from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from csj.v3.pair_probe import (
    PairProbe,
    PanelProbeDataset,
    assert_same_case_keys,
    paired_block_bootstrap,
    prediction_day_sampling_summary,
    prediction_day_uniform_weights,
)
from csj.v3.config import load_v3_config
from csj.v3.experiment import _p1_gate
from csj.v3.p0 import ProductDenseWindowDataset, target_path_metrics
from csj.v3.panel_data import (
    ConcreteContract,
    PanelArchive,
    PanelSnapshot,
    build_panel_cases,
    delivery_month_distance,
)


def _contract(
    contract_id: str,
    *,
    days: int = 8,
    first_timestamp_shift_minutes: int = 0,
) -> ConcreteContract:
    rows: list[dict[str, object]] = []
    for day_index in range(days):
        day = pd.Timestamp("2024-01-02") + pd.offsets.BDay(day_index)
        for bar in range(7):
            close = 100.0 + day_index + bar * 0.1
            timestamp = day.to_pydatetime() + timedelta(hours=int(9 + bar))
            if day_index == 0 and bar == 0:
                timestamp += timedelta(minutes=int(first_timestamp_shift_minutes))
            rows.append(
                {
                    "instrument": contract_id,
                    "contract_id": contract_id,
                    "product": "rb",
                    "timestamps": timestamp,
                    "trading_day": day,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0 + day_index * 10 + bar,
                    "amount": 100000.0 + day_index * 100 + bar,
                    "open_interest": 100.0,
                    "source_snapshot_id": "synthetic",
                }
            )
    frame = pd.DataFrame(rows).sort_values("timestamps", kind="stable").reset_index(drop=True)
    frame.attrs["timestamp_positions"] = {
        pd.Timestamp(timestamp): index
        for index, timestamp in enumerate(frame["timestamps"].tolist())
    }
    return ConcreteContract(
        contract_id=contract_id,
        product="rb",
        frame=frame,
        source_snapshot_ids=("synthetic",),
    )


def _archive(
    *,
    snapshot_at: pd.Timestamp,
    completeness: str = "complete",
    mismatch_nearest: bool = False,
    daily_snapshots: bool = True,
) -> PanelArchive:
    target = _contract("rb2609")
    later = _contract(
        "rb2610",
        first_timestamp_shift_minutes=-30 if mismatch_nearest else 0,
    )
    earlier = _contract("rb2608")
    snapshot_days = (
        list(pd.bdate_range("2024-01-02", periods=8))
        if daily_snapshots
        else [snapshot_at.normalize()]
    )
    snapshots = tuple(
        PanelSnapshot(
            snapshot_id=f"synthetic-{index}",
            snapshot_at=(
                pd.Timestamp(day).replace(hour=8)
                if daily_snapshots
                else snapshot_at
            ),
            panel_completeness=completeness,
            contracts_by_product={"rb": ("rb2608", "rb2609", "rb2610")},
            contract_status={"rb2608": "ok", "rb2609": "ok", "rb2610": "ok"},
        )
        for index, day in enumerate(snapshot_days)
    )
    return PanelArchive(
        root=Path("synthetic"),
        snapshots=snapshots,
        contracts={"rb2608": earlier, "rb2609": target, "rb2610": later},
        source_audit={},
    )


def test_delivery_distance_crosses_year_and_tie_prefers_later_neighbor() -> None:
    assert delivery_month_distance("rb2612", "rb2701") == 1
    archive = _archive(snapshot_at=pd.Timestamp("2024-01-02 08:00"))

    bundle = build_panel_cases(archive, lookback=35, products=["rb"])

    target_case = next(
        case for case in bundle.strict_pair_cases if case.target_contract_id == "rb2609"
    )
    assert target_case.nearest_neighbor_id == "rb2610"
    assert target_case.neighbor_delta_month == 1
    assert target_case.target_context["timestamps"].tolist() == target_case.neighbor_context[  # type: ignore[union-attr]
        "timestamps"
    ].tolist()


def test_future_snapshot_is_exploratory_partial_not_a_strict_pair() -> None:
    archive = _archive(
        snapshot_at=pd.Timestamp("2024-01-11 08:00"),
        completeness="complete",
        daily_snapshots=False,
    )

    bundle = build_panel_cases(archive, lookback=35, products=["rb"])

    assert not bundle.strict_pair_cases
    assert bundle.partial_pair_cases
    assert {case.panel_source for case in bundle.partial_pair_cases} == {
        "retrospective_future_snapshot"
    }
    assert all(case.partial_panel for case in bundle.partial_pair_cases)


def test_nearest_neighbor_timestamp_gap_rejects_pair_without_skipping_to_next() -> None:
    archive = _archive(
        snapshot_at=pd.Timestamp("2024-01-02 08:00"),
        mismatch_nearest=True,
    )

    bundle = build_panel_cases(archive, lookback=35, products=["rb"])

    target_case = next(
        case for case in bundle.target_cases if case.target_contract_id == "rb2609"
    )
    assert target_case.nearest_neighbor_id == "rb2610"
    assert not target_case.has_pair
    assert target_case.pair_rejection_reason == "nearest_neighbor_context_timestamp_mismatch"


class _SpyTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def encode(self, values: torch.Tensor, half: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        del half
        self.calls += 1
        token = (values[..., 0] > 0).to(torch.long)
        return token, token


class _ToyPredictor(nn.Module):
    d_model = 4

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def decode_s1(
        self,
        token_s1: torch.Tensor,
        token_s2: torch.Tensor,
        stamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del token_s2, stamps
        hidden = token_s1.to(torch.float32).unsqueeze(-1).repeat(1, 1, self.d_model)
        return hidden, hidden * self.scale


def _probe_batch(dataset: PanelProbeDataset) -> tuple[torch.Tensor, ...]:
    items = [dataset[index] for index in range(len(dataset))]
    return tuple(torch.stack([item[position] for item in items]) for position in range(len(items[0])))


def test_target_only_probe_never_encodes_neighbor_and_keeps_head_parameter_count() -> None:
    archive = _archive(snapshot_at=pd.Timestamp("2024-01-02 08:00"))
    cases = build_panel_cases(archive, lookback=35, products=["rb"]).strict_pair_cases
    assert cases
    assert_same_case_keys(cases, cases)
    tokenizer = _SpyTokenizer()
    predictor = _ToyPredictor()
    probe = PairProbe(tokenizer, predictor, fusion_hidden_dim=8)
    target_batch = _probe_batch(PanelProbeDataset(cases, mode="target_only_probe"))
    target_logits = probe(*target_batch[:6])
    assert target_logits.shape == (len(cases),)
    assert tokenizer.calls == 1

    pair_probe = PairProbe(_SpyTokenizer(), _ToyPredictor(), fusion_hidden_dim=8)
    pair_batch = _probe_batch(PanelProbeDataset(cases, mode="pair_probe"))
    pair_logits = pair_probe(*pair_batch[:6])
    assert pair_logits.shape == (len(cases),)
    assert pair_probe.tokenizer.calls == 2  # type: ignore[attr-defined]
    assert probe.trainable_parameter_count == pair_probe.trainable_parameter_count


def test_prediction_day_uniform_weights_prevent_multi_contract_days_from_dominating() -> None:
    cases = build_panel_cases(
        _archive(snapshot_at=pd.Timestamp("2024-01-02 08:00")),
        lookback=35,
        products=["rb"],
    ).strict_pair_cases
    assert cases

    weights = prediction_day_uniform_weights(cases)
    days = pd.Series([case.origin_trading_day.normalize() for case in cases])
    masses = pd.DataFrame({"day": days, "weight": weights.numpy()}).groupby("day")["weight"].sum()
    summary = prediction_day_sampling_summary(cases, strategy="prediction_day_uniform")

    np.testing.assert_allclose(masses.to_numpy(), np.ones(len(masses)))
    assert summary["strategy"] == "prediction_day_uniform"
    assert summary["unique_prediction_days"] == len(masses)
    assert summary["per_prediction_day_sampling_mass"] == pytest.approx(1.0)


def test_paired_block_bootstrap_preserves_case_pairing() -> None:
    days = pd.bdate_range("2024-01-02", periods=6)
    labels = [0, 1, 0, 1, 0, 1]
    target = pd.DataFrame(
        {
            "case_key": [f"case-{index}" for index in range(6)],
            "target_end_day": days,
            "actual_label": labels,
            "predicted_label": [1 - value for value in labels],
            "valid_direction": True,
        }
    )
    pair = target.copy()
    pair["predicted_label"] = labels

    result = paired_block_bootstrap(pair, target, block_days=2, iterations=100, seed=7)

    assert result["point_estimate"] == pytest.approx(1.0)
    assert result["probability_improvement_positive"] == pytest.approx(1.0)


def test_p0_dense_windows_keep_target_only_context_normalization() -> None:
    archive = _archive(snapshot_at=pd.Timestamp("2024-01-02 08:00"))
    dataset = ProductDenseWindowDataset(
        tuple(archive.contracts.values()),
        product="rb",
        fit_end_day=pd.Timestamp("2024-01-11"),
        lookback=7,
    )

    values, stamps = dataset[0]

    assert values.shape == (28, 6)
    assert stamps.shape == (28, 5)
    np.testing.assert_allclose(values[:7].numpy().mean(axis=0), 0.0, atol=1e-6)


def test_p0_metrics_use_day3_path_direction_and_path_fields() -> None:
    records = pd.DataFrame(
        {
            "product": ["rb", "rb", "rb", "rb"],
            "day3_actual_direction": [1, -1, 1, -1],
            "day3_predicted_direction": [1, -1, 1, -1],
            "day3_actual_return": [0.01, -0.01, 0.02, -0.02],
            "day3_predicted_return": [0.01, -0.01, 0.02, -0.02],
            "path_return_correlation": [0.1, 0.2, 0.3, 0.4],
            "z_normalized_dtw": [0.2, 0.2, 0.2, 0.2],
        }
    )

    metrics = target_path_metrics(records)

    assert metrics["day3_path_balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["day3_return_mae"] == pytest.approx(0.0)
    assert metrics["mean_return_path_correlation"] == pytest.approx(0.25)


def test_v3_config_is_cuda_first_and_disallows_partial_panel_training_by_default() -> None:
    config = load_v3_config("csj/configs/active_contract_panel_v3.yaml")
    exploratory = load_v3_config("csj/configs/active_contract_panel_v3_partial.yaml")

    assert config["experiment"]["version"] == 3
    assert config["runtime"]["device"] == "cuda"
    assert config["data"]["lookbacks"] == [256, 512]
    assert not config["data"]["allow_partial_panel_training"]
    assert config["evaluation"]["bootstrap_block_days"] == [5, 10]
    assert config["p1"]["training"]["sampling_strategy"] == "prediction_day_uniform"
    assert exploratory["runtime"]["device"] == "cuda"
    assert exploratory["data"]["allow_partial_panel_training"]
    assert exploratory["p1"]["training"]["sampling_strategy"] == "prediction_day_uniform"
    assert exploratory["output"]["root"].endswith("active_contract_panel_v3_partial")


def test_partial_panel_probe_can_report_statistics_but_cannot_open_p2_gate() -> None:
    records = pd.DataFrame(
        {
            "case_key": ["a", "b", "c", "d"],
            "product": ["rb", "rb", "rb", "rb"],
            "neighbor_direction": ["earlier", "later", "earlier", "later"],
            "valid_direction": [True, True, True, True],
            "actual_label": [0, 1, 0, 1],
            "predicted_label": [0, 1, 0, 1],
        }
    )
    target_only = records.copy()
    target_only["predicted_label"] = [1, 0, 1, 0]
    gate = _p1_gate(
        records,
        target_only,
        by_fold={"fold_00": (records, target_only)},
        bootstrap={
            "block_5": {"available": True, "probability_improvement_positive": 1.0},
            "block_10": {"available": True, "probability_improvement_positive": 1.0},
        },
        formal_complete_panel_provenance=False,
    )

    assert gate["statistical_gate_passed"]
    assert not gate["formal_complete_panel_provenance"]
    assert not gate["passes_p1_to_p2_gate"]
