from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csj.v3.panel_data import ConcreteContract
from csj.v5.target_data import TargetOnlyCase
from csj.v6.audit import evaluate_p0_gate, prediction_day_atomicity_check
from csj.v6.config import load_v6_config, validate_v6_config
from csj.v6.plotting import render_p0_audit_plots
from csj.v6.risk_labels import (
    RiskLabelError,
    RiskLabelSpec,
    apply_tail_thresholds,
    context_diagnostics,
    fit_tail_thresholds,
    future_mutation_context_leakage_checks,
    risk_outcome,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec() -> RiskLabelSpec:
    return RiskLabelSpec(
        lookback=70,
        horizon_trading_days=3,
        valid_bar_counts=(5, 7),
        volatility_bars=60,
        volatility_halflife_bars=20,
        volatility_adjust=False,
        volatility_bias=False,
        volatility_floor=1e-5,
        tail_quantile=0.8,
        tail_quantile_method="linear",
        clip=5.0,
        epsilon=1e-5,
    )


def _case(*, product: str = "rb", contract_id: str = "rb2609") -> TargetOnlyCase:
    rows: list[dict[str, object]] = []
    context_days = [pd.Timestamp("2025-01-02") + pd.offsets.BDay(index) for index in range(10)]
    target_days = [context_days[-1] + pd.offsets.BDay(index) for index in range(1, 4)]
    context_closes = np.linspace(91.0, 100.0, 70) + np.sin(np.arange(70)) * 0.35
    context_closes[-1] = 100.0
    for index, close in enumerate(context_closes):
        day_index, bar = divmod(index, 7)
        timestamp = context_days[day_index].to_pydatetime() + timedelta(hours=9 + bar)
        rows.append(
            {
                "instrument": contract_id,
                "contract_id": contract_id,
                "product": product,
                "timestamps": timestamp,
                "trading_day": context_days[day_index],
                "open": close - 0.1,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": 1000 + index,
                "amount": 100000 + index * 10,
                "open_interest": 100.0,
                "source_snapshot_id": "synthetic",
            }
        )
    target_closes = np.asarray(
        [99.0, 98.0, 97.0, 96.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    )
    target_lows = target_closes - 0.5
    target_highs = target_closes + 0.5
    target_lows[4] = 90.0
    target_highs[12] = 110.0
    for index, close in enumerate(target_closes):
        day_index, bar = divmod(index, 5)
        timestamp = target_days[day_index].to_pydatetime() + timedelta(hours=9 + bar)
        rows.append(
            {
                "instrument": contract_id,
                "contract_id": contract_id,
                "product": product,
                "timestamps": timestamp,
                "trading_day": target_days[day_index],
                "open": close,
                "high": target_highs[index],
                "low": target_lows[index],
                "close": close,
                "volume": 1200 + index,
                "amount": 120000 + index * 10,
                "open_interest": 110.0,
                "source_snapshot_id": "synthetic",
            }
        )
    frame = pd.DataFrame(rows).sort_values("timestamps", kind="stable").reset_index(drop=True)
    contract = ConcreteContract(
        contract_id=contract_id,
        product=product,
        frame=frame,
        source_snapshot_ids=("synthetic",),
    )
    return TargetOnlyCase(
        target_contract_id=contract_id,
        product=product,
        origin_timestamp=pd.Timestamp(frame["timestamps"].iloc[69]),
        origin_trading_day=pd.Timestamp(frame["trading_day"].iloc[69]).normalize(),
        target_contract=contract,
        target_context_start=0,
        target_start=70,
        target_end_exclusive=85,
        target_days=tuple(pd.Timestamp(day).normalize() for day in target_days),
        day_end_indices=(4, 9, 14),
        data_fingerprint="synthetic-fingerprint",
    )


def test_v6_config_loads_and_rejects_label_protocol_drift() -> None:
    config = load_v6_config(REPO_ROOT / "csj/configs/risk_control_v6.yaml")

    assert config["experiment"]["version"] == 6
    assert config["risk_labels"]["context_volatility"]["adjust"] is False
    assert Path(config["data"]["snapshot_root"]).is_absolute()

    mutated = deepcopy(config)
    mutated["risk_labels"]["tail_event"]["quantile"] = 0.75
    with pytest.raises(ValueError, match="tail_event.quantile"):
        validate_v6_config(mutated)


def test_v6_risk_outcome_uses_past_scale_and_declared_adverse_prices() -> None:
    case = _case()
    spec = _spec()

    outcome = risk_outcome(case, spec)

    scale = float(outcome["context_horizon_scale"])
    assert outcome["origin_close"] == pytest.approx(100.0)
    assert outcome["long_mae"] == pytest.approx(-np.log(90.0 / 100.0) / scale)
    assert outcome["short_mae"] == pytest.approx(np.log(110.0 / 100.0) / scale)
    future_closes = np.concatenate(([100.0], case.target["close"].to_numpy(dtype=float)))
    expected_future_scale = np.sqrt(np.square(np.diff(np.log(future_closes))).sum())
    assert outcome["future_realized_scale"] == pytest.approx(expected_future_scale)
    assert outcome["future_vol_ratio"] == pytest.approx(
        np.log((expected_future_scale + spec.epsilon) / (scale + spec.epsilon))
    )


def test_v6_future_ohlcva_mutations_cannot_change_context_scale_or_signature() -> None:
    case = _case()
    spec = _spec()
    baseline = context_diagnostics(case, spec)
    frame = case.target_contract.frame.copy(deep=True)
    frame[["open", "high", "low", "close", "volume", "amount"]] = frame[
        ["open", "high", "low", "close", "volume", "amount"]
    ].astype(float)
    frame.loc[70:, ["open", "high", "low", "close", "volume", "amount"]] *= 1.5
    mutated_case = replace(
        case,
        target_contract=replace(case.target_contract, frame=frame),
    )

    observed = context_diagnostics(mutated_case, spec)
    checks = future_mutation_context_leakage_checks(case, spec)

    assert observed == baseline
    assert {check["mutated_future_feature"] for check in checks} == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    assert all(check["passed"] for check in checks)


def _threshold_records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product_index, product in enumerate(("i", "jm", "rb")):
        for index in range(5):
            rows.append(
                {
                    "case_key": f"{product}2609|2025-01-{index + 2:02d}",
                    "fold_id": "fold_00",
                    "split": "fit",
                    "product": product,
                    "target_end_day": pd.Timestamp("2025-01-02") + pd.offsets.BDay(index),
                    "long_mae": float(index + product_index),
                    "short_mae": float(10 - index + product_index),
                    "data_fingerprint": "fingerprint",
                }
            )
    return pd.DataFrame(rows)


def test_v6_tail_thresholds_are_fit_primary_only_and_do_not_read_evaluation() -> None:
    fit = _threshold_records()
    thresholds = fit_tail_thresholds(
        fit,
        fold_id="fold_00",
        fit_start_day=pd.Timestamp("2025-01-02"),
        fit_end_day=pd.Timestamp("2025-01-31"),
        primary_products=("i", "jm", "rb"),
        quantile=0.8,
        quantile_method="linear",
    )
    evaluation = fit.copy()
    evaluation["case_key"] = "evaluation|" + evaluation["case_key"]
    evaluation["split"] = "evaluation"
    evaluation["long_mae"] = 1_000_000.0
    evaluation["short_mae"] = 1_000_000.0

    labeled = apply_tail_thresholds(evaluation, thresholds, split="evaluation")

    assert thresholds.long_tail_threshold == pytest.approx(
        np.quantile(fit["long_mae"], 0.8, method="linear")
    )
    assert thresholds.short_tail_threshold == pytest.approx(
        np.quantile(fit["short_mae"], 0.8, method="linear")
    )
    assert labeled["long_tail_event"].all()
    assert labeled["short_tail_event"].all()

    contaminated = pd.concat(
        [fit, evaluation.assign(split="evaluation")], ignore_index=True
    )
    with pytest.raises(RiskLabelError, match="non-fit split"):
        fit_tail_thresholds(
            contaminated,
            fold_id="fold_00",
            fit_start_day=pd.Timestamp("2025-01-02"),
            fit_end_day=pd.Timestamp("2025-01-31"),
            primary_products=("i", "jm", "rb"),
            quantile=0.8,
            quantile_method="linear",
        )

    transfer = pd.concat(
        [fit, fit.iloc[[0]].assign(case_key="j2609|x", product="j")],
        ignore_index=True,
    )
    with pytest.raises(RiskLabelError, match="primary product"):
        fit_tail_thresholds(
            transfer,
            fold_id="fold_00",
            fit_start_day=pd.Timestamp("2025-01-02"),
            fit_end_day=pd.Timestamp("2025-01-31"),
            primary_products=("i", "jm", "rb"),
            quantile=0.8,
            quantile_method="linear",
        )


def _gate_records(*, fail_one: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in ("fold_00", "fold_01"):
        for split in ("fit", "inner_validation", "evaluation"):
            for product in ("i", "jm", "rb"):
                for index in range(2):
                    rows.append(
                        {
                            "case_key": f"{fold_id}|{split}|{product}|{index}",
                            "fold_id": fold_id,
                            "split": split,
                            "product": product,
                            "long_tail_event": not (
                                fail_one
                                and fold_id == "fold_00"
                                and split == "inner_validation"
                                and index == 1
                            ),
                            "short_tail_event": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_v6_p0_gate_requires_every_fold_split_side_and_unique_pooled_cases() -> None:
    p0 = {
        "minimum_fit_events_per_side": 6,
        "minimum_validation_events_per_side": 6,
        "minimum_evaluation_events_per_side": 6,
        "minimum_pooled_evaluation_events_per_product_side": 2,
        "maximum_integrity_failures": 0,
        "maximum_leakage_failures": 0,
    }
    passing = evaluate_p0_gate(
        _gate_records(),
        expected_fold_ids=("fold_00", "fold_01"),
        primary_products=("i", "jm", "rb"),
        p0_config=p0,
        integrity_failures=(),
        leakage_checks=(),
    )
    failing = evaluate_p0_gate(
        _gate_records(fail_one=True),
        expected_fold_ids=("fold_00", "fold_01"),
        primary_products=("i", "jm", "rb"),
        p0_config=p0,
        integrity_failures=(),
        leakage_checks=(),
    )

    assert passing["allows_next_phase"] is True
    assert failing["allows_next_phase"] is False
    assert "fold_00:inner_validation:long_event_support" in failing[
        "failed_condition_ids"
    ]


def test_p0_gate_can_keep_fold_pool_wider_than_per_product_gate() -> None:
    records = _gate_records()
    records = pd.concat(
        [
            records,
            records.loc[records["product"] == "rb"].assign(
                product="a",
                case_key=lambda frame: "extra|" + frame["case_key"],
            ),
        ],
        ignore_index=True,
    )
    gate = evaluate_p0_gate(
        records,
        expected_fold_ids=("fold_00", "fold_01"),
        primary_products=("i", "jm", "rb", "a"),
        pooled_evaluation_products=("i", "jm", "rb"),
        p0_config={
            "minimum_fit_events_per_side": 8,
            "minimum_validation_events_per_side": 8,
            "minimum_evaluation_events_per_side": 8,
            "minimum_pooled_evaluation_events_per_product_side": 2,
            "maximum_integrity_failures": 0,
            "maximum_leakage_failures": 0,
        },
        integrity_failures=(),
        leakage_checks=(),
    )

    assert gate["allows_next_phase"] is True
    assert set(gate["support"]["pooled_evaluation_by_product"]) == {"i", "jm", "rb"}


def test_v6_prediction_day_atomicity_rejects_one_origin_in_two_splits() -> None:
    records = pd.DataFrame(
        {
            "origin_trading_day": [pd.Timestamp("2026-05-13")] * 2,
            "split": ["fit", "inner_validation"],
        }
    )

    check = prediction_day_atomicity_check(records, fold_id="fold_04")

    assert check["passed"] is False
    assert check["maximum_split_assignments_per_origin_trading_day"] == 2
    assert check["violating_origin_trading_days"] == [pd.Timestamp("2026-05-13")]


def test_v6_p0_plot_contract_generates_nonempty_pngs_and_summary(tmp_path: Path) -> None:
    outcomes = pd.DataFrame(
        [
            {
                "case_key": f"{product}|{index}",
                "product": product,
                "long_mae": 0.1 + index / 10,
                "short_mae": 0.2 + index / 12,
                "future_vol_ratio": -0.5 + index / 8,
            }
            for product in ("i", "jm", "rb")
            for index in range(12)
        ]
    )
    fold_records = pd.DataFrame(
        [
            {
                "case_key": f"{fold_id}|{split}|{product}|{index}",
                "fold_id": fold_id,
                "split": split,
                "product": product,
                "long_tail_event": index >= 8,
                "short_tail_event": index >= 9,
                "long_tail_threshold": 0.8 + fold_index / 10,
                "short_tail_threshold": 0.9 + fold_index / 10,
            }
            for fold_index, fold_id in enumerate(("fold_00", "fold_01"))
            for split in ("fit", "inner_validation", "evaluation")
            for product in ("i", "jm", "rb")
            for index in range(12)
        ]
    )
    artifacts = render_p0_audit_plots(
        outcomes,
        fold_records,
        primary_products=("i", "jm", "rb"),
        p0_config={
            "minimum_fit_events_per_side": 100,
            "minimum_validation_events_per_side": 20,
            "minimum_evaluation_events_per_side": 20,
        },
        output_dir=tmp_path,
        metadata={"strategy_version": 6, "phase": "p0"},
    )

    for path in (
        artifacts.risk_label_distributions,
        artifacts.fold_event_support,
        artifacts.fold_tail_thresholds,
        artifacts.summary_json,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0
    summary = pd.read_json(artifacts.summary_json, typ="series")
    assert summary["plot_contract_version"] == "v6-p0-risk-label-audit-v1"
