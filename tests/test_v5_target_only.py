from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from csj.v3.panel_data import ConcreteContract
from csj.v3.pair_probe import PairProbe
from csj.v5.target_data import (
    TargetOnlyObservedCohort,
    build_target_only_cases,
)
from csj.v5.target_probe import TargetOnlyProbe, TargetOnlyProbeDataset


def _contract(contract_id: str, *, product: str = "rb") -> ConcreteContract:
    rows: list[dict[str, object]] = []
    for day_index in range(14):
        day = pd.Timestamp("2025-01-02") + pd.offsets.BDay(day_index)
        for bar in range(7):
            timestamp = day.to_pydatetime() + timedelta(hours=9 + bar)
            close = 100.0 + day_index + bar * 0.1
            rows.append(
                {
                    "instrument": contract_id,
                    "contract_id": contract_id,
                    "product": product,
                    "timestamps": timestamp,
                    "trading_day": day,
                    "open": close - 0.1,
                    "high": close + 0.3,
                    "low": close - 0.3,
                    "close": close,
                    "volume": 1000.0 + day_index * 10 + bar,
                    "amount": 100000.0 + day_index * 100 + bar,
                    "open_interest": 100.0,
                    "source_snapshot_id": "synthetic",
                }
            )
    frame = pd.DataFrame(rows).sort_values("timestamps", kind="stable").reset_index(drop=True)
    return ConcreteContract(
        contract_id=contract_id,
        product=product,
        frame=frame,
        source_snapshot_ids=("synthetic",),
    )


def _cohort() -> TargetOnlyObservedCohort:
    contract = _contract("rb2609")
    return TargetOnlyObservedCohort(
        manifest_path=Path("synthetic/manifest.json"),
        manifest_sha256="manifest",
        snapshot_id="synthetic",
        snapshot_at=pd.Timestamp("2025-01-01"),
        contracts={contract.contract_id: contract},
        contract_status={contract.contract_id: {"status": "ok", "product": "rb"}},
        payload_sha256={contract.contract_id: "payload"},
        data_fingerprint="target-only-fingerprint",
    )


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


def test_v5_target_cases_expose_no_neighbor_state() -> None:
    bundle = build_target_only_cases(_cohort(), lookback=35, products=["rb"])

    assert bundle.target_cases
    case = bundle.target_cases[0]
    assert not hasattr(case, "has_pair")
    assert not hasattr(case, "nearest_neighbor_id")
    assert not hasattr(case, "selected_neighbor_id")
    assert not hasattr(case, "neighbor_context")
    assert not hasattr(case, "term_structure")
    assert len(case.target_context) == 35
    assert len(case.target) in {15, 17, 19, 21}


def test_v5_target_case_builder_never_looks_at_another_contract() -> None:
    target = _contract("rb2609")
    unrelated = _contract("rb2610")
    first = _cohort()
    second = TargetOnlyObservedCohort(
        manifest_path=first.manifest_path,
        manifest_sha256=first.manifest_sha256,
        snapshot_id=first.snapshot_id,
        snapshot_at=first.snapshot_at,
        contracts={target.contract_id: target, unrelated.contract_id: unrelated},
        contract_status={
            target.contract_id: {"status": "ok", "product": "rb"},
            unrelated.contract_id: {"status": "ok", "product": "rb"},
        },
        payload_sha256={target.contract_id: "payload", unrelated.contract_id: "other-payload"},
        data_fingerprint=first.data_fingerprint,
    )

    first_cases = build_target_only_cases(first, lookback=35, products=["rb"]).target_cases
    second_cases = build_target_only_cases(second, lookback=35, products=["rb"]).target_cases

    assert [case.case_key for case in first_cases] == [
        case.case_key for case in second_cases if case.target_contract_id == target.contract_id
    ]


def test_v5_target_probe_dataset_and_encoder_have_no_neighbor_input() -> None:
    cases = build_target_only_cases(_cohort(), lookback=35, products=["rb"]).target_cases
    dataset = TargetOnlyProbeDataset(cases)
    batch = tuple(
        torch.stack([dataset[index][position] for index in range(len(dataset))])
        for position in range(5)
    )
    tokenizer = _SpyTokenizer()
    probe = TargetOnlyProbe(tokenizer, _ToyPredictor(), fusion_hidden_dim=8)

    logits = probe(*batch[:2])

    assert logits.shape == (len(cases),)
    assert tokenizer.calls == 1
    assert len(batch) == 5
    assert np.isfinite(logits.detach().numpy()).all()


def test_v5_target_only_logits_match_v4_target_branch_with_same_head_state() -> None:
    cases = build_target_only_cases(_cohort(), lookback=35, products=["rb"]).target_cases
    dataset = TargetOnlyProbeDataset(cases)
    batch = tuple(
        torch.stack([dataset[index][position] for index in range(len(dataset))])
        for position in range(5)
    )
    torch.manual_seed(17)
    v4 = PairProbe(_SpyTokenizer(), _ToyPredictor(), fusion_hidden_dim=8)
    v5 = TargetOnlyProbe(_SpyTokenizer(), _ToyPredictor(), fusion_hidden_dim=8)
    v5.load_head_state_dict(v4.head_state_dict())

    values, stamps, _, _, _ = batch
    v4_logits = v4(
        values,
        stamps,
        torch.zeros_like(values),
        torch.zeros_like(stamps),
        torch.zeros((len(cases), 3), dtype=torch.float32),
        torch.zeros(len(cases), dtype=torch.float32),
    )
    v5_logits = v5(values, stamps)

    torch.testing.assert_close(v5_logits, v4_logits)
