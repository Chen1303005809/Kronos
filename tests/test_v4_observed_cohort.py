from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from csj.v3.panel_data import ConcreteContract
from csj.v4.cohort_data import (
    ObservedContractCohort,
    build_observed_cohort_cases,
)


def _contract(contract_id: str, *, timestamp_shift_minutes: int = 0) -> ConcreteContract:
    rows: list[dict[str, object]] = []
    for day_index in range(12):
        day = pd.Timestamp("2025-01-02") + pd.offsets.BDay(day_index)
        for bar in range(7):
            timestamp = day.to_pydatetime() + timedelta(hours=9 + bar)
            timestamp += timedelta(minutes=timestamp_shift_minutes)
            close = 100.0 + day_index + bar * 0.1
            rows.append(
                {
                    "instrument": contract_id,
                    "contract_id": contract_id,
                    "product": "rb",
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


def _cohort(*, shift_later: bool) -> ObservedContractCohort:
    contracts = {
        "rb2608": _contract("rb2608"),
        "rb2609": _contract("rb2609"),
        "rb2610": _contract("rb2610", timestamp_shift_minutes=1 if shift_later else 0),
    }
    return ObservedContractCohort(
        manifest_path=Path("synthetic/manifest.json"),
        manifest_sha256="manifest",
        snapshot_id="synthetic",
        snapshot_at=pd.Timestamp("2025-01-01 08:00"),
        contracts=contracts,
        contract_status={
            contract_id: {"product": "rb", "status": "ok"}
            for contract_id in contracts
        },
        payload_sha256={contract_id: f"payload-{contract_id}" for contract_id in contracts},
        cohort_fingerprint="cohort-fingerprint",
    )


def test_v4_selects_next_available_neighbor_when_nominal_closest_is_unavailable() -> None:
    bundle = build_observed_cohort_cases(_cohort(shift_later=True), lookback=35, products=["rb"])

    target_case = next(case for case in bundle.pair_cases if case.target_contract_id == "rb2609")

    # rb2610 is equally close but its timestamps fail availability filtering;
    # V4 must move on to rb2608 rather than discard this target case as V3 did.
    assert target_case.selected_neighbor_id == "rb2608"
    assert target_case.candidate_count == 1
    assert target_case.neighbor_context_available_at_origin
    assert target_case.selection_rule_version == "nearest-available-observed-neighbor-v1"


def test_v4_tie_prefers_later_delivery_after_availability_filtering() -> None:
    bundle = build_observed_cohort_cases(_cohort(shift_later=False), lookback=35, products=["rb"])

    target_case = next(case for case in bundle.pair_cases if case.target_contract_id == "rb2609")

    assert target_case.selected_neighbor_id == "rb2610"
    assert target_case.neighbor_delta_month == 1
    assert target_case.candidate_count == 2
