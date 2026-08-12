from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from csj.v5.experiment import (
    V5Experiment,
    V5ExperimentError,
    _assert_same_direction_records,
    _probe_seed_ensemble,
    resolve_device,
)
from csj.v5.path_bank import PathBankError, selected_baseline


def test_v5_cuda_is_required_when_requested(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(V5ExperimentError, match="requested CUDA"):
        resolve_device("cuda")


def test_v5_p0_selection_cannot_be_reused_by_other_run(tmp_path) -> None:
    experiment = object.__new__(V5Experiment)
    experiment.run_dir = tmp_path / "runs" / "active"
    experiment.run_id = "active"
    experiment.cohort = SimpleNamespace(data_fingerprint="fingerprint")
    experiment.config = {
        "p0": {
            "selection_order": [
                "zero_shot_mean_path",
                "zero_shot_sample_vote",
                "fit_product_majority",
                "context_3day_momentum",
            ]
        }
    }
    path = experiment.run_dir / "p0" / "fold_00" / "p0_baseline_selection.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "strategy_version": 5,
                "phase": "p0",
                "run_id": "other",
                "result_scope": "retrospective_observed_contracts",
                "production_eligible": False,
                "data_fingerprint": "fingerprint",
                "selected_direction_baseline": "zero_shot_mean_path",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(V5ExperimentError, match="does not belong"):
        experiment._require_p0_selection(fold_id="fold_00")


def test_v5_p1_gate_smoke_is_never_promotable() -> None:
    experiment = object.__new__(V5Experiment)
    gate = experiment._p1_gate(
        records_by_seed={},
        baseline=pd.DataFrame(),
        fit_probabilities_by_fold={},
        smoke=True,
    )

    assert gate["available"] is False
    assert gate["allows_next_phase"] is False


def test_v5_formal_p0_rejects_unselectable_inner_validation() -> None:
    records = pd.DataFrame(
        {
            "valid_direction": [True, True],
            "actual_direction": [1, 1],
            "predicted_direction": [1, 1],
        }
    )

    with pytest.raises(PathBankError, match="no finite balanced accuracy"):
        selected_baseline({"zero_shot_mean_path": records}, selection_order=("zero_shot_mean_path",))

    selected, details = selected_baseline(
        {"zero_shot_mean_path": records},
        selection_order=("zero_shot_mean_path",),
        allow_unavailable=True,
    )
    assert selected == "zero_shot_mean_path"
    assert details["selection_available"] is False


def _direction_records_with_target_end_day(target_end_day: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_key": ["i2601|2026-01-01T00:00:00"],
            "fold_id": ["fold_00"],
            "product": ["i"],
            "target_end_day": [target_end_day],
            "target_contract_id": ["i2601"],
            "actual_label": [1],
            "actual_direction": [1],
            "valid_direction": [True],
            "probability_up": [0.5],
            "predicted_label": [1],
            "predicted_direction": [1],
        }
    )


def test_v5_p1_alignment_accepts_json_round_tripped_target_end_day() -> None:
    """P0 JSON dates and fresh P1 Timestamps identify the same trading day."""

    fresh_probe_records = _direction_records_with_target_end_day(pd.Timestamp("2026-01-05"))
    cached_p0_records = _direction_records_with_target_end_day("2026-01-05T00:00:00")

    _assert_same_direction_records(
        fresh_probe_records,
        cached_p0_records,
        label="V5 P1/fold_00",
    )


def test_v5_p1_seed_ensemble_accepts_mixed_cached_and_fresh_dates() -> None:
    fresh = _direction_records_with_target_end_day(pd.Timestamp("2026-01-05"))
    cached = _direction_records_with_target_end_day("2026-01-05T00:00:00")

    ensemble = _probe_seed_ensemble({42: fresh, 43: cached, 44: fresh.copy()})

    assert pd.api.types.is_datetime64_any_dtype(ensemble["target_end_day"])
    assert ensemble.loc[0, "target_end_day"] == pd.Timestamp("2026-01-05")
