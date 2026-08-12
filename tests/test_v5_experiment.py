from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from csj.v5.experiment import V5Experiment, V5ExperimentError, resolve_device
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
