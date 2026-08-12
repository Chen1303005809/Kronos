from __future__ import annotations

import pandas as pd

from csj.v5.plotting import render_fold_baseline_path_plots, render_fold_path_comparisons


def _records(model: str) -> pd.DataFrame:
    actual_path = [[100.0, 101.0, 99.0, 100.0], [101.0, 102.0, 100.0, 101.0]]
    predicted_path = [[100.0, 101.0, 99.0, 100.5], [101.0, 102.0, 100.0, 101.5]]
    return pd.DataFrame(
        {
            "case_key": ["case-a", "case-b", "case-c", "case-d"],
            "fold_id": ["fold_00"] * 4,
            "product": ["i", "i", "rb", "rb"],
            "target_end_day": pd.bdate_range("2026-01-05", periods=4),
            "target_contract_id": ["i2609", "i2610", "rb2609", "rb2610"],
            "origin_close": [100.0] * 4,
            "actual_path": [actual_path] * 4,
            "predicted_path": [predicted_path] * 4,
            "day_end_indices": [[0, 1]] * 4,
            "day3_actual_return": [0.01, -0.01, 0.02, -0.02],
            "day3_predicted_return": [0.015, -0.005, 0.01, -0.01],
            "day3_actual_direction": [1, -1, 1, -1],
            "day3_predicted_direction": [1, -1, 1, -1],
            "model": [model] * 4,
        }
    )


def test_v5_baseline_plot_writes_return_and_close_artifacts(tmp_path) -> None:
    artifacts = render_fold_baseline_path_plots(
        _records("zero_shot_mean_path"),
        fold_id="fold_00",
        output_dir=tmp_path,
        stage="p0",
        baseline_label="zero_shot_mean_path",
        metadata={"strategy_version": 5},
    )

    assert artifacts.return_comparison.name == "day3_close_return_comparison.png"
    assert artifacts.close_comparison.name == "close_price_comparison.png"
    assert all(path.is_file() and path.stat().st_size > 0 for path in (
        artifacts.return_comparison,
        artifacts.close_comparison,
        artifacts.summary_json,
    ))


def test_v5_path_comparison_writes_return_and_close_artifacts(tmp_path) -> None:
    candidate = _records("candidate")
    baseline = _records("baseline")
    artifacts = render_fold_path_comparisons(
        candidate,
        baseline,
        fold_id="fold_00",
        output_dir=tmp_path,
        stage="p2_path_bridge",
        candidate_label="probe_weighted_path",
        baseline_label="zero_shot_mean_path",
        metadata={"strategy_version": 5},
    )

    assert all(path.is_file() and path.stat().st_size > 0 for path in (
        artifacts.return_comparison,
        artifacts.close_comparison,
        artifacts.summary_json,
    ))
