from __future__ import annotations

import json

import pandas as pd

from csj.v3.evaluation_plotter import (
    METRIC_CONTRACT_VERSION,
    P0_FIXED_METRICS,
    P1_FIXED_METRICS,
    render_p0_evaluation_report,
    render_p1_evaluation_report,
)


def _p0_records(model: str, *, inverse: bool = False) -> pd.DataFrame:
    actual_directions = [1, -1, 1, -1]
    predicted_directions = [-value for value in actual_directions] if inverse else actual_directions
    actual_returns = [0.02, -0.02, 0.01, -0.01]
    predicted_returns = [-value if inverse else value for value in actual_returns]
    return pd.DataFrame(
        {
            "case_key": [f"{model}-{index}" for index in range(4)],
            "model": [model] * 4,
            "product": ["i", "i", "rb", "rb"],
            "day3_actual_direction": actual_directions,
            "day3_predicted_direction": predicted_directions,
            "day3_actual_return": actual_returns,
            "day3_predicted_return": predicted_returns,
            "path_return_correlation": [-0.5 if inverse else 0.5] * 4,
            "z_normalized_dtw": [1.0 if inverse else 0.25] * 4,
        }
    )


def _p1_records(*, pair: bool) -> pd.DataFrame:
    labels = [0, 1, 0, 1]
    return pd.DataFrame(
        {
            "case_key": [f"case-{index}" for index in range(4)],
            "product": ["i", "i", "rb", "rb"],
            "target_end_day": pd.bdate_range("2026-01-02", periods=4),
            "neighbor_direction": ["earlier", "later", "earlier", "later"],
            "actual_label": labels,
            "predicted_label": labels if pair else [1 - value for value in labels],
            "valid_direction": [True] * 4,
        }
    )


def test_p0_plotter_writes_the_fixed_metric_contract_and_required_figures(tmp_path) -> None:
    artifacts = render_p0_evaluation_report(
        {
            "zero_shot": _p0_records("zero_shot", inverse=True),
            "ce_only": _p0_records("ce_only"),
        },
        output_dir=tmp_path,
        stage="p0_ce_only",
        metadata={"result_scope": "exploratory_partial_panel", "context_length": 256},
    )

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report["metric_contract"]["version"] == METRIC_CONTRACT_VERSION
    assert [item["key"] for item in report["metric_contract"]["metrics"]] == [
        metric.key for metric in P0_FIXED_METRICS
    ]
    assert set(report["overall"]) == {"zero_shot", "ce_only"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.figure_paths)


def test_p1_plotter_writes_paired_fixed_metrics_and_stratified_figure(tmp_path) -> None:
    pair = _p1_records(pair=True)
    target_only = _p1_records(pair=False)
    artifacts = render_p1_evaluation_report(
        pair,
        target_only,
        bootstrap={
            "block_5": {"available": True, "probability_improvement_positive": 1.0},
            "block_10": {"available": True, "probability_improvement_positive": 1.0},
        },
        gate={"fold_balanced_accuracy_improvements": {"fold_00": 1.0}},
        output_dir=tmp_path,
        stage="p1_pair_probe",
        metadata={"result_scope": "exploratory_partial_panel", "context_length": 256},
    )

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert [item["key"] for item in report["metric_contract"]["metrics"]] == [
        metric.key for metric in P1_FIXED_METRICS
    ]
    assert report["overall"]["balanced_accuracy_improvement"] == 1.0
    assert report["by_product"]["i"]["balanced_accuracy_improvement"] == 1.0
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.figure_paths)
