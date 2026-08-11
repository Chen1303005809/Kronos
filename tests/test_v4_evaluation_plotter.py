from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from csj.evaluation_plotter import (
    DirectionComparisonError,
    _bar_returns,
    render_fold_ablation_direction_comparison,
    render_fold_direction_comparison,
)


def _records(model: str, predictions: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_key": ["case-a", "case-b", "case-c", "case-d"],
            "fold_id": ["fold_00"] * 4,
            "product": ["i", "i", "rb", "rb"],
            "target_end_day": pd.bdate_range("2026-01-05", periods=4),
            "target_contract_id": ["i2609", "i2610", "rb2609", "rb2610"],
            "actual_direction": [1, -1, 0, 1],
            "predicted_direction": predictions,
            "probability_up": [0.8, 0.2, 0.5, 0.7],
            "model": [model] * 4,
        }
    )


def test_bar_returns_use_each_preceding_close_not_the_origin_for_every_bar() -> None:
    returns = _bar_returns(
        np.array([110.0, 121.0, 108.9]),
        origin_close=100.0,
        case_key="case-a",
    )

    np.testing.assert_allclose(returns, np.array([0.10, 0.10, -0.10]))


def test_fold_direction_plot_writes_required_png_json_and_case_records(tmp_path) -> None:
    artifacts = render_fold_direction_comparison(
        {
            "candidate": _records("candidate", [1, -1, 0, -1]),
            "baseline": _records("baseline", [-1, 1, 0, 1]),
        },
        fold_id="fold_00",
        candidate_model="candidate",
        baseline_model="baseline",
        output_dir=tmp_path / "evaluation",
        stage="p1_ablation",
        metadata={"strategy_version": 4, "production_eligible": False},
    )

    assert artifacts.figure_paths == (tmp_path / "evaluation" / "fold_00" / "prediction_vs_actual.png",)
    report = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert report["comparison"]["case_coverage"]["cases"] == 4
    assert report["comparison"]["metrics"]["candidate"]["valid_direction_cases"] == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.figure_paths)


def test_fold_direction_plot_rejects_mismatched_case_coverage(tmp_path) -> None:
    baseline = _records("baseline", [-1, 1, 0, 1]).iloc[:-1]
    with pytest.raises(DirectionComparisonError, match="case keys"):
        render_fold_direction_comparison(
            {"candidate": _records("candidate", [1, -1, 0, -1]), "baseline": baseline},
            fold_id="fold_00",
            candidate_model="candidate",
            baseline_model="baseline",
            output_dir=tmp_path,
            stage="p1_ablation",
            metadata={},
        )


def test_path_records_render_v2_style_continuous_paths(tmp_path) -> None:
    candidate = _records("candidate", [1, -1, 0, 1])
    baseline = _records("baseline", [1, -1, 0, 1])
    actual_path = [[100.0, 101.0, 99.0, 100.0], [101.0, 102.0, 100.0, 101.0]]
    predicted_path = [[100.0, 101.0, 99.0, 100.5], [101.0, 102.0, 100.0, 101.5]]
    sample_paths = [predicted_path, predicted_path]
    candidate["origin_close"] = 100.0
    candidate["actual_path"] = [actual_path] * len(candidate)
    candidate["predicted_path"] = [predicted_path] * len(candidate)
    candidate["sample_paths"] = [sample_paths] * len(candidate)
    candidate["day_end_indices"] = [[0, 1]] * len(candidate)

    artifacts = render_fold_direction_comparison(
        {"candidate": candidate, "baseline": baseline},
        fold_id="fold_00",
        candidate_model="candidate",
        baseline_model="baseline",
        output_dir=tmp_path,
        stage="p0",
        metadata={},
    )

    report = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert report["visualization"]["kind"] == "close_to_close_return_examples"
    assert report["visualization"]["additional_kind"] == "close_price_examples"
    assert len(artifacts.figure_paths) == 2
    assert artifacts.figure_paths[1].name == "close_price_comparison.png"
    assert all(path.stat().st_size > 0 for path in artifacts.figure_paths)


def test_ablation_plot_keeps_two_granularities_in_one_png(tmp_path) -> None:
    shared_target = _records("shared_target", [-1, 1, 0, 1])
    shared_pair = _records("shared_pair", [1, -1, 0, 1])
    product_target = _records("product_target", [-1, 1, 0, 1])
    product_pair = _records("product_pair", [1, -1, 0, 1])
    artifacts = render_fold_ablation_direction_comparison(
        {
            "shared_target": shared_target,
            "shared_pair": shared_pair,
            "product_target": product_target,
            "product_pair": product_pair,
        },
        fold_id="fold_00",
        comparisons={
            "shared": ("shared_pair", "shared_target"),
            "per-product": ("product_pair", "product_target"),
        },
        output_dir=tmp_path,
        stage="p1_ablation",
        metadata={},
    )

    report = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert set(report["comparisons"]) == {"shared", "per-product"}
    assert len(artifacts.figure_paths) == 1
