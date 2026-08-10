from __future__ import annotations

import pandas as pd
import pytest

from csj.config import load_config
from csj.futures_data import ForecastCase
from csj.metrics import (
    balanced_direction_accuracy,
    compute_metrics,
    ensemble_records,
    make_naive_baselines,
    paired_block_bootstrap_improvement,
)


def _records(predictions: list[int]) -> pd.DataFrame:
    actual = [-1, 1, -1, 1, -1, 1]
    return pd.DataFrame(
        {
            "instrument": ["a", "b", "a", "b", "a", "b"],
            "target_day": pd.date_range("2024-01-01", periods=3).repeat(2),
            "actual_direction": actual,
            "predicted_direction": predictions,
            "actual_return": [value * 0.01 for value in actual],
            "predicted_return": [value * 0.01 for value in predictions],
        }
    )


def test_balanced_accuracy_weights_directions_equally() -> None:
    assert balanced_direction_accuracy(
        [-1, -1, 1, 1],
        [-1, 1, 1, 1],
    ) == pytest.approx(0.75)


def test_ensemble_averages_returns_before_classification() -> None:
    first = _records([-1, 1, -1, 1, -1, 1])
    second = first.copy()
    second["predicted_return"] = -first["predicted_return"] * 0.5
    second["predicted_direction"] = -first["predicted_direction"]

    ensemble = ensemble_records([first, second])

    ensemble_indexed = ensemble.set_index(["instrument", "target_day"]).sort_index()
    first_indexed = first.set_index(["instrument", "target_day"]).sort_index()
    assert ensemble_indexed["predicted_direction"].tolist() == first_indexed[
        "predicted_direction"
    ].tolist()
    assert compute_metrics(ensemble)["direction_balanced_accuracy"] == 1.0


def test_ensemble_recomputes_path_metrics_and_combines_samples() -> None:
    first = _records([-1, 1, -1, 1, -1, 1]).iloc[:1].copy()
    second = first.copy()
    for records, predicted_path, predicted_range, samples in (
        (first, [1.0, 1.0], 0.05, [-0.01, 0.01]),
        (second, [1.0, 3.0], 0.15, [0.02, 0.03]),
    ):
        records["actual_close_path"] = [[1.0, 2.0]]
        records["predicted_close_path"] = [predicted_path]
        records["actual_range"] = 0.10
        records["predicted_range"] = predicted_range
        records["range_relative_error"] = 99.0
        records["close_path_correlation"] = -99.0
        records["sample_final_returns"] = [samples]

    ensemble = ensemble_records([first, second])

    assert ensemble.loc[0, "predicted_close_path"] == [1.0, 2.0]
    assert ensemble.loc[0, "close_path_correlation"] == pytest.approx(1.0)
    assert ensemble.loc[0, "range_relative_error"] == pytest.approx(0.0)
    assert ensemble.loc[0, "sample_final_returns"] == [-0.01, 0.01, 0.02, 0.03]


def test_paired_block_bootstrap_preserves_paired_days() -> None:
    model = _records([-1, 1, -1, 1, -1, 1])
    baseline = _records([1, -1, 1, -1, 1, -1])

    result = paired_block_bootstrap_improvement(
        model,
        baseline,
        iterations=100,
        block_days=2,
        seed=7,
    )

    assert result["point_estimate"] == pytest.approx(1.0)
    assert result["ci_lower_95"] == pytest.approx(1.0)
    assert result["ci_upper_95"] == pytest.approx(1.0)


def test_agreed_config_is_valid() -> None:
    config = load_config("csj/configs/futures_hourly.yaml")

    assert config["data"]["lookback"] == 256
    assert config["evaluation"]["sample_count"] == 10
    assert len(config["training"]["learning_rates"]) == 3
    assert len(config["training"]["seeds"]) == 3


def test_naive_baselines_share_opening_gap_sensitivity_fields() -> None:
    context = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
            "trading_day": [pd.Timestamp("2024-01-01")] * 2,
        }
    )
    train_case = ForecastCase(
        instrument="demo",
        target_day=pd.Timestamp("2024-01-02"),
        context=context,
        target=pd.DataFrame({"open": [100.0], "close": [102.0]}),
    )
    evaluation_case = ForecastCase(
        instrument="demo",
        target_day=pd.Timestamp("2024-01-03"),
        context=context,
        target=pd.DataFrame({"open": [104.0], "close": [99.0]}),
    )

    baselines = make_naive_baselines([train_case], [evaluation_case])

    for records in baselines.values():
        assert records.loc[0, "opening_gap"] == pytest.approx(0.04)
        assert bool(records.loc[0, "large_opening_gap"])
