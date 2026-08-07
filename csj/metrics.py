from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from csj.futures_data import ForecastCase


KEY_COLUMNS = ["instrument", "target_day"]


def direction_label(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def cumulative_return_path(
    close_path: Sequence[float] | np.ndarray,
    origin_close: float,
) -> np.ndarray:
    values = np.asarray(close_path, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("close_path must be a non-empty one-dimensional sequence")
    if not np.isfinite(origin_close) or origin_close == 0:
        raise ValueError("origin_close must be finite and non-zero")
    return values / float(origin_close) - 1.0


def three_day_endpoint_returns(
    close_path: Sequence[float] | np.ndarray,
    origin_close: float,
    day_end_indices: Sequence[int],
) -> tuple[float, float, float]:
    returns = cumulative_return_path(close_path, origin_close)
    indices = tuple(int(value) for value in day_end_indices)
    if len(indices) != 3 or tuple(sorted(indices)) != indices:
        raise ValueError("day_end_indices must contain three increasing indices")
    if indices[0] < 0 or indices[-1] >= len(returns):
        raise ValueError("day_end_indices fall outside the close path")
    return cast(
        tuple[float, float, float],
        tuple(float(returns[index]) for index in indices),
    )


def return_path_correlation(
    actual_close: Sequence[float] | np.ndarray,
    predicted_close: Sequence[float] | np.ndarray,
    origin_close: float,
) -> float:
    actual = cumulative_return_path(actual_close, origin_close)
    predicted = cumulative_return_path(predicted_close, origin_close)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted paths must have the same shape")
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def _z_normalize_path(path: np.ndarray) -> np.ndarray:
    std = float(np.std(path))
    if std == 0:
        return np.zeros_like(path)
    return (path - float(np.mean(path))) / std


def dtw_distance(
    actual_path: Sequence[float] | np.ndarray,
    predicted_path: Sequence[float] | np.ndarray,
) -> float:
    """Mean absolute dynamic-time-warping cost along the best alignment."""

    actual = np.asarray(actual_path, dtype=np.float64)
    predicted = np.asarray(predicted_path, dtype=np.float64)
    if actual.ndim != 1 or predicted.ndim != 1 or not len(actual) or not len(predicted):
        raise ValueError("DTW paths must be non-empty one-dimensional sequences")

    costs = np.full((len(actual) + 1, len(predicted) + 1), np.inf)
    steps = np.full((len(actual) + 1, len(predicted) + 1), np.iinfo(np.int32).max)
    costs[0, 0] = 0.0
    steps[0, 0] = 0
    for actual_index in range(1, len(actual) + 1):
        for predicted_index in range(1, len(predicted) + 1):
            candidates = (
                (
                    costs[actual_index - 1, predicted_index],
                    steps[actual_index - 1, predicted_index],
                ),
                (
                    costs[actual_index, predicted_index - 1],
                    steps[actual_index, predicted_index - 1],
                ),
                (
                    costs[actual_index - 1, predicted_index - 1],
                    steps[actual_index - 1, predicted_index - 1],
                ),
            )
            previous_cost, previous_steps = min(candidates, key=lambda item: item)
            costs[actual_index, predicted_index] = previous_cost + abs(
                actual[actual_index - 1] - predicted[predicted_index - 1]
            )
            steps[actual_index, predicted_index] = previous_steps + 1
    return float(costs[-1, -1] / steps[-1, -1])


def return_space_dtw_distance(
    actual_close: Sequence[float] | np.ndarray,
    predicted_close: Sequence[float] | np.ndarray,
    origin_close: float,
) -> float:
    return dtw_distance(
        cumulative_return_path(actual_close, origin_close),
        cumulative_return_path(predicted_close, origin_close),
    )


def z_normalized_dtw_distance(
    actual_close: Sequence[float] | np.ndarray,
    predicted_close: Sequence[float] | np.ndarray,
    origin_close: float,
) -> float:
    actual = _z_normalize_path(cumulative_return_path(actual_close, origin_close))
    predicted = _z_normalize_path(
        cumulative_return_path(predicted_close, origin_close)
    )
    return dtw_distance(actual, predicted)


def slope_sign_agreement(
    actual_close: Sequence[float] | np.ndarray,
    predicted_close: Sequence[float] | np.ndarray,
    origin_close: float,
) -> float:
    actual = cumulative_return_path(actual_close, origin_close)
    predicted = cumulative_return_path(predicted_close, origin_close)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted paths must have the same shape")
    if len(actual) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))))


def _turning_points(path: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for index in range(1, len(path) - 1):
        incoming = float(path[index] - path[index - 1])
        outgoing = float(path[index + 1] - path[index])
        if abs(incoming) < threshold or abs(outgoing) < threshold:
            continue
        if incoming > 0 and outgoing < 0:
            points.add((index, 1))
        elif incoming < 0 and outgoing > 0:
            points.add((index, -1))
    return points


def turning_point_similarity(
    actual_close: Sequence[float] | np.ndarray,
    predicted_close: Sequence[float] | np.ndarray,
    origin_close: float,
    *,
    threshold: float,
) -> float:
    if threshold < 0:
        raise ValueError("Turning-point threshold must be non-negative")
    actual = cumulative_return_path(actual_close, origin_close)
    predicted = cumulative_return_path(predicted_close, origin_close)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted paths must have the same shape")
    actual_points = _turning_points(actual, threshold)
    predicted_points = _turning_points(predicted, threshold)
    union = actual_points | predicted_points
    if not union:
        return 1.0
    return float(len(actual_points & predicted_points) / len(union))


def range_relative_error(
    actual_high: Sequence[float] | np.ndarray,
    actual_low: Sequence[float] | np.ndarray,
    predicted_high: Sequence[float] | np.ndarray,
    predicted_low: Sequence[float] | np.ndarray,
    origin_close: float,
) -> float:
    actual_range = (
        float(np.max(np.asarray(actual_high, dtype=np.float64)))
        - float(np.min(np.asarray(actual_low, dtype=np.float64)))
    ) / origin_close
    predicted_range = (
        float(np.max(np.asarray(predicted_high, dtype=np.float64)))
        - float(np.min(np.asarray(predicted_low, dtype=np.float64)))
    ) / origin_close
    return float(abs(predicted_range - actual_range) / (abs(actual_range) + 1e-12))


def compute_three_day_path_metrics(
    *,
    actual_close: Sequence[float] | np.ndarray,
    predicted_close: Sequence[float] | np.ndarray,
    origin_close: float,
    day_end_indices: Sequence[int],
    turning_point_threshold: float,
) -> dict[str, float | int]:
    actual_endpoints = three_day_endpoint_returns(
        actual_close, origin_close, day_end_indices
    )
    predicted_endpoints = three_day_endpoint_returns(
        predicted_close, origin_close, day_end_indices
    )
    metrics: dict[str, float | int] = {
        "return_path_correlation": return_path_correlation(
            actual_close, predicted_close, origin_close
        ),
        "z_normalized_dtw_distance": z_normalized_dtw_distance(
            actual_close, predicted_close, origin_close
        ),
        "return_space_dtw_distance": return_space_dtw_distance(
            actual_close, predicted_close, origin_close
        ),
        "slope_sign_agreement": slope_sign_agreement(
            actual_close, predicted_close, origin_close
        ),
        "turning_point_similarity": turning_point_similarity(
            actual_close,
            predicted_close,
            origin_close,
            threshold=turning_point_threshold,
        ),
    }
    for day_number, (actual_return, predicted_return) in enumerate(
        zip(actual_endpoints, predicted_endpoints, strict=True), start=1
    ):
        metrics[f"day{day_number}_actual_return"] = actual_return
        metrics[f"day{day_number}_predicted_return"] = predicted_return
        metrics[f"day{day_number}_actual_direction"] = direction_label(actual_return)
        metrics[f"day{day_number}_path_direction"] = direction_label(predicted_return)
        metrics[f"day{day_number}_endpoint_absolute_error"] = abs(
            predicted_return - actual_return
        )
    return metrics


def compute_three_day_record_metrics(records: pd.DataFrame) -> dict[str, object]:
    if records.empty:
        raise ValueError("Cannot compute three-day metrics from an empty table")
    output: dict[str, object] = {"samples": int(len(records)), "endpoints": {}}
    endpoints = output["endpoints"]
    assert isinstance(endpoints, dict)
    for day_number in (1, 2, 3):
        actual_direction = records[
            f"day{day_number}_actual_direction"
        ].to_numpy(dtype=np.int8)
        predicted_direction = records[
            f"day{day_number}_path_direction"
        ].to_numpy(dtype=np.int8)
        valid = actual_direction != 0
        actual_return = records[
            f"day{day_number}_actual_return"
        ].to_numpy(dtype=np.float64)
        predicted_return = records[
            f"day{day_number}_predicted_return"
        ].to_numpy(dtype=np.float64)
        if np.std(actual_return) > 0 and np.std(predicted_return) > 0:
            return_correlation = float(
                np.corrcoef(actual_return, predicted_return)[0, 1]
            )
        else:
            return_correlation = float("nan")
        confusion = {
            str(actual_label): {
                str(predicted_label): int(
                    np.sum(
                        (actual_direction == actual_label)
                        & (predicted_direction == predicted_label)
                    )
                )
                for predicted_label in (-1, 0, 1)
            }
            for actual_label in (-1, 1)
        }
        endpoints[f"day{day_number}"] = {
            "direction_samples": int(valid.sum()),
            "excluded_zero_actual_returns": int((~valid).sum()),
            "path_direction_balanced_accuracy": balanced_direction_accuracy(
                actual_direction,
                predicted_direction,
            ),
            "path_direction_accuracy": float(
                np.mean(actual_direction[valid] == predicted_direction[valid])
            )
            if valid.any()
            else float("nan"),
            "actual_direction_counts": {
                str(label): int(np.sum(actual_direction == label))
                for label in (-1, 0, 1)
            },
            "predicted_direction_counts": {
                str(label): int(np.sum(predicted_direction == label))
                for label in (-1, 0, 1)
            },
            "confusion_matrix": confusion,
            "endpoint_return_mae": float(
                np.mean(np.abs(predicted_return - actual_return))
            ),
            "endpoint_return_bias": float(
                np.mean(predicted_return - actual_return)
            ),
            "endpoint_return_correlation": return_correlation,
            "mean_up_probability": float(
                records[f"day{day_number}_up_probability"].mean()
            ),
        }

    path_columns = (
        "return_path_correlation",
        "z_normalized_dtw_distance",
        "return_space_dtw_distance",
        "slope_sign_agreement",
        "turning_point_similarity",
        "range_relative_error",
    )
    output["path"] = {
        f"mean_{column}": float(
            pd.to_numeric(records[column], errors="coerce").mean()
        )
        for column in path_columns
        if column in records
    }
    diagnostic_columns = (
        "raw_nonfinite_rate",
        "raw_ohlc_violation_rate",
        "raw_negative_flow_rate",
    )
    output["generation_diagnostics"] = {
        f"mean_{column}": float(
            pd.to_numeric(records[column], errors="coerce").mean()
        )
        for column in diagnostic_columns
        if column in records
    }
    return output


def three_day_metrics_with_instruments(
    records: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    output = {"pooled": compute_three_day_record_metrics(records)}
    for instrument, group in records.groupby("instrument", sort=True):
        output[str(instrument)] = compute_three_day_record_metrics(group)
    return output


def balanced_direction_accuracy(actual: Sequence[int], predicted: Sequence[int]) -> float:
    actual_array = np.asarray(actual, dtype=np.int8)
    predicted_array = np.asarray(predicted, dtype=np.int8)
    valid = actual_array != 0
    actual_array = actual_array[valid]
    predicted_array = predicted_array[valid]
    recalls: list[float] = []
    for label in (-1, 1):
        label_mask = actual_array == label
        if label_mask.any():
            recalls.append(float(np.mean(predicted_array[label_mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def compute_metrics(records: pd.DataFrame) -> dict[str, float | int]:
    if records.empty:
        raise ValueError("Cannot compute metrics from an empty table")
    actual_direction = records["actual_direction"].to_numpy(dtype=np.int8)
    predicted_direction = records["predicted_direction"].to_numpy(dtype=np.int8)
    valid_direction = actual_direction != 0
    actual_return = records["actual_return"].to_numpy(dtype=np.float64)
    predicted_return = records["predicted_return"].to_numpy(dtype=np.float64)

    metrics: dict[str, float | int] = {
        "samples": int(len(records)),
        "direction_samples": int(valid_direction.sum()),
        "direction_balanced_accuracy": balanced_direction_accuracy(
            actual_direction, predicted_direction
        ),
        "direction_accuracy": float(
            np.mean(actual_direction[valid_direction] == predicted_direction[valid_direction])
        ),
        "return_mae": float(np.mean(np.abs(actual_return - predicted_return))),
        "return_bias": float(np.mean(predicted_return - actual_return)),
    }
    if np.std(actual_return) > 0 and np.std(predicted_return) > 0:
        metrics["return_correlation"] = float(
            np.corrcoef(actual_return, predicted_return)[0, 1]
        )
    else:
        metrics["return_correlation"] = float("nan")

    for optional_column in (
        "close_path_correlation",
        "range_relative_error",
        "up_probability",
    ):
        if optional_column in records:
            values = pd.to_numeric(records[optional_column], errors="coerce")
            if values.notna().any():
                metrics[f"mean_{optional_column}"] = float(values.mean())
    return metrics


def metrics_with_instruments(records: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    output = {"pooled": compute_metrics(records)}
    for instrument, group in records.groupby("instrument", sort=True):
        output[str(instrument)] = compute_metrics(group)
    return output


def _actual_case_fields(case: ForecastCase) -> dict[str, object]:
    current_close = float(case.context["close"].iloc[-1])
    target_open = float(case.target["open"].iloc[0])
    target_close = float(case.target["close"].iloc[-1])
    actual_return = target_close / current_close - 1.0
    opening_gap = target_open / current_close - 1.0
    return {
        "instrument": case.instrument,
        "target_day": case.target_day,
        "current_close": current_close,
        "actual_close": target_close,
        "actual_return": actual_return,
        "actual_direction": direction_label(actual_return),
        "opening_gap": opening_gap,
        "large_opening_gap": bool(abs(opening_gap) >= 0.03),
    }


def make_naive_baselines(
    train_cases: Sequence[ForecastCase],
    evaluation_cases: Sequence[ForecastCase],
) -> dict[str, pd.DataFrame]:
    train_directions: dict[str, list[int]] = {}
    for case in train_cases:
        fields = _actual_case_fields(case)
        train_directions.setdefault(case.instrument, []).append(
            int(fields["actual_direction"])
        )

    majority_by_instrument: dict[str, int] = {}
    for instrument, directions in train_directions.items():
        nonzero = [value for value in directions if value != 0]
        up_count = sum(value == 1 for value in nonzero)
        down_count = sum(value == -1 for value in nonzero)
        majority_by_instrument[instrument] = 1 if up_count >= down_count else -1

    majority_rows: list[dict[str, object]] = []
    momentum_rows: list[dict[str, object]] = []
    for case in evaluation_cases:
        actual = _actual_case_fields(case)
        majority_direction = majority_by_instrument[case.instrument]
        majority_rows.append(
            {
                **actual,
                "predicted_return": majority_direction * 1e-12,
                "predicted_direction": majority_direction,
            }
        )

        origin_day = case.context["trading_day"].iloc[-1]
        origin = case.context.loc[case.context["trading_day"] == origin_day]
        origin_return = float(origin["close"].iloc[-1] / origin["open"].iloc[0] - 1.0)
        momentum_direction = direction_label(origin_return)
        if momentum_direction == 0:
            momentum_direction = majority_direction
        momentum_rows.append(
            {
                **actual,
                "predicted_return": origin_return,
                "predicted_direction": momentum_direction,
            }
        )

    return {
        "majority": pd.DataFrame(majority_rows),
        "momentum": pd.DataFrame(momentum_rows),
    }


def select_strongest_baseline(
    validation_baselines: Mapping[str, pd.DataFrame],
    zero_shot_validation: pd.DataFrame,
) -> str:
    candidates = {"zero_shot": zero_shot_validation, **validation_baselines}
    scores = {
        name: float(compute_metrics(records)["direction_balanced_accuracy"])
        for name, records in candidates.items()
    }
    return max(scores, key=scores.get)


def ensemble_records(record_sets: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not record_sets:
        raise ValueError("At least one record set is required")
    indexed = [records.set_index(KEY_COLUMNS).sort_index() for records in record_sets]
    reference_index = indexed[0].index
    if any(not frame.index.equals(reference_index) for frame in indexed[1:]):
        raise ValueError("Cannot ensemble record sets with different forecast cases")

    result = indexed[0].copy()
    stacked_returns = np.stack(
        [frame["predicted_return"].to_numpy(dtype=np.float64) for frame in indexed],
        axis=0,
    )
    result["predicted_return"] = stacked_returns.mean(axis=0)
    result["predicted_direction"] = [
        direction_label(value) for value in result["predicted_return"]
    ]

    if all("predicted_close" in frame for frame in indexed):
        stacked_close = np.stack(
            [frame["predicted_close"].to_numpy(dtype=np.float64) for frame in indexed],
            axis=0,
        )
        result["predicted_close"] = stacked_close.mean(axis=0)
    for scalar_column in ("up_probability", "predicted_range"):
        if all(scalar_column in frame for frame in indexed):
            stacked_values = np.stack(
                [frame[scalar_column].to_numpy(dtype=np.float64) for frame in indexed],
                axis=0,
            )
            result[scalar_column] = np.nanmean(stacked_values, axis=0)

    if all("predicted_close_path" in frame for frame in indexed):
        ensemble_paths: list[list[float]] = []
        for row_index in range(len(reference_index)):
            paths = np.stack(
                [
                    np.asarray(frame.iloc[row_index]["predicted_close_path"], dtype=np.float64)
                    for frame in indexed
                ],
                axis=0,
            )
            ensemble_paths.append(paths.mean(axis=0).tolist())
        result["predicted_close_path"] = ensemble_paths

    if all("sample_final_returns" in frame for frame in indexed):
        result["sample_final_returns"] = [
            [
                float(value)
                for frame in indexed
                for value in frame.iloc[row_index]["sample_final_returns"]
            ]
            for row_index in range(len(reference_index))
        ]

    if "predicted_close_path" in result and "actual_close_path" in result:
        correlations: list[float] = []
        for actual_values, predicted_values in zip(
            result["actual_close_path"],
            result["predicted_close_path"],
            strict=True,
        ):
            actual = np.asarray(actual_values, dtype=np.float64)
            predicted = np.asarray(predicted_values, dtype=np.float64)
            if (
                len(actual) < 2
                or np.std(actual) == 0
                or np.std(predicted) == 0
            ):
                correlations.append(float("nan"))
            else:
                correlations.append(float(np.corrcoef(actual, predicted)[0, 1]))
        result["close_path_correlation"] = correlations

    if "predicted_range" in result and "actual_range" in result:
        result["range_relative_error"] = (
            (result["predicted_range"] - result["actual_range"]).abs()
            / (result["actual_range"] + 1e-12)
        )
    return result.reset_index()


def paired_block_bootstrap_improvement(
    model_records: pd.DataFrame,
    baseline_records: pd.DataFrame,
    *,
    iterations: int = 2_000,
    block_days: int = 5,
    seed: int = 0,
) -> dict[str, float | int]:
    model = model_records.set_index(KEY_COLUMNS).sort_index()
    baseline = baseline_records.set_index(KEY_COLUMNS).sort_index()
    common_index = model.index.intersection(baseline.index)
    if len(common_index) == 0:
        raise ValueError("Model and baseline have no paired forecasts")
    model = model.loc[common_index]
    baseline = baseline.loc[common_index]

    dates = sorted({pd.Timestamp(day) for _, day in common_index})
    if block_days < 1 or block_days > len(dates):
        raise ValueError("Invalid bootstrap block length")
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(len(dates) / block_days))
    improvements = np.empty(iterations, dtype=np.float64)

    model_by_day = {
        day: model.loc[model.index.get_level_values("target_day") == day]
        for day in dates
    }
    baseline_by_day = {
        day: baseline.loc[baseline.index.get_level_values("target_day") == day]
        for day in dates
    }

    for iteration in range(iterations):
        starts = rng.integers(0, len(dates), size=blocks_needed)
        sampled_days: list[pd.Timestamp] = []
        for start in starts:
            sampled_days.extend(
                dates[(int(start) + offset) % len(dates)]
                for offset in range(block_days)
            )
        sampled_days = sampled_days[: len(dates)]

        model_actual: list[int] = []
        model_predicted: list[int] = []
        baseline_predicted: list[int] = []
        for day in sampled_days:
            model_day = model_by_day[day]
            baseline_day = baseline_by_day[day]
            model_actual.extend(model_day["actual_direction"].astype(int).tolist())
            model_predicted.extend(model_day["predicted_direction"].astype(int).tolist())
            baseline_predicted.extend(
                baseline_day["predicted_direction"].astype(int).tolist()
            )
        improvements[iteration] = balanced_direction_accuracy(
            model_actual, model_predicted
        ) - balanced_direction_accuracy(model_actual, baseline_predicted)

    point_estimate = balanced_direction_accuracy(
        model["actual_direction"].to_numpy(dtype=np.int8),
        model["predicted_direction"].to_numpy(dtype=np.int8),
    ) - balanced_direction_accuracy(
        model["actual_direction"].to_numpy(dtype=np.int8),
        baseline["predicted_direction"].to_numpy(dtype=np.int8),
    )
    return {
        "samples": int(len(common_index)),
        "unique_days": int(len(dates)),
        "iterations": int(iterations),
        "block_days": int(block_days),
        "point_estimate": point_estimate,
        "ci_lower_95": float(np.quantile(improvements, 0.025)),
        "ci_upper_95": float(np.quantile(improvements, 0.975)),
        "probability_improvement_positive": float(np.mean(improvements > 0)),
    }
