"""Mandatory per-fold direction-comparison evaluation artifacts.

This module owns the cross-strategy visual evaluation contract.  A fold is not
considered evaluated until its case-level direction comparison PNG *and* its
machine-readable JSON companion have both been written successfully.

The renderer deliberately validates the paired comparison before drawing.  It
is easy for aggregate metrics to hide a dropped contract or a split mismatch;
the checks here make that class of error a hard failure instead.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "kronos-evaluation-matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DIRECTION_COMPARISON_CONTRACT_VERSION = "direction-comparison-v1"
_REQUIRED_COLUMNS = frozenset(
    {
        "case_key",
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "actual_direction",
        "predicted_direction",
        "probability_up",
        "model",
    }
)
_MATCHED_CASE_COLUMNS = (
    "product",
    "target_end_day",
    "target_contract_id",
    "actual_direction",
)
_DIRECTION_COLORS = {
    -1: "#2e8b57",  # green: down
    0: "#9ca3af",  # gray: zero / invalid direction
    1: "#d94841",  # red: up
}
_PROBABILITY_SERIES_STYLES = (
    ("candidate", "candidate P(up)", "#1f77b4", "-", "s"),
    ("baseline", "baseline P(up)", "#7c3aed", "--", "D"),
)


class DirectionComparisonError(RuntimeError):
    """A fold cannot satisfy the mandatory direction-plot contract."""


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Required case-level evaluation files for one fold or ablation fold."""

    json_path: Path
    figure_paths: tuple[Path, ...]
    evaluation_contract_version: str = DIRECTION_COMPARISON_CONTRACT_VERSION

    @property
    def report_path(self) -> Path:
        """Compatibility alias for callers that call the JSON a report."""

        return self.json_path

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluation_contract_version": self.evaluation_contract_version,
            "report": str(self.json_path),
            "figures": [str(path) for path in self.figure_paths],
        }


@dataclass(frozen=True)
class _Comparison:
    candidate_model: str
    baseline_model: str
    candidate: pd.DataFrame
    baseline: pd.DataFrame
    merged: pd.DataFrame
    metrics: Mapping[str, Mapping[str, object]]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _as_frame(records: object, *, model_name: str, fold_id: str) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        try:
            frame = pd.DataFrame(records)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DirectionComparisonError(
                f"{model_name} records cannot be converted to a dataframe"
            ) from exc
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise DirectionComparisonError(
            f"{model_name} records miss required direction-plot columns: {missing!r}"
        )
    if frame.empty:
        raise DirectionComparisonError(f"{model_name} records are empty")
    if frame["case_key"].isna().any() or (frame["case_key"].astype(str).str.len() == 0).any():
        raise DirectionComparisonError(f"{model_name} records contain an empty case_key")
    if frame["case_key"].duplicated().any():
        duplicates = frame.loc[frame["case_key"].duplicated(), "case_key"].astype(str).tolist()
        raise DirectionComparisonError(
            f"{model_name} records contain duplicate case keys: {duplicates[:3]!r}"
        )
    actual_model_values = set(frame["model"].astype(str))
    if actual_model_values != {model_name}:
        raise DirectionComparisonError(
            f"{model_name} records must identify exactly model={model_name!r}, "
            f"got {sorted(actual_model_values)!r}"
        )
    actual_fold_values = set(frame["fold_id"].astype(str))
    if actual_fold_values != {fold_id}:
        raise DirectionComparisonError(
            f"{model_name} records must identify exactly fold_id={fold_id!r}, "
            f"got {sorted(actual_fold_values)!r}"
        )
    for column in ("product", "target_contract_id"):
        if frame[column].isna().any() or (frame[column].astype(str).str.len() == 0).any():
            raise DirectionComparisonError(f"{model_name} records contain an empty {column}")
    parsed_days = pd.to_datetime(frame["target_end_day"], errors="coerce")
    if parsed_days.isna().any():
        raise DirectionComparisonError(f"{model_name} records contain an invalid target_end_day")
    frame["target_end_day"] = parsed_days.dt.normalize()
    for column in ("actual_direction", "predicted_direction"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not set(numeric.astype(int)).issubset({-1, 0, 1}):
            raise DirectionComparisonError(
                f"{model_name} {column} must use only -1, 0, or 1 direction values"
            )
        frame[column] = numeric.astype(np.int8)
    probabilities = pd.to_numeric(frame["probability_up"], errors="coerce")
    invalid_probabilities = probabilities.notna() & (
        (probabilities < 0.0) | (probabilities > 1.0)
    )
    if invalid_probabilities.any():
        raise DirectionComparisonError(
            f"{model_name} probability_up must be in [0, 1] when present"
        )
    frame["probability_up"] = probabilities
    return frame.sort_values(
        ["product", "target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _direction_metrics(records: pd.DataFrame) -> dict[str, object]:
    valid = records.loc[records["actual_direction"] != 0]
    samples = len(valid)
    if not samples:
        return {
            "cases": int(len(records)),
            "valid_direction_cases": 0,
            "balanced_accuracy": None,
            "accuracy": None,
            "up_cases": 0,
            "down_cases": 0,
        }
    actual = valid["actual_direction"].to_numpy(dtype=np.int8)
    predicted = valid["predicted_direction"].to_numpy(dtype=np.int8)
    up = actual == 1
    down = actual == -1
    balanced_accuracy: float | None
    if up.any() and down.any():
        balanced_accuracy = float(
            0.5 * (np.mean(predicted[up] == 1) + np.mean(predicted[down] == -1))
        )
    else:
        balanced_accuracy = None
    return {
        "cases": int(len(records)),
        "valid_direction_cases": int(samples),
        "balanced_accuracy": balanced_accuracy,
        "accuracy": float(np.mean(predicted == actual)),
        "up_cases": int(up.sum()),
        "down_cases": int(down.sum()),
    }


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.1%}"


def _validate_comparison(
    records_by_model: Mapping[str, object],
    *,
    fold_id: str,
    candidate_model: str,
    baseline_model: str,
) -> _Comparison:
    if candidate_model == baseline_model:
        raise DirectionComparisonError("candidate_model and baseline_model must differ")
    missing_models = {candidate_model, baseline_model}.difference(records_by_model)
    if missing_models:
        raise DirectionComparisonError(
            f"Direction comparison is missing model records: {sorted(missing_models)!r}"
        )
    candidate = _as_frame(
        records_by_model[candidate_model], model_name=candidate_model, fold_id=fold_id
    )
    baseline = _as_frame(
        records_by_model[baseline_model], model_name=baseline_model, fold_id=fold_id
    )
    candidate_indexed = candidate.set_index("case_key", verify_integrity=True)
    baseline_indexed = baseline.set_index("case_key", verify_integrity=True)
    if set(candidate_indexed.index) != set(baseline_indexed.index):
        missing = sorted(set(candidate_indexed.index).difference(baseline_indexed.index))
        extra = sorted(set(baseline_indexed.index).difference(candidate_indexed.index))
        raise DirectionComparisonError(
            "candidate and baseline case keys must match exactly "
            f"(missing_baseline={len(missing)}, extra_baseline={len(extra)})"
        )
    baseline_indexed = baseline_indexed.loc[candidate_indexed.index]
    for column in _MATCHED_CASE_COLUMNS:
        candidate_values = candidate_indexed[column]
        baseline_values = baseline_indexed[column]
        if not candidate_values.equals(baseline_values):
            raise DirectionComparisonError(
                "candidate and baseline disagree for the same case on "
                f"{column}; paired plots require identical case provenance"
            )
    if not bool((candidate["actual_direction"] != 0).any()):
        raise DirectionComparisonError(
            f"fold {fold_id} contains no valid non-zero direction cases to evaluate"
        )
    merged = candidate_indexed[
        [
            "fold_id",
            "product",
            "target_end_day",
            "target_contract_id",
            "actual_direction",
            "predicted_direction",
            "probability_up",
        ]
    ].rename(
        columns={
            "predicted_direction": "candidate_direction",
            "probability_up": "candidate_probability_up",
        }
    )
    merged["baseline_direction"] = baseline_indexed["predicted_direction"]
    merged["baseline_probability_up"] = baseline_indexed["probability_up"]
    merged = merged.reset_index().sort_values(
        ["product", "target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)
    return _Comparison(
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        candidate=candidate,
        baseline=baseline,
        merged=merged,
        metrics={
            "candidate": _direction_metrics(candidate),
            "baseline": _direction_metrics(baseline),
        },
    )


def _fold_destination(output_dir: str | Path, fold_id: str) -> Path:
    destination = Path(output_dir)
    return destination if destination.name == fold_id else destination / fold_id


def _draw_product_tracks(
    axis: plt.Axes,
    records: pd.DataFrame,
    *,
    candidate_label: str,
    baseline_label: str,
    metrics: Mapping[str, Mapping[str, object]],
    zone_label: str | None = None,
) -> None:
    """Draw a continuous probability line plot when no path forecast exists.

    P1 probes are direction classifiers, so they do not produce a future price
    path.  Their honest continuous output is P(up).  Realized directions are
    shown only as top/bottom event markers; they are deliberately not connected
    into a misleading 0/1 "return" line.  Path-producing phases use the
    close-to-close-return renderer below instead.
    """

    positions = np.arange(len(records), dtype=np.float64)
    actual = records["actual_direction"].to_numpy(dtype=np.int8)
    valid = actual != 0
    values = {
        "candidate": pd.to_numeric(
            records["candidate_probability_up"], errors="coerce"
        ).to_numpy(dtype=np.float64),
        "baseline": pd.to_numeric(
            records["baseline_probability_up"], errors="coerce"
        ).to_numpy(dtype=np.float64),
    }
    for role, _, line_color, line_style, marker in _PROBABILITY_SERIES_STYLES:
        y_values = values[role]
        axis.plot(
            positions,
            y_values,
            color=line_color,
            linestyle=line_style,
            linewidth=1.7,
            alpha=0.85,
            zorder=2,
        )
        finite = np.isfinite(y_values)
        axis.scatter(
            positions[finite],
            y_values[finite],
            color=line_color,
            marker=marker,
            s=20,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        direction_column = f"{role}_direction"
        direction = records[direction_column].to_numpy(dtype=np.int8)
        incorrect = valid & (
            direction != actual
        )
        wrong_positions = np.flatnonzero(incorrect)
        if len(wrong_positions):
            axis.scatter(
                positions[wrong_positions],
                y_values[wrong_positions],
                marker="x",
                s=38,
                color="black",
                linewidths=1.0,
                zorder=4,
            )
    for direction, vertical_position, marker, color in (
        (1, 1.04, "^", _DIRECTION_COLORS[1]),
        (-1, -0.04, "v", _DIRECTION_COLORS[-1]),
        (0, 0.5, "o", _DIRECTION_COLORS[0]),
    ):
        event_positions = np.flatnonzero(actual == direction)
        if not len(event_positions):
            continue
        axis.scatter(
            positions[event_positions],
            np.full(len(event_positions), vertical_position),
            color=color,
            marker=marker,
            s=20,
            edgecolors="white",
            linewidths=0.3,
            zorder=4,
        )
    product = str(records["product"].iloc[0])
    candidate_metrics = metrics["candidate"]
    baseline_metrics = metrics["baseline"]
    title_prefix = f"{zone_label} · " if zone_label else ""
    axis.set_title(
        f"{title_prefix}{product} · n={len(records)} · "
        f"{candidate_label} BA {_format_metric(candidate_metrics['balanced_accuracy'])} "
        f"Acc {_format_metric(candidate_metrics['accuracy'])} · "
        f"{baseline_label} BA {_format_metric(baseline_metrics['balanced_accuracy'])} "
        f"Acc {_format_metric(baseline_metrics['accuracy'])}\n"
        "direction-probability diagnostic (no close-path output)",
        fontsize=9,
    )
    tick_count = min(6, len(records))
    tick_positions = np.unique(np.linspace(0, len(records) - 1, tick_count, dtype=int))
    tick_labels = [
        pd.Timestamp(records["target_end_day"].iloc[position]).strftime("%m-%d")
        for position in tick_positions
    ]
    axis.axhline(0.5, color="#9ca3af", linestyle=":", linewidth=0.9, zorder=1)
    axis.set_yticks((0.0, 0.5, 1.0), ("0.0", "threshold", "1.0"))
    axis.set_xticks(tick_positions, tick_labels, fontsize=8)
    axis.set_xlabel("case order: target date / contract / case key", fontsize=8)
    axis.set_xlim(-0.45, len(records) - 0.55)
    axis.set_ylim(-0.10, 1.10)
    axis.set_ylabel("Predicted P(up)", fontsize=8)
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _has_nested_values(series: pd.Series) -> bool:
    def nonempty(value: object) -> bool:
        if value is None or value is pd.NA:
            return False
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return False
        try:
            return np.asarray(value).size > 0
        except (TypeError, ValueError):
            return False

    return bool(series.map(nonempty).all())


def _candidate_has_path_payload(records: pd.DataFrame) -> bool:
    """Return whether every candidate record can render a V2-style path."""

    path_fields = {"origin_close", "actual_path", "predicted_path", "sample_paths"}
    present = path_fields.intersection(records.columns)
    if not present:
        return False
    required = {"origin_close", "actual_path"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise DirectionComparisonError(
            "Candidate records partially declare a path forecast but miss "
            f"required fields: {missing!r}"
        )
    if not _has_nested_values(records["actual_path"]):
        raise DirectionComparisonError("Candidate path plot has an empty actual_path")
    if records["origin_close"].isna().any():
        raise DirectionComparisonError("Candidate path plot has an empty origin_close")
    available_prediction_fields = [
        field
        for field in ("sample_paths", "predicted_path")
        if field in records.columns and _has_nested_values(records[field])
    ]
    if not available_prediction_fields:
        raise DirectionComparisonError(
            "Candidate path plot needs non-empty sample_paths or predicted_path"
        )
    return True


def _close_series(value: object, *, field: str, case_key: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DirectionComparisonError(f"{case_key} has a non-numeric {field}") from exc
    if array.ndim != 2 or array.shape[1] < 4 or array.shape[0] < 1:
        raise DirectionComparisonError(
            f"{case_key} {field} must have shape [prediction_bar, feature>=4]"
        )
    close = array[:, 3]
    if not np.isfinite(close).all():
        raise DirectionComparisonError(f"{case_key} {field} close values are not finite")
    return close


def _bar_returns(
    close: np.ndarray,
    *,
    origin_close: float,
    case_key: str,
) -> np.ndarray:
    """Calculate each bar's move from the immediately preceding close."""

    if not math.isfinite(origin_close) or origin_close <= 0.0:
        raise DirectionComparisonError(f"{case_key} origin_close must be positive and finite")
    if close.ndim != 1 or len(close) < 1 or not np.isfinite(close).all() or (close <= 0.0).any():
        raise DirectionComparisonError(f"{case_key} close path must be positive and finite")
    previous = np.concatenate(([origin_close], close[:-1]))
    return close / previous - 1.0


def _prediction_return_distribution(
    record: pd.Series,
    *,
    origin_close: float,
    case_key: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return median, q10 and q90 *bar returns* from one model record."""

    sample_value = record.get("sample_paths")
    if sample_value is not None and sample_value is not pd.NA:
        try:
            samples = np.asarray(sample_value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DirectionComparisonError(f"{case_key} has non-numeric sample_paths") from exc
        if samples.ndim == 3 and samples.shape[0] >= 1 and samples.shape[1] >= 1 and samples.shape[2] >= 4:
            close = samples[:, :, 3]
            if not np.isfinite(close).all() or (close <= 0.0).any():
                raise DirectionComparisonError(
                    f"{case_key} sample_paths close values must be positive and finite"
                )
            previous = np.concatenate(
                (np.full((close.shape[0], 1), origin_close, dtype=np.float64), close[:, :-1]),
                axis=1,
            )
            returns = close / previous - 1.0
            return (
                np.quantile(returns, 0.5, axis=0),
                np.quantile(returns, 0.1, axis=0),
                np.quantile(returns, 0.9, axis=0),
            )
    predicted_value = record.get("predicted_path")
    if predicted_value is not None and predicted_value is not pd.NA:
        close = _close_series(predicted_value, field="predicted_path", case_key=case_key)
        return _bar_returns(close, origin_close=origin_close, case_key=case_key), None, None
    raise DirectionComparisonError(
        f"{case_key} needs a numeric sample_paths or predicted_path forecast"
    )


def _prediction_close_distribution(
    record: pd.Series,
    *,
    case_key: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return median, q10 and q90 close paths from one model record."""

    sample_value = record.get("sample_paths")
    if sample_value is not None and sample_value is not pd.NA:
        try:
            samples = np.asarray(sample_value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DirectionComparisonError(f"{case_key} has non-numeric sample_paths") from exc
        if samples.ndim == 3 and samples.shape[0] >= 1 and samples.shape[1] >= 1 and samples.shape[2] >= 4:
            close = samples[:, :, 3]
            if not np.isfinite(close).all() or (close <= 0.0).any():
                raise DirectionComparisonError(
                    f"{case_key} sample_paths close values must be positive and finite"
                )
            return (
                np.quantile(close, 0.5, axis=0),
                np.quantile(close, 0.1, axis=0),
                np.quantile(close, 0.9, axis=0),
            )
    predicted_value = record.get("predicted_path")
    if predicted_value is not None and predicted_value is not pd.NA:
        return _close_series(predicted_value, field="predicted_path", case_key=case_key), None, None
    raise DirectionComparisonError(
        f"{case_key} needs a numeric sample_paths or predicted_path forecast"
    )


def _optional_prediction_returns(
    record: pd.Series,
    *,
    origin_close: float,
    case_key: str,
) -> np.ndarray | None:
    """Read a baseline trajectory when that baseline actually predicts one."""

    has_samples = "sample_paths" in record.index and record.get("sample_paths") is not None
    has_path = "predicted_path" in record.index and record.get("predicted_path") is not None
    if not has_samples and not has_path:
        return None
    try:
        median, _, _ = _prediction_return_distribution(
            record,
            origin_close=origin_close,
            case_key=case_key,
        )
    except DirectionComparisonError:
        return None
    return median


def _optional_prediction_close(record: pd.Series, *, case_key: str) -> np.ndarray | None:
    """Read a baseline close trajectory only if it genuinely predicts one."""

    has_samples = "sample_paths" in record.index and record.get("sample_paths") is not None
    has_path = "predicted_path" in record.index and record.get("predicted_path") is not None
    if not has_samples and not has_path:
        return None
    try:
        median, _, _ = _prediction_close_distribution(record, case_key=case_key)
    except DirectionComparisonError:
        return None
    return median


def _direction_text(value: object) -> str:
    direction = int(value)
    return "up" if direction > 0 else "down" if direction < 0 else "flat"


def _example_positions(record_count: int) -> tuple[int, ...]:
    return tuple(sorted({0, record_count // 2, record_count - 1}))


def _draw_path_example(
    axis: plt.Axes,
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    candidate_label: str,
    baseline_label: str,
    product_metrics: Mapping[str, Mapping[str, object]],
    example_number: int,
    total_examples: int,
    value_kind: str,
) -> None:
    """Draw one actual/predicted close or close-to-close-return example."""

    case_key = str(candidate["case_key"])
    origin_close = float(candidate["origin_close"])
    if not math.isfinite(origin_close) or origin_close <= 0.0:
        raise DirectionComparisonError(f"{case_key} origin_close must be positive and finite")
    actual_close = _close_series(candidate["actual_path"], field="actual_path", case_key=case_key)
    if value_kind == "bar_return":
        candidate_values, candidate_q10, candidate_q90 = _prediction_return_distribution(
            candidate,
            origin_close=origin_close,
            case_key=case_key,
        )
        actual_values = _bar_returns(
            actual_close, origin_close=origin_close, case_key=case_key
        )
        baseline_values = _optional_prediction_returns(
            baseline,
            origin_close=origin_close,
            case_key=case_key,
        )
        y_label = "Bar return: close[t] / close[t−1] − 1"
    elif value_kind == "close_price":
        candidate_values, candidate_q10, candidate_q90 = _prediction_close_distribution(
            candidate,
            case_key=case_key,
        )
        actual_values = actual_close
        baseline_values = _optional_prediction_close(baseline, case_key=case_key)
        y_label = "Close price"
    else:
        raise ValueError(f"Unsupported path plot value kind: {value_kind}")
    length = min(len(actual_values), len(candidate_values))
    if length < 1:
        raise DirectionComparisonError(f"{case_key} has no overlapping actual/predicted bars")
    x_values = np.arange(1, length + 1)
    axis.plot(x_values, actual_values[:length], color="black", label="actual", linewidth=2.0)
    axis.plot(
        x_values,
        candidate_values[:length],
        color="#1f77b4",
        label=candidate_label,
        linewidth=2.0,
    )
    if candidate_q10 is not None and candidate_q90 is not None:
        axis.fill_between(
            x_values,
            candidate_q10[:length],
            candidate_q90[:length],
            color="#1f77b4",
            alpha=0.2,
            label=f"{candidate_label} 10–90%",
        )
    if baseline_values is not None:
        baseline_length = min(length, len(baseline_values))
        axis.plot(
            x_values[:baseline_length],
            baseline_values[:baseline_length],
            color="#7c3aed",
            label=baseline_label,
            linewidth=1.7,
            linestyle="--",
        )
    else:
        axis.text(
            0.02,
            0.03,
            f"{baseline_label}: direction-only\nP(up)={float(baseline['probability_up']):.2f}",
            transform=axis.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#d1d5db"},
        )
    for day_end in list(candidate.get("day_end_indices", ()))[0:-1]:
        if int(day_end) < length:
            axis.axvline(float(day_end) + 1.5, color="gray", linestyle="--", linewidth=0.8)
    if value_kind == "bar_return":
        axis.axhline(0.0, color="gray", linewidth=0.8)
    product = str(candidate["product"])
    date = pd.Timestamp(candidate["target_end_day"]).strftime("%Y-%m-%d")
    candidate_metrics = product_metrics["candidate"]
    baseline_metrics = product_metrics["baseline"]
    axis.set_title(
        f"{product} {date} · example {example_number}/{total_examples}\n"
        f"actual={_direction_text(candidate['actual_direction'])}; "
        f"{candidate_label}={_direction_text(candidate['predicted_direction'])}; "
        f"{baseline_label}={_direction_text(baseline['predicted_direction'])}\n"
        f"n={candidate_metrics['cases']} · {candidate_label} BA "
        f"{_format_metric(candidate_metrics['balanced_accuracy'])} · "
        f"{baseline_label} BA {_format_metric(baseline_metrics['balanced_accuracy'])}",
        fontsize=8,
    )
    axis.set_xlabel("Target hourly bar")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.2)


def _render_path_examples(
    comparison: _Comparison,
    *,
    stage: str,
    fold: str,
    value_kind: str,
) -> plt.Figure:
    """Render a one-fold close or close-to-close-return figure per product."""

    products = sorted(comparison.candidate["product"].astype(str).unique())
    product_rows = {
        product: comparison.candidate.loc[
            comparison.candidate["product"].astype(str) == product
        ].reset_index(drop=True)
        for product in products
    }
    max_examples = max(len(_example_positions(len(rows))) for rows in product_rows.values())
    figure, axes = plt.subplots(
        len(products),
        max_examples,
        figsize=(5.4 * max_examples, 4.1 * len(products)),
        squeeze=False,
        sharey=False,
    )
    baseline_by_key = comparison.baseline.set_index("case_key", verify_integrity=True)
    for row_index, product in enumerate(products):
        records = product_rows[product]
        positions = _example_positions(len(records))
        product_metrics = {
            side: _direction_metrics(
                frame.loc[frame["product"].astype(str) == product]
            )
            for side, frame in (("candidate", comparison.candidate), ("baseline", comparison.baseline))
        }
        for column_index in range(max_examples):
            axis = axes[row_index, column_index]
            if column_index >= len(positions):
                axis.axis("off")
                continue
            candidate = records.iloc[positions[column_index]]
            baseline = baseline_by_key.loc[str(candidate["case_key"])]
            _draw_path_example(
                axis,
                candidate,
                baseline,
                candidate_label=comparison.candidate_model,
                baseline_label=comparison.baseline_model,
                product_metrics=product_metrics,
                example_number=column_index + 1,
                total_examples=len(positions),
                value_kind=value_kind,
            )
            if row_index == 0 and column_index == 0:
                axis.legend(loc="best", fontsize=7)
    title = (
        "close-to-close actual versus predicted returns"
        if value_kind == "bar_return"
        else "actual versus predicted close prices"
    )
    figure.suptitle(
        f"{stage} · {fold} · {title}",
        fontsize=13,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _add_legend(figure: plt.Figure) -> None:
    handles: list[object] = [
        Line2D([], [], color="#1f77b4", marker="s", linestyle="-", label="candidate P(up)"),
        Line2D([], [], color="#7c3aed", marker="D", linestyle="--", label="baseline P(up)"),
        Line2D([], [], color="#9ca3af", linestyle=":", label="P(up)=0.5"),
        Line2D([], [], color=_DIRECTION_COLORS[1], marker="^", linestyle="None", label="actual up"),
        Line2D([], [], color=_DIRECTION_COLORS[-1], marker="v", linestyle="None", label="actual down"),
        Line2D([], [], color=_DIRECTION_COLORS[0], marker="o", linestyle="None", label="actual flat"),
        Line2D([], [], color="black", marker="x", linestyle="None", label="wrong prediction"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncols=4,
        fontsize=8,
    )


def _verify_artifacts(artifacts: EvaluationArtifacts) -> EvaluationArtifacts:
    for path in (artifacts.json_path, *artifacts.figure_paths):
        if not path.is_file() or path.stat().st_size == 0:
            raise DirectionComparisonError(f"Required evaluation artifact was not written: {path}")
    return artifacts


def _comparison_payload(comparison: _Comparison) -> dict[str, object]:
    return {
        "candidate_model": comparison.candidate_model,
        "baseline_model": comparison.baseline_model,
        "case_coverage": {
            "cases": int(len(comparison.merged)),
            "by_product": {
                str(product): int(len(group))
                for product, group in comparison.merged.groupby("product", sort=True)
            },
        },
        "metrics": dict(comparison.metrics),
        "records": {
            "candidate": comparison.candidate.to_dict("records"),
            "baseline": comparison.baseline.to_dict("records"),
        },
    }


def render_fold_direction_comparison(
    records_by_model: Mapping[str, object],
    *,
    fold_id: str,
    candidate_model: str,
    baseline_model: str,
    output_dir: str | Path,
    stage: str,
    metadata: Mapping[str, object],
) -> EvaluationArtifacts:
    """Render one mandatory actual/candidate/baseline plot for a fold.

    ``output_dir`` is normally the stage's ``evaluation`` directory.  The
    function creates ``<output_dir>/<fold_id>/prediction_vs_actual.{png,json}``.
    When the candidate contains full generated paths it additionally writes
    ``close_price_comparison.png``; passing a directory already named
    ``fold_XX`` is also supported for an explicit destination.
    """

    fold = str(fold_id)
    comparison = _validate_comparison(
        records_by_model,
        fold_id=fold,
        candidate_model=str(candidate_model),
        baseline_model=str(baseline_model),
    )
    destination = _fold_destination(output_dir, fold)
    destination.mkdir(parents=True, exist_ok=True)
    path_mode = _candidate_has_path_payload(comparison.candidate)
    visualization_kind = (
        "close_to_close_return_examples" if path_mode else "direction_probability_diagnostic"
    )
    figure_paths: tuple[Path, ...]
    if path_mode:
        return_figure = _render_path_examples(
            comparison,
            stage=str(stage),
            fold=fold,
            value_kind="bar_return",
        )
        close_figure = _render_path_examples(
            comparison,
            stage=str(stage),
            fold=fold,
            value_kind="close_price",
        )
        figure_path = destination / "prediction_vs_actual.png"
        close_price_path = destination / "close_price_comparison.png"
        try:
            return_figure.savefig(figure_path, dpi=180, bbox_inches="tight")
            close_figure.savefig(close_price_path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(return_figure)
            plt.close(close_figure)
        figure_paths = (figure_path, close_price_path)
    else:
        products = sorted(comparison.merged["product"].astype(str).unique())
        max_cases = max(
            len(comparison.merged.loc[comparison.merged["product"].astype(str) == product])
            for product in products
        )
        figure, axes = plt.subplots(
            len(products),
            1,
            figsize=(
                max(12.0, min(24.0, 8.0 + max_cases / 20.0)),
                max(4.8, 3.5 * len(products)),
            ),
            squeeze=False,
        )
        try:
            for axis, product in zip(axes[:, 0], products, strict=True):
                product_records = comparison.merged.loc[
                    comparison.merged["product"].astype(str) == product
                ].reset_index(drop=True)
                product_metrics = {
                    side: _direction_metrics(
                        frame.loc[frame["product"].astype(str) == product]
                    )
                    for side, frame in (
                        ("candidate", comparison.candidate),
                        ("baseline", comparison.baseline),
                    )
                }
                _draw_product_tracks(
                    axis,
                    product_records,
                    candidate_label=comparison.candidate_model,
                    baseline_label=comparison.baseline_model,
                    metrics=product_metrics,
                )
            figure.suptitle(
                f"{stage} · {fold} · candidate / baseline direction-probability diagnostic",
                fontsize=13,
                y=0.98,
            )
            figure.subplots_adjust(
                top=0.82 if len(products) == 1 else 0.90,
                bottom=0.18,
                hspace=0.42,
            )
            _add_legend(figure)
            figure_path = destination / "prediction_vs_actual.png"
            figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
        figure_paths = (figure_path,)
    json_path = destination / "prediction_vs_actual.json"
    _write_json(
        json_path,
        {
            "evaluation_contract_version": DIRECTION_COMPARISON_CONTRACT_VERSION,
            "stage": str(stage),
            "fold_id": fold,
            "metadata": dict(metadata),
            "visualization": {
                "kind": visualization_kind,
                "additional_kind": "close_price_examples" if path_mode else None,
            },
            "comparison": _comparison_payload(comparison),
            "artifacts": {
                "png": str(figure_path),
                "close_price_png": str(close_price_path) if path_mode else None,
                "json": str(json_path),
            },
        },
    )
    return _verify_artifacts(EvaluationArtifacts(json_path, figure_paths))


def render_fold_ablation_direction_comparison(
    records_by_model: Mapping[str, object],
    *,
    fold_id: str,
    comparisons: Mapping[str, tuple[str, str]],
    output_dir: str | Path,
    stage: str,
    metadata: Mapping[str, object],
) -> EvaluationArtifacts:
    """Render a multi-granularity ablation in one PNG, partitioned by comparison.

    ``comparisons`` maps a visible zone name (for example ``"shared"``) to a
    ``(candidate_model, baseline_model)`` pair.  All zones must use identical
    evaluation case keys, which prevents a granularity result from silently
    benefiting from different coverage.
    """

    if not comparisons:
        raise DirectionComparisonError("An ablation plot requires at least one comparison zone")
    fold = str(fold_id)
    validated = {
        str(name): _validate_comparison(
            records_by_model,
            fold_id=fold,
            candidate_model=str(pair[0]),
            baseline_model=str(pair[1]),
        )
        for name, pair in comparisons.items()
    }
    first = next(iter(validated.values()))
    first_keys = set(first.merged["case_key"])
    first_actual = first.merged.set_index("case_key")[list(_MATCHED_CASE_COLUMNS)]
    for name, comparison in validated.items():
        if set(comparison.merged["case_key"]) != first_keys:
            raise DirectionComparisonError(
                f"Ablation zone {name!r} does not use the same evaluation case keys"
            )
        compared_actual = comparison.merged.set_index("case_key").loc[
            first_actual.index, list(_MATCHED_CASE_COLUMNS)
        ]
        if not first_actual.equals(compared_actual):
            raise DirectionComparisonError(
                f"Ablation zone {name!r} disagrees on shared-case provenance"
            )
    destination = _fold_destination(output_dir, fold)
    destination.mkdir(parents=True, exist_ok=True)
    products = sorted(first.merged["product"].astype(str).unique())
    comparison_items = list(validated.items())
    max_cases = max(
        len(first.merged.loc[first.merged["product"].astype(str) == product])
        for product in products
    )
    figure, axes = plt.subplots(
        len(products),
        len(comparison_items),
        figsize=(
            max(15.0, min(34.0, 11.0 * len(comparison_items) + max_cases / 25.0)),
            max(4.8, 3.5 * len(products)),
        ),
        squeeze=False,
    )
    try:
        for row, product in enumerate(products):
            for column, (zone, comparison) in enumerate(comparison_items):
                product_records = comparison.merged.loc[
                    comparison.merged["product"].astype(str) == product
                ].reset_index(drop=True)
                product_metrics = {
                    side: _direction_metrics(
                        frame.loc[frame["product"].astype(str) == product]
                    )
                    for side, frame in (
                        ("candidate", comparison.candidate),
                        ("baseline", comparison.baseline),
                    )
                }
                _draw_product_tracks(
                    axes[row, column],
                    product_records,
                    candidate_label=comparison.candidate_model,
                    baseline_label=comparison.baseline_model,
                    metrics=product_metrics,
                    zone_label=zone,
                )
        figure.suptitle(
            f"{stage} · {fold} · direction-probability ablation (no close-path output)",
            fontsize=13,
            y=0.98,
        )
        figure.subplots_adjust(
            top=0.82 if len(products) == 1 else 0.90,
            bottom=0.18,
            hspace=0.42,
            wspace=0.27,
        )
        _add_legend(figure)
        figure_path = destination / "prediction_vs_actual.png"
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    json_path = destination / "prediction_vs_actual.json"
    _write_json(
        json_path,
        {
            "evaluation_contract_version": DIRECTION_COMPARISON_CONTRACT_VERSION,
            "stage": str(stage),
            "fold_id": fold,
            "metadata": dict(metadata),
            "visualization": {"kind": "direction_probability_diagnostic"},
            "comparisons": {
                name: _comparison_payload(comparison)
                for name, comparison in comparison_items
            },
            "artifacts": {"png": str(figure_path), "json": str(json_path)},
        },
    )
    return _verify_artifacts(EvaluationArtifacts(json_path, (figure_path,)))


def write_direction_stage_report(
    output_dir: str | Path,
    *,
    stage: str,
    fold_artifacts: Mapping[str, EvaluationArtifacts],
    metadata: Mapping[str, object],
) -> Path:
    """Write the stage-level index required to review every fold artifact."""

    if not fold_artifacts:
        raise DirectionComparisonError("A direction stage report requires at least one fold")
    for fold_id, artifacts in fold_artifacts.items():
        if not str(fold_id):
            raise DirectionComparisonError("A direction stage report has an empty fold ID")
        _verify_artifacts(artifacts)
    destination = Path(output_dir)
    path = destination / "direction_comparison_report.json"
    _write_json(
        path,
        {
            "evaluation_contract_version": DIRECTION_COMPARISON_CONTRACT_VERSION,
            "stage": str(stage),
            "metadata": dict(metadata),
            "fold_artifacts": {
                str(fold_id): artifacts.as_dict()
                for fold_id, artifacts in sorted(fold_artifacts.items())
            },
        },
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise DirectionComparisonError(f"Direction stage report was not written: {path}")
    return path


__all__ = [
    "DIRECTION_COMPARISON_CONTRACT_VERSION",
    "DirectionComparisonError",
    "EvaluationArtifacts",
    "render_fold_ablation_direction_comparison",
    "render_fold_direction_comparison",
    "write_direction_stage_report",
]
