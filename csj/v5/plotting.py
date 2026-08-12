"""Compact, mandatory V5 per-fold realized-return and close comparisons."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "kronos-v5-matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class V5PlotError(RuntimeError):
    """A required V5 fold plot cannot be produced truthfully."""


@dataclass(frozen=True)
class V5FoldPlotArtifacts:
    return_comparison: Path
    close_comparison: Path
    summary_json: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "return_comparison": str(self.return_comparison),
            "close_comparison": str(self.close_comparison),
            "summary_json": str(self.summary_json),
        }


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
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _direction_metrics(records: pd.DataFrame, *, actual: str, predicted: str) -> dict[str, object]:
    valid = records.loc[records[actual].astype(np.int8) != 0]
    if valid.empty:
        return {"cases": int(len(records)), "balanced_accuracy": None, "accuracy": None}
    actual_values = valid[actual].to_numpy(dtype=np.int8)
    predicted_values = valid[predicted].to_numpy(dtype=np.int8)
    up = actual_values == 1
    down = actual_values == -1
    return {
        "cases": int(len(records)),
        "valid_direction_cases": int(len(valid)),
        "balanced_accuracy": (
            float(0.5 * (np.mean(predicted_values[up] == 1) + np.mean(predicted_values[down] == -1)))
            if up.any() and down.any()
            else None
        ),
        "accuracy": float(np.mean(actual_values == predicted_values)),
    }


def _require_common_records(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    fold_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "case_key",
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "origin_close",
        "actual_path",
        "predicted_path",
        "day3_actual_return",
        "day3_predicted_return",
        "day3_actual_direction",
        "day3_predicted_direction",
    }
    for label, records in (("candidate", candidate), ("baseline", baseline)):
        missing = sorted(required.difference(records.columns))
        if missing:
            raise V5PlotError(f"{label} records miss required V5 plot columns: {missing!r}")
        if records.empty:
            raise V5PlotError(f"{label} records are empty")
        if records["case_key"].duplicated().any():
            raise V5PlotError(f"{label} records contain duplicate case keys")
        if set(records["fold_id"].astype(str)) != {str(fold_id)}:
            raise V5PlotError(f"{label} records do not belong exactly to {fold_id}")
    candidate_indexed = candidate.set_index("case_key", verify_integrity=True).sort_index()
    baseline_indexed = baseline.set_index("case_key", verify_integrity=True).sort_index()
    if set(candidate_indexed.index) != set(baseline_indexed.index):
        raise V5PlotError("V5 candidate and baseline case keys do not match")
    baseline_indexed = baseline_indexed.loc[candidate_indexed.index]
    for column in ("fold_id", "product", "target_end_day", "target_contract_id", "origin_close"):
        if not candidate_indexed[column].equals(baseline_indexed[column]):
            raise V5PlotError(f"V5 candidate and baseline disagree on {column}")
    return candidate_indexed.reset_index(), baseline_indexed.reset_index()


def _compact_positions(count: int, limit: int = 36) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, limit, dtype=int))


def _draw_return_panel(
    axis: plt.Axes,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    product: str,
    candidate_label: str,
    baseline_label: str,
) -> None:
    candidate_group = candidate.loc[candidate["product"].astype(str) == product].reset_index(drop=True)
    baseline_by_key = baseline.set_index("case_key", verify_integrity=True)
    baseline_group = baseline_by_key.loc[candidate_group["case_key"]].reset_index()
    positions = np.arange(len(candidate_group), dtype=float)
    actual = candidate_group["day3_actual_return"].to_numpy(dtype=np.float64)
    candidate_return = candidate_group["day3_predicted_return"].to_numpy(dtype=np.float64)
    baseline_return = baseline_group["day3_predicted_return"].to_numpy(dtype=np.float64)
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, zorder=0)
    axis.plot(positions, actual, color="black", linewidth=1.6, marker="o", markersize=2.8, label="actual")
    axis.plot(positions, candidate_return, color="#1f77b4", linewidth=1.4, marker="s", markersize=2.5, label=candidate_label)
    axis.plot(positions, baseline_return, color="#7c3aed", linewidth=1.2, linestyle="--", marker="D", markersize=2.4, label=baseline_label)
    metrics_candidate = _direction_metrics(candidate_group, actual="day3_actual_direction", predicted="day3_predicted_direction")
    metrics_baseline = _direction_metrics(baseline_group, actual="day3_actual_direction", predicted="day3_predicted_direction")
    axis.set_title(
        f"{product} · n={len(candidate_group)} · {candidate_label} BA {_format_metric(metrics_candidate['balanced_accuracy'])} · {baseline_label} BA {_format_metric(metrics_baseline['balanced_accuracy'])}",
        fontsize=8,
    )
    ticks = _compact_positions(len(candidate_group))
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [pd.Timestamp(candidate_group["target_end_day"].iloc[index]).strftime("%m-%d") for index in ticks],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel("Day3 close / origin close − 1")
    axis.grid(alpha=0.22)


def _close_path(value: object, *, field: str, case_key: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V5PlotError(f"{case_key} has non-numeric {field}") from exc
    if array.ndim != 2 or array.shape[1] < 4 or array.shape[0] < 1:
        raise V5PlotError(f"{case_key} {field} must have shape [bar, feature>=4]")
    close = array[:, 3]
    if not np.isfinite(close).all() or (close <= 0.0).any():
        raise V5PlotError(f"{case_key} {field} close must be finite and positive")
    return close


def _selected_positions(records: pd.DataFrame) -> tuple[int, ...]:
    return tuple(sorted({0, len(records) // 2, len(records) - 1}))


def _draw_close_panel(
    axis: plt.Axes,
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    candidate_label: str,
    baseline_label: str,
) -> None:
    key = str(candidate["case_key"])
    actual = _close_path(candidate["actual_path"], field="actual_path", case_key=key)
    predicted = _close_path(candidate["predicted_path"], field="predicted_path", case_key=key)
    baseline_path = _close_path(baseline["predicted_path"], field="predicted_path", case_key=key)
    length = min(len(actual), len(predicted), len(baseline_path))
    x = np.arange(1, length + 1)
    axis.plot(x, actual[:length], color="black", linewidth=1.9, label="actual")
    axis.plot(x, predicted[:length], color="#1f77b4", linewidth=1.5, label=candidate_label)
    axis.plot(x, baseline_path[:length], color="#7c3aed", linewidth=1.3, linestyle="--", label=baseline_label)
    for day_end in list(candidate["day_end_indices"])[:-1]:
        if int(day_end) < length:
            axis.axvline(int(day_end) + 1.5, color="#9ca3af", linestyle=":", linewidth=0.8)
    fallback = bool(candidate.get("fallback_to_uniform", False))
    axis.set_title(
        f"{candidate['product']} {pd.Timestamp(candidate['target_end_day']).strftime('%Y-%m-%d')} · {key.split('|')[0]}"
        + (" · fallback" if fallback else ""),
        fontsize=8,
    )
    axis.set_xlabel("target hourly bar")
    axis.set_ylabel("close")
    axis.grid(alpha=0.22)


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.1%}" if math.isfinite(number) else "n/a"


def render_fold_path_comparisons(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    fold_id: str,
    output_dir: str | Path,
    stage: str,
    candidate_label: str,
    baseline_label: str,
    metadata: Mapping[str, object],
) -> V5FoldPlotArtifacts:
    """Save exactly two compact mandatory figures for one fold.

    The first is the full-fold Day3 close-return comparison by product; the
    second is representative actual/candidate/baseline close paths.  It avoids
    persisting a third visualization or redundant data copies.
    """

    candidate, baseline = _require_common_records(candidate, baseline, fold_id=str(fold_id))
    destination = Path(output_dir) / str(fold_id)
    destination.mkdir(parents=True, exist_ok=True)
    products = sorted(candidate["product"].astype(str).unique())

    return_figure, return_axes = plt.subplots(
        len(products), 1, figsize=(13, max(3.4, 3.0 * len(products))), squeeze=False
    )
    try:
        for axis, product in zip(return_axes[:, 0], products, strict=True):
            _draw_return_panel(
                axis,
                candidate,
                baseline,
                product=product,
                candidate_label=candidate_label,
                baseline_label=baseline_label,
            )
        return_axes[0, 0].legend(loc="best", fontsize=7)
        return_figure.suptitle(f"{stage} · {fold_id} · realized Day3 close return comparison", fontsize=12)
        return_figure.tight_layout(rect=(0, 0, 1, 0.95))
        return_path = destination / "day3_close_return_comparison.png"
        return_figure.savefig(return_path, dpi=170, bbox_inches="tight")
    finally:
        plt.close(return_figure)

    representative: list[tuple[pd.Series, pd.Series]] = []
    baseline_by_key = baseline.set_index("case_key", verify_integrity=True)
    for product in products:
        group = candidate.loc[candidate["product"].astype(str) == product].reset_index(drop=True)
        for position in _selected_positions(group):
            row = group.iloc[position]
            representative.append((row, baseline_by_key.loc[str(row["case_key"])]))
    columns = 3
    rows = max(1, math.ceil(len(representative) / columns))
    close_figure, close_axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 3.5 * rows), squeeze=False)
    try:
        for index, axis in enumerate(close_axes.flat):
            if index >= len(representative):
                axis.axis("off")
                continue
            candidate_row, baseline_row = representative[index]
            _draw_close_panel(
                axis,
                candidate_row,
                baseline_row,
                candidate_label=candidate_label,
                baseline_label=baseline_label,
            )
        if representative:
            close_axes.flat[0].legend(loc="best", fontsize=7)
        close_figure.suptitle(f"{stage} · {fold_id} · actual versus predicted close paths", fontsize=12)
        close_figure.tight_layout(rect=(0, 0, 1, 0.95))
        close_path = destination / "close_price_comparison.png"
        close_figure.savefig(close_path, dpi=170, bbox_inches="tight")
    finally:
        plt.close(close_figure)

    summary_path = destination / "path_comparison_summary.json"
    _write_json(
        summary_path,
        {
            "stage": str(stage),
            "fold_id": str(fold_id),
            "metadata": dict(metadata),
            "candidate_model": candidate_label,
            "baseline_model": baseline_label,
            "case_count": int(len(candidate)),
            "by_product": {
                product: {
                    "candidate": _direction_metrics(
                        candidate.loc[candidate["product"].astype(str) == product],
                        actual="day3_actual_direction",
                        predicted="day3_predicted_direction",
                    ),
                    "baseline": _direction_metrics(
                        baseline.loc[baseline["product"].astype(str) == product],
                        actual="day3_actual_direction",
                        predicted="day3_predicted_direction",
                    ),
                }
                for product in products
            },
            "artifacts": {
                "day3_close_return_comparison": str(return_path),
                "close_price_comparison": str(close_path),
            },
        },
    )
    artifacts = V5FoldPlotArtifacts(return_path, close_path, summary_path)
    for path in (artifacts.return_comparison, artifacts.close_comparison, artifacts.summary_json):
        if not path.is_file() or path.stat().st_size == 0:
            raise V5PlotError(f"Required V5 fold plot artifact was not written: {path}")
    return artifacts


def render_fold_baseline_path_plots(
    records: pd.DataFrame,
    *,
    fold_id: str,
    output_dir: str | Path,
    stage: str,
    baseline_label: str,
    metadata: Mapping[str, object],
) -> V5FoldPlotArtifacts:
    """Render the requested actual-versus-baseline plots after a P0 fold.

    Unlike a candidate comparison, P0 has one genuine path-producing baseline:
    the fixed zero-shot mean path.  Reusing it as both sides of a generic
    comparison would duplicate lines and mislabel a direction-only baseline as
    a close forecast, so this dedicated renderer is deliberately single-arm.
    """

    required = {
        "case_key",
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "origin_close",
        "actual_path",
        "predicted_path",
        "day3_actual_return",
        "day3_predicted_return",
        "day3_actual_direction",
        "day3_predicted_direction",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise V5PlotError(f"V5 baseline records miss required plot columns: {missing!r}")
    if records.empty or records["case_key"].duplicated().any():
        raise V5PlotError("V5 baseline records must be non-empty with unique case keys")
    if set(records["fold_id"].astype(str)) != {str(fold_id)}:
        raise V5PlotError(f"V5 baseline records do not belong exactly to {fold_id}")
    baseline = records.sort_values(
        ["product", "target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)
    products = sorted(baseline["product"].astype(str).unique())
    destination = Path(output_dir) / str(fold_id)
    destination.mkdir(parents=True, exist_ok=True)

    return_figure, return_axes = plt.subplots(
        len(products), 1, figsize=(13, max(3.4, 3.0 * len(products))), squeeze=False
    )
    try:
        for axis, product in zip(return_axes[:, 0], products, strict=True):
            group = baseline.loc[baseline["product"].astype(str) == product].reset_index(drop=True)
            positions = np.arange(len(group), dtype=float)
            axis.axhline(0.0, color="#9ca3af", linewidth=0.8, zorder=0)
            axis.plot(
                positions,
                group["day3_actual_return"].to_numpy(dtype=np.float64),
                color="black",
                linewidth=1.6,
                marker="o",
                markersize=2.8,
                label="actual",
            )
            axis.plot(
                positions,
                group["day3_predicted_return"].to_numpy(dtype=np.float64),
                color="#1f77b4",
                linewidth=1.4,
                marker="s",
                markersize=2.5,
                label=baseline_label,
            )
            metrics = _direction_metrics(
                group,
                actual="day3_actual_direction",
                predicted="day3_predicted_direction",
            )
            axis.set_title(
                f"{product} · n={len(group)} · {baseline_label} BA {_format_metric(metrics['balanced_accuracy'])} · Acc {_format_metric(metrics['accuracy'])}",
                fontsize=8,
            )
            ticks = _compact_positions(len(group))
            axis.set_xticks(ticks)
            axis.set_xticklabels(
                [pd.Timestamp(group["target_end_day"].iloc[index]).strftime("%m-%d") for index in ticks],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            axis.set_ylabel("Day3 close / origin close − 1")
            axis.grid(alpha=0.22)
        return_axes[0, 0].legend(loc="best", fontsize=7)
        return_figure.suptitle(f"{stage} · {fold_id} · baseline realized Day3 close return", fontsize=12)
        return_figure.tight_layout(rect=(0, 0, 1, 0.95))
        return_path = destination / "day3_close_return_comparison.png"
        return_figure.savefig(return_path, dpi=170, bbox_inches="tight")
    finally:
        plt.close(return_figure)

    representative: list[pd.Series] = []
    for product in products:
        group = baseline.loc[baseline["product"].astype(str) == product].reset_index(drop=True)
        representative.extend(group.iloc[position] for position in _selected_positions(group))
    columns = 3
    rows = max(1, math.ceil(len(representative) / columns))
    close_figure, close_axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 3.5 * rows), squeeze=False)
    try:
        for index, axis in enumerate(close_axes.flat):
            if index >= len(representative):
                axis.axis("off")
                continue
            row = representative[index]
            key = str(row["case_key"])
            actual = _close_path(row["actual_path"], field="actual_path", case_key=key)
            predicted = _close_path(row["predicted_path"], field="predicted_path", case_key=key)
            length = min(len(actual), len(predicted))
            x = np.arange(1, length + 1)
            axis.plot(x, actual[:length], color="black", linewidth=1.9, label="actual")
            axis.plot(x, predicted[:length], color="#1f77b4", linewidth=1.5, label=baseline_label)
            for day_end in list(row["day_end_indices"])[:-1]:
                if int(day_end) < length:
                    axis.axvline(int(day_end) + 1.5, color="#9ca3af", linestyle=":", linewidth=0.8)
            axis.set_title(
                f"{row['product']} {pd.Timestamp(row['target_end_day']).strftime('%Y-%m-%d')} · {key.split('|')[0]}",
                fontsize=8,
            )
            axis.set_xlabel("target hourly bar")
            axis.set_ylabel("close")
            axis.grid(alpha=0.22)
        if representative:
            close_axes.flat[0].legend(loc="best", fontsize=7)
        close_figure.suptitle(f"{stage} · {fold_id} · actual versus baseline close paths", fontsize=12)
        close_figure.tight_layout(rect=(0, 0, 1, 0.95))
        close_path = destination / "close_price_comparison.png"
        close_figure.savefig(close_path, dpi=170, bbox_inches="tight")
    finally:
        plt.close(close_figure)

    summary_path = destination / "path_comparison_summary.json"
    _write_json(
        summary_path,
        {
            "stage": str(stage),
            "fold_id": str(fold_id),
            "metadata": dict(metadata),
            "baseline_model": baseline_label,
            "case_count": int(len(baseline)),
            "by_product": {
                product: _direction_metrics(
                    baseline.loc[baseline["product"].astype(str) == product],
                    actual="day3_actual_direction",
                    predicted="day3_predicted_direction",
                )
                for product in products
            },
            "artifacts": {
                "day3_close_return_comparison": str(return_path),
                "close_price_comparison": str(close_path),
            },
        },
    )
    artifacts = V5FoldPlotArtifacts(return_path, close_path, summary_path)
    for path in (artifacts.return_comparison, artifacts.close_comparison, artifacts.summary_json):
        if not path.is_file() or path.stat().st_size == 0:
            raise V5PlotError(f"Required V5 baseline plot artifact was not written: {path}")
    return artifacts


__all__ = [
    "V5FoldPlotArtifacts",
    "V5PlotError",
    "render_fold_baseline_path_plots",
    "render_fold_path_comparisons",
]
