"""Mandatory, fixed-metric evaluation plots for every V3 model stage.

The module is intentionally the only V3 code path that renders evaluation
figures.  P0/P1 runners call it synchronously; a plotting failure therefore
fails the stage instead of leaving an unreviewable checkpoint behind.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "kronos-v3-matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csj.evaluation_plotter import (
    DirectionComparisonError,
    EvaluationArtifacts as DirectionComparisonArtifacts,
    render_fold_direction_comparison,
    write_direction_stage_report,
)

from .p0 import target_path_metrics
from .pair_probe import probe_metrics


METRIC_CONTRACT_VERSION = "v3-evaluation-v1"


class EvaluationPlotError(RuntimeError):
    """Raised when a V3 evaluation cannot produce its required artifacts."""


@dataclass(frozen=True)
class FixedMetric:
    """A metric locked into the V3 evaluation contract."""

    key: str
    label: str
    optimization: str


P0_FIXED_METRICS: tuple[FixedMetric, ...] = (
    FixedMetric("day3_path_balanced_accuracy", "Day3 path balanced accuracy", "maximize"),
    FixedMetric("day3_return_mae", "Day3 return MAE", "minimize"),
    FixedMetric("mean_return_path_correlation", "Mean return-path correlation", "maximize"),
    FixedMetric("mean_z_normalized_dtw", "Mean z-normalized DTW", "minimize"),
)

P1_FIXED_METRICS: tuple[FixedMetric, ...] = (
    FixedMetric("target_only_balanced_accuracy", "Target-only Day3 balanced accuracy", "maximize"),
    FixedMetric("pair_balanced_accuracy", "Pair-probe Day3 balanced accuracy", "maximize"),
    FixedMetric("balanced_accuracy_improvement", "Pair minus target-only balanced accuracy", "maximize"),
    FixedMetric(
        "bootstrap_5_day_probability_improvement_positive",
        "5-day block P(improvement > 0)",
        "maximize",
    ),
    FixedMetric(
        "bootstrap_10_day_probability_improvement_positive",
        "10-day block P(improvement > 0)",
        "maximize",
    ),
)

P0_REQUIRED_COLUMNS = frozenset(
    {
        "case_key",
        "model",
        "product",
        "day3_actual_direction",
        "day3_predicted_direction",
        "day3_actual_return",
        "day3_predicted_return",
        "path_return_correlation",
        "z_normalized_dtw",
    }
)
P1_REQUIRED_COLUMNS = frozenset(
    {
        "case_key",
        "product",
        "target_end_day",
        "neighbor_direction",
        "actual_label",
        "predicted_label",
        "valid_direction",
    }
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    """The files that make a V3 stage reviewable."""

    report_path: Path
    figure_paths: tuple[Path, ...]
    metric_contract_version: str = METRIC_CONTRACT_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_contract_version": self.metric_contract_version,
            "report": str(self.report_path),
            "figures": [str(path) for path in self.figure_paths],
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
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


def _require_columns(records: pd.DataFrame, required: frozenset[str], *, label: str) -> None:
    missing = sorted(required.difference(records.columns))
    if missing:
        raise EvaluationPlotError(f"{label} records miss fixed evaluation columns: {missing!r}")
    if records.empty:
        raise EvaluationPlotError(f"Cannot render a fixed evaluation report from empty {label} records")


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _p0_scalar_metrics(records: pd.DataFrame) -> tuple[dict[str, float | None], dict[str, object]]:
    full = target_path_metrics(records)
    scalar = {metric.key: _finite_or_none(full[metric.key]) for metric in P0_FIXED_METRICS}
    coverage = {
        "cases": int(full["samples"]),
        "direction_cases": int(full["direction_samples"]),
    }
    return scalar, coverage


def _p1_scalar_metrics(
    pair_records: pd.DataFrame,
    target_only_records: pd.DataFrame,
    bootstrap: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, float | None], dict[str, object]]:
    pair = probe_metrics(pair_records)
    target = probe_metrics(target_only_records)
    target_balanced_accuracy = _finite_or_none(target["balanced_accuracy"])
    pair_balanced_accuracy = _finite_or_none(pair["balanced_accuracy"])
    improvement = (
        pair_balanced_accuracy - target_balanced_accuracy
        if pair_balanced_accuracy is not None and target_balanced_accuracy is not None
        else None
    )
    output: dict[str, float | None] = {
        "target_only_balanced_accuracy": target_balanced_accuracy,
        "pair_balanced_accuracy": pair_balanced_accuracy,
        "balanced_accuracy_improvement": improvement,
    }
    for block_days in (5, 10):
        result = bootstrap.get(f"block_{block_days}", {})
        output[f"bootstrap_{block_days}_day_probability_improvement_positive"] = (
            _finite_or_none(result.get("probability_improvement_positive"))
            if bool(result.get("available"))
            else None
        )
    coverage = {
        "pair_cases": int(len(pair_records)),
        "target_only_cases": int(len(target_only_records)),
        "pair_direction_cases": int(pair["samples"]),
        "target_only_direction_cases": int(target["samples"]),
    }
    return output, coverage


def _metric_contract(stage: str, metrics: Sequence[FixedMetric]) -> dict[str, object]:
    return {
        "version": METRIC_CONTRACT_VERSION,
        "stage": stage,
        "metrics": [asdict(metric) for metric in metrics],
    }


def _annotate_bars(axis: plt.Axes, bars: object, values: Sequence[float | None]) -> None:
    for bar, value in zip(bars, values, strict=True):
        if value is not None and math.isfinite(value):
            axis.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _plot_p0_overview(
    overall: Mapping[str, Mapping[str, float | None]],
    destination: Path,
) -> None:
    model_names = list(overall)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, metric in zip(axes.flat, P0_FIXED_METRICS, strict=True):
        values = [overall[name][metric.key] for name in model_names]
        bars = axis.bar(model_names, [np.nan if value is None else value for value in values])
        _annotate_bars(axis, bars, values)
        axis.set_title(f"{metric.label} ({metric.optimization})")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("V3 fixed P0 evaluation metrics", fontsize=14)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_p0_by_product(
    by_product: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    model_names: Sequence[str],
    destination: Path,
) -> None:
    product_names = list(by_product)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    positions = np.arange(len(product_names), dtype=np.float64)
    width = 0.75 / max(len(model_names), 1)
    palette = plt.get_cmap("tab10")
    for axis, metric in zip(axes.flat, P0_FIXED_METRICS, strict=True):
        for model_index, model_name in enumerate(model_names):
            values = [by_product[product][model_name][metric.key] for product in product_names]
            offsets = positions + (model_index - (len(model_names) - 1) / 2) * width
            bars = axis.bar(
                offsets,
                [np.nan if value is None else value for value in values],
                width=width,
                label=model_name,
                color=palette(model_index),
            )
            _annotate_bars(axis, bars, values)
        axis.set_title(f"{metric.label} by product")
        axis.set_xticks(positions, product_names)
        axis.grid(axis="y", alpha=0.25)
        if len(model_names) > 1:
            axis.legend(fontsize=8)
    figure.suptitle("V3 fixed P0 metrics by product", fontsize=14)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_p1_overview(overall: Mapping[str, float | None], destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    balanced_accuracy_values = [
        overall["target_only_balanced_accuracy"],
        overall["pair_balanced_accuracy"],
    ]
    bars = axes[0].bar(
        ["target-only", "pair"],
        [np.nan if value is None else value for value in balanced_accuracy_values],
    )
    _annotate_bars(axes[0], bars, balanced_accuracy_values)
    axes[0].set_title("Day3 balanced accuracy")
    axes[0].grid(axis="y", alpha=0.25)

    improvement = overall["balanced_accuracy_improvement"]
    bars = axes[1].bar(["pair − target-only"], [np.nan if improvement is None else improvement])
    _annotate_bars(axes[1], bars, [improvement])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Balanced accuracy improvement")
    axes[1].grid(axis="y", alpha=0.25)

    bootstrap_values = [
        overall["bootstrap_5_day_probability_improvement_positive"],
        overall["bootstrap_10_day_probability_improvement_positive"],
    ]
    bars = axes[2].bar(
        ["5-day", "10-day"],
        [np.nan if value is None else value for value in bootstrap_values],
    )
    _annotate_bars(axes[2], bars, bootstrap_values)
    axes[2].axhline(0.80, color="tab:red", linestyle="--", linewidth=1, label="P1 threshold")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("P(improvement > 0) by block")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("V3 fixed P1 evaluation metrics", fontsize=14)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_p1_strata(
    by_product: Mapping[str, Mapping[str, float | None]],
    by_fold: Mapping[str, float | None],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, title, values in (
        (
            axes[0],
            "Balanced accuracy improvement by product",
            {name: metrics["balanced_accuracy_improvement"] for name, metrics in by_product.items()},
        ),
        (axes[1], "Balanced accuracy improvement by fold", by_fold),
    ):
        names = list(values)
        numeric_values = [values[name] for name in names]
        bars = axis.bar(names, [np.nan if value is None else value for value in numeric_values])
        _annotate_bars(axis, bars, numeric_values)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("V3 fixed P1 stratified metrics", fontsize=14)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _verify_artifacts(artifacts: EvaluationArtifacts) -> EvaluationArtifacts:
    for path in (artifacts.report_path, *artifacts.figure_paths):
        if not path.is_file() or path.stat().st_size == 0:
            raise EvaluationPlotError(f"Required evaluation artifact was not written: {path}")
    return artifacts


def render_p0_evaluation_report(
    records_by_model: Mapping[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    stage: str,
    metadata: Mapping[str, object],
) -> EvaluationArtifacts:
    """Render the fixed P0/P2 path metrics for every target-path model run."""

    if not records_by_model:
        raise EvaluationPlotError("P0 evaluation plotter requires at least one model record set")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ordered_records = {str(name): records for name, records in records_by_model.items()}
    for model_name, records in ordered_records.items():
        _require_columns(records, P0_REQUIRED_COLUMNS, label=f"P0/{model_name}")

    overall: dict[str, dict[str, float | None]] = {}
    coverage: dict[str, dict[str, object]] = {}
    product_names = sorted(
        {str(product) for records in ordered_records.values() for product in records["product"].unique()}
    )
    by_product: dict[str, dict[str, dict[str, float | None]]] = {
        product: {} for product in product_names
    }
    for model_name, records in ordered_records.items():
        overall[model_name], coverage[model_name] = _p0_scalar_metrics(records)
        for product in product_names:
            product_records = records.loc[records["product"] == product]
            if product_records.empty:
                raise EvaluationPlotError(
                    f"P0/{model_name} has no records for product {product!r}; "
                    "model comparisons must use the same product coverage."
                )
            by_product[product][model_name], _ = _p0_scalar_metrics(product_records)

    overview_path = destination / "fixed_metrics_overview.png"
    product_path = destination / "fixed_metrics_by_product.png"
    _plot_p0_overview(overall, overview_path)
    _plot_p0_by_product(by_product, list(ordered_records), product_path)
    report_path = destination / "evaluation_report.json"
    payload: dict[str, object] = {
        "metric_contract": _metric_contract(stage, P0_FIXED_METRICS),
        "metadata": dict(metadata),
        "coverage": coverage,
        "overall": overall,
        "by_product": by_product,
        "artifacts": {
            "overview": str(overview_path),
            "by_product": str(product_path),
        },
    }
    _write_json(report_path, payload)
    return _verify_artifacts(EvaluationArtifacts(report_path, (overview_path, product_path)))


def render_p1_evaluation_report(
    pair_records: pd.DataFrame,
    target_only_records: pd.DataFrame,
    *,
    bootstrap: Mapping[str, Mapping[str, object]],
    gate: Mapping[str, object],
    output_dir: str | Path,
    stage: str,
    metadata: Mapping[str, object],
) -> EvaluationArtifacts:
    """Render the fixed, paired P1 Probe comparison and its stratification."""

    _require_columns(pair_records, P1_REQUIRED_COLUMNS, label="P1/pair")
    _require_columns(target_only_records, P1_REQUIRED_COLUMNS, label="P1/target-only")
    pair_keys = set(pair_records["case_key"])
    target_keys = set(target_only_records["case_key"])
    if pair_keys != target_keys:
        raise EvaluationPlotError("P1 plotter requires exactly matched pair and target-only case keys")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    overall, coverage = _p1_scalar_metrics(pair_records, target_only_records, bootstrap)
    product_names = sorted(str(product) for product in pair_records["product"].unique())
    by_product: dict[str, dict[str, float | None]] = {}
    for product in product_names:
        pair_product = pair_records.loc[pair_records["product"] == product]
        target_product = target_only_records.loc[target_only_records["product"] == product]
        if set(pair_product["case_key"]) != set(target_product["case_key"]):
            raise EvaluationPlotError(
                f"P1/{product} plot records are not a paired target-only comparison"
            )
        product_metrics, _ = _p1_scalar_metrics(pair_product, target_product, {})
        by_product[product] = product_metrics
    fold_values = {
        str(fold_id): _finite_or_none(value)
        for fold_id, value in dict(gate.get("fold_balanced_accuracy_improvements", {})).items()
    }

    overview_path = destination / "fixed_metrics_overview.png"
    strata_path = destination / "fixed_metrics_strata.png"
    _plot_p1_overview(overall, overview_path)
    _plot_p1_strata(by_product, fold_values, strata_path)
    report_path = destination / "evaluation_report.json"
    payload: dict[str, object] = {
        "metric_contract": _metric_contract(stage, P1_FIXED_METRICS),
        "metadata": dict(metadata),
        "coverage": coverage,
        "overall": overall,
        "by_product": by_product,
        "by_fold": fold_values,
        "gate": dict(gate),
        "artifacts": {
            "overview": str(overview_path),
            "strata": str(strata_path),
        },
    }
    _write_json(report_path, payload)
    return _verify_artifacts(EvaluationArtifacts(report_path, (overview_path, strata_path)))


def _v3_p1_direction_records(
    records: pd.DataFrame,
    *,
    fold_id: str,
    model: str,
) -> pd.DataFrame:
    """Adapt existing V3 probe records to the shared per-fold plot contract.

    V3 predates the common renderer, so its historical records do not carry a
    fold/model column and represented a zero-return label as ``-1``.  The
    adapter changes only the in-memory plotting view: it never rewrites the
    source predictions or their metrics.
    """

    required = {
        "case_key",
        "product",
        "target_end_day",
        "target_contract_id",
        "actual_direction",
        "predicted_direction",
        "probability_up",
        "valid_direction",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise EvaluationPlotError(
            f"V3 P1 direction comparison records miss columns: {missing!r}"
        )
    adapted = records.copy()
    invalid = ~adapted["valid_direction"].astype(bool)
    adapted.loc[invalid, "actual_direction"] = 0
    adapted.loc[invalid, "predicted_direction"] = 0
    adapted["fold_id"] = str(fold_id)
    adapted["model"] = str(model)
    return adapted


def render_p1_fold_direction_comparison(
    pair_records: pd.DataFrame,
    target_only_records: pd.DataFrame,
    *,
    fold_id: str,
    output_dir: str | Path,
    stage: str,
    metadata: Mapping[str, object],
) -> DirectionComparisonArtifacts:
    """Compatibility wrapper for V3's pair-probe per-fold comparison.

    New V4 code imports :mod:`csj.evaluation_plotter` directly.  This wrapper
    keeps the V3 record schema usable for historical backfills without making
    V4 depend on the V3 plotting module.
    """

    try:
        return render_fold_direction_comparison(
            {
                "pair_probe": _v3_p1_direction_records(
                    pair_records, fold_id=fold_id, model="pair_probe"
                ),
                "target_only_probe": _v3_p1_direction_records(
                    target_only_records,
                    fold_id=fold_id,
                    model="target_only_probe",
                ),
            },
            fold_id=fold_id,
            candidate_model="pair_probe",
            baseline_model="target_only_probe",
            output_dir=output_dir,
            stage=stage,
            metadata=metadata,
        )
    except DirectionComparisonError as exc:
        raise EvaluationPlotError(str(exc)) from exc


__all__ = [
    "EvaluationArtifacts",
    "EvaluationPlotError",
    "FixedMetric",
    "METRIC_CONTRACT_VERSION",
    "P0_FIXED_METRICS",
    "P1_FIXED_METRICS",
    "render_p1_fold_direction_comparison",
    "render_p0_evaluation_report",
    "render_p1_evaluation_report",
    "write_direction_stage_report",
]
