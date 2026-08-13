"""Mandatory V6 P0 label-distribution and event-support figures."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "kronos-v6-matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class V6PlotError(RuntimeError):
    """A mandatory V6 P0 figure cannot be generated truthfully."""


@dataclass(frozen=True)
class P0PlotArtifacts:
    risk_label_distributions: Path
    fold_event_support: Path
    fold_tail_thresholds: Path
    summary_json: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "risk_label_distributions": str(self.risk_label_distributions),
            "fold_event_support": str(self.fold_event_support),
            "fold_tail_thresholds": str(self.fold_tail_thresholds),
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
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise V6PlotError(f"V6 P0 figure is missing or empty: {path}")


def _validate_records(
    outcomes: pd.DataFrame,
    fold_records: pd.DataFrame,
    *,
    primary_products: Sequence[str],
) -> tuple[str, ...]:
    products = tuple(str(value) for value in primary_products)
    if not products:
        raise V6PlotError("V6 P0 plotting requires primary products")
    outcome_required = {
        "case_key",
        "product",
        "long_mae",
        "short_mae",
        "future_vol_ratio",
    }
    fold_required = {
        "case_key",
        "fold_id",
        "split",
        "product",
        "long_tail_event",
        "short_tail_event",
        "long_tail_threshold",
        "short_tail_threshold",
    }
    for label, records, required in (
        ("outcome", outcomes, outcome_required),
        ("fold", fold_records, fold_required),
    ):
        missing = sorted(required.difference(records.columns))
        if missing:
            raise V6PlotError(f"V6 P0 {label} records miss columns: {missing!r}")
        if records.empty:
            raise V6PlotError(f"V6 P0 {label} records are empty")
    if outcomes["case_key"].duplicated().any():
        raise V6PlotError("V6 P0 continuous outcomes contain duplicate case keys")
    observed = set(outcomes["product"].astype(str))
    missing_products = sorted(set(products).difference(observed))
    if missing_products:
        raise V6PlotError(f"V6 P0 outcomes miss primary products: {missing_products!r}")
    return products


def _display_values(values: np.ndarray, *, metric: str) -> tuple[np.ndarray, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise V6PlotError(f"V6 P0 {metric} contains no finite values")
    if metric in {"long_mae", "short_mae"}:
        low = 0.0
        high = float(np.quantile(finite, 0.99, method="linear"))
    else:
        low, high = (
            float(value)
            for value in np.quantile(finite, [0.01, 0.99], method="linear")
        )
    if high <= low:
        high = low + max(abs(low) * 0.05, 1e-6)
    return np.clip(finite, low, high), low, high


def _render_distribution_figure(
    outcomes: pd.DataFrame,
    *,
    primary_products: Sequence[str],
    output_path: Path,
    version_label: str,
) -> dict[str, object]:
    metrics = (
        ("long_mae", "Long-position adverse excursion / past scale", "#dc2626"),
        ("short_mae", "Short-position adverse excursion / past scale", "#2563eb"),
        ("future_vol_ratio", "log(future realized scale / past horizon scale)", "#7c3aed"),
    )
    figure, axes = plt.subplots(
        len(primary_products),
        len(metrics),
        figsize=(14.2, 3.15 * len(primary_products)),
        squeeze=False,
    )
    display_limits: dict[str, object] = {}
    for row, product in enumerate(primary_products):
        selected = outcomes.loc[outcomes["product"].astype(str) == str(product)]
        display_limits[str(product)] = {}
        for column, (metric, title, color) in enumerate(metrics):
            axis = axes[row, column]
            values = selected[metric].to_numpy(dtype=np.float64)
            displayed, low, high = _display_values(values, metric=metric)
            axis.hist(displayed, bins=30, color=color, alpha=0.78, edgecolor="white")
            median = float(np.median(values[np.isfinite(values)]))
            p80 = float(
                np.quantile(values[np.isfinite(values)], 0.8, method="linear")
            )
            axis.axvline(median, color="#111827", linewidth=1.2, label="median")
            axis.axvline(p80, color="#f59e0b", linestyle="--", linewidth=1.2, label="P80")
            axis.set_xlim(low, high)
            axis.set_title(f"{product} · {title}\nn={len(selected)}", fontsize=9)
            axis.set_ylabel("case count")
            axis.grid(axis="y", alpha=0.2)
            if row == len(primary_products) - 1:
                axis.set_xlabel(f"display clipped to [{low:.3g}, {high:.3g}]")
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
            display_limits[str(product)][metric] = {
                "display_minimum": low,
                "display_maximum": high,
                "values_outside_display_range": int(
                    np.sum((values < low) | (values > high))
                ),
            }
    figure.suptitle(
        f"{version_label} P0 continuous risk-label distributions (display clipping only)",
        fontsize=13,
        y=1.01,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)
    return display_limits


def _render_support_figure(
    fold_records: pd.DataFrame,
    *,
    primary_products: Sequence[str],
    p0_config: Mapping[str, Any],
    output_path: Path,
    version_label: str,
) -> list[dict[str, object]]:
    primary = fold_records.loc[
        fold_records["product"].astype(str).isin(tuple(primary_products))
    ]
    fold_ids = sorted(primary["fold_id"].astype(str).unique())
    split_order = ("fit", "inner_validation", "evaluation")
    minima = {
        "fit": int(p0_config["minimum_fit_events_per_side"]),
        "inner_validation": int(p0_config["minimum_validation_events_per_side"]),
        "evaluation": int(p0_config["minimum_evaluation_events_per_side"]),
    }
    rows: list[dict[str, object]] = []
    for fold_id in fold_ids:
        for split in split_order:
            selected = primary.loc[
                (primary["fold_id"].astype(str) == fold_id)
                & (primary["split"].astype(str) == split)
            ]
            rows.append(
                {
                    "fold_id": fold_id,
                    "split": split,
                    "label": f"{fold_id.replace('fold_', 'F')}\n{split.replace('inner_validation', 'val')}",
                    "cases": int(len(selected)),
                    "long_events": int(selected["long_tail_event"].astype(bool).sum()),
                    "short_events": int(selected["short_tail_event"].astype(bool).sum()),
                    "required_minimum": minima[split],
                }
            )
    positions = np.arange(len(rows), dtype=float)
    width = 0.34
    figure, axis = plt.subplots(figsize=(15.2, 6.3))
    long_values = np.asarray([row["long_events"] for row in rows], dtype=float)
    short_values = np.asarray([row["short_events"] for row in rows], dtype=float)
    required = np.asarray([row["required_minimum"] for row in rows], dtype=float)
    long_bars = axis.bar(
        positions - width / 2,
        long_values,
        width,
        color="#dc2626",
        label="long adverse events",
    )
    short_bars = axis.bar(
        positions + width / 2,
        short_values,
        width,
        color="#2563eb",
        label="short adverse events",
    )
    axis.scatter(
        positions,
        required,
        marker="_",
        s=360,
        linewidths=2.1,
        color="#111827",
        label="pre-registered minimum",
        zorder=4,
    )
    for bars, values in ((long_bars, long_values), (short_bars, short_values)):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(2.0, max(long_values.max(), short_values.max()) * 0.01),
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for index in range(2, len(rows) - 1, 3):
        axis.axvline(index + 0.5, color="#d1d5db", linewidth=0.9)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(row["label"]) for row in rows], fontsize=8)
    axis.set_ylabel("primary-product tail-event count")
    axis.set_title(f"{version_label} P0 fold/split event support versus frozen gate")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=3, fontsize=9)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return rows


def _render_threshold_figure(
    fold_records: pd.DataFrame,
    *,
    output_path: Path,
    version_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_id, group in fold_records.groupby("fold_id", sort=True):
        long_values = group["long_tail_threshold"].to_numpy(dtype=np.float64)
        short_values = group["short_tail_threshold"].to_numpy(dtype=np.float64)
        if not (np.all(long_values == long_values[0]) and np.all(short_values == short_values[0])):
            raise V6PlotError(f"{fold_id} has more than one frozen P0 threshold")
        rows.append(
            {
                "fold_id": str(fold_id),
                "long_tail_threshold": float(long_values[0]),
                "short_tail_threshold": float(short_values[0]),
            }
        )
    if not rows:
        raise V6PlotError("V6 P0 has no fold thresholds to plot")
    positions = np.arange(len(rows), dtype=float)
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    axis.plot(
        positions,
        [row["long_tail_threshold"] for row in rows],
        color="#dc2626",
        marker="o",
        linewidth=2,
        label="long threshold",
    )
    axis.plot(
        positions,
        [row["short_tail_threshold"] for row in rows],
        color="#2563eb",
        marker="s",
        linewidth=2,
        label="short threshold",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([row["fold_id"] for row in rows])
    axis.set_ylabel("fit-only normalized adverse excursion P80")
    axis.set_title(f"{version_label} P0 fit-only tail thresholds by outer fold")
    axis.grid(alpha=0.22)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)
    return rows


def render_p0_audit_plots(
    outcomes: pd.DataFrame,
    fold_records: pd.DataFrame,
    *,
    primary_products: Sequence[str],
    p0_config: Mapping[str, Any],
    output_dir: str | Path,
    metadata: Mapping[str, object],
) -> P0PlotArtifacts:
    """Render every mandatory P0 visual and a machine-readable plot summary."""

    products = _validate_records(
        outcomes,
        fold_records,
        primary_products=primary_products,
    )
    root = Path(output_dir)
    version_value = metadata.get("strategy_version", 6)
    version_label = f"V{version_value}"
    distributions_path = root / "risk_label_distributions.png"
    support_path = root / "fold_event_support.png"
    thresholds_path = root / "fold_tail_thresholds.png"
    summary_path = root / "plot_summary.json"
    display_limits = _render_distribution_figure(
        outcomes,
        primary_products=products,
        output_path=distributions_path,
        version_label=version_label,
    )
    support_rows = _render_support_figure(
        fold_records,
        primary_products=products,
        p0_config=p0_config,
        output_path=support_path,
        version_label=version_label,
    )
    threshold_rows = _render_threshold_figure(
        fold_records,
        output_path=thresholds_path,
        version_label=version_label,
    )
    artifacts = P0PlotArtifacts(
        risk_label_distributions=distributions_path,
        fold_event_support=support_path,
        fold_tail_thresholds=thresholds_path,
        summary_json=summary_path,
    )
    _write_json(
        summary_path,
        {
            **dict(metadata),
            "plot_contract_version": f"v{version_value}-p0-risk-label-audit-v1",
            "artifacts": artifacts.as_dict(),
            "display_limits": display_limits,
            "fold_split_event_support": support_rows,
            "fit_only_thresholds": threshold_rows,
            "note": "Only axes are clipped for readability; audit statistics and gate use full values.",
        },
    )
    return artifacts


__all__ = [
    "P0PlotArtifacts",
    "V6PlotError",
    "render_p0_audit_plots",
]
