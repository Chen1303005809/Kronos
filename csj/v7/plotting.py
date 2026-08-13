"""Required P1 path-bank and validation-baseline figures for V7."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "kronos-v7-matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class V7PlotError(RuntimeError):
    """A mandatory V7 P1 visual cannot be rendered truthfully."""


@dataclass(frozen=True)
class P1PlotArtifacts:
    coverage: Path
    valid_path_distribution: Path
    risk_probability_distribution: Path
    validation_pr: Path
    validation_reliability: Path
    validation_probability_event_rate: Path
    summary_json: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "coverage": str(self.coverage),
            "valid_path_distribution": str(self.valid_path_distribution),
            "risk_probability_distribution": str(self.risk_probability_distribution),
            "validation_pr": str(self.validation_pr),
            "validation_reliability": str(self.validation_reliability),
            "validation_probability_event_rate": str(self.validation_probability_event_rate),
            "summary_json": str(self.summary_json),
        }


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise V7PlotError(f"V7 P1 figure was not written: {path}")
    return path


def _require(records: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns).difference(records.columns))
    if missing:
        raise V7PlotError(f"V7 {label} records miss columns: {missing!r}")
    if records.empty:
        raise V7PlotError(f"V7 {label} records are empty")


def _pr_curve(probability: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-probability, kind="stable")
    sorted_labels = labels[order]
    total = int(sorted_labels.sum())
    if total == 0:
        return np.asarray([0.0]), np.asarray([1.0])
    true_positive = np.cumsum(sorted_labels)
    precision = true_positive / np.arange(1, len(sorted_labels) + 1)
    recall = true_positive / total
    return recall, precision


def _bin_rows(records: pd.DataFrame, *, side: str, bins: int = 10) -> list[dict[str, object]]:
    probability = records[f"p_{side}"].to_numpy(dtype=np.float64)
    labels = records[f"{side}_tail_event"].astype(float).to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all():
        raise V7PlotError("V7 validation probabilities are non-finite")
    index = np.minimum((np.clip(probability, 0.0, 1.0) * bins).astype(int), bins - 1)
    rows: list[dict[str, object]] = []
    for bucket in range(bins):
        selected = index == bucket
        if not selected.any():
            rows.append(
                {
                    "side": side,
                    "bin": bucket,
                    "count": 0,
                    "mean_probability": None,
                    "actual_event_rate": None,
                }
            )
            continue
        rows.append(
            {
                "side": side,
                "bin": bucket,
                "count": int(selected.sum()),
                "mean_probability": float(probability[selected].mean()),
                "actual_event_rate": float(labels[selected].mean()),
            }
        )
    return rows


def render_p1_plots(
    *,
    path_records: pd.DataFrame,
    fold_path_records: pd.DataFrame,
    validation_records: pd.DataFrame,
    selected_baselines: Mapping[str, str],
    output_dir: str | Path,
    metadata: Mapping[str, object],
) -> P1PlotArtifacts:
    """Render the P1 figure contract from raw cache and validation only."""

    _require(
        path_records,
        ("case_key", "product", "pred_len", "valid_path_count", "sample_count", "eligible_for_risk"),
        label="path-bank",
    )
    _require(
        fold_path_records,
        ("case_key", "fold_id", "split", "p_long", "p_short", "eligible_for_risk"),
        label="fold path-risk",
    )
    _require(
        validation_records,
        ("fold_id", "product", "p_long", "p_short", "long_tail_event", "short_tail_event"),
        label="validation",
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    products = sorted(path_records["product"].astype(str).unique())
    coverage = (
        path_records.groupby("product", sort=True)
        .agg(cases=("case_key", "size"), valid_cases=("eligible_for_risk", "sum"))
        .reindex(products)
    )
    figure, axis = plt.subplots(figsize=(13.5, 5.6))
    position = np.arange(len(products))
    total = coverage["cases"].to_numpy(dtype=float)
    valid = coverage["valid_cases"].to_numpy(dtype=float)
    axis.bar(position, total, color="#cbd5e1", label="cached cases")
    axis.bar(position, valid, color="#059669", label="≥ minimum valid paths")
    axis.set_xticks(position)
    axis.set_xticklabels(products, rotation=45, ha="right")
    axis.set_ylabel("unique cases")
    axis.set_title("V7 P1 unique raw-path cache coverage by product")
    axis.legend()
    axis.grid(axis="y", alpha=0.22)
    coverage_path = _save(figure, root / "path_bank_coverage.png")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    valid_paths = path_records["valid_path_count"].to_numpy(dtype=float)
    fractions = valid_paths / path_records["sample_count"].to_numpy(dtype=float)
    axes[0].hist(valid_paths, bins=np.arange(valid_paths.min() - 0.5, valid_paths.max() + 1.5), color="#2563eb", edgecolor="white")
    axes[0].set_title("valid raw paths per case")
    axes[0].set_xlabel("valid path count")
    axes[0].set_ylabel("cases")
    axes[1].hist(fractions, bins=20, color="#7c3aed", edgecolor="white")
    axes[1].set_title("valid raw-path fraction per case")
    axes[1].set_xlabel("fraction")
    axes[1].set_ylabel("cases")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    valid_path_path = _save(figure, root / "valid_path_distribution.png")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, side, color in zip(axes, ("long", "short"), ("#dc2626", "#2563eb"), strict=True):
        values = fold_path_records.loc[
            fold_path_records["eligible_for_risk"].astype(bool), f"p_{side}"
        ].dropna().to_numpy(dtype=float)
        if not len(values):
            raise V7PlotError(f"V7 P1 has no eligible fold-derived {side} probabilities")
        axis.hist(values, bins=20, color=color, edgecolor="white", alpha=0.82)
        axis.set_title(f"zero-shot raw-path {side} risk")
        axis.set_xlabel("crossing fraction")
        axis.set_ylabel("cases")
        axis.grid(axis="y", alpha=0.22)
    probability_path = _save(figure, root / "risk_probability_distribution.png")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, side, color in zip(axes, ("long", "short"), ("#dc2626", "#2563eb"), strict=True):
        labels = validation_records[f"{side}_tail_event"].astype(int).to_numpy()
        probabilities = validation_records[f"p_{side}"].to_numpy(dtype=float)
        recall, precision = _pr_curve(probabilities, labels)
        axis.plot(recall, precision, color=color, linewidth=2.0)
        axis.set_xlabel("recall")
        axis.set_ylabel("precision")
        axis.set_title(f"validation PR · selected baselines · {side}")
        axis.grid(alpha=0.22)
    pr_path = _save(figure, root / "validation_pr_curves.png")

    binned = {side: _bin_rows(validation_records, side=side) for side in ("long", "short")}
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, side, color in zip(axes, ("long", "short"), ("#dc2626", "#2563eb"), strict=True):
        rows = [row for row in binned[side] if row["count"]]
        x = np.asarray([row["mean_probability"] for row in rows], dtype=float)
        y = np.asarray([row["actual_event_rate"] for row in rows], dtype=float)
        axis.plot([0.0, 1.0], [0.0, 1.0], color="#9ca3af", linestyle="--", label="perfect calibration")
        axis.plot(x, y, marker="o", color=color, label="selected baseline")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("mean predicted probability")
        axis.set_ylabel("actual event rate")
        axis.set_title(f"validation reliability · {side}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.22)
    reliability_path = _save(figure, root / "validation_reliability.png")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, side, color in zip(axes, ("long", "short"), ("#dc2626", "#2563eb"), strict=True):
        rows = binned[side]
        counts = np.asarray([row["count"] for row in rows], dtype=float)
        rates = np.asarray(
            [np.nan if row["actual_event_rate"] is None else row["actual_event_rate"] for row in rows],
            dtype=float,
        )
        position = np.arange(len(rows))
        axis.bar(position, np.nan_to_num(rates), color=color, alpha=0.8)
        axis.set_xticks(position)
        axis.set_xticklabels([f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in position], rotation=45, ha="right", fontsize=8)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("predicted-probability bin")
        axis.set_ylabel("actual event rate")
        axis.set_title(f"validation event rate by probability bin · {side}")
        for pos, count in zip(position, counts, strict=True):
            axis.text(pos, 0.02, f"n={int(count)}", ha="center", va="bottom", fontsize=7, rotation=90)
        axis.grid(axis="y", alpha=0.22)
    event_path = _save(figure, root / "validation_probability_event_rate.png")

    summary_path = _write_json(
        root / "plot_summary.json",
        {
            "metadata": dict(metadata),
            "selected_baselines": dict(selected_baselines),
            "path_bank_cases": int(len(path_records)),
            "fold_path_risk_records": int(len(fold_path_records)),
            "validation_cases": int(len(validation_records)),
            "validation_probability_bins": binned,
            "artifacts": {
                "coverage": str(coverage_path),
                "valid_path_distribution": str(valid_path_path),
                "risk_probability_distribution": str(probability_path),
                "validation_pr": str(pr_path),
                "validation_reliability": str(reliability_path),
                "validation_probability_event_rate": str(event_path),
            },
        },
    )
    return P1PlotArtifacts(
        coverage_path,
        valid_path_path,
        probability_path,
        pr_path,
        reliability_path,
        event_path,
        summary_path,
    )


__all__ = ["P1PlotArtifacts", "V7PlotError", "render_p1_plots"]
