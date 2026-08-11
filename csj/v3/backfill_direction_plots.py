"""Backfill mandatory per-fold direction plots from immutable V3 P1 records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from csj.evaluation_plotter import DirectionComparisonError, write_direction_stage_report
from csj.v3.evaluation_plotter import render_p1_fold_direction_comparison


def _read_records(path: Path) -> pd.DataFrame:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionComparisonError(f"Cannot read V3 record file: {path}") from exc
    if not isinstance(value, list):
        raise DirectionComparisonError(f"V3 record file must contain a JSON list: {path}")
    return pd.DataFrame(value)


def backfill_v3_p1_direction_plots(run_root: str | Path) -> Path:
    """Create only new evaluation artifacts for every complete V3 P1 fold.

    ``run_root`` may be the run directory or its ``p1`` subdirectory.  The
    source ``*_records.json`` files, aggregate metrics, and checkpoints remain
    read-only throughout this operation.
    """

    root = Path(run_root)
    stage_root = root if root.name == "p1" else root / "p1"
    if not stage_root.is_dir():
        raise DirectionComparisonError(f"V3 P1 stage directory does not exist: {stage_root}")
    fold_artifacts = {}
    for fold_dir in sorted(path for path in stage_root.glob("fold_*") if path.is_dir()):
        pair_frames: list[pd.DataFrame] = []
        target_frames: list[pd.DataFrame] = []
        for product_dir in sorted(path for path in fold_dir.iterdir() if path.is_dir()):
            pair_path = product_dir / "pair_records.json"
            target_path = product_dir / "target_only_records.json"
            if pair_path.is_file() != target_path.is_file():
                raise DirectionComparisonError(
                    f"V3 fold has only one arm's records for {product_dir}: "
                    f"pair={pair_path.is_file()} target={target_path.is_file()}"
                )
            if pair_path.is_file():
                pair_frames.append(_read_records(pair_path))
                target_frames.append(_read_records(target_path))
        if not pair_frames:
            raise DirectionComparisonError(f"V3 fold has no paired probe records: {fold_dir}")
        fold_id = fold_dir.name
        fold_artifacts[fold_id] = render_p1_fold_direction_comparison(
            pd.concat(pair_frames, ignore_index=True),
            pd.concat(target_frames, ignore_index=True),
            fold_id=fold_id,
            output_dir=stage_root / "evaluation",
            stage="v3_p1_pair_probe_backfill",
            metadata={
                "strategy_version": 3,
                "phase": "p1",
                "result_scope": "exploratory_partial_panel",
                "production_eligible": False,
                "backfilled_from": str(fold_dir),
            },
        )
    if not fold_artifacts:
        raise DirectionComparisonError(f"No V3 P1 fold directories found below: {stage_root}")
    return write_direction_stage_report(
        stage_root / "evaluation",
        stage="v3_p1_pair_probe_backfill",
        fold_artifacts=fold_artifacts,
        metadata={
            "strategy_version": 3,
            "phase": "p1",
            "result_scope": "exploratory_partial_panel",
            "production_eligible": False,
            "source_run": str(root),
        },
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backfill V3 P1 per-fold direction comparison artifacts"
    )
    parser.add_argument(
        "--run-root",
        default=(
            "csj/runs/active_contract_panel_v3_partial/"
            "v3_partial_day_balanced_cuda"
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = backfill_v3_p1_direction_plots(args.run_root)
    except DirectionComparisonError as exc:
        parser.exit(2, f"V3 direction-plot backfill failed: {exc}\n")
    print(json.dumps({"direction_comparison_report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
