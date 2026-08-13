"""Command-line P0 runner for V7 diversified active-risk-control research."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from csj.v5.target_data import load_target_only_observed_cohort
from csj.v6.plotting import render_p0_audit_plots
from csj.v7 import PRODUCTION_ELIGIBLE, RESULT_SCOPE, STRATEGY_VERSION
from csj.v7.audit import build_v7_p0_audit
from csj.v7.config import load_v7_config


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V7Experiment:
    def __init__(self, config_path: str | Path, run_id: str) -> None:
        self.config = load_v7_config(config_path)
        self.run_id = str(run_id)
        self.run_dir = Path(self.config["output"]["root"]) / self.run_id
        self.results_dir = Path(self.config["output"]["results_root"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = load_target_only_observed_cohort(self.config["data"]["snapshot_root"])
        _write(self.run_dir / "resolved_config.json", self.config)

    def audit(self) -> Path:
        bundle = build_v7_p0_audit(self.cohort, self.config)
        metadata = {
            "strategy_version": STRATEGY_VERSION,
            "phase": "p0",
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": PRODUCTION_ELIGIBLE,
            "snapshot_id": self.cohort.snapshot_id,
            "data_fingerprint": self.cohort.data_fingerprint,
        }
        figures = render_p0_audit_plots(
            bundle.outcomes,
            bundle.fold_records,
            primary_products=tuple(self.config["data"]["products"]),
            p0_config=self.config["p0"],
            output_dir=self.run_dir / "p0" / "figures",
            metadata=metadata,
        )
        outcomes_path = self.run_dir / "p0" / "risk_outcomes.json"
        records_path = self.run_dir / "p0" / "fold_label_records.json"
        audit_path = self.run_dir / "data_audit.json"
        gate_path = self.run_dir / "p0_gate.json"
        _write(outcomes_path, {**metadata, "records": bundle.outcomes.to_dict("records")})
        _write(records_path, {**metadata, "records": bundle.fold_records.to_dict("records")})
        _write(
            audit_path,
            {
                **metadata,
                **bundle.audit,
                "figures": {key: str(value) for key, value in figures.as_dict().items()},
            },
        )
        gate = {
            **metadata,
            **bundle.gate,
            "data_audit": {"path": str(audit_path), "sha256": _sha(audit_path)},
            "figures": {key: str(value) for key, value in figures.as_dict().items()},
        }
        _write(gate_path, gate)
        _write(self.results_dir / "v7_data_audit.json", {**metadata, **bundle.audit})
        _write(self.results_dir / "p0_gate.json", gate)
        report = [
            "# V7 P0 多品种主动风控数据门控",
            "",
            f"- 结论：**{'通过' if gate['allows_next_phase'] else '未通过'}**",
            f"- 快照：`{self.cohort.snapshot_id}`",
            f"- 产品数：{len(self.config['data']['products'])}",
            f"- 案例数：{bundle.audit['case_universe']['included_cases']}",
            f"- 预测原点日：{bundle.audit['case_universe']['origin_days']}",
            "",
            "## Fold 支持度",
            "",
            "| fold | split | cases | long | short |",
            "|---|---:|---:|---:|---:|",
        ]
        for fold, splits in gate["support"]["folds"].items():
            for split, values in splits.items():
                report.append(
                    f"| {fold} | {split} | {values['cases']} | "
                    f"{values['long_events']} | {values['short_events']} |"
                )
        report.extend(["", "## 失败项", ""])
        if gate["failed_condition_ids"]:
            report.extend(f"- `{value}`" for value in gate["failed_condition_ids"])
        else:
            report.append("- 无。")
        report.extend(
            [
                "",
                "本结果仍是回看式 observed-contract 证据，不能直接宣称生产有效。",
                "产品池只按上市时间和 bar 数冻结，未使用风险标签或 evaluation 表现筛选。",
            ]
        )
        text = "\n".join(report) + "\n"
        (self.run_dir / "P0_REPORT.md").write_text(text, encoding="utf-8")
        (self.results_dir / "REPORT.md").write_text(text, encoding="utf-8")
        print(json.dumps(_safe({"p0_gate": gate_path, "allows_next_phase": gate["allows_next_phase"], "failed_condition_ids": gate["failed_condition_ids"]}), ensure_ascii=False))
        return gate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos V7 risk-control P0")
    parser.add_argument("stage", choices=("audit",))
    parser.add_argument("--config", default="csj/configs/risk_control_v7.yaml")
    parser.add_argument("--run-id", default="v7_p0")
    args = parser.parse_args()
    V7Experiment(args.config, args.run_id).audit()


if __name__ == "__main__":
    main()
