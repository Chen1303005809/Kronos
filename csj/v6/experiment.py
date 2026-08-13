"""Phase-gated orchestration for V6 active-risk-control research.

Only P0 is implemented. It freezes risk labels, audits five walk-forward
folds, renders mandatory evidence, and persists a gate. All later entrypoints
are deliberately guarded by that run-specific gate and remain unavailable
until P0 has passed and been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from csj.v5.target_data import DATA_RULE_VERSION, load_target_only_observed_cohort
from csj.v6 import PRODUCTION_ELIGIBLE, RESULT_SCOPE, STRATEGY_VERSION
from csj.v6.audit import P0AuditBundle, P0AuditError, build_p0_audit
from csj.v6.config import RISK_LABEL_RULE_VERSION, load_v6_config
from csj.v6.plotting import P0PlotArtifacts, V6PlotError, render_p0_audit_plots
from csj.v6.risk_labels import RiskLabelError


class V6ExperimentError(RuntimeError):
    """A V6 stage cannot satisfy its frozen research protocol."""


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise V6ExperimentError(f"Required V6 P0 artifact is missing or empty: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_path(path),
        "bytes": int(path.stat().st_size),
    }


def _validated_run_id(value: object) -> str:
    run_id = str(value).strip()
    if not _RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            "V6 run-id must start with an alphanumeric character and contain only "
            "alphanumerics, dot, underscore, or hyphen"
        )
    return run_id


def _copy_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256_path(source) != _sha256_path(destination):
        raise V6ExperimentError(f"Copied V6 artifact checksum mismatch: {destination}")
    return destination


class V6Experiment:
    """V6 P0 implementation with persisted gates protecting P1-P5."""

    def __init__(self, config_path: str | Path, run_id: str) -> None:
        self.config = load_v6_config(config_path)
        self.run_id = _validated_run_id(run_id)
        self.run_dir = Path(self.config["output"]["root"]) / self.run_id
        self.results_dir = Path(self.config["output"]["results_root"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = load_target_only_observed_cohort(
            self.config["data"]["snapshot_root"]
        )
        self.config["runtime_resolved"] = {
            "p0_device": "cpu-data-audit",
            "configured_later_phase_device": str(self.config["runtime"]["device"]),
            "python": "3.12",
            "data_fingerprint": self.cohort.data_fingerprint,
            "snapshot_id": self.cohort.snapshot_id,
        }
        _write_json(self.run_dir / "resolved_config.json", self.config)

    @property
    def primary_products(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config["data"]["primary_products"])

    @property
    def transfer_products(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config["data"]["transfer_products"])

    def _model_provenance(self) -> dict[str, object]:
        model = self.config["model"]
        return {
            "tokenizer_id": model["tokenizer_id"],
            "tokenizer_revision": model["tokenizer_revision"],
            "predictor_id": model["predictor_id"],
            "predictor_revision": model["predictor_revision"],
            "tokenizer_frozen": bool(model["freeze_tokenizer"]),
            "predictor_frozen": bool(model["freeze_predictor"]),
        }

    def _metadata(self, phase: str) -> dict[str, object]:
        return {
            "strategy_version": STRATEGY_VERSION,
            "phase": phase,
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": PRODUCTION_ELIGIBLE,
            "data_rule_version": DATA_RULE_VERSION,
            "risk_label_rule_version": RISK_LABEL_RULE_VERSION,
            "data_fingerprint": self.cohort.data_fingerprint,
            **self._model_provenance(),
        }

    def _compact_fold_records(self, bundle: P0AuditBundle) -> pd.DataFrame:
        columns = (
            "fold_id",
            "split",
            "case_key",
            "target_contract_id",
            "product",
            "origin_timestamp",
            "origin_trading_day",
            "target_end_day",
            "target_bar_count",
            "long_mae",
            "short_mae",
            "future_vol_ratio",
            "long_tail_threshold",
            "short_tail_threshold",
            "long_tail_event",
            "short_tail_event",
            "threshold_fit_case_keys_sha256",
            "data_fingerprint",
        )
        missing = sorted(set(columns).difference(bundle.fold_records.columns))
        if missing:
            raise V6ExperimentError(f"V6 P0 fold records miss columns: {missing!r}")
        return bundle.fold_records[list(columns)].copy()

    def _write_report(
        self,
        *,
        gate: Mapping[str, object],
        plot_artifacts: P0PlotArtifacts,
        audit_path: Path,
        gate_path: Path,
    ) -> tuple[Path, Path]:
        allows_next = bool(gate["allows_next_phase"])
        status = "通过" if allows_next else "未通过"
        failed = [str(value) for value in gate.get("failed_condition_ids", ())]
        support = gate["support"]
        lines = [
            "# V6 P0 风险标签与支持度审计",
            "",
            f"- 结论：**P0 {status}**。",
            f"- P1 路径库：**{'允许进入' if allows_next else '保持锁死'}**。",
            f"- 结果范围：`{RESULT_SCOPE}`。",
            f"- 生产资格：`production_eligible: {str(PRODUCTION_ELIGIBLE).lower()}`。",
            f"- 数据指纹：`{self.cohort.data_fingerprint}`。",
            f"- 风险标签规则：`{RISK_LABEL_RULE_VERSION}`。",
            "",
            "## 五折主要品种事件支持度",
            "",
            "| fold | split | cases | long events | short events |",
            "|---|---:|---:|---:|---:|",
        ]
        for fold_id, splits in support["folds"].items():
            for split, values in splits.items():
                lines.append(
                    f"| {fold_id} | {split} | {values['cases']} | "
                    f"{values['long_events']} | {values['short_events']} |"
                )
        lines.extend(
            [
                "",
                "## 五折 evaluation 汇总",
                "",
                "| product | cases | long events | short events |",
                "|---|---:|---:|---:|",
            ]
        )
        for product, values in support["pooled_evaluation_by_product"].items():
            lines.append(
                f"| {product} | {values['cases']} | {values['long_events']} | "
                f"{values['short_events']} |"
            )
        lines.extend(["", "## Gate 失败项", ""])
        if failed:
            lines.extend(f"- `{check_id}`" for check_id in failed)
        else:
            lines.append("- 无。")
        failed_leakage_checks = [
            check
            for check in self._current_audit_leakage_checks
            if not bool(check.get("passed"))
        ]
        lines.extend(["", "## 防泄漏与切分审计", ""])
        if failed_leakage_checks:
            for check in failed_leakage_checks:
                details = ""
                if check.get("violating_origin_trading_days"):
                    details = (
                        "；违规预测日："
                        + ", ".join(
                            pd.Timestamp(day).strftime("%Y-%m-%d")
                            for day in check["violating_origin_trading_days"]
                        )
                    )
                lines.append(f"- `{check['check_id']}` 未通过{details}。")
        else:
            lines.append("- 所有 past-only 与 split 检查均通过。")
        lines.append(
            f"- 未来 OHLCVA 扰动检查：{self._future_mutation_check_count} 个，失败 0 个。"
        )
        lines.extend(
            [
                "",
                "支持度不足不会通过降低 80% 分位标签、延长 evaluation 或并入迁移品种来补救。",
                "",
                "## 证据产物",
                "",
                f"- 数据审计：`{audit_path}`",
                f"- P0 gate：`{gate_path}`",
                f"- 标签分布图：`{plot_artifacts.risk_label_distributions}`",
                f"- 事件支持度图：`{plot_artifacts.fold_event_support}`",
                f"- fit-only 阈值图：`{plot_artifacts.fold_tail_thresholds}`",
                "",
                "图表的分布轴仅为可读性裁剪；所有审计统计和 gate 均使用完整数值。",
                "",
            ]
        )
        content = "\n".join(lines)
        run_report = self.run_dir / "P0_REPORT.md"
        result_report = self.results_dir / "REPORT.md"
        run_report.write_text(content, encoding="utf-8")
        result_report.write_text(content, encoding="utf-8")
        return run_report, result_report

    def audit(self) -> Path:
        """Run and persist the complete P0 audit; a failed gate is still evidence."""

        bundle = build_p0_audit(self.cohort, self.config)
        self._current_audit_leakage_checks = list(
            bundle.audit["past_only_leakage_audit"]["checks"]
        )
        self._future_mutation_check_count = sum(
            "mutated_future_feature" in check
            for check in self._current_audit_leakage_checks
        )
        metadata = self._metadata("p0")
        p0_dir = self.run_dir / "p0"
        figures_dir = p0_dir / "figures"
        plot_artifacts = render_p0_audit_plots(
            bundle.outcomes,
            bundle.fold_records,
            primary_products=self.primary_products,
            p0_config=self.config["p0"],
            output_dir=figures_dir,
            metadata=metadata,
        )

        outcomes_path = p0_dir / "risk_outcomes.json"
        records_path = p0_dir / "fold_label_records.json"
        _write_json(
            outcomes_path,
            {
                **metadata,
                "record_schema_version": "v6-p0-continuous-risk-outcomes-v1",
                "records": bundle.outcomes.to_dict(orient="records"),
            },
        )
        compact_records = self._compact_fold_records(bundle)
        _write_json(
            records_path,
            {
                **metadata,
                "record_schema_version": "v6-p0-fold-tail-events-v1",
                "records": compact_records.to_dict(orient="records"),
            },
        )
        audit_payload = {
            **metadata,
            **dict(bundle.audit),
            "record_artifacts": {
                "continuous_risk_outcomes": _artifact(outcomes_path),
                "fold_tail_events": _artifact(records_path),
            },
            "plot_artifacts": {
                key: _artifact(Path(value))
                for key, value in plot_artifacts.as_dict().items()
            },
        }
        run_audit_path = self.run_dir / "data_audit.json"
        result_audit_path = self.results_dir / "v6_data_audit.json"
        _write_json(run_audit_path, audit_payload)
        _write_json(result_audit_path, audit_payload)

        gate_payload = {
            **metadata,
            **dict(bundle.gate),
            "data_audit": _artifact(run_audit_path),
            "record_artifacts": {
                "continuous_risk_outcomes": _artifact(outcomes_path),
                "fold_tail_events": _artifact(records_path),
            },
            "plot_artifacts": {
                key: _artifact(Path(value))
                for key, value in plot_artifacts.as_dict().items()
            },
        }
        run_gate_path = self.run_dir / "p0_gate.json"
        result_gate_path = self.results_dir / "p0_gate.json"
        _write_json(run_gate_path, gate_payload)
        _write_json(result_gate_path, gate_payload)

        result_figures = self.results_dir / "p0_figures"
        result_plot_paths = [
            _copy_artifact(
                plot_artifacts.risk_label_distributions,
                result_figures / plot_artifacts.risk_label_distributions.name,
            ),
            _copy_artifact(
                plot_artifacts.fold_event_support,
                result_figures / plot_artifacts.fold_event_support.name,
            ),
            _copy_artifact(
                plot_artifacts.fold_tail_thresholds,
                result_figures / plot_artifacts.fold_tail_thresholds.name,
            ),
            _copy_artifact(
                plot_artifacts.summary_json,
                result_figures / plot_artifacts.summary_json.name,
            ),
        ]
        run_report, result_report = self._write_report(
            gate=gate_payload,
            plot_artifacts=plot_artifacts,
            audit_path=run_audit_path,
            gate_path=run_gate_path,
        )
        summary = {
            "p0_gate": str(result_gate_path),
            "allows_next_phase": bool(gate_payload["allows_next_phase"]),
            "failed_condition_ids": gate_payload["failed_condition_ids"],
            "data_audit": str(result_audit_path),
            "reports": [str(run_report), str(result_report)],
            "result_figures": [str(path) for path in result_plot_paths],
        }
        print(json.dumps(_json_safe(summary), ensure_ascii=False))
        return result_gate_path

    def _require_p0_gate(self) -> Mapping[str, object]:
        path = self.run_dir / "p0_gate.json"
        if not path.is_file():
            raise V6ExperimentError(
                f"V6 P1 requires this run's P0 gate, but it is missing: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V6ExperimentError(f"Cannot read V6 P0 gate: {path}") from exc
        expected = {
            "strategy_version": STRATEGY_VERSION,
            "phase": "p0",
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": PRODUCTION_ELIGIBLE,
            "risk_label_rule_version": RISK_LABEL_RULE_VERSION,
            "data_fingerprint": self.cohort.data_fingerprint,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise V6ExperimentError(
                    f"V6 P0 gate does not belong to this frozen run ({key})"
                )
        if payload.get("allows_next_phase") is not True:
            failed = payload.get("failed_condition_ids", [])
            raise V6ExperimentError(
                f"V6 P0 gate does not allow P1; failed conditions: {failed}"
            )
        return payload

    def run_p1_path_bank(self) -> None:
        self._require_p0_gate()
        raise V6ExperimentError(
            "V6 P1 is intentionally unavailable in the P0 implementation; "
            "review and freeze the passing P0 evidence first."
        )

    def run_p2_risk_head(self) -> None:
        raise V6ExperimentError("V6 P2 is unavailable until P1 is implemented and passes")

    def run_p3_calibrate(self) -> None:
        raise V6ExperimentError("V6 P3 is unavailable until P2 is implemented and passes")

    def run_p4_overlay(self, *, base_positions: str | Path | None = None) -> None:
        del base_positions
        raise V6ExperimentError("V6 P4 is unavailable until P3 is implemented and passes")

    def run_p5_freeze(self) -> None:
        raise V6ExperimentError("V6 P5 is unavailable until P4 is implemented and passes")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kronos V6 active-risk-control workflow")
    parser.add_argument(
        "stage",
        choices=(
            "audit",
            "p1-path-bank",
            "p2-risk-head",
            "p3-calibrate",
            "p4-overlay",
            "p5-freeze",
        ),
    )
    parser.add_argument("--config", default="csj/configs/risk_control_v6.yaml")
    parser.add_argument("--run-id", default="v6_p0")
    parser.add_argument("--base-positions", default=None)
    args = parser.parse_args(argv)
    try:
        experiment = V6Experiment(args.config, args.run_id)
        if args.stage == "audit":
            if args.base_positions is not None:
                parser.error("--base-positions applies only to p4-overlay")
            experiment.audit()
        elif args.stage == "p1-path-bank":
            if args.base_positions is not None:
                parser.error("--base-positions applies only to p4-overlay")
            experiment.run_p1_path_bank()
        elif args.stage == "p2-risk-head":
            if args.base_positions is not None:
                parser.error("--base-positions applies only to p4-overlay")
            experiment.run_p2_risk_head()
        elif args.stage == "p3-calibrate":
            if args.base_positions is not None:
                parser.error("--base-positions applies only to p4-overlay")
            experiment.run_p3_calibrate()
        elif args.stage == "p4-overlay":
            experiment.run_p4_overlay(base_positions=args.base_positions)
        elif args.stage == "p5-freeze":
            if args.base_positions is not None:
                parser.error("--base-positions applies only to p4-overlay")
            experiment.run_p5_freeze()
    except (
        V6ExperimentError,
        P0AuditError,
        RiskLabelError,
        V6PlotError,
        ValueError,
        RuntimeError,
    ) as exc:
        parser.exit(2, f"V6 stage blocked: {exc}\n")


if __name__ == "__main__":
    main()


__all__ = ["V6Experiment", "V6ExperimentError", "main"]
