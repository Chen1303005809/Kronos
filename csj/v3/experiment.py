"""CUDA-first orchestration for the V3 active concrete-contract panel strategy."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from csj.v3.config import load_v3_config
from csj.v3.p0 import (
    P0TrainingConfig,
    ProductDenseWindowDataset,
    evaluate_target_paths,
    load_p0_checkpoint,
    target_path_metrics,
    train_p0_ce,
)
from csj.v3.pair_probe import (
    PairProbe,
    ProbeTrainingConfig,
    assert_same_case_keys,
    evaluate_probe,
    load_probe_head,
    paired_block_bootstrap,
    probe_metrics,
    train_probe,
)
from csj.v3.panel_data import (
    PanelCase,
    PanelCaseBundle,
    V3WalkForwardFold,
    build_coverage_audit,
    load_panel_archive,
)
from model import Kronos, KronosTokenizer


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V3 requested CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported V3 device: {requested}")


def _cases_in_period(
    cases: Sequence[PanelCase],
    *,
    product: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> tuple[PanelCase, ...]:
    return tuple(
        case
        for case in cases
        if case.product == product and start_day <= case.target_end_day <= end_day
    )


def _p1_gate(
    pair_records: pd.DataFrame,
    target_records: pd.DataFrame,
    *,
    by_fold: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    bootstrap: dict[str, dict[str, object]],
    formal_complete_panel_provenance: bool,
) -> dict[str, object]:
    assert_same_case_keys_from_records(pair_records, target_records)
    pair_metric = probe_metrics(pair_records)
    target_metric = probe_metrics(target_records)
    pooled_improvement = float(pair_metric["balanced_accuracy"]) - float(
        target_metric["balanced_accuracy"]
    )
    products = sorted(set(pair_records["product"]))
    product_improvements: dict[str, float] = {}
    for product in products:
        product_improvements[product] = float(
            probe_metrics(pair_records.loc[pair_records["product"] == product])["balanced_accuracy"]
        ) - float(
            probe_metrics(target_records.loc[target_records["product"] == product])["balanced_accuracy"]
        )
    fold_improvements: dict[str, float] = {}
    for fold_id, (pair_fold, target_fold) in by_fold.items():
        fold_improvements[fold_id] = float(probe_metrics(pair_fold)["balanced_accuracy"]) - float(
            probe_metrics(target_fold)["balanced_accuracy"]
        )
    directions = set(pair_records["neighbor_direction"].astype(str))
    bootstrap_pass = all(
        bool(result.get("available"))
        and float(result.get("probability_improvement_positive", 0.0)) >= 0.80
        for result in bootstrap.values()
    )
    conditions = {
        "pooled_improvement_at_least_2pp": pooled_improvement >= 0.02,
        "majority_folds_improve": sum(value > 0.0 for value in fold_improvements.values())
        > len(fold_improvements) / 2,
        "no_product_degrades_more_than_1pp": all(
            value >= -0.01 for value in product_improvements.values()
        ),
        "both_neighbor_directions_present": directions == {"earlier", "later"},
        "bootstrap_5_and_10_day_probability_at_least_80pct": bootstrap_pass,
    }
    statistical_gate_passed = all(conditions.values())
    return {
        "pair_probe": pair_metric,
        "target_only_probe": target_metric,
        "pooled_balanced_accuracy_improvement": pooled_improvement,
        "product_balanced_accuracy_improvements": product_improvements,
        "fold_balanced_accuracy_improvements": fold_improvements,
        "conditions": conditions,
        "statistical_gate_passed": statistical_gate_passed,
        "formal_complete_panel_provenance": formal_complete_panel_provenance,
        "passes_p1_to_p2_gate": (
            formal_complete_panel_provenance and statistical_gate_passed
        ),
    }


def assert_same_case_keys_from_records(
    pair_records: pd.DataFrame,
    target_records: pd.DataFrame,
) -> None:
    pair_keys = pair_records["case_key"].tolist()
    target_keys = target_records["case_key"].tolist()
    if len(set(pair_keys)) != len(pair_keys) or len(set(target_keys)) != len(target_keys):
        raise RuntimeError("P1 result files contain duplicate case keys")
    if set(pair_keys) != set(target_keys):
        raise RuntimeError("P1 result files are not a strict paired comparison")


class V3Experiment:
    """Phase-gated V3 runner; P2 is intentionally absent until P1 passes."""

    def __init__(
        self,
        config_path: str | Path,
        run_id: str,
        *,
        device_override: str | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.config = load_v3_config(config_path)
        if device_override is not None:
            self.config["runtime"]["device"] = device_override
        self.device = resolve_device(str(self.config["runtime"]["device"]))
        self.run_id = run_id
        self.run_dir = Path(self.config["output"]["root"]) / run_id
        self.results_dir = Path(self.config["output"]["results_root"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.local_files_only = not allow_model_download
        self.archive = load_panel_archive(self.config["data"]["snapshot_root"])
        walk = self.config["walk_forward"]
        self.audit_payload, self.bundles, self.folds = build_coverage_audit(
            self.archive,
            lookbacks=self.config["data"]["lookbacks"],
            products=self.config["data"]["products"],
            minimum_fit_days=int(walk["minimum_fit_days"]),
            inner_validation_days=int(walk["inner_validation_days"]),
            evaluation_days=int(walk["evaluation_days"]),
            step_days=int(walk["step_days"]),
            purge_days=int(walk["purge_days"]),
        )
        self.lookback = int(
            self.audit_payload["context_length_decision"]["primary_context_length"]
        )
        self.bundle = self.bundles[self.lookback]
        self.config["runtime_resolved"] = {
            "device": str(self.device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else None,
            "model_download_allowed": allow_model_download,
            "primary_context_length": self.lookback,
        }
        _write_json(self.run_dir / "resolved_config.json", self.config)

    @property
    def products(self) -> tuple[str, ...]:
        return tuple(str(product) for product in self.config["data"]["products"])

    @property
    def uses_partial_panel_training(self) -> bool:
        """Whether this run deliberately admits non-formal panel cases."""

        return bool(self.config["data"]["allow_partial_panel_training"])

    @property
    def result_scope(self) -> str:
        return (
            "exploratory_partial_panel"
            if self.uses_partial_panel_training
            else "formal_complete_panel"
        )

    def audit(self) -> Path:
        path = self.results_dir / "v3_data_audit.json"
        _write_json(path, self.audit_payload)
        _write_json(self.run_dir / "v3_data_audit.json", self.audit_payload)
        print(json.dumps(_json_safe({"data_audit": path, "primary_context_length": self.lookback}), ensure_ascii=False))
        return path

    def _load_tokenizer(self) -> KronosTokenizer:
        model = self.config["model"]
        tokenizer = KronosTokenizer.from_pretrained(
            model["tokenizer_id"],
            revision=model["tokenizer_revision"],
            cache_dir=model["cache_dir"],
            local_files_only=self.local_files_only,
        )
        tokenizer.requires_grad_(False)
        tokenizer.eval()
        return tokenizer

    def _load_predictor(self) -> Kronos:
        model = self.config["model"]
        return Kronos.from_pretrained(
            model["predictor_id"],
            revision=model["predictor_revision"],
            cache_dir=model["cache_dir"],
            local_files_only=self.local_files_only,
        )

    def _p0_evaluation_arguments(self) -> dict[str, object]:
        evaluation = self.config["evaluation"]
        data = self.config["data"]
        return {
            "device": self.device,
            "max_context": int(self.config["model"]["max_context"]),
            "clip": float(data["clip"]),
            "epsilon": float(data["normalization_epsilon"]),
            "sample_count": int(evaluation["sample_count"]),
            "temperature": float(evaluation["temperature"]),
            "top_k": int(evaluation["top_k"]),
            "top_p": float(evaluation["top_p"]),
            "batch_size": int(evaluation["inference_batch_size"]),
            "seed": int(evaluation["random_seed"]),
        }

    def _release(self, *models: torch.nn.Module) -> None:
        for model in models:
            model.to("cpu")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _require_folds(self) -> tuple[V3WalkForwardFold, ...]:
        if not self.folds:
            raise RuntimeError(
                "No complete V3 walk-forward folds are available. Continue snapshot archiving "
                "before running CUDA training."
            )
        return self.folds

    def run_p0(self, *, train: bool) -> Path:
        """Run zero-shot P0 or CE-only P0, separately for each product/fold."""

        folds = self._require_folds()
        all_outputs: dict[str, list[pd.DataFrame]] = {"zero_shot": [], "ce_only": []}
        stage = "p0_ce_only" if train else "p0_zero_shot"
        for fold in folds:
            for product in self.products:
                validation_cases = _cases_in_period(
                    self.bundle.target_cases,
                    product=product,
                    start_day=fold.inner_validation_start_day,
                    end_day=fold.inner_validation_end_day,
                )
                evaluation_cases = _cases_in_period(
                    self.bundle.target_cases,
                    product=product,
                    start_day=fold.evaluation_start_day,
                    end_day=fold.evaluation_end_day,
                )
                if not validation_cases or not evaluation_cases:
                    continue
                tokenizer = self._load_tokenizer()
                predictor = self._load_predictor()
                stage_dir = self.run_dir / stage / fold.fold_id / product
                try:
                    evaluation_args = self._p0_evaluation_arguments()
                    zero_records = evaluate_target_paths(
                        tokenizer,
                        predictor,
                        evaluation_cases,
                        model_name="zero_shot",
                        **evaluation_args,
                    )
                    _write_json(stage_dir / "zero_shot_records.json", zero_records.to_dict("records"))
                    all_outputs["zero_shot"].append(zero_records)
                    if not train:
                        continue
                    train_dataset = ProductDenseWindowDataset(
                        tuple(self.archive.contracts.values()),
                        product=product,
                        fit_end_day=fold.fit_end_day,
                        lookback=self.lookback,
                        clip=float(self.config["data"]["clip"]),
                        epsilon=float(self.config["data"]["normalization_epsilon"]),
                    )
                    result = train_p0_ce(
                        predictor,
                        tokenizer,
                        train_dataset,
                        validation_cases,
                        config=P0TrainingConfig(**self.config["p0"]["training"]),
                        device=self.device,
                        output_dir=stage_dir / "training",
                        max_context=int(self.config["model"]["max_context"]),
                        clip=float(self.config["data"]["clip"]),
                        epsilon=float(self.config["data"]["normalization_epsilon"]),
                        validation_sample_count=int(self.config["evaluation"]["sample_count"]),
                        validation_temperature=float(self.config["evaluation"]["temperature"]),
                        validation_top_k=int(self.config["evaluation"]["top_k"]),
                        validation_top_p=float(self.config["evaluation"]["top_p"]),
                        validation_batch_size=int(self.config["evaluation"]["inference_batch_size"]),
                    )
                    load_p0_checkpoint(predictor, result.checkpoint_path)
                    ce_records = evaluate_target_paths(
                        tokenizer,
                        predictor,
                        evaluation_cases,
                        model_name="ce_only",
                        **evaluation_args,
                    )
                    _write_json(stage_dir / "ce_only_records.json", ce_records.to_dict("records"))
                    _write_json(
                        stage_dir / "training_summary.json",
                        {
                            "checkpoint": result.checkpoint_path,
                            "best_epoch": result.best_epoch,
                            "best_metrics": result.best_metrics,
                            "elapsed_seconds": result.elapsed_seconds,
                        },
                    )
                    all_outputs["ce_only"].append(ce_records)
                finally:
                    self._release(tokenizer, predictor)
        payload: dict[str, object] = {
            "stage": stage,
            "context_length": self.lookback,
            "device": str(self.device),
            "result_scope": self.result_scope,
            "formal_panel_conclusion_eligible": not self.uses_partial_panel_training,
            "records": {},
        }
        for name, records in all_outputs.items():
            if records:
                combined = pd.concat(records, ignore_index=True)
                payload["records"][name] = {
                    "count": len(combined),
                    "metrics": target_path_metrics(combined),
                }
        if not payload["records"]:
            raise RuntimeError("P0 produced no fold/product records")
        path = self.results_dir / f"{stage}_metrics.json"
        _write_json(path, payload)
        _write_json(self.run_dir / stage / "metrics.json", payload)
        print(json.dumps(_json_safe({"stage": stage, "metrics": path}), ensure_ascii=False))
        return path

    def _p1_cases(self) -> tuple[PanelCase, ...]:
        if self.uses_partial_panel_training:
            cases = self.bundle.all_pair_cases
            provenance = "exploratory_partial_panel_opt_in"
        else:
            cases = self.bundle.strict_pair_cases
            provenance = "strict_complete_panel_only"
        if not cases and self.uses_partial_panel_training:
            raise RuntimeError(
                "P1 exploratory training is enabled, but the archive contains no valid "
                "timestamp-aligned nearest-neighbour pair cases."
            )
        if not cases:
            raise RuntimeError(
                "P1 is blocked by the V3 data gate: no strict complete-panel pair cases exist. "
                "Archive a complete active-contract manifest on each trading day; do not set "
                "allow_partial_panel_training merely to turn this into a formal result."
            )
        _write_json(
            self.run_dir / "p1_case_provenance.json",
            {
                "source": provenance,
                "strict_pair_cases": len(self.bundle.strict_pair_cases),
                "partial_pair_cases": len(self.bundle.partial_pair_cases),
                "used_cases": len(cases),
                "formal_p1_to_p2_eligible": not self.uses_partial_panel_training,
            },
        )
        return cases

    def run_p1(self) -> Path:
        """Run paired P1 probes, optionally as explicitly-labelled exploratory research."""

        folds = self._require_folds()
        pair_cases = self._p1_cases()
        target_records_by_fold: dict[str, list[pd.DataFrame]] = {}
        pair_records_by_fold: dict[str, list[pd.DataFrame]] = {}
        p1 = self.config["p1"]
        training_config = ProbeTrainingConfig(**p1["training"])
        data = self.config["data"]
        for fold in folds:
            target_records_by_fold[fold.fold_id] = []
            pair_records_by_fold[fold.fold_id] = []
            for product in self.products:
                fit_cases = _cases_in_period(
                    pair_cases,
                    product=product,
                    start_day=fold.fit_start_day,
                    end_day=fold.fit_end_day,
                )
                validation_cases = _cases_in_period(
                    pair_cases,
                    product=product,
                    start_day=fold.inner_validation_start_day,
                    end_day=fold.inner_validation_end_day,
                )
                evaluation_cases = _cases_in_period(
                    pair_cases,
                    product=product,
                    start_day=fold.evaluation_start_day,
                    end_day=fold.evaluation_end_day,
                )
                if not fit_cases or not validation_cases or not evaluation_cases:
                    continue
                assert_same_case_keys(fit_cases, fit_cases)
                assert_same_case_keys(validation_cases, validation_cases)
                assert_same_case_keys(evaluation_cases, evaluation_cases)
                tokenizer = self._load_tokenizer()
                predictor = self._load_predictor()
                stage_dir = self.run_dir / "p1" / fold.fold_id / product
                try:
                    torch.manual_seed(training_config.seed)
                    template = PairProbe(
                        tokenizer,
                        predictor,
                        fusion_hidden_dim=int(p1["fusion_hidden_dim"]),
                        dropout=float(p1["dropout"]),
                    )
                    initial_head = template.head_state_dict()
                    target_probe = PairProbe(
                        tokenizer,
                        predictor,
                        fusion_hidden_dim=int(p1["fusion_hidden_dim"]),
                        dropout=float(p1["dropout"]),
                    )
                    pair_probe = PairProbe(
                        tokenizer,
                        predictor,
                        fusion_hidden_dim=int(p1["fusion_hidden_dim"]),
                        dropout=float(p1["dropout"]),
                    )
                    target_probe.load_head_state_dict(initial_head)
                    pair_probe.load_head_state_dict(initial_head)
                    target_result = train_probe(
                        target_probe,
                        fit_cases,
                        validation_cases,
                        mode="target_only_probe",
                        config=training_config,
                        device=self.device,
                        output_dir=stage_dir / "target_only_probe",
                        clip=float(data["clip"]),
                        epsilon=float(data["normalization_epsilon"]),
                    )
                    pair_result = train_probe(
                        pair_probe,
                        fit_cases,
                        validation_cases,
                        mode="pair_probe",
                        config=training_config,
                        device=self.device,
                        output_dir=stage_dir / "pair_probe",
                        clip=float(data["clip"]),
                        epsilon=float(data["normalization_epsilon"]),
                    )
                    load_probe_head(target_probe, target_result.checkpoint_path)
                    load_probe_head(pair_probe, pair_result.checkpoint_path)
                    target_records = evaluate_probe(
                        target_probe,
                        evaluation_cases,
                        mode="target_only_probe",
                        device=self.device,
                        batch_size=training_config.batch_size,
                        clip=float(data["clip"]),
                        epsilon=float(data["normalization_epsilon"]),
                    )
                    pair_records = evaluate_probe(
                        pair_probe,
                        evaluation_cases,
                        mode="pair_probe",
                        device=self.device,
                        batch_size=training_config.batch_size,
                        clip=float(data["clip"]),
                        epsilon=float(data["normalization_epsilon"]),
                    )
                    assert_same_case_keys_from_records(pair_records, target_records)
                    _write_json(stage_dir / "target_only_records.json", target_records.to_dict("records"))
                    _write_json(stage_dir / "pair_records.json", pair_records.to_dict("records"))
                    _write_json(
                        stage_dir / "summary.json",
                        {
                            "target_only": {
                                "checkpoint": target_result.checkpoint_path,
                                "best_epoch": target_result.best_epoch,
                                "best_balanced_accuracy": target_result.best_balanced_accuracy,
                                "elapsed_seconds": target_result.elapsed_seconds,
                            },
                            "pair": {
                                "checkpoint": pair_result.checkpoint_path,
                                "best_epoch": pair_result.best_epoch,
                                "best_balanced_accuracy": pair_result.best_balanced_accuracy,
                                "elapsed_seconds": pair_result.elapsed_seconds,
                            },
                            "target_only_metrics": probe_metrics(target_records),
                            "pair_metrics": probe_metrics(pair_records),
                        },
                    )
                    target_records_by_fold[fold.fold_id].append(target_records)
                    pair_records_by_fold[fold.fold_id].append(pair_records)
                finally:
                    self._release(tokenizer, predictor)

        usable_folds = {
            fold_id: (
                pd.concat(pair_records_by_fold[fold_id], ignore_index=True),
                pd.concat(target_records_by_fold[fold_id], ignore_index=True),
            )
            for fold_id in pair_records_by_fold
            if pair_records_by_fold[fold_id] and target_records_by_fold[fold_id]
        }
        if not usable_folds:
            raise RuntimeError("P1 produced no complete fold/product paired records")
        combined_pair = pd.concat([value[0] for value in usable_folds.values()], ignore_index=True)
        combined_target = pd.concat([value[1] for value in usable_folds.values()], ignore_index=True)
        evaluation = self.config["evaluation"]
        bootstrap: dict[str, dict[str, object]] = {}
        unique_days = combined_pair["target_end_day"].nunique()
        for block_days in evaluation["bootstrap_block_days"]:
            block = int(block_days)
            if unique_days < block:
                bootstrap[f"block_{block}"] = {
                    "available": False,
                    "reason": f"only {unique_days} evaluation days for {block}-day blocks",
                }
                continue
            bootstrap[f"block_{block}"] = {
                "available": True,
                **paired_block_bootstrap(
                    combined_pair,
                    combined_target,
                    block_days=block,
                    iterations=int(evaluation["bootstrap_iterations"]),
                    seed=int(evaluation["random_seed"]) + block,
                ),
            }
        gate = _p1_gate(
            combined_pair,
            combined_target,
            by_fold=usable_folds,
            bootstrap=bootstrap,
            formal_complete_panel_provenance=not self.uses_partial_panel_training,
        )
        payload = {
            "stage": "p1_pair_probe",
            "context_length": self.lookback,
            "device": str(self.device),
            "result_scope": self.result_scope,
            "formal_p1_to_p2_eligible": not self.uses_partial_panel_training,
            "record_count": len(combined_pair),
            "folds": list(usable_folds),
            "bootstrap": bootstrap,
            "gate": gate,
        }
        path = self.results_dir / "p1_pair_probe_metrics.json"
        _write_json(path, payload)
        _write_json(self.run_dir / "p1" / "metrics.json", payload)
        print(json.dumps(_json_safe({"stage": "p1", "metrics": path, "gate": gate}), ensure_ascii=False))
        return path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Kronos V3 active concrete-contract panel experiment"
    )
    parser.add_argument("stage", choices=("audit", "p0-zero-shot", "p0", "p1"))
    parser.add_argument("--config", default="csj/configs/active_contract_panel_v3.yaml")
    parser.add_argument("--run-id", default="v3_cuda")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args(argv)
    try:
        experiment = V3Experiment(
            args.config,
            args.run_id,
            device_override=args.device,
            allow_model_download=args.allow_model_download,
        )
        if args.stage == "audit":
            experiment.audit()
        elif args.stage == "p0-zero-shot":
            experiment.run_p0(train=False)
        elif args.stage == "p0":
            experiment.run_p0(train=True)
        elif args.stage == "p1":
            experiment.run_p1()
    except RuntimeError as exc:
        parser.exit(2, f"V3 stage blocked: {exc}\n")


if __name__ == "__main__":
    main()
