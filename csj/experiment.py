from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from csj.config import load_config
from csj.evaluation import predict_cases, resolve_device
from csj.futures_data import (
    MultiContractWindowDataset,
    build_forecast_cases,
    chronological_split,
    describe_splits,
    load_contracts,
)
from csj.metrics import (
    compute_metrics,
    ensemble_records,
    make_naive_baselines,
    metrics_with_instruments,
    paired_block_bootstrap_improvement,
    select_strongest_baseline,
)
from csj.reporting import (
    plot_forecast_examples,
    plot_metric_comparison,
    plot_training_histories,
    write_report,
)
from csj.training import TrainingResult, load_checkpoint, train_predictor
from csj.utils.tool import MODEL_FEATURES
from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import auto_regressive_inference


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _save_records(records: pd.DataFrame, path: Path) -> None:
    _write_json(path, records.to_dict(orient="records"))
    csv_records = records.copy()
    for column in csv_records.columns:
        if csv_records[column].map(lambda value: isinstance(value, list)).any():
            csv_records[column] = csv_records[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, list)
                else value
            )
    csv_records.to_csv(path.with_suffix(".csv"), index=False)


def _load_records(path: Path) -> pd.DataFrame:
    records = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    records["target_day"] = pd.to_datetime(records["target_day"])
    return records


def _load_training_result(output_dir: Path) -> TrainingResult | None:
    summary_path = output_dir / "summary.json"
    history_path = output_dir / "history.json"
    checkpoint_path = output_dir / "best_model.pt"
    if not (summary_path.exists() and history_path.exists() and checkpoint_path.exists()):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=int(summary["best_epoch"]),
        best_score=float(summary["best_score"]),
        best_return_mae=float(summary["best_return_mae"]),
        history=history,
        elapsed_seconds=float(summary["elapsed_seconds"]),
    )


def _release_model(model: torch.nn.Module, device: torch.device) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


class FuturesExperiment:
    def __init__(self, config_path: str | Path, run_id: str) -> None:
        self.config = load_config(config_path)
        self.run_dir = Path(self.config["output"]["root"]) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(str(self.config["training"]["device"]))

        data_config = self.config["data"]
        self.frames, self.audits = load_contracts(
            data_config["paths"],
            valid_bar_counts=data_config["valid_bar_counts"],
        )
        self.boundaries = chronological_split(
            self.frames,
            train_ratio=float(data_config["train_ratio"]),
            val_ratio=float(data_config["val_ratio"]),
            test_ratio=float(data_config["test_ratio"]),
        )
        self.lookback = int(data_config["lookback"])
        self.train_cases = build_forecast_cases(
            self.frames, self.boundaries, "train", self.lookback
        )
        self.val_cases = build_forecast_cases(
            self.frames, self.boundaries, "val", self.lookback
        )
        self.test_cases = build_forecast_cases(
            self.frames, self.boundaries, "test", self.lookback
        )
        self.split_summary = describe_splits(
            self.frames, self.boundaries, self.lookback
        )
        self.train_dataset = MultiContractWindowDataset(
            self.frames,
            self.boundaries,
            split="train",
            lookback=self.lookback,
            horizon=int(data_config["train_horizon"]),
            clip=float(data_config["clip"]),
        )
        self.evaluation_config = {
            **self.config["evaluation"],
            "max_context": int(self.config["model"]["max_context"]),
            "clip": float(data_config["clip"]),
        }

        _write_json(self.run_dir / "resolved_config.json", self.config)
        _write_json(self.run_dir / "data_audit.json", self.audits)
        _write_json(self.run_dir / "split_summary.json", self.split_summary)

    def audit(self) -> None:
        print(json.dumps(self.audits, ensure_ascii=False, indent=2))
        print(json.dumps(self.split_summary, ensure_ascii=False, indent=2))
        print(f"training_windows={len(self.train_dataset)}")
        print(f"device={self.device}")

    def load_tokenizer(self, *, local_files_only: bool = False) -> KronosTokenizer:
        model_config = self.config["model"]
        tokenizer = KronosTokenizer.from_pretrained(
            model_config["tokenizer_id"],
            revision=model_config["tokenizer_revision"],
            cache_dir=model_config["cache_dir"],
            local_files_only=local_files_only,
        )
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        return tokenizer

    def load_fresh_model(self, *, local_files_only: bool = False) -> Kronos:
        model_config = self.config["model"]
        return Kronos.from_pretrained(
            model_config["predictor_id"],
            revision=model_config["predictor_revision"],
            cache_dir=model_config["cache_dir"],
            local_files_only=local_files_only,
        )

    def download(self) -> None:
        tokenizer = self.load_tokenizer(local_files_only=False)
        model = self.load_fresh_model(local_files_only=False)
        print(
            f"downloaded tokenizer_params={sum(p.numel() for p in tokenizer.parameters()):,} "
            f"predictor_params={sum(p.numel() for p in model.parameters()):,}"
        )

    def _predict_or_load(
        self,
        output_path: Path,
        model: torch.nn.Module,
        tokenizer: torch.nn.Module,
        cases: list,
        *,
        evaluation_overrides: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if output_path.exists():
            return _load_records(output_path)
        evaluation = {**self.evaluation_config, **(evaluation_overrides or {})}
        records = predict_cases(
            model,
            tokenizer,
            cases,
            device=self.device,
            max_context=int(evaluation["max_context"]),
            clip=float(evaluation["clip"]),
            sample_count=int(evaluation["sample_count"]),
            temperature=float(evaluation["temperature"]),
            top_k=int(evaluation["top_k"]),
            top_p=float(evaluation["top_p"]),
            batch_size=int(evaluation["inference_batch_size"]),
            seed=int(evaluation["random_seed"]),
        )
        _save_records(records, output_path)
        return records

    def _train_or_load(
        self,
        tokenizer: torch.nn.Module,
        *,
        learning_rate: float,
        seed: int,
        validation_cases: list,
        output_dir: Path,
        max_epochs: int | None = None,
        max_train_batches: int | None = None,
        evaluation_overrides: dict[str, Any] | None = None,
    ) -> TrainingResult:
        existing = _load_training_result(output_dir)
        if existing is not None:
            print(f"resuming completed run: {output_dir.name}")
            return existing
        model = self.load_fresh_model(local_files_only=True)
        training = self.config["training"]
        evaluation = {**self.evaluation_config, **(evaluation_overrides or {})}
        try:
            return train_predictor(
                model,
                tokenizer,
                self.train_dataset,
                validation_cases,
                device=self.device,
                output_dir=output_dir,
                learning_rate=learning_rate,
                seed=seed,
                max_epochs=max_epochs or int(training["max_epochs"]),
                early_stopping_patience=int(training["early_stopping_patience"]),
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
                weight_decay=float(training["weight_decay"]),
                gradient_clip=float(training["gradient_clip"]),
                warmup_ratio=float(training["warmup_ratio"]),
                evaluation_config=evaluation,
                max_train_batches=max_train_batches,
            )
        finally:
            _release_model(model, self.device)

    def smoke(self) -> None:
        tokenizer = self.load_tokenizer(local_files_only=True)
        selected_cases = []
        for pred_len in sorted({case.pred_len for case in self.val_cases}):
            selected_cases.extend(
                [case for case in self.val_cases if case.pred_len == pred_len][:1]
            )
        smoke_dir = self.run_dir / "smoke_train"
        result = self._train_or_load(
            tokenizer,
            learning_rate=3e-6,
            seed=42,
            validation_cases=selected_cases,
            output_dir=smoke_dir,
            max_epochs=1,
            max_train_batches=20,
            evaluation_overrides={"sample_count": 2, "inference_batch_size": 1},
        )
        print(
            f"smoke complete: best_epoch={result.best_epoch} "
            f"val_bal_acc={result.best_score:.4f} device={self.device}"
        )

    def baseline(self) -> None:
        tokenizer = self.load_tokenizer(local_files_only=True)
        validation_dir = self.run_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        naive_validation = make_naive_baselines(self.train_cases, self.val_cases)
        for name, records in naive_validation.items():
            _save_records(records, validation_dir / f"{name}.json")

        model = self.load_fresh_model(local_files_only=True)
        started_at = time.monotonic()
        records = self._predict_or_load(
            validation_dir / "zero_shot.json",
            model,
            tokenizer,
            self.val_cases,
        )
        elapsed = time.monotonic() - started_at
        _release_model(model, self.device)
        metrics = {
            "zero_shot": compute_metrics(records),
            **{
                name: compute_metrics(baseline_records)
                for name, baseline_records in naive_validation.items()
            },
        }
        _write_json(validation_dir / "baseline_metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"validation baseline elapsed_seconds={elapsed:.2f} device={self.device}")

    def consistency(self) -> None:
        tokenizer = self.load_tokenizer(local_files_only=True)
        model = self.load_fresh_model(local_files_only=True)
        tokenizer.eval()
        model.eval()
        case = self.val_cases[0]

        predictor = KronosPredictor(
            model,
            tokenizer,
            device=str(self.device),
            max_context=int(self.config["model"]["max_context"]),
            clip=float(self.config["data"]["clip"]),
        )
        native = predictor.predict(
            df=case.context[MODEL_FEATURES],
            x_timestamp=case.context["timestamps"],
            y_timestamp=case.target["timestamps"],
            pred_len=case.pred_len,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )
        custom = predict_cases(
            model,
            tokenizer,
            [case],
            device=self.device,
            max_context=int(self.config["model"]["max_context"]),
            clip=float(self.config["data"]["clip"]),
            sample_count=1,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            batch_size=1,
            seed=int(self.config["evaluation"]["random_seed"]),
        )

        native_close = native["close"].to_numpy(dtype=np.float64)
        custom_close = np.asarray(custom.iloc[0]["predicted_close_path"], dtype=np.float64)
        max_abs_close_diff = float(np.max(np.abs(native_close - custom_close)))
        max_rel_close_diff = float(
            np.max(np.abs(native_close - custom_close) / (np.abs(native_close) + 1e-12))
        )

        # Compare every decoded feature before the production OHLC sanitation.
        # This explicitly verifies the scale contract: the network output is in
        # standardized space and must be inverted with this context's statistics.
        context = case.context[MODEL_FEATURES].to_numpy(dtype=np.float64)
        mean = context.mean(axis=0)
        std = context.std(axis=0)
        normalized = (context - mean) / (std + 1e-5)
        model_input = np.clip(
            normalized,
            -float(self.config["data"]["clip"]),
            float(self.config["data"]["clip"]),
        ).astype(np.float32)
        x = torch.from_numpy(model_input[None, ...]).to(self.device)
        x_stamp = torch.from_numpy(
            case.context[["minute", "hour", "weekday", "day", "month"]]
            .to_numpy(dtype=np.float32)[None, ...]
        ).to(self.device)
        y_stamp = torch.from_numpy(
            case.target[["minute", "hour", "weekday", "day", "month"]]
            .to_numpy(dtype=np.float32)[None, ...]
        ).to(self.device)
        raw_normalized = auto_regressive_inference(
            tokenizer,
            model,
            x,
            x_stamp,
            y_stamp,
            max_context=int(self.config["model"]["max_context"]),
            pred_len=case.pred_len,
            clip=float(self.config["data"]["clip"]),
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
            return_samples=True,
        )[0, 0, -case.pred_len :, :]
        raw_custom = raw_normalized * (std + 1e-5) + mean
        native_values = native[MODEL_FEATURES].to_numpy(dtype=np.float64)
        feature_diffs = np.abs(native_values - raw_custom)
        feature_relative_diffs = feature_diffs / (np.abs(native_values) + 1e-12)
        per_feature = {
            feature: {
                "max_abs_diff": float(feature_diffs[:, index].max()),
                "max_rel_diff": float(feature_relative_diffs[:, index].max()),
            }
            for index, feature in enumerate(MODEL_FEATURES)
        }
        reconstructed = normalized * (std + 1e-5) + mean
        roundtrip_max_abs = float(np.max(np.abs(context - reconstructed)))
        result = {
            "instrument": case.instrument,
            "target_day": case.target_day,
            "native_close": native_close,
            "custom_close": custom_close,
            "max_abs_close_diff": max_abs_close_diff,
            "max_rel_close_diff": max_rel_close_diff,
            "raw_all_features_max_abs_diff": float(feature_diffs.max()),
            "raw_all_features_max_rel_diff": float(feature_relative_diffs.max()),
            "per_feature": per_feature,
            "normalization_roundtrip_max_abs": roundtrip_max_abs,
        }
        _write_json(self.run_dir / "normalization_consistency.json", result)
        print(json.dumps(result, default=_json_default, ensure_ascii=False, indent=2))
        _release_model(model, self.device)
        if max(max_rel_close_diff, float(feature_relative_diffs.max())) > 1e-5:
            raise RuntimeError(
                "Custom inverse normalization does not match native KronosPredictor"
            )

    def full(self) -> Path:
        consistency_path = self.run_dir / "normalization_consistency.json"
        if not consistency_path.exists():
            self.consistency()
        tokenizer = self.load_tokenizer(local_files_only=True)
        validation_dir = self.run_dir / "validation"
        test_dir = self.run_dir / "sealed_test"
        validation_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        naive_validation = make_naive_baselines(self.train_cases, self.val_cases)
        for name, records in naive_validation.items():
            _save_records(records, validation_dir / f"{name}.json")

        zero_shot_model = self.load_fresh_model(local_files_only=True)
        zero_shot_validation = self._predict_or_load(
            validation_dir / "zero_shot.json",
            zero_shot_model,
            tokenizer,
            self.val_cases,
        )
        _release_model(zero_shot_model, self.device)

        training = self.config["training"]
        primary_seed = int(training["seeds"][0])
        search_results: dict[float, TrainingResult] = {}
        all_results: dict[str, TrainingResult] = {}
        for learning_rate in map(float, training["learning_rates"]):
            label = f"lr_{learning_rate:.0e}_seed_{primary_seed}"
            result = self._train_or_load(
                tokenizer,
                learning_rate=learning_rate,
                seed=primary_seed,
                validation_cases=self.val_cases,
                output_dir=self.run_dir / "training" / label,
            )
            search_results[learning_rate] = result
            all_results[label] = result

        selected_learning_rate = max(
            search_results,
            key=lambda lr: (
                search_results[lr].best_score,
                -search_results[lr].best_return_mae,
            ),
        )
        _write_json(
            self.run_dir / "selected_learning_rate.json",
            {
                "selected_learning_rate": selected_learning_rate,
                "validation_scores": {
                    str(lr): {
                        "best_score": result.best_score,
                        "best_return_mae": result.best_return_mae,
                        "best_epoch": result.best_epoch,
                    }
                    for lr, result in search_results.items()
                },
            },
        )

        selected_results: dict[int, TrainingResult] = {
            primary_seed: search_results[selected_learning_rate]
        }
        for seed in map(int, training["seeds"][1:]):
            label = f"lr_{selected_learning_rate:.0e}_seed_{seed}"
            result = self._train_or_load(
                tokenizer,
                learning_rate=selected_learning_rate,
                seed=seed,
                validation_cases=self.val_cases,
                output_dir=self.run_dir / "training" / label,
            )
            selected_results[seed] = result
            all_results[label] = result

        validation_seed_records: list[pd.DataFrame] = []
        for seed, result in sorted(selected_results.items()):
            model = self.load_fresh_model(local_files_only=True)
            load_checkpoint(model, result.checkpoint_path)
            records = self._predict_or_load(
                validation_dir / f"fine_tuned_seed_{seed}.json",
                model,
                tokenizer,
                self.val_cases,
            )
            validation_seed_records.append(records)
            _release_model(model, self.device)
        validation_ensemble = ensemble_records(validation_seed_records)
        _save_records(validation_ensemble, validation_dir / "fine_tuned_ensemble.json")

        strongest_baseline = select_strongest_baseline(
            naive_validation,
            zero_shot_validation,
        )
        _write_json(
            self.run_dir / "strongest_baseline.json",
            {"name": strongest_baseline},
        )

        # The sealed test is first touched only after LR selection and all seeds finish.
        naive_test = make_naive_baselines(self.train_cases, self.test_cases)
        for name, records in naive_test.items():
            _save_records(records, test_dir / f"{name}.json")
        zero_shot_test_model = self.load_fresh_model(local_files_only=True)
        zero_shot_test = self._predict_or_load(
            test_dir / "zero_shot.json",
            zero_shot_test_model,
            tokenizer,
            self.test_cases,
        )
        _release_model(zero_shot_test_model, self.device)

        test_seed_records: list[pd.DataFrame] = []
        for seed, result in sorted(selected_results.items()):
            model = self.load_fresh_model(local_files_only=True)
            load_checkpoint(model, result.checkpoint_path)
            records = self._predict_or_load(
                test_dir / f"fine_tuned_seed_{seed}.json",
                model,
                tokenizer,
                self.test_cases,
            )
            test_seed_records.append(records)
            _release_model(model, self.device)
        test_ensemble = ensemble_records(test_seed_records)
        _save_records(test_ensemble, test_dir / "fine_tuned_ensemble.json")

        validation_records = {
            "zero_shot": zero_shot_validation,
            **naive_validation,
            "fine_tuned_ensemble": validation_ensemble,
        }
        test_records = {
            "zero_shot": zero_shot_test,
            **naive_test,
            "fine_tuned_ensemble": test_ensemble,
        }
        baseline_test_records = (
            zero_shot_test
            if strongest_baseline == "zero_shot"
            else naive_test[strongest_baseline]
        )
        evaluation = self.config["evaluation"]
        bootstrap = paired_block_bootstrap_improvement(
            test_ensemble,
            baseline_test_records,
            iterations=int(evaluation["bootstrap_iterations"]),
            block_days=int(evaluation["bootstrap_block_days"]),
            seed=int(evaluation["random_seed"]),
        )

        sensitivity_records = {
            label: records.loc[~records["large_opening_gap"].astype(bool)].copy()
            if "large_opening_gap" in records
            else records.copy()
            for label, records in test_records.items()
        }
        sensitivity_metrics = {
            label: compute_metrics(records)
            for label, records in sensitivity_records.items()
        }

        metrics_payload = {
            "validation": {
                label: metrics_with_instruments(records)
                for label, records in validation_records.items()
            },
            "sealed_test": {
                label: metrics_with_instruments(records)
                for label, records in test_records.items()
            },
            "strongest_baseline": strongest_baseline,
            "bootstrap": bootstrap,
            "sensitivity_without_3pct_opening_gaps": sensitivity_metrics,
        }
        _write_json(self.run_dir / "metrics.json", metrics_payload)

        plot_metric_comparison(test_records, self.run_dir / "test_metrics.png")
        plot_training_histories(
            {label: result.history for label, result in all_results.items()},
            self.run_dir / "training_curves.png",
        )
        plot_forecast_examples(test_ensemble, self.run_dir / "forecast_examples.png")

        selected_summaries = [
            {
                "seed": seed,
                "learning_rate": selected_learning_rate,
                "best_epoch": result.best_epoch,
                "best_score": result.best_score,
                "best_return_mae": result.best_return_mae,
                "elapsed_seconds": result.elapsed_seconds,
            }
            for seed, result in sorted(selected_results.items())
        ]
        report_path = self.run_dir / "REPORT.md"
        normalization_consistency = json.loads(
            consistency_path.read_text(encoding="utf-8")
        )
        write_report(
            report_path,
            audits=self.audits,
            split_summary=self.split_summary,
            selected_learning_rate=selected_learning_rate,
            training_summaries=selected_summaries,
            validation_records=validation_records,
            test_records=test_records,
            strongest_baseline=strongest_baseline,
            bootstrap=bootstrap,
            sensitivity_metrics=sensitivity_metrics,
            normalization_consistency=normalization_consistency,
        )

        deliverable_dir = Path(__file__).resolve().parent / "results" / "futures_hourly"
        deliverable_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "REPORT.md",
            "metrics.json",
            "data_audit.json",
            "split_summary.json",
            "test_metrics.png",
            "training_curves.png",
            "forecast_examples.png",
        ):
            shutil.copy2(self.run_dir / filename, deliverable_dir / filename)
        print(f"full experiment complete: {report_path}")
        return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kronos hourly futures experiment")
    parser.add_argument(
        "stage",
        choices=("audit", "download", "smoke", "baseline", "consistency", "full"),
        help="Experiment stage to execute",
    )
    parser.add_argument(
        "--config",
        default="csj/configs/futures_hourly.yaml",
        help="Path to the YAML experiment config",
    )
    parser.add_argument(
        "--run-id",
        default="full",
        help="Subdirectory under output.root; completed training runs are resumed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = FuturesExperiment(args.config, args.run_id)
    if args.stage == "audit":
        experiment.audit()
    elif args.stage == "download":
        experiment.download()
    elif args.stage == "smoke":
        experiment.smoke()
    elif args.stage == "baseline":
        experiment.baseline()
    elif args.stage == "consistency":
        experiment.consistency()
    else:
        experiment.full()


if __name__ == "__main__":
    main()
