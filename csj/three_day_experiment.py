from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from csj.config import load_config
from csj.evaluation import resolve_device
from csj.futures_data import (
    CasePeriod,
    DenseInstrumentWindowDataset,
    ThreeTradingDayCase,
    build_expanding_walk_forward_folds,
    build_three_trading_day_cases,
    chronological_split,
    common_trading_days,
    fit_context_normalization,
    load_contracts,
    split_case_period,
)
from csj.metrics import three_day_metrics_with_instruments
from csj.three_day_evaluation import (
    make_three_day_baselines,
    predict_three_day_cases,
)
from csj.three_day_reporting import (
    plot_three_day_return_examples,
    write_phase1_report,
)
from csj.three_day_training import (
    load_ce_only_checkpoint,
    load_ce_only_training_result,
    train_ce_only_predictor,
)
from csj.utils.tool import MODEL_FEATURES
from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import auto_regressive_inference


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
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _save_records(records: pd.DataFrame, path: Path) -> None:
    _write_json(path, records.to_dict(orient="records"))


def _load_records(path: Path) -> pd.DataFrame:
    records = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    for column in ("target_day", "origin_timestamp", "origin_trading_day"):
        if column in records:
            records[column] = pd.to_datetime(records[column])
    return records


def _case_summary(cases: Sequence[ThreeTradingDayCase]) -> dict[str, Any]:
    by_instrument = Counter(case.instrument for case in cases)
    by_pred_len = Counter(str(case.pred_len) for case in cases)
    by_bar_pattern = Counter(
        "-".join(
            str(
                int(
                    (
                        case.target["trading_day"]
                        == case.target_days[day_number]
                    ).sum()
                )
            )
            for day_number in range(3)
        )
        for case in cases
    )
    return {
        "total": len(cases),
        "by_instrument": dict(sorted(by_instrument.items())),
        "by_pred_len": dict(sorted(by_pred_len.items())),
        "by_bar_pattern": dict(sorted(by_bar_pattern.items())),
        "first_target_day": min(
            (case.target_days[0] for case in cases), default=None
        ),
        "last_target_day": max(
            (case.target_days[-1] for case in cases), default=None
        ),
    }


def _assert_case_integrity(
    cases: Sequence[ThreeTradingDayCase],
    period: CasePeriod,
    *,
    lookback: int,
) -> None:
    for case in cases:
        if len(case.context) != lookback:
            raise RuntimeError("Three-day case has an invalid context length")
        if case.context["timestamps"].max() >= case.target["timestamps"].min():
            raise RuntimeError("Three-day case has context/target time leakage")
        if not all(period.contains(day) for day in case.target_days):
            raise RuntimeError("Three-day target crosses its split or fold")
        if set(case.context["instrument"].astype(str)) != {case.instrument}:
            raise RuntimeError("Three-day context crosses instruments")
        if set(case.target["instrument"].astype(str)) != {case.instrument}:
            raise RuntimeError("Three-day target crosses instruments")
        if case.day_end_indices[-1] != case.pred_len - 1:
            raise RuntimeError("Three-day endpoint indices are inconsistent")


def _clipping_audit(
    cases: Sequence[ThreeTradingDayCase],
    *,
    clip: float,
    epsilon: float,
) -> dict[str, Any]:
    counts: dict[str, dict[str, dict[str, list[int]]]] = {}
    for case in cases:
        instrument_counts = counts.setdefault(case.instrument, {})
        stats = fit_context_normalization(
            case.context,
            clip=clip,
            epsilon=epsilon,
        )
        for segment, frame in (("context", case.context), ("target", case.target)):
            segment_counts = instrument_counts.setdefault(
                segment,
                {feature: [0, 0] for feature in MODEL_FEATURES},
            )
            values = frame[MODEL_FEATURES].to_numpy(dtype=np.float64)
            clipped = stats.clipping_mask(values)
            for feature_index, feature in enumerate(MODEL_FEATURES):
                segment_counts[feature][0] += int(clipped[:, feature_index].sum())
                segment_counts[feature][1] += int(len(values))

    output: dict[str, Any] = {}
    for instrument, instrument_counts in sorted(counts.items()):
        output[instrument] = {}
        for segment, segment_counts in sorted(instrument_counts.items()):
            output[instrument][segment] = {
                feature: {
                    "clipped_values": clipped_values,
                    "total_values": total_values,
                    "rate": clipped_values / total_values if total_values else None,
                }
                for feature, (clipped_values, total_values) in segment_counts.items()
            }
    return output


class ThreeDayExperiment:
    def __init__(
        self,
        config_path: str | Path,
        run_id: str,
        *,
        device_override: str | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.config = load_config(config_path)
        if device_override is not None:
            self.config["training"]["device"] = device_override
            self.config["evaluation"]["device"] = device_override
        self.run_dir = Path(self.config["output"]["root"]) / run_id
        self.results_dir = Path(self.config["output"]["results_root"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(str(self.config["training"]["device"]))
        self.evaluation_device = resolve_device(
            str(self.config["evaluation"].get("device", "cpu"))
        )
        self.local_files_only = not allow_model_download
        self.config["runtime"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "requested_device": device_override or self.config["training"]["device"],
            "training_device": str(self.device),
            "evaluation_device": str(self.evaluation_device),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "model_download_allowed": allow_model_download,
        }

        data = self.config["data"]
        self.lookback = int(data["lookback"])
        self.frames, self.contract_audits = load_contracts(
            data["paths"], valid_bar_counts=data["valid_bar_counts"]
        )
        self.boundaries = chronological_split(
            self.frames,
            train_ratio=float(data["train_ratio"]),
            val_ratio=float(data["val_ratio"]),
            test_ratio=float(data["test_ratio"]),
        )
        self.split_periods = {
            split: split_case_period(self.boundaries, split)
            for split in ("train", "val", "test")
        }
        self.split_cases = {
            split: build_three_trading_day_cases(
                self.frames, period, lookback=self.lookback
            )
            for split, period in self.split_periods.items()
        }

        walk = self.config["walk_forward"]
        self.folds = build_expanding_walk_forward_folds(
            self.frames,
            minimum_train_days=int(walk["minimum_train_days"]),
            evaluation_days=int(walk["evaluation_days"]),
            step_days=int(walk["step_days"]),
            inner_validation_days=int(walk["inner_validation_days"]),
        )
        _write_json(self.run_dir / "resolved_config.json", self.config)

    def audit(self) -> tuple[Path, Path]:
        for split, cases in self.split_cases.items():
            _assert_case_integrity(
                cases,
                self.split_periods[split],
                lookback=self.lookback,
            )

        fold_rows: list[dict[str, Any]] = []
        for fold in self.folds:
            period_summaries: dict[str, Any] = {}
            for label, period in (
                ("fit", fold.fit_period),
                ("inner_validation", fold.inner_validation_period),
                ("evaluation", fold.evaluation_period),
            ):
                cases = build_three_trading_day_cases(
                    self.frames, period, lookback=self.lookback
                )
                _assert_case_integrity(cases, period, lookback=self.lookback)
                period_summaries[label] = {
                    "start_day": period.start_day,
                    "end_day": period.end_day,
                    "cases": _case_summary(cases),
                }
            fold_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "available_train_start": fold.train_period.start_day,
                    "available_train_end": fold.train_period.end_day,
                    **period_summaries,
                }
            )

        data = self.config["data"]
        all_cases = [case for cases in self.split_cases.values() for case in cases]
        data_audit = {
            "task": "three_complete_trading_days",
            "lookback": self.lookback,
            "allowed_pred_lengths": data["possible_pred_lengths"],
            "contract_audit": self.contract_audits,
            "chronological_boundaries": {
                "first_day": self.boundaries.first_day,
                "train_end": self.boundaries.train_end,
                "val_end": self.boundaries.val_end,
                "last_day": self.boundaries.last_day,
            },
            "split_cases": {
                split: _case_summary(cases)
                for split, cases in self.split_cases.items()
            },
            "integrity": {
                "context_length": self.lookback,
                "all_contexts_exact": all(
                    len(case.context) == self.lookback for case in all_cases
                ),
                "all_contexts_strictly_before_targets": all(
                    case.context["timestamps"].max()
                    < case.target["timestamps"].min()
                    for case in all_cases
                ),
                "all_targets_have_three_days": all(
                    len(case.target_days) == 3 for case in all_cases
                ),
                "all_pred_lengths_allowed": all(
                    case.pred_len in data["possible_pred_lengths"]
                    for case in all_cases
                ),
                "instrument_leakage_cases": 0,
                "split_or_fold_leakage_cases": 0,
            },
            "clipping": _clipping_audit(
                all_cases,
                clip=float(data["clip"]),
                epsilon=float(data["normalization_epsilon"]),
            ),
        }
        shared_days = common_trading_days(self.frames)
        fold_summary = {
            "common_trading_days": len(shared_days),
            "first_common_day": shared_days[0],
            "last_common_day": shared_days[-1],
            "protocol": self.config["walk_forward"],
            "complete_folds": len(self.folds),
            "folds": fold_rows,
        }

        data_audit_path = self.results_dir / "data_audit.json"
        fold_summary_path = self.results_dir / "fold_summary.json"
        for destination in (data_audit_path, self.run_dir / "data_audit.json"):
            _write_json(destination, data_audit)
        for destination in (fold_summary_path, self.run_dir / "fold_summary.json"):
            _write_json(destination, fold_summary)
        print(json.dumps(_json_safe(data_audit["split_cases"]), ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "complete_folds": len(self.folds),
                    "results_dir": str(self.results_dir),
                    "device": str(self.device),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return data_audit_path, fold_summary_path

    def _load_tokenizer(self) -> KronosTokenizer:
        model = self.config["model"]
        tokenizer = KronosTokenizer.from_pretrained(
            model["tokenizer_id"],
            revision=model["tokenizer_revision"],
            cache_dir=model["cache_dir"],
            local_files_only=self.local_files_only,
        )
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        return tokenizer

    def _load_predictor(self) -> Kronos:
        model = self.config["model"]
        return Kronos.from_pretrained(
            model["predictor_id"],
            revision=model["predictor_revision"],
            cache_dir=model["cache_dir"],
            local_files_only=self.local_files_only,
        )

    def consistency(self) -> Path:
        case = next(
            (
                candidate
                for split in ("val", "test", "train")
                for candidate in self.split_cases[split]
                if candidate.pred_len == 21
            ),
            None,
        )
        if case is None:
            raise RuntimeError("No 21-bar three-day case is available for consistency")

        tokenizer = self._load_tokenizer().to(self.device)
        model = self._load_predictor().to(self.device)
        tokenizer.eval()
        model.eval()
        model_config = self.config["model"]
        data = self.config["data"]
        predictor = KronosPredictor(
            model,
            tokenizer,
            device=str(self.device),
            max_context=int(model_config["max_context"]),
            clip=float(data["clip"]),
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

        stats = fit_context_normalization(
            case.context,
            clip=float(data["clip"]),
            epsilon=float(data["normalization_epsilon"]),
        )
        context_values = case.context[MODEL_FEATURES].to_numpy(dtype=np.float64)
        model_input = stats.transform(context_values).astype(np.float32)
        x = torch.from_numpy(model_input[None, ...]).to(self.device)
        x_stamp = torch.from_numpy(
            case.context[["minute", "hour", "weekday", "day", "month"]]
            .to_numpy(dtype=np.float32)[None, ...]
        ).to(self.device)
        y_stamp = torch.from_numpy(
            case.target[["minute", "hour", "weekday", "day", "month"]]
            .to_numpy(dtype=np.float32)[None, ...]
        ).to(self.device)
        normalized_output = auto_regressive_inference(
            tokenizer,
            model,
            x,
            x_stamp,
            y_stamp,
            max_context=int(model_config["max_context"]),
            pred_len=case.pred_len,
            clip=float(data["clip"]),
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
            return_samples=True,
        )[0, 0, -case.pred_len :, :]
        custom = stats.inverse(normalized_output)
        native_values = native[MODEL_FEATURES].to_numpy(dtype=np.float64)
        absolute = np.abs(native_values - custom)
        relative = absolute / (np.abs(native_values) + 1e-12)
        roundtrip = stats.inverse(
            stats.transform(context_values, apply_clip=False)
        )
        result = {
            "instrument": case.instrument,
            "target_days": case.target_days,
            "pred_len": case.pred_len,
            "day_end_indices": case.day_end_indices,
            "max_abs_diff": float(absolute.max()),
            "max_rel_diff": float(relative.max()),
            "normalization_roundtrip_max_abs": float(
                np.abs(roundtrip - context_values).max()
            ),
            "per_feature": {
                feature: {
                    "max_abs_diff": float(absolute[:, index].max()),
                    "max_rel_diff": float(relative[:, index].max()),
                }
                for index, feature in enumerate(MODEL_FEATURES)
            },
        }
        if result["max_rel_diff"] > 1e-6:
            raise RuntimeError(
                "V2 inverse normalization differs from native predictor by more than 1e-6"
            )

        output_path = self.run_dir / "normalization_consistency.json"
        _write_json(output_path, result)
        _write_json(self.results_dir / "normalization_consistency.json", result)
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))

        model.to("cpu")
        tokenizer.to("cpu")
        del predictor, model, tokenizer, x, x_stamp, y_stamp
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        return output_path

    def _select_smoke_cases(
        self,
        cases: Sequence[ThreeTradingDayCase],
    ) -> list[ThreeTradingDayCase]:
        selected: list[ThreeTradingDayCase] = []
        for instrument in sorted({case.instrument for case in cases}):
            instrument_cases = [
                case for case in cases if case.instrument == instrument
            ]
            for pred_len in sorted({case.pred_len for case in instrument_cases}):
                selected.append(
                    next(
                        case
                        for case in instrument_cases
                        if case.pred_len == pred_len
                    )
                )
        return sorted(
            selected,
            key=lambda case: (case.target_days[0], case.instrument, case.pred_len),
        )

    def phase1(self, *, smoke: bool = False, pilot: bool = False) -> Path:
        if smoke and pilot:
            raise ValueError("Phase 1 cannot be both smoke and pilot")
        stage_name = (
            f"phase1_smoke_{self.evaluation_device.type}"
            if smoke
            else (
                f"phase1_pilot_{self.evaluation_device.type}"
                if pilot
                else f"phase1_{self.evaluation_device.type}"
            )
        )
        stage_dir = self.run_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        evaluation = self.config["evaluation"]
        data = self.config["data"]
        sample_count = int(
            evaluation["smoke_sample_count"]
            if smoke
            else evaluation["sample_count"]
        )
        point_estimate = str(evaluation["path_point_estimate"])
        turning_threshold = float(evaluation["turning_point_return_threshold"])
        base_seed = int(evaluation["random_seed"])
        folds = self.folds[:1] if smoke or pilot else self.folds
        started_at = time.monotonic()

        tokenizer = self._load_tokenizer()
        model = self._load_predictor()
        model.eval()
        tokenizer.eval()
        records_by_model: dict[str, list[pd.DataFrame]] = {
            "zero_shot": [],
            "majority": [],
            "momentum": [],
            "persistence": [],
        }
        try:
            for fold_index, fold in enumerate(folds):
                evaluation_cases = build_three_trading_day_cases(
                    self.frames,
                    fold.evaluation_period,
                    lookback=self.lookback,
                )
                if smoke:
                    evaluation_cases = self._select_smoke_cases(evaluation_cases)
                training_cases = build_three_trading_day_cases(
                    self.frames,
                    fold.train_period,
                    lookback=self.lookback,
                )

                for instrument_index, instrument in enumerate(
                    sorted({case.instrument for case in evaluation_cases})
                ):
                    instrument_cases = [
                        case
                        for case in evaluation_cases
                        if case.instrument == instrument
                    ]
                    output_path = (
                        stage_dir
                        / "zero_shot"
                        / f"{fold.fold_id}_{instrument}.json"
                    )
                    cache_is_current = False
                    if output_path.exists():
                        records = _load_records(output_path)
                        cache_is_current = {
                            "raw_sample_paths",
                            "sampling_seed",
                            "inference_device",
                        }.issubset(records.columns)
                        cache_is_current = cache_is_current and bool(
                            (
                                records["inference_device"]
                                == str(self.evaluation_device)
                            ).all()
                        )
                    if cache_is_current:
                        print(f"resuming cached zero-shot records: {output_path}")
                    else:
                        if output_path.exists():
                            print(f"refreshing stale zero-shot cache: {output_path}")
                        estimated_batches = int(
                            np.ceil(
                                len(instrument_cases)
                                / int(evaluation["inference_batch_size"])
                            )
                        )
                        run_seed = base_seed + fold_index * 10 + instrument_index
                        print(
                            f"phase1 zero_shot instrument={instrument} "
                            f"fold={fold.fold_id} cases={len(instrument_cases)} "
                            f"samples={sample_count} pred_lengths="
                            f"{sorted({case.pred_len for case in instrument_cases})} "
                            f"batches={estimated_batches} seed={run_seed}"
                            f" device={self.evaluation_device}"
                        )
                        inference_started = time.monotonic()
                        records = predict_three_day_cases(
                            model,
                            tokenizer,
                            instrument_cases,
                            device=self.evaluation_device,
                            max_context=int(self.config["model"]["max_context"]),
                            clip=float(data["clip"]),
                            normalization_epsilon=float(
                                data["normalization_epsilon"]
                            ),
                            sample_count=sample_count,
                            temperature=float(evaluation["temperature"]),
                            top_k=int(evaluation["top_k"]),
                            top_p=float(evaluation["top_p"]),
                            batch_size=int(evaluation["inference_batch_size"]),
                            seed=run_seed,
                            point_estimate=point_estimate,
                            turning_point_threshold=turning_threshold,
                        )
                        _save_records(records, output_path)
                        print(
                            f"phase1 zero_shot completed instrument={instrument} "
                            f"fold={fold.fold_id} seconds="
                            f"{time.monotonic() - inference_started:.2f}"
                        )
                    records_by_model["zero_shot"].append(records)

                baselines = make_three_day_baselines(
                    training_cases,
                    evaluation_cases,
                    point_estimate=point_estimate,
                    turning_point_threshold=turning_threshold,
                )
                for model_name, records in baselines.items():
                    output_path = (
                        stage_dir
                        / "baselines"
                        / f"{fold.fold_id}_{model_name}.json"
                    )
                    _save_records(records, output_path)
                    records_by_model[model_name].append(records)
        finally:
            model.to("cpu")
            tokenizer.to("cpu")
            del model, tokenizer
            gc.collect()
            if self.evaluation_device.type == "mps":
                torch.mps.empty_cache()
            elif self.evaluation_device.type == "cuda":
                torch.cuda.empty_cache()

        combined = {
            model_name: pd.concat(record_sets, ignore_index=True).sort_values(
                ["target_day", "instrument"], kind="stable"
            ).reset_index(drop=True)
            for model_name, record_sets in records_by_model.items()
        }
        overall_metrics = {
            model_name: three_day_metrics_with_instruments(records)
            for model_name, records in combined.items()
        }
        by_fold_metrics = {
            fold.fold_id: {
                model_name: three_day_metrics_with_instruments(
                    records.loc[records["fold_id"] == fold.fold_id].copy()
                )
                for model_name, records in combined.items()
            }
            for fold in folds
        }
        elapsed_seconds = time.monotonic() - started_at
        payload = {
            "stage": stage_name,
            "historical_evidence": "retrospective expanding walk-forward development",
            "sample_count": sample_count,
            "point_estimate": point_estimate,
            "turning_point_return_threshold": turning_threshold,
            "inference_device": str(self.evaluation_device),
            "record_counts": {
                model_name: len(records)
                for model_name, records in combined.items()
            },
            "elapsed_seconds": elapsed_seconds,
            "overall": overall_metrics,
            "by_fold": by_fold_metrics,
        }
        metrics_path = self.results_dir / f"{stage_name}_metrics.json"
        _write_json(metrics_path, payload)
        _write_json(stage_dir / "metrics.json", payload)

        figure_path = self.results_dir / f"{stage_name}_path_examples.png"
        plot_three_day_return_examples(combined["zero_shot"], figure_path)
        report_label = "SMOKE_" if smoke else ("PILOT_" if pilot else "")
        report_path = self.results_dir / (
            f"PHASE1_{report_label}{self.evaluation_device.type.upper()}_REPORT.md"
        )
        write_phase1_report(
            overall_metrics,
            by_fold_metrics=by_fold_metrics,
            output_path=report_path,
            elapsed_seconds=elapsed_seconds,
            record_counts=payload["record_counts"],
            figure_name=figure_path.name,
            metrics_name=metrics_path.name,
            smoke=smoke,
        )
        print(
            json.dumps(
                {
                    "stage": stage_name,
                    "metrics": str(metrics_path),
                    "report": str(report_path),
                    "records": payload["record_counts"],
                    "elapsed_seconds": elapsed_seconds,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return report_path

    def phase2_smoke(self) -> Path:
        fold = self.folds[0]
        training = self.config["training"]
        evaluation = self.config["evaluation"]
        data = self.config["data"]
        model_config = self.config["model"]
        learning_rate = float(training["learning_rates"][0])
        seed = int(training["seeds"][0])
        stage_dir = self.run_dir / "phase2_smoke" / fold.fold_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = self._load_tokenizer()
        results: dict[str, Any] = {}
        try:
            for instrument in sorted(self.frames):
                dataset = DenseInstrumentWindowDataset(
                    self.frames[instrument],
                    fold.fit_period,
                    instrument=instrument,
                    lookback=self.lookback,
                    horizon=int(data["dense_horizon"]),
                    clip=float(data["clip"]),
                    epsilon=float(data["normalization_epsilon"]),
                )
                validation_cases = build_three_trading_day_cases(
                    {instrument: self.frames[instrument]},
                    fold.inner_validation_period,
                    lookback=self.lookback,
                )
                validation_cases = self._select_smoke_cases(validation_cases)
                output_dir = (
                    stage_dir
                    / instrument
                    / f"lr_{learning_rate:.0e}_seed_{seed}"
                )
                print(
                    f"phase2 smoke model_source={model_config['predictor_id']}@"
                    f"{model_config['predictor_revision']} instrument={instrument} "
                    f"fold={fold.fold_id} lr={learning_rate:.1e} seed={seed} "
                    f"lambda_dir=0 windows={len(dataset)} "
                    f"validation_cases={len(validation_cases)} max_batches=2 "
                    f"device={self.device}"
                )
                model = self._load_predictor()
                metadata = {
                    "phase": "phase2_ce_only_smoke",
                    "model_source": {
                        "predictor_id": model_config["predictor_id"],
                        "predictor_revision": model_config["predictor_revision"],
                        "tokenizer_id": model_config["tokenizer_id"],
                        "tokenizer_revision": model_config["tokenizer_revision"],
                    },
                    "instrument": instrument,
                    "fold_id": fold.fold_id,
                    "fit_start": fold.fit_period.start_day,
                    "fit_end": fold.fit_period.end_day,
                    "inner_validation_start": (
                        fold.inner_validation_period.start_day
                    ),
                    "inner_validation_end": fold.inner_validation_period.end_day,
                    "normalization": {
                        "context_only": True,
                        "lookback": self.lookback,
                        "clip": float(data["clip"]),
                        "epsilon": float(data["normalization_epsilon"]),
                    },
                    "dense_horizon": int(data["dense_horizon"]),
                    "dense_windows": len(dataset),
                    "lambda_dir": 0.0,
                    "stream_ratio": {
                        "dense_batches": 1,
                        "direction_batches": 0,
                    },
                    "smoke_max_batches": 2,
                }
                try:
                    result = train_ce_only_predictor(
                        model,
                        tokenizer,
                        dataset,
                        validation_cases,
                        device=self.device,
                        output_dir=output_dir,
                        learning_rate=learning_rate,
                        seed=seed,
                        max_epochs=1,
                        early_stopping_patience=1,
                        batch_size=int(training["batch_size"]),
                        num_workers=int(training["num_workers"]),
                        weight_decay=float(training["weight_decay"]),
                        gradient_clip=float(training["gradient_clip"]),
                        warmup_ratio=float(training["warmup_ratio"]),
                        evaluation_config={
                            **evaluation,
                            "sample_count": int(
                                evaluation["smoke_sample_count"]
                            ),
                            "max_context": int(model_config["max_context"]),
                            "clip": float(data["clip"]),
                            "normalization_epsilon": float(
                                data["normalization_epsilon"]
                            ),
                        },
                        checkpoint_metadata=metadata,
                        max_train_batches=2,
                    )
                    results[instrument] = {
                        "checkpoint": result.checkpoint_path,
                        "best_epoch": result.best_epoch,
                        "day3_balanced_accuracy": (
                            result.best_day3_balanced_accuracy
                        ),
                        "day3_return_mae": result.best_day3_return_mae,
                        "z_normalized_dtw": result.best_z_normalized_dtw,
                        "elapsed_seconds": result.elapsed_seconds,
                        "dense_windows": len(dataset),
                        "validation_cases": len(validation_cases),
                        "device": str(self.device),
                    }
                finally:
                    model.to("cpu")
                    del model
                    gc.collect()
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    elif self.device.type == "cuda":
                        torch.cuda.empty_cache()
        finally:
            tokenizer.to("cpu")
            del tokenizer
            gc.collect()

        output_path = self.results_dir / "phase2_smoke_summary.json"
        payload = {
            "stage": "phase2_ce_only_smoke",
            "fold_id": fold.fold_id,
            "learning_rate": learning_rate,
            "seed": seed,
            "lambda_dir": 0.0,
            "max_train_batches": 2,
            "results": results,
        }
        _write_json(output_path, payload)
        _write_json(self.run_dir / "phase2_smoke" / "summary.json", payload)
        print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
        return output_path

    def phase2(self, *, pilot: bool = False) -> Path:
        stage_name = (
            f"phase2_pilot_{self.evaluation_device.type}"
            if pilot
            else f"phase2_{self.evaluation_device.type}"
        )
        folds = self.folds[:1] if pilot else self.folds
        training = self.config["training"]
        evaluation = self.config["evaluation"]
        data = self.config["data"]
        model_config = self.config["model"]
        seed = int(training["seeds"][0])
        learning_rates = [float(value) for value in training["learning_rates"]]
        base_seed = int(evaluation["random_seed"])
        started_at = time.monotonic()
        tokenizer = self._load_tokenizer()
        selected_runs: dict[str, dict[str, Any]] = {}
        ce_records_by_fold: dict[str, list[pd.DataFrame]] = {}
        zero_records_by_fold: dict[str, list[pd.DataFrame]] = {}
        try:
            for fold_index, fold in enumerate(folds):
                selected_runs[fold.fold_id] = {}
                ce_records_by_fold[fold.fold_id] = []
                zero_records_by_fold[fold.fold_id] = []
                for instrument_index, instrument in enumerate(sorted(self.frames)):
                    dataset = DenseInstrumentWindowDataset(
                        self.frames[instrument],
                        fold.fit_period,
                        instrument=instrument,
                        lookback=self.lookback,
                        horizon=int(data["dense_horizon"]),
                        clip=float(data["clip"]),
                        epsilon=float(data["normalization_epsilon"]),
                    )
                    validation_cases = build_three_trading_day_cases(
                        {instrument: self.frames[instrument]},
                        fold.inner_validation_period,
                        lookback=self.lookback,
                    )
                    run_results = {}
                    validation_seed = (
                        base_seed + fold_index * 10 + instrument_index
                    )
                    for learning_rate in learning_rates:
                        output_dir = (
                            self.run_dir
                            / "phase2"
                            / "training"
                            / fold.fold_id
                            / instrument
                            / f"lr_{learning_rate:.0e}_seed_{seed}"
                        )
                        existing = load_ce_only_training_result(output_dir)
                        if existing is not None:
                            print(f"resuming completed CE-only run: {output_dir}")
                            run_results[learning_rate] = existing
                            continue
                        print(
                            f"phase2 model_source={model_config['predictor_id']}@"
                            f"{model_config['predictor_revision']} "
                            f"instrument={instrument} fold={fold.fold_id} "
                            f"lr={learning_rate:.1e} seed={seed} lambda_dir=0 "
                            f"windows={len(dataset)} "
                            f"validation_cases={len(validation_cases)} "
                            f"batches={int(np.ceil(len(dataset) / int(training['batch_size'])))} "
                            f"device={self.device}"
                        )
                        model = self._load_predictor()
                        metadata = {
                            "phase": "phase2_ce_only",
                            "model_source": {
                                "predictor_id": model_config["predictor_id"],
                                "predictor_revision": model_config[
                                    "predictor_revision"
                                ],
                                "tokenizer_id": model_config["tokenizer_id"],
                                "tokenizer_revision": model_config[
                                    "tokenizer_revision"
                                ],
                            },
                            "instrument": instrument,
                            "fold_id": fold.fold_id,
                            "fit_start": fold.fit_period.start_day,
                            "fit_end": fold.fit_period.end_day,
                            "inner_validation_start": (
                                fold.inner_validation_period.start_day
                            ),
                            "inner_validation_end": (
                                fold.inner_validation_period.end_day
                            ),
                            "evaluation_start": fold.evaluation_period.start_day,
                            "evaluation_end": fold.evaluation_period.end_day,
                            "normalization": {
                                "context_only": True,
                                "lookback": self.lookback,
                                "clip": float(data["clip"]),
                                "epsilon": float(data["normalization_epsilon"]),
                            },
                            "dense_horizon": int(data["dense_horizon"]),
                            "dense_windows": len(dataset),
                            "lambda_dir": 0.0,
                            "stream_ratio": {
                                "dense_batches": 1,
                                "direction_batches": 0,
                            },
                            "devices": {
                                "training": str(self.device),
                                "inner_validation": str(
                                    self.evaluation_device
                                ),
                                "evaluation": str(self.evaluation_device),
                            },
                            "selection_rule": [
                                "maximize_day3_path_balanced_accuracy",
                                "minimize_day3_return_mae",
                                "minimize_z_normalized_dtw",
                            ],
                        }
                        try:
                            run_results[learning_rate] = train_ce_only_predictor(
                                model,
                                tokenizer,
                                dataset,
                                validation_cases,
                                device=self.device,
                                output_dir=output_dir,
                                learning_rate=learning_rate,
                                seed=seed,
                                max_epochs=int(training["max_epochs"]),
                                early_stopping_patience=int(
                                    training["early_stopping_patience"]
                                ),
                                batch_size=int(training["batch_size"]),
                                num_workers=int(training["num_workers"]),
                                weight_decay=float(training["weight_decay"]),
                                gradient_clip=float(training["gradient_clip"]),
                                warmup_ratio=float(training["warmup_ratio"]),
                                evaluation_config={
                                    **evaluation,
                                    "random_seed": validation_seed,
                                    "max_context": int(
                                        model_config["max_context"]
                                    ),
                                    "clip": float(data["clip"]),
                                    "normalization_epsilon": float(
                                        data["normalization_epsilon"]
                                    ),
                                },
                                checkpoint_metadata=metadata,
                                validation_device=self.evaluation_device,
                            )
                        finally:
                            model.to("cpu")
                            del model
                            gc.collect()
                            if self.device.type == "mps":
                                torch.mps.empty_cache()
                            elif self.device.type == "cuda":
                                torch.cuda.empty_cache()

                    selected_lr, selected_result = max(
                        run_results.items(),
                        key=lambda item: (
                            item[1].best_day3_balanced_accuracy,
                            -item[1].best_day3_return_mae,
                            -item[1].best_z_normalized_dtw,
                        ),
                    )
                    selected_runs[fold.fold_id][instrument] = {
                        "learning_rate": selected_lr,
                        "seed": seed,
                        "best_epoch": selected_result.best_epoch,
                        "validation_day3_balanced_accuracy": (
                            selected_result.best_day3_balanced_accuracy
                        ),
                        "validation_day3_return_mae": (
                            selected_result.best_day3_return_mae
                        ),
                        "validation_z_normalized_dtw": (
                            selected_result.best_z_normalized_dtw
                        ),
                        "checkpoint": selected_result.checkpoint_path,
                        "candidate_runs": {
                            f"{candidate_lr:.0e}": {
                                "best_epoch": candidate_result.best_epoch,
                                "day3_balanced_accuracy": (
                                    candidate_result.best_day3_balanced_accuracy
                                ),
                                "day3_return_mae": (
                                    candidate_result.best_day3_return_mae
                                ),
                                "z_normalized_dtw": (
                                    candidate_result.best_z_normalized_dtw
                                ),
                                "elapsed_seconds": (
                                    candidate_result.elapsed_seconds
                                ),
                            }
                            for candidate_lr, candidate_result in run_results.items()
                        },
                    }

                    evaluation_cases = build_three_trading_day_cases(
                        {instrument: self.frames[instrument]},
                        fold.evaluation_period,
                        lookback=self.lookback,
                    )
                    evaluation_path = (
                        self.run_dir
                        / "phase2"
                        / "evaluation"
                        / f"{fold.fold_id}_{instrument}.json"
                    )
                    if evaluation_path.exists():
                        ce_records = _load_records(evaluation_path)
                        cache_is_current = {
                            "raw_sample_paths",
                            "sampling_seed",
                            "inference_device",
                        }.issubset(ce_records.columns)
                        cache_is_current = cache_is_current and bool(
                            (
                                ce_records["inference_device"]
                                == str(self.evaluation_device)
                            ).all()
                        )
                    else:
                        cache_is_current = False
                    if not cache_is_current:
                        model = self._load_predictor()
                        try:
                            load_ce_only_checkpoint(
                                model, selected_result.checkpoint_path
                            )
                            ce_records = predict_three_day_cases(
                                model,
                                tokenizer,
                                evaluation_cases,
                                device=self.evaluation_device,
                                max_context=int(model_config["max_context"]),
                                clip=float(data["clip"]),
                                normalization_epsilon=float(
                                    data["normalization_epsilon"]
                                ),
                                sample_count=int(evaluation["sample_count"]),
                                temperature=float(evaluation["temperature"]),
                                top_k=int(evaluation["top_k"]),
                                top_p=float(evaluation["top_p"]),
                                batch_size=int(
                                    evaluation["inference_batch_size"]
                                ),
                                seed=validation_seed,
                                point_estimate=str(
                                    evaluation["path_point_estimate"]
                                ),
                                turning_point_threshold=float(
                                    evaluation[
                                        "turning_point_return_threshold"
                                    ]
                                ),
                                model_name="ce_only",
                            )
                            _save_records(ce_records, evaluation_path)
                        finally:
                            model.to("cpu")
                            del model
                            gc.collect()
                            if self.device.type == "mps":
                                torch.mps.empty_cache()
                            elif self.device.type == "cuda":
                                torch.cuda.empty_cache()
                    ce_records_by_fold[fold.fold_id].append(ce_records)

                    phase1_stage = (
                        f"phase1_pilot_{self.evaluation_device.type}"
                        if pilot
                        else f"phase1_{self.evaluation_device.type}"
                    )
                    zero_path = (
                        self.run_dir
                        / phase1_stage
                        / "zero_shot"
                        / f"{fold.fold_id}_{instrument}.json"
                    )
                    if not zero_path.exists():
                        raise RuntimeError(
                            f"Phase 1 zero-shot cache is missing: {zero_path}"
                        )
                    zero_records_by_fold[fold.fold_id].append(
                        _load_records(zero_path)
                    )
        finally:
            tokenizer.to("cpu")
            del tokenizer
            gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

        ce_by_fold = {
            fold_id: pd.concat(records, ignore_index=True)
            for fold_id, records in ce_records_by_fold.items()
        }
        zero_by_fold = {
            fold_id: pd.concat(records, ignore_index=True)
            for fold_id, records in zero_records_by_fold.items()
        }
        combined_ce = pd.concat(ce_by_fold.values(), ignore_index=True)
        combined_zero = pd.concat(zero_by_fold.values(), ignore_index=True)
        metrics = {
            "ce_only": three_day_metrics_with_instruments(combined_ce),
            "zero_shot": three_day_metrics_with_instruments(combined_zero),
        }
        by_fold = {
            fold_id: {
                "ce_only": three_day_metrics_with_instruments(
                    ce_by_fold[fold_id]
                ),
                "zero_shot": three_day_metrics_with_instruments(
                    zero_by_fold[fold_id]
                ),
            }
            for fold_id in ce_by_fold
        }
        elapsed_seconds = time.monotonic() - started_at
        payload = {
            "stage": stage_name,
            "historical_evidence": (
                "retrospective expanding walk-forward development"
            ),
            "folds": [fold.fold_id for fold in folds],
            "seed": seed,
            "lambda_dir": 0.0,
            "devices": {
                "training": str(self.device),
                "inner_validation": str(self.evaluation_device),
                "evaluation": str(self.evaluation_device),
            },
            "learning_rate_candidates": learning_rates,
            "selection_rule": [
                "maximize_day3_path_balanced_accuracy",
                "minimize_day3_return_mae",
                "minimize_z_normalized_dtw",
            ],
            "selected_runs": selected_runs,
            "record_counts": {
                "ce_only": len(combined_ce),
                "zero_shot": len(combined_zero),
            },
            "elapsed_seconds": elapsed_seconds,
            "overall": metrics,
            "by_fold": by_fold,
        }
        metrics_path = self.results_dir / f"{stage_name}_metrics.json"
        _write_json(metrics_path, payload)
        _write_json(self.run_dir / "phase2" / f"{stage_name}_metrics.json", payload)
        figure_path = self.results_dir / f"{stage_name}_path_examples.png"
        plot_three_day_return_examples(combined_ce, figure_path)
        print(
            json.dumps(
                {
                    "stage": stage_name,
                    "metrics": str(metrics_path),
                    "records": payload["record_counts"],
                    "elapsed_seconds": elapsed_seconds,
                    "selected_runs": _json_safe(selected_runs),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos three-trading-day V2")
    parser.add_argument(
        "stage",
        choices=(
            "audit",
            "consistency",
            "phase0",
            "phase1_smoke",
            "phase1_pilot",
            "phase1",
            "phase2_smoke",
            "phase2_pilot",
            "phase2",
        ),
    )
    parser.add_argument(
        "--config",
        default="csj/configs/futures_3day_trend.yaml",
    )
    parser.add_argument("--run-id", default="phase0")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="Override both training and inference devices",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow pinned model revisions to be downloaded into cache",
    )
    args = parser.parse_args()

    experiment = ThreeDayExperiment(
        args.config,
        args.run_id,
        device_override=args.device,
        allow_model_download=args.allow_model_download,
    )
    if args.stage in ("audit", "phase0"):
        experiment.audit()
    if args.stage in ("consistency", "phase0"):
        experiment.consistency()
    if args.stage == "phase1_smoke":
        experiment.phase1(smoke=True)
    if args.stage == "phase1_pilot":
        experiment.phase1(pilot=True)
    if args.stage == "phase1":
        experiment.phase1(smoke=False)
    if args.stage == "phase2_smoke":
        experiment.phase2_smoke()
    if args.stage == "phase2_pilot":
        experiment.phase2(pilot=True)
    if args.stage == "phase2":
        experiment.phase2(pilot=False)


if __name__ == "__main__":
    main()
