"""Phase-gated orchestration for V5 target-only direction-guided paths.

Implemented stages intentionally stop after P1.  P0 creates the fixed K=32
zero-shot sample banks and chooses a direction baseline only on inner
validation.  P1 trains the frozen-backbone target-only direction probe and
writes its full gate.  P2/P3/P4 are guarded placeholders by design until the
preceding CUDA gate has been synchronized and reviewed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from csj.v5.config import PRODUCTION_ELIGIBLE, RESULT_SCOPE, load_v5_config
from csj.v5.path_bank import (
    PathBankError,
    context_3day_momentum_direction_records,
    fit_product_majority_direction_records,
    fit_product_up_probabilities,
    generate_sample_bank,
    read_sample_bank,
    rehydrate_path_records,
    selected_baseline,
    write_sample_bank,
    zero_shot_mean_direction_records,
    zero_shot_vote_direction_records,
)
from csj.v5.plotting import (
    V5FoldPlotArtifacts,
    V5PlotError,
    render_fold_baseline_path_plots,
)
from csj.v5.target_data import (
    DATA_RULE_VERSION,
    TargetOnlyCase,
    TargetOnlyObservedCohort,
    cases_in_period,
    build_target_only_audit,
    load_target_only_observed_cohort,
)
from csj.v5.target_probe import (
    TargetOnlyProbe,
    TargetOnlyProbeError,
    TargetProbeTrainingConfig,
    evaluate_target_probe,
    load_target_probe_head,
    probe_metrics,
    train_target_probe,
)
from model import Kronos, KronosTokenizer


class V5ExperimentError(RuntimeError):
    """A V5 stage cannot meet its frozen research protocol."""


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
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
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
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise V5ExperimentError("V5 requested CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported V5 device: {requested}")


def _direction_metrics(records: pd.DataFrame) -> dict[str, object]:
    if records.empty:
        return {
            "cases": 0,
            "valid_direction_cases": 0,
            "balanced_accuracy": None,
            "accuracy": None,
        }
    valid = records.loc[records["valid_direction"].astype(bool)]
    if valid.empty:
        return {
            "cases": int(len(records)),
            "valid_direction_cases": 0,
            "balanced_accuracy": None,
            "accuracy": None,
        }
    actual = valid["actual_direction"].to_numpy(dtype=np.int8)
    predicted = valid["predicted_direction"].to_numpy(dtype=np.int8)
    up = actual == 1
    down = actual == -1
    return {
        "cases": int(len(records)),
        "valid_direction_cases": int(len(valid)),
        "balanced_accuracy": (
            float(0.5 * (np.mean(predicted[up] == 1) + np.mean(predicted[down] == -1)))
            if up.any() and down.any()
            else None
        ),
        "accuracy": float(np.mean(actual == predicted)),
    }


def _canonicalize_target_end_day(records: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Give fresh and JSON-round-tripped records one trading-day representation.

    P0 persists its compact record index as JSON, so ``target_end_day`` comes
    back as an ISO string.  P1 evaluation produces ``pandas.Timestamp``
    values.  Both identify the same V5 trading day and must compare equal.
    Normalize only this declared day key; all other provenance fields remain
    subject to exact equality checks.
    """

    if "target_end_day" not in records:
        raise V5ExperimentError(f"{label} records miss target_end_day")
    normalized: list[pd.Timestamp] = []
    for raw_day in records["target_end_day"].tolist():
        try:
            day = pd.Timestamp(raw_day)
        except (TypeError, ValueError, OverflowError) as exc:
            raise V5ExperimentError(f"{label} has an invalid target_end_day: {raw_day!r}") from exc
        if pd.isna(day):
            raise V5ExperimentError(f"{label} has a null target_end_day")
        if day.tzinfo is not None:
            raise V5ExperimentError(f"{label} target_end_day must be timezone-naive: {raw_day!r}")
        normalized.append(day.normalize())
    output = records.copy()
    output["target_end_day"] = pd.DatetimeIndex(normalized)
    return output


def _canonicalize_direction_truth(records: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Use one checked integer representation for shared ground-truth fields."""

    output = records.copy()
    for column, allowed in (("actual_label", {0, 1}), ("actual_direction", {-1, 0, 1})):
        if column not in output:
            raise V5ExperimentError(f"{label} records miss {column}")
        numeric = pd.to_numeric(output[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise V5ExperimentError(f"{label} has non-finite {column}")
        rounded = np.rint(numeric.to_numpy(dtype=np.float64))
        if not np.array_equal(numeric.to_numpy(dtype=np.float64), rounded):
            raise V5ExperimentError(f"{label} has non-integral {column}")
        values = rounded.astype(np.int8)
        if not set(values).issubset(allowed):
            raise V5ExperimentError(f"{label} has invalid {column} values")
        output[column] = values
    return output


def _assert_same_direction_records(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    label: str,
) -> None:
    required = {
        "case_key",
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "actual_label",
        "actual_direction",
        "valid_direction",
        "probability_up",
        "predicted_label",
        "predicted_direction",
    }
    for side, records in (("candidate", candidate), ("baseline", baseline)):
        missing = sorted(required.difference(records.columns))
        if missing:
            raise V5ExperimentError(f"{label}/{side} records miss columns: {missing!r}")
        if records["case_key"].duplicated().any():
            raise V5ExperimentError(f"{label}/{side} records contain duplicate case keys")
    candidate = _canonicalize_target_end_day(candidate, label=f"{label}/candidate")
    baseline = _canonicalize_target_end_day(baseline, label=f"{label}/baseline")
    candidate = _canonicalize_direction_truth(candidate, label=f"{label}/candidate")
    baseline = _canonicalize_direction_truth(baseline, label=f"{label}/baseline")
    candidate_indexed = candidate.set_index("case_key", verify_integrity=True).sort_index()
    baseline_indexed = baseline.set_index("case_key", verify_integrity=True).sort_index()
    if set(candidate_indexed.index) != set(baseline_indexed.index):
        raise V5ExperimentError(f"{label} arms do not use the same case keys")
    baseline_indexed = baseline_indexed.loc[candidate_indexed.index]
    for column in (
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "actual_label",
        "actual_direction",
        "valid_direction",
    ):
        if not candidate_indexed[column].equals(baseline_indexed[column]):
            raise V5ExperimentError(f"{label} arms disagree on {column}")


def _attach_fold(records: pd.DataFrame, *, fold_id: str, model_name: str, seed: int | str | None = None) -> pd.DataFrame:
    output = records.copy()
    output["fold_id"] = str(fold_id)
    output["model"] = str(model_name)
    if seed is not None:
        output["seed"] = seed
    invalid = ~output["valid_direction"].astype(bool)
    output.loc[invalid, "actual_direction"] = 0
    output.loc[invalid, "predicted_direction"] = 0
    return output.sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _path_records_with_fold(records: pd.DataFrame, *, fold_id: str, model_name: str) -> pd.DataFrame:
    output = records.copy()
    output["fold_id"] = str(fold_id)
    output["model"] = str(model_name)
    output["actual_direction"] = output["day3_actual_direction"].astype(np.int8)
    output["predicted_direction"] = output["day3_predicted_direction"].astype(np.int8)
    output["valid_direction"] = output["actual_direction"] != 0
    output["actual_label"] = (output["actual_direction"] == 1).astype(np.int8)
    output["predicted_label"] = (output["predicted_direction"] == 1).astype(np.int8)
    output["probability_up"] = output["sample_vote_probability_up"].astype(float)
    output.loc[~output["valid_direction"], "predicted_direction"] = 0
    return output.sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _path_metric_summary(records: pd.DataFrame) -> dict[str, object]:
    valid = records.loc[records["day3_actual_direction"].astype(np.int8) != 0]
    actual = valid["day3_actual_direction"].to_numpy(dtype=np.int8)
    predicted = valid["day3_predicted_direction"].to_numpy(dtype=np.int8)
    up = actual == 1
    down = actual == -1
    correlations = records["path_return_correlation"].to_numpy(dtype=np.float64)
    dtw = records["z_normalized_dtw"].to_numpy(dtype=np.float64)
    return {
        "samples": int(len(records)),
        "direction_samples": int(len(valid)),
        "day3_path_balanced_accuracy": (
            float(0.5 * (np.mean(predicted[up] == 1) + np.mean(predicted[down] == -1)))
            if up.any() and down.any()
            else None
        ),
        "day3_return_mae": float(
            np.mean(
                np.abs(
                    records["day3_predicted_return"].to_numpy(dtype=np.float64)
                    - records["day3_actual_return"].to_numpy(dtype=np.float64)
                )
            )
        ),
        "mean_return_path_correlation": float(np.nanmean(correlations)),
        "mean_z_normalized_dtw": float(np.nanmean(dtw)),
        "by_product": {
            str(product): _path_metric_summary(group)
            for product, group in records.groupby("product", sort=True)
        }
        if len(records["product"].unique()) > 1
        else {},
    }


def _probe_seed_ensemble(records_by_seed: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    expected = (42, 43, 44)
    if tuple(sorted(records_by_seed)) != expected:
        raise V5ExperimentError(f"V5 P1 ensemble requires seeds {expected!r}")
    ordered = [
        (
            seed,
            _canonicalize_direction_truth(
                _canonicalize_target_end_day(
                    records_by_seed[seed], label=f"V5 P1 seed {seed}"
                ),
                label=f"V5 P1 seed {seed}",
            ),
        )
        for seed in expected
    ]
    _, first = ordered[0]
    first_indexed = first.set_index("case_key", verify_integrity=True).sort_index()
    probabilities = [first_indexed["probability_up"].to_numpy(dtype=np.float64)]
    for seed, records in ordered[1:]:
        indexed = records.set_index("case_key", verify_integrity=True).sort_index()
        if set(indexed.index) != set(first_indexed.index):
            raise V5ExperimentError(f"V5 P1 seed {seed} does not use matching case keys")
        indexed = indexed.loc[first_indexed.index]
        for column in (
            "fold_id",
            "product",
            "target_end_day",
            "target_contract_id",
            "actual_label",
            "actual_direction",
            "valid_direction",
        ):
            if not first_indexed[column].equals(indexed[column]):
                raise V5ExperimentError(f"V5 P1 seeds disagree on {column}")
        probabilities.append(indexed["probability_up"].to_numpy(dtype=np.float64))
    output = first_indexed.reset_index().copy()
    output["probability_up"] = np.vstack(probabilities).mean(axis=0)
    output["predicted_label"] = (output["probability_up"] >= 0.5).astype(np.int8)
    output["predicted_direction"] = np.where(output["predicted_label"] == 1, 1, -1)
    output.loc[~output["valid_direction"].astype(bool), "predicted_direction"] = 0
    output["seed"] = "ensemble"
    output["model"] = "target_only_probe_ensemble"
    return output.sort_values(
        ["fold_id", "target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _balanced_accuracy_difference(candidate: pd.DataFrame, baseline: pd.DataFrame) -> float | None:
    _assert_same_direction_records(candidate, baseline, label="V5 P1 balanced accuracy")
    candidate_ba = _direction_metrics(candidate)["balanced_accuracy"]
    baseline_ba = _direction_metrics(baseline)["balanced_accuracy"]
    if candidate_ba is None or baseline_ba is None:
        return None
    return float(candidate_ba) - float(baseline_ba)


def _paired_block_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    block_days: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Moving-day-block bootstrap of probe-minus-selected-baseline BA."""

    _assert_same_direction_records(candidate, baseline, label="V5 P1 bootstrap")
    candidate_valid = candidate.loc[candidate["valid_direction"].astype(bool)].set_index(
        "case_key", verify_integrity=True
    )
    baseline_valid = baseline.loc[baseline["valid_direction"].astype(bool)].set_index(
        "case_key", verify_integrity=True
    ).loc[candidate_valid.index]
    days = sorted(pd.to_datetime(candidate_valid["target_end_day"]).dt.normalize().unique())
    if block_days < 1 or block_days > len(days):
        return {"available": False, "reason": f"only {len(days)} evaluation days for {block_days}-day blocks"}
    candidate_by_day = {
        day: candidate_valid.loc[pd.to_datetime(candidate_valid["target_end_day"]).dt.normalize() == day]
        for day in days
    }
    baseline_by_day = {
        day: baseline_valid.loc[pd.to_datetime(baseline_valid["target_end_day"]).dt.normalize() == day]
        for day in days
    }
    rng = np.random.default_rng(seed)
    blocks_needed = int(math.ceil(len(days) / block_days))
    improvements = np.full(iterations, np.nan, dtype=np.float64)
    for iteration in range(iterations):
        sampled_days: list[pd.Timestamp] = []
        for start in rng.integers(0, len(days), size=blocks_needed):
            sampled_days.extend(days[(int(start) + offset) % len(days)] for offset in range(block_days))
        sampled_days = sampled_days[: len(days)]
        actual: list[int] = []
        candidate_predicted: list[int] = []
        baseline_predicted: list[int] = []
        for day in sampled_days:
            candidate_day = candidate_by_day[day]
            baseline_day = baseline_by_day[day].loc[candidate_day.index]
            actual.extend(candidate_day["actual_direction"].astype(int).tolist())
            candidate_predicted.extend(candidate_day["predicted_direction"].astype(int).tolist())
            baseline_predicted.extend(baseline_day["predicted_direction"].astype(int).tolist())
        actual_values = np.asarray(actual, dtype=np.int8)
        candidate_values = np.asarray(candidate_predicted, dtype=np.int8)
        baseline_values = np.asarray(baseline_predicted, dtype=np.int8)
        up = actual_values == 1
        down = actual_values == -1
        if up.any() and down.any():
            candidate_ba = 0.5 * (
                np.mean(candidate_values[up] == 1) + np.mean(candidate_values[down] == -1)
            )
            baseline_ba = 0.5 * (
                np.mean(baseline_values[up] == 1) + np.mean(baseline_values[down] == -1)
            )
            improvements[iteration] = candidate_ba - baseline_ba
    finite = improvements[np.isfinite(improvements)]
    point = _balanced_accuracy_difference(candidate, baseline)
    return {
        "available": bool(len(finite)),
        "block_days": int(block_days),
        "iterations": int(iterations),
        "point_estimate": point,
        "probability_improvement_positive": float(np.mean(finite > 0.0)) if len(finite) else None,
        "interval_5pct": float(np.quantile(finite, 0.05)) if len(finite) else None,
        "interval_95pct": float(np.quantile(finite, 0.95)) if len(finite) else None,
    }


class V5Experiment:
    """V5 implementation through P1, with later phases persisted-gate guarded."""

    def __init__(
        self,
        config_path: str | Path,
        run_id: str,
        *,
        device_override: str | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.config = load_v5_config(config_path)
        if device_override is not None:
            self.config["runtime"]["device"] = device_override
        self.run_id = str(run_id)
        self.run_dir = Path(self.config["output"]["root"]) / self.run_id
        self.results_dir = Path(self.config["output"]["results_root"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.local_files_only = not allow_model_download
        self._device: torch.device | None = None
        self.cohort: TargetOnlyObservedCohort = load_target_only_observed_cohort(
            self.config["data"]["snapshot_root"]
        )
        walk = self.config["walk_forward"]
        self.audit_payload, self.bundle, self.folds = build_target_only_audit(
            self.cohort,
            lookback=int(self.config["data"]["lookback"]),
            products=tuple(str(value) for value in self.config["data"]["products"]),
            minimum_fit_days=int(walk["minimum_fit_days"]),
            inner_validation_days=int(walk["inner_validation_days"]),
            evaluation_days=int(walk["evaluation_days"]),
            step_days=int(walk["step_days"]),
            purge_days=int(walk["purge_days"]),
            model_provenance=self._model_provenance(),
            data_provenance={
                "lookback": int(self.config["data"]["lookback"]),
                "clip": float(self.config["data"]["clip"]),
                "normalization_epsilon": float(self.config["data"]["normalization_epsilon"]),
            },
            include_v4_pair_only_comparison=False,
        )
        self.lookback = int(self.config["data"]["lookback"])
        self.config["runtime_resolved"] = {
            "device_requested": str(self.config["runtime"]["device"]),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "model_download_allowed": allow_model_download,
            "primary_context_length": self.lookback,
            "data_fingerprint": self.cohort.data_fingerprint,
        }
        _write_json(self.run_dir / "resolved_config.json", self.config)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = resolve_device(str(self.config["runtime"]["device"]))
            self.config["runtime_resolved"].update(
                {
                    "device": str(self._device),
                    "cuda_device": torch.cuda.get_device_name(self._device)
                    if self._device.type == "cuda"
                    else None,
                }
            )
            _write_json(self.run_dir / "resolved_config.json", self.config)
        return self._device

    @property
    def primary_products(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config["data"]["primary_selection_products"])

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
        }

    def _metadata(self, phase: str, *, upstream_gate_path: Path | None = None) -> dict[str, object]:
        upstream: dict[str, object] = {}
        if upstream_gate_path is not None:
            if not upstream_gate_path.is_file():
                raise V5ExperimentError(f"Required upstream V5 gate is missing: {upstream_gate_path}")
            upstream = {
                "upstream_gate_path": str(upstream_gate_path),
                "upstream_gate_sha256": _sha256_path(upstream_gate_path),
            }
        return {
            "strategy_version": 5,
            "phase": phase,
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": PRODUCTION_ELIGIBLE,
            "data_rule_version": DATA_RULE_VERSION,
            "data_fingerprint": self.cohort.data_fingerprint,
            "evaluation_contract_version": "v5-price-path-plot-v1",
            **self._model_provenance(),
            **upstream,
        }

    def _require_folds(self) -> tuple[Any, ...]:
        if not self.folds:
            raise V5ExperimentError("V5 target-only audit has no complete walk-forward folds")
        return self.folds

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

    def _release(self, *models: torch.nn.Module) -> None:
        for model in models:
            model.to("cpu")
        gc.collect()
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def audit(self) -> Path:
        full_audit, _, _ = build_target_only_audit(
            self.cohort,
            lookback=self.lookback,
            products=tuple(str(value) for value in self.config["data"]["products"]),
            minimum_fit_days=int(self.config["walk_forward"]["minimum_fit_days"]),
            inner_validation_days=int(self.config["walk_forward"]["inner_validation_days"]),
            evaluation_days=int(self.config["walk_forward"]["evaluation_days"]),
            step_days=int(self.config["walk_forward"]["step_days"]),
            purge_days=int(self.config["walk_forward"]["purge_days"]),
            model_provenance=self._model_provenance(),
            data_provenance={
                "lookback": self.lookback,
                "clip": float(self.config["data"]["clip"]),
                "normalization_epsilon": float(self.config["data"]["normalization_epsilon"]),
            },
            include_v4_pair_only_comparison=True,
        )
        payload = {**self._metadata("audit"), **full_audit}
        result_path = self.results_dir / "v5_data_audit.json"
        _write_json(result_path, payload)
        _write_json(self.run_dir / "data_audit.json", payload)
        print(json.dumps(_json_safe({"data_audit": result_path, "lookback": self.lookback})))
        return result_path

    def _p0_paths(self, *, stage_name: str, fold_id: str, split: str) -> tuple[Path, Path]:
        root = self.run_dir / stage_name / fold_id / split
        return root / "sample_bank.npz", root / "records.json"

    def _p0_generation_arguments(self) -> dict[str, object]:
        p0 = self.config["p0"]
        data = self.config["data"]
        return {
            "device": self.device,
            "max_context": int(self.config["model"]["max_context"]),
            "clip": float(data["clip"]),
            "epsilon": float(data["normalization_epsilon"]),
            "sample_count": int(p0["sample_count"]),
            "temperature": float(p0["temperature"]),
            "top_k": int(p0["top_k"]),
            "top_p": float(p0["top_p"]),
            "batch_size": int(p0["inference_batch_size"]),
            "random_seed": int(p0["random_seed"]),
        }

    def _create_p0_sample_bank(
        self,
        *,
        stage_name: str,
        fold_id: str,
        split: str,
        cases: Sequence[TargetOnlyCase],
    ) -> pd.DataFrame:
        bank_path, records_path = self._p0_paths(
            stage_name=stage_name, fold_id=fold_id, split=split
        )
        if bank_path.is_file() or records_path.is_file():
            # A partial/bad cache should not be silently trusted or overwritten.
            if not (bank_path.is_file() and records_path.is_file()):
                raise V5ExperimentError(f"Incomplete V5 P0 cache for {fold_id}/{split}")
            try:
                raw = json.loads(records_path.read_text(encoding="utf-8"))
                compact_records = pd.DataFrame(raw)
                bank, _ = read_sample_bank(bank_path)
                records = rehydrate_path_records(
                    compact_records,
                    bank=bank,
                    cases=cases,
                )
            except (OSError, json.JSONDecodeError, ValueError, PathBankError) as exc:
                raise V5ExperimentError(f"Cannot read V5 P0 records for {fold_id}/{split}") from exc
            expected_keys = {case.case_key for case in cases}
            if set(records.get("case_key", ())) != expected_keys:
                raise V5ExperimentError(f"V5 P0 cache case keys do not match {fold_id}/{split}")
            return records.sort_values(["target_end_day", "target_contract_id", "case_key"], kind="stable").reset_index(drop=True)
        tokenizer = self._load_tokenizer()
        predictor = self._load_predictor()
        try:
            records, bank = generate_sample_bank(
                tokenizer,
                predictor,
                cases,
                model_name="zero_shot_mean_path",
                **self._p0_generation_arguments(),
            )
        finally:
            self._release(tokenizer, predictor)
        records = _path_records_with_fold(records, fold_id=fold_id, model_name="zero_shot_mean_path")
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        write_sample_bank(
            bank_path,
            bank,
            metadata={
                **self._metadata("p0"),
                "fold_id": fold_id,
                "split": split,
                "case_keys_sha256": hashlib.sha256(
                    "\n".join(sorted(bank)).encode("utf-8")
                ).hexdigest(),
                "sampling": {
                    "sample_count": int(self.config["p0"]["sample_count"]),
                    "temperature": float(self.config["p0"]["temperature"]),
                    "top_k": int(self.config["p0"]["top_k"]),
                    "top_p": float(self.config["p0"]["top_p"]),
                    "random_seed": int(self.config["p0"]["random_seed"]),
                },
            },
        )
        # The raw paths exist only in sample_bank.npz; this index keeps metrics,
        # plots, and later P2 pairing inspectable without a duplicate JSON bank.
        compact_records = records.drop(
            columns=["sample_paths", "actual_path", "predicted_path"], errors="ignore"
        )
        _write_json(records_path, compact_records.to_dict("records"))
        return records

    def _p0_baselines(
        self,
        *,
        path_records: pd.DataFrame,
        fit_cases: Sequence[TargetOnlyCase],
        all_cases: Sequence[TargetOnlyCase],
    ) -> dict[str, pd.DataFrame]:
        fit_probabilities = fit_product_up_probabilities(fit_cases)
        zero_mean = zero_shot_mean_direction_records(path_records)
        zero_vote = zero_shot_vote_direction_records(path_records)
        majority = fit_product_majority_direction_records(path_records, fit_cases=fit_cases)
        momentum = context_3day_momentum_direction_records(
            path_records,
            cases=all_cases,
            fallback_probabilities=fit_probabilities,
        )
        baseline_map = {
            "zero_shot_mean_path": zero_mean,
            "zero_shot_sample_vote": zero_vote,
            "fit_product_majority": majority,
            "context_3day_momentum": momentum,
        }
        expected = set(path_records["case_key"])
        for name, records in baseline_map.items():
            if set(records["case_key"]) != expected:
                raise V5ExperimentError(f"V5 P0 baseline {name} changed case coverage")
        return baseline_map

    @staticmethod
    def _filter_path_records_by_case_keys(
        records: pd.DataFrame,
        cases: Sequence[TargetOnlyCase],
        *,
        label: str,
    ) -> pd.DataFrame:
        """Select one exact case universe after validating its source bank."""

        case_keys = {case.case_key for case in cases}
        selected = records.loc[records["case_key"].astype(str).isin(case_keys)].copy()
        if set(selected["case_key"].astype(str)) != case_keys:
            raise V5ExperimentError(f"V5 {label} records do not cover the requested cases")
        return selected.sort_values(
            ["target_end_day", "target_contract_id", "case_key"], kind="stable"
        ).reset_index(drop=True)

    def run_p0(
        self,
        *,
        fold_id: str | None = None,
        max_cases_per_split: int | None = None,
    ) -> Path:
        """Generate P0 validation/evaluation banks, select baseline, and plot each fold."""

        folds = self._require_folds()
        if fold_id is not None:
            folds = tuple(fold for fold in folds if fold.fold_id == str(fold_id))
            if not folds:
                raise V5ExperimentError(f"Unknown V5 P0 fold: {fold_id!r}")
        if max_cases_per_split is not None and max_cases_per_split < 1:
            raise ValueError("V5 --max-cases-per-split must be positive when supplied")
        smoke = fold_id is not None or max_cases_per_split is not None
        stage_name = "p0_smoke" if smoke else "p0"
        path_artifacts: dict[str, V5FoldPlotArtifacts] = {}
        selection_payloads: dict[str, object] = {}
        evaluation_payloads: dict[str, object] = {}
        evaluation_records: list[pd.DataFrame] = []
        all_baseline_metrics: dict[str, list[pd.DataFrame]] = defaultdict(list)
        for fold in folds:
            fit_cases = cases_in_period(
                self.bundle.target_cases,
                start_day=fold.fit_start_day,
                end_day=fold.fit_end_day,
                products=self.primary_products,
            )
            validation_cases = cases_in_period(
                self.bundle.target_cases,
                start_day=fold.inner_validation_start_day,
                end_day=fold.inner_validation_end_day,
            )
            evaluation_cases = cases_in_period(
                self.bundle.target_cases,
                start_day=fold.evaluation_start_day,
                end_day=fold.evaluation_end_day,
            )
            if max_cases_per_split is not None:
                validation_cases = validation_cases[:max_cases_per_split]
                evaluation_cases = evaluation_cases[:max_cases_per_split]
            if not fit_cases or not validation_cases or not evaluation_cases:
                raise V5ExperimentError(f"V5 P0 lacks fit/validation/evaluation cases for {fold.fold_id}")
            validation_paths = self._create_p0_sample_bank(
                stage_name=stage_name,
                fold_id=fold.fold_id,
                split="inner_validation",
                cases=validation_cases,
            )
            validation_selection_paths = self._filter_path_records_by_case_keys(
                validation_paths,
                cases=cases_in_period(
                    validation_cases,
                    start_day=fold.inner_validation_start_day,
                    end_day=fold.inner_validation_end_day,
                    products=self.primary_products,
                ),
                label=f"P0 validation primary/{fold.fold_id}",
            )
            validation_baselines = self._p0_baselines(
                path_records=validation_selection_paths,
                fit_cases=fit_cases,
                all_cases=self.bundle.target_cases,
            )
            selected_name, selection = selected_baseline(
                validation_baselines,
                selection_order=tuple(self.config["p0"]["selection_order"]),
                allow_unavailable=smoke,
            )
            # This file is committed before the evaluation bank is generated
            # or its realized labels/path metrics are inspected.
            selection_payload = {
                **self._metadata("p0"),
                "fold_id": fold.fold_id,
                "fit_case_count": int(len(fit_cases)),
                "inner_validation_case_count": int(len(validation_cases)),
                "inner_validation_primary_selection_case_count": int(len(validation_selection_paths)),
                "planned_evaluation_case_count": int(len(evaluation_cases)),
                "fit_product_probability_up": fit_product_up_probabilities(fit_cases),
                "selection_excludes_transfer_products": list(self.transfer_products),
                **selection,
            }
            selection_path = self.run_dir / stage_name / fold.fold_id / "p0_baseline_selection.json"
            _write_json(selection_path, selection_payload)
            selection_payloads[fold.fold_id] = {
                "selected_direction_baseline": selected_name,
                "selection_available": bool(selection["selection_available"]),
                "path": str(selection_path),
            }

            evaluation_paths = self._create_p0_sample_bank(
                stage_name=stage_name,
                fold_id=fold.fold_id,
                split="evaluation",
                cases=evaluation_cases,
            )
            evaluation_primary_paths = self._filter_path_records_by_case_keys(
                evaluation_paths,
                cases=cases_in_period(
                    evaluation_cases,
                    start_day=fold.evaluation_start_day,
                    end_day=fold.evaluation_end_day,
                    products=self.primary_products,
                ),
                label=f"P0 evaluation primary/{fold.fold_id}",
            )
            evaluation_baselines = self._p0_baselines(
                path_records=evaluation_primary_paths,
                fit_cases=fit_cases,
                all_cases=self.bundle.target_cases,
            )
            path_artifacts[fold.fold_id] = render_fold_baseline_path_plots(
                evaluation_paths,
                fold_id=fold.fold_id,
                output_dir=self.run_dir / stage_name / "path_evaluation",
                stage=stage_name,
                baseline_label="zero_shot_mean_path",
                metadata={
                    **self._metadata("p0"),
                    "selected_direction_baseline": selected_name,
                    "note": "P0 close plots use the path-producing zero-shot mean; selected direction baselines may be classifier-only.",
                },
            )
            evaluation_summary_path = (
                self.run_dir / stage_name / fold.fold_id / "p0_evaluation_summary.json"
            )
            evaluation_summary = {
                **self._metadata("p0"),
                "fold_id": fold.fold_id,
                "selection_path": str(selection_path),
                "selected_direction_baseline": selected_name,
                "evaluation_primary_case_count": int(len(evaluation_primary_paths)),
                "evaluation_transfer_case_count": int(
                    len(evaluation_paths) - len(evaluation_primary_paths)
                ),
                "evaluation_metrics_all_declared_baselines_primary": {
                    name: _direction_metrics(records)
                    for name, records in evaluation_baselines.items()
                },
                "zero_shot_path_metrics_primary": _path_metric_summary(evaluation_primary_paths),
                "zero_shot_path_metrics_transfer": self._p0_transfer_path_metrics(
                    evaluation_paths=evaluation_paths,
                    evaluation_cases=evaluation_cases,
                    fold_id=fold.fold_id,
                ),
                "path_plot_artifacts": path_artifacts[fold.fold_id].as_dict(),
            }
            _write_json(evaluation_summary_path, evaluation_summary)
            evaluation_payloads[fold.fold_id] = {"path": str(evaluation_summary_path)}
            evaluation_records.append(evaluation_primary_paths)
            for name, records in evaluation_baselines.items():
                all_baseline_metrics[name].append(records)
        combined_paths = pd.concat(evaluation_records, ignore_index=True)
        payload = {
            **self._metadata("p0"),
            "run_mode": "single_fold_smoke" if smoke else "full",
            "record_count": int(len(combined_paths)),
            "zero_shot_mean_path": _path_metric_summary(combined_paths),
            "direction_baselines": {
                name: _direction_metrics(pd.concat(frames, ignore_index=True))
                for name, frames in sorted(all_baseline_metrics.items())
            },
            "baseline_selection_by_fold": selection_payloads,
            "evaluation_by_fold": evaluation_payloads,
            "path_plot_artifacts": {
                fold: artifact.as_dict() for fold, artifact in sorted(path_artifacts.items())
            },
        }
        result_path = self.results_dir / ("p0_smoke_baselines.json" if smoke else "p0_baselines.json")
        _write_json(result_path, payload)
        _write_json(self.run_dir / stage_name / "metrics.json", payload)
        print(json.dumps(_json_safe({"stage": stage_name, "metrics": result_path})))
        return result_path

    def _require_p0_selection(
        self,
        *,
        fold_id: str,
    ) -> dict[str, object]:
        path = self.run_dir / "p0" / fold_id / "p0_baseline_selection.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V5ExperimentError(
                f"V5 P1 requires P0 baseline selection before evaluation: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise V5ExperimentError(f"V5 P0 selection must be a JSON object: {path}")
        expected = {
            "strategy_version": 5,
            "phase": "p0",
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": False,
            "data_fingerprint": self.cohort.data_fingerprint,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise V5ExperimentError(f"V5 P0 selection does not belong to active run/data: {key}")
        selected = value.get("selected_direction_baseline")
        if selected not in set(self.config["p0"]["selection_order"]):
            raise V5ExperimentError(f"V5 P0 selection contains an undeclared baseline: {selected!r}")
        return value

    def _p0_transfer_path_metrics(
        self,
        *,
        evaluation_paths: pd.DataFrame,
        evaluation_cases: Sequence[TargetOnlyCase],
        fold_id: str,
    ) -> dict[str, object]:
        transfer_cases = tuple(
            case for case in evaluation_cases if case.product in self.transfer_products
        )
        if not transfer_cases:
            return {
                "available": False,
                "products": list(self.transfer_products),
                "reason": "no_eligible_transfer_cases",
            }
        transfer_paths = self._filter_path_records_by_case_keys(
            evaluation_paths,
            cases=transfer_cases,
            label=f"P0 evaluation transfer/{fold_id}",
        )
        return {
            "available": True,
            "products": list(self.transfer_products),
            "zero_shot_path_metrics": _path_metric_summary(transfer_paths),
        }

    def _all_fold_evaluation_cases(self, *, fold_id: str) -> tuple[TargetOnlyCase, ...]:
        """Resolve the exact all-product P0 evaluation universe for one fold."""

        fold = next((item for item in self._require_folds() if item.fold_id == fold_id), None)
        if fold is None:
            raise V5ExperimentError(f"Unknown V5 fold: {fold_id!r}")
        return cases_in_period(
            self.bundle.target_cases,
            start_day=fold.evaluation_start_day,
            end_day=fold.evaluation_end_day,
        )

    def _p0_path_artifact_references(self, *, fold_id: str) -> dict[str, str]:
        """Return the one required P0 close artifact pair for a later P1 fold.

        P1 emits only direction probabilities, so rendering another pair of
        close figures would be a duplicate of the immutable P0 sample bank.
        Verify the P0 artifacts instead and retain their exact paths.
        """

        root = self.run_dir / "p0" / "path_evaluation" / fold_id
        artifacts = {
            "day3_close_return_comparison": root / "day3_close_return_comparison.png",
            "close_price_comparison": root / "close_price_comparison.png",
            "summary_json": root / "path_comparison_summary.json",
        }
        missing = [str(path) for path in artifacts.values() if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise V5ExperimentError(
                f"V5 P1 requires the completed P0 close artifacts for {fold_id}: {missing!r}"
            )
        return {name: str(path) for name, path in artifacts.items()}

    def _load_p0_direction_baseline(
        self,
        *,
        fold_id: str,
        selected_name: str,
        fit_cases: Sequence[TargetOnlyCase],
        evaluation_cases: Sequence[TargetOnlyCase],
    ) -> pd.DataFrame:
        bank_path, records_path = self._p0_paths(
            stage_name="p0", fold_id=fold_id, split="evaluation"
        )
        try:
            compact_records = pd.DataFrame(json.loads(records_path.read_text(encoding="utf-8")))
            bank, _ = read_sample_bank(bank_path)
            all_evaluation_cases = self._all_fold_evaluation_cases(fold_id=fold_id)
            all_path_records = rehydrate_path_records(
                compact_records,
                bank=bank,
                cases=all_evaluation_cases,
            )
        except (OSError, json.JSONDecodeError, ValueError, PathBankError) as exc:
            raise V5ExperimentError(f"Cannot load V5 P0 evaluation records for {fold_id}") from exc
        path_records = self._filter_path_records_by_case_keys(
            all_path_records,
            evaluation_cases,
            label=f"P1 primary baseline/{fold_id}",
        )
        if set(path_records.get("case_key", ())) != {case.case_key for case in evaluation_cases}:
            raise V5ExperimentError(f"V5 P0 evaluation record keys differ from P1 evaluation: {fold_id}")
        return self._p0_baselines(
            path_records=path_records,
            fit_cases=fit_cases,
            all_cases=self.bundle.target_cases,
        )[selected_name]

    def _probe_training_config(self, *, seed: int) -> TargetProbeTrainingConfig:
        training = self.config["p1"]["training"]
        return TargetProbeTrainingConfig(
            learning_rate=float(training["learning_rate"]),
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            weight_decay=float(training["weight_decay"]),
            gradient_clip=float(training["gradient_clip"]),
            num_workers=int(training["num_workers"]),
            seed=int(seed),
            sampling_strategy=str(training["sampling_strategy"]),
        )

    def _run_p1_seed(
        self,
        *,
        tokenizer: KronosTokenizer,
        predictor: Kronos,
        storage_stage_name: str,
        fold_id: str,
        seed: int,
        fit_cases: Sequence[TargetOnlyCase],
        validation_cases: Sequence[TargetOnlyCase],
        evaluation_cases: Sequence[TargetOnlyCase],
    ) -> pd.DataFrame:
        training = self._probe_training_config(seed=seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        probe = TargetOnlyProbe(
            tokenizer,
            predictor,
            fusion_hidden_dim=int(self.config["p1"]["fusion_hidden_dim"]),
            dropout=float(self.config["p1"]["dropout"]),
        )
        output_dir = self.run_dir / storage_stage_name / fold_id / f"seed_{seed}"
        checkpoint_path = output_dir / "best_probe_head.pt"
        records_path = output_dir / "evaluation_records.json"
        if checkpoint_path.is_file() and records_path.is_file():
            try:
                cached = pd.DataFrame(json.loads(records_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise V5ExperimentError(
                    f"Cannot read cached V5 P1 evaluation records: {records_path}"
                ) from exc
            if set(cached.get("case_key", ())) != {case.case_key for case in evaluation_cases}:
                raise V5ExperimentError(
                    f"Cached V5 P1 records do not match evaluation cases: {fold_id}/seed_{seed}"
                )
            return cached.sort_values(
                ["target_end_day", "target_contract_id", "case_key"], kind="stable"
            ).reset_index(drop=True)
        if checkpoint_path.is_file() or records_path.is_file():
            raise V5ExperimentError(
                f"Incomplete V5 P1 cache for {fold_id}/seed_{seed}; preserve it and use a new run ID"
            )
        result = train_target_probe(
            probe,
            fit_cases,
            validation_cases,
            config=training,
            device=self.device,
            output_dir=output_dir,
            clip=float(self.config["data"]["clip"]),
            epsilon=float(self.config["data"]["normalization_epsilon"]),
        )
        load_target_probe_head(probe, result.checkpoint_path)
        raw = evaluate_target_probe(
            probe,
            evaluation_cases,
            device=self.device,
            batch_size=training.batch_size,
            clip=float(self.config["data"]["clip"]),
            epsilon=float(self.config["data"]["normalization_epsilon"]),
        )
        records = _attach_fold(
            raw,
            fold_id=fold_id,
            model_name="target_only_probe",
            seed=seed,
        )
        _write_json(output_dir / "evaluation_records.json", records.to_dict("records"))
        _write_json(
            output_dir / "summary.json",
            {
                **self._metadata("p1_signal"),
                "fold_id": fold_id,
                "seed": seed,
                "checkpoint": str(result.checkpoint_path),
                "best_epoch": result.best_epoch,
                "best_balanced_accuracy": result.best_balanced_accuracy,
                "sampling": result.sampling_summary,
                "elapsed_seconds": result.elapsed_seconds,
                "evaluation_metrics": probe_metrics(records),
            },
        )
        del probe
        return records

    def _p1_gate(
        self,
        *,
        records_by_seed: Mapping[int, pd.DataFrame],
        baseline: pd.DataFrame,
        fit_probabilities_by_fold: Mapping[str, Mapping[str, float]],
        smoke: bool,
    ) -> dict[str, object]:
        if smoke:
            return {
                "available": False,
                "reason": "single_fold_smoke_does_not_satisfy_the_five_fold_p1_gate",
                "allows_next_phase": False,
            }
        expected = (42, 43, 44)
        if tuple(sorted(records_by_seed)) != expected:
            raise V5ExperimentError("V5 P1 gate requires exactly three declared seeds")
        seed_improvements: dict[str, float | None] = {}
        for seed in expected:
            seed_improvements[str(seed)] = _balanced_accuracy_difference(records_by_seed[seed], baseline)
        ensemble = _probe_seed_ensemble(records_by_seed)
        _assert_same_direction_records(ensemble, baseline, label="V5 P1 ensemble")
        ensemble_metrics = probe_metrics(ensemble)
        baseline_metrics = _direction_metrics(baseline)
        fold_improvements: dict[str, float | None] = {}
        for fold_id, candidate_fold in ensemble.groupby("fold_id", sort=True):
            baseline_fold = baseline.loc[baseline["fold_id"].astype(str) == str(fold_id)]
            fold_improvements[str(fold_id)] = _balanced_accuracy_difference(candidate_fold, baseline_fold)
        product_degradations: dict[str, float | None] = {}
        for product in self.primary_products:
            candidate_product = ensemble.loc[ensemble["product"].astype(str) == product]
            baseline_product = baseline.loc[baseline["product"].astype(str) == product]
            product_degradations[product] = (
                _balanced_accuracy_difference(candidate_product, baseline_product)
                if not candidate_product.empty and not baseline_product.empty
                else None
            )
        bootstrap = {
            f"block_{block}": _paired_block_bootstrap(
                ensemble,
                baseline,
                block_days=int(block),
                iterations=int(self.config["evaluation"]["bootstrap_iterations"]),
                seed=int(self.config["evaluation"]["bootstrap_random_seed"]) + int(block),
            )
            for block in self.config["evaluation"]["bootstrap_block_days"]
        }
        fit_brier_frames: list[pd.DataFrame] = []
        for fold_id, candidate_fold in ensemble.groupby("fold_id", sort=True):
            fit_probabilities = fit_probabilities_by_fold[str(fold_id)]
            fit_brier_frames.append(
                pd.DataFrame(
                    {
                        "fold_id": candidate_fold["fold_id"],
                        "product": candidate_fold["product"],
                        "actual_label": candidate_fold["actual_label"],
                        "valid_direction": candidate_fold["valid_direction"],
                        "fit_probability_up": candidate_fold["product"].map(fit_probabilities),
                    }
                )
            )
        fit_brier_records = pd.concat(fit_brier_frames, ignore_index=True)
        fit_brier_valid = fit_brier_records.loc[
            fit_brier_records["valid_direction"].astype(bool)
        ]
        brier_probability_baseline = (
            float(
                np.mean(
                    (
                        fit_brier_valid["fit_probability_up"].to_numpy(dtype=np.float64)
                        - fit_brier_valid["actual_label"].to_numpy(dtype=np.float64)
                    )
                    ** 2
                )
            )
            if not fit_brier_valid.empty
            else float("nan")
        )
        finite_seed_improvements = [
            float(value) for value in seed_improvements.values() if value is not None and math.isfinite(value)
        ]
        median_seed_improvement = float(np.median(finite_seed_improvements)) if finite_seed_improvements else None
        conditions = {
            "median_seed_improvement_at_least_2pp": (
                median_seed_improvement is not None and median_seed_improvement >= 0.02
            ),
            "at_least_two_of_three_seed_improvements_positive": sum(
                value is not None and value > 0.0 for value in seed_improvements.values()
            ) >= 2,
            "at_least_three_of_five_folds_improve": (
                len(fold_improvements) == 5
                and sum(value is not None and value > 0.0 for value in fold_improvements.values()) >= 3
            ),
            "no_primary_product_degrades_more_than_1pp": all(
                value is not None and value >= -0.01 for value in product_degradations.values()
            ),
            "bootstrap_5_and_10_day_probability_at_least_80pct": all(
                bool(value.get("available"))
                and float(value.get("probability_improvement_positive") or 0.0) >= 0.80
                for value in bootstrap.values()
            ),
            "ensemble_roc_auc_at_least_055": (
                ensemble_metrics["roc_auc"] is not None and float(ensemble_metrics["roc_auc"]) >= 0.55
            ),
            "ensemble_brier_at_most_025": (
                ensemble_metrics["brier_score"] is not None and float(ensemble_metrics["brier_score"]) <= 0.25
            ),
            "ensemble_brier_not_worse_than_fit_product_probability": (
                ensemble_metrics["brier_score"] is not None
                and math.isfinite(brier_probability_baseline)
                and float(ensemble_metrics["brier_score"]) <= brier_probability_baseline
            ),
        }
        passed = bool(all(conditions.values()))
        return {
            "available": True,
            "selected_baseline": baseline_metrics,
            "seed_balanced_accuracy_improvements": seed_improvements,
            "median_seed_balanced_accuracy_improvement": median_seed_improvement,
            "ensemble": {
                "target_only_probe": ensemble_metrics,
                "selected_direction_baseline": baseline_metrics,
                "balanced_accuracy_improvement": _balanced_accuracy_difference(ensemble, baseline),
                "pooled_fit_product_probability_brier": brier_probability_baseline,
                "records": ensemble,
            },
            "fold_balanced_accuracy_improvements": fold_improvements,
            "primary_product_balanced_accuracy_improvements": product_degradations,
            "bootstrap": bootstrap,
            "conditions": conditions,
            "passes_p1_gate": passed,
            "allows_next_phase": passed,
        }

    @staticmethod
    def _gate_for_json(gate: Mapping[str, object]) -> dict[str, object]:
        output = dict(gate)
        ensemble = output.get("ensemble")
        if isinstance(ensemble, Mapping):
            output["ensemble"] = {key: value for key, value in ensemble.items() if key != "records"}
        return output

    def run_p1_signal(self, *, fold_id: str | None = None) -> Path:
        """Fit frozen target-only probes and write the pre-registered P1 gate."""

        folds = self._require_folds()
        if fold_id is not None:
            folds = tuple(fold for fold in folds if fold.fold_id == str(fold_id))
            if not folds:
                raise V5ExperimentError(f"Unknown V5 P1 fold: {fold_id!r}")
        smoke = fold_id is not None
        stage_name = "p1_signal_smoke" if smoke else "p1_signal"
        seeds = tuple(int(seed) for seed in self.config["p1"]["seeds"])
        all_seed_records: dict[int, list[pd.DataFrame]] = {seed: [] for seed in seeds}
        all_baselines: list[pd.DataFrame] = []
        transfer_seed_records: dict[int, list[pd.DataFrame]] = {seed: [] for seed in seeds}
        p0_path_artifacts: dict[str, dict[str, str]] = {}
        fit_probabilities_by_fold: dict[str, dict[str, float]] = {}
        selected_baselines: dict[str, str] = {}
        for fold in folds:
            selection = self._require_p0_selection(fold_id=fold.fold_id)
            selected_name = str(selection["selected_direction_baseline"])
            selected_baselines[fold.fold_id] = selected_name
            fit_cases = cases_in_period(
                self.bundle.target_cases,
                start_day=fold.fit_start_day,
                end_day=fold.fit_end_day,
                products=self.primary_products,
            )
            validation_cases = cases_in_period(
                self.bundle.target_cases,
                start_day=fold.inner_validation_start_day,
                end_day=fold.inner_validation_end_day,
                products=self.primary_products,
            )
            evaluation_primary = cases_in_period(
                self.bundle.target_cases,
                start_day=fold.evaluation_start_day,
                end_day=fold.evaluation_end_day,
                products=self.primary_products,
            )
            evaluation_transfer = cases_in_period(
                self.bundle.target_cases,
                start_day=fold.evaluation_start_day,
                end_day=fold.evaluation_end_day,
                products=self.transfer_products,
            )
            if not fit_cases or not validation_cases or not evaluation_primary:
                raise V5ExperimentError(f"V5 P1 lacks primary fit/validation/evaluation cases for {fold.fold_id}")
            baseline_primary = self._load_p0_direction_baseline(
                fold_id=fold.fold_id,
                selected_name=selected_name,
                fit_cases=fit_cases,
                evaluation_cases=evaluation_primary,
            )
            if set(baseline_primary["case_key"]) != {case.case_key for case in evaluation_primary}:
                raise V5ExperimentError(f"Selected V5 P0 baseline keys mismatch primary P1 evaluation: {fold.fold_id}")
            fit_probabilities_by_fold[fold.fold_id] = fit_product_up_probabilities(fit_cases)
            fold_seed_primary: dict[int, pd.DataFrame] = {}
            fold_seed_transfer: dict[int, pd.DataFrame] = {}
            for seed in seeds:
                # One frozen backbone is loaded per seed rather than kept
                # alongside all three heads. This bounds GPU memory and avoids
                # cross-seed state carryover.
                tokenizer = self._load_tokenizer()
                predictor = self._load_predictor()
                try:
                    primary_records = self._run_p1_seed(
                        tokenizer=tokenizer,
                        predictor=predictor,
                        storage_stage_name=stage_name,
                        fold_id=fold.fold_id,
                        seed=seed,
                        fit_cases=fit_cases,
                        validation_cases=validation_cases,
                        evaluation_cases=evaluation_primary,
                    )
                    fold_seed_primary[seed] = primary_records
                    if evaluation_transfer:
                        # The shared probe is evaluated on j only; it was never
                        # fit or selected using j labels.
                        probe = TargetOnlyProbe(
                            tokenizer,
                            predictor,
                            fusion_hidden_dim=int(self.config["p1"]["fusion_hidden_dim"]),
                            dropout=float(self.config["p1"]["dropout"]),
                        )
                        checkpoint = (
                            self.run_dir
                            / stage_name
                            / fold.fold_id
                            / f"seed_{seed}"
                            / "best_probe_head.pt"
                        )
                        load_target_probe_head(probe, checkpoint)
                        transfer_raw = evaluate_target_probe(
                            probe,
                            evaluation_transfer,
                            device=self.device,
                            batch_size=int(self.config["p1"]["training"]["batch_size"]),
                            clip=float(self.config["data"]["clip"]),
                            epsilon=float(self.config["data"]["normalization_epsilon"]),
                        )
                        transfer_records = _attach_fold(
                            transfer_raw,
                            fold_id=fold.fold_id,
                            model_name="target_only_probe_transfer",
                            seed=seed,
                        )
                        _write_json(
                            self.run_dir
                            / stage_name
                            / fold.fold_id
                            / f"seed_{seed}"
                            / "transfer_evaluation_records.json",
                            transfer_records.to_dict("records"),
                        )
                        fold_seed_transfer[seed] = transfer_records
                        del probe
                finally:
                    self._release(tokenizer, predictor)
            ensemble_primary = _probe_seed_ensemble(fold_seed_primary)
            _assert_same_direction_records(
                ensemble_primary, baseline_primary, label=f"V5 P1/{fold.fold_id}"
            )
            p0_paths = self._p0_path_artifact_references(fold_id=fold.fold_id)
            p0_path_artifacts[fold.fold_id] = p0_paths
            for seed, records in fold_seed_primary.items():
                all_seed_records[seed].append(records)
            for seed, records in fold_seed_transfer.items():
                transfer_seed_records[seed].append(records)
            all_baselines.append(baseline_primary)
        combined_seed_records = {
            seed: pd.concat(frames, ignore_index=True) for seed, frames in all_seed_records.items() if frames
        }
        combined_baseline = pd.concat(all_baselines, ignore_index=True)
        gate = self._p1_gate(
            records_by_seed=combined_seed_records,
            baseline=combined_baseline,
            fit_probabilities_by_fold=fit_probabilities_by_fold,
            smoke=smoke,
        )
        transfer_metrics: dict[str, object]
        if all(transfer_seed_records[seed] for seed in seeds):
            transfer_ensemble = _probe_seed_ensemble(
                {seed: pd.concat(transfer_seed_records[seed], ignore_index=True) for seed in seeds}
            )
            transfer_metrics = {
                "available": True,
                "products": list(self.transfer_products),
                "target_only_probe": probe_metrics(transfer_ensemble),
                "record_count": int(len(transfer_ensemble)),
            }
        else:
            transfer_metrics = {
                "available": False,
                "products": list(self.transfer_products),
                "reason": "no_transfer_cases_in_every_seed_and_fold",
            }
        metrics_payload = {
            **self._metadata("p1_signal"),
            "run_mode": "single_fold_smoke" if smoke else "full",
            "selected_direction_baseline_by_fold": selected_baselines,
            "record_counts_by_seed": {str(seed): int(len(records)) for seed, records in combined_seed_records.items()},
            "p0_path_plot_artifacts_reused_without_duplication": p0_path_artifacts,
            "transfer_report": transfer_metrics,
            "gate_path": str(self.results_dir / "p1_signal_gate.json") if not smoke else None,
        }
        result_path = self.results_dir / ("p1_signal_smoke_metrics.json" if smoke else "p1_signal_metrics.json")
        _write_json(result_path, metrics_payload)
        _write_json(self.run_dir / stage_name / "metrics.json", metrics_payload)
        if not smoke:
            gate_payload = {
                **self._metadata("p1_signal"),
                **self._gate_for_json(gate),
                "selected_direction_baseline_by_fold": selected_baselines,
                "p0_path_plot_artifacts_reused_without_duplication": p0_path_artifacts,
            }
            gate_path = self.results_dir / "p1_signal_gate.json"
            _write_json(gate_path, gate_payload)
            _write_json(self.run_dir / stage_name / "p1_signal_gate.json", gate_payload)
        print(
            json.dumps(
                _json_safe(
                    {
                        "stage": stage_name,
                        "metrics": result_path,
                        "gate": None if smoke else self.results_dir / "p1_signal_gate.json",
                    }
                )
            )
        )
        return result_path

    def _require_gate(self, *, phase: str, filename: str) -> Mapping[str, object]:
        path = self.results_dir / filename
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V5ExperimentError(f"V5 {phase} requires persisted gate: {path}") from exc
        if not isinstance(value, dict):
            raise V5ExperimentError(f"V5 {phase} gate must be a JSON object")
        expected = {
            "strategy_version": 5,
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": False,
            "data_fingerprint": self.cohort.data_fingerprint,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise V5ExperimentError(f"V5 gate does not belong to active run/data: {key}")
        if not bool(value.get("allows_next_phase")):
            raise V5ExperimentError(f"V5 {phase} gate did not pass; refusing later phase")
        return value

    def run_p2_path_bridge(self) -> None:
        self._require_gate(phase="P2", filename="p1_signal_gate.json")
        raise V5ExperimentError(
            "V5 P2 is intentionally not implemented until synchronized P1 CUDA results are reviewed."
        )

    def run_p3_adapter(self) -> None:
        self._require_gate(phase="P3", filename="p2_path_bridge_gate.json")
        raise V5ExperimentError(
            "V5 P3 is intentionally unavailable until P2 is implemented and passes."
        )

    def run_p4_stability(self) -> None:
        self._require_gate(phase="P4", filename="p3_adapter_gate.json")
        raise V5ExperimentError(
            "V5 P4 is intentionally unavailable until P3 is implemented and passes."
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kronos V5 target-only direction-guided paths")
    parser.add_argument(
        "stage",
        choices=("audit", "p0", "p1-signal", "p2-path-bridge", "p3-adapter", "p4-stability"),
    )
    parser.add_argument("--config", default="csj/configs/target_only_path_v5.yaml")
    parser.add_argument("--run-id", default="v5_cuda")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--fold-id",
        default=None,
        help="Run one P0/P1 fold as a non-gating CUDA smoke; omit for five-fold formal stage.",
    )
    parser.add_argument(
        "--max-cases-per-split",
        type=int,
        default=None,
        help="Limit P0 inner-validation/evaluation cases for a local smoke only; omitted is formal coverage.",
    )
    args = parser.parse_args(argv)
    try:
        experiment = V5Experiment(
            args.config,
            args.run_id,
            device_override=args.device,
            allow_model_download=args.allow_model_download,
        )
        if args.stage == "audit":
            if args.fold_id is not None or args.max_cases_per_split is not None:
                parser.error("P0 selection arguments do not apply to audit")
            experiment.audit()
        elif args.stage == "p0":
            experiment.run_p0(
                fold_id=args.fold_id,
                max_cases_per_split=args.max_cases_per_split,
            )
        elif args.stage == "p1-signal":
            if args.max_cases_per_split is not None:
                parser.error("--max-cases-per-split applies only to P0")
            experiment.run_p1_signal(fold_id=args.fold_id)
        elif args.stage == "p2-path-bridge":
            if args.fold_id is not None or args.max_cases_per_split is not None:
                parser.error("P2 does not accept smoke-limit arguments")
            experiment.run_p2_path_bridge()
        elif args.stage == "p3-adapter":
            if args.fold_id is not None or args.max_cases_per_split is not None:
                parser.error("P3 does not accept smoke-limit arguments")
            experiment.run_p3_adapter()
        elif args.stage == "p4-stability":
            if args.fold_id is not None or args.max_cases_per_split is not None:
                parser.error("P4 does not accept smoke-limit arguments")
            experiment.run_p4_stability()
    except (
        V5ExperimentError,
        TargetOnlyProbeError,
        PathBankError,
        V5PlotError,
        RuntimeError,
    ) as exc:
        parser.exit(2, f"V5 stage blocked: {exc}\n")


if __name__ == "__main__":
    main()


__all__ = ["V5Experiment", "V5ExperimentError", "main", "resolve_device"]
