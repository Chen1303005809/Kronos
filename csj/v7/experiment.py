"""Phase-gated V7 orchestration through the CUDA P1 path-bank and baselines.

P1 is intentionally the last implemented phase.  It proves that frozen Kronos
paths can be cached and inspected without touching a risk-head or any outer
evaluation performance metric.  P2--P5 remain explicit guarded placeholders.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from csj.v5.target_data import TargetOnlyCase, load_target_only_observed_cohort
from csj.v7 import PRODUCTION_ELIGIBLE, RESULT_SCOPE, STRATEGY_VERSION
from csj.v7.audit import V7AuditBundle, build_v7_p0_audit
from csj.v7.baselines import (
    V7BaselineError,
    attach_context_features,
    choose_baseline,
    classification_metrics,
    fit_and_predict_baselines,
)
from csj.v7.config import REPO_ROOT, load_v7_config, validate_v7_config
from csj.v7.path_bank import (
    PATH_BANK_SCHEMA_VERSION,
    V7PathBankError,
    build_shard_arrays,
    cache_key_for_case,
    generate_case_output,
    make_shard_entry,
    read_shard,
    sampling_seed,
    sha256_case_keys,
    sha256_path,
    validate_raw_paths,
    validate_shard_entry,
    write_shard_atomic,
)
from csj.v7.plotting import V7PlotError, render_p1_plots
from csj.v7.provenance import (
    V7ProvenanceError,
    phase_metadata,
    read_json,
    runtime_provenance,
    sha256_json,
    write_json,
)
from model import Kronos, KronosTokenizer


CANONICAL_P0_RUN_ID = "v7_p0_20260813_verified"
CANONICAL_SNAPSHOT_ID = "20260813T105916+0800"
CANONICAL_DATA_FINGERPRINT = "2387ff95a1f7bedc7a96377d3af0a45a4bf02f707c2edab1d3f47731d6e0ea1c"
CANONICAL_P0_AUDIT_SHA256 = "45bc5866e9dadbf897c07fc688809e0906a763076c1002b111c3709f6fabd32b"
CANONICAL_P0_GATE_SHA256 = "7e606cc45acbf2cfaa99e92bba8373e8e26e2b19fc6df2fc80e8859a950887bf"
CANONICAL_P0_RECORDS_SHA256 = "17b78903b5eaf018682900048e53ed25c5fb3ec1b228d49d5d3e853e9bdad24c"


class V7ExperimentError(RuntimeError):
    """A V7 phase cannot proceed under its frozen protocol."""


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise V7ExperimentError("V7 requested CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported V7 device: {requested!r}")


def _module_sha256(module: torch.nn.Module) -> str:
    """Hash frozen parameters/buffers without relying on serialization format."""

    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _assert_frozen(module: torch.nn.Module, *, label: str) -> None:
    if module.training:
        raise V7ExperimentError(f"V7 {label} must be in eval mode")
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise V7ExperimentError(f"V7 {label} is not frozen: {trainable[:3]!r}")
    gradients = [name for name, parameter in module.named_parameters() if parameter.grad is not None]
    if gradients:
        raise V7ExperimentError(f"V7 {label} unexpectedly has gradients: {gradients[:3]!r}")


class V7Experiment:
    """Owns one V7 run directory while reusing only validated immutable cache shards."""

    def __init__(
        self,
        config_path: str | Path,
        run_id: str,
        *,
        device_override: str | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.config = load_v7_config(config_path)
        if device_override is not None:
            self.config["runtime"]["device"] = str(device_override)
            validate_v7_config(self.config)
        self.run_id = str(run_id)
        self.run_dir = Path(self.config["output"]["root"]) / self.run_id
        self.results_dir = Path(self.config["output"]["results_root"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.local_files_only = not bool(allow_model_download)
        self._device: torch.device | None = None
        self.cohort = load_target_only_observed_cohort(self.config["data"]["snapshot_root"])
        self._verify_canonical_snapshot()
        self.p0_gate_path, self.p0_audit_path, self.p0_records_path = self._canonical_p0_paths()
        self.p0_gate = self._verify_canonical_p0()
        self.audit_bundle: V7AuditBundle = build_v7_p0_audit(self.cohort, self.config)
        self.fold_records = self._load_canonical_fold_records()
        self.case_by_key = {
            case.case_key: case for case in self.audit_bundle.case_bundle.target_cases
        }
        self.unique_case_keys = tuple(sorted(set(self.fold_records["case_key"].astype(str))))
        expected_cases = int(self.config["path_bank"]["expected_unique_cases"])
        if len(self.unique_case_keys) != expected_cases:
            raise V7ExperimentError(
                f"Canonical P0 case universe is {len(self.unique_case_keys)}, expected {expected_cases}"
            )
        missing_cases = sorted(set(self.unique_case_keys).difference(self.case_by_key))
        if missing_cases:
            raise V7ExperimentError(
                f"Canonical P0 records refer to unavailable target cases: {missing_cases[:3]!r}"
            )
        self.unique_cases = tuple(
            sorted(
                (self.case_by_key[key] for key in self.unique_case_keys),
                key=lambda case: (
                    case.pred_len,
                    case.target_end_day,
                    case.target_contract_id,
                    case.case_key,
                ),
            )
        )
        self.config["runtime_resolved"] = {
            "device_requested": str(self.config["runtime"]["device"]),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "model_download_allowed": bool(allow_model_download),
            "canonical_p0_run_id": CANONICAL_P0_RUN_ID,
            "canonical_p0_gate_sha256": CANONICAL_P0_GATE_SHA256,
            "canonical_unique_case_count": len(self.unique_cases),
        }
        # Resolve once before hashing resolved_config.json.  Every formal P1
        # artifact must retain the actual runtime/device provenance rather than
        # a pre-CUDA placeholder hash that changes on first model loading.
        self._device = resolve_device(str(self.config["runtime"]["device"]))
        self.config["runtime_resolved"].update(runtime_provenance(self._device))
        self.resolved_config_path = write_json(self.run_dir / "resolved_config.json", self.config)
        self._write_upstream_p0_reference()
        print(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "canonical_p0_verified": True,
                    "unique_cases": len(self.unique_cases),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = resolve_device(str(self.config["runtime"]["device"]))
            self.config["runtime_resolved"].update(runtime_provenance(self._device))
            write_json(self.resolved_config_path, self.config)
        return self._device

    def _canonical_p0_paths(self) -> tuple[Path, Path, Path]:
        root = REPO_ROOT / "csj" / "runs" / "risk_control_v7" / CANONICAL_P0_RUN_ID
        return root / "p0_gate.json", root / "data_audit.json", root / "p0" / "fold_label_records.json"

    def _verify_canonical_snapshot(self) -> None:
        expected = REPO_ROOT / "csj" / "data" / "active_contract_snapshots" / CANONICAL_SNAPSHOT_ID
        configured = Path(self.config["data"]["snapshot_root"]).resolve()
        if configured != expected.resolve():
            raise V7ExperimentError(
                f"V7 requires the canonical P0 snapshot {expected}, got {configured}"
            )
        if self.cohort.snapshot_id != CANONICAL_SNAPSHOT_ID:
            raise V7ExperimentError("Loaded V7 cohort does not have the canonical snapshot ID")
        if self.cohort.data_fingerprint != CANONICAL_DATA_FINGERPRINT:
            raise V7ExperimentError("Loaded V7 cohort data fingerprint does not match canonical P0")

    def _verify_canonical_p0(self) -> dict[str, Any]:
        p0_gate_path, p0_audit_path, p0_records_path = self._canonical_p0_paths()
        required = (
            (p0_gate_path, CANONICAL_P0_GATE_SHA256, "gate"),
            (p0_audit_path, CANONICAL_P0_AUDIT_SHA256, "audit"),
            (p0_records_path, CANONICAL_P0_RECORDS_SHA256, "fold records"),
        )
        for path, expected_hash, label in required:
            observed = sha256_path(path)
            if observed != expected_hash:
                raise V7ExperimentError(
                    f"Canonical V7 P0 {label} SHA-256 mismatch: {observed} != {expected_hash}"
                )
        gate = read_json(p0_gate_path, label="canonical V7 P0 gate")
        expected = {
            "strategy_version": STRATEGY_VERSION,
            "phase": "p0",
            "run_id": CANONICAL_P0_RUN_ID,
            "result_scope": RESULT_SCOPE,
            "production_eligible": PRODUCTION_ELIGIBLE,
            "snapshot_id": CANONICAL_SNAPSHOT_ID,
            "data_fingerprint": CANONICAL_DATA_FINGERPRINT,
        }
        for key, expected_value in expected.items():
            if gate.get(key) != expected_value:
                raise V7ExperimentError(
                    f"Canonical V7 P0 gate mismatch at {key}: {gate.get(key)!r} != {expected_value!r}"
                )
        if not bool(gate.get("allows_next_phase")):
            raise V7ExperimentError("Canonical V7 P0 gate does not unlock P1")
        if gate.get("failed_condition_ids"):
            raise V7ExperimentError("Canonical V7 P0 gate has failed conditions")
        return gate

    def _load_canonical_fold_records(self) -> pd.DataFrame:
        payload = read_json(self.p0_records_path, label="canonical V7 P0 fold records")
        for key, value in {
            "strategy_version": STRATEGY_VERSION,
            "phase": "p0",
            "run_id": CANONICAL_P0_RUN_ID,
            "data_fingerprint": self.cohort.data_fingerprint,
        }.items():
            if payload.get(key) != value:
                raise V7ExperimentError(f"Canonical P0 record metadata mismatch at {key}")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise V7ExperimentError("Canonical V7 P0 fold records are not a list")
        records = pd.DataFrame(raw_records)
        required = {
            "case_key", "fold_id", "split", "product", "origin_trading_day",
            "long_tail_event", "short_tail_event", "long_tail_threshold",
            "short_tail_threshold", "context_horizon_scale", "origin_close",
            "context_clip_fraction", "data_fingerprint",
        }
        missing = sorted(required.difference(records.columns))
        if missing:
            raise V7ExperimentError(f"Canonical P0 records miss columns: {missing!r}")
        if len(records) != 17263 or records["case_key"].isna().any():
            raise V7ExperimentError("Canonical P0 fold-record universe is incomplete")
        if set(records["data_fingerprint"].astype(str)) != {self.cohort.data_fingerprint}:
            raise V7ExperimentError("Canonical P0 records mix data fingerprints")
        return records.sort_values(
            ["fold_id", "split", "origin_trading_day", "case_key"], kind="stable"
        ).reset_index(drop=True)

    def _write_upstream_p0_reference(self) -> None:
        payload = {
            "canonical_only": True,
            "run_id": CANONICAL_P0_RUN_ID,
            "snapshot_id": CANONICAL_SNAPSHOT_ID,
            "data_fingerprint": self.cohort.data_fingerprint,
            "p0_gate": {"path": str(self.p0_gate_path), "sha256": sha256_path(self.p0_gate_path)},
            "p0_audit": {"path": str(self.p0_audit_path), "sha256": sha256_path(self.p0_audit_path)},
            "fold_label_records": {
                "path": str(self.p0_records_path),
                "sha256": sha256_path(self.p0_records_path),
            },
        }
        write_json(self.run_dir / "upstream_p0.json", payload)

    def _metadata(self, phase: str, *, smoke: bool) -> dict[str, object]:
        runtime: Mapping[str, object] = self.config.get("runtime_resolved", {})
        return phase_metadata(
            phase=phase,
            run_id=self.run_id,
            config=self.config,
            data_fingerprint=self.cohort.data_fingerprint,
            resolved_config_path=self.resolved_config_path,
            repo_root=REPO_ROOT,
            smoke=smoke,
            runtime=runtime,
            upstream_gate_path=self.p0_gate_path,
        )

    def _load_models(self) -> tuple[KronosTokenizer, Kronos, dict[str, str]]:
        model_config = self.config["model"]
        tokenizer = KronosTokenizer.from_pretrained(
            model_config["tokenizer_id"],
            revision=model_config["tokenizer_revision"],
            cache_dir=model_config["cache_dir"],
            local_files_only=self.local_files_only,
        )
        predictor = Kronos.from_pretrained(
            model_config["predictor_id"],
            revision=model_config["predictor_revision"],
            cache_dir=model_config["cache_dir"],
            local_files_only=self.local_files_only,
        )
        tokenizer.requires_grad_(False).eval().to(self.device)
        predictor.requires_grad_(False).eval().to(self.device)
        _assert_frozen(tokenizer, label="tokenizer")
        _assert_frozen(predictor, label="predictor")
        return tokenizer, predictor, {
            "tokenizer_parameter_sha256": _module_sha256(tokenizer),
            "predictor_parameter_sha256": _module_sha256(predictor),
        }

    def _release_models(self, *models: torch.nn.Module) -> None:
        for model in models:
            model.to("cpu")
        gc.collect()
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _stage_name(self, *, smoke: bool) -> str:
        return "p1_path_bank_smoke" if smoke else "p1_path_bank"

    def _selected_cases(self, *, smoke: bool) -> tuple[TargetOnlyCase, ...]:
        if not smoke:
            return self.unique_cases
        count = int(self.config["path_bank"]["smoke_case_count"])
        if count > len(self.unique_cases):
            raise V7ExperimentError("V7 smoke case count exceeds the canonical case universe")
        indices = np.unique(np.linspace(0, len(self.unique_cases) - 1, count, dtype=int))
        if len(indices) != count:
            raise V7ExperimentError("V7 cannot build the declared smoke case selection")
        return tuple(self.unique_cases[int(index)] for index in indices)

    def _sampling(self, *, smoke: bool) -> dict[str, object]:
        path_bank = self.config["path_bank"]
        return {
            "sample_count": int(
                path_bank["smoke_sample_count"] if smoke else path_bank["sample_count"]
            ),
            "temperature": float(path_bank["temperature"]),
            "top_k": int(path_bank["top_k"]),
            "top_p": float(path_bank["top_p"]),
            "minimum_valid_paths_per_case": int(path_bank["minimum_valid_paths_per_case"]),
        }

    def _cache_namespace(self, *, smoke: bool) -> str:
        model = self.config["model"]
        return sha256_json(
            {
                "strategy_version": STRATEGY_VERSION,
                "data_fingerprint": self.cohort.data_fingerprint,
                "tokenizer_revision": model["tokenizer_revision"],
                "predictor_revision": model["predictor_revision"],
                "smoke": bool(smoke),
                **self._sampling(smoke=smoke),
            }
        )

    def _cache_root(self, *, smoke: bool) -> Path:
        return Path(self.config["path_bank"]["artifact_root"]) / self._cache_namespace(smoke=smoke)

    def _planned_shards(
        self, cases: Sequence[TargetOnlyCase]
    ) -> tuple[tuple[str, tuple[TargetOnlyCase, ...]], ...]:
        by_length: defaultdict[int, list[TargetOnlyCase]] = defaultdict(list)
        for case in cases:
            by_length[int(case.pred_len)].append(case)
        case_count = int(self.config["path_bank"]["shard_case_count"])
        planned: list[tuple[str, tuple[TargetOnlyCase, ...]]] = []
        for pred_len, grouped in sorted(by_length.items()):
            ordered = sorted(
                grouped,
                key=lambda case: (case.target_end_day, case.target_contract_id, case.case_key),
            )
            for offset in range(0, len(ordered), case_count):
                index = offset // case_count
                planned.append(
                    (f"length_{pred_len:02d}/shard_{index:04d}", tuple(ordered[offset : offset + case_count])))
        return tuple(planned)

    def _new_manifest(self, *, smoke: bool, cases: Sequence[TargetOnlyCase]) -> dict[str, object]:
        cache_root = self._cache_root(smoke=smoke)
        try:
            cache_root_relative = str(cache_root.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError as exc:
            raise V7ExperimentError("V7 path-bank artifact root must remain inside the repository") from exc
        return {
            **self._metadata("p1_path_bank", smoke=smoke),
            "schema_version": PATH_BANK_SCHEMA_VERSION,
            "cache_namespace": self._cache_namespace(smoke=smoke),
            "artifact_cache_root_relative": cache_root_relative,
            "case_keys_sha256": sha256_case_keys(case.case_key for case in cases),
            "planned_case_count": int(len(cases)),
            "sampling": self._sampling(smoke=smoke),
            "shards": [],
            "determinism_checks": [],
            "complete": False,
        }

    def _manifest_path(self, *, smoke: bool) -> Path:
        return self.run_dir / self._stage_name(smoke=smoke) / "manifest.json"

    def _failures_path(self, *, smoke: bool) -> Path:
        return self.run_dir / self._stage_name(smoke=smoke) / "failures.json"

    def _validate_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        smoke: bool,
        cases: Sequence[TargetOnlyCase],
    ) -> None:
        expected = self._new_manifest(smoke=smoke, cases=cases)
        for key in (
            "strategy_version", "phase", "run_id", "result_scope", "production_eligible",
            "data_fingerprint", "tokenizer_revision", "predictor_revision", "risk_label_version",
            "smoke", "schema_version", "cache_namespace", "artifact_cache_root_relative",
            "case_keys_sha256", "planned_case_count", "sampling",
        ):
            if manifest.get(key) != expected.get(key):
                raise V7ExperimentError(f"V7 P1 manifest does not match active protocol at {key}")
        if not isinstance(manifest.get("shards"), list):
            raise V7ExperimentError("V7 P1 manifest shards must be a list")

    def _load_or_create_manifest(
        self, *, smoke: bool, cases: Sequence[TargetOnlyCase]
    ) -> tuple[dict[str, Any], Path, Path]:
        manifest_path = self._manifest_path(smoke=smoke)
        failures_path = self._failures_path(smoke=smoke)
        cache_root = self._cache_root(smoke=smoke)
        cache_root.mkdir(parents=True, exist_ok=True)
        if manifest_path.is_file():
            manifest = read_json(manifest_path, label="V7 P1 path-bank manifest")
            self._validate_manifest(manifest, smoke=smoke, cases=cases)
        else:
            manifest = self._new_manifest(smoke=smoke, cases=cases)
            write_json(manifest_path, manifest)
        if not failures_path.exists():
            write_json(failures_path, {**self._metadata("p1_path_bank", smoke=smoke), "failures": []})
        return manifest, manifest_path, failures_path

    @staticmethod
    def _shard_path(cache_root: Path, shard_id: str) -> Path:
        return cache_root / f"{shard_id}.npz"

    def _record_failure(
        self,
        failures_path: Path,
        *,
        smoke: bool,
        case: TargetOnlyCase,
        error: BaseException,
    ) -> None:
        current = read_json(failures_path, label="V7 P1 failures")
        failures = current.get("failures", [])
        if not isinstance(failures, list):
            failures = []
        entry = {
            "case_key": case.case_key,
            "target_contract_id": case.target_contract_id,
            "product": case.product,
            "pred_len": int(case.pred_len),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failures = [value for value in failures if value.get("case_key") != case.case_key]
        failures.append(entry)
        write_json(
            failures_path,
            {**self._metadata("p1_path_bank", smoke=smoke), "failures": failures},
        )

    def _existing_entries(
        self,
        manifest: dict[str, Any],
        *,
        cache_root: Path,
    ) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for raw_entry in manifest["shards"]:
            if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("shard_id"), str):
                raise V7ExperimentError("V7 P1 manifest has an invalid shard entry")
            shard_id = str(raw_entry["shard_id"])
            if shard_id in entries:
                raise V7ExperimentError(f"V7 P1 manifest duplicates shard {shard_id}")
            validate_shard_entry(raw_entry, artifact_root=cache_root)
            entries[shard_id] = raw_entry
        return entries

    def _recover_orphan_shard(
        self,
        *,
        cache_root: Path,
        shard_id: str,
        cases: Sequence[TargetOnlyCase],
        smoke: bool,
    ) -> dict[str, Any] | None:
        path = self._shard_path(cache_root, shard_id)
        if not path.exists():
            return None
        _, metadata = read_shard(path)
        if (
            metadata.get("cache_namespace") != self._cache_namespace(smoke=smoke)
            or metadata.get("shard_id") != shard_id
            or tuple(metadata.get("case_keys", ())) != tuple(case.case_key for case in cases)
        ):
            raise V7ExperimentError(f"V7 orphan shard does not match planned shard: {path}")
        entry = make_shard_entry(path, artifact_root=cache_root, case_keys=[case.case_key for case in cases])
        entry["shard_id"] = shard_id
        return entry

    def _case_cache_key(self, case: TargetOnlyCase, *, smoke: bool) -> str:
        sampling = self._sampling(smoke=smoke)
        model = self.config["model"]
        return cache_key_for_case(
            strategy_version=STRATEGY_VERSION,
            data_fingerprint=self.cohort.data_fingerprint,
            case_key=case.case_key,
            tokenizer_revision=str(model["tokenizer_revision"]),
            predictor_revision=str(model["predictor_revision"]),
            sample_count=int(sampling["sample_count"]),
            temperature=float(sampling["temperature"]),
            top_k=int(sampling["top_k"]),
            top_p=float(sampling["top_p"]),
        )

    def _generate_shard(
        self,
        *,
        tokenizer: KronosTokenizer,
        predictor: Kronos,
        model_hashes: Mapping[str, str],
        cache_root: Path,
        shard_id: str,
        cases: Sequence[TargetOnlyCase],
        smoke: bool,
    ) -> dict[str, Any]:
        sampling = self._sampling(smoke=smoke)
        outputs: list[tuple[dict[str, np.ndarray], Mapping[str, object]]] = []
        for case in cases:
            cache_key = self._case_cache_key(case, smoke=smoke)
            seed = sampling_seed(cache_key)
            payload, record = generate_case_output(
                tokenizer,
                predictor,
                case,
                device=self.device,
                max_context=int(self.config["model"]["max_context"]),
                clip=float(self.config["data"]["clip"]),
                epsilon=float(self.config["data"]["normalization_epsilon"]),
                sample_count=int(sampling["sample_count"]),
                temperature=float(sampling["temperature"]),
                top_k=int(sampling["top_k"]),
                top_p=float(sampling["top_p"]),
                seed=seed,
            )
            outputs.append(
                (
                    payload,
                    {
                        **record,
                        "cache_key": cache_key,
                        "sampling_seed": int(seed),
                        "tokenizer_revision": self.config["model"]["tokenizer_revision"],
                        "predictor_revision": self.config["model"]["predictor_revision"],
                    },
                )
            )
        arrays = build_shard_arrays(outputs)
        metadata = {
            "schema_version": PATH_BANK_SCHEMA_VERSION,
            "cache_namespace": self._cache_namespace(smoke=smoke),
            "shard_id": shard_id,
            "case_keys": [case.case_key for case in cases],
            "case_metadata": [dict(metadata) for _, metadata in outputs],
            "sampling": sampling,
            "model_parameter_sha256": dict(model_hashes),
        }
        path = self._shard_path(cache_root, shard_id)
        write_shard_atomic(path, arrays=arrays, metadata=metadata)
        entry = make_shard_entry(path, artifact_root=cache_root, case_keys=[case.case_key for case in cases])
        entry["shard_id"] = shard_id
        return entry

    def _determinism_check(
        self,
        *,
        manifest: dict[str, Any],
        cache_root: Path,
        cases: Sequence[TargetOnlyCase],
        tokenizer: KronosTokenizer,
        predictor: Kronos,
        smoke: bool,
    ) -> list[dict[str, object]]:
        needed = int(self.config["path_bank"]["determinism_check_cases"])
        by_key = {case.case_key: case for case in cases}
        stored: dict[str, tuple[dict[str, np.ndarray], Mapping[str, object], int]] = {}
        for entry in manifest["shards"]:
            arrays, metadata = read_shard(
                self._resolve_artifact_path(entry["relative_path"], cache_root=cache_root)
            )
            case_metadata = metadata.get("case_metadata")
            if not isinstance(case_metadata, list):
                raise V7ExperimentError("V7 shard lacks per-case metadata for determinism check")
            for index, record in enumerate(case_metadata):
                if isinstance(record, Mapping):
                    stored[str(record.get("case_key"))] = (arrays, record, index)
        selected_keys = sorted(by_key)[:needed]
        if len(selected_keys) != needed:
            raise V7ExperimentError("V7 path bank cannot provide required determinism checks")
        checks: list[dict[str, object]] = []
        sampling = self._sampling(smoke=smoke)
        for key in selected_keys:
            arrays, record, index = stored.get(key, (None, None, None))  # type: ignore[assignment]
            if arrays is None or record is None or index is None:
                raise V7ExperimentError(f"V7 determinism check cache entry is missing: {key}")
            case = by_key[key]
            payload, regenerated = generate_case_output(
                tokenizer,
                predictor,
                case,
                device=self.device,
                max_context=int(self.config["model"]["max_context"]),
                clip=float(self.config["data"]["clip"]),
                epsilon=float(self.config["data"]["normalization_epsilon"]),
                sample_count=int(sampling["sample_count"]),
                temperature=float(sampling["temperature"]),
                top_k=int(sampling["top_k"]),
                top_p=float(sampling["top_p"]),
                seed=int(record["sampling_seed"]),
            )
            passed = bool(
                regenerated["output_signature_sha256"] == record.get("output_signature_sha256")
                and np.array_equal(payload["raw_paths"], arrays["raw_paths"][index])
                and np.array_equal(payload["hidden"], arrays["hidden"][index])
                and np.array_equal(payload["context_mean"], arrays["context_mean"][index])
                and np.array_equal(payload["context_std"], arrays["context_std"][index])
                and np.array_equal(payload["target_timestamps"], arrays["target_timestamps"][index])
            )
            checks.append(
                {
                    "case_key": key,
                    "cache_key": record.get("cache_key"),
                    "passed": passed,
                    "expected_output_signature_sha256": record.get("output_signature_sha256"),
                    "observed_output_signature_sha256": regenerated["output_signature_sha256"],
                }
            )
        return checks

    def p1_path_bank(self, *, resume: bool = False, smoke: bool = False) -> Path:
        """Create (or verify/reuse) P1's raw, unmodified unique-case cache."""

        cases = self._selected_cases(smoke=smoke)
        manifest, manifest_path, failures_path = self._load_or_create_manifest(
            smoke=smoke, cases=cases
        )
        cache_root = self._cache_root(smoke=smoke)
        planned = self._planned_shards(cases)
        existing = self._existing_entries(manifest, cache_root=cache_root)
        planned_ids = {shard_id for shard_id, _ in planned}
        unexpected = sorted(set(existing).difference(planned_ids))
        if unexpected:
            raise V7ExperimentError(f"V7 P1 manifest has unplanned shards: {unexpected[:3]!r}")
        if manifest.get("complete") and set(existing) == planned_ids:
            if not resume:
                return manifest_path
        elif manifest.get("complete"):
            raise V7ExperimentError("V7 P1 manifest claims complete but misses planned shards")
        if not resume and existing:
            raise V7ExperimentError(
                "V7 P1 has partial cache state; rerun with --resume after inspecting failures"
            )
        if manifest.get("complete") and set(existing) == planned_ids and resume:
            tokenizer, predictor, _ = self._load_models()
            try:
                checks = self._determinism_check(
                    manifest=manifest,
                    cache_root=cache_root,
                    cases=cases,
                    tokenizer=tokenizer,
                    predictor=predictor,
                    smoke=smoke,
                )
                if not all(bool(check.get("passed")) for check in checks):
                    raise V7ExperimentError("V7 completed cache failed its resume determinism recheck")
            finally:
                self._release_models(tokenizer, predictor)
            return manifest_path

        stale_failures = read_json(failures_path, label="V7 P1 failures").get("failures", [])
        if stale_failures:
            raise V7ExperimentError(
                "V7 P1 has recorded failed cases; preserve the evidence and use a new run ID "
                "rather than silently retrying with the same formal run"
            )

        for shard_id, shard_cases in planned:
            if shard_id in existing:
                continue
            orphan = self._recover_orphan_shard(
                cache_root=cache_root,
                shard_id=shard_id,
                cases=shard_cases,
                smoke=smoke,
            )
            if orphan is not None:
                manifest["shards"].append(orphan)
                manifest["shards"].sort(key=lambda entry: str(entry["shard_id"]))
                write_json(manifest_path, manifest)
                existing[shard_id] = orphan

        missing = [(shard_id, shard_cases) for shard_id, shard_cases in planned if shard_id not in existing]
        tokenizer: KronosTokenizer | None = None
        predictor: Kronos | None = None
        model_hashes: dict[str, str] | None = None
        try:
            if missing or not manifest.get("determinism_checks"):
                tokenizer, predictor, model_hashes = self._load_models()
            for shard_id, shard_cases in missing:
                try:
                    entry = self._generate_shard(
                        tokenizer=tokenizer,  # type: ignore[arg-type]
                        predictor=predictor,  # type: ignore[arg-type]
                        model_hashes=model_hashes or {},
                        cache_root=cache_root,
                        shard_id=shard_id,
                        cases=shard_cases,
                        smoke=smoke,
                    )
                except BaseException as exc:
                    failed_case = shard_cases[0]
                    self._record_failure(
                        failures_path, smoke=smoke, case=failed_case, error=exc
                    )
                    raise V7ExperimentError(
                        f"V7 P1 raw-path generation failed in {shard_id}; failure was recorded"
                    ) from exc
                manifest["shards"].append(entry)
                manifest["shards"].sort(key=lambda item: str(item["shard_id"]))
                write_json(manifest_path, manifest)
                existing[shard_id] = entry
            if tokenizer is not None and predictor is not None:
                _assert_frozen(tokenizer, label="tokenizer after P1 generation")
                _assert_frozen(predictor, label="predictor after P1 generation")
                after_hashes = {
                    "tokenizer_parameter_sha256": _module_sha256(tokenizer),
                    "predictor_parameter_sha256": _module_sha256(predictor),
                }
                if model_hashes is not None and after_hashes != model_hashes:
                    raise V7ExperimentError("Frozen V7 model parameter hash changed during P1")
                checks = self._determinism_check(
                    manifest=manifest,
                    cache_root=cache_root,
                    cases=cases,
                    tokenizer=tokenizer,
                    predictor=predictor,
                    smoke=smoke,
                )
                manifest["determinism_checks"] = checks
            elif not manifest.get("determinism_checks"):
                raise V7ExperimentError("V7 P1 cache lacks required determinism checks")
        finally:
            if tokenizer is not None and predictor is not None:
                self._release_models(tokenizer, predictor)

        completed_keys = [
            key for entry in manifest["shards"] for key in entry.get("case_keys", [])
        ]
        if len(completed_keys) != len(cases) or set(completed_keys) != {case.case_key for case in cases}:
            raise V7ExperimentError("V7 P1 cache manifest does not cover the planned unique cases exactly")
        manifest["completed_case_count"] = int(len(completed_keys))
        manifest["completed_case_keys_sha256"] = sha256_case_keys(completed_keys)
        manifest["complete"] = True
        write_json(manifest_path, manifest)
        if not smoke:
            # Baseline selection is a separate formal P1 sub-stage.  A stale
            # gate from a prior completed selection must never coexist with a
            # newly materialized cache under the same run ID.
            stale_gate = self.run_dir / "p1_gate.json"
            if stale_gate.exists():
                raise V7ExperimentError(
                    "V7 formal P1 gate already exists for this run ID; preserve it and use a new run ID"
                )
        print(
            json.dumps(
                {
                    "stage": self._stage_name(smoke=smoke),
                    "manifest": str(manifest_path),
                    "cache_root": str(cache_root),
                    "cases": len(cases),
                    "smoke": smoke,
                },
                ensure_ascii=False,
            )
        )
        return manifest_path

    def p1_smoke_gate(self) -> Path:
        """Validate smoke cache artifacts and emit the deliberately non-unlocking gate."""

        manifest, _ = self._load_path_manifest(smoke=True)
        manifest_path = self._manifest_path(smoke=True)
        checks = manifest.get("determinism_checks", [])
        failures = read_json(self._failures_path(smoke=True), label="V7 P1 smoke failures").get(
            "failures", []
        )
        gate = {
            **self._metadata("p1", smoke=True),
            "available": True,
            "allows_next_phase": False,
            "failed_condition_ids": ["smoke_run_cannot_unlock_formal_p2"],
            "path_bank_manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
            "engineering_checks": {
                "manifest_complete": bool(manifest.get("complete")),
                "determinism_checks": checks,
                "determinism_passed": bool(checks) and all(bool(check.get("passed")) for check in checks),
                "failure_count": len(failures) if isinstance(failures, list) else None,
            },
            "note": "Smoke validates cache engineering only; it is not a formal P1 gate.",
        }
        path = self.run_dir / "p1_smoke_gate.json"
        write_json(path, gate)
        return path

    def _load_path_manifest(self, *, smoke: bool) -> tuple[dict[str, Any], Path]:
        cases = self._selected_cases(smoke=smoke)
        manifest_path = self._manifest_path(smoke=smoke)
        if not manifest_path.is_file():
            raise V7ExperimentError(f"V7 P1 path bank is missing: {manifest_path}")
        manifest = read_json(manifest_path, label="V7 P1 path-bank manifest")
        self._validate_manifest(manifest, smoke=smoke, cases=cases)
        if not bool(manifest.get("complete")):
            raise V7ExperimentError("V7 P1 path bank is not complete; resume it before baselines")
        cache_root = self._cache_root(smoke=smoke)
        entries = self._existing_entries(manifest, cache_root=cache_root)
        expected = {shard_id for shard_id, _ in self._planned_shards(cases)}
        if set(entries) != expected:
            raise V7ExperimentError("V7 P1 path bank does not have exactly the planned shards")
        checks = manifest.get("determinism_checks")
        if not isinstance(checks, list) or not checks or not all(
            isinstance(check, Mapping) and bool(check.get("passed")) for check in checks
        ):
            raise V7ExperimentError("V7 P1 path bank lacks passing determinism checks")
        return manifest, cache_root

    def _resolve_artifact_path(self, relative_path: object, *, cache_root: Path) -> Path:
        path = (cache_root / str(relative_path)).resolve()
        root = cache_root.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise V7ExperimentError("V7 cache artifact path escapes its cache root") from exc
        return path

    def _derive_path_features(
        self,
        *,
        manifest: Mapping[str, Any],
        cache_root: Path,
    ) -> dict[str, object]:
        """Derive raw summaries once and fold-specific crossings from fit thresholds."""

        p0_by_key = {
            str(key): group.copy()
            for key, group in self.fold_records.groupby("case_key", sort=False)
        }
        case_rows: list[dict[str, object]] = []
        fold_rows: list[dict[str, object]] = []
        global_valid = 0
        global_total = 0
        seen: set[str] = set()
        epsilon = float(self.config["data"]["normalization_epsilon"])
        minimum_valid = int(self.config["path_bank"]["minimum_valid_paths_per_case"])
        for entry in manifest["shards"]:
            arrays, shard_metadata = read_shard(
                self._resolve_artifact_path(entry["relative_path"], cache_root=cache_root)
            )
            case_metadata = shard_metadata.get("case_metadata")
            if not isinstance(case_metadata, list) or len(case_metadata) != len(entry["case_keys"]):
                raise V7ExperimentError("V7 shard case metadata cannot be matched to manifest keys")
            raw_paths = arrays["raw_paths"]
            hidden = arrays["hidden"]
            if raw_paths.shape[0] != len(case_metadata) or hidden.shape[0] != len(case_metadata):
                raise V7ExperimentError("V7 shard arrays have inconsistent case dimensions")
            for index, raw_metadata in enumerate(case_metadata):
                if not isinstance(raw_metadata, Mapping):
                    raise V7ExperimentError("V7 shard has malformed case metadata")
                metadata = dict(raw_metadata)
                key = str(metadata.get("case_key"))
                if key in seen or key not in p0_by_key:
                    raise V7ExperimentError(f"V7 cache has duplicate or unknown case key: {key}")
                seen.add(key)
                paths = np.asarray(raw_paths[index], dtype=np.float64)
                hidden_value = np.asarray(hidden[index], dtype=np.float64)
                timestamps = np.asarray(arrays["target_timestamps"][index], dtype=np.int64)
                if hidden_value.ndim != 1 or not np.isfinite(hidden_value).all():
                    raise V7ExperimentError(f"V7 cached hidden state is invalid: {key}")
                if timestamps.shape != (int(metadata["pred_len"]),) or (
                    len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0)
                ):
                    raise V7ExperimentError(f"V7 cached target timestamps are invalid: {key}")
                validity = validate_raw_paths(paths)
                if validity.valid_count != int(metadata.get("valid_path_count", -1)):
                    raise V7ExperimentError(f"V7 cached valid-path count changed: {key}")
                if validity.invalid_reason_counts != metadata.get("invalid_path_reason_counts", {}):
                    raise V7ExperimentError(f"V7 cached invalid-path reasons changed: {key}")
                global_valid += validity.valid_count
                global_total += validity.total_count
                records = p0_by_key[key]
                origin_close_values = records["origin_close"].to_numpy(dtype=np.float64)
                scale_values = records["context_horizon_scale"].to_numpy(dtype=np.float64)
                if not np.allclose(origin_close_values, origin_close_values[0]) or not np.allclose(scale_values, scale_values[0]):
                    raise V7ExperimentError(f"V7 P0 context scale differs across folds for {key}")
                origin_close = float(origin_close_values[0])
                scale = float(scale_values[0])
                if not np.isfinite(origin_close) or origin_close <= 0.0 or not np.isfinite(scale) or scale <= 0.0:
                    raise V7ExperimentError(f"V7 P0 context scale is invalid: {key}")
                valid_paths = paths[validity.valid_mask]
                if len(valid_paths):
                    lows = valid_paths[:, :, 2]
                    highs = valid_paths[:, :, 1]
                    closes = valid_paths[:, :, 3]
                    long_mae = np.maximum(
                        0.0, -np.min(np.log(lows / origin_close), axis=1)
                    ) / scale
                    short_mae = np.maximum(
                        0.0, np.max(np.log(highs / origin_close), axis=1)
                    ) / scale
                    future_returns = np.diff(
                        np.log(np.concatenate([np.full((len(valid_paths), 1), origin_close), closes], axis=1)),
                        axis=1,
                    )
                    future_scale = np.sqrt(np.sum(np.square(future_returns), axis=1))
                    future_vol_ratio = np.log((future_scale + epsilon) / (scale + epsilon))
                    final_returns = closes[:, -1] / origin_close - 1.0
                    up_fraction = float(np.mean(final_returns > 0.0))
                    down_fraction = float(np.mean(final_returns < 0.0))
                    direction_entropy = float(
                        -sum(
                            probability * np.log(probability)
                            for probability in (up_fraction, down_fraction)
                            if probability > 0.0
                        )
                    )
                    summary = {
                        "long_mae_median": float(np.quantile(long_mae, 0.5, method="linear")),
                        "long_mae_p80": float(np.quantile(long_mae, 0.8, method="linear")),
                        "long_mae_p95": float(np.quantile(long_mae, 0.95, method="linear")),
                        "short_mae_median": float(np.quantile(short_mae, 0.5, method="linear")),
                        "short_mae_p80": float(np.quantile(short_mae, 0.8, method="linear")),
                        "short_mae_p95": float(np.quantile(short_mae, 0.95, method="linear")),
                        "future_vol_ratio_median": float(np.quantile(future_vol_ratio, 0.5, method="linear")),
                        "future_vol_ratio_p90": float(np.quantile(future_vol_ratio, 0.9, method="linear")),
                        "path_return_iqr": float(np.quantile(final_returns, 0.75, method="linear") - np.quantile(final_returns, 0.25, method="linear")),
                        "direction_entropy": direction_entropy,
                    }
                else:
                    long_mae = short_mae = np.asarray([], dtype=np.float64)
                    summary = {
                        name: None
                        for name in (
                            "long_mae_median", "long_mae_p80", "long_mae_p95",
                            "short_mae_median", "short_mae_p80", "short_mae_p95",
                            "future_vol_ratio_median", "future_vol_ratio_p90",
                            "path_return_iqr", "direction_entropy",
                        )
                    }
                eligible = bool(validity.valid_count >= minimum_valid)
                case_rows.append(
                    {
                        "case_key": key,
                        "product": metadata["product"],
                        "pred_len": int(metadata["pred_len"]),
                        "valid_path_count": validity.valid_count,
                        "sample_count": validity.total_count,
                        "valid_path_fraction": float(validity.valid_count / validity.total_count),
                        "invalid_path_rate": float(1.0 - validity.valid_count / validity.total_count),
                        "eligible_for_risk": eligible,
                        "hidden_dim": int(hidden_value.shape[0]),
                        **summary,
                    }
                )
                for row in records.to_dict("records"):
                    long_probability = (
                        float(np.mean(long_mae >= float(row["long_tail_threshold"])))
                        if eligible else None
                    )
                    short_probability = (
                        float(np.mean(short_mae >= float(row["short_tail_threshold"])))
                        if eligible else None
                    )
                    fold_rows.append(
                        {
                            "case_key": key,
                            "fold_id": str(row["fold_id"]),
                            "split": str(row["split"]),
                            "p_long": long_probability,
                            "p_short": short_probability,
                            "valid_path_count": validity.valid_count,
                            "sample_count": validity.total_count,
                            "eligible_for_risk": eligible,
                            "invalid_path_rate": float(1.0 - validity.valid_count / validity.total_count),
                            **summary,
                        }
                    )
        expected_keys = set(self.unique_case_keys)
        if seen != expected_keys:
            raise V7ExperimentError("V7 P1 path bank does not cover canonical P0 case keys exactly")
        if global_total == 0:
            raise V7ExperimentError("V7 P1 path bank has no raw paths")
        return {
            "case_summaries": case_rows,
            "fold_features": fold_rows,
            "statistics": {
                "unique_cases": len(case_rows),
                "fold_records": len(fold_rows),
                "global_valid_path_count": int(global_valid),
                "global_path_count": int(global_total),
                "global_valid_path_fraction": float(global_valid / global_total),
                "abstain_path_quality_cases": int(sum(not bool(row["eligible_for_risk"]) for row in case_rows)),
                "eligible_cases": int(sum(bool(row["eligible_for_risk"]) for row in case_rows)),
            },
        }

    def _write_path_summaries(
        self,
        *,
        manifest_path: Path,
        features: Mapping[str, object],
    ) -> Path:
        destination = self.run_dir / "p1_path_bank" / "path_summaries.json"
        payload = {
            **self._metadata("p1_path_bank", smoke=False),
            "path_bank_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_path(manifest_path),
            },
            **features,
        }
        return write_json(destination, payload)

    @staticmethod
    def _path_risk_non_degenerate(records: pd.DataFrame, *, fold_id: str, split: str, side: str) -> dict[str, object]:
        selected = records.loc[
            (records["fold_id"].astype(str) == str(fold_id))
            & (records["split"].astype(str) == str(split))
            & records["eligible_for_risk"].astype(bool)
        ]
        values = selected[f"p_{side}"].to_numpy(dtype=np.float64)
        passed = bool(
            len(values)
            and np.isfinite(values).all()
            and not np.all(values == 0.0)
            and not np.all(values == 1.0)
            and float(np.var(values)) > 0.0
        )
        return {
            "condition_id": f"{fold_id}:{split}:{side}_raw_path_risk_non_degenerate",
            "passed": passed,
            "eligible_cases": int(len(values)),
            "minimum": float(np.min(values)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
            "variance": float(np.var(values)) if len(values) else None,
        }

    def p1_baselines(self, *, smoke: bool = False) -> Path:
        """Select frozen simple baselines on validation, render P1 evidence, and gate P2."""

        if smoke:
            raise V7ExperimentError(
                "V7 smoke path banks verify engineering only and cannot run formal P1 baseline selection"
            )
        manifest, cache_root = self._load_path_manifest(smoke=False)
        manifest_path = self._manifest_path(smoke=False)
        features = self._derive_path_features(manifest=manifest, cache_root=cache_root)
        summaries_path = self._write_path_summaries(
            manifest_path=manifest_path, features=features
        )
        case_summaries = pd.DataFrame(features["case_summaries"])
        fold_features = pd.DataFrame(features["fold_features"])
        base_records = attach_context_features(self.fold_records, cases=self.unique_cases)
        records = base_records.merge(
            fold_features,
            on=["case_key", "fold_id", "split"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_path"),
        )
        if records["eligible_for_risk"].isna().any():
            raise V7ExperimentError("V7 P1 path summaries do not cover all P0 fold records")
        selections: dict[str, object] = {}
        selected_validation_frames: list[pd.DataFrame] = []
        selected_baseline_by_fold: dict[str, str] = {}
        selected_parameters: dict[str, object] = {}
        for fold_id, fold_records in records.groupby("fold_id", sort=True):
            eligible = fold_records.loc[fold_records["eligible_for_risk"].astype(bool)].copy()
            fit = eligible.loc[eligible["split"].astype(str) == "fit"].copy()
            validation = eligible.loc[
                eligible["split"].astype(str) == "inner_validation"
            ].copy()
            if fit.empty or validation.empty:
                raise V7ExperimentError(f"V7 P1 lacks eligible fit/validation cases for {fold_id}")
            validation_by_name, parameters = fit_and_predict_baselines(
                fit_records=fit,
                destination_records=validation,
                config=self.config,
                path_features=fold_features,
            )
            selected_name, selection = choose_baseline(
                validation_by_name,
                selection_order=tuple(self.config["baselines"]["selection_order"]),
                gate_products=tuple(self.config["data"]["gate_products"]),
                all_products=tuple(self.config["data"]["products"]),
            )
            selected = validation_by_name[selected_name].copy()
            selected["baseline_name"] = selected_name
            selected_validation_frames.append(selected)
            selected_baseline_by_fold[str(fold_id)] = selected_name
            selected_parameters[str(fold_id)] = parameters
            selections[str(fold_id)] = {
                "selected_baseline": selected_name,
                "fit_cases": int(len(fit)),
                "validation_cases": int(len(validation)),
                "path_quality_abstain_fit_cases": int(
                    len(fold_records.loc[
                        (fold_records["split"].astype(str) == "fit")
                        & ~fold_records["eligible_for_risk"].astype(bool)
                    ])
                ),
                "path_quality_abstain_validation_cases": int(
                    len(fold_records.loc[
                        (fold_records["split"].astype(str) == "inner_validation")
                        & ~fold_records["eligible_for_risk"].astype(bool)
                    ])
                ),
                **selection,
                "selected_validation_metrics": classification_metrics(selected),
            }
        selected_validation = pd.concat(selected_validation_frames, ignore_index=True)
        selection_path = self.run_dir / "p1_baselines" / "selection.json"
        selection_payload = {
            **self._metadata("p1_baselines", smoke=False),
            "path_bank_manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
            "path_summaries": {"path": str(summaries_path), "sha256": sha256_path(summaries_path)},
            "selection_by_fold": selections,
            "frozen_baseline_parameters_by_fold": selected_parameters,
        }
        if selection_path.exists():
            existing = read_json(selection_path, label="V7 P1 baseline selection")
            if sha256_json(existing) != sha256_json(selection_payload):
                raise V7ExperimentError(
                    "V7 P1 baseline selection is immutable and differs from this attempt; use a new run ID"
                )
        else:
            write_json(selection_path, selection_payload)
        selected_validation_path = self.run_dir / "p1_baselines" / "selected_validation_records.json"
        write_json(
            selected_validation_path,
            {
                **self._metadata("p1_baselines", smoke=False),
                "records": selected_validation.to_dict("records"),
            },
        )
        figures = render_p1_plots(
            path_records=case_summaries,
            fold_path_records=fold_features,
            validation_records=selected_validation,
            selected_baselines=selected_baseline_by_fold,
            output_dir=self.run_dir / "p1_baselines" / "figures",
            metadata={
                **self._metadata("p1_baselines", smoke=False),
                "selection_path": str(selection_path),
            },
        )
        conditions: list[dict[str, object]] = []
        statistics = dict(features["statistics"])
        conditions.append(
            {
                "condition_id": "global_finite_ohlc_path_validity",
                "passed": float(statistics["global_valid_path_fraction"])
                >= float(self.config["path_bank"]["minimum_valid_path_fraction"]),
                "observed": statistics["global_valid_path_fraction"],
                "required": self.config["path_bank"]["minimum_valid_path_fraction"],
            }
        )
        conditions.append(
            {
                "condition_id": "all_non_abstain_cases_meet_minimum_valid_paths",
                "passed": bool(
                    (
                        case_summaries.loc[case_summaries["eligible_for_risk"].astype(bool), "valid_path_count"]
                        >= int(self.config["path_bank"]["minimum_valid_paths_per_case"])
                    ).all()
                ),
                "minimum_required": int(self.config["path_bank"]["minimum_valid_paths_per_case"]),
                "eligible_cases": int(statistics["eligible_cases"]),
                "path_quality_abstain_cases": int(statistics["abstain_path_quality_cases"]),
            }
        )
        for fold_id in sorted(records["fold_id"].astype(str).unique()):
            for split in ("fit", "inner_validation"):
                for side in ("long", "short"):
                    conditions.append(
                        self._path_risk_non_degenerate(
                            records, fold_id=fold_id, split=split, side=side
                        )
                    )
        determinism_checks = manifest.get("determinism_checks", [])
        conditions.append(
            {
                "condition_id": "same_cache_key_determinism_check",
                "passed": bool(determinism_checks) and all(
                    bool(check.get("passed")) for check in determinism_checks
                ),
                "checks": determinism_checks,
            }
        )
        evaluation_keys = set(
            self.fold_records.loc[
                self.fold_records["split"].astype(str) == "evaluation", "case_key"
            ].astype(str)
        )
        conditions.append(
            {
                "condition_id": "evaluation_contexts_generated_without_evaluation_metric_selection",
                "passed": evaluation_keys.issubset(set(case_summaries["case_key"].astype(str))),
                "evaluation_unique_cases": int(len(evaluation_keys)),
                "note": "P1 reports no evaluation prediction-performance metric.",
            }
        )
        figure_paths = figures.as_dict()
        conditions.append(
            {
                "condition_id": "path_manifest_failures_and_required_figures_present",
                "passed": all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in figure_paths.values())
                and self._failures_path(smoke=False).is_file()
                and manifest_path.is_file(),
                "figures": figure_paths,
                "failures_path": str(self._failures_path(smoke=False)),
            }
        )
        failures_payload = read_json(self._failures_path(smoke=False), label="V7 P1 failures")
        conditions.append(
            {
                "condition_id": "no_raw_path_generation_failures",
                "passed": not bool(failures_payload.get("failures")),
                "failure_count": len(failures_payload.get("failures", [])),
            }
        )
        failed = [str(condition["condition_id"]) for condition in conditions if not bool(condition["passed"])]
        gate = {
            **self._metadata("p1", smoke=False),
            "available": True,
            "allows_next_phase": not failed,
            "failed_condition_ids": failed,
            "conditions": conditions,
            "path_bank_manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
            "path_summaries": {"path": str(summaries_path), "sha256": sha256_path(summaries_path)},
            "baseline_selection": {"path": str(selection_path), "sha256": sha256_path(selection_path)},
            "selected_validation_records": {
                "path": str(selected_validation_path),
                "sha256": sha256_path(selected_validation_path),
            },
            "figures": figure_paths,
            "statistics": statistics,
        }
        gate_path = self.run_dir / "p1_gate.json"
        write_json(gate_path, gate)
        write_json(self.results_dir / "p1_gate.json", gate)
        report = [
            "# V7 P1 冻结路径与简单风险基线门控",
            "",
            f"- 结论：**{'通过' if gate['allows_next_phase'] else '未通过'}**",
            f"- 原始路径缓存：{statistics['unique_cases']} 个唯一 case",
            f"- 全局有效路径比例：{statistics['global_valid_path_fraction']:.4%}",
            f"- path-quality abstain：{statistics['abstain_path_quality_cases']} 个 case",
            "- P1 的基线只使用 inner validation 选择；未输出 outer evaluation 预测性能指标。",
            "",
            "## 每折选中的基线",
            "",
            "| fold | selected baseline | validation cases |",
            "|---|---|---:|",
        ]
        for fold_id, payload in selections.items():
            report.append(
                f"| {fold_id} | {payload['selected_baseline']} | {payload['validation_cases']} |"
            )
        report.extend(["", "## 失败项", ""])
        report.extend([f"- `{item}`" for item in failed] if failed else ["- 无。"])
        report.extend(
            [
                "",
                "本结果是 observed-contract 回看研究证据，`production_eligible: false`。",
            ]
        )
        report_text = "\n".join(report) + "\n"
        (self.run_dir / "P1_REPORT.md").write_text(report_text, encoding="utf-8")
        (self.results_dir / "P1_REPORT.md").write_text(report_text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "p1_gate": str(gate_path),
                    "allows_next_phase": gate["allows_next_phase"],
                    "failed_condition_ids": failed,
                },
                ensure_ascii=False,
            )
        )
        return gate_path

    def audit(self) -> Path:
        """Validate and record the only permissible P0 predecessor without rerunning it."""

        destination = self.run_dir / "canonical_p0_verified.json"
        write_json(
            destination,
            {
                **self._metadata("audit", smoke=False),
                "canonical_p0": {
                    "run_id": CANONICAL_P0_RUN_ID,
                    "gate_path": str(self.p0_gate_path),
                    "gate_sha256": sha256_path(self.p0_gate_path),
                    "audit_path": str(self.p0_audit_path),
                    "audit_sha256": sha256_path(self.p0_audit_path),
                    "fold_label_records_path": str(self.p0_records_path),
                    "fold_label_records_sha256": sha256_path(self.p0_records_path),
                    "allows_p1": bool(self.p0_gate["allows_next_phase"]),
                },
            },
        )
        print(json.dumps({"canonical_p0_verified": str(destination)}, ensure_ascii=False))
        return destination

    def unavailable_stage(self, stage: str) -> None:
        raise V7ExperimentError(
            f"V7 {stage} is intentionally unavailable. It requires a synchronized and reviewed "
            "successful P1 gate before any P2 implementation, training, or evaluation."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos V7 diversified active-risk-control")
    parser.add_argument(
        "stage",
        choices=(
            "audit", "p1-path-bank", "p1-baselines", "p2-train", "p2-evaluate",
            "p3-calibrate", "p4-overlay", "p5-freeze",
        ),
    )
    parser.add_argument("--config", default="csj/configs/risk_control_v7.yaml")
    parser.add_argument("--run-id", default="v7_p1")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    experiment = V7Experiment(
        args.config,
        args.run_id,
        device_override=args.device,
        allow_model_download=args.allow_model_download,
    )
    try:
        if args.stage == "audit":
            experiment.audit()
        elif args.stage == "p1-path-bank":
            experiment.p1_path_bank(resume=args.resume, smoke=args.smoke)
            if args.smoke:
                smoke_gate = experiment.p1_smoke_gate()
                print(json.dumps({"p1_smoke_gate": str(smoke_gate)}, ensure_ascii=False))
        elif args.stage == "p1-baselines":
            experiment.p1_baselines(smoke=args.smoke)
        else:
            experiment.unavailable_stage(args.stage)
    except (V7ExperimentError, V7PathBankError, V7BaselineError, V7PlotError, V7ProvenanceError) as exc:
        parser.exit(2, f"V7 {args.stage} failed: {exc}\n")


if __name__ == "__main__":
    main()
