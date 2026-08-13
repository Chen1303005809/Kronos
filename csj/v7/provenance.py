"""Immutable V7 phase provenance and persisted-gate validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from csj.v7 import PRODUCTION_ELIGIBLE, RESULT_SCOPE, STRATEGY_VERSION


class V7ProvenanceError(RuntimeError):
    """A persisted V7 artifact cannot safely be used as phase input."""


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise V7ProvenanceError(f"Required V7 artifact is missing: {source}")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def write_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(safe_json(payload), ensure_ascii=False, indent=2, allow_nan=False)
                + "\n"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V7ProvenanceError(f"Cannot read {label}: {source}") from exc
    if not isinstance(value, dict):
        raise V7ProvenanceError(f"{label} must be a JSON object: {source}")
    return value


def git_commit(repo_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def runtime_provenance(device: torch.device) -> dict[str, object]:
    value: dict[str, object] = {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    if device.type == "cuda":
        value.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return value


def phase_metadata(
    *,
    phase: str,
    run_id: str,
    config: Mapping[str, Any],
    data_fingerprint: str,
    resolved_config_path: str | Path,
    repo_root: str | Path,
    smoke: bool,
    runtime: Mapping[str, object] | None = None,
    upstream_gate_path: str | Path | None = None,
) -> dict[str, object]:
    """Build the complete provenance contract required for V7 artifacts."""

    model = config["model"]
    upstream: dict[str, object] = {}
    if upstream_gate_path is not None:
        upstream_path = Path(upstream_gate_path)
        upstream = {
            "upstream_gate_path": str(upstream_path),
            "upstream_gate_sha256": sha256_path(upstream_path),
        }
    return {
        "strategy_version": STRATEGY_VERSION,
        "phase": str(phase),
        "run_id": str(run_id),
        "result_scope": RESULT_SCOPE,
        "production_eligible": PRODUCTION_ELIGIBLE,
        "git_commit": git_commit(repo_root),
        "resolved_config_path": str(resolved_config_path),
        "resolved_config_sha256": sha256_path(resolved_config_path),
        "data_fingerprint": str(data_fingerprint),
        "tokenizer_id": model["tokenizer_id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "predictor_id": model["predictor_id"],
        "predictor_revision": model["predictor_revision"],
        "risk_label_version": config["risk_labels"]["version"],
        "smoke": bool(smoke),
        "runtime": dict(runtime or {}),
        **upstream,
    }


def verify_persisted_gate(
    path: str | Path,
    *,
    expected_phase: str,
    config: Mapping[str, Any],
    data_fingerprint: str,
    run_id: str | None = None,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Require a matching, successful, non-smoke predecessor gate."""

    source = Path(path)
    gate = read_json(source, label=f"V7 {expected_phase} gate")
    expected = {
        "strategy_version": STRATEGY_VERSION,
        "phase": str(expected_phase),
        "result_scope": RESULT_SCOPE,
        "production_eligible": PRODUCTION_ELIGIBLE,
        "data_fingerprint": str(data_fingerprint),
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "predictor_revision": config["model"]["predictor_revision"],
        "risk_label_version": config["risk_labels"]["version"],
    }
    if run_id is not None:
        expected["run_id"] = str(run_id)
    for key, expected_value in expected.items():
        if gate.get(key) != expected_value:
            raise V7ProvenanceError(
                f"V7 {expected_phase} gate does not match active protocol at {key}: "
                f"{gate.get(key)!r} != {expected_value!r}"
            )
    if not bool(gate.get("allows_next_phase")):
        raise V7ProvenanceError(
            f"V7 {expected_phase} gate does not allow the next phase: {source}"
        )
    if bool(gate.get("smoke")) and not allow_smoke:
        raise V7ProvenanceError(
            f"V7 {expected_phase} smoke gate cannot unlock a formal phase: {source}"
        )
    return gate


__all__ = [
    "V7ProvenanceError",
    "git_commit",
    "phase_metadata",
    "read_json",
    "runtime_provenance",
    "safe_json",
    "sha256_json",
    "sha256_path",
    "verify_persisted_gate",
    "write_json",
]
