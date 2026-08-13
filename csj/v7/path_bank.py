"""Resumable, unique-case frozen Kronos raw-path and hidden-state cache for V7 P1.

The cache deliberately stores model outputs only once per unique case.  Fold
labels remain in the immutable P0 record table and fold-specific crossing
features are derived later, so no P1 inference can accidentally fit or read an
evaluation threshold.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from csj.utils.tool import MODEL_FEATURES
from csj.v3.panel_data import TIME_FEATURES, add_time_features
from csj.v5.target_data import TargetOnlyCase
from model.kronos import auto_regressive_inference


PATH_BANK_SCHEMA_VERSION = 1


class V7PathBankError(RuntimeError):
    """A raw-path cache cannot satisfy V7's immutable cache contract."""


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
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class PathValidity:
    valid_mask: np.ndarray
    invalid_reason_counts: Mapping[str, int]

    @property
    def valid_count(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def total_count(self) -> int:
        return int(len(self.valid_mask))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise V7PathBankError(f"V7 path-bank file is missing: {source}")
    return sha256_bytes(source.read_bytes())


def sha256_case_keys(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def cache_key_for_case(
    *,
    strategy_version: int,
    data_fingerprint: str,
    case_key: str,
    tokenizer_revision: str,
    predictor_revision: str,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    """Hash exactly the pre-registered cache-key fields, in a stable format."""

    payload = {
        "strategy_version": int(strategy_version),
        "data_fingerprint": str(data_fingerprint),
        "case_key": str(case_key),
        "tokenizer_revision": str(tokenizer_revision),
        "predictor_revision": str(predictor_revision),
        "sample_count": int(sample_count),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
    }
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def sampling_seed(cache_key: str) -> int:
    """Derive a deterministic positive 63-bit seed without Python's hash()."""

    digest = hashlib.sha256(str(cache_key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def input_signature(
    context_values: np.ndarray,
    context_stamps: np.ndarray,
    target_stamps: np.ndarray,
    *,
    context_mean: np.ndarray | None = None,
    context_std: np.ndarray | None = None,
    target_timestamps: np.ndarray | None = None,
) -> str:
    digest = hashlib.sha256()
    values_and_dtypes: list[tuple[np.ndarray, str]] = [
        (context_values, "<f4"),
        (context_stamps, "<f4"),
        (target_stamps, "<f4"),
    ]
    if context_mean is not None:
        values_and_dtypes.append((context_mean, "<f4"))
    if context_std is not None:
        values_and_dtypes.append((context_std, "<f4"))
    if target_timestamps is not None:
        values_and_dtypes.append((target_timestamps, "<i8"))
    for values, dtype in values_and_dtypes:
        digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def output_signature(
    paths: np.ndarray,
    hidden: np.ndarray,
    *,
    context_mean: np.ndarray | None = None,
    context_std: np.ndarray | None = None,
    target_timestamps: np.ndarray | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(paths, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(hidden, dtype="<f4").tobytes())
    if context_mean is not None:
        digest.update(np.ascontiguousarray(context_mean, dtype="<f4").tobytes())
    if context_std is not None:
        digest.update(np.ascontiguousarray(context_std, dtype="<f4").tobytes())
    if target_timestamps is not None:
        digest.update(np.ascontiguousarray(target_timestamps, dtype="<i8").tobytes())
    return digest.hexdigest()


def normalize_context(
    values: np.ndarray, *, clip: float, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(MODEL_FEATURES):
        raise V7PathBankError(f"Invalid V7 context shape: {raw.shape!r}")
    if not np.isfinite(raw).all():
        raise V7PathBankError("V7 context contains non-finite OHLCVA")
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    z_values = (raw - mean) / (std + float(epsilon))
    clip_fraction = float(np.mean(np.abs(z_values) > float(clip)))
    normalized = np.clip(z_values, -float(clip), float(clip)).astype(np.float32)
    return normalized, mean.astype(np.float32), std.astype(np.float32), clip_fraction


def validate_raw_paths(paths: np.ndarray) -> PathValidity:
    """Validate every path without projecting invalid values back into validity."""

    values = np.asarray(paths, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(MODEL_FEATURES):
        raise V7PathBankError(f"Invalid V7 raw path tensor shape: {values.shape!r}")
    reasons: defaultdict[str, int] = defaultdict(int)
    valid = np.ones(values.shape[0], dtype=bool)
    for index, sample in enumerate(values):
        sample_reasons: list[str] = []
        if not np.isfinite(sample).all():
            sample_reasons.append("nonfinite_ohlcva")
        else:
            open_values = sample[:, 0]
            high_values = sample[:, 1]
            low_values = sample[:, 2]
            close_values = sample[:, 3]
            flows = sample[:, 4:]
            if (open_values <= 0.0).any() or (high_values <= 0.0).any() or (low_values <= 0.0).any() or (close_values <= 0.0).any():
                sample_reasons.append("nonpositive_price")
            if (high_values < np.maximum(open_values, close_values)).any():
                sample_reasons.append("high_below_open_or_close")
            if (low_values > np.minimum(open_values, close_values)).any():
                sample_reasons.append("low_above_open_or_close")
            if (high_values < low_values).any():
                sample_reasons.append("high_below_low")
            if (flows < 0.0).any():
                sample_reasons.append("negative_volume_or_amount")
        if sample_reasons:
            valid[index] = False
            for reason in sorted(set(sample_reasons)):
                reasons[reason] += 1
    return PathValidity(valid, dict(sorted(reasons.items())))


def _case_arrays(
    case: TargetOnlyCase,
    *,
    clip: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    context = add_time_features(case.target_context)
    target = add_time_features(case.target)
    raw_context = context[MODEL_FEATURES].to_numpy(dtype=np.float64)
    normalized, mean, std, clip_fraction = normalize_context(
        raw_context, clip=clip, epsilon=epsilon
    )
    context_stamps = context[list(TIME_FEATURES)].to_numpy(dtype=np.float32)
    target_stamps = target[list(TIME_FEATURES)].to_numpy(dtype=np.float32)
    target_timestamps = pd.to_datetime(target["timestamps"]).astype("int64").to_numpy(
        dtype=np.int64
    )
    if target_timestamps.shape != (int(case.pred_len),) or (
        len(target_timestamps) > 1 and np.any(np.diff(target_timestamps) <= 0)
    ):
        raise V7PathBankError(
            f"V7 target timestamps are not strictly ordered for {case.case_key}"
        )
    return normalized, mean, std, context_stamps, target_stamps, clip_fraction, target_timestamps


def _extract_hidden(
    tokenizer: torch.nn.Module,
    predictor: torch.nn.Module,
    normalized_context: np.ndarray,
    context_stamps: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(normalized_context[None, ...]).to(device)
    stamps = torch.from_numpy(context_stamps[None, ...]).to(device)
    with torch.no_grad():
        tokens = tokenizer.encode(x, half=True)
        _, hidden = predictor.decode_s1(tokens[0], tokens[1], stamps)
        last = hidden[:, -1, :].detach().float().cpu().numpy()
    if last.ndim != 2 or last.shape[0] != 1 or not np.isfinite(last).all():
        raise V7PathBankError(f"Invalid frozen Kronos hidden shape: {last.shape!r}")
    return last[0].astype(np.float32, copy=False)


def generate_case_output(
    tokenizer: torch.nn.Module,
    predictor: torch.nn.Module,
    case: TargetOnlyCase,
    *,
    device: torch.device,
    max_context: int,
    clip: float,
    epsilon: float,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Generate exactly one atomic raw-path/hidden output for one V7 case."""

    if sample_count < 1:
        raise ValueError("V7 sample_count must be positive")
    arrays = _case_arrays(case, clip=clip, epsilon=epsilon)
    normalized, mean, std, context_stamps, target_stamps, clip_fraction, timestamps = arrays
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    # The model's multinomial sampler is torch-based.  This only ensures any
    # incidental NumPy consumer remains deterministic without using hash().
    np.random.seed(int(seed) % (2**32 - 1))
    hidden = _extract_hidden(
        tokenizer, predictor, normalized, context_stamps, device=device
    )
    x = torch.from_numpy(normalized[None, ...]).to(device)
    x_stamp = torch.from_numpy(context_stamps[None, ...]).to(device)
    y_stamp = torch.from_numpy(target_stamps[None, ...]).to(device)
    samples = auto_regressive_inference(
        tokenizer,
        predictor,
        x,
        x_stamp,
        y_stamp,
        max_context=int(max_context),
        pred_len=int(case.pred_len),
        clip=float(clip),
        T=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        sample_count=int(sample_count),
        verbose=False,
        return_samples=True,
    )[:, :, -int(case.pred_len) :, :]
    raw_paths = np.asarray(
        samples[0] * (std[None, None, :] + float(epsilon)) + mean[None, None, :],
        dtype=np.float32,
    )
    if raw_paths.shape != (int(sample_count), int(case.pred_len), len(MODEL_FEATURES)):
        raise V7PathBankError(
            f"V7 path tensor has wrong shape for {case.case_key}: {raw_paths.shape!r}"
        )
    validity = validate_raw_paths(raw_paths)
    payload = {
        "raw_paths": raw_paths,
        "hidden": hidden,
        "context_mean": mean,
        "context_std": std,
        "target_timestamps": timestamps,
    }
    metadata = {
        "case_key": case.case_key,
        "target_contract_id": case.target_contract_id,
        "product": case.product,
        "origin_timestamp": case.origin_timestamp,
        "origin_trading_day": case.origin_trading_day,
        "target_end_day": case.target_end_day,
        "pred_len": int(case.pred_len),
        "day_end_indices": list(case.day_end_indices),
        "origin_close": float(case.target_context["close"].iloc[-1]),
        "context_clip_fraction": float(clip_fraction),
        "input_signature_sha256": input_signature(
            normalized,
            context_stamps,
            target_stamps,
            context_mean=mean,
            context_std=std,
            target_timestamps=timestamps,
        ),
        "output_signature_sha256": output_signature(
            raw_paths,
            hidden,
            context_mean=mean,
            context_std=std,
            target_timestamps=timestamps,
        ),
        "valid_path_count": validity.valid_count,
        "sample_count": validity.total_count,
        "valid_path_fraction": (
            float(validity.valid_count / validity.total_count)
            if validity.total_count
            else 0.0
        ),
        "invalid_path_reason_counts": dict(validity.invalid_reason_counts),
    }
    return payload, metadata


def _encode_metadata(value: Mapping[str, object]) -> np.ndarray:
    return np.asarray(
        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False),
        dtype=np.str_,
    )


def write_shard_atomic(
    path: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> Path:
    """Atomically write a compressed shard and return it after SHA validation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise V7PathBankError(f"Refusing to overwrite immutable V7 shard: {destination}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                __metadata__=_encode_metadata(metadata),
                **{key: np.asarray(value) for key, value in arrays.items()},
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise V7PathBankError(f"V7 shard write produced no data: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    # Full readback validates the compressed container before it enters a manifest.
    read_shard(destination)
    return destination


def read_shard(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as loaded:
            if "__metadata__" not in loaded.files:
                raise V7PathBankError(f"V7 shard has no metadata: {source}")
            metadata = json.loads(str(loaded["__metadata__"].item()))
            arrays = {
                key: np.asarray(loaded[key])
                for key in loaded.files
                if key != "__metadata__"
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise V7PathBankError(f"Cannot read V7 shard: {source}") from exc
    if not isinstance(metadata, dict) or int(metadata.get("schema_version", -1)) != PATH_BANK_SCHEMA_VERSION:
        raise V7PathBankError(f"Unsupported V7 shard schema: {source}")
    required = {"raw_paths", "hidden", "context_mean", "context_std", "target_timestamps"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise V7PathBankError(f"V7 shard misses arrays {missing!r}: {source}")
    return arrays, metadata


def validate_shard_entry(entry: Mapping[str, object], *, artifact_root: str | Path) -> None:
    required = {"relative_path", "sha256", "case_keys", "shapes", "dtypes"}
    missing = sorted(required.difference(entry))
    if missing:
        raise V7PathBankError(f"V7 manifest shard entry misses fields: {missing!r}")
    root = Path(artifact_root).resolve()
    path = (root / str(entry["relative_path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise V7PathBankError("V7 manifest shard path escapes artifact root") from exc
    if not path.is_file():
        raise V7PathBankError(f"V7 manifest shard path is missing: {path}")
    if sha256_path(path) != str(entry["sha256"]):
        raise V7PathBankError(f"V7 shard checksum mismatch: {path}")
    arrays, metadata = read_shard(path)
    keys = tuple(str(value) for value in entry["case_keys"])
    if tuple(str(value) for value in metadata.get("case_keys", ())) != keys:
        raise V7PathBankError(f"V7 shard case-key metadata mismatch: {path}")
    shapes = entry["shapes"]
    dtypes = entry["dtypes"]
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        raise V7PathBankError(f"V7 manifest shard shape/dtype fields are invalid: {path}")
    for name, expected_shape in shapes.items():
        if name not in arrays or list(arrays[name].shape) != list(expected_shape):
            raise V7PathBankError(f"V7 shard shape mismatch for {name}: {path}")
    for name, expected_dtype in dtypes.items():
        if name not in arrays or str(arrays[name].dtype) != str(expected_dtype):
            raise V7PathBankError(f"V7 shard dtype mismatch for {name}: {path}")


def make_shard_entry(
    path: str | Path,
    *,
    artifact_root: str | Path,
    case_keys: Sequence[str],
) -> dict[str, object]:
    source = Path(path).resolve()
    root = Path(artifact_root).resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise V7PathBankError(f"V7 shard escapes artifact root: {source}") from exc
    arrays, metadata = read_shard(source)
    return {
        "relative_path": str(relative),
        "sha256": sha256_path(source),
        "case_keys": [str(value) for value in case_keys],
        "case_keys_sha256": sha256_case_keys(case_keys),
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "schema_version": int(metadata["schema_version"]),
    }


def build_shard_arrays(
    outputs: Sequence[tuple[dict[str, np.ndarray], Mapping[str, object]]]
) -> dict[str, np.ndarray]:
    if not outputs:
        raise V7PathBankError("Cannot build an empty V7 path shard")
    pred_lengths = {int(metadata["pred_len"]) for _, metadata in outputs}
    if len(pred_lengths) != 1:
        raise V7PathBankError("Each V7 shard must contain one target length")
    arrays = {
        name: np.stack([payload[name] for payload, _ in outputs], axis=0)
        for name in ("raw_paths", "hidden", "context_mean", "context_std", "target_timestamps")
    }
    # A non-finite generated OHLCVA sample is an invalid *path*, not a reason
    # to erase the whole atomic case shard.  It remains in the raw bank so P1
    # can truthfully count it against the validity gate.  Frozen hidden and
    # context-normalization state, however, must always be finite.
    if not (
        np.isfinite(arrays["hidden"]).all()
        and np.isfinite(arrays["context_mean"]).all()
        and np.isfinite(arrays["context_std"]).all()
    ):
        raise V7PathBankError("V7 path shard contains non-finite frozen state")
    return arrays


__all__ = [
    "PATH_BANK_SCHEMA_VERSION",
    "PathValidity",
    "V7PathBankError",
    "build_shard_arrays",
    "cache_key_for_case",
    "generate_case_output",
    "input_signature",
    "make_shard_entry",
    "normalize_context",
    "output_signature",
    "read_shard",
    "sampling_seed",
    "sha256_case_keys",
    "sha256_path",
    "validate_raw_paths",
    "validate_shard_entry",
    "write_shard_atomic",
]
