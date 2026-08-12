"""Fixed P0 sample-bank creation and V5's declared direction baselines."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from csj.metrics import direction_label
from csj.utils.tool import MODEL_FEATURES
from csj.v3.panel_data import TIME_FEATURES, add_time_features
from csj.v5.target_data import TargetOnlyCase
from model.kronos import auto_regressive_inference


SAMPLE_BANK_SCHEMA_VERSION = 1


class PathBankError(RuntimeError):
    """A V5 fixed path-bank invariant failed."""


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
    return value


def _normalize_context(values: np.ndarray, *, clip: float, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    normalized = np.clip((values - mean) / (std + epsilon), -clip, clip).astype(np.float32)
    return normalized, mean, std


def _enforce_path_constraints(samples: np.ndarray) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64).copy()
    result[..., 1] = np.maximum.reduce([result[..., 1], result[..., 0], result[..., 3]])
    result[..., 2] = np.minimum.reduce([result[..., 2], result[..., 0], result[..., 3]])
    result[..., 4:] = np.maximum(result[..., 4:], 0.0)
    return result


def _path_metrics(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    origin_close: float,
    day_end_indices: Sequence[int],
) -> dict[str, object]:
    actual_return = float(actual[int(day_end_indices[-1]), 3] / origin_close - 1.0)
    predicted_return = float(predicted[int(day_end_indices[-1]), 3] / origin_close - 1.0)
    actual_direction = direction_label(actual_return)
    predicted_direction = 1 if predicted_return >= 0.0 else -1
    actual_path_returns = actual[:, 3] / origin_close - 1.0
    predicted_path_returns = predicted[:, 3] / origin_close - 1.0
    correlation = (
        float(np.corrcoef(actual_path_returns, predicted_path_returns)[0, 1])
        if len(actual) > 1
        and np.std(actual_path_returns) > 0.0
        and np.std(predicted_path_returns) > 0.0
        else float("nan")
    )
    return {
        "day3_actual_return": actual_return,
        "day3_predicted_return": predicted_return,
        "day3_actual_direction": actual_direction,
        "day3_predicted_direction": predicted_direction,
        "path_return_correlation": correlation,
        "z_normalized_dtw": _z_normalized_dtw(actual_path_returns, predicted_path_returns),
    }


def _z_normalized_dtw(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or len(actual) != len(predicted):
        return float("nan")
    actual_std = float(np.std(actual))
    predicted_std = float(np.std(predicted))
    if actual_std == 0.0 or predicted_std == 0.0:
        return float("nan")
    a = (actual - np.mean(actual)) / actual_std
    b = (predicted - np.mean(predicted)) / predicted_std
    distance = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    distance[0, 0] = 0.0
    for row in range(1, len(a) + 1):
        for column in range(1, len(b) + 1):
            distance[row, column] = abs(a[row - 1] - b[column - 1]) + min(
                distance[row - 1, column],
                distance[row, column - 1],
                distance[row - 1, column - 1],
            )
    return float(distance[-1, -1] / max(len(a), 1))


def generate_sample_bank(
    tokenizer: torch.nn.Module,
    predictor: torch.nn.Module,
    cases: Sequence[TargetOnlyCase],
    *,
    device: torch.device,
    max_context: int,
    clip: float,
    epsilon: float,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    batch_size: int,
    random_seed: int,
    model_name: str = "zero_shot_mean_path",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate one fixed raw-path bank and a light per-case index table.

    The case order is fixed and the configured global sampling seed is reset at
    the beginning of each split.  Cases of the same target length are batched
    on CUDA to keep P0 tractable; the saved bank is the sole raw-path copy.
    """

    if not cases:
        raise PathBankError("Cannot generate a V5 P0 sample bank from zero cases")
    if sample_count < 1 or batch_size < 1:
        raise ValueError("V5 P0 sample_count and batch_size must be positive")
    tokenizer = tokenizer.to(device)
    predictor = predictor.to(device)
    tokenizer.eval()
    predictor.eval()
    torch.manual_seed(int(random_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(random_seed))
    np.random.seed(int(random_seed))
    rows: list[dict[str, object]] = []
    bank: dict[str, np.ndarray] = {}
    by_length: dict[int, list[TargetOnlyCase]] = defaultdict(list)
    for case in cases:
        by_length[case.pred_len].append(case)
    sampling_order = 0
    for pred_len, same_length_cases in sorted(by_length.items()):
        for offset in range(0, len(same_length_cases), batch_size):
            batch_cases = same_length_cases[offset : offset + batch_size]
            normalized_contexts: list[np.ndarray] = []
            context_stamps: list[np.ndarray] = []
            target_stamps: list[np.ndarray] = []
            statistics: list[tuple[np.ndarray, np.ndarray]] = []
            for case in batch_cases:
                context = add_time_features(case.target_context)
                target = add_time_features(case.target)
                raw_context = context[MODEL_FEATURES].to_numpy(dtype=np.float64)
                normalized, mean, std = _normalize_context(raw_context, clip=clip, epsilon=epsilon)
                normalized_contexts.append(normalized)
                context_stamps.append(
                    context[list(TIME_FEATURES)].to_numpy(dtype=np.float32)
                )
                target_stamps.append(target[list(TIME_FEATURES)].to_numpy(dtype=np.float32))
                statistics.append((mean, std))
            x = torch.from_numpy(np.stack(normalized_contexts)).to(device)
            x_stamp = torch.from_numpy(np.stack(context_stamps)).to(device)
            y_stamp = torch.from_numpy(np.stack(target_stamps)).to(device)
            normalized_samples = auto_regressive_inference(
                tokenizer,
                predictor,
                x,
                x_stamp,
                y_stamp,
                max_context=max_context,
                pred_len=pred_len,
                clip=clip,
                T=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
                return_samples=True,
            )[:, :, -pred_len:, :]
            for index, case in enumerate(batch_cases):
                mean, std = statistics[index]
                samples = _enforce_path_constraints(
                    normalized_samples[index] * (std + epsilon) + mean
                )
                if samples.shape != (sample_count, case.pred_len, len(MODEL_FEATURES)):
                    raise PathBankError(
                        f"Invalid sample-bank shape for {case.case_key}: {samples.shape!r}"
                    )
                predicted = samples.mean(axis=0)
                actual = case.target[MODEL_FEATURES].to_numpy(dtype=np.float64)
                origin_close = float(case.target_context["close"].iloc[-1])
                if not np.isfinite(origin_close) or origin_close <= 0.0:
                    raise PathBankError(f"Non-positive origin close for {case.case_key}")
                day_end = int(case.day_end_indices[-1])
                sample_returns = samples[:, day_end, 3] / origin_close - 1.0
                row = {
                    "case_key": case.case_key,
                    "model": model_name,
                    "target_contract_id": case.target_contract_id,
                    "product": case.product,
                    "target_end_day": case.target_end_day,
                    "origin_timestamp": case.origin_timestamp,
                    "origin_close": origin_close,
                    "pred_len": int(case.pred_len),
                    "day_end_indices": list(case.day_end_indices),
                    "actual_path": actual.tolist(),
                    "predicted_path": predicted.tolist(),
                    "sample_bank_key": case.case_key,
                    "sample_count": int(sample_count),
                    "sampling_seed": int(random_seed),
                    "sampling_order": int(sampling_order),
                    "sample_up_count": int(np.sum(sample_returns > 0.0)),
                    "sample_down_count": int(np.sum(sample_returns < 0.0)),
                    "sample_zero_count": int(np.sum(sample_returns == 0.0)),
                    "sample_vote_probability_up": float(np.mean(sample_returns > 0.0)),
                    "oracle_direction_supported": bool(
                        np.any(
                            (sample_returns > 0.0)
                            if case.day3_return > 0.0
                            else (sample_returns < 0.0)
                            if case.day3_return < 0.0
                            else (sample_returns == 0.0)
                        )
                    ),
                    **_path_metrics(
                        actual=actual,
                        predicted=predicted,
                        origin_close=origin_close,
                        day_end_indices=case.day_end_indices,
                    ),
                }
                rows.append(row)
                bank[case.case_key] = samples.astype(np.float32)
                sampling_order += 1
            del x, x_stamp, y_stamp, normalized_samples
    records = pd.DataFrame(rows).sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)
    return records, bank


def write_sample_bank(path: str | Path, bank: Mapping[str, np.ndarray], *, metadata: Mapping[str, object]) -> Path:
    """Write raw sample paths once in compressed form, plus an auditable index."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(bank)
    if not keys:
        raise PathBankError("Cannot save an empty V5 sample bank")
    arrays = {f"sample_{index:06d}": np.asarray(bank[key], dtype=np.float32) for index, key in enumerate(keys)}
    index = {key: f"sample_{index:06d}" for index, key in enumerate(keys)}
    payload = {
        "schema_version": SAMPLE_BANK_SCHEMA_VERSION,
        "keys": keys,
        "index": index,
        "metadata": _json_safe(dict(metadata)),
    }
    np.savez_compressed(destination, __metadata__=json.dumps(payload, ensure_ascii=False), **arrays)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise PathBankError(f"V5 sample bank was not written: {destination}")
    return destination


def read_sample_bank(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load a V5 compressed bank and validate its one-key-per-case mapping."""

    source = Path(path)
    if not source.is_file():
        raise PathBankError(f"V5 sample bank is missing: {source}")
    with np.load(source, allow_pickle=False) as loaded:
        if "__metadata__" not in loaded.files:
            raise PathBankError(f"V5 sample bank has no metadata: {source}")
        metadata = json.loads(str(loaded["__metadata__"].item()))
        if not isinstance(metadata, dict) or int(metadata.get("schema_version", -1)) != SAMPLE_BANK_SCHEMA_VERSION:
            raise PathBankError(f"V5 sample bank has unsupported schema: {source}")
        index = metadata.get("index")
        if not isinstance(index, dict):
            raise PathBankError(f"V5 sample bank has invalid index: {source}")
        bank = {str(key): np.asarray(loaded[str(array_key)], dtype=np.float32) for key, array_key in index.items()}
    return bank, metadata


def rehydrate_path_records(
    records: pd.DataFrame,
    *,
    bank: Mapping[str, np.ndarray],
    cases: Sequence[TargetOnlyCase],
) -> pd.DataFrame:
    """Restore plot-only actual/mean paths from the single compressed bank.

    P0's on-disk JSON deliberately omits nested paths so it is not a second raw
    sample store.  This helper is the one explicit route that reconstructs the
    two small plot paths in memory when a later stage needs them.
    """

    if records.empty or "case_key" not in records:
        raise PathBankError("Cannot rehydrate V5 path records without case keys")
    by_key = {case.case_key: case for case in cases}
    output = records.copy()
    expected = set(output["case_key"].astype(str))
    if expected != set(bank) or expected != set(by_key):
        raise PathBankError("V5 compact records, sample bank, and cases do not have matching keys")
    actual_paths: list[list[list[float]]] = []
    predicted_paths: list[list[list[float]]] = []
    for key in output["case_key"].astype(str):
        case = by_key[key]
        samples = np.asarray(bank[key], dtype=np.float64)
        if samples.ndim != 3 or samples.shape[1:] != (case.pred_len, len(MODEL_FEATURES)):
            raise PathBankError(f"V5 sample bank shape no longer matches {key}")
        actual_paths.append(case.target[MODEL_FEATURES].to_numpy(dtype=np.float64).tolist())
        predicted_paths.append(samples.mean(axis=0).tolist())
    output["actual_path"] = actual_paths
    output["predicted_path"] = predicted_paths
    return output


def _direction_records_from_paths(
    records: pd.DataFrame,
    *,
    model_name: str,
    probability_column: str,
    predicted_direction_column: str,
) -> pd.DataFrame:
    output = records.copy()
    output["fold_id"] = output["fold_id"].astype(str)
    output["actual_direction"] = output["day3_actual_direction"].astype(np.int8)
    output["predicted_direction"] = output[predicted_direction_column].astype(np.int8)
    output["probability_up"] = pd.to_numeric(output[probability_column], errors="raise")
    output["valid_direction"] = output["actual_direction"] != 0
    output["actual_label"] = (output["actual_direction"] == 1).astype(np.int8)
    output["predicted_label"] = (output["predicted_direction"] == 1).astype(np.int8)
    output.loc[~output["valid_direction"], "predicted_direction"] = 0
    output["model"] = model_name
    return output.sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def zero_shot_mean_direction_records(records: pd.DataFrame) -> pd.DataFrame:
    return _direction_records_from_paths(
        records,
        model_name="zero_shot_mean_path",
        probability_column="sample_vote_probability_up",
        predicted_direction_column="day3_predicted_direction",
    )


def zero_shot_vote_direction_records(records: pd.DataFrame) -> pd.DataFrame:
    output = records.copy()
    output["sample_vote_direction"] = np.where(
        output["sample_vote_probability_up"].to_numpy(dtype=np.float64) >= 0.5, 1, -1
    )
    return _direction_records_from_paths(
        output,
        model_name="zero_shot_sample_vote",
        probability_column="sample_vote_probability_up",
        predicted_direction_column="sample_vote_direction",
    )


def fit_product_majority_direction_records(
    records: pd.DataFrame,
    *,
    fit_cases: Sequence[TargetOnlyCase],
) -> pd.DataFrame:
    by_product: dict[str, float] = {}
    for product, product_cases in _cases_by_product(fit_cases).items():
        returns = np.asarray([case.day3_return for case in product_cases], dtype=np.float64)
        nonzero = returns[returns != 0.0]
        by_product[product] = float(np.mean(nonzero > 0.0)) if len(nonzero) else 0.5
    output = records.copy()
    output["fit_product_probability_up"] = output["product"].map(by_product)
    if output["fit_product_probability_up"].isna().any():
        missing = sorted(output.loc[output["fit_product_probability_up"].isna(), "product"].unique())
        raise PathBankError(f"Fit-period majority lacks products in evaluation: {missing!r}")
    output["fit_product_direction"] = np.where(
        output["fit_product_probability_up"].to_numpy(dtype=np.float64) >= 0.5, 1, -1
    )
    return _direction_records_from_paths(
        output,
        model_name="fit_product_majority",
        probability_column="fit_product_probability_up",
        predicted_direction_column="fit_product_direction",
    )


def context_3day_momentum_direction_records(
    records: pd.DataFrame,
    *,
    cases: Sequence[TargetOnlyCase],
    fallback_probabilities: Mapping[str, float],
) -> pd.DataFrame:
    """Use the last three fully completed context days, with fit majority ties."""

    by_key = {case.case_key: case for case in cases}
    rows: list[dict[str, object]] = []
    for row in records.to_dict("records"):
        case_key = str(row["case_key"])
        case = by_key.get(case_key)
        if case is None:
            raise PathBankError(f"Momentum case missing from V5 target cases: {case_key}")
        context = case.target_context
        groups = list(context.groupby("trading_day", sort=False))
        complete_groups = [group for _, group in groups if len(group) in {5, 7}]
        if len(complete_groups) < 3:
            raise PathBankError(f"V5 context lacks three complete days: {case_key}")
        selected = complete_groups[-3:]
        first_close = float(selected[0]["close"].iloc[0])
        last_close = float(selected[-1]["close"].iloc[-1])
        momentum_return = last_close / first_close - 1.0
        fallback_probability = float(fallback_probabilities[case.product])
        direction = direction_label(momentum_return)
        if direction == 0:
            direction = 1 if fallback_probability >= 0.5 else -1
        row["context_3day_momentum_return"] = momentum_return
        row["context_3day_momentum_probability_up"] = (
            1.0 if direction == 1 else 0.0
        )
        row["context_3day_momentum_direction"] = direction
        rows.append(row)
    return _direction_records_from_paths(
        pd.DataFrame(rows),
        model_name="context_3day_momentum",
        probability_column="context_3day_momentum_probability_up",
        predicted_direction_column="context_3day_momentum_direction",
    )


def _cases_by_product(cases: Sequence[TargetOnlyCase]) -> dict[str, tuple[TargetOnlyCase, ...]]:
    grouped: dict[str, list[TargetOnlyCase]] = defaultdict(list)
    for case in cases:
        grouped[case.product].append(case)
    return {product: tuple(values) for product, values in grouped.items()}


def fit_product_up_probabilities(cases: Sequence[TargetOnlyCase]) -> dict[str, float]:
    values: dict[str, float] = {}
    for product, product_cases in _cases_by_product(cases).items():
        returns = np.asarray([case.day3_return for case in product_cases], dtype=np.float64)
        nonzero = returns[returns != 0.0]
        values[product] = float(np.mean(nonzero > 0.0)) if len(nonzero) else 0.5
    return values


def selected_baseline(
    validation_records: Mapping[str, pd.DataFrame],
    *,
    selection_order: Sequence[str],
    allow_unavailable: bool = False,
) -> tuple[str, dict[str, object]]:
    """Choose one baseline only from inner validation, with a fixed tie order."""

    scores: dict[str, float | None] = {}
    for name in selection_order:
        records = validation_records.get(name)
        if records is None:
            raise PathBankError(f"Declared V5 baseline is missing validation records: {name}")
        valid = records.loc[records["valid_direction"].astype(bool)]
        actual = valid["actual_direction"].to_numpy(dtype=np.int8)
        predicted = valid["predicted_direction"].to_numpy(dtype=np.int8)
        up = actual == 1
        down = actual == -1
        scores[name] = (
            float(0.5 * (np.mean(predicted[up] == 1) + np.mean(predicted[down] == -1)))
            if up.any() and down.any()
            else None
        )
    finite = [(name, score) for name, score in scores.items() if score is not None and math.isfinite(score)]
    if not finite:
        if not allow_unavailable:
            raise PathBankError(
                "V5 formal P0 inner validation has no finite balanced accuracy"
            )
        # A tiny smoke subset can contain only one realized direction.  It is
        # explicitly non-gating, but still needs a deterministic selected arm
        # to exercise the P0 artifact path.
        selected = str(selection_order[0])
        return selected, {
            "selection_order": list(selection_order),
            "validation_balanced_accuracy": scores,
            "selected_direction_baseline": selected,
            "selection_available": False,
            "selection_reason": "no_finite_balanced_accuracy_in_smoke_subset",
        }
    best_score = max(float(score) for _, score in finite)
    selected = next(
        name for name in selection_order if scores[name] is not None and float(scores[name]) == best_score
    )
    return selected, {
        "selection_order": list(selection_order),
        "validation_balanced_accuracy": scores,
        "selected_direction_baseline": selected,
        "selection_available": True,
    }


__all__ = [
    "PathBankError",
    "SAMPLE_BANK_SCHEMA_VERSION",
    "context_3day_momentum_direction_records",
    "fit_product_majority_direction_records",
    "fit_product_up_probabilities",
    "generate_sample_bank",
    "read_sample_bank",
    "rehydrate_path_records",
    "selected_baseline",
    "write_sample_bank",
    "zero_shot_mean_direction_records",
    "zero_shot_vote_direction_records",
]
