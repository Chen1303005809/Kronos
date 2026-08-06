from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from csj.futures_data import ForecastCase, TIME_FEATURES
from csj.metrics import direction_label
from csj.utils.tool import MODEL_FEATURES
from model.kronos import auto_regressive_inference


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _sanitize_paths(paths: np.ndarray) -> np.ndarray:
    sanitized = paths.copy()
    open_values = sanitized[..., 0]
    close_values = sanitized[..., 3]
    sanitized[..., 1] = np.maximum.reduce(
        [sanitized[..., 1], open_values, close_values]
    )
    sanitized[..., 2] = np.minimum.reduce(
        [sanitized[..., 2], open_values, close_values]
    )
    sanitized[..., 4:] = np.maximum(sanitized[..., 4:], 0.0)
    return sanitized


def _path_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def predict_cases(
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    cases: Sequence[ForecastCase],
    *,
    device: torch.device,
    max_context: int,
    clip: float,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    if not cases:
        raise ValueError("At least one forecast case is required")
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    model.eval()
    tokenizer.eval()
    set_seed(seed)

    grouped: dict[int, list[ForecastCase]] = defaultdict(list)
    for case in cases:
        grouped[case.pred_len].append(case)

    rows: list[dict[str, object]] = []
    for pred_len in sorted(grouped):
        same_length_cases = grouped[pred_len]
        for offset in range(0, len(same_length_cases), batch_size):
            batch_cases = same_length_cases[offset : offset + batch_size]
            normalized_contexts: list[np.ndarray] = []
            context_stamps: list[np.ndarray] = []
            target_stamps: list[np.ndarray] = []
            means: list[np.ndarray] = []
            stds: list[np.ndarray] = []

            for case in batch_cases:
                context = case.context[MODEL_FEATURES].to_numpy(dtype=np.float64)
                mean = context.mean(axis=0)
                std = context.std(axis=0)
                normalized = (context - mean) / (std + 1e-5)
                normalized = np.clip(normalized, -clip, clip).astype(np.float32)
                normalized_contexts.append(normalized)
                context_stamps.append(
                    case.context[TIME_FEATURES].to_numpy(dtype=np.float32)
                )
                target_stamps.append(
                    case.target[TIME_FEATURES].to_numpy(dtype=np.float32)
                )
                means.append(mean)
                stds.append(std)

            x = torch.from_numpy(np.stack(normalized_contexts)).to(device)
            x_stamp = torch.from_numpy(np.stack(context_stamps)).to(device)
            y_stamp = torch.from_numpy(np.stack(target_stamps)).to(device)
            normalized_samples = auto_regressive_inference(
                tokenizer,
                model,
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
            )
            normalized_samples = normalized_samples[:, :, -pred_len:, :]

            means_array = np.stack(means)[:, None, None, :]
            stds_array = np.stack(stds)[:, None, None, :]
            samples = normalized_samples * (stds_array + 1e-5) + means_array
            samples = _sanitize_paths(samples)

            for batch_index, case in enumerate(batch_cases):
                sample_paths = samples[batch_index]
                mean_path = sample_paths.mean(axis=0)
                current_close = float(case.context["close"].iloc[-1])
                actual_close = float(case.target["close"].iloc[-1])
                predicted_close = float(mean_path[-1, 3])
                actual_return = actual_close / current_close - 1.0
                predicted_return = predicted_close / current_close - 1.0
                actual_close_path = case.target["close"].to_numpy(dtype=np.float64)
                predicted_close_path = mean_path[:, 3].astype(np.float64)
                actual_range = float(
                    (case.target["high"].max() - case.target["low"].min())
                    / current_close
                )
                predicted_range = float(
                    (mean_path[:, 1].max() - mean_path[:, 2].min()) / current_close
                )
                opening_gap = float(case.target["open"].iloc[0] / current_close - 1.0)
                sample_final_returns = sample_paths[:, -1, 3] / current_close - 1.0

                rows.append(
                    {
                        "instrument": case.instrument,
                        "target_day": case.target_day,
                        "pred_len": pred_len,
                        "current_close": current_close,
                        "actual_close": actual_close,
                        "predicted_close": predicted_close,
                        "actual_return": actual_return,
                        "predicted_return": predicted_return,
                        "actual_direction": direction_label(actual_return),
                        "predicted_direction": direction_label(predicted_return),
                        "up_probability": float(np.mean(sample_final_returns > 0)),
                        "opening_gap": opening_gap,
                        "large_opening_gap": bool(abs(opening_gap) >= 0.03),
                        "actual_range": actual_range,
                        "predicted_range": predicted_range,
                        "range_relative_error": float(
                            abs(predicted_range - actual_range) / (actual_range + 1e-12)
                        ),
                        "close_path_correlation": _path_correlation(
                            actual_close_path, predicted_close_path
                        ),
                        "actual_close_path": actual_close_path.tolist(),
                        "predicted_close_path": predicted_close_path.tolist(),
                        "sample_final_returns": sample_final_returns.astype(float).tolist(),
                        "target_timestamps": [
                            timestamp.isoformat()
                            for timestamp in case.target["timestamps"].tolist()
                        ],
                    }
                )

            del x, x_stamp, y_stamp
    return pd.DataFrame(rows).sort_values(["target_day", "instrument"]).reset_index(drop=True)
