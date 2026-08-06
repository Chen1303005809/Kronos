from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from csj.evaluation import set_seed
from csj.futures_data import (
    TIME_FEATURES,
    ThreeTradingDayCase,
    fit_context_normalization,
    group_three_day_cases_by_length,
)
from csj.metrics import (
    compute_three_day_path_metrics,
    direction_label,
    range_relative_error,
    three_day_endpoint_returns,
)
from csj.utils.tool import MODEL_FEATURES
from model.kronos import auto_regressive_inference


def _sanitize_paths(paths: np.ndarray) -> np.ndarray:
    sanitized = np.asarray(paths, dtype=np.float64).copy()
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


def _path_diagnostics(paths: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(paths, dtype=np.float64)
    nonfinite = ~np.isfinite(values)
    ohlc_violations = (
        (values[..., 1] < values[..., 0])
        | (values[..., 1] < values[..., 3])
        | (values[..., 2] > values[..., 0])
        | (values[..., 2] > values[..., 3])
    )
    negative_flows = np.any(values[..., 4:] < 0, axis=-1)
    return {
        "raw_nonfinite_values": int(nonfinite.sum()),
        "raw_nonfinite_rate": float(nonfinite.mean()),
        "raw_ohlc_violation_bars": int(ohlc_violations.sum()),
        "raw_ohlc_violation_rate": float(ohlc_violations.mean()),
        "raw_negative_flow_bars": int(negative_flows.sum()),
        "raw_negative_flow_rate": float(negative_flows.mean()),
    }


def _case_record(
    case: ThreeTradingDayCase,
    sample_paths: np.ndarray,
    *,
    model_name: str,
    point_estimate: str,
    turning_point_threshold: float,
    diagnostics: Mapping[str, float | int] | None = None,
    raw_sample_paths: np.ndarray | None = None,
    sampling_seed: int | None = None,
) -> dict[str, object]:
    samples = np.asarray(sample_paths, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != (case.pred_len, len(MODEL_FEATURES)):
        raise ValueError("sample_paths must have [sample, pred_len, feature] shape")
    if not np.isfinite(samples).all():
        raise ValueError("Non-finite generated paths cannot be evaluated")
    if point_estimate not in ("mean", "median"):
        raise ValueError("point_estimate must be 'mean' or 'median'")

    mean_path = samples.mean(axis=0)
    median_path = np.median(samples, axis=0)
    q10_path = np.quantile(samples, 0.10, axis=0)
    q90_path = np.quantile(samples, 0.90, axis=0)
    predicted_path = mean_path if point_estimate == "mean" else median_path
    actual_path = case.target[MODEL_FEATURES].to_numpy(dtype=np.float64)
    origin_close = float(case.context["close"].iloc[-1])
    path_metrics = compute_three_day_path_metrics(
        actual_close=actual_path[:, 3],
        predicted_close=predicted_path[:, 3],
        origin_close=origin_close,
        day_end_indices=case.day_end_indices,
        turning_point_threshold=turning_point_threshold,
    )
    path_metrics["range_relative_error"] = range_relative_error(
        actual_path[:, 1],
        actual_path[:, 2],
        predicted_path[:, 1],
        predicted_path[:, 2],
        origin_close,
    )

    sample_endpoint_returns = np.stack(
        [
            np.asarray(
                three_day_endpoint_returns(
                    sample[:, 3], origin_close, case.day_end_indices
                ),
                dtype=np.float64,
            )
            for sample in samples
        ],
        axis=0,
    )
    endpoint_up_probabilities = np.mean(sample_endpoint_returns > 0, axis=0)
    opening_gap = float(actual_path[0, 0] / origin_close - 1.0)
    return {
        "model": model_name,
        "instrument": case.instrument,
        "fold_id": case.fold_id,
        "split": case.split,
        "origin_timestamp": case.origin_timestamp,
        "origin_trading_day": case.origin_trading_day,
        "target_day": case.target_days[0],
        "target_days": case.target_days,
        "target_timestamps": case.target["timestamps"].tolist(),
        "pred_len": case.pred_len,
        "day_end_indices": case.day_end_indices,
        "point_estimate": point_estimate,
        "sampling_seed": sampling_seed,
        "origin_close": origin_close,
        "opening_gap": opening_gap,
        "large_opening_gap": bool(abs(opening_gap) >= 0.03),
        "actual_path": actual_path.tolist(),
        "sample_paths": samples.tolist(),
        "raw_sample_paths": (
            np.asarray(raw_sample_paths, dtype=np.float64).tolist()
            if raw_sample_paths is not None
            else samples.tolist()
        ),
        "mean_path": mean_path.tolist(),
        "median_path": median_path.tolist(),
        "q10_path": q10_path.tolist(),
        "q90_path": q90_path.tolist(),
        "predicted_path": predicted_path.tolist(),
        "sample_endpoint_returns": sample_endpoint_returns.tolist(),
        "day1_up_probability": float(endpoint_up_probabilities[0]),
        "day2_up_probability": float(endpoint_up_probabilities[1]),
        "day3_up_probability": float(endpoint_up_probabilities[2]),
        **path_metrics,
        **(dict(diagnostics) if diagnostics is not None else {}),
    }


def predict_three_day_cases(
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    cases: Sequence[ThreeTradingDayCase],
    *,
    device: torch.device,
    max_context: int,
    clip: float,
    normalization_epsilon: float,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    batch_size: int,
    seed: int,
    point_estimate: str,
    turning_point_threshold: float,
    model_name: str = "zero_shot",
) -> pd.DataFrame:
    if not cases:
        raise ValueError("At least one three-day case is required")
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    model.eval()
    tokenizer.eval()
    set_seed(seed)

    rows: list[dict[str, object]] = []
    for pred_len, same_length_cases in group_three_day_cases_by_length(cases).items():
        for offset in range(0, len(same_length_cases), batch_size):
            batch_cases = same_length_cases[offset : offset + batch_size]
            normalized_contexts: list[np.ndarray] = []
            context_stamps: list[np.ndarray] = []
            target_stamps: list[np.ndarray] = []
            statistics = []
            for case in batch_cases:
                stats = fit_context_normalization(
                    case.context,
                    clip=clip,
                    epsilon=normalization_epsilon,
                )
                context = case.context[MODEL_FEATURES].to_numpy(dtype=np.float64)
                normalized_contexts.append(stats.transform(context).astype(np.float32))
                context_stamps.append(
                    case.context[TIME_FEATURES].to_numpy(dtype=np.float32)
                )
                target_stamps.append(
                    case.target[TIME_FEATURES].to_numpy(dtype=np.float32)
                )
                statistics.append(stats)

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
            )[:, :, -pred_len:, :]

            for batch_index, case in enumerate(batch_cases):
                raw_paths = statistics[batch_index].inverse(
                    normalized_samples[batch_index]
                )
                diagnostics = _path_diagnostics(raw_paths)
                if diagnostics["raw_nonfinite_values"]:
                    raise RuntimeError(
                        f"Non-finite path generated for {case.instrument} "
                        f"{case.target_days[0].date()}"
                    )
                sample_paths = _sanitize_paths(raw_paths)
                rows.append(
                    _case_record(
                        case,
                        sample_paths,
                        model_name=model_name,
                        point_estimate=point_estimate,
                        turning_point_threshold=turning_point_threshold,
                        diagnostics=diagnostics,
                        raw_sample_paths=raw_paths,
                        sampling_seed=seed,
                    )
                )
            del x, x_stamp, y_stamp, normalized_samples

    records = pd.DataFrame(rows).sort_values(
        ["target_day", "instrument"], kind="stable"
    ).reset_index(drop=True)
    records["inference_device"] = str(device)
    return records


def _interpolated_feature_path(
    case: ThreeTradingDayCase,
    endpoint_returns: Sequence[float],
) -> np.ndarray:
    if len(endpoint_returns) != 3:
        raise ValueError("Three endpoint returns are required")
    origin_close = float(case.context["close"].iloc[-1])
    return_values: list[float] = []
    previous_return = 0.0
    previous_end = -1
    for endpoint_return, day_end in zip(
        endpoint_returns, case.day_end_indices, strict=True
    ):
        bar_count = day_end - previous_end
        return_values.extend(
            np.linspace(
                previous_return,
                float(endpoint_return),
                bar_count + 1,
                dtype=np.float64,
            )[1:].tolist()
        )
        previous_return = float(endpoint_return)
        previous_end = day_end

    close = origin_close * (1.0 + np.asarray(return_values, dtype=np.float64))
    last_features = case.context[MODEL_FEATURES].iloc[-1].to_numpy(dtype=np.float64)
    path = np.repeat(last_features[None, :], case.pred_len, axis=0)
    path[:, 0] = close
    path[:, 1] = close
    path[:, 2] = close
    path[:, 3] = close
    return path


def make_three_day_baselines(
    training_cases: Sequence[ThreeTradingDayCase],
    evaluation_cases: Sequence[ThreeTradingDayCase],
    *,
    point_estimate: str,
    turning_point_threshold: float,
) -> dict[str, pd.DataFrame]:
    directions: dict[str, list[list[int]]] = defaultdict(list)
    for case in training_cases:
        origin_close = float(case.context["close"].iloc[-1])
        returns = three_day_endpoint_returns(
            case.target["close"].to_numpy(dtype=np.float64),
            origin_close,
            case.day_end_indices,
        )
        directions[case.instrument].append(
            [direction_label(value) for value in returns]
        )

    majority: dict[str, tuple[int, int, int]] = {}
    for instrument, instrument_directions in directions.items():
        direction_array = np.asarray(instrument_directions, dtype=np.int8)
        endpoint_majorities: list[int] = []
        for endpoint_index in range(3):
            values = direction_array[:, endpoint_index]
            up_count = int(np.sum(values == 1))
            down_count = int(np.sum(values == -1))
            endpoint_majorities.append(1 if up_count >= down_count else -1)
        majority[instrument] = tuple(endpoint_majorities)  # type: ignore[assignment]

    rows: dict[str, list[dict[str, object]]] = {
        "majority": [],
        "momentum": [],
        "persistence": [],
    }
    for case in evaluation_cases:
        if case.instrument not in majority:
            raise ValueError(f"No training majority for {case.instrument}")
        majority_returns = [
            direction * 1e-12 for direction in majority[case.instrument]
        ]
        origin_day = case.origin_trading_day
        recent_day = case.context.loc[
            case.context["trading_day"] == origin_day
        ]
        recent_return = float(
            recent_day["close"].iloc[-1] / recent_day["open"].iloc[0] - 1.0
        )
        momentum_returns = [
            (1.0 + recent_return) ** day_number - 1.0
            for day_number in (1, 2, 3)
        ]
        paths = {
            "majority": _interpolated_feature_path(case, majority_returns),
            "momentum": _interpolated_feature_path(case, momentum_returns),
            "persistence": _interpolated_feature_path(case, [0.0, 0.0, 0.0]),
        }
        for model_name, path in paths.items():
            rows[model_name].append(
                _case_record(
                    case,
                    path[None, ...],
                    model_name=model_name,
                    point_estimate=point_estimate,
                    turning_point_threshold=turning_point_threshold,
                )
            )
    return {
        model_name: pd.DataFrame(model_rows).sort_values(
            ["target_day", "instrument"], kind="stable"
        ).reset_index(drop=True)
        for model_name, model_rows in rows.items()
    }
