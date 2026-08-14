"""Standalone three-trading-day probabilistic Kronos forecast workflow.

This module deliberately has no dependency on the project's ``csj`` package.
It only consumes the same futures-provider JSON payload shape and the native
Kronos model implementation from ``model``.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kronos-standalone-three-day-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import torch

from model.kronos import auto_regressive_inference


MODEL_FEATURES = ["open", "high", "low", "close", "volume", "amount"]
TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]
VALID_BAR_COUNTS = (5, 7)

DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-small"
DEFAULT_MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
DEFAULT_MAX_CONTEXT = 512
DEFAULT_LOOKBACK = 256
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 0.9
DEFAULT_CLIP = 5.0
DEFAULT_NORMALIZATION_EPSILON = 1e-5
DEFAULT_TURNING_POINT_THRESHOLD = 0.0005
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ThreeDayForecastConfig:
    lookback: int = DEFAULT_LOOKBACK
    sample_count: int = DEFAULT_SAMPLE_COUNT
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    max_context: int = DEFAULT_MAX_CONTEXT
    clip: float = DEFAULT_CLIP
    normalization_epsilon: float = DEFAULT_NORMALIZATION_EPSILON
    turning_point_threshold: float = DEFAULT_TURNING_POINT_THRESHOLD
    seed: int = DEFAULT_SEED
    device: str = "auto"
    tokenizer_id: str = DEFAULT_TOKENIZER_ID
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION

    def __post_init__(self) -> None:
        if self.lookback < 1 or self.max_context < 1:
            raise ValueError("lookback and max_context must be positive")
        if self.lookback > self.max_context:
            raise ValueError("lookback cannot exceed max_context")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.clip <= 0 or self.normalization_epsilon <= 0:
            raise ValueError("clip and normalization_epsilon must be positive")
        if self.turning_point_threshold < 0:
            raise ValueError("turning_point_threshold cannot be negative")


@dataclass(frozen=True)
class TradingDayCase:
    instrument: str
    origin_timestamp: pd.Timestamp
    origin_trading_day: pd.Timestamp
    context: pd.DataFrame
    target: pd.DataFrame
    target_days: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]
    day_end_indices: tuple[int, int, int]
    split: str

    @property
    def pred_len(self) -> int:
        return len(self.target)


@dataclass(frozen=True)
class PreparedThreeDayForecast:
    source_path: Path
    instrument: str
    frame: pd.DataFrame
    audit: dict[str, object]
    target_case: TradingDayCase
    training_cases: tuple[TradingDayCase, ...]


@dataclass(frozen=True)
class ThreeDayForecastResult:
    prepared: PreparedThreeDayForecast
    config: ThreeDayForecastConfig
    device: torch.device
    kronos_records: pd.DataFrame
    baseline_records: dict[str, pd.DataFrame]
    path_table: pd.DataFrame
    metrics: dict[str, object]
    diagnostics: dict[str, object]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_provider_payload(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Input JSON must be a provider object with a list-valued data field")
    instrument = str(payload.get("Ins", "unknown"))
    records: list[dict[str, object]] = []
    for item in payload["data"]:
        if not isinstance(item, dict):
            raise ValueError("Every provider data record must be an object")
        records.append(
            {
                "instrument": instrument,
                "open": item.get("O"),
                "high": item.get("H"),
                "low": item.get("L"),
                "close": item.get("C"),
                "cumulative_volume": item.get("V"),
                "volume_delta": item.get("VD"),
                "cumulative_amount": item.get("A"),
                "open_interest": item.get("OI"),
                "calendar_day": item.get("TeD"),
                "trading_day_raw": item.get("TiD"),
                "bar_time": item.get("T"),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("Input JSON contains no K-line records")
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "cumulative_volume",
        "volume_delta",
        "cumulative_amount",
        "open_interest",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame["timestamps"] = pd.to_datetime(
        frame["calendar_day"].astype(str) + " " + frame["bar_time"].astype(str),
        format="%Y%m%d %H:%M:%S",
        errors="raise",
    )
    frame["trading_day"] = pd.to_datetime(
        frame["trading_day_raw"].astype(str), format="%Y%m%d", errors="raise"
    )
    frame = frame.sort_values("timestamps", kind="stable").reset_index(drop=True)
    grouped = frame.groupby("trading_day", sort=False)
    derived_volume = grouped["cumulative_volume"].diff()
    derived_amount = grouped["cumulative_amount"].diff()
    first_bar = grouped.cumcount().eq(0)
    derived_volume.loc[first_bar] = frame.loc[first_bar, "cumulative_volume"]
    derived_amount.loc[first_bar] = frame.loc[first_bar, "cumulative_amount"]
    frame["volume"] = frame["volume_delta"].fillna(derived_volume)
    frame["amount"] = derived_amount
    frame = frame[
        [
            "instrument",
            "timestamps",
            "trading_day",
            *MODEL_FEATURES,
            "open_interest",
            "calendar_day",
            "bar_time",
            "cumulative_volume",
            "cumulative_amount",
        ]
    ]
    _validate_frame(frame)
    frame.attrs["source"] = str(source)
    audit = {
        "instrument": instrument,
        "source_path": str(source),
        "raw_bars": int(len(frame)),
        "raw_trading_days": int(frame["trading_day"].nunique()),
    }
    return frame, audit


def _validate_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("K-line payload contains no bars")
    if frame[MODEL_FEATURES + ["timestamps", "trading_day"]].isna().any().any():
        raise ValueError("K-line payload contains missing model fields")
    if frame["timestamps"].duplicated().any():
        raise ValueError("K-line payload contains duplicate timestamps")
    if not frame["timestamps"].is_monotonic_increasing:
        raise ValueError("K-line timestamps are not monotonically increasing")
    if (frame[["volume", "amount"]] < 0).any().any():
        raise ValueError("Volume and amount must be non-negative")
    high_floor = frame[["open", "close", "low"]].max(axis=1)
    low_ceiling = frame[["open", "close", "high"]].min(axis=1)
    if (frame["high"] < high_floor).any() or (frame["low"] > low_ceiling).any():
        raise ValueError("Invalid OHLC relationship found")


def _clean_structural_anomalies(
    frame: pd.DataFrame,
    valid_bar_counts: Sequence[int] = VALID_BAR_COUNTS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    valid_counts = {int(value) for value in valid_bar_counts}
    day_counts = frame.groupby("trading_day", sort=True).size()
    invalid_counts = day_counts.loc[~day_counts.isin(valid_counts)]
    valid_days = day_counts.loc[day_counts.isin(valid_counts)].index
    cleaned = frame.loc[frame["trading_day"].isin(valid_days)].copy()
    cleaned = cleaned.sort_values("timestamps", kind="stable").reset_index(drop=True)
    audit = {
        "instrument": str(frame["instrument"].iloc[0]),
        "raw_bars": int(len(frame)),
        "clean_bars": int(len(cleaned)),
        "raw_trading_days": int(len(day_counts)),
        "clean_trading_days": int(len(valid_days)),
        "valid_day_counts": {
            str(count): int((day_counts == count).sum())
            for count in sorted(valid_counts)
        },
        "removed_days": [
            {"trading_day": day.strftime("%Y-%m-%d"), "bars": int(count)}
            for day, count in invalid_counts.items()
        ],
        "first_timestamp": (
            cleaned["timestamps"].min().isoformat() if not cleaned.empty else None
        ),
        "last_timestamp": (
            cleaned["timestamps"].max().isoformat() if not cleaned.empty else None
        ),
    }
    return cleaned, audit


def _day_end_indices(bar_counts: Sequence[int]) -> tuple[int, int, int]:
    counts = tuple(int(value) for value in bar_counts)
    if len(counts) != 3 or any(count not in VALID_BAR_COUNTS for count in counts):
        raise ValueError("Each three-day target must contain 5 or 7 bars per day")
    cumulative = np.cumsum(counts, dtype=np.int64) - 1
    return tuple(int(value) for value in cumulative)  # type: ignore[return-value]


def _build_cases(
    frame: pd.DataFrame,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    lookback: int,
    split: str,
) -> list[TradingDayCase]:
    instrument = str(frame["instrument"].iloc[0])
    groups = [
        (pd.Timestamp(day).normalize(), group.index.to_numpy(dtype=np.int64))
        for day, group in frame.groupby("trading_day", sort=True)
    ]
    cases: list[TradingDayCase] = []
    for offset in range(max(len(groups) - 2, 0)):
        selected = groups[offset : offset + 3]
        target_days = tuple(day for day, _ in selected)
        if not all(start_day.normalize() <= day <= end_day.normalize() for day in target_days):
            continue
        counts = [len(indices) for _, indices in selected]
        day_ends = _day_end_indices(counts)
        target_indices = np.concatenate([indices for _, indices in selected])
        target_start = int(target_indices[0])
        if target_start < lookback:
            continue
        context = frame.iloc[target_start - lookback : target_start].copy()
        target = frame.loc[target_indices].copy()
        if len(context) != lookback:
            continue
        cases.append(
            TradingDayCase(
                instrument=instrument,
                origin_timestamp=pd.Timestamp(context["timestamps"].iloc[-1]),
                origin_trading_day=pd.Timestamp(context["trading_day"].iloc[-1]).normalize(),
                context=context,
                target=target,
                target_days=target_days,  # type: ignore[arg-type]
                day_end_indices=day_ends,
                split=split,
            )
        )
    return sorted(cases, key=lambda case: case.target_days[0])


def prepare_json_three_day_forecast(
    input_path: str | Path,
    *,
    target_start_day: str | pd.Timestamp | None = None,
    lookback: int = DEFAULT_LOOKBACK,
) -> PreparedThreeDayForecast:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    frame, audit = _parse_provider_payload(input_path)
    frame, clean_audit = _clean_structural_anomalies(frame)
    if frame.empty:
        raise ValueError("JSON payload has no complete trading days after cleaning")
    day_counts = frame.groupby("trading_day", sort=True).size()
    days = [pd.Timestamp(day).normalize() for day in day_counts.index]
    starts = days[:-2]
    if not starts:
        raise ValueError("JSON payload does not contain three complete trading days")
    selected = starts[-1] if target_start_day is None else pd.Timestamp(target_start_day).normalize()
    if selected not in set(starts):
        raise ValueError(f"target_start_day {selected.date()} is not a complete three-day window start")
    index = days.index(selected)
    target_days = days[index : index + 3]
    target_cases = _build_cases(
        frame,
        start_day=target_days[0],
        end_day=target_days[-1],
        lookback=lookback,
        split="evaluation",
    )
    if len(target_cases) != 1:
        raise ValueError(f"Expected one target case, found {len(target_cases)}")
    target_case = target_cases[0]
    training_end = target_days[0] - pd.DateOffset(days=1)
    training_cases = _build_cases(
        frame,
        start_day=days[0],
        end_day=training_end,
        lookback=lookback,
        split="train",
    )
    if not training_cases:
        raise ValueError("At least one historical three-day case is required for majority baseline")
    audit = {
        **audit,
        **clean_audit,
        "source_path": str(Path(input_path).expanduser().resolve()),
        "lookback": int(lookback),
        "target_start_day": target_case.target_days[0],
        "target_days": list(target_case.target_days),
        "target_pred_len": target_case.pred_len,
        "training_case_count": len(training_cases),
    }
    return PreparedThreeDayForecast(
        source_path=Path(input_path).expanduser().resolve(),
        instrument=target_case.instrument,
        frame=frame,
        audit=audit,
        target_case=target_case,
        training_cases=tuple(training_cases),
    )


def load_native_kronos_models(
    *,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    from model import Kronos, KronosTokenizer

    kwargs: dict[str, object] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    tokenizer = KronosTokenizer.from_pretrained(
        tokenizer_id, revision=tokenizer_revision, **kwargs
    )
    model = Kronos.from_pretrained(model_id, revision=model_revision, **kwargs)
    tokenizer.eval()
    model.eval()
    tokenizer.requires_grad_(False)
    model.requires_grad_(False)
    return model, tokenizer


def _time_features(timestamps: pd.Series) -> np.ndarray:
    values = pd.to_datetime(timestamps)
    return np.column_stack(
        [
            values.dt.minute.to_numpy(),
            values.dt.hour.to_numpy(),
            values.dt.weekday.to_numpy(),
            values.dt.day.to_numpy(),
            values.dt.month.to_numpy(),
        ]
    ).astype(np.float32)


def _sanitize_paths(paths: np.ndarray) -> np.ndarray:
    sanitized = np.asarray(paths, dtype=np.float64).copy()
    sanitized[..., 1] = np.maximum.reduce(
        [sanitized[..., 1], sanitized[..., 0], sanitized[..., 3]]
    )
    sanitized[..., 2] = np.minimum.reduce(
        [sanitized[..., 2], sanitized[..., 0], sanitized[..., 3]]
    )
    sanitized[..., 4:] = np.maximum(sanitized[..., 4:], 0.0)
    return sanitized


def _path_diagnostics(paths: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(paths, dtype=np.float64)
    ohlc_violations = (
        (values[..., 1] < values[..., 0])
        | (values[..., 1] < values[..., 3])
        | (values[..., 2] > values[..., 0])
        | (values[..., 2] > values[..., 3])
    )
    negative_flows = np.any(values[..., 4:] < 0, axis=-1)
    nonfinite = ~np.isfinite(values)
    return {
        "raw_nonfinite_values": int(nonfinite.sum()),
        "raw_nonfinite_rate": float(nonfinite.mean()),
        "raw_ohlc_violation_bars": int(ohlc_violations.sum()),
        "raw_ohlc_violation_rate": float(ohlc_violations.mean()),
        "raw_negative_flow_bars": int(negative_flows.sum()),
        "raw_negative_flow_rate": float(negative_flows.mean()),
    }


def _direction_label(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _endpoint_returns(close: np.ndarray, origin_close: float, indices: Sequence[int]) -> np.ndarray:
    cumulative = np.asarray(close, dtype=np.float64) / origin_close - 1.0
    return cumulative[np.asarray(indices, dtype=np.int64)]


def _path_correlation(actual: np.ndarray, predicted: np.ndarray, origin_close: float) -> float:
    actual_returns = actual / origin_close - 1.0
    predicted_returns = predicted / origin_close - 1.0
    if np.std(actual_returns) == 0 or np.std(predicted_returns) == 0:
        return float("nan")
    return float(np.corrcoef(actual_returns, predicted_returns)[0, 1])


def _case_record(
    case: TradingDayCase,
    sample_paths: np.ndarray,
    *,
    model_name: str,
    sampling_seed: int | None = None,
    raw_sample_paths: np.ndarray | None = None,
    point_estimate: str = "median",
) -> dict[str, object]:
    samples = np.asarray(sample_paths, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != (case.pred_len, len(MODEL_FEATURES)):
        raise ValueError("sample_paths must have [sample, pred_len, feature] shape")
    if not np.isfinite(samples).all():
        raise ValueError("Non-finite generated paths cannot be evaluated")
    if point_estimate not in {"mean", "median"}:
        raise ValueError("point_estimate must be mean or median")
    mean_path = samples.mean(axis=0)
    median_path = np.median(samples, axis=0)
    q10_path = np.quantile(samples, 0.10, axis=0)
    q90_path = np.quantile(samples, 0.90, axis=0)
    predicted_path = mean_path if point_estimate == "mean" else median_path
    actual_path = case.target[MODEL_FEATURES].to_numpy(dtype=np.float64)
    origin_close = float(case.context["close"].iloc[-1])
    actual_endpoints = _endpoint_returns(actual_path[:, 3], origin_close, case.day_end_indices)
    predicted_endpoints = _endpoint_returns(predicted_path[:, 3], origin_close, case.day_end_indices)
    sample_endpoint_returns = np.asarray(
        [_endpoint_returns(sample[:, 3], origin_close, case.day_end_indices) for sample in samples]
    )
    result: dict[str, object] = {
        "model": model_name,
        "instrument": case.instrument,
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
        "actual_path": actual_path.tolist(),
        "sample_endpoint_returns": sample_endpoint_returns.tolist(),
        "day1_up_probability": float(np.mean(sample_endpoint_returns[:, 0] > 0)),
        "day2_up_probability": float(np.mean(sample_endpoint_returns[:, 1] > 0)),
        "day3_up_probability": float(np.mean(sample_endpoint_returns[:, 2] > 0)),
        "return_path_correlation": _path_correlation(
            actual_path[:, 3], predicted_path[:, 3], origin_close
        ),
        "range_relative_error": float(
            abs(
                (
                    (predicted_path[:, 1].max() - predicted_path[:, 2].min())
                    - (actual_path[:, 1].max() - actual_path[:, 2].min())
                )
                / origin_close
            )
            / (
                abs((actual_path[:, 1].max() - actual_path[:, 2].min()) / origin_close)
                + 1e-12
            )
        ),
    }
    for day_number, (actual_return, predicted_return) in enumerate(
        zip(actual_endpoints, predicted_endpoints, strict=True), start=1
    ):
        result[f"day{day_number}_actual_return"] = float(actual_return)
        result[f"day{day_number}_predicted_return"] = float(predicted_return)
        result[f"day{day_number}_actual_direction"] = _direction_label(float(actual_return))
        result[f"day{day_number}_path_direction"] = _direction_label(float(predicted_return))
        result[f"day{day_number}_endpoint_absolute_error"] = float(
            abs(predicted_return - actual_return)
        )
    return result


def _interpolated_feature_path(
    case: TradingDayCase,
    endpoint_returns: Sequence[float],
) -> np.ndarray:
    values: list[float] = []
    previous_return = 0.0
    previous_end = -1
    for endpoint_return, day_end in zip(endpoint_returns, case.day_end_indices, strict=True):
        bar_count = day_end - previous_end
        values.extend(
            np.linspace(previous_return, float(endpoint_return), bar_count + 1)[1:].tolist()
        )
        previous_return = float(endpoint_return)
        previous_end = day_end
    origin_close = float(case.context["close"].iloc[-1])
    close = origin_close * (1.0 + np.asarray(values, dtype=np.float64))
    last_features = case.context[MODEL_FEATURES].iloc[-1].to_numpy(dtype=np.float64)
    path = np.repeat(last_features[None, :], case.pred_len, axis=0)
    path[:, :4] = close[:, None]
    return path


def _make_baselines(
    training_cases: Sequence[TradingDayCase],
    evaluation_case: TradingDayCase,
) -> dict[str, pd.DataFrame]:
    training_directions: list[list[int]] = []
    for case in training_cases:
        origin = float(case.context["close"].iloc[-1])
        endpoints = _endpoint_returns(
            case.target["close"].to_numpy(dtype=np.float64), origin, case.day_end_indices
        )
        training_directions.append([_direction_label(float(value)) for value in endpoints])
    directions = np.asarray(training_directions, dtype=np.int8)
    majority = [
        1 if int(np.sum(directions[:, index] == 1)) >= int(np.sum(directions[:, index] == -1)) else -1
        for index in range(3)
    ]
    origin_day = evaluation_case.origin_trading_day
    recent_day = evaluation_case.context.loc[
        evaluation_case.context["trading_day"] == origin_day
    ]
    recent_return = float(
        recent_day["close"].iloc[-1] / recent_day["open"].iloc[0] - 1.0
    )
    momentum = [(1.0 + recent_return) ** day_number - 1.0 for day_number in (1, 2, 3)]
    endpoint_paths = {
        "majority": [direction * 1e-12 for direction in majority],
        "momentum": momentum,
        "persistence": [0.0, 0.0, 0.0],
    }
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in endpoint_paths}
    for name, endpoints in endpoint_paths.items():
        path = _interpolated_feature_path(evaluation_case, endpoints)
        rows[name].append(
            _case_record(evaluation_case, path[None, ...], model_name=name)
        )
    return {name: pd.DataFrame(values) for name, values in rows.items()}


def _run_native_sampling(
    prepared: PreparedThreeDayForecast,
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    config: ThreeDayForecastConfig,
) -> tuple[pd.DataFrame, torch.device]:
    device = resolve_device(config.device)
    set_seed(config.seed)
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    model.eval()
    tokenizer.eval()
    case = prepared.target_case
    context = case.context[MODEL_FEATURES].to_numpy(dtype=np.float64)
    mean = context.mean(axis=0)
    std = context.std(axis=0)
    normalized_context = np.clip(
        (context - mean) / (std + config.normalization_epsilon),
        -config.clip,
        config.clip,
    ).astype(np.float32)
    x = torch.from_numpy(normalized_context[None, ...]).to(device)
    x_stamp = torch.from_numpy(_time_features(case.context["timestamps"])[None, ...]).to(device)
    y_stamp = torch.from_numpy(_time_features(case.target["timestamps"])[None, ...]).to(device)
    normalized_samples = auto_regressive_inference(
        tokenizer,
        model,
        x,
        x_stamp,
        y_stamp,
        max_context=config.max_context,
        pred_len=case.pred_len,
        clip=config.clip,
        T=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        sample_count=config.sample_count,
        verbose=False,
        return_samples=True,
    )[:, :, -case.pred_len :, :]
    raw_paths = normalized_samples[0] * (std + config.normalization_epsilon) + mean
    raw_paths = np.asarray(raw_paths, dtype=np.float64)
    diagnostics = _path_diagnostics(raw_paths)
    if diagnostics["raw_nonfinite_values"]:
        raise RuntimeError("Native Kronos produced non-finite values")
    paths = _sanitize_paths(raw_paths)
    record = _case_record(
        case,
        paths,
        model_name="kronos_native",
        sampling_seed=config.seed,
        raw_sample_paths=raw_paths,
    )
    record.update(diagnostics)
    return pd.DataFrame([record]), device


def _record_dict(frame: pd.DataFrame) -> dict[str, object]:
    if len(frame) != 1:
        raise ValueError("A single-case forecast record was expected")
    return frame.iloc[0].to_dict()


def _feature_array(record: Mapping[str, object], field: str) -> np.ndarray:
    values = np.asarray(record[field], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(MODEL_FEATURES):
        raise ValueError(f"{field} must have shape [bars, {len(MODEL_FEATURES)}]")
    return values


def _close_metrics(record: Mapping[str, object], with_interval: bool) -> dict[str, object]:
    actual = _feature_array(record, "actual_path")[:, 3]
    predicted = _feature_array(record, "predicted_path")[:, 3]
    origin_close = float(record["origin_close"])
    actual_returns = actual / origin_close - 1.0
    predicted_returns = predicted / origin_close - 1.0
    metrics: dict[str, object] = {
        "close_mae": float(np.mean(np.abs(predicted - actual))),
        "close_rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "close_mape": float(np.mean(np.abs((predicted - actual) / np.maximum(np.abs(actual), 1e-12)))),
        "cumulative_return_mae": float(np.mean(np.abs(predicted_returns - actual_returns))),
        "actual_final_return": float(actual_returns[-1]),
        "predicted_final_return": float(predicted_returns[-1]),
        "final_return_absolute_error": float(abs(predicted_returns[-1] - actual_returns[-1])),
        "day_endpoint_returns": {
            f"day{day}": {
                "actual": float(record[f"day{day}_actual_return"]),
                "predicted": float(record[f"day{day}_predicted_return"]),
                "absolute_error": float(record[f"day{day}_endpoint_absolute_error"]),
                "actual_direction": int(record[f"day{day}_actual_direction"]),
                "predicted_direction": int(record[f"day{day}_path_direction"]),
            }
            for day in (1, 2, 3)
        },
    }
    if with_interval:
        q10 = _feature_array(record, "q10_path")[:, 3]
        q90 = _feature_array(record, "q90_path")[:, 3]
        inside = (actual >= q10) & (actual <= q90)
        metrics.update(
            {
                "interval_coverage": float(np.mean(inside)),
                "interval_mean_width": float(np.mean(q90 - q10)),
                "interval_mean_width_return": float(np.mean((q90 - q10) / origin_close)),
                "interval_outside_bars": int(np.sum(~inside)),
            }
        )
    return metrics


def _three_day_metrics(record: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {"endpoints": {}, "path": {}}
    endpoints = output["endpoints"]
    for day in (1, 2, 3):
        actual = int(record[f"day{day}_actual_direction"])
        predicted = int(record[f"day{day}_path_direction"])
        endpoints[f"day{day}"] = {
            "actual_return": float(record[f"day{day}_actual_return"]),
            "predicted_return": float(record[f"day{day}_predicted_return"]),
            "absolute_error": float(record[f"day{day}_endpoint_absolute_error"]),
            "actual_direction": actual,
            "predicted_direction": predicted,
            "direction_hit": bool(actual != 0 and actual == predicted),
        }
    output["path"] = {
        "return_path_correlation": record.get("return_path_correlation"),
        "range_relative_error": record.get("range_relative_error"),
    }
    return output


def _build_metrics(
    kronos_records: pd.DataFrame,
    baseline_records: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    all_records = {"kronos": kronos_records, **baseline_records}
    models: dict[str, object] = {}
    three_day: dict[str, object] = {}
    for name, records in all_records.items():
        record = _record_dict(records)
        models[name] = _close_metrics(record, with_interval=name == "kronos")
        three_day[name] = _three_day_metrics(record)
    kronos_mae = float(models["kronos"]["close_mae"])
    comparisons = {
        name: {
            "baseline_close_mae_minus_kronos": float(models[name]["close_mae"]) - kronos_mae,
            "kronos_better_close_mae": kronos_mae < float(models[name]["close_mae"]),
        }
        for name in baseline_records
    }
    return {"models": models, "three_day": three_day, "baseline_comparison": comparisons}


def build_path_table(
    prepared: PreparedThreeDayForecast,
    kronos_records: pd.DataFrame,
    baseline_records: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    target = prepared.target_case.target.reset_index(drop=True)
    kronos = _record_dict(kronos_records)
    target_days = list(dict.fromkeys(pd.Timestamp(day).normalize() for day in target["trading_day"]))
    table: dict[str, object] = {
        "bar_index": np.arange(len(target), dtype=np.int64),
        "timestamp": target["timestamps"].tolist(),
        "trading_day": target["trading_day"].tolist(),
        "day_number": [target_days.index(pd.Timestamp(day).normalize()) + 1 for day in target["trading_day"]],
    }
    actual = target[MODEL_FEATURES].to_numpy(dtype=np.float64)
    for index, feature in enumerate(MODEL_FEATURES):
        table[f"actual_{feature}"] = actual[:, index]
    for quantile, field in (("q10", "q10_path"), ("q50", "median_path"), ("q90", "q90_path")):
        values = _feature_array(kronos, field)
        for index, feature in enumerate(MODEL_FEATURES):
            table[f"kronos_{quantile}_{feature}"] = values[:, index]
    origin_close = float(prepared.target_case.context["close"].iloc[-1])
    for column in ("actual_close", "kronos_q10_close", "kronos_q50_close", "kronos_q90_close"):
        table[column.replace("close", "close_return")] = np.asarray(table[column]) / origin_close - 1.0
    for name, records in baseline_records.items():
        values = _feature_array(_record_dict(records), "predicted_path")
        for index, feature in enumerate(MODEL_FEATURES):
            table[f"{name}_{feature}"] = values[:, index]
        table[f"{name}_close_return"] = values[:, 3] / origin_close - 1.0
    return pd.DataFrame(table)


def run_native_three_day_forecast(
    prepared: PreparedThreeDayForecast,
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    *,
    config: ThreeDayForecastConfig = ThreeDayForecastConfig(),
) -> ThreeDayForecastResult:
    kronos_records, device = _run_native_sampling(prepared, model, tokenizer, config)
    baseline_records = _make_baselines(prepared.training_cases, prepared.target_case)
    path_table = build_path_table(prepared, kronos_records, baseline_records)
    metrics = _build_metrics(kronos_records, baseline_records)
    kronos_record = _record_dict(kronos_records)
    diagnostics = {
        "raw_generation": {
            field: kronos_record[field]
            for field in (
                "raw_nonfinite_values",
                "raw_nonfinite_rate",
                "raw_ohlc_violation_bars",
                "raw_ohlc_violation_rate",
                "raw_negative_flow_bars",
                "raw_negative_flow_rate",
            )
        },
        "target": {
            "origin_timestamp": prepared.target_case.origin_timestamp,
            "origin_trading_day": prepared.target_case.origin_trading_day,
            "target_days": list(prepared.target_case.target_days),
            "day_end_indices": list(prepared.target_case.day_end_indices),
            "pred_len": prepared.target_case.pred_len,
        },
    }
    return ThreeDayForecastResult(
        prepared=prepared,
        config=config,
        device=device,
        kronos_records=kronos_records,
        baseline_records=baseline_records,
        path_table=path_table,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _relative_close(values: Sequence[float], origin_close: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / origin_close - 1.0


def plot_three_day_forecast(result: ThreeDayForecastResult, output_path: str | Path) -> Path:
    prepared = result.prepared
    target = prepared.target_case.target.reset_index(drop=True)
    context = prepared.target_case.context.reset_index(drop=True)
    kronos = _record_dict(result.kronos_records)
    origin_close = float(context["close"].iloc[-1])
    forecast_x = pd.DatetimeIndex(
        [pd.Timestamp(prepared.target_case.origin_timestamp), *pd.DatetimeIndex(target["timestamps"]).tolist()]
    )
    actual_y = np.concatenate(([0.0], _relative_close(target["close"], origin_close)))
    q10_y = np.concatenate(([0.0], _relative_close(_feature_array(kronos, "q10_path")[:, 3], origin_close)))
    q50_y = np.concatenate(([0.0], _relative_close(_feature_array(kronos, "median_path")[:, 3], origin_close)))
    q90_y = np.concatenate(([0.0], _relative_close(_feature_array(kronos, "q90_path")[:, 3], origin_close)))
    fig, axis = plt.subplots(figsize=(14, 7))
    axis.plot(context["timestamps"], _relative_close(context["close"], origin_close), color="#6b7280", linewidth=1.2, label="input close")
    axis.plot(forecast_x, actual_y, color="black", linewidth=2.1, label="actual close")
    axis.fill_between(forecast_x, q10_y, q90_y, color="#2563eb", alpha=0.18, label="Kronos 10–90%")
    axis.plot(forecast_x, q50_y, color="#2563eb", linewidth=2.0, linestyle="-", label="Kronos median")
    styles = {
        "persistence": ("#dc2626", "persistence"),
        "momentum": ("#eab308", "momentum"),
        "majority": ("#7c3aed", "majority"),
    }
    for name, (color, label) in styles.items():
        baseline = _record_dict(result.baseline_records[name])
        values = np.concatenate(([0.0], _relative_close(_feature_array(baseline, "predicted_path")[:, 3], origin_close)))
        axis.plot(forecast_x, values, color=color, linewidth=1.4, linestyle="--", label=label)
    for day_end in prepared.target_case.day_end_indices[:-1]:
        axis.axvline(pd.Timestamp(target["timestamps"].iloc[day_end]), color="#9ca3af", linewidth=0.9, linestyle=":")
    target_days = [day.strftime("%Y-%m-%d") for day in prepared.target_case.target_days]
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8)
    axis.set_title(f"{prepared.instrument} · standalone native Kronos forecast\ntarget: {', '.join(target_days)} · samples={result.config.sample_count}")
    axis.set_ylabel("Close return from forecast origin")
    axis.set_xlabel("Timestamp")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}"))
    locator = mdates.AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.grid(alpha=0.22)
    axis.legend(loc="best", ncol=2)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def _record_summary(record: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "model", "instrument", "split", "origin_timestamp", "origin_trading_day",
        "target_day", "target_days", "target_timestamps", "pred_len", "day_end_indices",
        "point_estimate", "sampling_seed", "origin_close", "sample_paths", "raw_sample_paths",
        "mean_path", "median_path", "q10_path", "q90_path", "predicted_path", "actual_path",
        "sample_endpoint_returns", "day1_up_probability", "day2_up_probability", "day3_up_probability",
        "raw_nonfinite_values", "raw_ohlc_violation_bars", "raw_negative_flow_bars",
    )
    return {field: record[field] for field in fields if field in record}


def write_forecast_artifacts(result: ThreeDayForecastResult, output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "paths_csv": destination / "forecast_paths.csv",
        "result_json": destination / "forecast_result.json",
        "metrics_json": destination / "metrics.json",
        "plot": destination / "forecast_plot.png",
    }
    result.path_table.to_csv(paths["paths_csv"], index=False)
    _write_json(
        paths["result_json"],
        {
            "source": {
                "path": result.prepared.source_path,
                "instrument": result.prepared.instrument,
                "audit": result.prepared.audit,
            },
            "config": {
                **{
                    field: getattr(result.config, field)
                    for field in (
                        "lookback", "sample_count", "temperature", "top_k", "top_p",
                        "max_context", "clip", "normalization_epsilon", "turning_point_threshold",
                        "seed", "tokenizer_id", "tokenizer_revision", "model_id", "model_revision",
                    )
                },
                "device": str(result.device),
            },
            "target": result.diagnostics["target"],
            "kronos": _record_summary(_record_dict(result.kronos_records)),
            "baselines": {
                name: _record_summary(_record_dict(records))
                for name, records in result.baseline_records.items()
            },
            "metrics": result.metrics,
            "diagnostics": result.diagnostics,
        },
    )
    _write_json(paths["metrics_json"], result.metrics)
    plot_three_day_forecast(result, paths["plot"])
    return paths

