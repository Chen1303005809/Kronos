from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from csj.utils.tool import MODEL_FEATURES, d_to_df


SplitName = Literal["train", "val", "test"]
TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    first_day: pd.Timestamp
    last_day: pd.Timestamp
    train_days: int
    val_days: int
    test_days: int

    def contains(self, day: pd.Timestamp, split: SplitName) -> bool:
        if split == "train":
            return day <= self.train_end
        if split == "val":
            return self.train_end < day <= self.val_end
        if split == "test":
            return self.val_end < day <= self.last_day
        raise ValueError(f"Unknown split: {split}")


@dataclass(frozen=True)
class ForecastCase:
    instrument: str
    target_day: pd.Timestamp
    context: DataFrame
    target: DataFrame

    @property
    def pred_len(self) -> int:
        return len(self.target)


@dataclass(frozen=True)
class CasePeriod:
    """Inclusive target-day boundary for one split or walk-forward fold."""

    split: str
    start_day: pd.Timestamp
    end_day: pd.Timestamp
    fold_id: str | None = None

    def __post_init__(self) -> None:
        start_day = pd.Timestamp(self.start_day).normalize()
        end_day = pd.Timestamp(self.end_day).normalize()
        if start_day > end_day:
            raise ValueError("Case period start_day must not exceed end_day")
        object.__setattr__(self, "start_day", start_day)
        object.__setattr__(self, "end_day", end_day)

    def contains(self, day: pd.Timestamp) -> bool:
        normalized = pd.Timestamp(day).normalize()
        return self.start_day <= normalized <= self.end_day


@dataclass(frozen=True)
class ThreeTradingDayCase:
    """A 256-bar context followed by exactly three complete trading days."""

    instrument: str
    origin_timestamp: pd.Timestamp
    origin_trading_day: pd.Timestamp
    context: DataFrame
    target: DataFrame
    target_days: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]
    day_end_indices: tuple[int, int, int]
    split: str
    fold_id: str | None = None

    def __post_init__(self) -> None:
        target_days = tuple(pd.Timestamp(day).normalize() for day in self.target_days)
        if len(target_days) != 3 or len(set(target_days)) != 3:
            raise ValueError("A three-day case must contain three distinct target days")
        if len(self.context) == 0 or len(self.target) == 0:
            raise ValueError("Context and target must both be non-empty")

        observed_days = tuple(
            pd.Timestamp(day).normalize()
            for day in self.target["trading_day"].drop_duplicates().tolist()
        )
        if observed_days != target_days:
            raise ValueError("target_days must match the target trading-day groups")

        bar_counts = [
            int((self.target["trading_day"] == day).sum()) for day in target_days
        ]
        expected_indices = day_end_indices_from_bar_counts(bar_counts)
        if tuple(self.day_end_indices) != expected_indices:
            raise ValueError("day_end_indices do not match the complete target days")
        if expected_indices[-1] != len(self.target) - 1:
            raise ValueError("The third day-end index must be the final target row")

        context_end = pd.Timestamp(self.context["timestamps"].iloc[-1])
        target_start = pd.Timestamp(self.target["timestamps"].iloc[0])
        if context_end >= target_start:
            raise ValueError("Context must end strictly before the target begins")
        if pd.Timestamp(self.origin_timestamp) != context_end:
            raise ValueError("origin_timestamp must be the final context timestamp")
        context_day = pd.Timestamp(self.context["trading_day"].iloc[-1]).normalize()
        if pd.Timestamp(self.origin_trading_day).normalize() != context_day:
            raise ValueError("origin_trading_day must be the final context trading day")

        object.__setattr__(self, "origin_timestamp", context_end)
        object.__setattr__(self, "origin_trading_day", context_day)
        object.__setattr__(self, "target_days", target_days)
        object.__setattr__(self, "day_end_indices", expected_indices)

    @property
    def pred_len(self) -> int:
        return len(self.target)

    @property
    def target_day(self) -> pd.Timestamp:
        """First target day, retained as the forecast-case key."""

        return self.target_days[0]


@dataclass(frozen=True)
class ContextNormalization:
    """Feature statistics fitted only on a forecast case's context."""

    mean: np.ndarray
    std: np.ndarray
    clip: float = 5.0
    epsilon: float = 1e-5

    def transform(self, values: np.ndarray, *, apply_clip: bool = True) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.mean):
            raise ValueError("Feature count does not match normalization statistics")
        normalized = (array - self.mean) / (self.std + self.epsilon)
        if apply_clip:
            normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        array = np.asarray(normalized, dtype=np.float64)
        if array.shape[-1] != len(self.mean):
            raise ValueError("Feature count does not match normalization statistics")
        return array * (self.std + self.epsilon) + self.mean

    def clipping_mask(self, values: np.ndarray) -> np.ndarray:
        return np.abs(self.transform(values, apply_clip=False)) > self.clip


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_period: CasePeriod
    fit_period: CasePeriod
    inner_validation_period: CasePeriod
    evaluation_period: CasePeriod


def load_contract_json(path: str | Path) -> DataFrame:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    frame = d_to_df(None, payload, persist_raw=False)
    frame.attrs["source"] = str(source)
    return frame


def clean_structural_anomalies(
    frame: DataFrame,
    valid_bar_counts: Sequence[int] = (5, 7),
) -> tuple[DataFrame, dict[str, object]]:
    """Remove structurally incomplete trading days without filtering volatility."""

    instrument = str(frame["instrument"].iloc[0])
    day_counts = frame.groupby("trading_day", sort=True).size()
    valid_counts = {int(value) for value in valid_bar_counts}
    invalid_counts = day_counts.loc[~day_counts.isin(valid_counts)]
    valid_days = day_counts.loc[day_counts.isin(valid_counts)].index

    cleaned = frame.loc[frame["trading_day"].isin(valid_days)].copy()
    cleaned = cleaned.sort_values("timestamps", kind="stable").reset_index(drop=True)
    cleaned.attrs.update(frame.attrs)
    cleaned.attrs["instrument"] = instrument

    audit: dict[str, object] = {
        "instrument": instrument,
        "raw_bars": int(len(frame)),
        "clean_bars": int(len(cleaned)),
        "raw_trading_days": int(len(day_counts)),
        "clean_trading_days": int(len(valid_days)),
        "valid_day_counts": {
            str(int(bar_count)): int((day_counts == bar_count).sum())
            for bar_count in sorted(valid_counts)
        },
        "removed_days": [
            {"trading_day": day.strftime("%Y-%m-%d"), "bars": int(count)}
            for day, count in invalid_counts.items()
        ],
        "first_timestamp": cleaned["timestamps"].min().isoformat(),
        "last_timestamp": cleaned["timestamps"].max().isoformat(),
    }
    return cleaned, audit


def load_contracts(
    paths: Iterable[str | Path],
    valid_bar_counts: Sequence[int] = (5, 7),
) -> tuple[dict[str, DataFrame], list[dict[str, object]]]:
    frames: dict[str, DataFrame] = {}
    audits: list[dict[str, object]] = []
    for path in paths:
        raw_frame = load_contract_json(path)
        clean_frame, audit = clean_structural_anomalies(raw_frame, valid_bar_counts)
        instrument = str(clean_frame["instrument"].iloc[0])
        if instrument in frames:
            raise ValueError(f"Duplicate instrument: {instrument}")
        frames[instrument] = clean_frame
        audits.append(audit)
    if not frames:
        raise ValueError("At least one contract file is required")
    return frames, audits


def chronological_split(
    frames: Mapping[str, DataFrame],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitBoundaries:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Train, validation, and test ratios must be positive and sum to one")

    all_days = sorted(
        {
            pd.Timestamp(day)
            for frame in frames.values()
            for day in frame["trading_day"].unique()
        }
    )
    if len(all_days) < 3:
        raise ValueError("At least three trading days are required")

    train_days = int(len(all_days) * train_ratio)
    val_days = int(len(all_days) * val_ratio)
    test_days = len(all_days) - train_days - val_days
    if min(train_days, val_days, test_days) < 1:
        raise ValueError("Each chronological split must contain at least one trading day")

    return SplitBoundaries(
        train_end=pd.Timestamp(all_days[train_days - 1]),
        val_end=pd.Timestamp(all_days[train_days + val_days - 1]),
        first_day=pd.Timestamp(all_days[0]),
        last_day=pd.Timestamp(all_days[-1]),
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
    )


def add_time_features(frame: DataFrame) -> DataFrame:
    enriched = frame.copy()
    timestamps = enriched["timestamps"].dt
    enriched["minute"] = timestamps.minute
    enriched["hour"] = timestamps.hour
    enriched["weekday"] = timestamps.weekday
    enriched["day"] = timestamps.day
    enriched["month"] = timestamps.month
    return enriched


def split_case_period(
    boundaries: SplitBoundaries,
    split: SplitName,
) -> CasePeriod:
    """Convert the V1 chronological boundaries to an inclusive V2 period."""

    if split == "train":
        return CasePeriod(split, boundaries.first_day, boundaries.train_end)
    if split == "val":
        return CasePeriod(
            split,
            boundaries.train_end + pd.Timedelta(days=1),
            boundaries.val_end,
        )
    if split == "test":
        return CasePeriod(
            split,
            boundaries.val_end + pd.Timedelta(days=1),
            boundaries.last_day,
        )
    raise ValueError(f"Unknown split: {split}")


def day_end_indices_from_bar_counts(
    bar_counts: Sequence[int],
) -> tuple[int, int, int]:
    counts = tuple(int(value) for value in bar_counts)
    if len(counts) != 3:
        raise ValueError("Exactly three trading-day bar counts are required")
    if any(value not in (5, 7) for value in counts):
        raise ValueError("Each complete trading day must contain 5 or 7 bars")
    cumulative = np.cumsum(counts, dtype=np.int64) - 1
    return cast(tuple[int, int, int], tuple(int(value) for value in cumulative))


def build_three_trading_day_cases(
    frames: Mapping[str, DataFrame],
    period: CasePeriod,
    lookback: int = 256,
) -> list[ThreeTradingDayCase]:
    """Build per-instrument cases whose three target days stay inside ``period``."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    cases: list[ThreeTradingDayCase] = []
    for instrument, raw_frame in sorted(frames.items()):
        frame = add_time_features(raw_frame).sort_values(
            "timestamps", kind="stable"
        ).reset_index(drop=True)
        if frame.empty:
            continue
        frame_instruments = set(frame["instrument"].astype(str).unique())
        if frame_instruments != {instrument}:
            raise ValueError(
                f"Frame {instrument!r} contains instruments {sorted(frame_instruments)!r}"
            )

        day_groups = [
            (pd.Timestamp(day).normalize(), day_frame.index.to_numpy(dtype=np.int64))
            for day, day_frame in frame.groupby("trading_day", sort=False)
        ]
        for target_offset in range(max(len(day_groups) - 2, 0)):
            selected_groups = day_groups[target_offset : target_offset + 3]
            target_days = cast(
                tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp],
                tuple(day for day, _ in selected_groups),
            )
            if not all(period.contains(day) for day in target_days):
                continue

            bar_counts = [len(indices) for _, indices in selected_groups]
            day_end_indices = day_end_indices_from_bar_counts(bar_counts)
            target_indices = np.concatenate(
                [indices for _, indices in selected_groups]
            )
            target_start = int(target_indices[0])
            if target_start < lookback:
                continue
            context = frame.iloc[target_start - lookback : target_start].copy()
            target = frame.loc[target_indices].copy()
            if len(context) != lookback:
                continue

            cases.append(
                ThreeTradingDayCase(
                    instrument=instrument,
                    origin_timestamp=pd.Timestamp(context["timestamps"].iloc[-1]),
                    origin_trading_day=pd.Timestamp(
                        context["trading_day"].iloc[-1]
                    ),
                    context=context,
                    target=target,
                    target_days=target_days,
                    day_end_indices=day_end_indices,
                    split=period.split,
                    fold_id=period.fold_id,
                )
            )
    return sorted(cases, key=lambda case: (case.target_days[0], case.instrument))


def group_three_day_cases_by_length(
    cases: Sequence[ThreeTradingDayCase],
) -> dict[int, list[ThreeTradingDayCase]]:
    grouped: dict[int, list[ThreeTradingDayCase]] = {}
    for case in cases:
        if case.pred_len not in (15, 17, 19, 21):
            raise ValueError(f"Unexpected three-day target length: {case.pred_len}")
        grouped.setdefault(case.pred_len, []).append(case)
    return {length: grouped[length] for length in sorted(grouped)}


def fit_context_normalization(
    context: DataFrame | np.ndarray,
    *,
    clip: float = 5.0,
    epsilon: float = 1e-5,
    features: Sequence[str] = MODEL_FEATURES,
) -> ContextNormalization:
    if isinstance(context, pd.DataFrame):
        values = context[list(features)].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(context, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Context must be a non-empty [time, feature] matrix")
    if clip <= 0 or epsilon <= 0:
        raise ValueError("clip and epsilon must be positive")
    return ContextNormalization(
        mean=values.mean(axis=0),
        std=values.std(axis=0),
        clip=float(clip),
        epsilon=float(epsilon),
    )


def common_trading_days(frames: Mapping[str, DataFrame]) -> list[pd.Timestamp]:
    if not frames:
        raise ValueError("At least one instrument is required")
    day_sets = [
        {pd.Timestamp(day).normalize() for day in frame["trading_day"].unique()}
        for frame in frames.values()
    ]
    return sorted(set.intersection(*day_sets))


def build_expanding_walk_forward_folds(
    frames: Mapping[str, DataFrame],
    *,
    minimum_train_days: int,
    evaluation_days: int,
    step_days: int,
    inner_validation_days: int,
) -> list[WalkForwardFold]:
    """Create complete expanding folds on trading days shared by all instruments."""

    if min(minimum_train_days, evaluation_days, step_days, inner_validation_days) < 1:
        raise ValueError("Walk-forward day counts must be positive")
    if inner_validation_days >= minimum_train_days:
        raise ValueError("inner_validation_days must be smaller than minimum_train_days")

    days = common_trading_days(frames)
    folds: list[WalkForwardFold] = []
    evaluation_start = minimum_train_days
    fold_number = 0
    while evaluation_start + evaluation_days <= len(days):
        fold_id = f"fold_{fold_number:02d}"
        train_end_index = evaluation_start - 1
        inner_start_index = evaluation_start - inner_validation_days
        evaluation_end_index = evaluation_start + evaluation_days - 1
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_period=CasePeriod(
                    split="train",
                    start_day=days[0],
                    end_day=days[train_end_index],
                    fold_id=fold_id,
                ),
                fit_period=CasePeriod(
                    split="fit",
                    start_day=days[0],
                    end_day=days[inner_start_index - 1],
                    fold_id=fold_id,
                ),
                inner_validation_period=CasePeriod(
                    split="inner_validation",
                    start_day=days[inner_start_index],
                    end_day=days[train_end_index],
                    fold_id=fold_id,
                ),
                evaluation_period=CasePeriod(
                    split="evaluation",
                    start_day=days[evaluation_start],
                    end_day=days[evaluation_end_index],
                    fold_id=fold_id,
                ),
            )
        )
        fold_number += 1
        evaluation_start += step_days
    if not folds:
        raise ValueError("No complete walk-forward folds can be built")
    return folds


def build_forecast_cases(
    frames: Mapping[str, DataFrame],
    boundaries: SplitBoundaries,
    split: SplitName,
    lookback: int,
) -> list[ForecastCase]:
    cases: list[ForecastCase] = []
    for instrument, raw_frame in sorted(frames.items()):
        frame = add_time_features(raw_frame).reset_index(drop=True)
        for target_day, day_frame in frame.groupby("trading_day", sort=True):
            day = pd.Timestamp(target_day)
            if not boundaries.contains(day, split):
                continue
            target_indices = day_frame.index.to_numpy()
            target_start = int(target_indices[0])
            if target_start < lookback:
                continue
            context = frame.iloc[target_start - lookback : target_start].copy()
            target = frame.loc[target_indices].copy()
            if len(context) != lookback:
                continue
            cases.append(
                ForecastCase(
                    instrument=instrument,
                    target_day=day,
                    context=context,
                    target=target,
                )
            )
    return sorted(cases, key=lambda case: (case.target_day, case.instrument))


class MultiContractWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Hourly sliding windows with context-only normalization.

    Every sample contains ``lookback + horizon`` rows. Normalization statistics
    are computed only from the first ``lookback`` rows. The trainer is expected
    to calculate next-token loss only for the final ``horizon`` target rows.
    """

    def __init__(
        self,
        frames: Mapping[str, DataFrame],
        boundaries: SplitBoundaries,
        split: Literal["train", "val"],
        lookback: int = 256,
        horizon: int = 7,
        clip: float = 5.0,
    ) -> None:
        if lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must be positive")
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.clip = float(clip)
        self.boundaries = boundaries
        self.split = split
        self.series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.indices: list[tuple[str, int]] = []

        for instrument, raw_frame in sorted(frames.items()):
            frame = add_time_features(raw_frame).reset_index(drop=True)
            features = frame[MODEL_FEATURES].to_numpy(dtype=np.float32)
            stamps = frame[TIME_FEATURES].to_numpy(dtype=np.float32)
            trading_days = frame["trading_day"].to_numpy(dtype="datetime64[ns]")
            self.series[instrument] = (features, stamps, trading_days)

            first_forecast = self.lookback
            last_forecast = len(frame) - self.horizon
            for forecast_start in range(first_forecast, last_forecast + 1):
                target_days = trading_days[
                    forecast_start : forecast_start + self.horizon
                ]
                target_start_day = pd.Timestamp(target_days[0])
                target_end_day = pd.Timestamp(target_days[-1])
                if split == "train":
                    valid = target_end_day <= boundaries.train_end
                else:
                    valid = (
                        target_start_day > boundaries.train_end
                        and target_end_day <= boundaries.val_end
                    )
                if valid:
                    self.indices.append((instrument, forecast_start))

        if not self.indices:
            raise ValueError(f"No valid {split} windows were generated")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        instrument, forecast_start = self.indices[index]
        features, stamps, _ = self.series[instrument]
        context_start = forecast_start - self.lookback
        target_end = forecast_start + self.horizon
        window = features[context_start:target_end].copy()
        stamp_window = stamps[context_start:target_end].copy()

        context = window[: self.lookback].astype(np.float64)
        mean = context.mean(axis=0)
        std = context.std(axis=0)
        normalized = (window.astype(np.float64) - mean) / (std + 1e-5)
        normalized = np.clip(normalized, -self.clip, self.clip).astype(np.float32)

        return torch.from_numpy(normalized), torch.from_numpy(stamp_window)


class DenseInstrumentWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """V2 dense 21-bar stream for one instrument and one inclusive fold period."""

    def __init__(
        self,
        frame: DataFrame,
        period: CasePeriod,
        *,
        instrument: str,
        lookback: int = 256,
        horizon: int = 21,
        clip: float = 5.0,
        epsilon: float = 1e-5,
    ) -> None:
        if lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must be positive")
        if horizon != 21:
            raise ValueError("The V2 dense token stream must use horizon=21")
        frame_instruments = set(frame["instrument"].astype(str).unique())
        if frame_instruments != {instrument}:
            raise ValueError(
                f"Dense stream {instrument!r} received {sorted(frame_instruments)!r}"
            )
        self.instrument = instrument
        self.period = period
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.clip = float(clip)
        self.epsilon = float(epsilon)

        enriched = add_time_features(frame).sort_values(
            "timestamps", kind="stable"
        ).reset_index(drop=True)
        self.features = enriched[MODEL_FEATURES].to_numpy(dtype=np.float32)
        self.stamps = enriched[TIME_FEATURES].to_numpy(dtype=np.float32)
        self.trading_days = enriched["trading_day"].to_numpy(
            dtype="datetime64[ns]"
        )
        self.forecast_starts: list[int] = []
        for forecast_start in range(
            self.lookback,
            len(enriched) - self.horizon + 1,
        ):
            target_days = self.trading_days[
                forecast_start : forecast_start + self.horizon
            ]
            if period.contains(pd.Timestamp(target_days[0])) and period.contains(
                pd.Timestamp(target_days[-1])
            ):
                self.forecast_starts.append(forecast_start)
        if not self.forecast_starts:
            raise ValueError(
                f"No dense windows for {instrument} inside {period.split} "
                f"{period.start_day.date()}..{period.end_day.date()}"
            )

    def __len__(self) -> int:
        return len(self.forecast_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        forecast_start = self.forecast_starts[index]
        context_start = forecast_start - self.lookback
        target_end = forecast_start + self.horizon
        window = self.features[context_start:target_end].astype(np.float64)
        stamps = self.stamps[context_start:target_end].copy()
        stats = fit_context_normalization(
            window[: self.lookback],
            clip=self.clip,
            epsilon=self.epsilon,
        )
        normalized = stats.transform(window).astype(np.float32)
        return torch.from_numpy(normalized), torch.from_numpy(stamps)


def describe_splits(
    frames: Mapping[str, DataFrame],
    boundaries: SplitBoundaries,
    lookback: int,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "boundaries": {
            "first_day": boundaries.first_day.strftime("%Y-%m-%d"),
            "train_end": boundaries.train_end.strftime("%Y-%m-%d"),
            "val_end": boundaries.val_end.strftime("%Y-%m-%d"),
            "last_day": boundaries.last_day.strftime("%Y-%m-%d"),
            "train_days": boundaries.train_days,
            "val_days": boundaries.val_days,
            "test_days": boundaries.test_days,
        },
        "forecast_cases": {},
    }
    case_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        cases = build_forecast_cases(frames, boundaries, split, lookback)
        per_instrument: dict[str, int] = {}
        for case in cases:
            per_instrument[case.instrument] = per_instrument.get(case.instrument, 0) + 1
        case_counts[split] = {"total": len(cases), **per_instrument}
    summary["forecast_cases"] = case_counts
    return summary
