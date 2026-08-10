from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from csj.futures_data import (
    MultiContractWindowDataset,
    build_forecast_cases,
    chronological_split,
    clean_structural_anomalies,
    load_contracts,
)
from csj.utils.tool import MODEL_FEATURES, d_to_df


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATHS = [
    REPO_ROOT / "csj/data/kline_rb8888.json",
    REPO_ROOT / "csj/data/kline_i8888.json",
]


def test_provider_conversion_uses_real_time_and_per_bar_flows() -> None:
    payload = {
        "Ins": "demo8888",
        "data": [
            {
                "TeD": "20240105",
                "TiD": "20240105",
                "T": "14:00:00",
                "O": 100,
                "H": 102,
                "L": 99,
                "C": 101,
                "V": 30,
                "VD": 10,
                "A": 3000,
                "OI": 50,
            },
            {
                "TeD": "20240105",
                "TiD": "20240108",
                "T": "21:00:00",
                "O": 101,
                "H": 103,
                "L": 100,
                "C": 102,
                "V": 12,
                "VD": 12,
                "A": 1200,
                "OI": 51,
            },
            {
                "TeD": "20240108",
                "TiD": "20240108",
                "T": "09:00:00",
                "O": 102,
                "H": 104,
                "L": 101,
                "C": 103,
                "V": 20,
                "VD": 8,
                "A": 2050,
                "OI": 52,
            },
        ],
    }

    frame = d_to_df(None, payload, persist_raw=False)

    assert frame["timestamps"].tolist() == [
        pd.Timestamp("2024-01-05 14:00:00"),
        pd.Timestamp("2024-01-05 21:00:00"),
        pd.Timestamp("2024-01-08 09:00:00"),
    ]
    assert frame["trading_day"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-08"),
    ]
    assert frame["volume"].tolist() == [10, 12, 8]
    assert frame["amount"].tolist() == [3000, 1200, 850]


def test_real_contract_cleaning_only_removes_invalid_day_shapes() -> None:
    frames, audits = load_contracts(CONTRACT_PATHS)

    assert set(frames) == {"rb8888", "i8888"}
    assert {audit["instrument"] for audit in audits} == {"rb8888", "i8888"}
    assert all(len(frame) < 4999 for frame in frames.values())
    for frame in frames.values():
        assert frame["timestamps"].is_monotonic_increasing
        assert set(frame.groupby("trading_day").size().unique()) <= {5, 7}
        assert not frame[MODEL_FEATURES].isna().any().any()
        assert (frame[["volume", "amount"]] >= 0).all().all()


def test_split_cases_are_chronological_and_keep_past_context() -> None:
    frames, _ = load_contracts(CONTRACT_PATHS)
    boundaries = chronological_split(frames)

    train_cases = build_forecast_cases(frames, boundaries, "train", lookback=256)
    val_cases = build_forecast_cases(frames, boundaries, "val", lookback=256)
    test_cases = build_forecast_cases(frames, boundaries, "test", lookback=256)

    assert train_cases and val_cases and test_cases
    assert max(case.target_day for case in train_cases) <= boundaries.train_end
    assert min(case.target_day for case in val_cases) > boundaries.train_end
    assert max(case.target_day for case in val_cases) <= boundaries.val_end
    assert min(case.target_day for case in test_cases) > boundaries.val_end
    for case in val_cases + test_cases:
        assert len(case.context) == 256
        assert len(case.target) in {5, 7}
        assert case.context["timestamps"].max() < case.target["timestamps"].min()


def test_training_windows_normalize_from_context_only() -> None:
    frames, _ = load_contracts(CONTRACT_PATHS)
    boundaries = chronological_split(frames)
    dataset = MultiContractWindowDataset(
        frames,
        boundaries,
        split="train",
        lookback=256,
        horizon=7,
    )

    normalized, stamps = dataset[0]

    assert normalized.shape == (263, 6)
    assert stamps.shape == (263, 5)
    instrument, forecast_start = dataset.indices[0]
    features, _, _ = dataset.series[instrument]
    raw_window = features[forecast_start - 256 : forecast_start + 7].astype(np.float64)
    context = raw_window[:256]
    expected = (raw_window - context.mean(axis=0)) / (context.std(axis=0) + 1e-5)
    expected = np.clip(expected, -5.0, 5.0).astype(np.float32)
    np.testing.assert_allclose(normalized.numpy(), expected, atol=1e-6)
    assert np.max(np.abs(normalized.numpy())) <= 5.0


def test_cleaner_reports_removed_days() -> None:
    raw = d_to_df(
        None,
        {
            "Ins": "tiny",
            "data": [
                {
                    "TeD": "20240101",
                    "TiD": "20240101",
                    "T": f"{hour:02d}:00:00",
                    "O": 1,
                    "H": 1,
                    "L": 1,
                    "C": 1,
                    "V": index + 1,
                    "VD": 1,
                    "A": index + 1,
                    "OI": 1,
                }
                for index, hour in enumerate([9, 10, 11, 13])
            ],
        },
        persist_raw=False,
    )

    cleaned, audit = clean_structural_anomalies(raw)

    assert cleaned.empty
    assert audit["removed_days"] == [{"trading_day": "2024-01-01", "bars": 4}]
