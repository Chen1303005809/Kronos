from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from csj.v7.audit import build_origin_day_folds
from csj.v7.config import load_v7_config, validate_v7_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v7_config_freezes_diversified_origin_day_protocol() -> None:
    config = load_v7_config(REPO_ROOT / "csj/configs/risk_control_v7.yaml")

    assert config["experiment"]["version"] == 7
    assert config["walk_forward"]["minimum_fit_days"] == 60
    assert config["walk_forward"]["split_key"] == "origin_trading_day"
    assert len(config["data"]["products"]) == 21
    assert config["data"]["gate_products"] == ["i", "jm", "rb"]

    mutated = deepcopy(config)
    mutated["walk_forward"]["minimum_fit_days"] = 90
    with pytest.raises(ValueError, match="minimum_fit_days"):
        validate_v7_config(mutated)


def test_v7_origin_day_folds_are_atomic_and_exactly_five() -> None:
    days = list(pd.bdate_range("2025-01-02", periods=220))
    folds = build_origin_day_folds(
        days,
        minimum_fit_days=60,
        inner_validation_days=20,
        evaluation_days=20,
        step_days=20,
        purge_days=3,
        fold_count=5,
    )

    assert len(folds) == 5
    assert folds[0].fit_end_day == days[59]
    assert folds[0].inner_validation_start_day == days[63]
    assert folds[0].evaluation_start_day == days[86]
    assert folds[-1].evaluation_end_day == days[185]
    for fold in folds:
        assert fold.fit_end_day < fold.inner_validation_start_day
        assert fold.inner_validation_end_day < fold.evaluation_start_day
