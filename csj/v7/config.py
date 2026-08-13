"""Strict configuration for the V7 P0 redesign."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from csj.v7 import PRODUCTION_ELIGIBLE, RESULT_SCOPE, STRATEGY_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_v7_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V7 configuration must be a mapping")
    data = value.setdefault("data", {})
    output = value.setdefault("output", {})
    data["snapshot_root"] = str(_resolve(data["snapshot_root"]))
    output["root"] = str(_resolve(output["root"]))
    output["results_root"] = str(_resolve(output["results_root"]))
    validate_v7_config(value)
    return value


def validate_v7_config(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    if int(experiment["version"]) != STRATEGY_VERSION:
        raise ValueError("V7 config must declare version 7")
    if experiment["result_scope"] != RESULT_SCOPE:
        raise ValueError(f"V7 result_scope must be {RESULT_SCOPE!r}")
    if bool(experiment["production_eligible"]) is not PRODUCTION_ELIGIBLE:
        raise ValueError("V7 remains non-production research")

    data = config["data"]
    if int(data["lookback"]) != 256:
        raise ValueError("V7 fixes lookback at 256")
    if int(data["horizon_trading_days"]) != 3:
        raise ValueError("V7 fixes risk horizon at three trading days")
    products = tuple(str(value) for value in data["products"])
    if len(products) < 10 or len(set(products)) != len(products):
        raise ValueError("V7 requires a unique diversified product pool")
    if tuple(str(value) for value in data["gate_products"]) != ("i", "jm", "rb"):
        raise ValueError("V7 preserves the core pooled gate for i/jm/rb")
    if str(data["study_start_day"]) != "2025-10-15":
        raise ValueError("V7 fixes the study start at the original V6 start day")
    coverage = data["coverage_filter"]
    if int(coverage["minimum_bars"]) != 1300:
        raise ValueError("V7 freezes coverage_filter.minimum_bars at 1300")
    if str(coverage["latest_first_bar"]) != "2025-10-01":
        raise ValueError("V7 freezes the latest allowed first bar at 2025-10-01")

    labels = config["risk_labels"]
    if float(labels["tail_quantile"]) != 0.8:
        raise ValueError("V7 fixes the adverse-event tail quantile at 0.8")
    if int(labels["volatility_bars"]) != 60 or int(labels["halflife_bars"]) != 20:
        raise ValueError("V7 fixes the past-volatility scale at EWMA 60/20")

    walk = config["walk_forward"]
    expected = {
        "minimum_fit_days": 60,
        "inner_validation_days": 20,
        "evaluation_days": 20,
        "step_days": 20,
        "purge_days": 3,
        "fold_count": 5,
    }
    for key, value in expected.items():
        if int(walk[key]) != value:
            raise ValueError(f"V7 fixes walk_forward.{key} at {value}")
    if walk["split_key"] != "origin_trading_day":
        raise ValueError("V7 requires prediction-origin-day atomic splits")

    gate = config["p0"]
    expected_gate = {
        "minimum_fit_events_per_side": 100,
        "minimum_validation_events_per_side": 20,
        "minimum_evaluation_events_per_side": 20,
        "minimum_pooled_evaluation_events_per_product_side": 20,
    }
    for key, value in expected_gate.items():
        if int(gate[key]) != value:
            raise ValueError(f"V7 fixes p0.{key} at {value}")


__all__ = ["load_v7_config", "validate_v7_config"]
