"""Configuration loading and frozen invariants for the V6 risk workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from csj.v6 import PRODUCTION_ELIGIBLE, RESULT_SCOPE, STRATEGY_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_LABEL_RULE_VERSION = "adverse-excursion-vol-scaled-v1"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _mapping(value: object, *, scope: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"V6 configuration section {scope} must be a mapping")
    return value


def _exact(mapping: Mapping[str, Any], key: str, expected: object, *, scope: str) -> None:
    if mapping.get(key) != expected:
        raise ValueError(f"V6 fixes {scope}.{key} at {expected!r}")


def _exact_int(mapping: Mapping[str, Any], key: str, expected: int, *, scope: str) -> None:
    try:
        actual = int(mapping.get(key, -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V6 fixes {scope}.{key} at {expected}") from exc
    if actual != expected:
        raise ValueError(f"V6 fixes {scope}.{key} at {expected}")


def _exact_float(
    mapping: Mapping[str, Any], key: str, expected: float, *, scope: str
) -> None:
    try:
        actual = float(mapping.get(key, float("nan")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V6 fixes {scope}.{key} at {expected}") from exc
    if actual != expected:
        raise ValueError(f"V6 fixes {scope}.{key} at {expected}")


def load_v6_config(path: str | Path) -> dict[str, Any]:
    """Load V6 YAML and resolve every repository-relative storage path."""

    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V6 configuration must be a YAML mapping")
    config: dict[str, Any] = value
    data = _mapping(config.setdefault("data", {}), scope="data")
    model = _mapping(config.setdefault("model", {}), scope="model")
    output = _mapping(config.setdefault("output", {}), scope="output")
    overlay = _mapping(config.setdefault("overlay", {}), scope="overlay")
    if "snapshot_root" not in data:
        raise ValueError("V6 config must set data.snapshot_root")
    data["snapshot_root"] = str(_resolve(data["snapshot_root"]))
    model["cache_dir"] = str(_resolve(model.get("cache_dir", "csj/artifacts/hf_cache")))
    output["root"] = str(_resolve(output.get("root", "csj/runs/risk_control_v6")))
    output["results_root"] = str(
        _resolve(output.get("results_root", "csj/results/risk_control_v6"))
    )
    base_positions = overlay.get("base_positions_path")
    if base_positions is not None:
        overlay["base_positions_path"] = str(_resolve(str(base_positions)))
    validate_v6_config(config)
    return config


def validate_v6_config(config: Mapping[str, Any]) -> None:
    """Reject protocol drift before a label, path, or model is evaluated."""

    experiment = _mapping(config.get("experiment"), scope="experiment")
    _exact_int(experiment, "version", STRATEGY_VERSION, scope="experiment")
    _exact(experiment, "result_scope", RESULT_SCOPE, scope="experiment")
    _exact(experiment, "production_eligible", PRODUCTION_ELIGIBLE, scope="experiment")

    data = _mapping(config.get("data"), scope="data")
    _exact(data, "data_rule_version", "target-only-observed-contract-data-v1", scope="data")
    _exact_int(data, "lookback", 256, scope="data")
    _exact_int(data, "horizon_trading_days", 3, scope="data")
    if tuple(int(value) for value in data.get("valid_bar_counts", ())) != (5, 7):
        raise ValueError("V6 accepts only five- or seven-bar complete trading days")
    if tuple(str(value) for value in data.get("products", ())) != ("i", "j", "jm", "rb"):
        raise ValueError("V6 fixes the observed product universe at i/j/jm/rb")
    if tuple(str(value) for value in data.get("primary_products", ())) != ("i", "jm", "rb"):
        raise ValueError("V6 fixes primary products at i/jm/rb")
    if tuple(str(value) for value in data.get("transfer_products", ())) != ("j",):
        raise ValueError("V6 fixes the transfer-only product at j")
    _exact_float(data, "clip", 5.0, scope="data")
    _exact_float(data, "normalization_epsilon", 1e-5, scope="data")

    labels = _mapping(config.get("risk_labels"), scope="risk_labels")
    _exact(labels, "version", RISK_LABEL_RULE_VERSION, scope="risk_labels")
    _exact(labels, "origin_price", "close", scope="risk_labels")
    _exact(labels, "long_adverse_price", "low", scope="risk_labels")
    _exact(labels, "short_adverse_price", "high", scope="risk_labels")
    _exact(labels, "severity_transform", "log1p", scope="risk_labels")
    _exact(
        labels,
        "future_volatility_target",
        "log_ratio_to_context_horizon_scale",
        scope="risk_labels",
    )
    volatility = _mapping(labels.get("context_volatility"), scope="risk_labels.context_volatility")
    _exact(volatility, "method", "ewma_close_log_return", scope="risk_labels.context_volatility")
    _exact_int(volatility, "bars", 60, scope="risk_labels.context_volatility")
    _exact_int(volatility, "halflife_bars", 20, scope="risk_labels.context_volatility")
    _exact(volatility, "adjust", False, scope="risk_labels.context_volatility")
    _exact(volatility, "bias", False, scope="risk_labels.context_volatility")
    _exact_float(volatility, "floor", 1e-5, scope="risk_labels.context_volatility")
    _exact(
        volatility,
        "horizon_scaling",
        "sqrt_target_bar_count",
        scope="risk_labels.context_volatility",
    )
    tail = _mapping(labels.get("tail_event"), scope="risk_labels.tail_event")
    _exact_float(tail, "quantile", 0.8, scope="risk_labels.tail_event")
    _exact(tail, "quantile_method", "linear", scope="risk_labels.tail_event")
    _exact(
        tail,
        "comparison",
        "greater_than_or_equal",
        scope="risk_labels.tail_event",
    )
    _exact(
        tail,
        "threshold_source",
        "outer_fit_primary_cases_only",
        scope="risk_labels.tail_event",
    )

    walk = _mapping(config.get("walk_forward"), scope="walk_forward")
    for key, expected in {
        "minimum_fit_days": 60,
        "inner_validation_days": 20,
        "evaluation_days": 20,
        "step_days": 20,
        "purge_days": 3,
        "fold_count": 5,
    }.items():
        _exact_int(walk, key, expected, scope="walk_forward")

    p0 = _mapping(config.get("p0"), scope="p0")
    for key, expected in {
        "minimum_fit_events_per_side": 100,
        "minimum_validation_events_per_side": 20,
        "minimum_evaluation_events_per_side": 20,
        "minimum_pooled_evaluation_events_per_product_side": 20,
        "maximum_integrity_failures": 0,
        "maximum_leakage_failures": 0,
    }.items():
        _exact_int(p0, key, expected, scope="p0")

    model = _mapping(config.get("model"), scope="model")
    _exact_int(model, "max_context", 512, scope="model")
    for key in ("tokenizer_id", "tokenizer_revision", "predictor_id", "predictor_revision"):
        if not str(model.get(key, "")).strip():
            raise ValueError(f"V6 model configuration is missing {key}")
    _exact(model, "freeze_tokenizer", True, scope="model")
    _exact(model, "freeze_predictor", True, scope="model")

    path_bank = _mapping(config.get("path_bank"), scope="path_bank")
    _exact_int(path_bank, "sample_count", 64, scope="path_bank")
    _exact_float(path_bank, "temperature", 1.0, scope="path_bank")
    _exact_int(path_bank, "top_k", 0, scope="path_bank")
    _exact_float(path_bank, "top_p", 0.9, scope="path_bank")
    _exact_int(path_bank, "random_seed", 20260812, scope="path_bank")
    _exact_int(path_bank, "minimum_valid_paths_per_case", 60, scope="path_bank")
    _exact_float(path_bank, "minimum_valid_path_fraction", 0.99, scope="path_bank")

    risk_head = _mapping(config.get("risk_head"), scope="risk_head")
    if tuple(int(seed) for seed in risk_head.get("seeds", ())) != (42, 43, 44):
        raise ValueError("V6 risk-head seeds are fixed at 42/43/44")
    for key, expected in {
        "kronos_projection_dim": 128,
        "feature_projection_dim": 64,
        "fusion_hidden_dim": 128,
    }.items():
        _exact_int(risk_head, key, expected, scope="risk_head")
    _exact_float(risk_head, "dropout", 0.0, scope="risk_head")
    expected_outputs = (
        "long_tail_logit",
        "short_tail_logit",
        "log1p_expected_long_mae",
        "log1p_expected_short_mae",
        "future_vol_ratio",
    )
    if tuple(str(value) for value in risk_head.get("outputs", ())) != expected_outputs:
        raise ValueError("V6 risk-head outputs must match the five pre-registered tasks")
    training = _mapping(risk_head.get("training"), scope="risk_head.training")
    for key, expected in {
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "gradient_clip": 3.0,
    }.items():
        _exact_float(training, key, expected, scope="risk_head.training")
    for key, expected in {
        "batch_size": 64,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "num_workers": 0,
    }.items():
        _exact_int(training, key, expected, scope="risk_head.training")
    _exact(
        training,
        "sampling_strategy",
        "prediction_day_product_uniform",
        scope="risk_head.training",
    )
    _exact(training, "checkpoint_metric", "validation_macro_brier", scope="risk_head.training")

    calibration = _mapping(config.get("calibration"), scope="calibration")
    _exact(calibration, "method", "affine_platt", scope="calibration")
    _exact(calibration, "source", "inner_validation_only", scope="calibration")
    _exact(calibration, "shared_across_primary_products", True, scope="calibration")
    _exact_float(calibration, "soft_alert_quantile", 0.8, scope="calibration")
    _exact_float(calibration, "hard_alert_quantile", 0.95, scope="calibration")

    overlay = _mapping(config.get("overlay"), scope="overlay")
    _exact(overlay, "policy", "position_reduction_only", scope="overlay")
    _exact_int(overlay, "execution_lag_bars", 1, scope="overlay")
    _exact(overlay, "overlapping_forecasts", "latest_only", scope="overlay")
    _exact(
        overlay,
        "require_external_base_positions_for_formal_gate",
        True,
        scope="overlay",
    )

    evaluation = _mapping(config.get("evaluation"), scope="evaluation")
    _exact_float(evaluation, "alert_budget", 0.2, scope="evaluation")
    _exact_int(evaluation, "bootstrap_iterations", 2000, scope="evaluation")
    if tuple(int(value) for value in evaluation.get("bootstrap_block_days", ())) != (5, 10):
        raise ValueError("V6 requires five- and ten-day paired block bootstraps")
    _exact_float(evaluation, "cvar_alpha", 0.05, scope="evaluation")
    _exact_float(evaluation, "minimum_return_retention", 0.85, scope="evaluation")

    runtime = _mapping(config.get("runtime"), scope="runtime")
    if str(runtime.get("device")) not in {"cuda", "cpu"}:
        raise ValueError("V6 runtime.device must be cuda or cpu")


__all__ = [
    "REPO_ROOT",
    "RISK_LABEL_RULE_VERSION",
    "load_v6_config",
    "validate_v6_config",
]
