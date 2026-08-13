"""Strict configuration loading and frozen protocol validation for V7."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from csj.v7 import PRODUCTION_ELIGIBLE, RESULT_SCOPE, STRATEGY_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _mapping(value: object, *, scope: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V7 configuration section {scope} must be a mapping")
    return value


def _exact(mapping: Mapping[str, Any], key: str, expected: object, *, scope: str) -> None:
    if mapping.get(key) != expected:
        raise ValueError(f"V7 fixes {scope}.{key} at {expected!r}")


def _exact_int(mapping: Mapping[str, Any], key: str, expected: int, *, scope: str) -> None:
    try:
        actual = int(mapping.get(key, -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V7 fixes {scope}.{key} at {expected}") from exc
    if actual != expected:
        raise ValueError(f"V7 fixes {scope}.{key} at {expected}")


def _exact_float(
    mapping: Mapping[str, Any], key: str, expected: float, *, scope: str
) -> None:
    try:
        actual = float(mapping.get(key, float("nan")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V7 fixes {scope}.{key} at {expected}") from exc
    if actual != expected:
        raise ValueError(f"V7 fixes {scope}.{key} at {expected}")


def load_v7_config(path: str | Path) -> dict[str, Any]:
    """Load V7 YAML and resolve all storage paths before validation."""

    value = yaml.safe_load(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V7 configuration must be a mapping")
    config: dict[str, Any] = value
    data = config.setdefault("data", {})
    model = config.setdefault("model", {})
    path_bank = config.setdefault("path_bank", {})
    output = config.setdefault("output", {})
    overlay = config.setdefault("overlay", {})
    if "snapshot_root" not in data:
        raise ValueError("V7 config must set data.snapshot_root")
    data["snapshot_root"] = str(_resolve(data["snapshot_root"]))
    model["cache_dir"] = str(_resolve(model.get("cache_dir", "csj/artifacts/hf_cache")))
    path_bank["artifact_root"] = str(
        _resolve(path_bank.get("artifact_root", "csj/artifacts/v7_path_bank"))
    )
    output["root"] = str(_resolve(output.get("root", "csj/runs/risk_control_v7")))
    output["results_root"] = str(
        _resolve(output.get("results_root", "csj/results/risk_control_v7"))
    )
    if overlay.get("base_positions_path") is not None:
        overlay["base_positions_path"] = str(_resolve(str(overlay["base_positions_path"])))
    validate_v7_config(config)
    return config


def validate_v7_config(config: Mapping[str, Any]) -> None:
    """Reject any P0--P5 protocol drift before CUDA work begins."""

    experiment = _mapping(config.get("experiment"), scope="experiment")
    _exact_int(experiment, "version", STRATEGY_VERSION, scope="experiment")
    _exact(experiment, "result_scope", RESULT_SCOPE, scope="experiment")
    _exact(experiment, "production_eligible", PRODUCTION_ELIGIBLE, scope="experiment")

    data = _mapping(config.get("data"), scope="data")
    _exact_int(data, "lookback", 256, scope="data")
    _exact_int(data, "horizon_trading_days", 3, scope="data")
    if tuple(int(value) for value in data.get("valid_bar_counts", ())) != (5, 7):
        raise ValueError("V7 accepts only five- or seven-bar complete trading days")
    products = tuple(str(value) for value in data.get("products", ()))
    if len(products) != 21 or len(set(products)) != len(products):
        raise ValueError("V7 requires the frozen unique diversified product pool")
    expected_products = (
        "a", "b", "br", "bu", "c", "eb", "eg", "fu", "hc", "i", "jm",
        "l", "m", "p", "pg", "pp", "rb", "ru", "sp", "v", "y",
    )
    if products != expected_products:
        raise ValueError("V7 fixes the diversified product pool and ordering")
    if tuple(str(value) for value in data.get("gate_products", ())) != ("i", "jm", "rb"):
        raise ValueError("V7 preserves the core pooled gate for i/jm/rb")
    _exact(data, "study_start_day", "2025-10-15", scope="data")
    _exact_float(data, "clip", 5.0, scope="data")
    _exact_float(data, "normalization_epsilon", 1e-5, scope="data")
    coverage = _mapping(data.get("coverage_filter"), scope="data.coverage_filter")
    _exact_int(coverage, "minimum_bars", 1300, scope="data.coverage_filter")
    _exact(coverage, "latest_first_bar", "2025-10-01", scope="data.coverage_filter")

    labels = _mapping(config.get("risk_labels"), scope="risk_labels")
    _exact(labels, "version", "adverse-excursion-vol-scaled-v1", scope="risk_labels")
    _exact_int(labels, "volatility_bars", 60, scope="risk_labels")
    _exact_int(labels, "halflife_bars", 20, scope="risk_labels")
    _exact_float(labels, "tail_quantile", 0.8, scope="risk_labels")
    _exact(labels, "quantile_method", "linear", scope="risk_labels")

    model = _mapping(config.get("model"), scope="model")
    _exact(model, "tokenizer_id", "NeoQuasar/Kronos-Tokenizer-base", scope="model")
    _exact(model, "tokenizer_revision", "0e0117387f39004a9016484a186a908917e22426", scope="model")
    _exact(model, "predictor_id", "NeoQuasar/Kronos-small", scope="model")
    _exact(model, "predictor_revision", "901c26c1332695a2a8f243eb2f37243a37bea320", scope="model")
    _exact_int(model, "max_context", 512, scope="model")
    _exact(model, "freeze_tokenizer", True, scope="model")
    _exact(model, "freeze_predictor", True, scope="model")

    walk = _mapping(config.get("walk_forward"), scope="walk_forward")
    for key, value in {
        "minimum_fit_days": 60,
        "inner_validation_days": 20,
        "evaluation_days": 20,
        "step_days": 20,
        "purge_days": 3,
        "fold_count": 5,
    }.items():
        _exact_int(walk, key, value, scope="walk_forward")
    _exact(walk, "split_key", "origin_trading_day", scope="walk_forward")

    p0 = _mapping(config.get("p0"), scope="p0")
    for key, value in {
        "minimum_fit_events_per_side": 100,
        "minimum_validation_events_per_side": 20,
        "minimum_evaluation_events_per_side": 20,
        "minimum_pooled_evaluation_events_per_product_side": 20,
        "maximum_integrity_failures": 0,
        "maximum_leakage_failures": 0,
    }.items():
        _exact_int(p0, key, value, scope="p0")

    path_bank = _mapping(config.get("path_bank"), scope="path_bank")
    for key, value in {
        "expected_unique_cases": 5154,
        "sample_count": 64,
        "minimum_valid_paths_per_case": 60,
        "shard_case_count": 32,
        "determinism_check_cases": 1,
        "smoke_case_count": 8,
        "smoke_sample_count": 4,
    }.items():
        _exact_int(path_bank, key, value, scope="path_bank")
    for key, value in {
        "temperature": 1.0,
        "top_p": 0.9,
        "minimum_valid_path_fraction": 0.99,
    }.items():
        _exact_float(path_bank, key, value, scope="path_bank")
    _exact_int(path_bank, "top_k", 0, scope="path_bank")

    baselines = _mapping(config.get("baselines"), scope="baselines")
    expected_baselines = (
        "fit_global_event_rate",
        "fit_product_event_rate",
        "ewma_volatility_rank",
        "atr20_rank",
        "fixed_context_logistic",
        "zero_shot_path_risk",
    )
    if tuple(str(value) for value in baselines.get("selection_order", ())) != expected_baselines:
        raise ValueError("V7 baseline selection order must be pre-registered")
    _exact_int(baselines, "rank_probability_bins", 10, scope="baselines")
    logistic = _mapping(baselines.get("fixed_context_logistic"), scope="baselines.fixed_context_logistic")
    _exact(logistic, "solver", "iteratively_reweighted_least_squares", scope="baselines.fixed_context_logistic")
    _exact_int(logistic, "maximum_iterations", 100, scope="baselines.fixed_context_logistic")
    _exact_float(logistic, "l2_penalty", 0.001, scope="baselines.fixed_context_logistic")

    risk_head = _mapping(config.get("risk_head"), scope="risk_head")
    if tuple(int(value) for value in risk_head.get("seeds", ())) != (42, 43, 44):
        raise ValueError("V7 risk-head seeds are fixed at 42/43/44")
    for key, value in {
        "kronos_projection_dim": 128,
        "feature_projection_dim": 64,
        "fusion_hidden_dim": 128,
    }.items():
        _exact_int(risk_head, key, value, scope="risk_head")
    _exact_float(risk_head, "dropout", 0.0, scope="risk_head")
    if tuple(str(value) for value in risk_head.get("outputs", ())) != (
        "long_tail_logit", "short_tail_logit", "log1p_expected_long_mae",
        "log1p_expected_short_mae", "future_vol_ratio",
    ):
        raise ValueError("V7 risk-head outputs must match the five pre-registered tasks")
    training = _mapping(risk_head.get("training"), scope="risk_head.training")
    for key, value in {
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "gradient_clip": 3.0,
    }.items():
        _exact_float(training, key, value, scope="risk_head.training")
    for key, value in {
        "batch_size": 64,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "num_workers": 0,
    }.items():
        _exact_int(training, key, value, scope="risk_head.training")
    _exact(training, "sampling_strategy", "prediction_day_product_uniform", scope="risk_head.training")
    _exact(training, "checkpoint_metric", "validation_macro_brier", scope="risk_head.training")
    control = _mapping(risk_head.get("matched_control"), scope="risk_head.matched_control")
    _exact(control, "name", "context_feature_control", scope="risk_head.matched_control")
    _exact(control, "zero_kronos_hidden", True, scope="risk_head.matched_control")
    _exact(control, "zero_path_summaries", True, scope="risk_head.matched_control")

    calibration = _mapping(config.get("calibration"), scope="calibration")
    _exact(calibration, "method", "affine_platt", scope="calibration")
    _exact(calibration, "source", "inner_validation_only", scope="calibration")
    _exact(calibration, "shared_across_primary_products", True, scope="calibration")
    _exact_float(calibration, "soft_alert_quantile", 0.8, scope="calibration")
    _exact_float(calibration, "hard_alert_quantile", 0.95, scope="calibration")

    abstention = _mapping(config.get("abstention"), scope="abstention")
    _exact_float(abstention, "maximum_context_clip_fraction", 0.05, scope="abstention")
    _exact_int(abstention, "minimum_valid_paths", 60, scope="abstention")
    _exact(abstention, "unsupported_products", True, scope="abstention")
    _exact_float(abstention, "formal_primary_coverage_minimum", 0.95, scope="abstention")

    overlay = _mapping(config.get("overlay"), scope="overlay")
    _exact(overlay, "policy", "position_reduction_only", scope="overlay")
    _exact_float(overlay, "soft_alert_multiplier_end", 0.5, scope="overlay")
    _exact_float(overlay, "hard_alert_multiplier", 0.25, scope="overlay")
    _exact_int(overlay, "execution_lag_bars", 1, scope="overlay")
    _exact(overlay, "overlapping_forecasts", "latest_only", scope="overlay")
    _exact(overlay, "require_external_base_positions_for_formal_gate", True, scope="overlay")

    evaluation = _mapping(config.get("evaluation"), scope="evaluation")
    _exact_float(evaluation, "alert_budget", 0.2, scope="evaluation")
    _exact_int(evaluation, "bootstrap_iterations", 2000, scope="evaluation")
    if tuple(int(value) for value in evaluation.get("bootstrap_block_days", ())) != (5, 10):
        raise ValueError("V7 requires five- and ten-day paired block bootstraps")
    _exact_int(evaluation, "bootstrap_random_seed", 20260812, scope="evaluation")
    _exact_float(evaluation, "cvar_alpha", 0.05, scope="evaluation")
    _exact_float(evaluation, "minimum_return_retention", 0.85, scope="evaluation")

    runtime = _mapping(config.get("runtime"), scope="runtime")
    if str(runtime.get("device")) not in {"cuda", "cpu"}:
        raise ValueError("V7 runtime.device must be cuda or cpu")


__all__ = ["REPO_ROOT", "load_v7_config", "validate_v7_config"]
