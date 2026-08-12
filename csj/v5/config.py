"""Configuration loading and frozen invariants for the V5 workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCOPE = "retrospective_observed_contracts"
PRODUCTION_ELIGIBLE = False


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_v5_config(path: str | Path) -> dict[str, Any]:
    """Load the V5 YAML and resolve repository-relative storage locations."""

    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V5 configuration must be a YAML mapping")
    config = value
    data = config.setdefault("data", {})
    model = config.setdefault("model", {})
    output = config.setdefault("output", {})
    if "snapshot_root" not in data:
        raise ValueError("V5 config must set data.snapshot_root")
    data["snapshot_root"] = str(_resolve(data["snapshot_root"]))
    model["cache_dir"] = str(_resolve(model.get("cache_dir", "csj/artifacts/hf_cache")))
    output["root"] = str(_resolve(output.get("root", "csj/runs/target_only_path_v5")))
    output["results_root"] = str(
        _resolve(output.get("results_root", "csj/results/target_only_path_v5"))
    )
    validate_v5_config(config)
    return config


def _require_exact_int(mapping: dict[str, Any], key: str, expected: int, *, scope: str) -> None:
    if int(mapping.get(key, -1)) != expected:
        raise ValueError(f"V5 fixes {scope}.{key} at {expected}")


def validate_v5_config(config: dict[str, Any]) -> None:
    """Reject accidental protocol changes before expensive CUDA work starts."""

    experiment = config.get("experiment", {})
    if int(experiment.get("version", 0)) != 5:
        raise ValueError("V5 config must declare experiment.version: 5")
    if experiment.get("result_scope") != RESULT_SCOPE:
        raise ValueError(f"V5 result_scope must be {RESULT_SCOPE!r}")
    if bool(experiment.get("production_eligible", True)) is not PRODUCTION_ELIGIBLE:
        raise ValueError("V5 production_eligible must remain false")

    data = config.get("data", {})
    if int(data.get("lookback", 0)) != 256:
        raise ValueError("V5 fixes data.lookback at 256")
    if sorted(int(value) for value in data.get("valid_bar_counts", [])) != [5, 7]:
        raise ValueError("V5 accepts only five- or seven-bar complete trading days")
    if float(data.get("clip", 0.0)) <= 0.0 or float(data.get("normalization_epsilon", 0.0)) <= 0.0:
        raise ValueError("V5 clip and normalization_epsilon must be positive")
    products = tuple(str(value) for value in data.get("products", ()))
    primary = tuple(str(value) for value in data.get("primary_selection_products", ()))
    transfer = tuple(str(value) for value in data.get("transfer_products", ()))
    if not products or not primary:
        raise ValueError("V5 requires products and primary_selection_products")
    if not set(primary).issubset(products) or not set(transfer).issubset(products):
        raise ValueError("V5 primary and transfer products must be declared in data.products")
    if set(primary).intersection(transfer):
        raise ValueError("V5 primary and transfer product sets must not overlap")
    if tuple(primary) != ("i", "jm", "rb") or tuple(transfer) != ("j",):
        raise ValueError("V5 fixes primary products to i/jm/rb and transfer product to j")

    model = config.get("model", {})
    if int(model.get("max_context", 0)) != 512:
        raise ValueError("V5 is pinned to Kronos's 512-token maximum context")
    for key in ("tokenizer_id", "tokenizer_revision", "predictor_id", "predictor_revision"):
        if not model.get(key):
            raise ValueError(f"V5 model configuration is missing {key}")

    walk = config.get("walk_forward", {})
    for key, expected in {
        "minimum_fit_days": 60,
        "inner_validation_days": 20,
        "evaluation_days": 20,
        "step_days": 20,
        "purge_days": 3,
    }.items():
        _require_exact_int(walk, key, expected, scope="walk_forward")

    if str(config.get("runtime", {}).get("device", "cuda")) not in {"cuda", "cpu"}:
        raise ValueError("V5 runtime.device must be cuda or cpu")

    p0 = config.get("p0", {})
    _require_exact_int(p0, "sample_count", 32, scope="p0")
    if (
        float(p0.get("temperature", -1.0)) != 1.0
        or int(p0.get("top_k", -1)) != 0
        or float(p0.get("top_p", -1.0)) != 0.9
        or int(p0.get("random_seed", -1)) != 20260812
    ):
        raise ValueError("V5 P0 sampling is frozen at K=32, T=1.0, top_k=0, top_p=0.9, seed=20260812")
    if int(p0.get("inference_batch_size", 0)) < 1:
        raise ValueError("V5 p0.inference_batch_size must be positive")
    selection_order = tuple(str(value) for value in p0.get("selection_order", ()))
    expected_order = (
        "zero_shot_mean_path",
        "zero_shot_sample_vote",
        "fit_product_majority",
        "context_3day_momentum",
    )
    if selection_order != expected_order:
        raise ValueError("V5 P0 baseline selection order must be pre-registered")

    p1 = config.get("p1", {})
    if tuple(int(seed) for seed in p1.get("seeds", ())) != (42, 43, 44):
        raise ValueError("V5 P1 seeds are fixed at [42, 43, 44]")
    _require_exact_int(p1, "fusion_hidden_dim", 256, scope="p1")
    if float(p1.get("dropout", -1.0)) != 0.0:
        raise ValueError("V5 fixes P1 dropout at 0")
    training = p1.get("training", {})
    expected_training = {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "weight_decay": 0.01,
        "gradient_clip": 3.0,
    }
    for key, expected in expected_training.items():
        actual = training.get(key)
        if isinstance(expected, int):
            if int(actual if actual is not None else -1) != expected:
                raise ValueError(f"V5 fixes p1.training.{key} at {expected}")
        elif float(actual if actual is not None else float("nan")) != expected:
            raise ValueError(f"V5 fixes p1.training.{key} at {expected}")
    if str(training.get("sampling_strategy")) != "prediction_day_product_uniform":
        raise ValueError("V5 P1 must use prediction_day_product_uniform sampling")
    if int(training.get("num_workers", 0)) < 0:
        raise ValueError("V5 p1.training.num_workers cannot be negative")

    evaluation = config.get("evaluation", {})
    if tuple(int(value) for value in evaluation.get("bootstrap_block_days", ())) != (5, 10):
        raise ValueError("V5 requires five- and ten-day paired block bootstraps")
    if int(evaluation.get("bootstrap_iterations", 0)) < 1:
        raise ValueError("V5 bootstrap_iterations must be positive")


__all__ = [
    "PRODUCTION_ELIGIBLE",
    "RESULT_SCOPE",
    "load_v5_config",
    "validate_v5_config",
]
