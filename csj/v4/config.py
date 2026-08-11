"""Configuration loading and invariants for the V4 observed-cohort workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from csj.v4.cohort_data import PRODUCTION_ELIGIBLE, RESULT_SCOPE


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_v4_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V4 configuration must be a YAML mapping")
    config = value
    data = config.setdefault("data", {})
    model = config.setdefault("model", {})
    output = config.setdefault("output", {})
    if "snapshot_root" not in data:
        raise ValueError("V4 config must set data.snapshot_root")
    data["snapshot_root"] = str(_resolve(data["snapshot_root"]))
    output["root"] = str(_resolve(output.get("root", "csj/runs/observed_contract_cohort_v4")))
    output["results_root"] = str(
        _resolve(output.get("results_root", "csj/results/observed_contract_cohort_v4"))
    )
    model["cache_dir"] = str(_resolve(model.get("cache_dir", "csj/artifacts/hf_cache")))
    validate_v4_config(config)
    return config


def validate_v4_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    if int(experiment.get("version", 0)) != 4:
        raise ValueError("V4 config must declare experiment.version: 4")
    if experiment.get("result_scope") != RESULT_SCOPE:
        raise ValueError(f"V4 result_scope must be {RESULT_SCOPE!r}")
    if bool(experiment.get("production_eligible", True)) is not PRODUCTION_ELIGIBLE:
        raise ValueError("V4 production_eligible must remain false")
    data = config["data"]
    if sorted(int(value) for value in data.get("lookbacks", [])) != [256, 512]:
        raise ValueError("V4 must audit lookbacks [256, 512]")
    if int(data.get("lookback", 0)) != 256:
        raise ValueError("V4 fixes data.lookback at 256 after the coverage decision")
    if sorted(int(value) for value in data.get("valid_bar_counts", [])) != [5, 7]:
        raise ValueError("V4 expects only 5- or 7-bar complete trading days")
    if float(data.get("clip", 0)) <= 0 or float(data.get("normalization_epsilon", 0)) <= 0:
        raise ValueError("V4 clip and normalization_epsilon must be positive")
    products = [str(value) for value in data.get("products", [])]
    primary = [str(value) for value in data.get("primary_selection_products", [])]
    transfer = [str(value) for value in data.get("transfer_products", [])]
    if not products or not primary:
        raise ValueError("V4 requires products and primary_selection_products")
    if not set(primary).issubset(products) or not set(transfer).issubset(products):
        raise ValueError("V4 primary/transfer products must be declared in data.products")
    if set(primary).intersection(transfer):
        raise ValueError("V4 primary and transfer product sets must not overlap")
    model = config["model"]
    if int(model.get("max_context", 0)) != 512:
        raise ValueError("V4 is pinned to Kronos's 512-token maximum context")
    for key in ("tokenizer_id", "tokenizer_revision", "predictor_id", "predictor_revision"):
        if not model.get(key):
            raise ValueError(f"V4 model configuration is missing {key}")
    walk = config.get("walk_forward", {})
    for key in ("minimum_fit_days", "inner_validation_days", "evaluation_days", "step_days"):
        if int(walk.get(key, 0)) < 1:
            raise ValueError(f"V4 walk_forward.{key} must be positive")
    if int(walk.get("purge_days", 0)) < 3:
        raise ValueError("V4 requires a three-trading-day purge")
    if str(config.get("runtime", {}).get("device", "cuda")) not in {"cuda", "cpu"}:
        raise ValueError("V4 runtime.device must be cuda or cpu")
    evaluation = config.get("evaluation", {})
    if [int(value) for value in evaluation.get("bootstrap_block_days", [])] != [5, 10]:
        raise ValueError("V4 P1 requires 5- and 10-day paired block bootstraps")
    if int(evaluation.get("bootstrap_iterations", 0)) < 1:
        raise ValueError("V4 bootstrap_iterations must be positive")
    p1 = config.get("p1", {})
    if [int(seed) for seed in p1.get("seeds", [])] != [42, 43, 44]:
        raise ValueError("V4 P1 seeds are fixed at [42, 43, 44]")
    if int(p1.get("fusion_hidden_dim", 0)) < 1:
        raise ValueError("V4 p1.fusion_hidden_dim must be positive")
    training = p1.get("training", {})
    if str(training.get("shared_sampling_strategy")) != "prediction_day_product_uniform":
        raise ValueError("V4 shared P1 must use prediction_day_product_uniform sampling")
    if str(training.get("per_product_sampling_strategy")) != "prediction_day_uniform":
        raise ValueError("V4 per-product P1 must use prediction_day_uniform sampling")


__all__ = ["load_v4_config", "validate_v4_config"]
