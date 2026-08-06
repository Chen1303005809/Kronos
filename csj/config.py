from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")

    data_paths = config.get("data", {}).get("paths", [])
    if not data_paths:
        raise ValueError("Config must provide at least one data path")
    config["data"]["paths"] = [str(_resolve_repo_path(value)) for value in data_paths]

    output_root = config.get("output", {}).get("root", "csj/runs/futures_hourly")
    config.setdefault("output", {})["root"] = str(_resolve_repo_path(output_root))
    if "results_root" in config["output"]:
        config["output"]["results_root"] = str(
            _resolve_repo_path(config["output"]["results_root"])
        )

    cache_root = config.get("model", {}).get("cache_dir", "csj/artifacts/hf_cache")
    config.setdefault("model", {})["cache_dir"] = str(_resolve_repo_path(cache_root))
    validate_config(config)
    return config


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    split_sum = sum(
        float(data[key]) for key in ("train_ratio", "val_ratio", "test_ratio")
    )
    if abs(split_sum - 1.0) > 1e-9:
        raise ValueError("Data split ratios must sum to one")
    if int(data["lookback"]) > int(config["model"]["max_context"]):
        raise ValueError("lookback cannot exceed model max_context")
    if sorted(int(value) for value in data["valid_bar_counts"]) != [5, 7]:
        raise ValueError("This experiment expects 5-bar and 7-bar trading days")

    training = config["training"]
    learning_rates = [float(value) for value in training["learning_rates"]]
    seeds = [int(value) for value in training["seeds"]]
    if len(seeds) != 3:
        raise ValueError("The agreed stability protocol requires three seeds")
    if int(config["evaluation"]["sample_count"]) != 10:
        raise ValueError("The agreed evaluation protocol requires ten sampled paths")

    if int(data.get("target_trading_days", 1)) == 3:
        if int(data["dense_horizon"]) != 21:
            raise ValueError("V2 dense token training must use a 21-bar horizon")
        if sorted(int(value) for value in data["possible_pred_lengths"]) != [
            15,
            17,
            19,
            21,
        ]:
            raise ValueError("V2 three-day targets must have lengths 15/17/19/21")
        if len(learning_rates) != 2:
            raise ValueError("V2 Phase 2 locks two learning-rate candidates")
        if float(training["direction_smoke_lambda"]) != 0.2:
            raise ValueError("V2 Phase 3 direction loss must use lambda_dir=0.2")
        if int(training.get("direction_batch_size", 0)) < 1:
            raise ValueError("V2 Phase 3 direction_batch_size must be positive")
        if int(training.get("dense_batches_per_direction_batch", 0)) < 1:
            raise ValueError(
                "V2 Phase 3 dense_batches_per_direction_batch must be positive"
            )
        walk_forward = config["walk_forward"]
        if int(walk_forward["minimum_train_days"]) != 360:
            raise ValueError("V2 walk-forward minimum_train_days must be 360")
        if int(walk_forward["evaluation_days"]) != 60:
            raise ValueError("V2 walk-forward evaluation_days must be 60")
        if int(walk_forward["step_days"]) != 60:
            raise ValueError("V2 walk-forward step_days must be 60")
        if int(config["evaluation"]["bootstrap_block_days"][0]) != 5:
            raise ValueError("V2 primary bootstrap block must be five trading days")
        if config["evaluation"]["path_point_estimate"] != "median":
            raise ValueError("V2 path metrics must use the sampled median path")
        allowed_devices = {"auto", "cpu", "cuda", "mps"}
        training_device = str(training.get("device", "auto"))
        evaluation_device = str(config["evaluation"].get("device", "auto"))
        if training_device not in allowed_devices:
            raise ValueError(f"Unsupported training device: {training_device}")
        if evaluation_device not in allowed_devices:
            raise ValueError(f"Unsupported evaluation device: {evaluation_device}")
        if training_device != evaluation_device:
            raise ValueError(
                "V2 training and evaluation must request the same device"
            )
        if float(config["evaluation"]["turning_point_return_threshold"]) != 0.0005:
            raise ValueError("V2 turning-point threshold must be locked before evaluation")
        if "results_root" not in config["output"]:
            raise ValueError("V2 config must provide an independent results_root")
    elif len(learning_rates) != 3:
        raise ValueError("The agreed V1 search protocol requires three learning rates")
