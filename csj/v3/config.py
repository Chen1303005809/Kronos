"""Configuration loading and invariants for the CUDA-first V3 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_v3_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V3 configuration must be a YAML mapping")
    config = value
    data = config.setdefault("data", {})
    output = config.setdefault("output", {})
    model = config.setdefault("model", {})
    if "snapshot_root" not in data:
        raise ValueError("V3 config must set data.snapshot_root")
    data["snapshot_root"] = str(_resolve(data["snapshot_root"]))
    output["root"] = str(_resolve(output.get("root", "csj/runs/active_contract_panel_v3")))
    output["results_root"] = str(
        _resolve(output.get("results_root", "csj/results/active_contract_panel_v3"))
    )
    model["cache_dir"] = str(_resolve(model.get("cache_dir", "csj/artifacts/hf_cache")))
    validate_v3_config(config)
    return config


def validate_v3_config(config: dict[str, Any]) -> None:
    if int(config.get("experiment", {}).get("version", 0)) != 3:
        raise ValueError("V3 config must declare experiment.version: 3")
    data = config["data"]
    lookbacks = sorted(int(value) for value in data.get("lookbacks", []))
    if lookbacks != [256, 512]:
        raise ValueError("V3 must audit exactly lookbacks 256 and 512")
    if not data.get("products"):
        raise ValueError("V3 config must declare at least one product")
    if float(data.get("clip", 0)) <= 0 or float(data.get("normalization_epsilon", 0)) <= 0:
        raise ValueError("V3 clip and normalization epsilon must be positive")
    if bool(data.get("allow_partial_panel_training", False)):
        # This is deliberately allowed only as an explicit research switch and
        # is recorded by the experiment; it must never be the default.
        pass

    model = config["model"]
    if int(model.get("max_context", 0)) != 512:
        raise ValueError("V3 is pinned to the Kronos 512-token maximum context")
    for key in ("tokenizer_id", "tokenizer_revision", "predictor_id", "predictor_revision"):
        if not model.get(key):
            raise ValueError(f"V3 model configuration is missing {key}")

    walk = config["walk_forward"]
    for key in ("minimum_fit_days", "inner_validation_days", "evaluation_days", "step_days"):
        if int(walk.get(key, 0)) < 1:
            raise ValueError(f"V3 walk_forward.{key} must be positive")
    if int(walk.get("purge_days", 0)) < 3:
        raise ValueError("V3 requires at least a three-trading-day purge")

    device = str(config.get("runtime", {}).get("device", "cuda"))
    if device not in {"cuda", "cpu"}:
        raise ValueError("V3 runtime.device must be cuda or cpu (cpu is smoke-test only)")
    evaluation = config["evaluation"]
    if int(evaluation.get("sample_count", 0)) < 1:
        raise ValueError("V3 evaluation.sample_count must be positive")
    blocks = [int(value) for value in evaluation.get("bootstrap_block_days", [])]
    if blocks != [5, 10]:
        raise ValueError("V3 P1 requires 5-day and 10-day bootstrap blocks")
    p1_training = config.get("p1", {}).get("training", {})
    sampling_strategy = str(p1_training.get("sampling_strategy", "prediction_day_uniform"))
    if sampling_strategy not in {"case_uniform", "prediction_day_uniform"}:
        raise ValueError(
            "V3 p1.training.sampling_strategy must be case_uniform or prediction_day_uniform"
        )
