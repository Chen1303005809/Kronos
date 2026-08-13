from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csj.v7.baselines import (
    attach_context_features,
    choose_baseline,
    fit_and_predict_baselines,
)
from csj.v7.config import load_v7_config, validate_v7_config
from csj.v7.path_bank import (
    cache_key_for_case,
    sampling_seed,
    validate_raw_paths,
    write_shard_atomic,
)
from csj.v7.plotting import render_p1_plots


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v7_config_freezes_p1_to_p5_protocol() -> None:
    config = load_v7_config(REPO_ROOT / "csj/configs/risk_control_v7.yaml")
    assert config["path_bank"]["sample_count"] == 64
    assert config["model"]["freeze_predictor"] is True
    assert config["risk_head"]["seeds"] == [42, 43, 44]

    cases = (
        ("path_bank", "sample_count", 32, "sample_count"),
        ("path_bank", "top_p", 0.8, "top_p"),
        ("model", "predictor_revision", "unfrozen", "predictor_revision"),
        ("risk_head", "dropout", 0.1, "dropout"),
        ("runtime", "device", "mps", "device"),
    )
    for section, key, value, message in cases:
        mutated = deepcopy(config)
        mutated[section][key] = value
        with pytest.raises(ValueError, match=message):
            validate_v7_config(mutated)


def test_v7_case_cache_key_and_seed_are_stable_and_sensitive() -> None:
    kwargs = {
        "strategy_version": 7,
        "data_fingerprint": "fingerprint",
        "case_key": "rb2609|2025-10-15T14:00:00",
        "tokenizer_revision": "tokenizer",
        "predictor_revision": "predictor",
        "sample_count": 64,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.9,
    }
    first = cache_key_for_case(**kwargs)
    second = cache_key_for_case(**kwargs)
    altered = cache_key_for_case(**{**kwargs, "top_p": 0.8})
    assert first == second
    assert first != altered
    assert sampling_seed(first) == sampling_seed(first)
    assert 0 <= sampling_seed(first) < 2**63


def test_v7_path_validity_does_not_silently_repair_invalid_ohlcva() -> None:
    paths = np.asarray(
        [
            [[10, 12, 9, 11, 1, 2], [11, 13, 10, 12, 1, 2]],
            [[10, 9, 11, 12, 1, 2], [11, 13, 10, 12, -1, 2]],
        ],
        dtype=np.float32,
    )
    validity = validate_raw_paths(paths)
    assert validity.valid_mask.tolist() == [True, False]
    assert validity.invalid_reason_counts["high_below_open_or_close"] == 1
    assert validity.invalid_reason_counts["low_above_open_or_close"] == 1
    assert validity.invalid_reason_counts["negative_volume_or_amount"] == 1
    assert paths[1, 0, 1] == 9  # validation must not mutate/reproject a raw path


def test_v7_atomic_shard_write_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "shard.npz"
    arrays = {
        "raw_paths": np.ones((1, 4, 17, 6), dtype=np.float32),
        "hidden": np.ones((1, 8), dtype=np.float32),
        "context_mean": np.ones((1, 6), dtype=np.float32),
        "context_std": np.ones((1, 6), dtype=np.float32),
        "target_timestamps": np.ones((1, 17), dtype=np.int64),
    }
    metadata = {"schema_version": 1, "case_keys": ["case"]}
    write_shard_atomic(destination, arrays=arrays, metadata=metadata)
    with pytest.raises(Exception, match="overwrite"):
        write_shard_atomic(destination, arrays=arrays, metadata=metadata)


def _baseline_records(split: str, *, rows: int = 24) -> pd.DataFrame:
    products = np.resize(np.asarray(["i", "jm", "rb", "a"], dtype=object), rows)
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "case_key": [f"{split}-{value}" for value in range(rows)],
            "fold_id": "fold_00",
            "split": split,
            "product": products,
            "long_tail_event": (index % 3 == 0),
            "short_tail_event": (index % 4 == 0),
            "ewma_volatility": 0.01 + index / 10_000,
            "atr20_over_price": 0.02 + index / 10_000,
            "range20_over_price": 0.03 + index / 10_000,
            "absolute_return_day_minus_3": 0.01 + index / 10_000,
            "absolute_return_day_minus_2": 0.011 + index / 10_000,
            "absolute_return_day_minus_1": 0.012 + index / 10_000,
            "volume_zscore": (index - rows / 2) / rows,
            "amount_zscore": (index - rows / 3) / rows,
            "context_clip_fraction": np.full(rows, 0.001),
        }
    )


def test_v7_baseline_selection_uses_validation_not_evaluation() -> None:
    config = load_v7_config(REPO_ROOT / "csj/configs/risk_control_v7.yaml")
    fit = _baseline_records("fit")
    validation = _baseline_records("inner_validation")
    path_features = pd.concat(
        [
            frame[["case_key", "fold_id", "split"]].assign(p_long=0.25, p_short=0.3)
            for frame in (fit, validation)
        ],
        ignore_index=True,
    )
    validation_by_name, _ = fit_and_predict_baselines(
        fit_records=fit,
        destination_records=validation,
        config=config,
        path_features=path_features,
    )
    selected, evidence = choose_baseline(
        validation_by_name,
        selection_order=config["baselines"]["selection_order"],
        gate_products=config["data"]["gate_products"],
        all_products=config["data"]["products"],
    )
    assert selected in config["baselines"]["selection_order"]
    assert evidence["validation_scores"][selected]["selection_brier"] is not None


def test_v7_p1_plots_use_case_and_fold_derived_path_records(tmp_path: Path) -> None:
    path_records = pd.DataFrame(
        {
            "case_key": ["a", "b"],
            "product": ["i", "rb"],
            "pred_len": [17, 21],
            "valid_path_count": [64, 63],
            "sample_count": [64, 64],
            "eligible_for_risk": [True, True],
        }
    )
    fold_paths = pd.DataFrame(
        {
            "case_key": ["a", "b"],
            "fold_id": ["fold_00", "fold_00"],
            "split": ["inner_validation", "inner_validation"],
            "p_long": [0.2, 0.8],
            "p_short": [0.7, 0.3],
            "eligible_for_risk": [True, True],
        }
    )
    validation = pd.DataFrame(
        {
            "fold_id": ["fold_00", "fold_00", "fold_01", "fold_01"],
            "product": ["i", "rb", "i", "rb"],
            "p_long": [0.2, 0.8, 0.3, 0.7],
            "p_short": [0.7, 0.3, 0.6, 0.4],
            "long_tail_event": [False, True, False, True],
            "short_tail_event": [True, False, True, False],
        }
    )
    artifacts = render_p1_plots(
        path_records=path_records,
        fold_path_records=fold_paths,
        validation_records=validation,
        selected_baselines={"fold_00": "zero_shot_path_risk"},
        output_dir=tmp_path,
        metadata={"smoke": False},
    )
    assert all(Path(path).is_file() for path in artifacts.as_dict().values())
