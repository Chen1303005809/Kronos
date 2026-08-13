from __future__ import annotations

from copy import deepcopy
import json
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
from csj.v7.experiment import V7Experiment
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


def test_v7_baseline_selection_reports_missing_core_coverage_without_crashing() -> None:
    """A path-quality abstain must become a P1 diagnostic, not a vague exception.

    The frozen P0 validation universe has all 21 products.  This deliberately
    models the post-path-quality subset in which every i/jm/rb case abstained,
    which is the exact condition that formerly surfaced as ``No V7 baseline``.
    """

    config = load_v7_config(REPO_ROOT / "csj/configs/risk_control_v7.yaml")
    validation = _baseline_records("inner_validation", rows=4)
    validation["product"] = "a"
    validation["p_long"] = 0.25
    validation["p_short"] = 0.30
    candidates = {
        name: validation.copy()
        for name in config["baselines"]["selection_order"]
    }

    selected, evidence = choose_baseline(
        candidates,
        selection_order=config["baselines"]["selection_order"],
        gate_products=config["data"]["gate_products"],
        all_products=config["data"]["products"],
        allow_unavailable=True,
    )

    assert selected is None
    assert evidence["selection_available"] is False
    assert evidence["selection_unavailable_reason"] == "no_finite_validation_selection_brier"
    assert evidence["validation_scores"]["fit_global_event_rate"][
        "missing_core_product_side_cells"
    ] == [
        {"product": "i", "side": "long"},
        {"product": "i", "side": "short"},
        {"product": "jm", "side": "long"},
        {"product": "jm", "side": "short"},
        {"product": "rb", "side": "long"},
        {"product": "rb", "side": "short"},
    ]


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


def test_v7_p1_failed_selection_still_writes_truthful_status_figures(tmp_path: Path) -> None:
    path_records = pd.DataFrame(
        {
            "case_key": ["a"],
            "product": ["i"],
            "pred_len": [17],
            "valid_path_count": [0],
            "sample_count": [64],
            "eligible_for_risk": [False],
        }
    )
    fold_paths = pd.DataFrame(
        {
            "case_key": ["a"],
            "fold_id": ["fold_00"],
            "split": ["inner_validation"],
            "p_long": [np.nan],
            "p_short": [np.nan],
            "eligible_for_risk": [False],
        }
    )
    validation = pd.DataFrame(
        columns=(
            "fold_id",
            "product",
            "p_long",
            "p_short",
            "long_tail_event",
            "short_tail_event",
        )
    )

    artifacts = render_p1_plots(
        path_records=path_records,
        fold_path_records=fold_paths,
        validation_records=validation,
        selected_baselines={"fold_00": None},
        output_dir=tmp_path,
        metadata={"smoke": False},
        selection_available=False,
        selection_note="fold_00: no_finite_validation_selection_brier",
    )

    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in artifacts.as_dict().values())


def test_v7_p1_unselectable_validation_writes_a_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original CUDA symptom must finish as an auditable P1 gate failure."""

    config = load_v7_config(REPO_ROOT / "csj/configs/risk_control_v7.yaml")
    base = _baseline_records("fit", rows=4)
    validation = _baseline_records("inner_validation", rows=4)
    # This is the critical post-path-quality condition: no eligible core
    # product remains on inner validation, though the P0 universe itself is
    # sound and the simple historical baselines can still be calculated.
    base["product"] = "a"
    validation["product"] = "a"
    fold_records = pd.concat([base, validation], ignore_index=True)
    fold_features = fold_records[["case_key", "fold_id", "split"]].copy()
    fold_features["p_long"] = [0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5]
    fold_features["p_short"] = [0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2]
    fold_features["eligible_for_risk"] = True
    fold_features["valid_path_count"] = 64
    fold_features["sample_count"] = 64
    fold_features["invalid_path_rate"] = 0.0
    case_summaries = pd.DataFrame(
        {
            "case_key": fold_records["case_key"],
            "product": "a",
            "pred_len": 17,
            "valid_path_count": 64,
            "sample_count": 64,
            "eligible_for_risk": True,
        }
    ).drop_duplicates("case_key")
    features = {
        "case_summaries": case_summaries.to_dict("records"),
        "fold_features": fold_features.to_dict("records"),
        "statistics": {
            "unique_cases": len(case_summaries),
            "fold_records": len(fold_features),
            "global_valid_path_fraction": 1.0,
            "abstain_path_quality_cases": 0,
            "eligible_cases": len(case_summaries),
        },
    }
    manifest_path = tmp_path / "p1_path_bank" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    summary_path = tmp_path / "p1_path_bank" / "path_summaries.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    failures_path = tmp_path / "p1_path_bank" / "failures.json"
    failures_path.write_text(json.dumps({"failures": []}), encoding="utf-8")

    experiment = object.__new__(V7Experiment)
    experiment.config = config
    experiment.run_id = "selection-unavailable"
    experiment.run_dir = tmp_path / "run"
    experiment.results_dir = tmp_path / "results"
    experiment.fold_records = fold_records
    experiment.unique_cases = ()
    experiment._metadata = lambda phase, smoke: {  # type: ignore[method-assign]
        "phase": phase,
        "smoke": smoke,
    }
    experiment._load_path_manifest = lambda *, smoke: (  # type: ignore[method-assign]
        {"determinism_checks": [{"passed": True}]}, tmp_path
    )
    experiment._manifest_path = lambda *, smoke: manifest_path  # type: ignore[method-assign]
    experiment._derive_path_features = lambda **_: features  # type: ignore[method-assign]
    experiment._write_path_summaries = lambda **_: summary_path  # type: ignore[method-assign]
    experiment._failures_path = lambda *, smoke: failures_path  # type: ignore[method-assign]
    monkeypatch.setattr(
        "csj.v7.experiment.attach_context_features",
        lambda records, *, cases: records.copy(),
    )

    gate_path = experiment.p1_baselines()

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["allows_next_phase"] is False
    assert "validation_baseline_selection_available_by_fold" in gate["failed_condition_ids"]
    selection = json.loads(
        (tmp_path / "run" / "p1_baselines" / "selection.json").read_text(encoding="utf-8")
    )
    fold = selection["selection_by_fold"]["fold_00"]
    assert fold["selected_baseline"] is None
    assert fold["selection_unavailable_reason"] == "no_finite_validation_selection_brier"
    assert fold["validation_scores"]["fit_global_event_rate"][
        "missing_core_product_side_cells"
    ]
