from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from csj.v3.pair_probe import prediction_day_product_uniform_weights
from csj.v4.experiment import (
    V4Experiment,
    V4ExperimentError,
    _gate_for_json,
    _granularity_gate,
    _select_granularity,
)


def _probe_records(model: str, *, seed: int, correct: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    products = ("i", "jm", "rb")
    directions = ("earlier", "later")
    for day_index, day in enumerate(pd.bdate_range("2026-01-05", periods=10)):
        fold_id = f"fold_{day_index // 2:02d}"
        for product_index, product in enumerate(products):
            for direction_index, neighbor_direction in enumerate(directions):
                actual_label = int((day_index + product_index + direction_index) % 2 == 0)
                predicted_label = actual_label if correct else 1 - actual_label
                rows.append(
                    {
                        "case_key": f"{product}|{neighbor_direction}|{day.date()}",
                        "fold_id": fold_id,
                        "product": product,
                        "target_end_day": day,
                        "target_contract_id": f"{product}26{day_index + 1:02d}",
                        "actual_label": actual_label,
                        "actual_direction": 1 if actual_label else -1,
                        "valid_direction": True,
                        "probability_up": 0.9 if predicted_label else 0.1,
                        "predicted_label": predicted_label,
                        "predicted_direction": 1 if predicted_label else -1,
                        "neighbor_direction": neighbor_direction,
                        "model": model,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def _seed_arms(seed: int) -> dict[str, pd.DataFrame]:
    return {
        "shared_target_only": _probe_records("shared_target_only", seed=seed, correct=False),
        "shared_pair": _probe_records("shared_pair", seed=seed, correct=True),
        "per_product_target_only": _probe_records(
            "per_product_target_only", seed=seed, correct=False
        ),
        "per_product_pair": _probe_records("per_product_pair", seed=seed, correct=True),
    }


def test_shared_sampler_gives_each_prediction_day_product_equal_mass() -> None:
    cases = (
        SimpleNamespace(origin_trading_day=pd.Timestamp("2026-01-05"), product="i"),
        SimpleNamespace(origin_trading_day=pd.Timestamp("2026-01-05"), product="i"),
        SimpleNamespace(origin_trading_day=pd.Timestamp("2026-01-05"), product="rb"),
        SimpleNamespace(origin_trading_day=pd.Timestamp("2026-01-06"), product="i"),
        SimpleNamespace(origin_trading_day=pd.Timestamp("2026-01-06"), product="rb"),
        SimpleNamespace(origin_trading_day=pd.Timestamp("2026-01-06"), product="rb"),
    )

    weights = prediction_day_product_uniform_weights(cases)
    groups = pd.DataFrame(
        {
            "day": [case.origin_trading_day for case in cases],
            "product": [case.product for case in cases],
            "weight": weights.numpy(),
        }
    )

    assert set(groups.groupby(["day", "product"])["weight"].sum()) == {1.0}


def test_v4_granularity_gate_uses_three_seeds_all_folds_and_seed_averaged_bootstrap() -> None:
    records_by_seed = {seed: _seed_arms(seed) for seed in (42, 43, 44)}
    gate = _granularity_gate(
        records_by_seed,
        pair_arm="shared_pair",
        target_arm="shared_target_only",
        primary_products=("i", "jm", "rb"),
        bootstrap_blocks=(5, 10),
        bootstrap_iterations=20,
        bootstrap_seed=7,
    )

    assert gate["passes_granularity_gate"]
    assert gate["median_seed_balanced_accuracy_improvement"] == 1.0
    assert gate["improved_fold_count"] == 5
    assert gate["bootstrap"]["block_5"]["probability_improvement_positive"] == 1.0
    assert "pair_records" not in _gate_for_json(gate)["ensemble"]

    selected = _select_granularity(
        gate,
        gate,
        bootstrap_blocks=(5, 10),
        bootstrap_iterations=20,
        bootstrap_seed=10,
        primary_products=("i", "jm", "rb"),
    )
    assert selected["selected_granularity"] == "shared"
    assert selected["allows_p2"]


def test_p1_gate_cannot_be_reused_by_a_different_run(tmp_path) -> None:
    experiment = object.__new__(V4Experiment)
    experiment.results_dir = tmp_path
    experiment.run_id = "active_run"
    experiment.cohort = SimpleNamespace(cohort_fingerprint="cohort-123")
    (tmp_path / "p1_granularity_gate.json").write_text(
        json.dumps(
            {
                "strategy_version": 4,
                "phase": "p1_ablation",
                "run_id": "other_run",
                "result_scope": "retrospective_observed_cohort",
                "production_eligible": False,
                "cohort_fingerprint": "cohort-123",
                "selection": {"allows_p2": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(V4ExperimentError, match="does not belong to the active run/cohort"):
        experiment._require_p1_gate()


def test_single_fold_p1_smoke_never_writes_a_phase_gate(tmp_path) -> None:
    fit_day = pd.Timestamp("2026-01-05")
    validation_day = pd.Timestamp("2026-01-06")
    evaluation_day = pd.Timestamp("2026-01-07")
    products = ("i", "jm", "rb", "j")
    cases = tuple(
        SimpleNamespace(
            case_key=f"{product}|{day.date()}",
            product=product,
            target_end_day=day,
        )
        for product in products
        for day in (fit_day, validation_day, evaluation_day)
    )
    fold = SimpleNamespace(
        fold_id="fold_00",
        fit_start_day=fit_day,
        fit_end_day=fit_day,
        inner_validation_start_day=validation_day,
        inner_validation_end_day=validation_day,
        evaluation_start_day=evaluation_day,
        evaluation_end_day=evaluation_day,
    )
    experiment = object.__new__(V4Experiment)
    experiment.config = {
        "data": {
            "products": list(products),
            "primary_selection_products": ["i", "jm", "rb"],
            "transfer_products": ["j"],
        },
        "model": {
            "tokenizer_id": "tokenizer",
            "tokenizer_revision": "revision",
            "predictor_id": "predictor",
            "predictor_revision": "revision",
        },
        "p1": {
            "seeds": [42, 43, 44],
            "training": {
                "shared_sampling_strategy": "prediction_day_product_uniform",
                "per_product_sampling_strategy": "prediction_day_uniform",
            },
        },
        "evaluation": {},
    }
    experiment.run_id = "p1_smoke"
    experiment.run_dir = tmp_path / "runs" / "p1_smoke"
    experiment.results_dir = tmp_path / "results"
    experiment.cohort = SimpleNamespace(cohort_fingerprint="cohort-123")
    experiment.bundle = SimpleNamespace(pair_cases=cases)
    experiment.folds = (fold,)

    def fake_matched_arms(
        *,
        evaluation_cases,
        auxiliary_evaluation_cases,
        fold_id,
        seed,
        target_model_name,
        pair_model_name,
        **_kwargs,
    ):
        def records(model: str, probability_up: float) -> pd.DataFrame:
            rows = []
            for case in evaluation_cases:
                rows.append(
                    {
                        "case_key": case.case_key,
                        "fold_id": fold_id,
                        "product": case.product,
                        "target_end_day": case.target_end_day,
                        "target_contract_id": f"{case.product}2601",
                        "actual_label": 1,
                        "actual_direction": 1,
                        "valid_direction": True,
                        "probability_up": probability_up,
                        "predicted_label": int(probability_up >= 0.5),
                        "predicted_direction": 1 if probability_up >= 0.5 else -1,
                        "neighbor_direction": "later",
                        "model": model,
                        "seed": seed,
                    }
                )
            return pd.DataFrame(rows)

        target = records(target_model_name, 0.2)
        pair = records(pair_model_name, 0.8)
        if not auxiliary_evaluation_cases:
            return target, pair, None, None
        original_cases = evaluation_cases
        try:
            evaluation_cases = auxiliary_evaluation_cases
            transfer_target = records(target_model_name, 0.2)
            transfer_pair = records(pair_model_name, 0.8)
        finally:
            evaluation_cases = original_cases
        return target, pair, transfer_target, transfer_pair

    experiment._load_tokenizer = lambda: object()
    experiment._load_predictor = lambda: object()
    experiment._release = lambda *_models: None
    experiment._run_matched_probe_arms = fake_matched_arms

    result = experiment.run_p1_ablation(fold_id="fold_00")

    assert result == experiment.results_dir / "p1_ablation_smoke_metrics.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["p1_gate"]["available"] is False
    assert not (experiment.results_dir / "p1_granularity_gate.json").exists()
    assert (
        experiment.run_dir
        / "p1_ablation_smoke"
        / "evaluation"
        / "fold_00"
        / "prediction_vs_actual.png"
    ).is_file()
