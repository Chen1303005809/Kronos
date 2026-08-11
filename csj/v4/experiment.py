"""Phase-gated V4 observed-cohort experiment orchestration.

The implementation stops at the Phase 1 shared-vs-per-product probe ablation.
Phase 2 path conditioning is deliberately represented only by a guarded CLI
entry: it cannot run until an observed Phase 1 gate exists, and its model is not
implemented in this first V4 delivery.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from csj.evaluation_plotter import (
    DIRECTION_COMPARISON_CONTRACT_VERSION,
    DirectionComparisonError,
    EvaluationArtifacts,
    render_fold_ablation_direction_comparison,
    render_fold_direction_comparison,
    write_direction_stage_report,
)
from csj.v3.p0 import evaluate_target_paths
from csj.v3.pair_probe import (
    PairProbe,
    ProbeTrainingConfig,
    assert_same_case_keys,
    evaluate_probe,
    load_probe_head,
    paired_block_bootstrap,
    probe_metrics,
    train_probe,
)
from csj.v4.cohort_data import (
    PRODUCTION_ELIGIBLE,
    RESULT_SCOPE,
    ObservedCohortCase,
    ObservedCohortCaseBundle,
    V3WalkForwardFold,
    build_observed_cohort_audit,
    load_observed_contract_cohort,
)
from csj.v4.config import load_v4_config
from model import Kronos, KronosTokenizer


class V4ExperimentError(RuntimeError):
    """A V4 stage cannot meet its frozen research protocol."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise V4ExperimentError("V4 requested CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported V4 device: {requested}")


def _cases_in_period(
    cases: Sequence[ObservedCohortCase],
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    products: Sequence[str] | None = None,
) -> tuple[ObservedCohortCase, ...]:
    accepted_products = None if products is None else {str(product) for product in products}
    return tuple(
        case
        for case in cases
        if start_day <= case.target_end_day <= end_day
        and (accepted_products is None or case.product in accepted_products)
    )


def _assert_paired_records(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    label: str,
) -> None:
    required = {
        "case_key",
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "actual_label",
        "valid_direction",
        "probability_up",
    }
    for name, records in (("candidate", candidate), ("baseline", baseline)):
        missing = sorted(required.difference(records.columns))
        if missing:
            raise V4ExperimentError(f"{label}/{name} records miss columns: {missing!r}")
        if records["case_key"].duplicated().any():
            raise V4ExperimentError(f"{label}/{name} records contain duplicate case keys")
    candidate_indexed = candidate.set_index("case_key", verify_integrity=True)
    baseline_indexed = baseline.set_index("case_key", verify_integrity=True)
    if set(candidate_indexed.index) != set(baseline_indexed.index):
        raise V4ExperimentError(f"{label} arms do not have exactly matched case keys")
    baseline_indexed = baseline_indexed.loc[candidate_indexed.index]
    for column in (
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "actual_label",
        "valid_direction",
    ):
        if not candidate_indexed[column].equals(baseline_indexed[column]):
            raise V4ExperimentError(
                f"{label} arms disagree for the same case on {column}"
            )


def _direction_metrics(records: pd.DataFrame) -> dict[str, object]:
    valid = records.loc[records["actual_direction"] != 0]
    if valid.empty:
        return {
            "cases": int(len(records)),
            "valid_direction_cases": 0,
            "balanced_accuracy": None,
            "accuracy": None,
        }
    actual = valid["actual_direction"].to_numpy(dtype=np.int8)
    predicted = valid["predicted_direction"].to_numpy(dtype=np.int8)
    up = actual == 1
    down = actual == -1
    balanced_accuracy = (
        float(0.5 * (np.mean(predicted[up] == 1) + np.mean(predicted[down] == -1)))
        if up.any() and down.any()
        else None
    )
    return {
        "cases": int(len(records)),
        "valid_direction_cases": int(len(valid)),
        "balanced_accuracy": balanced_accuracy,
        "accuracy": float(np.mean(actual == predicted)),
    }


def _probe_records_for_direction_plot(
    records: pd.DataFrame,
    *,
    fold_id: str,
    model_name: str,
    seed: int | str,
) -> pd.DataFrame:
    """Standardize probe output while keeping zero-return cases visibly neutral."""

    required = {
        "case_key",
        "target_contract_id",
        "product",
        "target_end_day",
        "actual_label",
        "actual_direction",
        "valid_direction",
        "probability_up",
        "predicted_label",
        "predicted_direction",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise V4ExperimentError(f"Probe records miss standardized columns: {missing!r}")
    output = records.copy()
    invalid = ~output["valid_direction"].astype(bool)
    output.loc[invalid, "actual_direction"] = 0
    output.loc[invalid, "predicted_direction"] = 0
    output["fold_id"] = str(fold_id)
    output["model"] = str(model_name)
    output["seed"] = seed
    return output.sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _attach_case_provenance(
    records: pd.DataFrame,
    cases: Sequence[ObservedCohortCase],
) -> pd.DataFrame:
    """Carry V4's per-case cohort/selection audit fields into every result file."""

    if records["case_key"].duplicated().any():
        raise V4ExperimentError("Cannot attach provenance to duplicate case-key records")
    by_key = {case.case_key: case for case in cases}
    record_keys = set(records["case_key"])
    if record_keys != set(by_key):
        raise V4ExperimentError("Evaluation records no longer match their V4 case provenance")
    output = records.copy()
    output["candidate_count"] = output["case_key"].map(
        lambda key: by_key[str(key)].candidate_count
    )
    output["selected_neighbor_id"] = output["case_key"].map(
        lambda key: by_key[str(key)].selected_neighbor_id
    )
    output["signed_month_distance"] = output["case_key"].map(
        lambda key: by_key[str(key)].neighbor_delta_month
    )
    output["selection_rule_version"] = output["case_key"].map(
        lambda key: by_key[str(key)].selection_rule_version
    )
    output["cohort_fingerprint"] = output["case_key"].map(
        lambda key: by_key[str(key)].cohort_fingerprint
    )
    output["neighbor_context_available_at_origin"] = output["case_key"].map(
        lambda key: by_key[str(key)].neighbor_context_available_at_origin
    )
    return output


def _path_records_for_direction_plot(
    records: pd.DataFrame,
    *,
    fold_id: str,
    model_name: str,
) -> pd.DataFrame:
    """Convert V3 path records to the generic direction-comparison schema."""

    required = {
        "case_key",
        "target_contract_id",
        "product",
        "target_end_day",
        "origin_close",
        "day_end_indices",
        "sample_paths",
        "day3_actual_direction",
        "day3_predicted_direction",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise V4ExperimentError(f"Path records miss direction columns: {missing!r}")
    output = records.copy()

    def probability_up(row: pd.Series) -> float:
        samples = np.asarray(row["sample_paths"], dtype=np.float64)
        if samples.ndim != 3 or samples.shape[-1] < 4:
            raise V4ExperimentError(f"Invalid sampled path tensor for {row['case_key']}")
        day_end = int(list(row["day_end_indices"])[-1])
        if day_end < 0 or day_end >= samples.shape[1]:
            raise V4ExperimentError(f"Invalid Day3 endpoint for {row['case_key']}")
        origin_close = float(row["origin_close"])
        if origin_close <= 0:
            raise V4ExperimentError(f"Non-positive origin close for {row['case_key']}")
        return float(np.mean(samples[:, day_end, 3] > origin_close))

    output["actual_direction"] = output["day3_actual_direction"].astype(np.int8)
    output["predicted_direction"] = output["day3_predicted_direction"].astype(np.int8)
    output["probability_up"] = output.apply(probability_up, axis=1)
    output["valid_direction"] = output["actual_direction"] != 0
    output["actual_label"] = (output["actual_direction"] == 1).astype(np.int8)
    output["predicted_label"] = (output["predicted_direction"] == 1).astype(np.int8)
    output["fold_id"] = str(fold_id)
    output["model"] = str(model_name)
    return output.sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _majority_direction_records(
    zero_shot_records: pd.DataFrame,
    *,
    fold_id: str,
    probability_up: float,
) -> pd.DataFrame:
    if not 0.0 <= probability_up <= 1.0:
        raise ValueError("majority probability must be in [0, 1]")
    records = _path_records_for_direction_plot(
        zero_shot_records, fold_id=fold_id, model_name="majority"
    )
    direction = 1 if probability_up >= 0.5 else -1
    records["probability_up"] = probability_up
    records["predicted_direction"] = direction
    records.loc[records["actual_direction"] == 0, "predicted_direction"] = 0
    # Majority is a direction-only baseline.  It must not inherit the
    # zero-shot sampled trajectory copied above, or a V2-style path plot would
    # incorrectly present the candidate forecast as a majority forecast.
    records = records.drop(columns=["predicted_path", "sample_paths"], errors="ignore")
    return records


def _balanced_accuracy_improvement(
    candidate: pd.DataFrame, baseline: pd.DataFrame
) -> float | None:
    _assert_paired_records(candidate, baseline, label="balanced-accuracy comparison")
    candidate_ba = float(probe_metrics(candidate)["balanced_accuracy"])
    baseline_ba = float(probe_metrics(baseline)["balanced_accuracy"])
    if not math.isfinite(candidate_ba) or not math.isfinite(baseline_ba):
        return None
    return candidate_ba - baseline_ba


def _median_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _seed_ensemble_records(
    records_by_seed: Mapping[int, pd.DataFrame],
    *,
    model_name: str,
) -> pd.DataFrame:
    """Average per-seed probabilities while preserving every matched case field."""

    if not records_by_seed:
        raise V4ExperimentError(f"Cannot ensemble zero seed records for {model_name}")
    ordered = [(int(seed), frame.copy()) for seed, frame in sorted(records_by_seed.items())]
    first_seed, first = ordered[0]
    if first.empty:
        raise V4ExperimentError(f"Cannot ensemble empty records for {model_name}")
    first_indexed = first.set_index("case_key", verify_integrity=True).sort_index()
    required = {
        "fold_id",
        "product",
        "target_end_day",
        "target_contract_id",
        "actual_label",
        "actual_direction",
        "valid_direction",
        "probability_up",
    }
    missing = sorted(required.difference(first_indexed.columns))
    if missing:
        raise V4ExperimentError(f"Seed ensemble records miss columns: {missing!r}")
    probabilities = [pd.to_numeric(first_indexed["probability_up"], errors="raise").to_numpy()]
    for seed, frame in ordered[1:]:
        indexed = frame.set_index("case_key", verify_integrity=True).sort_index()
        if set(indexed.index) != set(first_indexed.index):
            raise V4ExperimentError(
                f"{model_name} seed {seed} does not use the same evaluation cases as seed {first_seed}"
            )
        indexed = indexed.loc[first_indexed.index]
        for column in (
            "fold_id",
            "product",
            "target_end_day",
            "target_contract_id",
            "actual_label",
            "actual_direction",
            "valid_direction",
        ):
            if not first_indexed[column].equals(indexed[column]):
                raise V4ExperimentError(
                    f"{model_name} seeds disagree on {column} for the same case"
                )
        probabilities.append(pd.to_numeric(indexed["probability_up"], errors="raise").to_numpy())
    stacked = np.vstack(probabilities)
    if not np.isfinite(stacked).all():
        raise V4ExperimentError(f"{model_name} has non-finite seed probabilities")
    output = first_indexed.reset_index().copy()
    output["probability_up"] = stacked.mean(axis=0)
    output["predicted_label"] = (output["probability_up"] >= 0.5).astype(np.int8)
    output["predicted_direction"] = np.where(output["predicted_label"] == 1, 1, -1)
    invalid = ~output["valid_direction"].astype(bool)
    output.loc[invalid, "actual_direction"] = 0
    output.loc[invalid, "predicted_direction"] = 0
    output["model"] = model_name
    output["seed"] = "ensemble"
    return output.sort_values(
        ["fold_id", "target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _bootstrap_from_ensemble(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    block_days: Sequence[int],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    _assert_paired_records(candidate, baseline, label="bootstrap comparison")
    unique_days = candidate["target_end_day"].nunique()
    output: dict[str, dict[str, object]] = {}
    for raw_block in block_days:
        block = int(raw_block)
        if unique_days < block:
            output[f"block_{block}"] = {
                "available": False,
                "reason": f"only {unique_days} evaluation days for {block}-day blocks",
            }
            continue
        output[f"block_{block}"] = {
            "available": True,
            **paired_block_bootstrap(
                candidate,
                baseline,
                block_days=block,
                iterations=iterations,
                seed=seed + block,
            ),
        }
    return output


def _granularity_gate(
    records_by_seed: Mapping[int, Mapping[str, pd.DataFrame]],
    *,
    pair_arm: str,
    target_arm: str,
    primary_products: Sequence[str],
    bootstrap_blocks: Sequence[int],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Evaluate the pre-registered P1 threshold for one model granularity."""

    expected_seeds = (42, 43, 44)
    if tuple(sorted(records_by_seed)) != expected_seeds:
        raise V4ExperimentError(
            f"{pair_arm} gate requires exactly seeds {expected_seeds!r}, got {tuple(sorted(records_by_seed))!r}"
        )
    seed_metrics: dict[str, object] = {}
    seed_improvements: list[float | None] = []
    by_product: dict[str, list[float | None]] = {str(product): [] for product in primary_products}
    by_direction: dict[str, list[float | None]] = {"earlier": [], "later": []}
    pair_seed_frames: dict[int, pd.DataFrame] = {}
    target_seed_frames: dict[int, pd.DataFrame] = {}
    for seed in expected_seeds:
        arms = records_by_seed[seed]
        try:
            pair = arms[pair_arm]
            target = arms[target_arm]
        except KeyError as exc:
            raise V4ExperimentError(f"Seed {seed} lacks required P1 arm {exc.args[0]!r}") from exc
        _assert_paired_records(pair, target, label=f"{pair_arm}/seed_{seed}")
        pair_seed_frames[seed] = pair
        target_seed_frames[seed] = target
        improvement = _balanced_accuracy_improvement(pair, target)
        seed_improvements.append(improvement)
        product_metrics: dict[str, float | None] = {}
        for product in primary_products:
            pair_product = pair.loc[pair["product"] == product]
            target_product = target.loc[target["product"] == product]
            value = (
                _balanced_accuracy_improvement(pair_product, target_product)
                if not pair_product.empty and not target_product.empty
                else None
            )
            by_product[str(product)].append(value)
            product_metrics[str(product)] = value
        direction_metrics: dict[str, float | None] = {}
        for direction in ("earlier", "later"):
            pair_direction = pair.loc[pair["neighbor_direction"] == direction]
            target_direction = target.loc[target["neighbor_direction"] == direction]
            value = (
                _balanced_accuracy_improvement(pair_direction, target_direction)
                if not pair_direction.empty and not target_direction.empty
                else None
            )
            by_direction[direction].append(value)
            direction_metrics[direction] = value
        seed_metrics[str(seed)] = {
            "pair": probe_metrics(pair),
            "target_only": probe_metrics(target),
            "balanced_accuracy_improvement": improvement,
            "by_product": product_metrics,
            "by_neighbor_direction": direction_metrics,
        }
    pair_ensemble = _seed_ensemble_records(pair_seed_frames, model_name=f"{pair_arm}_seed_ensemble")
    target_ensemble = _seed_ensemble_records(
        target_seed_frames, model_name=f"{target_arm}_seed_ensemble"
    )
    _assert_paired_records(pair_ensemble, target_ensemble, label=f"{pair_arm}/ensemble")
    bootstrap = _bootstrap_from_ensemble(
        pair_ensemble,
        target_ensemble,
        block_days=bootstrap_blocks,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    fold_improvements: dict[str, float | None] = {}
    for fold_id, pair_fold in pair_ensemble.groupby("fold_id", sort=True):
        target_fold = target_ensemble.loc[
            target_ensemble["fold_id"].astype(str) == str(fold_id)
        ]
        fold_improvements[str(fold_id)] = _balanced_accuracy_improvement(pair_fold, target_fold)
    product_medians = {product: _median_or_none(values) for product, values in by_product.items()}
    direction_medians = {
        direction: _median_or_none(values) for direction, values in by_direction.items()
    }
    median_improvement = _median_or_none(seed_improvements)
    positive_seed_count = sum(value is not None and value > 0.0 for value in seed_improvements)
    improved_fold_count = sum(
        value is not None and value > 0.0 for value in fold_improvements.values()
    )
    bootstrap_pass = all(
        bool(result.get("available"))
        and float(result.get("probability_improvement_positive", 0.0)) >= 0.80
        for result in bootstrap.values()
    )
    conditions = {
        "median_seed_improvement_at_least_2pp": (
            median_improvement is not None and median_improvement >= 0.02
        ),
        "at_least_two_of_three_seed_improvements_positive": positive_seed_count >= 2,
        "at_least_three_of_five_folds_improve": (
            len(fold_improvements) == 5 and improved_fold_count >= 3
        ),
        "no_primary_product_median_degrades_more_than_1pp": all(
            value is not None and value >= -0.01 for value in product_medians.values()
        ),
        "no_neighbor_direction_median_degrades_more_than_1pp": all(
            value is not None and value >= -0.01 for value in direction_medians.values()
        ),
        "bootstrap_5_and_10_day_probability_at_least_80pct": bootstrap_pass,
    }
    return {
        "pair_arm": pair_arm,
        "target_arm": target_arm,
        "seed_metrics": seed_metrics,
        "seed_balanced_accuracy_improvements": {
            str(seed): value for seed, value in zip(expected_seeds, seed_improvements, strict=True)
        },
        "median_seed_balanced_accuracy_improvement": median_improvement,
        "positive_seed_count": positive_seed_count,
        "product_median_balanced_accuracy_improvements": product_medians,
        "neighbor_direction_median_balanced_accuracy_improvements": direction_medians,
        "fold_balanced_accuracy_improvements": fold_improvements,
        "improved_fold_count": improved_fold_count,
        "ensemble": {
            "pair": probe_metrics(pair_ensemble),
            "target_only": probe_metrics(target_ensemble),
            "pair_records": pair_ensemble,
            "target_only_records": target_ensemble,
        },
        "bootstrap": bootstrap,
        "conditions": conditions,
        "passes_granularity_gate": all(conditions.values()),
    }


def _select_granularity(
    shared: Mapping[str, object],
    per_product: Mapping[str, object],
    *,
    bootstrap_blocks: Sequence[int],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    primary_products: Sequence[str],
) -> dict[str, object]:
    """Apply the V4 post-ablation granularity selection rule exactly once."""

    shared_passes = bool(shared["passes_granularity_gate"])
    per_product_passes = bool(per_product["passes_granularity_gate"])
    result: dict[str, object] = {
        "shared_passes_granularity_gate": shared_passes,
        "per_product_passes_granularity_gate": per_product_passes,
        "selected_granularity": None,
        "allows_p2": False,
        "decision_reason": "",
    }
    if shared_passes and not per_product_passes:
        result.update(
            selected_granularity="shared",
            allows_p2=True,
            decision_reason="only_shared_passes_its_pre_registered_gate",
        )
        return result
    if per_product_passes and not shared_passes:
        result.update(
            selected_granularity="per_product",
            allows_p2=True,
            decision_reason="only_per_product_passes_its_pre_registered_gate",
        )
        return result
    if not shared_passes and not per_product_passes:
        result["decision_reason"] = "neither_granularity_passes_its_pre_registered_gate"
        return result
    shared_pair = shared["ensemble"]["pair_records"]  # type: ignore[index]
    per_product_pair = per_product["ensemble"]["pair_records"]  # type: ignore[index]
    if not isinstance(shared_pair, pd.DataFrame) or not isinstance(per_product_pair, pd.DataFrame):
        raise V4ExperimentError("Granularity gate has no ensemble pair records")
    _assert_paired_records(shared_pair, per_product_pair, label="granularity selection")
    shared_ba = float(probe_metrics(shared_pair)["balanced_accuracy"])
    per_product_ba = float(probe_metrics(per_product_pair)["balanced_accuracy"])
    if not math.isfinite(shared_ba) or not math.isfinite(per_product_ba):
        raise V4ExperimentError("Granularity selection has non-finite pair balanced accuracy")
    bootstrap = _bootstrap_from_ensemble(
        shared_pair,
        per_product_pair,
        block_days=bootstrap_blocks,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    by_product_difference: dict[str, float | None] = {}
    for product in primary_products:
        shared_product = shared_pair.loc[shared_pair["product"] == product]
        per_product_product = per_product_pair.loc[per_product_pair["product"] == product]
        if shared_product.empty or per_product_product.empty:
            by_product_difference[str(product)] = None
            continue
        by_product_difference[str(product)] = _balanced_accuracy_improvement(
            shared_product, per_product_product
        )
    shared_product_safe = all(
        value is not None and value >= -0.01 for value in by_product_difference.values()
    )
    per_product_safe = all(
        value is not None and value <= 0.01 for value in by_product_difference.values()
    )
    reverse_bootstrap = _bootstrap_from_ensemble(
        per_product_pair,
        shared_pair,
        block_days=bootstrap_blocks,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed + 100,
    )
    shared_probability = float(
        bootstrap["block_5"].get("probability_improvement_positive", 0.0)
    ) if bool(bootstrap["block_5"].get("available")) else 0.0
    shared_probability_10 = float(
        bootstrap["block_10"].get("probability_improvement_positive", 0.0)
    ) if bool(bootstrap["block_10"].get("available")) else 0.0
    shared_decisive = (
        shared_ba - per_product_ba >= 0.01
        and shared_probability >= 0.80
        and shared_probability_10 >= 0.80
        and shared_product_safe
    )
    per_product_probability = float(
        reverse_bootstrap["block_5"].get("probability_improvement_positive", 0.0)
    ) if bool(reverse_bootstrap["block_5"].get("available")) else 0.0
    per_product_probability_10 = float(
        reverse_bootstrap["block_10"].get("probability_improvement_positive", 0.0)
    ) if bool(reverse_bootstrap["block_10"].get("available")) else 0.0
    per_product_decisive = (
        per_product_ba - shared_ba >= 0.01
        and per_product_probability >= 0.80
        and per_product_probability_10 >= 0.80
        and per_product_safe
    )
    result["paired_absolute_ba_comparison"] = {
        "shared_pair_balanced_accuracy": shared_ba,
        "per_product_pair_balanced_accuracy": per_product_ba,
        "shared_minus_per_product": shared_ba - per_product_ba,
        "shared_minus_per_product_by_product": by_product_difference,
        "bootstrap_shared_minus_per_product": bootstrap,
        "bootstrap_per_product_minus_shared": reverse_bootstrap,
    }
    if shared_decisive:
        result.update(
            selected_granularity="shared",
            allows_p2=True,
            decision_reason="shared_has_a_decisive_paired_absolute_ba_advantage",
        )
    elif per_product_decisive:
        result.update(
            selected_granularity="per_product",
            allows_p2=True,
            decision_reason="per_product_has_a_decisive_paired_absolute_ba_advantage",
        )
    elif not shared_product_safe and per_product_safe:
        result.update(
            selected_granularity="per_product",
            allows_p2=True,
            decision_reason="shared_triggers_product_degradation_protection",
        )
    else:
        # Both arms passed and no decisive difference exists; shared is the
        # pre-registered parsimony default, retaining j transfer evaluation.
        result.update(
            selected_granularity="shared",
            allows_p2=True,
            decision_reason="no_decisive_granularity_difference_default_to_shared",
        )
    return result


def _gate_for_json(gate: Mapping[str, object]) -> dict[str, object]:
    """Remove in-memory record frames before persisting the P1 decision file."""

    output = dict(gate)
    ensemble = output.get("ensemble")
    if isinstance(ensemble, Mapping):
        output["ensemble"] = {
            key: value
            for key, value in ensemble.items()
            if key not in {"pair_records", "target_only_records"}
        }
    return output


class V4Experiment:
    """V4 runner through P1; later phases are intentionally guarded placeholders."""

    def __init__(
        self,
        config_path: str | Path,
        run_id: str,
        *,
        device_override: str | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.config = load_v4_config(config_path)
        if device_override is not None:
            self.config["runtime"]["device"] = device_override
        self.run_id = str(run_id)
        self.run_dir = Path(self.config["output"]["root"]) / self.run_id
        self.results_dir = Path(self.config["output"]["results_root"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.local_files_only = not allow_model_download
        self._device: torch.device | None = None
        self.cohort = load_observed_contract_cohort(self.config["data"]["snapshot_root"])
        walk = self.config["walk_forward"]
        self.audit_payload, bundles, self.folds = build_observed_cohort_audit(
            self.cohort,
            lookbacks=self.config["data"]["lookbacks"],
            products=self.config["data"]["products"],
            minimum_fit_days=int(walk["minimum_fit_days"]),
            inner_validation_days=int(walk["inner_validation_days"]),
            evaluation_days=int(walk["evaluation_days"]),
            step_days=int(walk["step_days"]),
            purge_days=int(walk["purge_days"]),
            model_provenance=self._model_provenance(),
            data_provenance={
                "lookback": int(self.config["data"]["lookback"]),
                "clip": float(self.config["data"]["clip"]),
                "normalization_epsilon": float(self.config["data"]["normalization_epsilon"]),
            },
        )
        self.lookback = int(self.config["data"]["lookback"])
        self.bundle: ObservedCohortCaseBundle = bundles[self.lookback]
        self.config["runtime_resolved"] = {
            "device_requested": str(self.config["runtime"]["device"]),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "model_download_allowed": allow_model_download,
            "primary_context_length": self.lookback,
            "cohort_fingerprint": self.cohort.cohort_fingerprint,
        }
        _write_json(self.run_dir / "resolved_config.json", self.config)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = resolve_device(str(self.config["runtime"]["device"]))
            self.config["runtime_resolved"].update(
                {
                    "device": str(self._device),
                    "cuda_device": torch.cuda.get_device_name(self._device)
                    if self._device.type == "cuda"
                    else None,
                }
            )
            _write_json(self.run_dir / "resolved_config.json", self.config)
        return self._device

    @property
    def products(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config["data"]["products"])

    @property
    def primary_products(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config["data"]["primary_selection_products"])

    @property
    def transfer_products(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config["data"]["transfer_products"])

    def _model_provenance(self) -> dict[str, object]:
        model = self.config["model"]
        return {
            "tokenizer_id": model["tokenizer_id"],
            "tokenizer_revision": model["tokenizer_revision"],
            "predictor_id": model["predictor_id"],
            "predictor_revision": model["predictor_revision"],
        }

    def _metadata(self, phase: str) -> dict[str, object]:
        return {
            "strategy_version": 4,
            "phase": phase,
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": PRODUCTION_ELIGIBLE,
            "cohort_fingerprint": self.cohort.cohort_fingerprint,
            "evaluation_contract_version": DIRECTION_COMPARISON_CONTRACT_VERSION,
            **self._model_provenance(),
        }

    def _require_folds(self) -> tuple[V3WalkForwardFold, ...]:
        if not self.folds:
            raise V4ExperimentError("V4 observed cohort has no complete walk-forward folds")
        return self.folds

    def _load_tokenizer(self) -> KronosTokenizer:
        model = self.config["model"]
        tokenizer = KronosTokenizer.from_pretrained(
            model["tokenizer_id"],
            revision=model["tokenizer_revision"],
            cache_dir=model["cache_dir"],
            local_files_only=self.local_files_only,
        )
        tokenizer.requires_grad_(False)
        tokenizer.eval()
        return tokenizer

    def _load_predictor(self) -> Kronos:
        model = self.config["model"]
        return Kronos.from_pretrained(
            model["predictor_id"],
            revision=model["predictor_revision"],
            cache_dir=model["cache_dir"],
            local_files_only=self.local_files_only,
        )

    def _release(self, *models: torch.nn.Module) -> None:
        for model in models:
            model.to("cpu")
        gc.collect()
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def audit(self) -> Path:
        payload = {**self._metadata("audit"), **self.audit_payload}
        result_path = self.results_dir / "v4_data_audit.json"
        _write_json(result_path, payload)
        _write_json(self.run_dir / "data_audit.json", payload)
        print(json.dumps(_json_safe({"data_audit": result_path, "lookback": self.lookback})))
        return result_path

    def _p0_evaluation_arguments(self) -> dict[str, object]:
        evaluation = self.config["evaluation"]
        data = self.config["data"]
        return {
            "device": self.device,
            "max_context": int(self.config["model"]["max_context"]),
            "clip": float(data["clip"]),
            "epsilon": float(data["normalization_epsilon"]),
            "sample_count": int(evaluation["sample_count"]),
            "temperature": float(evaluation["temperature"]),
            "top_k": int(evaluation["top_k"]),
            "top_p": float(evaluation["top_p"]),
            "batch_size": int(evaluation["inference_batch_size"]),
            "seed": int(evaluation["random_seed"]),
        }

    def run_p0(
        self,
        *,
        fold_id: str | None = None,
        max_evaluation_cases: int | None = None,
    ) -> Path:
        """Run V4 zero-shot versus fit-period majority on the common pair cases."""

        folds = self._require_folds()
        if fold_id is not None:
            folds = tuple(fold for fold in folds if fold.fold_id == str(fold_id))
            if not folds:
                raise V4ExperimentError(f"Unknown V4 fold for P0 smoke: {fold_id!r}")
        if max_evaluation_cases is not None and max_evaluation_cases < 1:
            raise ValueError("max_evaluation_cases must be positive when supplied")
        smoke = fold_id is not None or max_evaluation_cases is not None
        stage_name = "p0_smoke" if smoke else "p0"
        artifacts: dict[str, EvaluationArtifacts] = {}
        all_zero: list[pd.DataFrame] = []
        all_majority: list[pd.DataFrame] = []
        for fold in folds:
            fit_cases = _cases_in_period(
                self.bundle.pair_cases,
                start_day=fold.fit_start_day,
                end_day=fold.fit_end_day,
            )
            evaluation_cases = _cases_in_period(
                self.bundle.pair_cases,
                start_day=fold.evaluation_start_day,
                end_day=fold.evaluation_end_day,
            )
            if max_evaluation_cases is not None:
                evaluation_cases = evaluation_cases[:max_evaluation_cases]
            if not fit_cases or not evaluation_cases:
                raise V4ExperimentError(f"P0 {fold.fold_id} lacks common observed-cohort cases")
            nonzero_fit = [case.day3_return > 0.0 for case in fit_cases if case.day3_return != 0.0]
            majority_probability = float(np.mean(nonzero_fit)) if nonzero_fit else 0.5
            tokenizer = self._load_tokenizer()
            predictor = self._load_predictor()
            fold_dir = self.run_dir / stage_name / fold.fold_id
            try:
                zero_raw = evaluate_target_paths(
                    tokenizer,
                    predictor,
                    evaluation_cases,
                    model_name="zero_shot",
                    **self._p0_evaluation_arguments(),
                )
                zero_raw = _attach_case_provenance(zero_raw, evaluation_cases)
                zero = _path_records_for_direction_plot(
                    zero_raw, fold_id=fold.fold_id, model_name="zero_shot"
                )
                majority = _majority_direction_records(
                    zero_raw,
                    fold_id=fold.fold_id,
                    probability_up=majority_probability,
                )
                _assert_paired_records(zero, majority, label=f"p0/{fold.fold_id}")
                _write_json(fold_dir / "zero_shot_records.json", zero.to_dict("records"))
                _write_json(fold_dir / "majority_records.json", majority.to_dict("records"))
                artifacts[fold.fold_id] = render_fold_direction_comparison(
                    {"zero_shot": zero, "majority": majority},
                    fold_id=fold.fold_id,
                    candidate_model="zero_shot",
                    baseline_model="majority",
                    output_dir=self.run_dir / stage_name / "evaluation",
                    stage=stage_name,
                    metadata={
                        **self._metadata("p0"),
                        "run_mode": "cpu_smoke" if smoke else "full",
                        "majority_probability_up_from_fit_cases": majority_probability,
                    },
                )
                all_zero.append(zero)
                all_majority.append(majority)
            finally:
                self._release(tokenizer, predictor)
        stage_report = write_direction_stage_report(
            self.run_dir / stage_name / "evaluation",
            stage=stage_name,
            fold_artifacts=artifacts,
            metadata={**self._metadata("p0"), "run_mode": "cpu_smoke" if smoke else "full"},
        )
        zero_combined = pd.concat(all_zero, ignore_index=True)
        majority_combined = pd.concat(all_majority, ignore_index=True)
        payload = {
            **self._metadata("p0"),
            "run_mode": "cpu_smoke" if smoke else "full",
            "record_count": len(zero_combined),
            "zero_shot": _direction_metrics(zero_combined),
            "majority": _direction_metrics(majority_combined),
            "direction_stage_report": stage_report,
        }
        result_path = self.results_dir / ("p0_smoke_metrics.json" if smoke else "p0_metrics.json")
        _write_json(result_path, payload)
        _write_json(self.run_dir / stage_name / "metrics.json", payload)
        print(json.dumps(_json_safe({"stage": stage_name, "metrics": result_path})))
        return result_path

    def _probe_training_config(self, *, seed: int, sampling_strategy: str) -> ProbeTrainingConfig:
        training = self.config["p1"]["training"]
        return ProbeTrainingConfig(
            learning_rate=float(training["learning_rate"]),
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            weight_decay=float(training["weight_decay"]),
            gradient_clip=float(training["gradient_clip"]),
            num_workers=int(training["num_workers"]),
            seed=int(seed),
            sampling_strategy=sampling_strategy,
        )

    def _run_matched_probe_arms(
        self,
        *,
        tokenizer: KronosTokenizer,
        predictor: Kronos,
        fit_cases: Sequence[ObservedCohortCase],
        validation_cases: Sequence[ObservedCohortCase],
        evaluation_cases: Sequence[ObservedCohortCase],
        auxiliary_evaluation_cases: Sequence[ObservedCohortCase],
        fold_id: str,
        seed: int,
        target_model_name: str,
        pair_model_name: str,
        sampling_strategy: str,
        output_dir: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        if not fit_cases or not validation_cases or not evaluation_cases:
            raise V4ExperimentError(
                f"{fold_id}/{pair_model_name} lacks fit, validation, or evaluation cases"
            )
        assert_same_case_keys(fit_cases, fit_cases)
        assert_same_case_keys(validation_cases, validation_cases)
        assert_same_case_keys(evaluation_cases, evaluation_cases)
        p1 = self.config["p1"]
        data = self.config["data"]
        training = self._probe_training_config(seed=seed, sampling_strategy=sampling_strategy)
        torch.manual_seed(seed)
        np.random.seed(seed)
        template = PairProbe(
            tokenizer,
            predictor,
            fusion_hidden_dim=int(p1["fusion_hidden_dim"]),
            dropout=float(p1["dropout"]),
        )
        initial_head = template.head_state_dict()
        target_probe = PairProbe(
            tokenizer,
            predictor,
            fusion_hidden_dim=int(p1["fusion_hidden_dim"]),
            dropout=float(p1["dropout"]),
        )
        pair_probe = PairProbe(
            tokenizer,
            predictor,
            fusion_hidden_dim=int(p1["fusion_hidden_dim"]),
            dropout=float(p1["dropout"]),
        )
        target_probe.load_head_state_dict(initial_head)
        pair_probe.load_head_state_dict(initial_head)
        target_result = train_probe(
            target_probe,
            fit_cases,
            validation_cases,
            mode="target_only_probe",
            config=training,
            device=self.device,
            output_dir=output_dir / "target_only_probe",
            clip=float(data["clip"]),
            epsilon=float(data["normalization_epsilon"]),
        )
        pair_result = train_probe(
            pair_probe,
            fit_cases,
            validation_cases,
            mode="pair_probe",
            config=training,
            device=self.device,
            output_dir=output_dir / "pair_probe",
            clip=float(data["clip"]),
            epsilon=float(data["normalization_epsilon"]),
        )
        load_probe_head(target_probe, target_result.checkpoint_path)
        load_probe_head(pair_probe, pair_result.checkpoint_path)
        target_raw = _attach_case_provenance(
            evaluate_probe(
                target_probe,
                evaluation_cases,
                mode="target_only_probe",
                device=self.device,
                batch_size=training.batch_size,
                clip=float(data["clip"]),
                epsilon=float(data["normalization_epsilon"]),
            ),
            evaluation_cases,
        )
        pair_raw = _attach_case_provenance(
            evaluate_probe(
                pair_probe,
                evaluation_cases,
                mode="pair_probe",
                device=self.device,
                batch_size=training.batch_size,
                clip=float(data["clip"]),
                epsilon=float(data["normalization_epsilon"]),
            ),
            evaluation_cases,
        )
        target_records = _probe_records_for_direction_plot(
            target_raw,
            fold_id=fold_id,
            model_name=target_model_name,
            seed=seed,
        )
        pair_records = _probe_records_for_direction_plot(
            pair_raw,
            fold_id=fold_id,
            model_name=pair_model_name,
            seed=seed,
        )
        _assert_paired_records(pair_records, target_records, label=f"{fold_id}/{pair_model_name}")
        target_aux: pd.DataFrame | None = None
        pair_aux: pd.DataFrame | None = None
        if auxiliary_evaluation_cases:
            target_aux = _probe_records_for_direction_plot(
                _attach_case_provenance(
                    evaluate_probe(
                        target_probe,
                        auxiliary_evaluation_cases,
                        mode="target_only_probe",
                        device=self.device,
                        batch_size=training.batch_size,
                        clip=float(data["clip"]),
                        epsilon=float(data["normalization_epsilon"]),
                    ),
                    auxiliary_evaluation_cases,
                ),
                fold_id=fold_id,
                model_name=target_model_name,
                seed=seed,
            )
            pair_aux = _probe_records_for_direction_plot(
                _attach_case_provenance(
                    evaluate_probe(
                        pair_probe,
                        auxiliary_evaluation_cases,
                        mode="pair_probe",
                        device=self.device,
                        batch_size=training.batch_size,
                        clip=float(data["clip"]),
                        epsilon=float(data["normalization_epsilon"]),
                    ),
                    auxiliary_evaluation_cases,
                ),
                fold_id=fold_id,
                model_name=pair_model_name,
                seed=seed,
            )
            _assert_paired_records(pair_aux, target_aux, label=f"{fold_id}/{pair_model_name}/transfer")
        _write_json(output_dir / "target_only_records.json", target_records.to_dict("records"))
        _write_json(output_dir / "pair_records.json", pair_records.to_dict("records"))
        if target_aux is not None and pair_aux is not None:
            _write_json(output_dir / "transfer_target_only_records.json", target_aux.to_dict("records"))
            _write_json(output_dir / "transfer_pair_records.json", pair_aux.to_dict("records"))
        _write_json(
            output_dir / "summary.json",
            {
                **self._metadata("p1_ablation"),
                "seed": seed,
                "sampling_strategy": sampling_strategy,
                "target_only": {
                    "checkpoint": target_result.checkpoint_path,
                    "best_epoch": target_result.best_epoch,
                    "best_balanced_accuracy": target_result.best_balanced_accuracy,
                    "sampling": target_result.sampling_summary,
                    "elapsed_seconds": target_result.elapsed_seconds,
                },
                "pair": {
                    "checkpoint": pair_result.checkpoint_path,
                    "best_epoch": pair_result.best_epoch,
                    "best_balanced_accuracy": pair_result.best_balanced_accuracy,
                    "sampling": pair_result.sampling_summary,
                    "elapsed_seconds": pair_result.elapsed_seconds,
                },
                "target_only_metrics": probe_metrics(target_records),
                "pair_metrics": probe_metrics(pair_records),
            },
        )
        del template, target_probe, pair_probe
        return target_records, pair_records, target_aux, pair_aux

    def run_p1_ablation(self, *, fold_id: str | None = None) -> Path:
        """Run the frozen P1 ablation, or one non-gating CUDA smoke fold."""

        folds = self._require_folds()
        if fold_id is not None:
            folds = tuple(fold for fold in folds if fold.fold_id == str(fold_id))
            if not folds:
                raise V4ExperimentError(f"Unknown V4 fold for P1 smoke: {fold_id!r}")
        smoke = fold_id is not None
        stage_name = "p1_ablation_smoke" if smoke else "p1_ablation"
        p1 = self.config["p1"]
        seeds = tuple(int(seed) for seed in p1["seeds"])
        all_seed_records: dict[int, dict[str, list[pd.DataFrame]]] = {
            seed: defaultdict(list) for seed in seeds
        }
        transfer_seed_records: dict[int, dict[str, list[pd.DataFrame]]] = {
            seed: defaultdict(list) for seed in seeds
        }
        primary_artifacts: dict[str, EvaluationArtifacts] = {}
        transfer_artifacts: dict[str, EvaluationArtifacts] = {}
        for fold in folds:
            fold_seed_records: dict[int, dict[str, pd.DataFrame]] = {}
            fold_transfer_records: dict[int, dict[str, pd.DataFrame]] = {}
            shared_fit = _cases_in_period(
                self.bundle.pair_cases,
                start_day=fold.fit_start_day,
                end_day=fold.fit_end_day,
                products=self.primary_products,
            )
            shared_validation = _cases_in_period(
                self.bundle.pair_cases,
                start_day=fold.inner_validation_start_day,
                end_day=fold.inner_validation_end_day,
                products=self.primary_products,
            )
            shared_evaluation = _cases_in_period(
                self.bundle.pair_cases,
                start_day=fold.evaluation_start_day,
                end_day=fold.evaluation_end_day,
                products=self.primary_products,
            )
            transfer_evaluation = _cases_in_period(
                self.bundle.pair_cases,
                start_day=fold.evaluation_start_day,
                end_day=fold.evaluation_end_day,
                products=self.transfer_products,
            )
            if not shared_fit or not shared_validation or not shared_evaluation:
                raise V4ExperimentError(f"P1 shared arm lacks cases for {fold.fold_id}")
            for seed in seeds:
                tokenizer = self._load_tokenizer()
                predictor = self._load_predictor()
                try:
                    seed_root = self.run_dir / stage_name / fold.fold_id / f"seed_{seed}"
                    shared_target, shared_pair, transfer_target, transfer_pair = self._run_matched_probe_arms(
                        tokenizer=tokenizer,
                        predictor=predictor,
                        fit_cases=shared_fit,
                        validation_cases=shared_validation,
                        evaluation_cases=shared_evaluation,
                        auxiliary_evaluation_cases=transfer_evaluation,
                        fold_id=fold.fold_id,
                        seed=seed,
                        target_model_name="shared_target_only",
                        pair_model_name="shared_pair",
                        sampling_strategy=str(p1["training"]["shared_sampling_strategy"]),
                        output_dir=seed_root / "shared",
                    )
                    product_target: list[pd.DataFrame] = []
                    product_pair: list[pd.DataFrame] = []
                    for product in self.primary_products:
                        fit = _cases_in_period(
                            self.bundle.pair_cases,
                            start_day=fold.fit_start_day,
                            end_day=fold.fit_end_day,
                            products=(product,),
                        )
                        validation = _cases_in_period(
                            self.bundle.pair_cases,
                            start_day=fold.inner_validation_start_day,
                            end_day=fold.inner_validation_end_day,
                            products=(product,),
                        )
                        evaluation = _cases_in_period(
                            self.bundle.pair_cases,
                            start_day=fold.evaluation_start_day,
                            end_day=fold.evaluation_end_day,
                            products=(product,),
                        )
                        target, pair, _, _ = self._run_matched_probe_arms(
                            tokenizer=tokenizer,
                            predictor=predictor,
                            fit_cases=fit,
                            validation_cases=validation,
                            evaluation_cases=evaluation,
                            auxiliary_evaluation_cases=(),
                            fold_id=fold.fold_id,
                            seed=seed,
                            target_model_name="per_product_target_only",
                            pair_model_name="per_product_pair",
                            sampling_strategy=str(
                                p1["training"]["per_product_sampling_strategy"]
                            ),
                            output_dir=seed_root / "per_product" / product,
                        )
                        product_target.append(target)
                        product_pair.append(pair)
                    per_product_target = pd.concat(product_target, ignore_index=True)
                    per_product_pair = pd.concat(product_pair, ignore_index=True)
                    for candidate, baseline, label in (
                        (shared_pair, shared_target, "shared"),
                        (per_product_pair, per_product_target, "per_product"),
                        (shared_pair, per_product_pair, "four_arm_common_evaluation"),
                        (shared_target, per_product_target, "four_arm_common_evaluation"),
                    ):
                        _assert_paired_records(candidate, baseline, label=f"{fold.fold_id}/{label}")
                    fold_seed_records[seed] = {
                        "shared_target_only": shared_target,
                        "shared_pair": shared_pair,
                        "per_product_target_only": per_product_target,
                        "per_product_pair": per_product_pair,
                    }
                    if transfer_target is not None and transfer_pair is not None:
                        fold_transfer_records[seed] = {
                            "shared_target_only": transfer_target,
                            "shared_pair": transfer_pair,
                        }
                finally:
                    self._release(tokenizer, predictor)
            ensemble_records = {
                arm: _seed_ensemble_records(
                    {seed: fold_seed_records[seed][arm] for seed in seeds},
                    model_name=f"{arm}_seed_ensemble",
                )
                for arm in (
                    "shared_target_only",
                    "shared_pair",
                    "per_product_target_only",
                    "per_product_pair",
                )
            }
            _assert_paired_records(
                ensemble_records["shared_pair"],
                ensemble_records["per_product_pair"],
                label=f"{fold.fold_id}/ensemble_common_evaluation",
            )
            primary_plot_records = {
                str(records["model"].iloc[0]): records
                for records in ensemble_records.values()
            }
            primary_artifacts[fold.fold_id] = render_fold_ablation_direction_comparison(
                primary_plot_records,
                fold_id=fold.fold_id,
                comparisons={
                    "shared": ("shared_pair_seed_ensemble", "shared_target_only_seed_ensemble"),
                    "per-product": (
                        "per_product_pair_seed_ensemble",
                        "per_product_target_only_seed_ensemble",
                    ),
                },
                output_dir=self.run_dir / stage_name / "evaluation",
                stage="p1_ablation",
                metadata=self._metadata("p1_ablation"),
            )
            if len(fold_transfer_records) == len(seeds):
                transfer_ensemble = {
                    arm: _seed_ensemble_records(
                        {seed: fold_transfer_records[seed][arm] for seed in seeds},
                        model_name=f"transfer_{arm}_seed_ensemble",
                    )
                    for arm in ("shared_target_only", "shared_pair")
                }
                transfer_plot_records = {
                    str(records["model"].iloc[0]): records
                    for records in transfer_ensemble.values()
                }
                transfer_artifacts[fold.fold_id] = render_fold_direction_comparison(
                    transfer_plot_records,
                    fold_id=fold.fold_id,
                    candidate_model="transfer_shared_pair_seed_ensemble",
                    baseline_model="transfer_shared_target_only_seed_ensemble",
                    output_dir=self.run_dir / stage_name / "transfer_evaluation",
                    stage="p1_ablation_shared_transfer",
                    metadata={**self._metadata("p1_ablation"), "auxiliary_transfer_products": self.transfer_products},
                )
            for seed in seeds:
                for arm, records in fold_seed_records[seed].items():
                    all_seed_records[seed][arm].append(records)
                for arm, records in fold_transfer_records.get(seed, {}).items():
                    transfer_seed_records[seed][arm].append(records)
        primary_report = write_direction_stage_report(
            self.run_dir / stage_name / "evaluation",
            stage="p1_ablation",
            fold_artifacts=primary_artifacts,
            metadata=self._metadata("p1_ablation"),
        )
        transfer_report = (
            write_direction_stage_report(
                self.run_dir / stage_name / "transfer_evaluation",
                stage="p1_ablation_shared_transfer",
                fold_artifacts=transfer_artifacts,
                metadata={**self._metadata("p1_ablation"), "auxiliary_transfer_products": self.transfer_products},
            )
            if transfer_artifacts
            else None
        )
        combined_seed_records = {
            seed: {
                arm: pd.concat(frames, ignore_index=True)
                for arm, frames in arms.items()
                if frames
            }
            for seed, arms in all_seed_records.items()
        }
        if smoke:
            metrics_payload = {
                **self._metadata("p1_ablation"),
                "run_mode": "single_fold_smoke",
                "smoke_fold_id": str(fold_id),
                "record_counts_by_seed": {
                    str(seed): {arm: len(records) for arm, records in arms.items()}
                    for seed, arms in combined_seed_records.items()
                },
                "per_seed_metrics": {
                    str(seed): {
                        arm: probe_metrics(records) for arm, records in arms.items()
                    }
                    for seed, arms in combined_seed_records.items()
                },
                "primary_direction_stage_report": primary_report,
                "transfer_direction_stage_report": transfer_report,
                "p1_gate": {
                    "available": False,
                    "reason": "single_fold_smoke_does_not_satisfy_the_five_fold_p1_gate",
                },
                "production_eligible": False,
            }
            result_path = self.results_dir / "p1_ablation_smoke_metrics.json"
            _write_json(result_path, metrics_payload)
            _write_json(self.run_dir / stage_name / "metrics.json", metrics_payload)
            print(
                json.dumps(
                    _json_safe(
                        {
                            "stage": "p1-ablation-smoke",
                            "metrics": result_path,
                            "gate": "not_written_for_single_fold_smoke",
                        }
                    )
                )
            )
            return result_path
        shared_gate = _granularity_gate(
            combined_seed_records,
            pair_arm="shared_pair",
            target_arm="shared_target_only",
            primary_products=self.primary_products,
            bootstrap_blocks=self.config["evaluation"]["bootstrap_block_days"],
            bootstrap_iterations=int(self.config["evaluation"]["bootstrap_iterations"]),
            bootstrap_seed=int(self.config["evaluation"]["random_seed"]),
        )
        per_product_gate = _granularity_gate(
            combined_seed_records,
            pair_arm="per_product_pair",
            target_arm="per_product_target_only",
            primary_products=self.primary_products,
            bootstrap_blocks=self.config["evaluation"]["bootstrap_block_days"],
            bootstrap_iterations=int(self.config["evaluation"]["bootstrap_iterations"]),
            bootstrap_seed=int(self.config["evaluation"]["random_seed"]) + 100,
        )
        selection = _select_granularity(
            shared_gate,
            per_product_gate,
            bootstrap_blocks=self.config["evaluation"]["bootstrap_block_days"],
            bootstrap_iterations=int(self.config["evaluation"]["bootstrap_iterations"]),
            bootstrap_seed=int(self.config["evaluation"]["random_seed"]) + 200,
            primary_products=self.primary_products,
        )
        transfer_metrics: dict[str, object] = {}
        if all(transfer_seed_records[seed] for seed in seeds):
            try:
                transfer_pair = _seed_ensemble_records(
                    {
                        seed: pd.concat(transfer_seed_records[seed]["shared_pair"], ignore_index=True)
                        for seed in seeds
                    },
                    model_name="transfer_shared_pair_seed_ensemble",
                )
                transfer_target = _seed_ensemble_records(
                    {
                        seed: pd.concat(
                            transfer_seed_records[seed]["shared_target_only"], ignore_index=True
                        )
                        for seed in seeds
                    },
                    model_name="transfer_shared_target_only_seed_ensemble",
                )
                transfer_metrics = {
                    "available": True,
                    "products": list(self.transfer_products),
                    "pair": probe_metrics(transfer_pair),
                    "target_only": probe_metrics(transfer_target),
                    "balanced_accuracy_improvement": _balanced_accuracy_improvement(
                        transfer_pair, transfer_target
                    ),
                }
            except (KeyError, V4ExperimentError):
                transfer_metrics = {
                    "available": False,
                    "products": list(self.transfer_products),
                    "reason": "not_every_seed_had_auxiliary_transfer_cases",
                }
        else:
            transfer_metrics = {
                "available": False,
                "products": list(self.transfer_products),
                "reason": "no_auxiliary_transfer_evaluation_in_all_folds",
            }
        gate_payload = {
            **self._metadata("p1_ablation"),
            "sampling": {
                "shared": p1["training"]["shared_sampling_strategy"],
                "per_product": p1["training"]["per_product_sampling_strategy"],
            },
            "primary_selection_products": list(self.primary_products),
            "auxiliary_transfer_products": list(self.transfer_products),
            "shared": _gate_for_json(shared_gate),
            "per_product": _gate_for_json(per_product_gate),
            "selection": selection,
            "production_eligible": False,
        }
        metrics_payload = {
            **self._metadata("p1_ablation"),
            "record_counts_by_seed": {
                str(seed): {arm: len(records) for arm, records in arms.items()}
                for seed, arms in combined_seed_records.items()
            },
            "primary_direction_stage_report": primary_report,
            "transfer_direction_stage_report": transfer_report,
            "auxiliary_shared_transfer": transfer_metrics,
            "gate_path": self.results_dir / "p1_granularity_gate.json",
            "production_eligible": False,
        }
        result_path = self.results_dir / "p1_ablation_metrics.json"
        gate_path = self.results_dir / "p1_granularity_gate.json"
        _write_json(result_path, metrics_payload)
        _write_json(gate_path, gate_payload)
        _write_json(self.run_dir / stage_name / "metrics.json", metrics_payload)
        _write_json(self.run_dir / stage_name / "p1_granularity_gate.json", gate_payload)
        print(
            json.dumps(
                _json_safe(
                    {
                        "stage": "p1-ablation",
                        "metrics": result_path,
                        "gate": gate_path,
                        "selection": selection,
                    }
                )
            )
        )
        return result_path

    def _require_p1_gate(self) -> Mapping[str, object]:
        path = self.results_dir / "p1_granularity_gate.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V4ExperimentError(
                "V4 P2/P3 requires p1_granularity_gate.json from a completed observed P1 run"
            ) from exc
        if not isinstance(value, dict):
            raise V4ExperimentError("V4 P1 gate must be a JSON object")
        expected_metadata = {
            "strategy_version": 4,
            "phase": "p1_ablation",
            "run_id": self.run_id,
            "result_scope": RESULT_SCOPE,
            "production_eligible": False,
            "cohort_fingerprint": self.cohort.cohort_fingerprint,
        }
        for key, expected in expected_metadata.items():
            if value.get(key) != expected:
                raise V4ExperimentError(
                    f"V4 P1 gate does not belong to the active run/cohort: {key}"
                )
        selection = value.get("selection")
        if not isinstance(selection, dict) or not bool(selection.get("allows_p2")):
            raise V4ExperimentError(
                "V4 P1 gate did not pass; refusing to run a later phase"
            )
        return value

    def run_p2(self) -> None:
        self._require_p1_gate()
        raise V4ExperimentError(
            "V4 P2 is intentionally not implemented before synchronized Phase 1 results "
            "are reviewed; this delivery stops at the P1 ablation gate."
        )

    def run_p3_stability(self) -> None:
        self._require_p1_gate()
        raise V4ExperimentError(
            "V4 P3 is unavailable because P2 has not been implemented or passed."
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kronos V4 observed-cohort experiment")
    parser.add_argument("stage", choices=("audit", "p0", "p1-ablation", "p2", "p3-stability"))
    parser.add_argument("--config", default="csj/configs/observed_contract_cohort_v4.yaml")
    parser.add_argument("--run-id", default="v4_cuda")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--fold-id",
        default=None,
        help=(
            "Run one P0 fold for a local smoke, or one P1 CUDA smoke fold "
            "without writing a P1 gate; omitted runs the full phase."
        ),
    )
    parser.add_argument(
        "--max-evaluation-cases",
        type=int,
        default=None,
        help="Limit P0 evaluation cases for a CPU smoke; omitted evaluates the full fold.",
    )
    args = parser.parse_args(argv)
    try:
        experiment = V4Experiment(
            args.config,
            args.run_id,
            device_override=args.device,
            allow_model_download=args.allow_model_download,
        )
        if args.stage == "audit":
            if args.fold_id is not None or args.max_evaluation_cases is not None:
                parser.error("--fold-id and --max-evaluation-cases apply only to P0/P1 smoke runs")
            experiment.audit()
        elif args.stage == "p0":
            experiment.run_p0(
                fold_id=args.fold_id,
                max_evaluation_cases=args.max_evaluation_cases,
            )
        elif args.stage == "p1-ablation":
            if args.max_evaluation_cases is not None:
                parser.error("--max-evaluation-cases applies only to P0")
            experiment.run_p1_ablation(fold_id=args.fold_id)
        elif args.stage == "p2":
            if args.fold_id is not None or args.max_evaluation_cases is not None:
                parser.error("--fold-id and --max-evaluation-cases do not apply to P2")
            experiment.run_p2()
        elif args.stage == "p3-stability":
            if args.fold_id is not None or args.max_evaluation_cases is not None:
                parser.error("--fold-id and --max-evaluation-cases do not apply to P3")
            experiment.run_p3_stability()
    except (V4ExperimentError, DirectionComparisonError, RuntimeError) as exc:
        parser.exit(2, f"V4 stage blocked: {exc}\n")


if __name__ == "__main__":
    main()


__all__ = [
    "V4Experiment",
    "V4ExperimentError",
    "_granularity_gate",
    "_seed_ensemble_records",
    "_select_granularity",
    "main",
    "resolve_device",
]
