"""P0 data, label-support, and no-leakage audit for V6."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from csj.v3.panel_data import V3WalkForwardFold, build_expanding_walk_forward_folds
from csj.v5.target_data import (
    TargetOnlyCase,
    TargetOnlyCaseBundle,
    TargetOnlyObservedCohort,
    build_target_only_cases,
    cases_in_period,
)
from csj.v6.risk_labels import (
    RiskLabelError,
    RiskLabelSpec,
    TailThresholds,
    apply_tail_thresholds,
    fit_tail_thresholds,
    future_mutation_context_leakage_checks,
    risk_outcome,
)


class P0AuditError(RuntimeError):
    """The V6 P0 audit cannot produce a truthful, complete result."""


@dataclass(frozen=True)
class P0AuditBundle:
    """Pure in-memory P0 evidence, ready for persistence and plotting."""

    audit: Mapping[str, object]
    gate: Mapping[str, object]
    outcomes: pd.DataFrame
    fold_records: pd.DataFrame
    folds: tuple[V3WalkForwardFold, ...]
    case_bundle: TargetOnlyCaseBundle


def _case_keys_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(str(value) for value in values):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _distribution(values: Sequence[float] | pd.Series | np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    output: dict[str, object] = {
        "count": int(len(array)),
        "finite_count": int(len(finite)),
        "nonfinite_count": int(len(array) - len(finite)),
    }
    if not len(finite):
        output.update(
            {
                "minimum": None,
                "p01": None,
                "p05": None,
                "p20": None,
                "median": None,
                "p80": None,
                "p95": None,
                "p99": None,
                "maximum": None,
                "mean": None,
                "standard_deviation": None,
            }
        )
        return output
    quantiles = np.quantile(
        finite,
        [0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99],
        method="linear",
    )
    output.update(
        {
            "minimum": float(np.min(finite)),
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p20": float(quantiles[2]),
            "median": float(quantiles[3]),
            "p80": float(quantiles[4]),
            "p95": float(quantiles[5]),
            "p99": float(quantiles[6]),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "standard_deviation": float(np.std(finite)),
        }
    )
    return output


def _counts_by(records: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, object]]:
    if records.empty:
        return []
    grouped = records.groupby(list(columns), sort=True, dropna=False).size()
    output: list[dict[str, object]] = []
    for keys, count in grouped.items():
        values = keys if isinstance(keys, tuple) else (keys,)
        output.append(
            {
                **{column: value for column, value in zip(columns, values, strict=True)},
                "cases": int(count),
            }
        )
    return output


def _event_support(records: pd.DataFrame) -> dict[str, object]:
    return {
        "cases": int(len(records)),
        "long_events": int(records["long_tail_event"].astype(bool).sum()),
        "short_events": int(records["short_tail_event"].astype(bool).sum()),
        "long_event_rate": (
            float(records["long_tail_event"].astype(bool).mean()) if len(records) else None
        ),
        "short_event_rate": (
            float(records["short_tail_event"].astype(bool).mean()) if len(records) else None
        ),
    }


def _outcome_distributions(records: pd.DataFrame) -> dict[str, object]:
    return {
        metric: _distribution(records[metric].to_numpy(dtype=np.float64))
        for metric in (
            "long_mae",
            "short_mae",
            "future_realized_scale",
            "future_vol_ratio",
            "context_sigma",
            "context_horizon_scale",
            "context_clip_fraction",
        )
    }


def _records_for_cases(outcomes: pd.DataFrame, cases: Sequence[TargetOnlyCase]) -> pd.DataFrame:
    keys = [case.case_key for case in cases]
    if not keys:
        return outcomes.iloc[0:0].copy()
    indexed = outcomes.set_index("case_key", verify_integrity=True)
    available = [key for key in keys if key in indexed.index]
    if not available:
        return outcomes.iloc[0:0].copy()
    return indexed.loc[available].reset_index()


def prediction_day_atomicity_check(
    records: pd.DataFrame, *, fold_id: str
) -> dict[str, object]:
    """Prove every origin trading day belongs to at most one split in a fold."""

    required = {"origin_trading_day", "split"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise P0AuditError(f"Prediction-day atomicity records miss columns: {missing!r}")
    per_origin_day = records.groupby("origin_trading_day", sort=False)["split"].nunique()
    violating_days = sorted(
        pd.Timestamp(day).normalize()
        for day, count in per_origin_day.items()
        if int(count) > 1
    )
    return {
        "check_id": f"{fold_id}:prediction_day_atomic",
        "passed": not violating_days,
        "maximum_split_assignments_per_origin_trading_day": (
            int(per_origin_day.max()) if len(per_origin_day) else 0
        ),
        "violating_origin_trading_days": violating_days,
    }


def _without_heavy_frame_attrs(bundle: TargetOnlyCaseBundle) -> TargetOnlyCaseBundle:
    """Detach source-loader lookup metadata from repeated per-case slices.

    The immutable loader stores a timestamp-position dictionary in
    ``DataFrame.attrs``. Pandas deep-copies attrs while finalizing every slice,
    even though V6 labels never read that dictionary. Keep the exact same
    blocks and values in a shallow frame copy, clear only those non-semantic
    attrs, and let every case for a contract share that stripped view.
    """

    contracts = {}
    for case in bundle.target_cases:
        contract_id = case.target_contract_id
        if contract_id in contracts:
            continue
        frame = case.target_contract.frame.copy(deep=False)
        frame.attrs = {}
        contracts[contract_id] = replace(case.target_contract, frame=frame)
    return replace(
        bundle,
        target_cases=tuple(
            replace(case, target_contract=contracts[case.target_contract_id])
            for case in bundle.target_cases
        ),
    )


def _fold_leakage_checks(
    *,
    fold: V3WalkForwardFold,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    thresholds: TailThresholds,
    primary_fit: pd.DataFrame,
    global_days: Sequence[pd.Timestamp],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    keys = {
        "fit": set(fit["case_key"].astype(str)),
        "inner_validation": set(validation["case_key"].astype(str)),
        "evaluation": set(evaluation["case_key"].astype(str)),
    }
    overlaps = {
        "fit_inner_validation": len(keys["fit"].intersection(keys["inner_validation"])),
        "fit_evaluation": len(keys["fit"].intersection(keys["evaluation"])),
        "inner_validation_evaluation": len(
            keys["inner_validation"].intersection(keys["evaluation"])
        ),
    }
    checks.append(
        {
            "check_id": f"{fold.fold_id}:split_case_key_disjoint",
            "passed": all(value == 0 for value in overlaps.values()),
            "observed_overlaps": overlaps,
        }
    )
    label_boundaries: dict[str, object] = {}
    boundary_passed = True
    for earlier_name, earlier, later_name, later in (
        ("fit", fit, "inner_validation", validation),
        ("inner_validation", validation, "evaluation", evaluation),
    ):
        if earlier.empty or later.empty:
            boundary_passed = False
            label_boundaries[f"{earlier_name}_to_{later_name}"] = None
            continue
        earlier_end = pd.to_datetime(earlier["target_end_day"]).max().normalize()
        later_start = pd.to_datetime(later["target_start_day"]).min().normalize()
        passed = bool(earlier_end < later_start)
        boundary_passed = boundary_passed and passed
        label_boundaries[f"{earlier_name}_to_{later_name}"] = {
            "earlier_latest_label_day": earlier_end,
            "later_earliest_label_day": later_start,
            "passed": passed,
        }
    checks.append(
        {
            "check_id": f"{fold.fold_id}:three_day_labels_do_not_cross_splits",
            "passed": boundary_passed,
            "boundaries": label_boundaries,
        }
    )
    normalized_days = [pd.Timestamp(day).normalize() for day in global_days]
    fit_validation_purge = sum(
        fold.fit_end_day < day < fold.inner_validation_start_day for day in normalized_days
    )
    validation_evaluation_purge = sum(
        fold.inner_validation_end_day < day < fold.evaluation_start_day
        for day in normalized_days
    )
    checks.append(
        {
            "check_id": f"{fold.fold_id}:global_calendar_purge",
            "passed": bool(
                fit_validation_purge >= fold.purge_days
                and validation_evaluation_purge >= fold.purge_days
            ),
            "required_days_per_boundary": int(fold.purge_days),
            "fit_to_inner_validation_days": int(fit_validation_purge),
            "inner_validation_to_evaluation_days": int(validation_evaluation_purge),
        }
    )
    expected_hash = _case_keys_sha256(primary_fit["case_key"].astype(str).tolist())
    checks.append(
        {
            "check_id": f"{fold.fold_id}:threshold_fit_primary_only",
            "passed": bool(
                thresholds.fit_case_keys_sha256 == expected_hash
                and thresholds.fit_case_count == len(primary_fit)
                and set(primary_fit["product"].astype(str)).issubset(
                    thresholds.primary_products
                )
            ),
            "expected_fit_case_keys_sha256": expected_hash,
            "observed_fit_case_keys_sha256": thresholds.fit_case_keys_sha256,
            "fit_case_count": int(len(primary_fit)),
            "threshold_fit_case_count": int(thresholds.fit_case_count),
        }
    )
    combined = pd.concat((fit, validation, evaluation), ignore_index=True)
    checks.append(prediction_day_atomicity_check(combined, fold_id=fold.fold_id))
    per_target_end_day = combined.groupby("target_end_day", sort=False)["split"].nunique()
    checks.append(
        {
            "check_id": f"{fold.fold_id}:target_completion_day_atomic",
            "passed": bool(
                per_target_end_day.empty or int(per_target_end_day.max()) == 1
            ),
            "maximum_split_assignments_per_target_end_day": (
                int(per_target_end_day.max()) if len(per_target_end_day) else 0
            ),
        }
    )
    return checks


def evaluate_p0_gate(
    records: pd.DataFrame,
    *,
    expected_fold_ids: Sequence[str],
    primary_products: Sequence[str],
    pooled_evaluation_products: Sequence[str] | None = None,
    p0_config: Mapping[str, Any],
    integrity_failures: Sequence[Mapping[str, object]],
    leakage_checks: Sequence[Mapping[str, object]],
    protocol_checks: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Evaluate only the pre-registered support, integrity, and leakage gates."""

    required = {
        "case_key",
        "fold_id",
        "split",
        "product",
        "long_tail_event",
        "short_tail_event",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise P0AuditError(f"P0 gate records miss columns: {missing!r}")
    if records.duplicated(["fold_id", "split", "case_key"]).any():
        raise P0AuditError("P0 gate records duplicate a case within one fold/split")
    conditions: list[dict[str, object]] = []
    support: dict[str, object] = {"folds": {}, "pooled_evaluation_by_product": {}}
    primary = tuple(str(value) for value in primary_products)
    pooled_products = (
        primary
        if pooled_evaluation_products is None
        else tuple(str(value) for value in pooled_evaluation_products)
    )
    if not set(pooled_products).issubset(primary):
        raise P0AuditError("Pooled evaluation products must be included in primary products")
    split_requirements = {
        "fit": int(p0_config["minimum_fit_events_per_side"]),
        "inner_validation": int(p0_config["minimum_validation_events_per_side"]),
        "evaluation": int(p0_config["minimum_evaluation_events_per_side"]),
    }
    for fold_id in expected_fold_ids:
        fold_support: dict[str, object] = {}
        fold_records = records.loc[records["fold_id"].astype(str) == str(fold_id)]
        for split, minimum in split_requirements.items():
            selected = fold_records.loc[
                (fold_records["split"].astype(str) == split)
                & fold_records["product"].astype(str).isin(primary)
            ]
            observed = _event_support(selected)
            fold_support[split] = observed
            for side in ("long", "short"):
                event_count = int(observed[f"{side}_events"])
                conditions.append(
                    {
                        "check_id": f"{fold_id}:{split}:{side}_event_support",
                        "passed": event_count >= minimum,
                        "observed": event_count,
                        "required_minimum": minimum,
                    }
                )
        support["folds"][str(fold_id)] = fold_support

    evaluation = records.loc[
        (records["split"].astype(str) == "evaluation")
        & records["product"].astype(str).isin(primary)
    ]
    duplicate_pooled_keys = int(evaluation["case_key"].astype(str).duplicated().sum())
    conditions.append(
        {
            "check_id": "pooled_evaluation_case_keys_unique_across_folds",
            "passed": duplicate_pooled_keys == 0,
            "observed_duplicate_case_keys": duplicate_pooled_keys,
            "required_maximum": 0,
        }
    )
    pooled_minimum = int(p0_config["minimum_pooled_evaluation_events_per_product_side"])
    for product in pooled_products:
        selected = evaluation.loc[evaluation["product"].astype(str) == product]
        observed = _event_support(selected)
        support["pooled_evaluation_by_product"][product] = observed
        for side in ("long", "short"):
            event_count = int(observed[f"{side}_events"])
            conditions.append(
                {
                    "check_id": f"pooled_evaluation:{product}:{side}_event_support",
                    "passed": event_count >= pooled_minimum,
                    "observed": event_count,
                    "required_minimum": pooled_minimum,
                }
            )

    failed_leakage = [check for check in leakage_checks if not bool(check.get("passed"))]
    failed_protocol = [check for check in protocol_checks if not bool(check.get("passed"))]
    integrity_count = int(len(integrity_failures))
    leakage_count = int(len(failed_leakage))
    integrity_condition = {
        "check_id": "included_case_integrity_failures",
        "passed": integrity_count <= int(p0_config["maximum_integrity_failures"]),
        "observed": integrity_count,
        "required_maximum": int(p0_config["maximum_integrity_failures"]),
    }
    leakage_condition = {
        "check_id": "past_only_and_split_leakage_failures",
        "passed": leakage_count <= int(p0_config["maximum_leakage_failures"]),
        "observed": leakage_count,
        "required_maximum": int(p0_config["maximum_leakage_failures"]),
    }
    conditions.extend((integrity_condition, leakage_condition))
    conditions.extend(dict(check) for check in protocol_checks)
    passed = bool(all(bool(condition.get("passed")) for condition in conditions))
    return {
        "gate_name": "v6_p0_risk_label_support_and_leakage",
        "available": True,
        "allows_next_phase": passed,
        "support": support,
        "conditions": conditions,
        "integrity_failure_count": integrity_count,
        "leakage_failure_count": leakage_count,
        "protocol_failure_count": int(len(failed_protocol)),
        "failed_condition_ids": [
            str(condition["check_id"])
            for condition in conditions
            if not bool(condition.get("passed"))
        ],
    }


def _representative_cases(cases: Sequence[TargetOnlyCase]) -> tuple[TargetOnlyCase, ...]:
    selected: dict[tuple[str, int], TargetOnlyCase] = {}
    for case in sorted(cases, key=lambda item: (item.product, item.pred_len, item.case_key)):
        selected.setdefault((case.product, case.pred_len), case)
    return tuple(selected[key] for key in sorted(selected))


def build_p0_audit(
    cohort: TargetOnlyObservedCohort,
    config: Mapping[str, Any],
) -> P0AuditBundle:
    """Construct all P0 evidence without loading Kronos or touching CUDA."""

    data = config["data"]
    walk = config["walk_forward"]
    labels = config["risk_labels"]
    primary_products = tuple(str(value) for value in data["primary_products"])
    transfer_products = tuple(str(value) for value in data["transfer_products"])
    products = tuple(str(value) for value in data["products"])
    spec = RiskLabelSpec.from_config(config)
    case_bundle = build_target_only_cases(
        cohort,
        lookback=int(data["lookback"]),
        products=products,
    )
    case_bundle = _without_heavy_frame_attrs(case_bundle)
    if not case_bundle.target_cases:
        raise P0AuditError("V6 P0 has no target-only cases")
    global_days = sorted({case.target_end_day for case in case_bundle.target_cases})
    folds = build_expanding_walk_forward_folds(
        global_days,
        minimum_fit_days=int(walk["minimum_fit_days"]),
        inner_validation_days=int(walk["inner_validation_days"]),
        evaluation_days=int(walk["evaluation_days"]),
        step_days=int(walk["step_days"]),
        purge_days=int(walk["purge_days"]),
    )
    expected_fold_ids = tuple(f"fold_{index:02d}" for index in range(int(walk["fold_count"])))
    protocol_checks: list[dict[str, object]] = [
        {
            "check_id": "exact_pre_registered_fold_count",
            "passed": len(folds) == int(walk["fold_count"]),
            "observed": int(len(folds)),
            "required": int(walk["fold_count"]),
        },
        {
            "check_id": "exact_pre_registered_fold_ids",
            "passed": tuple(fold.fold_id for fold in folds) == expected_fold_ids,
            "observed": [fold.fold_id for fold in folds],
            "required": list(expected_fold_ids),
        },
    ]

    outcome_rows: list[dict[str, object]] = []
    integrity_failures: list[dict[str, object]] = []
    for case in case_bundle.target_cases:
        try:
            outcome_rows.append(risk_outcome(case, spec))
        except RiskLabelError as exc:
            integrity_failures.append(
                {
                    "case_key": case.case_key,
                    "product": case.product,
                    "target_contract_id": case.target_contract_id,
                    "issues": [str(exc)],
                }
            )
    if not outcome_rows:
        raise P0AuditError("V6 P0 could not derive any finite risk outcomes")
    outcomes = pd.DataFrame(outcome_rows).sort_values(
        ["target_end_day", "target_contract_id", "origin_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if outcomes["case_key"].duplicated().any():
        raise P0AuditError("V6 P0 produced duplicate continuous-outcome case keys")

    leakage_checks: list[dict[str, object]] = []
    for case in _representative_cases(case_bundle.target_cases):
        leakage_checks.extend(future_mutation_context_leakage_checks(case, spec))

    fold_rows: list[pd.DataFrame] = []
    fold_audits: list[dict[str, object]] = []
    thresholds_by_fold: list[dict[str, object]] = []
    for fold in folds:
        split_cases = {
            "fit": cases_in_period(
                case_bundle.target_cases,
                start_day=fold.fit_start_day,
                end_day=fold.fit_end_day,
            ),
            "inner_validation": cases_in_period(
                case_bundle.target_cases,
                start_day=fold.inner_validation_start_day,
                end_day=fold.inner_validation_end_day,
            ),
            "evaluation": cases_in_period(
                case_bundle.target_cases,
                start_day=fold.evaluation_start_day,
                end_day=fold.evaluation_end_day,
            ),
        }
        raw_records = {
            split: _records_for_cases(outcomes, cases)
            for split, cases in split_cases.items()
        }
        primary_fit = raw_records["fit"].loc[
            raw_records["fit"]["product"].astype(str).isin(primary_products)
        ].copy()
        primary_fit["fold_id"] = fold.fold_id
        primary_fit["split"] = "fit"
        try:
            thresholds = fit_tail_thresholds(
                primary_fit,
                fold_id=fold.fold_id,
                fit_start_day=fold.fit_start_day,
                fit_end_day=fold.fit_end_day,
                primary_products=primary_products,
                quantile=float(labels["tail_event"]["quantile"]),
                quantile_method=str(labels["tail_event"]["quantile_method"]),
            )
        except RiskLabelError as exc:
            raise P0AuditError(f"Cannot fit {fold.fold_id} V6 thresholds: {exc}") from exc
        labeled = {
            split: apply_tail_thresholds(records, thresholds, split=split)
            for split, records in raw_records.items()
        }
        for records in labeled.values():
            fold_rows.append(records)
        leakage_checks.extend(
            _fold_leakage_checks(
                fold=fold,
                fit=labeled["fit"],
                validation=labeled["inner_validation"],
                evaluation=labeled["evaluation"],
                thresholds=thresholds,
                primary_fit=primary_fit,
                global_days=global_days,
            )
        )
        split_summaries: dict[str, object] = {}
        for split, records in labeled.items():
            primary_records = records.loc[
                records["product"].astype(str).isin(primary_products)
            ]
            transfer_records = records.loc[
                records["product"].astype(str).isin(transfer_products)
            ]
            split_summaries[split] = {
                "start_day": (
                    fold.fit_start_day
                    if split == "fit"
                    else fold.inner_validation_start_day
                    if split == "inner_validation"
                    else fold.evaluation_start_day
                ),
                "end_day": (
                    fold.fit_end_day
                    if split == "fit"
                    else fold.inner_validation_end_day
                    if split == "inner_validation"
                    else fold.evaluation_end_day
                ),
                "all_products": _event_support(records),
                "primary_products": _event_support(primary_records),
                "transfer_products": _event_support(transfer_records),
                "counts_by_product": _counts_by(records, ("product",)),
                "counts_by_product_contract": _counts_by(
                    records, ("product", "target_contract_id")
                ),
                "primary_event_support_by_product": [
                    {
                        "product": product,
                        **_event_support(
                            primary_records.loc[
                                primary_records["product"].astype(str) == product
                            ]
                        ),
                    }
                    for product in primary_products
                ],
            }
        thresholds_by_fold.append(thresholds.as_dict())
        fold_audits.append(
            {
                "fold_id": fold.fold_id,
                "purge_days": int(fold.purge_days),
                "thresholds": thresholds.as_dict(),
                "splits": split_summaries,
            }
        )
    fold_records = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    gate = evaluate_p0_gate(
        fold_records,
        expected_fold_ids=expected_fold_ids,
        primary_products=primary_products,
        p0_config=config["p0"],
        integrity_failures=integrity_failures,
        leakage_checks=leakage_checks,
        protocol_checks=protocol_checks,
    )

    by_product_distribution = {
        product: _outcome_distributions(
            outcomes.loc[outcomes["product"].astype(str) == product]
        )
        for product in products
    }
    by_contract_clipping = [
        {
            "product": product,
            "target_contract_id": contract,
            **_distribution(group["context_clip_fraction"].to_numpy(dtype=np.float64)),
        }
        for (product, contract), group in outcomes.groupby(
            ["product", "target_contract_id"], sort=True
        )
    ]
    rejection_records = [dict(record) for record in case_bundle.target_rejection_records]
    audit: dict[str, object] = {
        "risk_label_rule": {
            "version": spec.version,
            "context_volatility": {
                "method": "ewma_close_log_return",
                "return_count": spec.volatility_bars,
                "halflife_bars": spec.volatility_halflife_bars,
                "adjust": spec.volatility_adjust,
                "bias": spec.volatility_bias,
                "floor": spec.volatility_floor,
                "horizon_scaling": "sqrt_target_bar_count",
            },
            "tail_quantile": spec.tail_quantile,
            "tail_quantile_method": spec.tail_quantile_method,
            "tail_comparison": "greater_than_or_equal",
            "threshold_source": "outer_fit_primary_cases_only",
        },
        "cohort": {
            "manifest_path": cohort.manifest_path,
            "manifest_sha256": cohort.manifest_sha256,
            "snapshot_id": cohort.snapshot_id,
            "snapshot_at": cohort.snapshot_at,
            "payload_sha256": dict(sorted(cohort.payload_sha256.items())),
            "contract_count": int(len(cohort.contracts)),
            "contracts": {
                contract_id: {
                    "product": contract.product,
                    "bars": int(len(contract.frame)),
                    "first_available_bar": contract.first_timestamp,
                    "last_available_bar": contract.last_timestamp,
                    **dict(cohort.contract_status[contract_id]),
                }
                for contract_id, contract in sorted(cohort.contracts.items())
            },
            "failed_contracts": {
                contract_id: dict(status)
                for contract_id, status in sorted(cohort.contract_status.items())
                if status.get("status") != "ok"
            },
        },
        "case_universe": {
            "lookback": int(case_bundle.lookback),
            "global_target_end_days": int(len(global_days)),
            "included_cases": int(len(case_bundle.target_cases)),
            "finite_labeled_cases": int(len(outcomes)),
            "case_keys_sha256": _case_keys_sha256(
                [case.case_key for case in case_bundle.target_cases]
            ),
            "by_product": dict(
                sorted(Counter(case.product for case in case_bundle.target_cases).items())
            ),
            "by_contract": dict(
                sorted(
                    Counter(
                        case.target_contract_id for case in case_bundle.target_cases
                    ).items()
                )
            ),
            "by_target_bar_count": dict(
                sorted(
                    (str(length), int(count))
                    for length, count in Counter(
                        case.pred_len for case in case_bundle.target_cases
                    ).items()
                )
            ),
            "rejections_by_reason": dict(case_bundle.target_rejections),
            "rejection_records": rejection_records,
        },
        "integrity": {
            "failure_count": int(len(integrity_failures)),
            "failures": integrity_failures,
        },
        "past_only_leakage_audit": {
            "check_count": int(len(leakage_checks)),
            "failure_count": int(
                sum(not bool(check.get("passed")) for check in leakage_checks)
            ),
            "checks": leakage_checks,
        },
        "protocol_checks": protocol_checks,
        "continuous_outcome_distributions": {
            "all_products": _outcome_distributions(outcomes),
            "by_product": by_product_distribution,
        },
        "context_clipping_by_contract": by_contract_clipping,
        "fit_only_thresholds": thresholds_by_fold,
        "folds": fold_audits,
        "p0_gate_summary": gate,
    }
    return P0AuditBundle(
        audit=audit,
        gate=gate,
        outcomes=outcomes,
        fold_records=fold_records,
        folds=folds,
        case_bundle=case_bundle,
    )


__all__ = [
    "P0AuditBundle",
    "P0AuditError",
    "build_p0_audit",
    "evaluate_p0_gate",
    "prediction_day_atomicity_check",
]
