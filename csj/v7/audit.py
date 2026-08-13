"""Prediction-origin-day atomic P0 audit for diversified V7 risk data."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from csj.v3.panel_data import V3WalkForwardFold
from csj.v5.target_data import (
    TargetOnlyCase,
    TargetOnlyCaseBundle,
    TargetOnlyObservedCohort,
    build_target_only_cases,
)
from csj.v6.audit import P0AuditError, evaluate_p0_gate, prediction_day_atomicity_check
from csj.v6.risk_labels import (
    RiskLabelError,
    RiskLabelSpec,
    TailThresholds,
    apply_tail_thresholds,
    fit_tail_thresholds,
    future_mutation_context_leakage_checks,
    risk_outcome,
)


@dataclass(frozen=True)
class V7AuditBundle:
    audit: Mapping[str, object]
    gate: Mapping[str, object]
    outcomes: pd.DataFrame
    fold_records: pd.DataFrame
    folds: tuple[V3WalkForwardFold, ...]
    case_bundle: TargetOnlyCaseBundle


def _sha256_case_keys(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strip_frame_attrs(bundle: TargetOnlyCaseBundle) -> TargetOnlyCaseBundle:
    contracts = {}
    for case in bundle.target_cases:
        if case.target_contract_id not in contracts:
            frame = case.target_contract.frame.copy(deep=False)
            frame.attrs = {}
            contracts[case.target_contract_id] = replace(case.target_contract, frame=frame)
    return replace(
        bundle,
        target_cases=tuple(
            replace(case, target_contract=contracts[case.target_contract_id])
            for case in bundle.target_cases
        ),
    )


def _risk_spec(config: Mapping[str, Any]) -> RiskLabelSpec:
    data = config["data"]
    labels = config["risk_labels"]
    return RiskLabelSpec(
        lookback=int(data["lookback"]),
        horizon_trading_days=int(data["horizon_trading_days"]),
        valid_bar_counts=tuple(int(value) for value in data["valid_bar_counts"]),
        volatility_bars=int(labels["volatility_bars"]),
        volatility_halflife_bars=int(labels["halflife_bars"]),
        volatility_adjust=False,
        volatility_bias=False,
        volatility_floor=1e-5,
        tail_quantile=float(labels["tail_quantile"]),
        tail_quantile_method=str(labels["quantile_method"]),
        clip=float(data["clip"]),
        epsilon=float(data["normalization_epsilon"]),
    )


def filter_coverage_products(
    cohort: TargetOnlyObservedCohort,
    *,
    declared_products: Sequence[str],
    minimum_bars: int,
    latest_first_bar: str,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """Apply only pre-label availability rules to the declared product pool."""

    cutoff = pd.Timestamp(latest_first_bar).normalize()
    selected: list[str] = []
    records: list[dict[str, object]] = []
    for product in declared_products:
        contracts = [item for item in cohort.contracts.values() if item.product == product]
        eligible = [
            item
            for item in contracts
            if len(item.frame) >= minimum_bars and item.first_timestamp.normalize() <= cutoff
        ]
        passed = bool(eligible)
        if passed:
            selected.append(product)
        records.append(
            {
                "product": product,
                "passed": passed,
                "minimum_bars": int(minimum_bars),
                "latest_first_bar": cutoff,
                "eligible_contracts": [item.contract_id for item in eligible],
                "observed_contracts": [
                    {
                        "contract_id": item.contract_id,
                        "bars": int(len(item.frame)),
                        "first_bar": item.first_timestamp,
                        "last_bar": item.last_timestamp,
                    }
                    for item in contracts
                ],
            }
        )
    if tuple(selected) != tuple(declared_products):
        rejected = [record["product"] for record in records if not record["passed"]]
        raise P0AuditError(f"V7 declared products fail the frozen coverage filter: {rejected}")
    return tuple(selected), records


def build_origin_day_folds(
    origin_days: Sequence[pd.Timestamp],
    *,
    minimum_fit_days: int,
    inner_validation_days: int,
    evaluation_days: int,
    step_days: int,
    purge_days: int,
    fold_count: int,
) -> tuple[V3WalkForwardFold, ...]:
    """Build exactly ``fold_count`` atomic folds on forecast-origin days."""

    if purge_days < 3:
        raise ValueError("V7 requires at least three purged origin days")
    days = sorted({pd.Timestamp(day).normalize() for day in origin_days})
    folds: list[V3WalkForwardFold] = []
    for index in range(fold_count):
        fit_end_index = minimum_fit_days + index * step_days - 1
        validation_start_index = fit_end_index + purge_days + 1
        validation_end_index = validation_start_index + inner_validation_days - 1
        evaluation_start_index = validation_end_index + purge_days + 1
        evaluation_end_index = evaluation_start_index + evaluation_days - 1
        if evaluation_end_index >= len(days):
            break
        folds.append(
            V3WalkForwardFold(
                fold_id=f"fold_{index:02d}",
                fit_start_day=days[0],
                fit_end_day=days[fit_end_index],
                inner_validation_start_day=days[validation_start_index],
                inner_validation_end_day=days[validation_end_index],
                evaluation_start_day=days[evaluation_start_index],
                evaluation_end_day=days[evaluation_end_index],
                purge_days=purge_days,
            )
        )
    return tuple(folds)


def _cases_by_origin(
    cases: Sequence[TargetOnlyCase], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[TargetOnlyCase, ...]:
    return tuple(case for case in cases if start <= case.origin_trading_day <= end)


def _records(outcomes: pd.DataFrame, cases: Sequence[TargetOnlyCase]) -> pd.DataFrame:
    keys = {case.case_key for case in cases}
    return outcomes.loc[outcomes["case_key"].astype(str).isin(keys)].copy()


def _support(records: pd.DataFrame) -> dict[str, object]:
    return {
        "cases": int(len(records)),
        "long_events": int(records["long_tail_event"].astype(bool).sum()),
        "short_events": int(records["short_tail_event"].astype(bool).sum()),
    }


def _split_checks(
    fold: V3WalkForwardFold,
    labeled: Mapping[str, pd.DataFrame],
    thresholds: TailThresholds,
    fit_records: pd.DataFrame,
    origin_days: Sequence[pd.Timestamp],
) -> list[dict[str, object]]:
    combined = pd.concat(tuple(labeled.values()), ignore_index=True)
    checks = [prediction_day_atomicity_check(combined, fold_id=fold.fold_id)]
    keys = {name: set(frame["case_key"].astype(str)) for name, frame in labeled.items()}
    checks.append(
        {
            "check_id": f"{fold.fold_id}:split_case_key_disjoint",
            "passed": not (
                keys["fit"] & keys["inner_validation"]
                or keys["fit"] & keys["evaluation"]
                or keys["inner_validation"] & keys["evaluation"]
            ),
        }
    )
    boundary_details = {}
    boundary_ok = True
    for earlier, later in (("fit", "inner_validation"), ("inner_validation", "evaluation")):
        latest_label = pd.to_datetime(labeled[earlier]["target_end_day"]).max().normalize()
        earliest_target = pd.to_datetime(labeled[later]["target_start_day"]).min().normalize()
        passed = bool(latest_label < earliest_target)
        boundary_ok &= passed
        boundary_details[f"{earlier}_to_{later}"] = {
            "earlier_latest_label_day": latest_label,
            "later_earliest_target_day": earliest_target,
            "passed": passed,
        }
    checks.append(
        {
            "check_id": f"{fold.fold_id}:three_day_labels_do_not_cross_splits",
            "passed": boundary_ok,
            "boundaries": boundary_details,
        }
    )
    normalized = [pd.Timestamp(day).normalize() for day in origin_days]
    checks.append(
        {
            "check_id": f"{fold.fold_id}:origin_calendar_purge",
            "passed": sum(fold.fit_end_day < day < fold.inner_validation_start_day for day in normalized) >= fold.purge_days
            and sum(fold.inner_validation_end_day < day < fold.evaluation_start_day for day in normalized) >= fold.purge_days,
        }
    )
    expected_hash = _sha256_case_keys(fit_records["case_key"].astype(str).tolist())
    checks.append(
        {
            "check_id": f"{fold.fold_id}:threshold_fit_only",
            "passed": thresholds.fit_case_keys_sha256 == expected_hash,
        }
    )
    return checks


def build_v7_p0_audit(
    cohort: TargetOnlyObservedCohort, config: Mapping[str, Any]
) -> V7AuditBundle:
    data = config["data"]
    coverage = data["coverage_filter"]
    products, coverage_records = filter_coverage_products(
        cohort,
        declared_products=tuple(str(value) for value in data["products"]),
        minimum_bars=int(coverage["minimum_bars"]),
        latest_first_bar=str(coverage["latest_first_bar"]),
    )
    bundle = _strip_frame_attrs(
        build_target_only_cases(cohort, lookback=int(data["lookback"]), products=products)
    )
    study_start_day = pd.Timestamp(data["study_start_day"]).normalize()
    bundle = replace(
        bundle,
        target_cases=tuple(
            case for case in bundle.target_cases if case.origin_trading_day >= study_start_day
        ),
    )
    products_without_cases = sorted(
        set(products).difference(case.product for case in bundle.target_cases)
    )
    if products_without_cases:
        raise P0AuditError(
            "V7 products produce no structurally eligible cases: "
            + ", ".join(products_without_cases)
        )
    spec = _risk_spec(config)
    rows: list[dict[str, object]] = []
    integrity_failures: list[dict[str, object]] = []
    for case in bundle.target_cases:
        try:
            rows.append(risk_outcome(case, spec))
        except RiskLabelError as exc:
            integrity_failures.append({"case_key": case.case_key, "error": str(exc)})
    outcomes = pd.DataFrame(rows).sort_values(
        ["origin_trading_day", "target_contract_id", "origin_timestamp"], kind="stable"
    ).reset_index(drop=True)
    origin_days = sorted({case.origin_trading_day for case in bundle.target_cases})
    walk = config["walk_forward"]
    folds = build_origin_day_folds(
        origin_days,
        minimum_fit_days=int(walk["minimum_fit_days"]),
        inner_validation_days=int(walk["inner_validation_days"]),
        evaluation_days=int(walk["evaluation_days"]),
        step_days=int(walk["step_days"]),
        purge_days=int(walk["purge_days"]),
        fold_count=int(walk["fold_count"]),
    )
    expected_fold_ids = tuple(f"fold_{index:02d}" for index in range(int(walk["fold_count"])))
    protocol_checks = [
        {
            "check_id": "exact_pre_registered_fold_count",
            "passed": len(folds) == int(walk["fold_count"]),
            "observed": len(folds),
            "required": int(walk["fold_count"]),
        }
    ]
    leakage_checks: list[dict[str, object]] = []
    representatives: dict[tuple[str, int], TargetOnlyCase] = {}
    for case in bundle.target_cases:
        representatives.setdefault((case.product, case.pred_len), case)
    for case in representatives.values():
        leakage_checks.extend(future_mutation_context_leakage_checks(case, spec))

    fold_frames: list[pd.DataFrame] = []
    fold_audits: list[dict[str, object]] = []
    threshold_audits: list[dict[str, object]] = []
    for fold in folds:
        split_cases = {
            "fit": _cases_by_origin(bundle.target_cases, fold.fit_start_day, fold.fit_end_day),
            "inner_validation": _cases_by_origin(
                bundle.target_cases,
                fold.inner_validation_start_day,
                fold.inner_validation_end_day,
            ),
            "evaluation": _cases_by_origin(
                bundle.target_cases, fold.evaluation_start_day, fold.evaluation_end_day
            ),
        }
        raw = {name: _records(outcomes, cases) for name, cases in split_cases.items()}
        fit = raw["fit"].copy()
        fit["fold_id"] = fold.fold_id
        fit["split"] = "fit"
        thresholds = fit_tail_thresholds(
            fit,
            fold_id=fold.fold_id,
            fit_start_day=pd.to_datetime(fit["target_end_day"]).min(),
            fit_end_day=pd.to_datetime(fit["target_end_day"]).max(),
            primary_products=tuple(sorted(fit["product"].astype(str).unique())),
            quantile=float(config["risk_labels"]["tail_quantile"]),
            quantile_method=str(config["risk_labels"]["quantile_method"]),
        )
        labeled = {
            name: apply_tail_thresholds(frame, thresholds, split=name)
            for name, frame in raw.items()
        }
        fold_frames.extend(labeled.values())
        leakage_checks.extend(
            _split_checks(fold, labeled, thresholds, fit, origin_days)
        )
        fold_audits.append(
            {
                "fold_id": fold.fold_id,
                "splits": {name: _support(frame) for name, frame in labeled.items()},
                "boundaries": {
                    "fit": [fold.fit_start_day, fold.fit_end_day],
                    "inner_validation": [
                        fold.inner_validation_start_day,
                        fold.inner_validation_end_day,
                    ],
                    "evaluation": [fold.evaluation_start_day, fold.evaluation_end_day],
                },
            }
        )
        threshold_audits.append(thresholds.as_dict())
    fold_records = pd.concat(fold_frames, ignore_index=True)
    gate_config = dict(config["p0"])
    # Fold-level support is pooled over the frozen diversified training pool.
    # The stricter per-product pooled-evaluation check remains on i/jm/rb.
    gate = evaluate_p0_gate(
        fold_records,
        expected_fold_ids=expected_fold_ids,
        primary_products=products,
        pooled_evaluation_products=tuple(
            str(value) for value in data["gate_products"]
        ),
        p0_config=gate_config,
        integrity_failures=integrity_failures,
        leakage_checks=leakage_checks,
        protocol_checks=protocol_checks,
    )
    gate = {
        **gate,
        "gate_name": "v7_p0_diversified_risk_label_support_and_leakage",
    }
    audit = {
        "strategy_version": 7,
        "result_scope": config["experiment"]["result_scope"],
        "production_eligible": False,
        "snapshot_id": cohort.snapshot_id,
        "data_fingerprint": cohort.data_fingerprint,
        "coverage_filter": {
            "selection_uses_labels": False,
            "selected_products": list(products),
            "records": coverage_records,
        },
        "case_universe": {
            "included_cases": int(len(bundle.target_cases)),
            "finite_labeled_cases": int(len(outcomes)),
            "origin_days": int(len(origin_days)),
            "first_origin_day": origin_days[0],
            "last_origin_day": origin_days[-1],
            "study_start_day": study_start_day,
            "by_product": dict(sorted(Counter(case.product for case in bundle.target_cases).items())),
        },
        "integrity_failures": integrity_failures,
        "leakage_checks": leakage_checks,
        "protocol_checks": protocol_checks,
        "fit_only_thresholds": threshold_audits,
        "folds": fold_audits,
        "p0_gate_summary": gate,
    }
    return V7AuditBundle(audit, gate, outcomes, fold_records, folds, bundle)


__all__ = [
    "V7AuditBundle",
    "build_origin_day_folds",
    "build_v7_p0_audit",
    "filter_coverage_products",
]
