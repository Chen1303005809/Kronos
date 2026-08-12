"""Pure target-contract data construction for V5.

The V5 corpus deliberately has no neighbour fields, selection state, or
term-structure features.  Every loaded concrete contract is merely another
possible *target* series; a case is built from one contract's own context and
three following complete trading days.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from csj.v3.panel_data import (
    VALID_TRADING_DAY_BAR_COUNTS,
    ConcreteContract,
    V3WalkForwardFold,
    build_expanding_walk_forward_folds,
    load_panel_archive,
)
from csj.v5.config import PRODUCTION_ELIGIBLE, RESULT_SCOPE


DATA_RULE_VERSION = "target-only-observed-contract-data-v1"


class TargetOnlyDataError(RuntimeError):
    """A target-only V5 corpus fails a no-leakage invariant."""


@dataclass(frozen=True)
class TargetOnlyObservedCohort:
    """Frozen observed concrete contracts, without any pair-selection state."""

    manifest_path: Path
    manifest_sha256: str
    snapshot_id: str
    snapshot_at: pd.Timestamp
    contracts: Mapping[str, ConcreteContract]
    contract_status: Mapping[str, Mapping[str, object]]
    payload_sha256: Mapping[str, str]
    data_fingerprint: str


@dataclass(frozen=True)
class TargetOnlyCase:
    """One contract-local context and its next three complete trading days."""

    target_contract_id: str
    product: str
    origin_timestamp: pd.Timestamp
    origin_trading_day: pd.Timestamp
    target_contract: ConcreteContract
    target_context_start: int
    target_start: int
    target_end_exclusive: int
    target_days: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]
    day_end_indices: tuple[int, int, int]
    data_fingerprint: str

    @property
    def target_context(self) -> DataFrame:
        return self.target_contract.frame.iloc[
            self.target_context_start : self.target_start
        ].copy()

    @property
    def target(self) -> DataFrame:
        return self.target_contract.frame.iloc[
            self.target_start : self.target_end_exclusive
        ].copy()

    @property
    def case_key(self) -> str:
        return f"{self.target_contract_id}|{self.origin_timestamp.isoformat()}"

    @property
    def target_end_day(self) -> pd.Timestamp:
        return self.target_days[-1]

    @property
    def pred_len(self) -> int:
        return self.target_end_exclusive - self.target_start

    @property
    def day3_return(self) -> float:
        origin_close = float(self.target_contract.frame["close"].iloc[self.target_start - 1])
        day3_close = float(
            self.target_contract.frame["close"].iloc[
                self.target_start + self.day_end_indices[-1]
            ]
        )
        return day3_close / origin_close - 1.0


@dataclass(frozen=True)
class TargetOnlyCaseBundle:
    """The complete V5 target-case universe for one fixed context length."""

    lookback: int
    target_cases: tuple[TargetOnlyCase, ...]
    target_rejections: Mapping[str, int]
    target_rejection_records: tuple[Mapping[str, object], ...]


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetOnlyDataError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TargetOnlyDataError(f"Expected a JSON object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_naive_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp


def _resolve_current_manifest(root: str | Path) -> Path:
    candidate = Path(root)
    if (candidate / "manifest.json").is_file():
        return candidate
    if not candidate.is_dir():
        raise TargetOnlyDataError(f"Snapshot root does not exist: {candidate}")
    directories = sorted(path for path in candidate.iterdir() if (path / "manifest.json").is_file())
    if not directories:
        raise TargetOnlyDataError(f"No snapshot manifest exists below: {candidate}")
    ranked: list[tuple[pd.Timestamp, str, Path]] = []
    for directory in directories:
        manifest = _read_json(directory / "manifest.json")
        try:
            snapshot_at = _local_naive_timestamp(manifest["snapshot_at"])
        except KeyError as exc:
            raise TargetOnlyDataError(f"Manifest is missing snapshot_at: {directory}") from exc
        ranked.append((snapshot_at, directory.name, directory))
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def _safe_payload_path(snapshot_dir: Path, value: object) -> Path:
    if not value:
        raise TargetOnlyDataError(f"Successful contract has no raw payload path: {snapshot_dir}")
    root = snapshot_dir.resolve()
    payload = (snapshot_dir / str(value)).resolve()
    try:
        payload.relative_to(root)
    except ValueError as exc:
        raise TargetOnlyDataError(f"Raw payload path escapes snapshot: {value!r}") from exc
    if not payload.is_file():
        raise TargetOnlyDataError(f"Raw payload is missing: {payload}")
    return payload


def load_target_only_observed_cohort(snapshot_root: str | Path) -> TargetOnlyObservedCohort:
    """Freeze one observed manifest without invoking any pair-data loader.

    V5's source loader validates the same raw manifest/payload invariants as
    V4, but only materializes target contracts.  It neither creates nor asks
    for neighbour eligibility, context alignment, or term-structure state.
    """

    snapshot_dir = _resolve_current_manifest(snapshot_root)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    records = manifest.get("contracts")
    if not isinstance(records, list) or not records:
        raise TargetOnlyDataError(f"Manifest contains no contracts: {manifest_path}")
    try:
        snapshot_at = _local_naive_timestamp(manifest["snapshot_at"])
    except KeyError as exc:
        raise TargetOnlyDataError(f"Manifest is missing snapshot_at: {manifest_path}") from exc
    statuses: dict[str, dict[str, object]] = {}
    payload_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TargetOnlyDataError(f"Invalid contract record in {manifest_path}")
        contract_id = str(record.get("contract_id", "")).lower().strip()
        product = str(record.get("product", "")).lower().strip()
        if not contract_id or not product or contract_id in statuses:
            raise TargetOnlyDataError(f"Invalid or duplicate contract in {manifest_path}")
        status = str(record.get("status", "failed")).lower()
        statuses[contract_id] = {
            "product": product,
            "status": status,
            "error": record.get("error"),
            "error_type": record.get("error_type"),
            "raw_payload_path": record.get("raw_payload_path"),
            "declared_raw_payload_sha256": record.get("raw_payload_sha256"),
        }
        if status != "ok":
            continue
        payload_path = _safe_payload_path(snapshot_dir, record.get("raw_payload_path"))
        observed_hash = _sha256_path(payload_path)
        declared_hash = record.get("raw_payload_sha256")
        if declared_hash and observed_hash != str(declared_hash):
            raise TargetOnlyDataError(
                f"Raw payload checksum mismatch for {contract_id}: {payload_path}"
            )
        payload_hashes[contract_id] = observed_hash
    archive = load_panel_archive(snapshot_dir)
    contracts = {
        contract_id: contract
        for contract_id, contract in archive.contracts.items()
        if statuses.get(contract_id, {}).get("status") == "ok"
    }
    expected_successes = {key for key, item in statuses.items() if item["status"] == "ok"}
    if set(contracts) != expected_successes:
        missing = sorted(expected_successes.difference(contracts))
        raise TargetOnlyDataError(
            f"Manifest successful contracts did not all load: {missing[:5]!r}"
        )
    if not contracts:
        raise TargetOnlyDataError("Target-only cohort has no successful contracts")
    manifest_sha256 = _sha256_path(manifest_path)
    fingerprint = _sha256_json(
        {
            "data_rule_version": DATA_RULE_VERSION,
            "manifest_sha256": manifest_sha256,
            "snapshot_id": snapshot_dir.name,
            "contracts": [
                {
                    "contract_id": contract_id,
                    "payload_sha256": payload_hashes[contract_id],
                }
                for contract_id in sorted(contracts)
            ],
        }
    )
    return TargetOnlyObservedCohort(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        snapshot_id=snapshot_dir.name,
        snapshot_at=snapshot_at,
        contracts=contracts,
        contract_status=statuses,
        payload_sha256=payload_hashes,
        data_fingerprint=fingerprint,
    )


def _target_day_groups(frame: DataFrame) -> list[tuple[pd.Timestamp, np.ndarray]]:
    return [
        (pd.Timestamp(day).normalize(), group.index.to_numpy(dtype=np.int64))
        for day, group in frame.groupby("trading_day", sort=False)
    ]


def build_target_only_cases(
    cohort: TargetOnlyObservedCohort,
    *,
    lookback: int,
    products: Iterable[str] | None = None,
) -> TargetOnlyCaseBundle:
    """Build target-only V5 cases without inspecting any other contract.

    A case's inclusion, context, and label depend solely on the target contract
    that owns it.  This is intentionally independent of any other contract
    becoming unavailable or changing its timestamp alignment.
    """

    if lookback < 1:
        raise ValueError("V5 lookback must be positive")
    selected_products = (
        {str(product).lower() for product in products}
        if products is not None
        else {contract.product for contract in cohort.contracts.values()}
    )
    known_products = {contract.product for contract in cohort.contracts.values()}
    unknown = selected_products.difference(known_products)
    if unknown:
        raise ValueError(f"No observed target contracts for products: {sorted(unknown)!r}")

    cases: list[TargetOnlyCase] = []
    rejections: Counter[str] = Counter()
    rejection_records: list[Mapping[str, object]] = []
    for target_contract in sorted(
        cohort.contracts.values(), key=lambda item: (item.product, item.contract_id)
    ):
        if target_contract.product not in selected_products:
            continue
        frame = target_contract.frame
        day_groups = _target_day_groups(frame)
        for offset in range(max(len(day_groups) - 2, 0)):
            selected_days = day_groups[offset : offset + 3]
            target_days = tuple(day for day, _ in selected_days)
            bar_counts = tuple(len(indices) for _, indices in selected_days)
            if any(count not in VALID_TRADING_DAY_BAR_COUNTS for count in bar_counts):
                reason = "target_requires_three_complete_days"
                rejections[reason] += 1
                rejection_records.append(
                    {
                        "target_contract_id": target_contract.contract_id,
                        "product": target_contract.product,
                        "target_days": [pd.Timestamp(day).normalize() for day in target_days],
                        "reason": reason,
                    }
                )
                continue
            target_indices = np.concatenate([indices for _, indices in selected_days])
            target_start = int(target_indices[0])
            if target_start < lookback:
                reason = "target_lacks_context"
                rejections[reason] += 1
                rejection_records.append(
                    {
                        "target_contract_id": target_contract.contract_id,
                        "product": target_contract.product,
                        "target_days": [pd.Timestamp(day).normalize() for day in target_days],
                        "reason": reason,
                    }
                )
                continue
            target_context_start = target_start - lookback
            origin_timestamp = pd.Timestamp(frame["timestamps"].iloc[target_start - 1])
            origin_trading_day = pd.Timestamp(
                frame["trading_day"].iloc[target_start - 1]
            ).normalize()
            day_end_indices = tuple(int(value) for value in np.cumsum(bar_counts) - 1)
            cases.append(
                TargetOnlyCase(
                    target_contract_id=target_contract.contract_id,
                    product=target_contract.product,
                    origin_timestamp=origin_timestamp,
                    origin_trading_day=origin_trading_day,
                    target_contract=target_contract,
                    target_context_start=target_context_start,
                    target_start=target_start,
                    target_end_exclusive=int(target_indices[-1]) + 1,
                    target_days=(
                        pd.Timestamp(target_days[0]).normalize(),
                        pd.Timestamp(target_days[1]).normalize(),
                        pd.Timestamp(target_days[2]).normalize(),
                    ),
                    day_end_indices=(
                        int(day_end_indices[0]),
                        int(day_end_indices[1]),
                        int(day_end_indices[2]),
                    ),
                    data_fingerprint=cohort.data_fingerprint,
                )
            )
    cases.sort(key=lambda case: (case.target_end_day, case.target_contract_id, case.case_key))
    return TargetOnlyCaseBundle(
        lookback=int(lookback),
        target_cases=tuple(cases),
        target_rejections=dict(sorted(rejections.items())),
        target_rejection_records=tuple(rejection_records),
    )


def cases_in_period(
    cases: Sequence[TargetOnlyCase],
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    products: Sequence[str] | None = None,
) -> tuple[TargetOnlyCase, ...]:
    """Filter a case sequence by its target-completion day and product set."""

    selected_products = None if products is None else {str(product) for product in products}
    return tuple(
        case
        for case in cases
        if start_day <= case.target_end_day <= end_day
        and (selected_products is None or case.product in selected_products)
    )


def _audit_v4_pair_only_coverage(
    cohort: TargetOnlyObservedCohort,
    *,
    target_cases: Sequence[TargetOnlyCase],
    lookback: int,
    products: Sequence[str],
) -> tuple[set[str], Mapping[str, str]]:
    """Return V4 pair-only keys for audit reporting, never as V5 inputs."""

    from csj.v4.cohort_data import (
        build_observed_cohort_cases,
        load_observed_contract_cohort,
    )

    v4_cohort = load_observed_contract_cohort(cohort.manifest_path.parent)
    v4_bundle = build_observed_cohort_cases(v4_cohort, lookback=lookback, products=products)
    target_keys = {case.case_key for case in target_cases}
    v4_target_keys = {case.case_key for case in v4_bundle.target_cases}
    if target_keys != v4_target_keys:
        raise TargetOnlyDataError(
            "V5 target-only construction no longer matches V4's target eligibility; "
            "a hidden non-target criterion may have been introduced"
        )
    return (
        {case.case_key for case in v4_bundle.pair_cases},
        {
            case.case_key: str(case.pair_rejection_reason or "unknown_pair_rejection")
            for case in v4_bundle.target_cases
        },
    )


def _direction_counts(cases: Sequence[TargetOnlyCase]) -> dict[str, int]:
    values = np.asarray([case.day3_return for case in cases], dtype=np.float64)
    return {
        "up": int(np.sum(values > 0.0)),
        "down": int(np.sum(values < 0.0)),
        "zero": int(np.sum(values == 0.0)),
    }


def _case_summary(cases: Sequence[TargetOnlyCase]) -> dict[str, object]:
    return {
        "cases": int(len(cases)),
        "case_keys": [case.case_key for case in cases],
        "case_keys_sha256": _sha256_json([case.case_key for case in cases]),
        "by_product": dict(sorted(Counter(case.product for case in cases).items())),
        "by_contract": dict(sorted(Counter(case.target_contract_id for case in cases).items())),
        "by_prediction_length": dict(
            sorted((str(length), int(count)) for length, count in Counter(case.pred_len for case in cases).items())
        ),
        "day3_direction": _direction_counts(cases),
    }


def build_target_only_audit(
    cohort: TargetOnlyObservedCohort,
    *,
    lookback: int,
    products: Sequence[str],
    minimum_fit_days: int,
    inner_validation_days: int,
    evaluation_days: int,
    step_days: int,
    purge_days: int,
    model_provenance: Mapping[str, object] | None = None,
    data_provenance: Mapping[str, object] | None = None,
    include_v4_pair_only_comparison: bool = True,
) -> tuple[dict[str, object], TargetOnlyCaseBundle, tuple[V3WalkForwardFold, ...]]:
    """Build V5's full target-case audit and global expanding folds."""

    if purge_days < 3:
        raise ValueError("V5 requires at least a three-trading-day purge")
    bundle = build_target_only_cases(cohort, lookback=lookback, products=products)
    target_by_key = {case.case_key: case for case in bundle.target_cases}
    v4_comparison: dict[str, object]
    if include_v4_pair_only_comparison:
        # This audit-only comparison documents exactly what V5 gains by
        # removing V4's pair requirement. It never feeds a V5 case, dataset,
        # normalizer, probe, sampler, or P0/P1 execution.
        v4_pair_keys, v4_pair_rejection_reasons = _audit_v4_pair_only_coverage(
            cohort,
            target_cases=bundle.target_cases,
            lookback=lookback,
            products=products,
        )
        newly_included = sorted(set(target_by_key).difference(v4_pair_keys))
        new_reason_counts = Counter(
            v4_pair_rejection_reasons[key]
            for key in newly_included
        )
        v4_comparison = {
            "audit_only": True,
            "included": True,
            "v4_pair_case_count": int(len(v4_pair_keys)),
            "v5_target_case_count": int(len(bundle.target_cases)),
            "new_v5_target_cases_vs_v4_pair_only": int(len(newly_included)),
            "new_case_keys": newly_included,
            "new_cases_by_product": dict(
                sorted(Counter(target_by_key[key].product for key in newly_included).items())
            ),
            "new_cases_by_v4_pair_rejection_reason": dict(sorted(new_reason_counts.items())),
        }
    else:
        v4_comparison = {
            "audit_only": True,
            "included": False,
            "reason": "excluded_from_cuda_p0_p1_execution",
        }
    global_days = sorted({case.target_end_day for case in bundle.target_cases})
    folds = build_expanding_walk_forward_folds(
        global_days,
        minimum_fit_days=minimum_fit_days,
        inner_validation_days=inner_validation_days,
        evaluation_days=evaluation_days,
        step_days=step_days,
        purge_days=purge_days,
    )
    fold_summaries: list[dict[str, object]] = []
    for fold in folds:
        fit = cases_in_period(
            bundle.target_cases,
            start_day=fold.fit_start_day,
            end_day=fold.fit_end_day,
        )
        validation = cases_in_period(
            bundle.target_cases,
            start_day=fold.inner_validation_start_day,
            end_day=fold.inner_validation_end_day,
        )
        evaluation = cases_in_period(
            bundle.target_cases,
            start_day=fold.evaluation_start_day,
            end_day=fold.evaluation_end_day,
        )
        fold_summaries.append(
            {
                "fold_id": fold.fold_id,
                "fit": _case_summary(fit),
                "inner_validation": {
                    "start_day": fold.inner_validation_start_day,
                    "end_day": fold.inner_validation_end_day,
                    **_case_summary(validation),
                },
                "evaluation": {
                    "start_day": fold.evaluation_start_day,
                    "end_day": fold.evaluation_end_day,
                    **_case_summary(evaluation),
                },
            }
        )
    audit: dict[str, object] = {
        "strategy_version": 5,
        "phase": "audit",
        "result_scope": RESULT_SCOPE,
        "production_eligible": PRODUCTION_ELIGIBLE,
        "data_rule_version": DATA_RULE_VERSION,
        "data_fingerprint": cohort.data_fingerprint,
        "cohort": {
            "manifest_path": str(cohort.manifest_path),
            "manifest_sha256": cohort.manifest_sha256,
            "snapshot_id": cohort.snapshot_id,
            "snapshot_at": cohort.snapshot_at,
            "payload_sha256": dict(sorted(cohort.payload_sha256.items())),
            "contracts": {
                contract_id: {
                    "product": contract.product,
                    "bars": len(contract.frame),
                    "first_available_bar": contract.first_timestamp,
                    "last_available_bar": contract.last_timestamp,
                    **dict(cohort.contract_status[contract_id]),
                }
                for contract_id, contract in sorted(cohort.contracts.items())
            },
            "failed_contracts": {
                contract_id: status
                for contract_id, status in sorted(cohort.contract_status.items())
                if status.get("status") != "ok"
            },
        },
        "data_provenance": dict(data_provenance or {}),
        "model_provenance": dict(model_provenance or {}),
        "walk_forward": {
            "minimum_fit_days": int(minimum_fit_days),
            "inner_validation_days": int(inner_validation_days),
            "evaluation_days": int(evaluation_days),
            "step_days": int(step_days),
            "purge_days": int(purge_days),
            "fold_count": int(len(folds)),
        },
        "target_cases": _case_summary(bundle.target_cases),
        "target_rejections": dict(bundle.target_rejections),
        "target_rejection_records": list(bundle.target_rejection_records),
        "target_only_invariant": {
            "forbidden_case_attributes": [
                "has_pair",
                "nearest_neighbor_id",
                "selected_neighbor_id",
                "term_structure",
                "neighbor_context",
            ],
            "forbidden_attributes_present": {
                name: any(hasattr(case, name) for case in bundle.target_cases)
                for name in (
                    "has_pair",
                    "nearest_neighbor_id",
                    "selected_neighbor_id",
                    "term_structure",
                    "neighbor_context",
                )
            },
            "passes": not any(
                hasattr(case, name)
                for case in bundle.target_cases
                for name in (
                    "has_pair",
                    "nearest_neighbor_id",
                    "selected_neighbor_id",
                    "term_structure",
                    "neighbor_context",
                )
            ),
        },
        "comparison_with_v4_pair_only": v4_comparison,
        "folds": fold_summaries,
    }
    if not bool(audit["target_only_invariant"]["passes"]):
        raise TargetOnlyDataError("V5 target-only audit found forbidden neighbour attributes")
    return audit, bundle, folds


__all__ = [
    "DATA_RULE_VERSION",
    "TargetOnlyCase",
    "TargetOnlyCaseBundle",
    "TargetOnlyDataError",
    "TargetOnlyObservedCohort",
    "build_target_only_audit",
    "build_target_only_cases",
    "cases_in_period",
    "load_target_only_observed_cohort",
]
