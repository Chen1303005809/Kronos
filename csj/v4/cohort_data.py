"""Observed-cohort data construction for the V4 research strategy.

V4 intentionally does not claim to reconstruct a historical active-contract
panel.  It freezes the successfully loaded contracts in one current manifest
and, at each forecast origin, chooses the closest *currently observed and
context-complete* neighbour.  The resulting cases are therefore useful
retrospective research evidence only; they are never production eligibility
evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from csj.v3.panel_data import (
    VALID_TRADING_DAY_BAR_COUNTS,
    ConcreteContract,
    TermStructureState,
    V3WalkForwardFold,
    build_expanding_walk_forward_folds,
    case_arrays as _v3_case_arrays,
    delivery_month_distance,
    load_panel_archive,
    normalize_context as _v3_normalize_context,
)


RESULT_SCOPE = "retrospective_observed_cohort"
PRODUCTION_ELIGIBLE = False
DATA_RULE_VERSION = "observed-cohort-data-v1"
SELECTION_RULE_VERSION = "nearest-available-observed-neighbor-v1"


class ObservedCohortError(RuntimeError):
    """The frozen V4 cohort cannot meet a no-leakage data invariant."""


@dataclass(frozen=True)
class ObservedContractCohort:
    """A reproducible frozen set of successfully loaded concrete contracts."""

    manifest_path: Path
    manifest_sha256: str
    snapshot_id: str
    snapshot_at: pd.Timestamp
    contracts: Mapping[str, ConcreteContract]
    contract_status: Mapping[str, Mapping[str, object]]
    payload_sha256: Mapping[str, str]
    cohort_fingerprint: str


@dataclass(frozen=True)
class ObservedCohortCase:
    """One target path and its closest eligible observed-cohort neighbour."""

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
    selected_neighbor_id: str | None
    neighbor_contract: ConcreteContract | None
    neighbor_context_start: int | None
    term_structure: TermStructureState | None
    candidate_count: int
    selection_rule_version: str
    cohort_fingerprint: str
    neighbor_context_available_at_origin: bool
    pair_rejection_reason: str | None

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
    def neighbor_context(self) -> DataFrame | None:
        if self.neighbor_contract is None or self.neighbor_context_start is None:
            return None
        return self.neighbor_contract.frame.iloc[
            self.neighbor_context_start : self.neighbor_context_start
            + (self.target_start - self.target_context_start)
        ].copy()

    @property
    def case_key(self) -> str:
        return f"{self.target_contract_id}|{self.origin_timestamp.isoformat()}"

    @property
    def target_end_day(self) -> pd.Timestamp:
        return self.target_days[-1]

    @property
    def pred_len(self) -> int:
        return len(self.target)

    @property
    def day3_return(self) -> float:
        origin_close = float(self.target_contract.frame["close"].iloc[self.target_start - 1])
        day3_close = float(
            self.target_contract.frame["close"].iloc[
                self.target_start + self.day_end_indices[-1]
            ]
        )
        return day3_close / origin_close - 1.0

    # These aliases make the frozen V3 P1 probe reusable without allowing V4
    # to reuse V3's historical-panel selection semantics.
    @property
    def nearest_neighbor_id(self) -> str | None:
        return self.selected_neighbor_id

    @property
    def neighbor_delta_month(self) -> int | None:
        return (
            None
            if self.term_structure is None
            else self.term_structure.signed_month_distance
        )

    @property
    def has_pair(self) -> bool:
        return (
            self.neighbor_contract is not None
            and self.neighbor_context_start is not None
            and self.term_structure is not None
            and self.neighbor_context_available_at_origin
        )


@dataclass(frozen=True)
class ObservedCohortCaseBundle:
    """Target cases plus the V4 nearest-available pair subset for one lookback."""

    lookback: int
    target_cases: tuple[ObservedCohortCase, ...]
    pair_cases: tuple[ObservedCohortCase, ...]
    target_rejections: Mapping[str, int]
    pair_rejections: Mapping[str, int]

    @property
    def all_pair_cases(self) -> tuple[ObservedCohortCase, ...]:
        return self.pair_cases


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservedCohortError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ObservedCohortError(f"Expected a JSON object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_naive_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp


def _manifest_directories(root: str | Path) -> tuple[Path, ...]:
    candidate = Path(root)
    if (candidate / "manifest.json").is_file():
        return (candidate,)
    if not candidate.is_dir():
        raise ObservedCohortError(f"Snapshot root does not exist: {candidate}")
    manifests = tuple(sorted(path for path in candidate.iterdir() if (path / "manifest.json").is_file()))
    if not manifests:
        raise ObservedCohortError(f"No snapshot manifest exists below: {candidate}")
    return manifests


def _resolve_current_manifest(root: str | Path) -> Path:
    """Use the latest manifest deterministically when a parent archive is passed."""

    directories = _manifest_directories(root)
    if len(directories) == 1:
        return directories[0]
    candidates: list[tuple[pd.Timestamp, str, Path]] = []
    for directory in directories:
        manifest = _read_json(directory / "manifest.json")
        try:
            snapshot_at = _local_naive_timestamp(manifest["snapshot_at"])
        except KeyError as exc:
            raise ObservedCohortError(f"Manifest is missing snapshot_at: {directory}") from exc
        candidates.append((snapshot_at, directory.name, directory))
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _safe_payload_path(snapshot_dir: Path, value: object) -> Path:
    if not value:
        raise ObservedCohortError(f"Successful contract has no raw payload path: {snapshot_dir}")
    root = snapshot_dir.resolve()
    payload = (snapshot_dir / str(value)).resolve()
    try:
        payload.relative_to(root)
    except ValueError as exc:
        raise ObservedCohortError(f"Raw payload path escapes snapshot: {value!r}") from exc
    if not payload.is_file():
        raise ObservedCohortError(f"Raw payload is missing: {payload}")
    return payload


def load_observed_contract_cohort(snapshot_root: str | Path) -> ObservedContractCohort:
    """Freeze one current manifest and record immutable source hashes.

    Only contracts with a successful payload load enter the candidate cohort.
    Failed contracts remain visible in ``contract_status`` for auditability.
    """

    snapshot_dir = _resolve_current_manifest(snapshot_root)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    records = manifest.get("contracts")
    if not isinstance(records, list) or not records:
        raise ObservedCohortError(f"Manifest contains no contracts: {manifest_path}")
    try:
        snapshot_at = _local_naive_timestamp(manifest["snapshot_at"])
    except KeyError as exc:
        raise ObservedCohortError(f"Manifest is missing snapshot_at: {manifest_path}") from exc
    archive = load_panel_archive(snapshot_dir)
    statuses: dict[str, dict[str, object]] = {}
    payload_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ObservedCohortError(f"Invalid contract record in {manifest_path}")
        contract_id = str(record.get("contract_id", "")).lower().strip()
        product = str(record.get("product", "")).lower().strip()
        if not contract_id or not product:
            raise ObservedCohortError(f"Manifest contract lacks product or ID: {manifest_path}")
        if contract_id in statuses:
            raise ObservedCohortError(f"Duplicate contract in manifest: {contract_id}")
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
            raise ObservedCohortError(
                f"Raw payload checksum mismatch for {contract_id}: {payload_path}"
            )
        payload_hashes[contract_id] = observed_hash
    contracts = {
        contract_id: contract
        for contract_id, contract in archive.contracts.items()
        if statuses.get(contract_id, {}).get("status") == "ok"
    }
    expected_successes = {key for key, item in statuses.items() if item["status"] == "ok"}
    if set(contracts) != expected_successes:
        missing = sorted(expected_successes.difference(contracts))
        raise ObservedCohortError(
            f"Manifest successful contracts did not all load: {missing[:5]!r}"
        )
    if not contracts:
        raise ObservedCohortError("Observed cohort has no successful contracts")
    manifest_sha256 = _sha256_path(manifest_path)
    fingerprint_payload = {
        "data_rule_version": DATA_RULE_VERSION,
        "manifest_sha256": manifest_sha256,
        "snapshot_id": snapshot_dir.name,
        "contracts": [
            {"contract_id": contract_id, "payload_sha256": payload_hashes[contract_id]}
            for contract_id in sorted(contracts)
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ObservedContractCohort(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        snapshot_id=snapshot_dir.name,
        snapshot_at=snapshot_at,
        contracts=contracts,
        contract_status=statuses,
        payload_sha256=payload_hashes,
        cohort_fingerprint=fingerprint,
    )


def _frame_arrays(frame: DataFrame) -> Mapping[str, np.ndarray]:
    cached = frame.attrs.get("v4_observed_cohort_arrays")
    if isinstance(cached, dict):
        return cached
    arrays: dict[str, np.ndarray] = {
        "timestamps": frame["timestamps"].to_numpy(),
        "trading_day": frame["trading_day"].to_numpy(),
        "volume": frame["volume"].to_numpy(dtype=np.float64),
        "close": frame["close"].to_numpy(dtype=np.float64),
    }
    frame.attrs["v4_observed_cohort_arrays"] = arrays
    return arrays


def _context_ending_at(
    frame: DataFrame,
    origin_timestamp: pd.Timestamp,
    lookback: int,
) -> tuple[int, int] | None:
    positions = frame.attrs.get("timestamp_positions")
    if not isinstance(positions, dict):
        positions = {
            pd.Timestamp(timestamp): index
            for index, timestamp in enumerate(frame["timestamps"].tolist())
        }
        frame.attrs["timestamp_positions"] = positions
    position = positions.get(pd.Timestamp(origin_timestamp))
    if position is None or position + 1 < lookback:
        return None
    start = int(position - lookback + 1)
    end = int(position + 1)
    return (start, end) if end - start == lookback else None


def _context_structure_reason(frame: DataFrame, start: int, end: int) -> str | None:
    """Reject a fully included context day with other than five or seven bars."""

    days = _frame_arrays(frame)["trading_day"]
    cursor = start
    while cursor < end:
        day = days[cursor]
        group_end = cursor + 1
        while group_end < end and days[group_end] == day:
            group_end += 1
        full_start = cursor == 0 or days[cursor - 1] != day
        full_end = group_end == len(days) or days[group_end] != day
        if full_start and full_end and group_end - cursor not in VALID_TRADING_DAY_BAR_COUNTS:
            return "context_has_nonstandard_complete_trading_day"
        cursor = group_end
    return None


def _last_five_day_volume_ratio(
    target_frame: DataFrame,
    target_start: int,
    target_end: int,
    neighbor_frame: DataFrame,
    neighbor_start: int,
    neighbor_end: int,
) -> tuple[float | None, str | None]:
    """Calculate the known-at-origin five-day volume state or reject the pair."""

    target = _frame_arrays(target_frame)
    neighbor = _frame_arrays(neighbor_frame)
    target_cursor = target_end
    neighbor_cursor = neighbor_end
    target_total = 0.0
    neighbor_total = 0.0
    for _ in range(5):
        if target_cursor <= target_start or neighbor_cursor <= neighbor_start:
            return None, "term_state_requires_five_complete_days"
        target_day = target["trading_day"][target_cursor - 1]
        neighbor_day = neighbor["trading_day"][neighbor_cursor - 1]
        target_day_start = target_cursor - 1
        neighbor_day_start = neighbor_cursor - 1
        while target_day_start > target_start and target["trading_day"][target_day_start - 1] == target_day:
            target_day_start -= 1
        while (
            neighbor_day_start > neighbor_start
            and neighbor["trading_day"][neighbor_day_start - 1] == neighbor_day
        ):
            neighbor_day_start -= 1
        target_count = target_cursor - target_day_start
        neighbor_count = neighbor_cursor - neighbor_day_start
        if (
            target_count not in VALID_TRADING_DAY_BAR_COUNTS
            or neighbor_count != target_count
            or target_day != neighbor_day
            or not np.array_equal(
                target["timestamps"][target_day_start:target_cursor],
                neighbor["timestamps"][neighbor_day_start:neighbor_cursor],
            )
        ):
            return None, "term_state_requires_five_complete_days"
        target_total += float(target["volume"][target_day_start:target_cursor].sum())
        neighbor_total += float(neighbor["volume"][neighbor_day_start:neighbor_cursor].sum())
        target_cursor = target_day_start
        neighbor_cursor = neighbor_day_start
    return float(np.log((neighbor_total + 1e-12) / (target_total + 1e-12))), None


def _select_nearest_available_neighbor(
    cohort: ObservedContractCohort,
    *,
    target_contract: ConcreteContract,
    target_context_start: int,
    target_context_end: int,
    lookback: int,
) -> tuple[
    str | None,
    ConcreteContract | None,
    int | None,
    TermStructureState | None,
    int,
    str | None,
]:
    """Filter candidate availability first, then rank maturity distance.

    This ordering is V4's central difference from V3.  An unavailable nominal
    closest contract never prevents selection of the next closest usable one.
    """

    target_arrays = _frame_arrays(target_contract.frame)
    origin_timestamp = pd.Timestamp(target_arrays["timestamps"][target_context_end - 1])
    target_timestamps = target_arrays["timestamps"][target_context_start:target_context_end]
    target_days = target_arrays["trading_day"][target_context_start:target_context_end]
    target_structure_reason = _context_structure_reason(
        target_contract.frame, target_context_start, target_context_end
    )
    candidates = [
        contract
        for contract in cohort.contracts.values()
        if contract.product == target_contract.product
        and contract.contract_id != target_contract.contract_id
    ]
    if not candidates:
        return None, None, None, None, 0, "no_same_product_observed_neighbor"
    rejection_counts: Counter[str] = Counter()
    eligible: list[tuple[str, ConcreteContract, int, TermStructureState]] = []
    for neighbor_contract in sorted(candidates, key=lambda item: item.contract_id):
        positions = _context_ending_at(neighbor_contract.frame, origin_timestamp, lookback)
        if positions is None:
            rejection_counts["candidate_lacks_context_at_origin"] += 1
            continue
        neighbor_start, neighbor_end = positions
        neighbor_arrays = _frame_arrays(neighbor_contract.frame)
        if not np.array_equal(
            target_timestamps,
            neighbor_arrays["timestamps"][neighbor_start:neighbor_end],
        ):
            rejection_counts["candidate_context_timestamp_mismatch"] += 1
            continue
        if not np.array_equal(
            target_days,
            neighbor_arrays["trading_day"][neighbor_start:neighbor_end],
        ):
            rejection_counts["candidate_context_trading_day_mismatch"] += 1
            continue
        if target_structure_reason is not None:
            rejection_counts[target_structure_reason] += 1
            continue
        # Timestamp equality has already established the entire observed
        # neighbour context.  The complete-day rule is evaluated on that shared
        # aligned sequence (the target slice); a neighbour may have an extra
        # bar immediately outside its 256-bar window, which must not turn an
        # otherwise aligned window into a synthetic rejection.
        target_close = float(target_arrays["close"][target_context_end - 1])
        neighbor_close = float(neighbor_arrays["close"][neighbor_end - 1])
        if target_close <= 0.0 or neighbor_close <= 0.0:
            rejection_counts["non_positive_close_for_term_structure"] += 1
            continue
        volume_ratio, volume_reason = _last_five_day_volume_ratio(
            target_contract.frame,
            target_context_start,
            target_context_end,
            neighbor_contract.frame,
            neighbor_start,
            neighbor_end,
        )
        if volume_reason is not None:
            rejection_counts[volume_reason] += 1
            continue
        assert volume_ratio is not None
        distance = delivery_month_distance(target_contract.contract_id, neighbor_contract.contract_id)
        eligible.append(
            (
                neighbor_contract.contract_id,
                neighbor_contract,
                neighbor_start,
                TermStructureState(
                    log_close_ratio=float(np.log(neighbor_close / target_close)),
                    signed_month_distance=distance,
                    log_volume_ratio_5d=volume_ratio,
                ),
            )
        )
    candidate_count = len(eligible)
    if not eligible:
        if rejection_counts:
            # Keep the leading, deterministic reason while the aggregate audit
            # records every rejection category separately.
            reason = min(rejection_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        else:
            reason = "no_available_neighbor_at_origin"
        return None, None, None, None, candidate_count, reason

    def rank(candidate: tuple[str, ConcreteContract, int, TermStructureState]) -> tuple[int, int, str]:
        contract_id, _, _, state = candidate
        # Equal absolute maturity distance deterministically prefers later month.
        return (abs(state.signed_month_distance), 0 if state.signed_month_distance > 0 else 1, contract_id)

    selected_id, selected_contract, selected_start, term_state = min(eligible, key=rank)
    return (
        selected_id,
        selected_contract,
        selected_start,
        term_state,
        candidate_count,
        None,
    )


def _target_day_groups(frame: DataFrame) -> list[tuple[pd.Timestamp, np.ndarray]]:
    return [
        (pd.Timestamp(day).normalize(), group.index.to_numpy(dtype=np.int64))
        for day, group in frame.groupby("trading_day", sort=False)
    ]


def build_observed_cohort_cases(
    cohort: ObservedContractCohort,
    *,
    lookback: int,
    products: Iterable[str] | None = None,
) -> ObservedCohortCaseBundle:
    """Build V4 target and nearest-*available* observed-cohort pair cases."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    selected_products = (
        {str(product).lower() for product in products}
        if products is not None
        else {contract.product for contract in cohort.contracts.values()}
    )
    known_products = {contract.product for contract in cohort.contracts.values()}
    unknown = selected_products.difference(known_products)
    if unknown:
        raise ValueError(f"No observed cohort contracts for products: {sorted(unknown)!r}")
    target_cases: list[ObservedCohortCase] = []
    pair_cases: list[ObservedCohortCase] = []
    target_rejections: Counter[str] = Counter()
    pair_rejections: Counter[str] = Counter()
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
                target_rejections["target_requires_three_complete_days"] += 1
                continue
            target_indices = np.concatenate([indices for _, indices in selected_days])
            target_start = int(target_indices[0])
            if target_start < lookback:
                target_rejections["target_lacks_context"] += 1
                continue
            target_context_start = target_start - lookback
            (
                neighbor_id,
                neighbor_contract,
                neighbor_context_start,
                term_structure,
                candidate_count,
                pair_reason,
            ) = _select_nearest_available_neighbor(
                cohort,
                target_contract=target_contract,
                target_context_start=target_context_start,
                target_context_end=target_start,
                lookback=lookback,
            )
            origin_timestamp = pd.Timestamp(frame["timestamps"].iloc[target_start - 1])
            origin_trading_day = pd.Timestamp(frame["trading_day"].iloc[target_start - 1]).normalize()
            day_end_indices = tuple(int(value) for value in np.cumsum(bar_counts) - 1)
            case = ObservedCohortCase(
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
                selected_neighbor_id=neighbor_id,
                neighbor_contract=neighbor_contract,
                neighbor_context_start=neighbor_context_start,
                term_structure=term_structure,
                candidate_count=candidate_count,
                selection_rule_version=SELECTION_RULE_VERSION,
                cohort_fingerprint=cohort.cohort_fingerprint,
                neighbor_context_available_at_origin=neighbor_contract is not None,
                pair_rejection_reason=pair_reason,
            )
            target_cases.append(case)
            if case.has_pair:
                pair_cases.append(case)
            else:
                pair_rejections[pair_reason or "unknown_pair_rejection"] += 1
    sort_key = lambda case: (case.target_end_day, case.target_contract_id, case.case_key)
    target_cases.sort(key=sort_key)
    pair_cases.sort(key=sort_key)
    return ObservedCohortCaseBundle(
        lookback=int(lookback),
        target_cases=tuple(target_cases),
        pair_cases=tuple(pair_cases),
        target_rejections=dict(sorted(target_rejections.items())),
        pair_rejections=dict(sorted(pair_rejections.items())),
    )


def _direction_counts(cases: Sequence[ObservedCohortCase]) -> dict[str, int]:
    counts = {"earlier": 0, "later": 0}
    for case in cases:
        distance = case.neighbor_delta_month
        if distance is None:
            continue
        if distance < 0:
            counts["earlier"] += 1
        elif distance > 0:
            counts["later"] += 1
    return counts


def _product_counts(cases: Sequence[ObservedCohortCase]) -> dict[str, int]:
    return dict(sorted(Counter(case.product for case in cases).items()))


def _cases_in_period(
    cases: Sequence[ObservedCohortCase], start_day: pd.Timestamp, end_day: pd.Timestamp
) -> tuple[ObservedCohortCase, ...]:
    return tuple(
        case for case in cases if start_day <= case.target_end_day <= end_day
    )


def _bundle_summary(
    bundle: ObservedCohortCaseBundle,
    *,
    folds: Sequence[V3WalkForwardFold],
) -> dict[str, object]:
    fold_summaries = []
    for fold in folds:
        pair = _cases_in_period(
            bundle.pair_cases, fold.evaluation_start_day, fold.evaluation_end_day
        )
        target = _cases_in_period(
            bundle.target_cases, fold.evaluation_start_day, fold.evaluation_end_day
        )
        fold_summaries.append(
            {
                "fold_id": fold.fold_id,
                "fit_end_day": fold.fit_end_day,
                "inner_validation": {
                    "start_day": fold.inner_validation_start_day,
                    "end_day": fold.inner_validation_end_day,
                },
                "evaluation": {
                    "start_day": fold.evaluation_start_day,
                    "end_day": fold.evaluation_end_day,
                },
                "target_cases": len(target),
                "pair_cases": len(pair),
                "pair_cases_by_product": _product_counts(pair),
                "pair_cases_by_neighbor_direction": _direction_counts(pair),
            }
        )
    return {
        "lookback": bundle.lookback,
        "target_cases": len(bundle.target_cases),
        "pair_cases": len(bundle.pair_cases),
        "pair_retention_vs_target": (
            len(bundle.pair_cases) / len(bundle.target_cases)
            if bundle.target_cases
            else None
        ),
        "target_cases_by_product": _product_counts(bundle.target_cases),
        "pair_cases_by_product": _product_counts(bundle.pair_cases),
        "pair_cases_by_neighbor_direction": _direction_counts(bundle.pair_cases),
        "target_rejections": dict(bundle.target_rejections),
        "pair_rejections": dict(bundle.pair_rejections),
        "folds": fold_summaries,
    }


def build_observed_cohort_audit(
    cohort: ObservedContractCohort,
    *,
    lookbacks: Sequence[int] = (256, 512),
    products: Iterable[str] | None = None,
    minimum_fit_days: int = 60,
    inner_validation_days: int = 20,
    evaluation_days: int = 20,
    step_days: int = 20,
    purge_days: int = 3,
    model_provenance: Mapping[str, object] | None = None,
    data_provenance: Mapping[str, object] | None = None,
) -> tuple[
    dict[str, object],
    Mapping[int, ObservedCohortCaseBundle],
    tuple[V3WalkForwardFold, ...],
]:
    """Audit V4 coverage and freeze the required 256-bar context length."""

    normalized_lookbacks = tuple(sorted({int(value) for value in lookbacks}))
    if normalized_lookbacks != (256, 512):
        raise ValueError("V4 audit must compare exactly lookbacks 256 and 512")
    if purge_days < 3:
        raise ValueError("V4 requires at least a three-trading-day purge")
    bundles = {
        lookback: build_observed_cohort_cases(cohort, lookback=lookback, products=products)
        for lookback in normalized_lookbacks
    }
    global_days = sorted(
        {
            case.target_end_day
            for bundle in bundles.values()
            for case in bundle.target_cases
        }
    )
    folds = build_expanding_walk_forward_folds(
        global_days,
        minimum_fit_days=minimum_fit_days,
        inner_validation_days=inner_validation_days,
        evaluation_days=evaluation_days,
        step_days=step_days,
        purge_days=purge_days,
    )
    summaries = {
        str(lookback): _bundle_summary(bundles[lookback], folds=folds)
        for lookback in normalized_lookbacks
    }
    fold_retention = []
    for fold in folds:
        cases_256 = _cases_in_period(
            bundles[256].pair_cases, fold.evaluation_start_day, fold.evaluation_end_day
        )
        cases_512 = _cases_in_period(
            bundles[512].pair_cases, fold.evaluation_start_day, fold.evaluation_end_day
        )
        count_256 = len(cases_256)
        count_512 = len(cases_512)
        fold_retention.append(
            {
                "fold_id": fold.fold_id,
                "pair_cases_256": count_256,
                "pair_cases_512": count_512,
                "retention_512_vs_256": None if count_256 == 0 else count_512 / count_256,
                "directions_256": _direction_counts(cases_256),
                "directions_512": _direction_counts(cases_512),
            }
        )
    overall_retention = (
        len(bundles[512].pair_cases) / len(bundles[256].pair_cases)
        if bundles[256].pair_cases
        else None
    )
    audit: dict[str, object] = {
        "strategy_version": 4,
        "phase": "audit",
        "result_scope": RESULT_SCOPE,
        "production_eligible": PRODUCTION_ELIGIBLE,
        "data_rule_version": DATA_RULE_VERSION,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "cohort": {
            "manifest_path": str(cohort.manifest_path),
            "manifest_sha256": cohort.manifest_sha256,
            "snapshot_id": cohort.snapshot_id,
            "snapshot_at": cohort.snapshot_at,
            "cohort_fingerprint": cohort.cohort_fingerprint,
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
                if status["status"] != "ok"
            },
        },
        "data_provenance": dict(data_provenance or {}),
        "model_provenance": dict(model_provenance or {}),
        "walk_forward": {
            "minimum_fit_days": minimum_fit_days,
            "inner_validation_days": inner_validation_days,
            "evaluation_days": evaluation_days,
            "step_days": step_days,
            "purge_days": purge_days,
            "fold_count": len(folds),
        },
        "coverage_by_lookback": summaries,
        "context_length_decision": {
            "primary_context_length": 256,
            "rule": "V4 fixes 256 because observed-cohort 512 pair retention must be >=80%",
            "overall_retention_512_vs_256": overall_retention,
            "passes_512_retention_rule": bool(
                overall_retention is not None and overall_retention >= 0.80
            ),
            "folds": fold_retention,
        },
    }
    return audit, bundles, folds


def case_arrays(
    case: ObservedCohortCase, *, include_neighbor: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return native six-feature streams using the shared V3-safe adapter."""

    return _v3_case_arrays(case, include_neighbor=include_neighbor)


def normalize_context(
    values: np.ndarray, *, clip: float = 5.0, epsilon: float = 1e-5
) -> np.ndarray:
    return _v3_normalize_context(values, clip=clip, epsilon=epsilon)


__all__ = [
    "DATA_RULE_VERSION",
    "PRODUCTION_ELIGIBLE",
    "RESULT_SCOPE",
    "SELECTION_RULE_VERSION",
    "ObservedCohortCase",
    "ObservedCohortCaseBundle",
    "ObservedCohortError",
    "ObservedContractCohort",
    "V3WalkForwardFold",
    "build_expanding_walk_forward_folds",
    "build_observed_cohort_audit",
    "build_observed_cohort_cases",
    "case_arrays",
    "load_observed_contract_cohort",
    "normalize_context",
]
