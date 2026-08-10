"""Concrete-contract panel loading, case construction, and coverage audits.

The module implements the data-side invariants of the V3 strategy:

* a target remains one concrete contract for its whole context and label path;
* panel membership comes from an immutable saved active-contract manifest;
* a nearest neighbour is selected by parsed delivery month, never string order;
* each selected stream keeps the native six-feature Kronos interface;
* pair contexts must be timestamp-aligned without synthetic filling; and
* a historical case without a same-day saved panel is explicitly partial.

The public builder returns target-only cases as well as strict and exploratory
pair subsets.  Callers must opt in before using exploratory partial-panel cases
for any training; they are never suitable for the final panel comparison.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from csj.active_contract_data import parse_delivery_year_month
from csj.utils.tool import MODEL_FEATURES, d_to_df


TIME_FEATURES: tuple[str, ...] = ("minute", "hour", "weekday", "day", "month")
VALID_TRADING_DAY_BAR_COUNTS = frozenset((5, 7))


class PanelDataError(RuntimeError):
    """The archived concrete-contract data cannot satisfy a V3 invariant."""


@dataclass(frozen=True)
class PanelSnapshot:
    """One immutable active-contract manifest, represented in local wall time."""

    snapshot_id: str
    snapshot_at: pd.Timestamp
    panel_completeness: str
    contracts_by_product: Mapping[str, tuple[str, ...]]
    contract_status: Mapping[str, str]

    @property
    def is_complete(self) -> bool:
        return self.panel_completeness == "complete"


@dataclass(frozen=True)
class ConcreteContract:
    """Merged history for one specific delivery-month contract."""

    contract_id: str
    product: str
    frame: DataFrame
    source_snapshot_ids: tuple[str, ...]

    @property
    def first_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame["timestamps"].iloc[0])

    @property
    def last_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame["timestamps"].iloc[-1])


@dataclass(frozen=True)
class PanelArchive:
    """All immutable snapshots plus de-duplicated concrete-contract histories."""

    root: Path
    snapshots: tuple[PanelSnapshot, ...]
    contracts: Mapping[str, ConcreteContract]
    source_audit: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class PanelResolution:
    """Panel selection provenance at one prediction origin."""

    snapshot: PanelSnapshot | None
    partial_panel: bool
    source: str


@dataclass(frozen=True)
class TermStructureState:
    """Known-at-origin state that survives per-contract normalization."""

    log_close_ratio: float
    signed_month_distance: int
    log_volume_ratio_5d: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.log_close_ratio,
                float(self.signed_month_distance),
                self.log_volume_ratio_5d,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class PanelCase:
    """One target path and, when eligible, its single selected nearest neighbour."""

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
    panel_snapshot_id: str | None
    panel_source: str
    partial_panel: bool
    nearest_neighbor_id: str | None
    neighbor_contract: ConcreteContract | None
    neighbor_context_start: int | None
    term_structure: TermStructureState | None
    pair_rejection_reason: str | None

    @property
    def target_context(self) -> DataFrame:
        """Materialize the target context only when a caller needs its values."""

        return self.target_contract.frame.iloc[
            self.target_context_start : self.target_start
        ].copy()

    @property
    def target(self) -> DataFrame:
        """Materialize the target's three full trading days on demand."""

        return self.target_contract.frame.iloc[
            self.target_start : self.target_end_exclusive
        ].copy()

    @property
    def neighbor_context(self) -> DataFrame | None:
        """Materialize the selected nearest-neighbour context on demand."""

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
        origin_close = float(
            self.target_contract.frame["close"].iloc[self.target_start - 1]
        )
        day3_close = float(
            self.target_contract.frame["close"].iloc[
                self.target_start + self.day_end_indices[-1]
            ]
        )
        return day3_close / origin_close - 1.0

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
        )


@dataclass(frozen=True)
class PanelCaseBundle:
    """Target-only cases and the strict/exploratory pair subsets for one lookback."""

    lookback: int
    target_cases: tuple[PanelCase, ...]
    strict_pair_cases: tuple[PanelCase, ...]
    partial_pair_cases: tuple[PanelCase, ...]
    target_rejections: Mapping[str, int]
    pair_rejections: Mapping[str, int]

    @property
    def all_pair_cases(self) -> tuple[PanelCase, ...]:
        return self.strict_pair_cases + self.partial_pair_cases


@dataclass(frozen=True)
class V3WalkForwardFold:
    """Expanding time fold with a three-trading-day purge between adjacent splits."""

    fold_id: str
    fit_start_day: pd.Timestamp
    fit_end_day: pd.Timestamp
    inner_validation_start_day: pd.Timestamp
    inner_validation_end_day: pd.Timestamp
    evaluation_start_day: pd.Timestamp
    evaluation_end_day: pd.Timestamp
    purge_days: int


def add_time_features(frame: DataFrame) -> DataFrame:
    """Return a copy with the five temporal features expected by Kronos."""

    enriched = frame.copy()
    values = pd.to_datetime(enriched["timestamps"])
    enriched["minute"] = values.dt.minute
    enriched["hour"] = values.dt.hour
    enriched["weekday"] = values.dt.weekday
    enriched["day"] = values.dt.day
    enriched["month"] = values.dt.month
    return enriched


@lru_cache(maxsize=None)
def delivery_month_distance(target_contract_id: str, neighbor_contract_id: str) -> int:
    """Return ``neighbor - target`` delivery month distance across years."""

    target_year, target_month = parse_delivery_year_month(target_contract_id)
    neighbor_year, neighbor_month = parse_delivery_year_month(neighbor_contract_id)
    return (neighbor_year - target_year) * 12 + neighbor_month - target_month


def _local_naive_timestamp(value: object) -> pd.Timestamp:
    """Use Asia/Shanghai wall time so manifest and provider timestamps compare safely."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelDataError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PanelDataError(f"Expected JSON object: {path}")
    return value


def _manifest_directories(root: str | Path) -> list[Path]:
    archive_root = Path(root)
    if (archive_root / "manifest.json").is_file():
        return [archive_root]
    if not archive_root.is_dir():
        raise PanelDataError(f"Snapshot archive root does not exist: {archive_root}")
    directories = sorted(
        path for path in archive_root.iterdir() if (path / "manifest.json").is_file()
    )
    if not directories:
        raise PanelDataError(f"No snapshot manifest exists below: {archive_root}")
    return directories


def _safe_raw_path(snapshot_dir: Path, relative_path: object) -> Path:
    if not relative_path:
        raise PanelDataError(f"Missing raw payload path in snapshot: {snapshot_dir}")
    root = snapshot_dir.resolve()
    raw_path = (snapshot_dir / str(relative_path)).resolve()
    try:
        raw_path.relative_to(root)
    except ValueError as exc:
        raise PanelDataError(
            f"Raw payload path escapes its snapshot directory: {relative_path!r}"
        ) from exc
    if not raw_path.is_file():
        raise PanelDataError(f"Raw payload is missing: {raw_path}")
    return raw_path


def _verify_payload_checksum(path: Path, expected: object) -> None:
    if not expected:
        return
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != str(expected):
        raise PanelDataError(f"Raw payload checksum mismatch: {path}")


def _merge_contract_frames(
    contract_id: str,
    product: str,
    frame_items: Sequence[tuple[pd.Timestamp, str, DataFrame]],
) -> ConcreteContract:
    if not frame_items:
        raise PanelDataError(f"No data frames supplied for {contract_id}")
    ordered_items = sorted(frame_items, key=lambda item: (item[0], item[1]))
    combined = pd.concat(
        [frame.copy() for _, _, frame in ordered_items],
        ignore_index=True,
    ).sort_values(["timestamps", "source_snapshot_id"], kind="stable")
    duplicate_rows = combined.loc[
        combined["timestamps"].duplicated(keep=False)
    ]
    for timestamp, rows in duplicate_rows.groupby("timestamps", sort=False):
        reference = rows.iloc[0]
        for _, candidate in rows.iloc[1:].iterrows():
            same_day = pd.Timestamp(candidate["trading_day"]) == pd.Timestamp(
                reference["trading_day"]
            )
            same_values = np.array_equal(
                candidate[MODEL_FEATURES + ["open_interest"]].to_numpy(dtype=np.float64),
                reference[MODEL_FEATURES + ["open_interest"]].to_numpy(dtype=np.float64),
                equal_nan=True,
            )
            if not same_day or not same_values:
                raise PanelDataError(
                    f"Conflicting archived values for {contract_id} at {timestamp}"
                )
    merged = combined.drop_duplicates("timestamps", keep="first").reset_index(drop=True)
    if not merged["timestamps"].is_monotonic_increasing:
        raise PanelDataError(f"Merged timestamps are not monotonic for {contract_id}")
    if merged["timestamps"].duplicated().any():
        raise PanelDataError(f"Merged timestamps remain duplicated for {contract_id}")
    merged.attrs["timestamp_positions"] = {
        pd.Timestamp(timestamp): index
        for index, timestamp in enumerate(merged["timestamps"].tolist())
    }
    merged.attrs["contract_id"] = contract_id
    merged.attrs["product"] = product
    return ConcreteContract(
        contract_id=contract_id,
        product=product,
        frame=merged,
        source_snapshot_ids=tuple(snapshot_id for _, snapshot_id, _ in ordered_items),
    )


def load_panel_archive(root: str | Path) -> PanelArchive:
    """Load and validate immutable manifests and merge their raw contract bars.

    Every raw response is converted through the same provider parser used by the
    previous experiments.  This retains OHLC, non-negative flow, timestamp, and
    trading-day validation while preserving a per-contract history.
    """

    snapshot_dirs = _manifest_directories(root)
    snapshots: list[PanelSnapshot] = []
    frames_by_contract: dict[str, list[tuple[pd.Timestamp, str, DataFrame]]] = defaultdict(list)
    product_by_contract: dict[str, str] = {}
    source_audit: dict[str, dict[str, object]] = {}

    for snapshot_dir in snapshot_dirs:
        manifest = _read_json(snapshot_dir / "manifest.json")
        try:
            snapshot_at = _local_naive_timestamp(manifest["snapshot_at"])
        except KeyError as exc:
            raise PanelDataError(f"Manifest is missing snapshot_at: {snapshot_dir}") from exc
        records = manifest.get("contracts")
        if not isinstance(records, list) or not records:
            raise PanelDataError(f"Manifest contains no contract list: {snapshot_dir}")

        contracts_by_product: dict[str, list[str]] = defaultdict(list)
        contract_status: dict[str, str] = {}
        succeeded = 0
        failed = 0
        for record in records:
            if not isinstance(record, dict):
                raise PanelDataError(f"Invalid contract manifest record: {snapshot_dir}")
            product = str(record.get("product", "")).lower().strip()
            contract_id = str(record.get("contract_id", "")).lower().strip()
            if not product or not contract_id:
                raise PanelDataError(f"Manifest contract is missing product or ID: {snapshot_dir}")
            try:
                parse_delivery_year_month(contract_id)
            except ValueError as exc:
                raise PanelDataError(
                    f"Invalid concrete contract in manifest: {contract_id!r}"
                ) from exc
            if not contract_id.startswith(product):
                raise PanelDataError(
                    f"Manifest product/contract mismatch: {product!r}/{contract_id!r}"
                )
            status = str(record.get("status", "failed")).lower()
            if contract_id in contract_status:
                raise PanelDataError(f"Duplicate manifest contract: {contract_id}")
            contract_status[contract_id] = status
            contracts_by_product[product].append(contract_id)

            if status != "ok":
                failed += 1
                continue
            succeeded += 1
            raw_path = _safe_raw_path(snapshot_dir, record.get("raw_payload_path"))
            _verify_payload_checksum(raw_path, record.get("raw_payload_sha256"))
            payload = _read_json(raw_path)
            try:
                frame = d_to_df(None, payload, persist_raw=False, validate=True)
            except (TypeError, ValueError) as exc:
                raise PanelDataError(
                    f"Invalid K-line payload for {contract_id}: {raw_path}"
                ) from exc
            reported = set(frame["instrument"].astype(str).str.lower().unique())
            if reported != {contract_id}:
                raise PanelDataError(
                    f"Payload contract mismatch for {contract_id}: {sorted(reported)!r}"
                )
            known_product = product_by_contract.setdefault(contract_id, product)
            if known_product != product:
                raise PanelDataError(
                    f"Contract {contract_id} changes product across manifests"
                )
            frame = frame.copy()
            frame["instrument"] = contract_id
            frame["contract_id"] = contract_id
            frame["product"] = product
            frame["source_snapshot_id"] = snapshot_dir.name
            frames_by_contract[contract_id].append(
                (snapshot_at, snapshot_dir.name, frame)
            )

        completeness = str(manifest.get("panel_completeness", "partial_panel"))
        snapshots.append(
            PanelSnapshot(
                snapshot_id=snapshot_dir.name,
                snapshot_at=snapshot_at,
                panel_completeness=completeness,
                contracts_by_product={
                    product: tuple(contract_ids)
                    for product, contract_ids in sorted(contracts_by_product.items())
                },
                contract_status=contract_status,
            )
        )
        source_audit[snapshot_dir.name] = {
            "snapshot_at": snapshot_at.isoformat(),
            "panel_completeness": completeness,
            "succeeded_contracts": succeeded,
            "failed_contracts": failed,
            "listed_contracts": succeeded + failed,
        }

    merged_contracts = {
        contract_id: _merge_contract_frames(
            contract_id,
            product_by_contract[contract_id],
            items,
        )
        for contract_id, items in sorted(frames_by_contract.items())
    }
    if not merged_contracts:
        raise PanelDataError("No successful concrete-contract payloads were loaded")
    return PanelArchive(
        root=Path(root),
        snapshots=tuple(sorted(snapshots, key=lambda item: (item.snapshot_at, item.snapshot_id))),
        contracts=merged_contracts,
        source_audit=source_audit,
    )


def resolve_panel_at(
    archive: PanelArchive,
    origin_timestamp: pd.Timestamp,
    *,
    allow_retrospective_partial: bool = True,
) -> PanelResolution:
    """Resolve a same-day known panel, otherwise mark the case explicitly partial.

    A later snapshot can be used only to count *exploratory* cases.  Its future
    membership is never represented as a valid historical panel and is excluded
    from ``strict_pair_cases``.
    """

    origin = _local_naive_timestamp(origin_timestamp)
    same_day_prior = [
        snapshot
        for snapshot in archive.snapshots
        if snapshot.snapshot_at.normalize() == origin.normalize()
        and snapshot.snapshot_at <= origin
    ]
    if same_day_prior:
        snapshot = max(same_day_prior, key=lambda item: (item.snapshot_at, item.snapshot_id))
        return PanelResolution(
            snapshot=snapshot,
            partial_panel=not snapshot.is_complete,
            source="same_day_snapshot",
        )

    # A previous day's list is known at the origin, but it is not a snapshot of
    # the origin's active panel. It may support exploratory data-quality work
    # only and must remain partial even when that old manifest was complete.
    stale_prior = [snapshot for snapshot in archive.snapshots if snapshot.snapshot_at <= origin]
    if stale_prior:
        snapshot = max(stale_prior, key=lambda item: (item.snapshot_at, item.snapshot_id))
        return PanelResolution(
            snapshot=snapshot,
            partial_panel=True,
            source="stale_prior_snapshot",
        )

    if allow_retrospective_partial:
        future = [snapshot for snapshot in archive.snapshots if snapshot.snapshot_at > origin]
        if future:
            snapshot = min(future, key=lambda item: (item.snapshot_at, item.snapshot_id))
            return PanelResolution(
                snapshot=snapshot,
                partial_panel=True,
                source="retrospective_future_snapshot",
            )
    return PanelResolution(snapshot=None, partial_panel=True, source="missing_same_day_snapshot")


def _last_context_ending_at(
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
    position = positions.get(pd.Timestamp(origin_timestamp))
    if position is None or position + 1 < lookback:
        return None
    start = int(position - lookback + 1)
    end = int(position + 1)
    return (start, end) if end - start == lookback else None


def _frame_arrays(frame: DataFrame) -> Mapping[str, np.ndarray]:
    """Cache immutable NumPy views used repeatedly while auditing many cases."""

    cached = frame.attrs.get("v3_panel_arrays")
    if isinstance(cached, dict):
        return cached
    arrays: dict[str, np.ndarray] = {
        "trading_day": frame["trading_day"].to_numpy(),
        "timestamps": frame["timestamps"].to_numpy(),
        "volume": frame["volume"].to_numpy(dtype=np.float64),
        "close": frame["close"].to_numpy(dtype=np.float64),
    }
    frame.attrs["v3_panel_arrays"] = arrays
    return arrays


def _last_five_day_volume_ratio(
    target_frame: DataFrame,
    target_context_start: int,
    target_context_end: int,
    neighbor_frame: DataFrame,
    neighbor_context_start: int,
    neighbor_context_end: int,
) -> tuple[float | None, str | None]:
    """Compute only when the last five context trading days are complete/aligned.

    This intentionally works on positional slices instead of allocating five
    pandas filters for every candidate case.  Coverage audits may contain many
    thousands of cases, while each case must still preserve the exact same
    missing-bar rejection rule.
    """

    target_arrays = _frame_arrays(target_frame)
    neighbor_arrays = _frame_arrays(neighbor_frame)
    target_days = target_arrays["trading_day"]
    neighbor_days = neighbor_arrays["trading_day"]
    target_timestamps = target_arrays["timestamps"]
    neighbor_timestamps = neighbor_arrays["timestamps"]
    target_volume = target_arrays["volume"]
    neighbor_volume = neighbor_arrays["volume"]
    target_end = target_context_end
    neighbor_end = neighbor_context_end
    target_total = 0.0
    neighbor_total = 0.0
    for _ in range(5):
        if target_end <= target_context_start or neighbor_end <= neighbor_context_start:
            return None, "term_state_requires_five_complete_days"
        target_day = target_days[target_end - 1]
        neighbor_day = neighbor_days[neighbor_end - 1]
        target_start = target_end - 1
        neighbor_start = neighbor_end - 1
        while target_start > target_context_start and target_days[target_start - 1] == target_day:
            target_start -= 1
        while neighbor_start > neighbor_context_start and neighbor_days[neighbor_start - 1] == neighbor_day:
            neighbor_start -= 1
        target_count = target_end - target_start
        neighbor_count = neighbor_end - neighbor_start
        if (
            target_count not in VALID_TRADING_DAY_BAR_COUNTS
            or neighbor_count != target_count
            or target_day != neighbor_day
        ):
            return None, "term_state_requires_five_complete_days"
        if not np.array_equal(
            target_timestamps[target_start:target_end],
            neighbor_timestamps[neighbor_start:neighbor_end],
        ):
            return None, "term_state_requires_five_complete_days"
        target_total += float(target_volume[target_start:target_end].sum())
        neighbor_total += float(neighbor_volume[neighbor_start:neighbor_end].sum())
        target_end = target_start
        neighbor_end = neighbor_start
    return float(np.log((neighbor_total + 1e-12) / (target_total + 1e-12))), None


def _pair_context_structure_reason(
    target_frame: DataFrame,
    target_context_start: int,
    target_context_end: int,
    neighbor_frame: DataFrame,
    neighbor_context_start: int,
    neighbor_context_end: int,
) -> str | None:
    """Reject a pair if a fully represented context day has a missing/extra bar."""

    target_days = _frame_arrays(target_frame)["trading_day"]
    neighbor_days = _frame_arrays(neighbor_frame)["trading_day"]
    target_slice = target_days[target_context_start:target_context_end]
    neighbor_slice = neighbor_days[neighbor_context_start:neighbor_context_end]
    if not np.array_equal(target_slice, neighbor_slice):
        return "nearest_neighbor_context_trading_day_mismatch"
    cursor = target_context_start
    while cursor < target_context_end:
        trading_day = target_days[cursor]
        group_end = cursor + 1
        while group_end < target_context_end and target_days[group_end] == trading_day:
            group_end += 1
        full_group_start = cursor == 0 or target_days[cursor - 1] != trading_day
        full_group_end = group_end == len(target_days) or target_days[group_end] != trading_day
        if full_group_start and full_group_end and group_end - cursor not in VALID_TRADING_DAY_BAR_COUNTS:
            return "selected_stream_has_nonstandard_context_day"
        cursor = group_end
    return None


def _select_nearest_pair(
    archive: PanelArchive,
    *,
    target_contract: ConcreteContract,
    target_context_start: int,
    target_context_end: int,
    resolution: PanelResolution,
    lookback: int,
) -> tuple[
    str | None,
    ConcreteContract | None,
    int | None,
    TermStructureState | None,
    str | None,
]:
    if resolution.snapshot is None:
        return None, None, None, None, resolution.source
    candidates = [
        contract_id
        for contract_id in resolution.snapshot.contracts_by_product.get(target_contract.product, ())
        if contract_id != target_contract.contract_id
    ]
    if not candidates:
        return None, None, None, None, "no_same_product_neighbor_in_manifest"

    def rank(contract_id: str) -> tuple[int, int, str]:
        distance = delivery_month_distance(target_contract.contract_id, contract_id)
        # At equal absolute distance, the later delivery month wins.
        return (abs(distance), 0 if distance > 0 else 1, contract_id)

    neighbor_id = min(candidates, key=rank)
    neighbor_contract = archive.contracts.get(neighbor_id)
    if neighbor_contract is None:
        return neighbor_id, None, None, None, "nearest_neighbor_payload_unavailable"
    target_arrays = _frame_arrays(target_contract.frame)
    neighbor_arrays = _frame_arrays(neighbor_contract.frame)
    neighbor_positions = _last_context_ending_at(
        neighbor_contract.frame,
        pd.Timestamp(target_arrays["timestamps"][target_context_end - 1]),
        lookback,
    )
    if neighbor_positions is None:
        return neighbor_id, None, None, None, "nearest_neighbor_lacks_context"
    neighbor_start, neighbor_end = neighbor_positions
    if not np.array_equal(
        target_arrays["timestamps"][target_context_start:target_context_end],
        neighbor_arrays["timestamps"][neighbor_start:neighbor_end],
    ):
        return (
            neighbor_id,
            None,
            None,
            None,
            "nearest_neighbor_context_timestamp_mismatch",
        )
    structure_reason = _pair_context_structure_reason(
        target_contract.frame,
        target_context_start,
        target_context_end,
        neighbor_contract.frame,
        neighbor_start,
        neighbor_end,
    )
    if structure_reason is not None:
        return neighbor_id, None, None, None, structure_reason
    target_close = float(target_arrays["close"][target_context_end - 1])
    neighbor_close = float(neighbor_arrays["close"][neighbor_end - 1])
    if target_close <= 0.0 or neighbor_close <= 0.0:
        return neighbor_id, None, None, None, "non_positive_close_for_term_structure"
    volume_ratio, volume_reason = _last_five_day_volume_ratio(
        target_contract.frame,
        target_context_start,
        target_context_end,
        neighbor_contract.frame,
        neighbor_start,
        neighbor_end,
    )
    if volume_reason is not None:
        return neighbor_id, None, None, None, volume_reason
    assert volume_ratio is not None
    return (
        neighbor_id,
        neighbor_contract,
        neighbor_start,
        TermStructureState(
            log_close_ratio=float(np.log(neighbor_close / target_close)),
            signed_month_distance=delivery_month_distance(
                target_contract.contract_id,
                neighbor_id,
            ),
            log_volume_ratio_5d=volume_ratio,
        ),
        None,
    )


def _target_day_groups(frame: DataFrame) -> list[tuple[pd.Timestamp, np.ndarray]]:
    return [
        (
            pd.Timestamp(day).normalize(),
            group.index.to_numpy(dtype=np.int64),
        )
        for day, group in frame.groupby("trading_day", sort=False)
    ]


def build_panel_cases(
    archive: PanelArchive,
    *,
    lookback: int,
    products: Iterable[str] | None = None,
    allow_retrospective_partial: bool = True,
) -> PanelCaseBundle:
    """Build three-complete-day target cases and strict/exploratory P1 pairs."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    selected_products = (
        {str(product).lower() for product in products}
        if products is not None
        else {contract.product for contract in archive.contracts.values()}
    )
    unknown_products = selected_products.difference(
        {contract.product for contract in archive.contracts.values()}
    )
    if unknown_products:
        raise ValueError(f"No loaded data for products: {sorted(unknown_products)!r}")

    target_cases: list[PanelCase] = []
    strict_pair_cases: list[PanelCase] = []
    partial_pair_cases: list[PanelCase] = []
    target_rejections: Counter[str] = Counter()
    pair_rejections: Counter[str] = Counter()

    for target_contract in sorted(
        archive.contracts.values(), key=lambda contract: (contract.product, contract.contract_id)
    ):
        if target_contract.product not in selected_products:
            continue
        frame = target_contract.frame
        day_groups = _target_day_groups(frame)
        for offset in range(max(len(day_groups) - 2, 0)):
            selected_groups = day_groups[offset : offset + 3]
            target_days = tuple(day for day, _ in selected_groups)
            bar_counts = tuple(len(indices) for _, indices in selected_groups)
            if any(count not in VALID_TRADING_DAY_BAR_COUNTS for count in bar_counts):
                target_rejections["target_requires_three_complete_days"] += 1
                continue
            target_indices = np.concatenate([indices for _, indices in selected_groups])
            target_start = int(target_indices[0])
            if target_start < lookback:
                target_rejections["target_lacks_context"] += 1
                continue
            target_context_start = target_start - lookback
            day_end_indices = tuple(int(value) for value in np.cumsum(bar_counts) - 1)
            origin_timestamp = pd.Timestamp(frame["timestamps"].iloc[target_start - 1])
            origin_trading_day = pd.Timestamp(
                frame["trading_day"].iloc[target_start - 1]
            ).normalize()
            resolution = resolve_panel_at(
                archive,
                origin_timestamp,
                allow_retrospective_partial=allow_retrospective_partial,
            )
            (
                neighbor_id,
                neighbor_contract,
                neighbor_context_start,
                term_structure,
                pair_reason,
            ) = _select_nearest_pair(
                archive,
                target_contract=target_contract,
                target_context_start=target_context_start,
                target_context_end=target_start,
                resolution=resolution,
                lookback=lookback,
            )
            case = PanelCase(
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
                panel_snapshot_id=(
                    None if resolution.snapshot is None else resolution.snapshot.snapshot_id
                ),
                panel_source=resolution.source,
                partial_panel=resolution.partial_panel,
                nearest_neighbor_id=neighbor_id,
                neighbor_contract=neighbor_contract,
                neighbor_context_start=neighbor_context_start,
                term_structure=term_structure,
                pair_rejection_reason=pair_reason,
            )
            target_cases.append(case)
            if case.has_pair:
                if case.partial_panel:
                    partial_pair_cases.append(case)
                else:
                    strict_pair_cases.append(case)
            else:
                pair_rejections[pair_reason or "unknown_pair_rejection"] += 1

    target_cases.sort(key=lambda case: (case.target_end_day, case.target_contract_id, case.origin_timestamp))
    strict_pair_cases.sort(key=lambda case: (case.target_end_day, case.target_contract_id, case.origin_timestamp))
    partial_pair_cases.sort(key=lambda case: (case.target_end_day, case.target_contract_id, case.origin_timestamp))
    return PanelCaseBundle(
        lookback=lookback,
        target_cases=tuple(target_cases),
        strict_pair_cases=tuple(strict_pair_cases),
        partial_pair_cases=tuple(partial_pair_cases),
        target_rejections=dict(sorted(target_rejections.items())),
        pair_rejections=dict(sorted(pair_rejections.items())),
    )


def build_expanding_walk_forward_folds(
    target_end_days: Sequence[pd.Timestamp],
    *,
    minimum_fit_days: int,
    inner_validation_days: int,
    evaluation_days: int,
    step_days: int,
    purge_days: int = 3,
) -> tuple[V3WalkForwardFold, ...]:
    """Create global-calendar folds using target completion day as the split key."""

    if min(minimum_fit_days, inner_validation_days, evaluation_days, step_days) < 1:
        raise ValueError("walk-forward day counts must be positive")
    if purge_days < 3:
        raise ValueError("V3 requires at least a three-trading-day purge")
    days = sorted({_local_naive_timestamp(day).normalize() for day in target_end_days})
    first_evaluation_index = minimum_fit_days + purge_days + inner_validation_days + purge_days
    folds: list[V3WalkForwardFold] = []
    evaluation_start = first_evaluation_index
    fold_index = 0
    while evaluation_start + evaluation_days <= len(days):
        fit_end_index = evaluation_start - purge_days - inner_validation_days - purge_days - 1
        inner_start_index = fit_end_index + purge_days + 1
        inner_end_index = inner_start_index + inner_validation_days - 1
        evaluation_end_index = evaluation_start + evaluation_days - 1
        folds.append(
            V3WalkForwardFold(
                fold_id=f"fold_{fold_index:02d}",
                fit_start_day=days[0],
                fit_end_day=days[fit_end_index],
                inner_validation_start_day=days[inner_start_index],
                inner_validation_end_day=days[inner_end_index],
                evaluation_start_day=days[evaluation_start],
                evaluation_end_day=days[evaluation_end_index],
                purge_days=purge_days,
            )
        )
        fold_index += 1
        evaluation_start += step_days
    return tuple(folds)


def _pair_direction_counts(cases: Sequence[PanelCase]) -> dict[str, int]:
    counts = {"earlier": 0, "later": 0}
    for case in cases:
        delta = case.neighbor_delta_month
        if delta is None:
            continue
        if delta < 0:
            counts["earlier"] += 1
        elif delta > 0:
            counts["later"] += 1
    return counts


def _case_counts_by_product(cases: Sequence[PanelCase]) -> dict[str, int]:
    return dict(sorted(Counter(case.product for case in cases).items()))


def _fold_case_count(cases: Sequence[PanelCase], start_day: pd.Timestamp, end_day: pd.Timestamp) -> tuple[PanelCase, ...]:
    return tuple(
        case
        for case in cases
        if start_day <= case.target_end_day <= end_day
    )


def summarize_case_bundle(
    bundle: PanelCaseBundle,
    *,
    folds: Sequence[V3WalkForwardFold] = (),
) -> dict[str, object]:
    """Summarize counts needed to freeze a V3 context length before training."""

    strict = bundle.strict_pair_cases
    partial = bundle.partial_pair_cases
    output: dict[str, object] = {
        "lookback": bundle.lookback,
        "target_cases": len(bundle.target_cases),
        "strict_pair_cases": len(strict),
        "exploratory_partial_pair_cases": len(partial),
        "target_cases_by_product": _case_counts_by_product(bundle.target_cases),
        "strict_pair_cases_by_product": _case_counts_by_product(strict),
        "exploratory_partial_pair_cases_by_product": _case_counts_by_product(partial),
        "strict_pair_cases_by_neighbor_direction": _pair_direction_counts(strict),
        "exploratory_partial_pair_cases_by_neighbor_direction": _pair_direction_counts(partial),
        "target_rejections": dict(bundle.target_rejections),
        "pair_rejections": dict(bundle.pair_rejections),
        "folds": [],
    }
    fold_summaries: list[dict[str, object]] = []
    for fold in folds:
        strict_cases = _fold_case_count(
            strict, fold.evaluation_start_day, fold.evaluation_end_day
        )
        partial_cases = _fold_case_count(
            partial, fold.evaluation_start_day, fold.evaluation_end_day
        )
        target_cases = _fold_case_count(
            bundle.target_cases, fold.evaluation_start_day, fold.evaluation_end_day
        )
        fold_summaries.append(
            {
                "fold_id": fold.fold_id,
                "fit_end_day": fold.fit_end_day.isoformat(),
                "inner_validation": {
                    "start_day": fold.inner_validation_start_day.isoformat(),
                    "end_day": fold.inner_validation_end_day.isoformat(),
                },
                "evaluation": {
                    "start_day": fold.evaluation_start_day.isoformat(),
                    "end_day": fold.evaluation_end_day.isoformat(),
                },
                "target_cases": len(target_cases),
                "strict_pair_cases": len(strict_cases),
                "exploratory_partial_pair_cases": len(partial_cases),
                "strict_pair_cases_by_neighbor_direction": _pair_direction_counts(
                    strict_cases
                ),
            }
        )
    output["folds"] = fold_summaries
    return output


def build_coverage_audit(
    archive: PanelArchive,
    *,
    lookbacks: Sequence[int] = (256, 512),
    products: Iterable[str] | None = None,
    minimum_fit_days: int = 60,
    inner_validation_days: int = 20,
    evaluation_days: int = 20,
    step_days: int = 20,
    purge_days: int = 3,
) -> tuple[dict[str, object], Mapping[int, PanelCaseBundle], tuple[V3WalkForwardFold, ...]]:
    """Audit 256/512 pair coverage and recommend one primary context length.

    The recommendation follows the V3 rule literally: 512 is primary only if
    every fold retains at least 80% of *strict* 256 pair cases and both neighbour
    directions are non-empty.  A partial snapshot therefore cannot accidentally
    qualify a context length for the final experiment.
    """

    normalized_lookbacks = tuple(sorted({int(value) for value in lookbacks}))
    if normalized_lookbacks != (256, 512):
        raise ValueError("V3 coverage audit must compare exactly lookbacks 256 and 512")
    bundles = {
        lookback: build_panel_cases(
            archive,
            lookback=lookback,
            products=products,
            allow_retrospective_partial=True,
        )
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
        str(lookback): summarize_case_bundle(bundles[lookback], folds=folds)
        for lookback in normalized_lookbacks
    }

    fold_decisions: list[dict[str, object]] = []
    all_512_conditions = bool(folds)
    for fold in folds:
        cases_256 = _fold_case_count(
            bundles[256].strict_pair_cases,
            fold.evaluation_start_day,
            fold.evaluation_end_day,
        )
        cases_512 = _fold_case_count(
            bundles[512].strict_pair_cases,
            fold.evaluation_start_day,
            fold.evaluation_end_day,
        )
        count_256 = len(cases_256)
        count_512 = len(cases_512)
        retention = None if count_256 == 0 else count_512 / count_256
        directions = _pair_direction_counts(cases_512)
        condition = (
            retention is not None
            and retention >= 0.80
            and directions["earlier"] > 0
            and directions["later"] > 0
        )
        all_512_conditions = all_512_conditions and condition
        fold_decisions.append(
            {
                "fold_id": fold.fold_id,
                "strict_pair_cases_256": count_256,
                "strict_pair_cases_512": count_512,
                "retention_512_vs_256": retention,
                "directions_512": directions,
                "passes_512_primary_rule": condition,
            }
        )
    if not folds:
        rationale = "No complete walk-forward folds are available for a strict 256/512 comparison."
    elif not bundles[256].strict_pair_cases:
        rationale = (
            "No strict pair cases are available; partial-panel cases remain exploratory "
            "and cannot choose the primary context length."
        )
    elif all_512_conditions:
        rationale = "All folds satisfy the pre-registered 512 coverage and direction rule."
    else:
        rationale = "At least one fold fails the pre-registered 512 coverage or direction rule."
    audit: dict[str, object] = {
        "schema_version": 1,
        "task": "active_concrete_contract_panel_v3",
        "archive_root": str(archive.root),
        "snapshot_audit": dict(archive.source_audit),
        "loaded_contracts": {
            contract_id: {
                "product": contract.product,
                "bars": len(contract.frame),
                "first_available_bar": contract.first_timestamp.isoformat(),
                "last_available_bar": contract.last_timestamp.isoformat(),
                "source_snapshots": list(contract.source_snapshot_ids),
            }
            for contract_id, contract in sorted(archive.contracts.items())
        },
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
            "primary_context_length": 512 if all_512_conditions else 256,
            "rule": "512 requires >=80% strict-pair retention and both directions in every fold",
            "folds": fold_decisions,
            "rationale": rationale,
        },
    }
    return audit, bundles, folds


def case_arrays(case: PanelCase, *, include_neighbor: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return raw context features/stamps without introducing a 12-feature stream."""

    target = add_time_features(case.target_context)
    target_values = target[MODEL_FEATURES].to_numpy(dtype=np.float64)
    target_stamps = target[list(TIME_FEATURES)].to_numpy(dtype=np.float32)
    if not include_neighbor:
        return target_values, target_stamps, None, None
    if not case.has_pair or case.neighbor_context is None:
        raise PanelDataError(f"Case has no usable pair: {case.case_key}")
    neighbor = add_time_features(case.neighbor_context)
    neighbor_values = neighbor[MODEL_FEATURES].to_numpy(dtype=np.float64)
    neighbor_stamps = neighbor[list(TIME_FEATURES)].to_numpy(dtype=np.float32)
    if not np.array_equal(target["timestamps"].to_numpy(), neighbor["timestamps"].to_numpy()):
        raise PanelDataError(f"Pair timestamps no longer align: {case.case_key}")
    return target_values, target_stamps, neighbor_values, neighbor_stamps


def normalize_context(values: np.ndarray, *, clip: float = 5.0, epsilon: float = 1e-5) -> np.ndarray:
    """Normalize a single contract stream using only its own context."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(MODEL_FEATURES):
        raise ValueError("Expected [time, 6] contract values")
    if len(array) == 0:
        raise ValueError("Cannot normalize an empty contract context")
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    return np.clip((array - mean) / (std + epsilon), -clip, clip).astype(np.float32)
