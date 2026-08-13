"""Archive hourly K-lines for the current concrete-contract panel.

The collection format deliberately keeps the provider response for every
contract separate.  It does not create a continuous contract and it does not
merge contracts into a wider feature vector.  A snapshot is immutable: every
run writes a fresh directory containing the raw payloads and an active-contract
manifest that can later be used to distinguish complete and partial panels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as multiprocessing
import re
import sys
import time as monotonic_time
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from csj.utils.kline_client import klineclient
from csj.utils.tool import d_to_df


SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_HOST = "192.168.1.40"
DEFAULT_PORT = 8891
DEFAULT_CYCLE_TYPE = 2  # Provider convention: hourly bars.
DEFAULT_QUERY_BARS = 5000

# This is the user-supplied complete active list at the snapshot origin.  Keep
# it here rather than deriving it retrospectively from volume or a main-contract
# rule, which would leak future information into panel construction.
ACTIVE_CONTRACTS: dict[str, tuple[str, ...]] = {
    "rb": (
        "rb2608",
        "rb2609",
        "rb2610",
        "rb2611",
        "rb2612",
        "rb2701",
        "rb2702",
        "rb2703",
        "rb2704",
    ),
    "i": (
        "i2608",
        "i2609",
        "i2610",
        "i2611",
        "i2612",
        "i2701",
        "i2702",
        "i2703",
        "i2704",
    ),
    "jm": (
        "jm2608",
        "jm2609",
        "jm2610",
        "jm2611",
        "jm2612",
        "jm2701",
        "jm2702",
        "jm2703",
        "jm2704",
    ),
    "j": (
        "j2608",
        "j2609",
        "j2610",
        "j2611",
        "j2612",
        "j2701",
        "j2702",
        "j2703",
        "j2704",
    ),
}

# Candidate months beyond the original user-supplied snapshot universe.  These
# are queried only while their delivery month has not passed.  A successful
# provider response is evidence that the concrete contract is available; an
# empty/timeout response remains an explicit failed manifest row.  The list is
# intentionally finite so collection cannot silently enumerate old contracts.
EXPANDED_CANDIDATE_CONTRACTS: dict[str, tuple[str, ...]] = {
    product: tuple(f"{product}27{month:02d}" for month in range(5, 13))
    for product in ("rb", "i", "jm", "j")
}

_CONTRACT_PATTERN = re.compile(r"^(?P<product>[a-z]+)(?P<year>\d{2})(?P<month>\d{2})$")


class ActiveContractDataError(RuntimeError):
    """Collection or raw-payload validation failed for one concrete contract."""


@dataclass(frozen=True)
class ProviderCutoff:
    """A date/time cutoff proven not to lie after the current Shanghai time."""

    requested_at: datetime
    effective_at: datetime

    @property
    def was_clamped(self) -> bool:
        return self.requested_at != self.effective_at

    @property
    def end_date(self) -> int:
        return int(self.effective_at.strftime("%Y%m%d"))

    @property
    def end_time(self) -> int:
        return int(self.effective_at.strftime("%H%M%S"))

    def to_manifest(self) -> dict[str, object]:
        return {
            "requested_at": self.requested_at.isoformat(),
            "effective_at": self.effective_at.isoformat(),
            "was_clamped": self.was_clamped,
            "provider_end_date": self.end_date,
            "provider_end_time": self.end_time,
        }


@dataclass(frozen=True)
class FetchSettings:
    host: str
    port: int
    cycle_type: int
    query_bars: int
    socket_timeout_seconds: float
    process_timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be in 1..65535")
        if self.cycle_type < 1:
            raise ValueError("cycle_type must be positive")
        if self.query_bars < 1:
            raise ValueError("query_bars must be positive")
        if self.socket_timeout_seconds <= 0:
            raise ValueError("socket_timeout_seconds must be positive")
        if self.process_timeout_seconds <= 0:
            raise ValueError("process_timeout_seconds must be positive")
        if self.socket_timeout_seconds > self.process_timeout_seconds:
            raise ValueError("socket timeout cannot exceed process timeout")


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_dir: Path
    manifest_path: Path
    active_contracts_path: Path
    succeeded: int
    failed: int


def parse_delivery_year_month(contract_id: str) -> tuple[int, int]:
    """Return a concrete contract's delivery year and month, not string order."""

    normalized = contract_id.strip().lower()
    match = _CONTRACT_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"Invalid concrete contract ID: {contract_id!r}")
    month = int(match.group("month"))
    if month < 1 or month > 12:
        raise ValueError(f"Invalid delivery month in contract ID: {contract_id!r}")
    return 2000 + int(match.group("year")), month


def delivery_year_month(contract_id: str) -> str:
    year, month = parse_delivery_year_month(contract_id)
    return f"{year:04d}-{month:02d}"


def _ensure_shanghai(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(SHANGHAI).replace(microsecond=0)
    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI, microsecond=0)
    return now.astimezone(SHANGHAI).replace(microsecond=0)


def _parse_end_date(value: date | datetime | str | int | None, *, default: date) -> date:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    digits = str(value).strip()
    if not re.fullmatch(r"\d{8}", digits):
        raise ValueError("end date must use YYYYMMDD")
    return datetime.strptime(digits, "%Y%m%d").date()


def _parse_end_time(value: time | str | int | None, *, default: time) -> time:
    if value is None:
        return default
    if isinstance(value, time):
        return value.replace(microsecond=0)
    digits = str(value).strip().replace(":", "")
    # The provider uses 0 to mean an unspecified time.  Treat it as the end of
    # the supplied date before applying the full timestamp safety clamp.
    if digits == "0":
        return time(23, 59, 59)
    if not re.fullmatch(r"\d{1,6}", digits):
        raise ValueError("end time must use HHMMSS")
    digits = digits.zfill(6)
    try:
        return time(int(digits[:2]), int(digits[2:4]), int(digits[4:6]))
    except ValueError as exc:
        raise ValueError("end time must use a valid HHMMSS value") from exc


def clamp_provider_cutoff(
    end_date: date | datetime | str | int | None = None,
    end_time: time | str | int | None = None,
    *,
    now: datetime | None = None,
) -> ProviderCutoff:
    """Clamp a requested cutoff to the current Shanghai timestamp.

    The service can remain silent when its requested end date is in the future.
    Clamping the full timestamp also avoids an accidental request for later in
    the current trading day.
    """

    current = _ensure_shanghai(now)
    requested_date = _parse_end_date(end_date, default=current.date())
    default_time = current.timetz().replace(tzinfo=None)
    if requested_date < current.date():
        default_time = time(23, 59, 59)
    requested_time = _parse_end_time(end_time, default=default_time)
    requested = datetime.combine(requested_date, requested_time, tzinfo=SHANGHAI)
    return ProviderCutoff(
        requested_at=requested,
        effective_at=min(requested, current),
    )


def contracts_for_products(
    products: Sequence[str] | None = None,
    *,
    include_candidates: bool = False,
    extra_contracts: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Return ordered concrete contracts, optionally including future candidates."""

    requested = tuple(product.lower() for product in (products or tuple(ACTIVE_CONTRACTS)))
    unknown = sorted(set(requested) - set(ACTIVE_CONTRACTS))
    if unknown:
        raise ValueError(f"Unknown products: {', '.join(unknown)}")
    pairs: list[tuple[str, str]] = []
    for product in requested:
        contracts = ACTIVE_CONTRACTS[product]
        if include_candidates:
            contracts = tuple(dict.fromkeys((*contracts, *EXPANDED_CANDIDATE_CONTRACTS[product])))
        if len(set(contracts)) != len(contracts):
            raise ValueError(f"Duplicate concrete contract in {product!r}")
        previous_maturity: tuple[int, int] | None = None
        for contract_id in contracts:
            normalized = contract_id.lower()
            match = _CONTRACT_PATTERN.fullmatch(normalized)
            if match is None or match.group("product") != product:
                raise ValueError(
                    f"Contract {contract_id!r} does not belong to product {product!r}"
                )
            maturity = parse_delivery_year_month(normalized)
            if previous_maturity is not None and maturity <= previous_maturity:
                raise ValueError(f"Contracts for {product!r} are not maturity ordered")
            previous_maturity = maturity
            pairs.append((product, normalized))
    seen = {contract_id for _, contract_id in pairs}
    for value in extra_contracts:
        normalized = str(value).strip().lower()
        match = _CONTRACT_PATTERN.fullmatch(normalized)
        if match is None:
            raise ValueError(f"Invalid explicit concrete contract ID: {value!r}")
        if normalized in seen:
            continue
        parse_delivery_year_month(normalized)
        pairs.append((match.group("product"), normalized))
        seen.add(normalized)
    return pairs


def reject_past_delivery_contracts(
    pairs: Sequence[tuple[str, str]], *, snapshot_at: datetime
) -> None:
    """Prevent the collector from probing a delivery month that has passed."""

    current_month = (snapshot_at.year, snapshot_at.month)
    past = [
        contract_id
        for _, contract_id in pairs
        if parse_delivery_year_month(contract_id) < current_month
    ]
    if past:
        raise ValueError(
            "Delivered/past delivery-month contracts cannot be queried: "
            + ", ".join(sorted(past))
        )


def _fetch_worker(
    result_queue: Any,
    contract_id: str,
    cutoff: ProviderCutoff,
    settings: FetchSettings,
) -> None:
    """Run one TCP query in a process that can be terminated by its parent."""

    client: klineclient | None = None
    try:
        client = klineclient(socket_timeout=settings.socket_timeout_seconds)
        client.connect(settings.host, settings.port)
        payload = client.reqhistorydatabynum(
            1,
            contract_id,
            cycletype=settings.cycle_type,
            qrynum=settings.query_bars,
            enddate=cutoff.end_date,
            endtime=cutoff.end_time,
        )
        result_queue.put({"ok": True, "payload": payload})
    except BaseException as exc:  # Marshal every provider failure back to the manifest.
        result_queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        if client is not None:
            try:
                client.close()
            except OSError:
                pass


def fetch_payload_in_isolated_process(
    contract_id: str,
    cutoff: ProviderCutoff,
    settings: FetchSettings,
) -> dict[str, Any]:
    """Fetch one payload with both socket and process-level time limits.

    There is intentionally no retry loop.  A failed request is evidence that is
    retained in the snapshot manifest, while the next listed contract can still
    be archived.
    """

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_fetch_worker,
        args=(result_queue, contract_id, cutoff, settings),
        daemon=False,
    )
    process.start()
    message: dict[str, Any] | None = None
    deadline = monotonic_time.monotonic() + settings.process_timeout_seconds
    deadline_exceeded = False
    try:
        while monotonic_time.monotonic() < deadline:
            remaining = deadline - monotonic_time.monotonic()
            try:
                candidate = result_queue.get(timeout=min(0.2, max(remaining, 0.01)))
            except Empty:
                if not process.is_alive() and process.exitcode is not None:
                    # Give Queue's feeder thread one short chance to flush its
                    # final result after the child has exited.
                    try:
                        candidate = result_queue.get(timeout=0.1)
                    except Empty:
                        break
                else:
                    continue
            if not isinstance(candidate, dict):
                raise ActiveContractDataError(
                    f"{contract_id}: worker returned a non-object result"
                )
            message = candidate
            break
        if message is None and monotonic_time.monotonic() >= deadline:
            deadline_exceeded = True
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        result_queue.close()
        result_queue.join_thread()

    if message is None:
        if deadline_exceeded:
            raise ActiveContractDataError(
                f"{contract_id}: request exceeded {settings.process_timeout_seconds:g}s"
            )
        raise ActiveContractDataError(
            f"{contract_id}: worker exited without a result (exit code {process.exitcode})"
        )
    if not message.get("ok"):
        error_type = str(message.get("error_type", "ProviderError"))
        error = str(message.get("error", "unknown provider failure"))
        raise ActiveContractDataError(f"{contract_id}: {error_type}: {error}")
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ActiveContractDataError(f"{contract_id}: provider payload is not an object")
    return payload


def audit_payload(payload: Mapping[str, Any], *, expected_contract: str) -> dict[str, object]:
    """Validate the raw response and return manifest-ready availability facts."""

    reported_contract = str(payload.get("Ins", "")).lower()
    if reported_contract != expected_contract.lower():
        raise ActiveContractDataError(
            f"{expected_contract}: provider identified payload as {reported_contract!r}"
        )
    if not isinstance(payload.get("data"), list) or not payload["data"]:
        raise ActiveContractDataError(f"{expected_contract}: provider returned no K-line bars")
    frame = d_to_df(None, dict(payload), persist_raw=False, validate=True)
    if len(frame) == 0:
        raise ActiveContractDataError(f"{expected_contract}: provider returned no usable K-line bars")

    trading_day_counts = frame.groupby("trading_day", sort=True).size()
    count_distribution = {
        str(int(bar_count)): int((trading_day_counts == bar_count).sum())
        for bar_count in sorted({int(value) for value in trading_day_counts.tolist()})
    }
    non_standard_days = [
        {
            "trading_day": day.strftime("%Y-%m-%d"),
            "bars": int(bar_count),
        }
        for day, bar_count in trading_day_counts.items()
        if int(bar_count) not in {5, 7}
    ]
    return {
        "bars": int(len(frame)),
        "trading_days": int(len(trading_day_counts)),
        "first_available_bar": frame["timestamps"].iloc[0].isoformat(),
        "last_available_bar": frame["timestamps"].iloc[-1].isoformat(),
        "trading_day_bar_count_distribution": count_distribution,
        # A current session can legitimately be incomplete.  Record it for the
        # later data audit instead of silently deleting bars during collection.
        "non_standard_trading_days": non_standard_days,
    }


def _write_new_json(path: Path, value: Any) -> str:
    """Create a JSON file once and return the checksum of its exact contents."""

    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as output:
        output.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_active_contract_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "snapshot_at",
        "product",
        "contract_id",
        "delivery_year_month",
        "first_available_bar",
        "last_available_bar",
        "status",
        "raw_payload_path",
        "bars",
    ]
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


FetchFunction = Callable[[str, ProviderCutoff, FetchSettings], dict[str, Any]]


def collect_active_contract_snapshot(
    *,
    output_root: str | Path,
    settings: FetchSettings,
    cutoff: ProviderCutoff,
    products: Sequence[str] | None = None,
    include_candidates: bool = False,
    extra_contracts: Sequence[str] = (),
    now: datetime | None = None,
    fetcher: FetchFunction = fetch_payload_in_isolated_process,
) -> SnapshotResult:
    """Fetch every selected concrete contract and archive one immutable snapshot."""

    snapshot_at = _ensure_shanghai(now)
    selected_contracts = contracts_for_products(
        products,
        include_candidates=include_candidates,
        extra_contracts=extra_contracts,
    )
    reject_past_delivery_contracts(selected_contracts, snapshot_at=snapshot_at)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_id = snapshot_at.strftime("%Y%m%dT%H%M%S%z")
    snapshot_dir = root / snapshot_id
    suffix = 1
    while True:
        try:
            snapshot_dir.mkdir()
            break
        except FileExistsError:
            # A second run in the same second is a new immutable snapshot, not
            # permission to replace the first run's raw provider responses.
            snapshot_dir = root / f"{snapshot_id}-{suffix:02d}"
            suffix += 1
    raw_dir = snapshot_dir / "raw"
    raw_dir.mkdir()

    records: list[dict[str, object]] = []
    for product, contract_id in selected_contracts:
        record: dict[str, object] = {
            "snapshot_at": snapshot_at.isoformat(),
            "product": product,
            "contract_id": contract_id,
            "delivery_year_month": delivery_year_month(contract_id),
            "first_available_bar": "",
            "last_available_bar": "",
            "status": "failed",
            "raw_payload_path": "",
            "bars": "",
        }
        try:
            payload = fetcher(contract_id, cutoff, settings)
            raw_path = raw_dir / f"kline_{contract_id}.json"
            checksum = _write_new_json(raw_path, payload)
            record["raw_payload_path"] = str(raw_path.relative_to(snapshot_dir))
            record["raw_payload_sha256"] = checksum
            record.update(audit_payload(payload, expected_contract=contract_id))
            record["status"] = "ok"
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
        records.append(record)

    succeeded = sum(record["status"] == "ok" for record in records)
    failed = len(records) - succeeded
    completed_at = datetime.now(SHANGHAI).replace(microsecond=0)
    manifest = {
        "schema_version": 1,
        "snapshot_at": snapshot_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "status": "complete" if failed == 0 else "partial",
        "panel_completeness": "complete" if failed == 0 else "partial_panel",
        "active_contract_list_source": (
            "user_supplied_active_contract_list_plus_explicit_candidates"
            if extra_contracts
            else "user_supplied_active_contract_list_plus_future_candidates"
            if include_candidates
            else "user_supplied_active_contract_list"
        ),
        "timezone": "Asia/Shanghai",
        "query": {
            "provider": {"host": settings.host, "port": settings.port},
            "cycle_type": settings.cycle_type,
            "query_bars": settings.query_bars,
            "socket_timeout_seconds": settings.socket_timeout_seconds,
            "process_timeout_seconds": settings.process_timeout_seconds,
            "cutoff": cutoff.to_manifest(),
            "request_isolation": "one fresh process and TCP connection per contract",
            "retry_policy": "no automatic retry",
            "included_future_candidates": include_candidates,
            "explicit_candidate_contracts": [
                str(value).strip().lower() for value in extra_contracts
            ],
        },
        "contracts": records,
    }
    manifest_path = snapshot_dir / "manifest.json"
    _write_new_json(manifest_path, manifest)
    active_contracts_path = snapshot_dir / "active_contracts.csv"
    _write_active_contract_csv(active_contracts_path, records)
    return SnapshotResult(
        snapshot_dir=snapshot_dir,
        manifest_path=manifest_path,
        active_contracts_path=active_contracts_path,
        succeeded=succeeded,
        failed=failed,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive hourly K-lines for the current concrete-contract panel."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "active_contract_snapshots",
        help="Parent directory for immutable timestamped snapshots.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="K-line provider host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="K-line provider port.")
    parser.add_argument(
        "--products",
        nargs="+",
        choices=tuple(ACTIVE_CONTRACTS),
        help="Optional subset of products; default archives all four products.",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Also probe the finite 2705-2712 candidate set; failures remain in the manifest.",
    )
    parser.add_argument(
        "--extra-contracts",
        nargs="+",
        default=(),
        help="Explicit additional non-past concrete contract IDs to probe.",
    )
    parser.add_argument(
        "--end-date",
        help="Requested provider cutoff as YYYYMMDD; future values are clamped to now.",
    )
    parser.add_argument(
        "--end-time",
        help="Requested provider cutoff as HHMMSS; future time today is clamped to now.",
    )
    parser.add_argument(
        "--query-bars",
        type=int,
        default=DEFAULT_QUERY_BARS,
        help="Maximum hourly bars requested per concrete contract.",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=15.0,
        help="Per-socket-operation timeout in seconds.",
    )
    parser.add_argument(
        "--process-timeout",
        type=float,
        default=30.0,
        help="Hard timeout for one isolated contract request in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    now = datetime.now(SHANGHAI).replace(microsecond=0)
    cutoff = clamp_provider_cutoff(args.end_date, args.end_time, now=now)
    settings = FetchSettings(
        host=args.host,
        port=args.port,
        cycle_type=DEFAULT_CYCLE_TYPE,
        query_bars=args.query_bars,
        socket_timeout_seconds=args.socket_timeout,
        process_timeout_seconds=args.process_timeout,
    )
    if cutoff.was_clamped:
        print(
            "Requested cutoff is after the current Shanghai time; "
            f"using {cutoff.effective_at.isoformat()} instead.",
            file=sys.stderr,
        )
    result = collect_active_contract_snapshot(
        output_root=args.output_root,
        settings=settings,
        cutoff=cutoff,
        products=args.products,
        include_candidates=args.include_candidates,
        extra_contracts=args.extra_contracts,
        now=now,
    )
    print(
        json.dumps(
            {
                "snapshot_dir": str(result.snapshot_dir),
                "manifest": str(result.manifest_path),
                "active_contracts": str(result.active_contracts_path),
                "succeeded": result.succeeded,
                "failed": result.failed,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
