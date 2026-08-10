from __future__ import annotations

import csv
import json
import zlib
from datetime import datetime
from pathlib import Path

import pytest

from csj.active_contract_data import (
    ACTIVE_CONTRACTS,
    SHANGHAI,
    ActiveContractDataError,
    FetchSettings,
    audit_payload,
    clamp_provider_cutoff,
    collect_active_contract_snapshot,
    contracts_for_products,
    delivery_year_month,
)
from csj.utils.kline_client import klineclient


NOW = datetime(2026, 8, 10, 11, 11, 55, tzinfo=SHANGHAI)


def _payload(contract_id: str) -> dict[str, object]:
    return {
        "Ins": contract_id,
        "data": [
            {
                "TeD": "20260807",
                "TiD": "20260807",
                "T": "14:00:00",
                "O": 100.0,
                "H": 102.0,
                "L": 99.0,
                "C": 101.0,
                "V": 10,
                "VD": 10,
                "A": 1000.0,
                "OI": 100,
            },
            {
                "TeD": "20260807",
                "TiD": "20260807",
                "T": "15:00:00",
                "O": 101.0,
                "H": 103.0,
                "L": 100.0,
                "C": 102.0,
                "V": 25,
                "VD": 15,
                "A": 2600.0,
                "OI": 101,
            },
        ],
    }


def _settings() -> FetchSettings:
    return FetchSettings(
        host="provider.test",
        port=8891,
        cycle_type=2,
        query_bars=5000,
        socket_timeout_seconds=5.0,
        process_timeout_seconds=10.0,
    )


def test_frozen_active_list_has_all_user_supplied_concrete_contracts() -> None:
    pairs = contracts_for_products()

    assert len(pairs) == 36
    assert set(ACTIVE_CONTRACTS) == {"rb", "i", "jm", "j"}
    assert pairs[0] == ("rb", "rb2608")
    assert pairs[-1] == ("j", "j2704")
    assert delivery_year_month("rb2612") == "2026-12"
    assert delivery_year_month("rb2701") == "2027-01"


def test_cutoff_clamps_future_date_and_future_time_to_current_shanghai_time() -> None:
    future_date = clamp_provider_cutoff("20260831", "210000", now=NOW)
    future_time = clamp_provider_cutoff("20260810", "210000", now=NOW)
    past_date = clamp_provider_cutoff("20260807", now=NOW)

    assert future_date.was_clamped
    assert future_date.effective_at == NOW
    assert future_date.end_date == 20260810
    assert future_date.end_time == 111155
    assert future_time.was_clamped
    assert future_time.effective_at == NOW
    assert not past_date.was_clamped
    assert past_date.end_time == 235959


def test_payload_audit_records_available_range_and_nonstandard_days() -> None:
    audit = audit_payload(_payload("rb2608"), expected_contract="rb2608")

    assert audit["bars"] == 2
    assert audit["first_available_bar"] == "2026-08-07T14:00:00"
    assert audit["last_available_bar"] == "2026-08-07T15:00:00"
    assert audit["trading_day_bar_count_distribution"] == {"2": 1}
    assert audit["non_standard_trading_days"] == [
        {"trading_day": "2026-08-07", "bars": 2}
    ]


def test_payload_audit_rejects_a_response_for_the_wrong_contract() -> None:
    with pytest.raises(ActiveContractDataError, match="provider identified payload"):
        audit_payload(_payload("rb2609"), expected_contract="rb2608")


def test_kline_client_reassembles_and_decompresses_a_provider_packet() -> None:
    decoded = json.dumps(_payload("rb2608"), ensure_ascii=False).encode("utf-8")
    compressed = zlib.compress(decoded)
    provider_header = b"".join(
        (
            len(decoded).to_bytes(4, "little"),
            len(compressed).to_bytes(4, "little"),
            (1).to_bytes(2, "little"),
            (1).to_bytes(2, "little"),
            (7).to_bytes(4, "little"),
        )
    )
    body = provider_header + compressed
    checksum = 0
    for value in body:
        checksum ^= value
    packet = b"".join(
        (
            (2).to_bytes(1, "little"),
            (2002).to_bytes(4, "little"),
            len(body).to_bytes(2, "little"),
            (1).to_bytes(1, "little"),
            body,
            checksum.to_bytes(1, "little"),
            (3).to_bytes(1, "little"),
        )
    )
    client = klineclient()
    try:
        complete, response, residual = client.processdata(bytearray(packet + b"tail"))
    finally:
        client.close()

    assert complete
    assert json.loads(response) == _payload("rb2608")
    assert residual == bytearray(b"tail")


def test_collection_creates_an_immutable_manifest_and_raw_payloads(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(contract_id: str, *_: object) -> dict[str, object]:
        calls.append(contract_id)
        return _payload(contract_id)

    cutoff = clamp_provider_cutoff(now=NOW)
    first = collect_active_contract_snapshot(
        output_root=tmp_path,
        settings=_settings(),
        cutoff=cutoff,
        products=["rb"],
        now=NOW,
        fetcher=fetcher,
    )
    second = collect_active_contract_snapshot(
        output_root=tmp_path,
        settings=_settings(),
        cutoff=cutoff,
        products=["rb"],
        now=NOW,
        fetcher=fetcher,
    )

    assert first.succeeded == 9
    assert first.failed == 0
    assert first.snapshot_dir != second.snapshot_dir
    assert calls == list(ACTIVE_CONTRACTS["rb"]) * 2
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["panel_completeness"] == "complete"
    assert manifest["query"]["cutoff"]["effective_at"] == NOW.isoformat()
    assert [entry["contract_id"] for entry in manifest["contracts"]] == list(
        ACTIVE_CONTRACTS["rb"]
    )
    assert (first.snapshot_dir / "raw" / "kline_rb2608.json").exists()
    with first.active_contracts_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert set(rows[0]) >= {
        "snapshot_at",
        "product",
        "contract_id",
        "delivery_year_month",
        "first_available_bar",
        "last_available_bar",
    }
