"""Small, timeout-aware client for the provider's K-line TCP protocol.

The provider does not reliably return an error for every invalid request.  In
particular, a date beyond the current trading date can leave ``recv`` waiting
forever.  Callers should therefore use this client with a socket timeout and,
for collection jobs, place each request in its own process.
"""

from __future__ import annotations

import json
import socket
import zlib
from typing import Any


class KlineClientError(RuntimeError):
    """The provider connection closed or returned an invalid protocol frame."""


class klineclient:
    """Client retained under its original public name for compatibility."""

    def __init__(self, *, socket_timeout: float | None = 15.0, verbose: bool = False) -> None:
        if socket_timeout is not None and socket_timeout <= 0:
            raise ValueError("socket_timeout must be positive or None")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(socket_timeout)
        self.cache: list[bytes] = []
        self.verbose = verbose

    def __enter__(self) -> "klineclient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            # Destructors must not surface an exception during interpreter exit.
            pass

    def connect(self, addr: str, port: int) -> None:
        self.socket.connect((addr, port))

    def close(self) -> None:
        self.socket.close()

    def processdata(self, data: bytearray) -> tuple[bool, bytes, bytearray]:
        """Consume complete response packets from ``data``.

        A response can be split across TCP reads and across provider packets.
        The returned residual buffer must be supplied on the next invocation.
        """

        while True:
            if len(data) < 8:
                break
            ver = int.from_bytes(data[:1], "little")
            funcid = int.from_bytes(data[1:5], "little")
            payload_length = int.from_bytes(data[5:7], "little")
            compressed = int.from_bytes(data[7:8], "little")
            package_length = 8 + payload_length + 2
            if len(data) < package_length:
                break
            if payload_length < 16:
                raise KlineClientError(
                    f"Provider packet payload is too short: {payload_length} bytes"
                )

            packet = bytes(data[:package_length])
            if packet[-1] != 3:
                raise KlineClientError("Provider packet has an invalid terminator")
            checksum = 0
            for value in packet[8 : 8 + payload_length]:
                checksum ^= value
            if packet[-2] != checksum:
                raise KlineClientError("Provider packet has an invalid checksum")

            source_length = int.from_bytes(packet[8:12], "little")
            compressed_length = int.from_bytes(packet[12:16], "little")
            current_sequence = int.from_bytes(packet[16:18], "little")
            maximum_sequence = int.from_bytes(packet[18:20], "little")
            request_id = int.from_bytes(packet[20:24], "little")
            if self.verbose:
                print(
                    "provider packet "
                    f"ver={ver} funcid={funcid} compressed={compressed} "
                    f"source_length={source_length} compressed_length={compressed_length} "
                    f"sequence={current_sequence}/{maximum_sequence} request_id={request_id}"
                )

            self.cache.append(packet)
            data = data[package_length:]
            if current_sequence == maximum_sequence:
                compressed_payload = b"".join(chunk[24:-2] for chunk in self.cache)
                self.cache.clear()
                try:
                    decoded = zlib.decompress(compressed_payload)
                except zlib.error as exc:
                    raise KlineClientError("Provider response cannot be decompressed") from exc
                return True, decoded, data

        return False, b"", data

    @staticmethod
    def _request_packet(function_id: int, payload: dict[str, Any]) -> bytes:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(encoded) > 0xFFFF:
            raise ValueError("Provider request payload exceeds the protocol limit")
        checksum = 0
        for value in encoded:
            checksum ^= value
        return b"".join(
            (
                (2).to_bytes(1, "little"),
                function_id.to_bytes(4, "little"),
                len(encoded).to_bytes(2, "little"),
                (0).to_bytes(1, "little"),
                encoded,
                checksum.to_bytes(1, "little"),
                (3).to_bytes(1, "little"),
            )
        )

    def _request(self, function_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        self.cache.clear()
        self.socket.sendall(self._request_packet(function_id, payload))
        buffer = bytearray()
        while True:
            try:
                chunk = self.socket.recv(64 * 1024)
            except socket.timeout as exc:
                raise TimeoutError("Timed out while waiting for the K-line provider") from exc
            if not chunk:
                raise KlineClientError("K-line provider closed the connection before responding")
            buffer.extend(chunk)
            complete, decoded, buffer = self.processdata(buffer)
            if not complete:
                continue
            try:
                response = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KlineClientError("Provider returned invalid JSON") from exc
            if not isinstance(response, dict):
                raise KlineClientError("Provider response JSON must be an object")
            return response

    def reqhistorydata(
        self,
        reqid: int,
        instrumentid: str,
        /,
        cycletype: int = 1,
        startdate: int = 0,
        starttime: int = 0,
        enddate: int = 0,
        endtime: int = 0,
    ) -> dict[str, Any]:
        """Query historical K-lines within an inclusive provider date range."""

        return self._request(
            1002,
            {
                "nRequestID": reqid,
                "GlobalID": 0,
                "ExchangeID": "",
                "InstrumentID": instrumentid,
                "CycleType": cycletype,
                "StartDate": startdate,
                "StartTime": starttime,
                "EndDate": enddate,
                "EndTime": endtime,
            },
        )

    def reqhistorydatabynum(
        self,
        reqid: int,
        instrumentid: str,
        /,
        cycletype: int = 1,
        qrynum: int = 4000,
        enddate: int = 0,
        endtime: int = 0,
    ) -> dict[str, Any]:
        """Query up to ``qrynum`` K-lines ending at the provider cutoff."""

        if qrynum < 1:
            raise ValueError("qrynum must be positive")
        return self._request(
            1003,
            {
                "nRequestID": reqid,
                "GlobalID": 0,
                "ExchangeID": "",
                "InstrumentID": instrumentid,
                "CycleType": cycletype,
                "QryNum": qrynum,
                "EndDate": enddate,
                "EndTime": endtime,
            },
        )


# A PEP 8 alias for new callers; the lower-case name is part of the existing API.
KlineClient = klineclient
