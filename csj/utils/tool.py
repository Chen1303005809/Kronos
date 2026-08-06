from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kronos-matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


MODEL_FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def _parse_payload(payload: dict[str, Any] | str) -> dict[str, Any]:
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(parsed, dict):
        raise TypeError("K-line payload must be a dictionary or a JSON object string")
    if not isinstance(parsed.get("data"), list):
        raise ValueError("K-line payload must contain a list-valued 'data' field")
    return parsed


def _validate_frame(frame: DataFrame) -> None:
    if frame.empty:
        raise ValueError("K-line payload contains no bars")
    if frame[MODEL_FEATURES + ["timestamps", "trading_day"]].isna().any().any():
        raise ValueError("K-line payload contains missing model fields")
    if frame["timestamps"].duplicated().any():
        duplicates = frame.loc[frame["timestamps"].duplicated(), "timestamps"].tolist()
        raise ValueError(f"Duplicate real timestamps found: {duplicates[:3]}")
    if not frame["timestamps"].is_monotonic_increasing:
        raise ValueError("Real timestamps are not monotonically increasing")
    if (frame[["volume", "amount"]] < 0).any().any():
        raise ValueError("Per-bar volume and amount must be non-negative")

    high_floor = frame[["open", "close", "low"]].max(axis=1)
    low_ceiling = frame[["open", "close", "high"]].min(axis=1)
    if (frame["high"] < high_floor).any() or (frame["low"] > low_ceiling).any():
        raise ValueError("Invalid OHLC relationship found")


def d_to_df(
    path: str | Path | None,
    d: dict[str, Any] | str,
    *,
    persist_raw: bool = True,
    validate: bool = True,
) -> DataFrame:
    """Convert the provider payload into correctly ordered, per-bar K-line data.

    ``TeD + T`` is the real calendar timestamp. ``TiD`` is retained separately as
    the futures trading day, so a Friday night session can be grouped into the
    following Monday without being sorted after Monday's day session.

    The provider's ``V`` and ``A`` fields are cumulative within a trading day.
    ``VD`` is used as per-bar volume and per-bar amount is derived by differencing
    ``A`` within ``TiD``.

    ``path`` and ``persist_raw`` preserve the original helper's optional raw JSON
    persistence behavior. Pass ``path=None`` or ``persist_raw=False`` for a
    side-effect-free conversion.
    """

    payload = _parse_payload(d)
    instrument = str(payload.get("Ins", "unknown"))

    if path is not None and persist_raw:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"kline_{instrument}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    records: list[dict[str, Any]] = []
    for item in payload["data"]:
        records.append(
            {
                "instrument": instrument,
                "open": item.get("O"),
                "high": item.get("H"),
                "low": item.get("L"),
                "close": item.get("C"),
                "cumulative_volume": item.get("V"),
                "volume_delta": item.get("VD"),
                "cumulative_amount": item.get("A"),
                "open_interest": item.get("OI"),
                "calendar_day": item.get("TeD"),
                "trading_day_raw": item.get("TiD"),
                "bar_time": item.get("T"),
            }
        )

    frame = pd.DataFrame.from_records(records)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "cumulative_volume",
        "volume_delta",
        "cumulative_amount",
        "open_interest",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame["timestamps"] = pd.to_datetime(
        frame["calendar_day"].astype(str) + " " + frame["bar_time"].astype(str),
        format="%Y%m%d %H:%M:%S",
        errors="raise",
    )
    frame["trading_day"] = pd.to_datetime(
        frame["trading_day_raw"].astype(str),
        format="%Y%m%d",
        errors="raise",
    )
    frame = frame.sort_values("timestamps", kind="stable").reset_index(drop=True)

    grouped = frame.groupby("trading_day", sort=False)
    derived_volume = grouped["cumulative_volume"].diff()
    derived_amount = grouped["cumulative_amount"].diff()
    first_bar = grouped.cumcount().eq(0)
    derived_volume.loc[first_bar] = frame.loc[first_bar, "cumulative_volume"]
    derived_amount.loc[first_bar] = frame.loc[first_bar, "cumulative_amount"]

    frame["volume"] = frame["volume_delta"].fillna(derived_volume)
    frame["amount"] = derived_amount

    ordered_columns = [
        "instrument",
        "timestamps",
        "trading_day",
        *MODEL_FEATURES,
        "open_interest",
        "calendar_day",
        "bar_time",
        "cumulative_volume",
        "cumulative_amount",
    ]
    frame = frame[ordered_columns]
    frame.attrs["instrument"] = instrument

    if validate:
        _validate_frame(frame)
    return frame


def plot_prediction(kline_df: DataFrame, pred_df: DataFrame, save_path: str = "prediction_plot.png") -> None:
    history = kline_df.copy()
    prediction = pred_df.copy()
    prediction.index = history.index[-prediction.shape[0] :]

    close_df = pd.concat(
        [history["close"].rename("Ground Truth"), prediction["close"].rename("Prediction")],
        axis=1,
    )
    volume_df = pd.concat(
        [history["volume"].rename("Ground Truth"), prediction["volume"].rename("Prediction")],
        axis=1,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    if os.environ.get("DISPLAY") is None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
