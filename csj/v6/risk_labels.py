"""Past-scaled adverse-excursion labels with fit-only tail thresholds.

The context diagnostic seam accepts a :class:`TargetOnlyCase` but slices only
its pre-origin context.  Future OHLCVA enters only ``risk_outcome`` after the
past-only horizon scale has been fixed.  Tail-event thresholds are a separate
fit-only operation so callers cannot silently fold evaluation labels into the
label rule.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from csj.utils.tool import MODEL_FEATURES
from csj.v5.target_data import TargetOnlyCase
from csj.v6.config import RISK_LABEL_RULE_VERSION


class RiskLabelError(RuntimeError):
    """A case or threshold violates the frozen V6 label protocol."""


@dataclass(frozen=True)
class RiskLabelSpec:
    """All numerical choices that can change a V6 P0 risk label."""

    lookback: int
    horizon_trading_days: int
    valid_bar_counts: tuple[int, ...]
    volatility_bars: int
    volatility_halflife_bars: int
    volatility_adjust: bool
    volatility_bias: bool
    volatility_floor: float
    tail_quantile: float
    tail_quantile_method: str
    clip: float
    epsilon: float
    version: str = RISK_LABEL_RULE_VERSION

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RiskLabelSpec":
        data = config["data"]
        labels = config["risk_labels"]
        volatility = labels["context_volatility"]
        tail = labels["tail_event"]
        return cls(
            lookback=int(data["lookback"]),
            horizon_trading_days=int(data["horizon_trading_days"]),
            valid_bar_counts=tuple(int(value) for value in data["valid_bar_counts"]),
            volatility_bars=int(volatility["bars"]),
            volatility_halflife_bars=int(volatility["halflife_bars"]),
            volatility_adjust=bool(volatility["adjust"]),
            volatility_bias=bool(volatility["bias"]),
            volatility_floor=float(volatility["floor"]),
            tail_quantile=float(tail["quantile"]),
            tail_quantile_method=str(tail["quantile_method"]),
            clip=float(data["clip"]),
            epsilon=float(data["normalization_epsilon"]),
            version=str(labels["version"]),
        )

    def __post_init__(self) -> None:
        if self.version != RISK_LABEL_RULE_VERSION:
            raise ValueError(f"Unsupported V6 risk-label version: {self.version!r}")
        if self.lookback < self.volatility_bars + 1:
            raise ValueError("V6 lookback must provide one more close than volatility returns")
        if self.horizon_trading_days != 3:
            raise ValueError("V6 labels require exactly three future trading days")
        if self.valid_bar_counts != (5, 7):
            raise ValueError("V6 labels accept only five- or seven-bar trading days")
        if self.volatility_bars < 2 or self.volatility_halflife_bars <= 0:
            raise ValueError("V6 EWMA window and halflife must be positive")
        if not 0.0 < self.tail_quantile < 1.0:
            raise ValueError("V6 tail quantile must lie strictly between zero and one")
        if self.tail_quantile_method != "linear":
            raise ValueError("V6 fixes the tail quantile method at linear")
        if min(self.volatility_floor, self.clip, self.epsilon) <= 0.0:
            raise ValueError("V6 volatility floor, clip, and epsilon must be positive")


@dataclass(frozen=True)
class TailThresholds:
    """One outer fold's thresholds fitted solely on primary fit labels."""

    fold_id: str
    fit_start_day: pd.Timestamp
    fit_end_day: pd.Timestamp
    primary_products: tuple[str, ...]
    quantile: float
    quantile_method: str
    long_tail_threshold: float
    short_tail_threshold: float
    fit_case_count: int
    fit_case_keys_sha256: str
    data_fingerprint: str
    risk_label_rule_version: str = RISK_LABEL_RULE_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "fit_start_day": self.fit_start_day,
            "fit_end_day": self.fit_end_day,
            "primary_products": list(self.primary_products),
            "quantile": self.quantile,
            "quantile_method": self.quantile_method,
            "comparison": "greater_than_or_equal",
            "long_tail_threshold": self.long_tail_threshold,
            "short_tail_threshold": self.short_tail_threshold,
            "fit_case_count": self.fit_case_count,
            "fit_case_keys_sha256": self.fit_case_keys_sha256,
            "data_fingerprint": self.data_fingerprint,
            "risk_label_rule_version": self.risk_label_rule_version,
            "threshold_source": "outer_fit_primary_cases_only",
        }


def _sha256_case_keys(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(str(value) for value in values):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _finite_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    try:
        values = frame[list(columns)].to_numpy(dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(values.size and np.isfinite(values).all())


def _frame_integrity_issues(frame: pd.DataFrame, *, scope: str) -> list[str]:
    issues: list[str] = []
    required = set(MODEL_FEATURES).union({"timestamps", "trading_day"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        return [f"{scope}_missing_columns:{','.join(missing)}"]
    if frame.empty:
        return [f"{scope}_empty"]
    if not _finite_matrix(frame, MODEL_FEATURES):
        issues.append(f"{scope}_nonfinite_ohlcva")
        return issues
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    if (prices <= 0.0).any():
        issues.append(f"{scope}_nonpositive_price")
    flows = frame[["volume", "amount"]].to_numpy(dtype=np.float64)
    if (flows < 0.0).any():
        issues.append(f"{scope}_negative_flow")
    high_floor = frame[["open", "close", "low"]].max(axis=1)
    low_ceiling = frame[["open", "close", "high"]].min(axis=1)
    if bool((frame["high"] < high_floor).any() or (frame["low"] > low_ceiling).any()):
        issues.append(f"{scope}_invalid_ohlc")
    timestamps = pd.to_datetime(frame["timestamps"], errors="coerce")
    trading_days = pd.to_datetime(frame["trading_day"], errors="coerce")
    if timestamps.isna().any() or trading_days.isna().any():
        issues.append(f"{scope}_invalid_timestamp")
    elif timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        issues.append(f"{scope}_unordered_or_duplicate_timestamp")
    return issues


def case_integrity_issues(case: TargetOnlyCase, spec: RiskLabelSpec) -> tuple[str, ...]:
    """Return all included-case integrity failures without deriving a label."""

    context = case.target_context
    target = case.target
    issues = _frame_integrity_issues(context, scope="context")
    issues.extend(_frame_integrity_issues(target, scope="target"))
    if len(context) != spec.lookback:
        issues.append("context_wrong_lookback")
    if len(target) != case.pred_len or case.pred_len != len(target):
        issues.append("target_length_mismatch")
    if case.target_start - case.target_context_start != spec.lookback:
        issues.append("case_boundary_wrong_lookback")
    if len(case.target_days) != spec.horizon_trading_days:
        issues.append("target_wrong_trading_day_count")
    if "trading_day" in target:
        observed_groups = tuple(
            (pd.Timestamp(day).normalize(), int(len(group)))
            for day, group in target.groupby("trading_day", sort=False)
        )
        observed_days = tuple(day for day, _ in observed_groups)
        observed_counts = tuple(count for _, count in observed_groups)
        declared_days = tuple(pd.Timestamp(day).normalize() for day in case.target_days)
        if observed_days != declared_days:
            issues.append("target_day_boundary_mismatch")
        if len(observed_counts) != spec.horizon_trading_days or any(
            count not in spec.valid_bar_counts for count in observed_counts
        ):
            issues.append("target_incomplete_trading_day")
        expected_day_ends = tuple(int(value) for value in np.cumsum(observed_counts) - 1)
        if expected_day_ends != tuple(case.day_end_indices):
            issues.append("target_day_end_index_mismatch")
    if not context.empty and "timestamps" in context:
        if pd.Timestamp(context["timestamps"].iloc[-1]) != pd.Timestamp(case.origin_timestamp):
            issues.append("origin_timestamp_mismatch")
    if not context.empty and not target.empty and "timestamps" in context and "timestamps" in target:
        if pd.Timestamp(target["timestamps"].iloc[0]) <= pd.Timestamp(context["timestamps"].iloc[-1]):
            issues.append("target_not_strictly_after_origin")
    return tuple(sorted(set(issues)))


def context_diagnostics(case: TargetOnlyCase, spec: RiskLabelSpec) -> dict[str, object]:
    """Compute only pre-origin statistics plus the known target bar count."""

    context = case.target_context
    if len(context) != spec.lookback:
        raise RiskLabelError(
            f"{case.case_key} context has {len(context)} bars, expected {spec.lookback}"
        )
    if not _finite_matrix(context, MODEL_FEATURES):
        raise RiskLabelError(f"{case.case_key} context OHLCVA is not finite")
    values = context[MODEL_FEATURES].to_numpy(dtype=np.float64)
    closes = context["close"].to_numpy(dtype=np.float64)
    if (closes <= 0.0).any():
        raise RiskLabelError(f"{case.case_key} context close is not positive")
    log_returns = np.diff(np.log(closes))
    if len(log_returns) < spec.volatility_bars:
        raise RiskLabelError(f"{case.case_key} lacks EWMA context returns")
    volatility_returns = pd.Series(log_returns[-spec.volatility_bars :])
    raw_sigma = float(
        volatility_returns.ewm(
            halflife=spec.volatility_halflife_bars,
            adjust=spec.volatility_adjust,
        )
        .std(bias=spec.volatility_bias)
        .iloc[-1]
    )
    if not np.isfinite(raw_sigma):
        raise RiskLabelError(f"{case.case_key} EWMA context volatility is not finite")
    sigma = max(raw_sigma, spec.volatility_floor)
    horizon_scale = sigma * float(np.sqrt(case.pred_len))
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    z_values = (values - mean) / (std + spec.epsilon)
    clipped = np.abs(z_values) > spec.clip
    signature = hashlib.sha256()
    signature.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    signature.update(
        np.ascontiguousarray(
            pd.to_datetime(context["timestamps"]).astype("int64").to_numpy(),
            dtype="<i8",
        ).tobytes()
    )
    return {
        "origin_close": float(closes[-1]),
        "context_sigma_raw": raw_sigma,
        "context_sigma": sigma,
        "context_horizon_scale": horizon_scale,
        "context_clip_count": int(clipped.sum()),
        "context_value_count": int(clipped.size),
        "context_clip_fraction": float(clipped.mean()),
        "context_signature_sha256": signature.hexdigest(),
    }


def risk_outcome(case: TargetOnlyCase, spec: RiskLabelSpec) -> dict[str, object]:
    """Derive continuous V6 outcomes after freezing all past-only statistics."""

    issues = case_integrity_issues(case, spec)
    if issues:
        raise RiskLabelError(f"{case.case_key} failed integrity: {', '.join(issues)}")
    context = context_diagnostics(case, spec)
    target = case.target
    origin_close = float(context["origin_close"])
    horizon_scale = float(context["context_horizon_scale"])
    lows = target["low"].to_numpy(dtype=np.float64)
    highs = target["high"].to_numpy(dtype=np.float64)
    closes = target["close"].to_numpy(dtype=np.float64)
    long_excursion = max(0.0, -float(np.min(np.log(lows / origin_close))))
    short_excursion = max(0.0, float(np.max(np.log(highs / origin_close))))
    future_closes = np.concatenate(([origin_close], closes))
    future_returns = np.diff(np.log(future_closes))
    future_scale = float(np.sqrt(np.sum(np.square(future_returns))))
    future_vol_ratio = float(
        np.log((future_scale + spec.epsilon) / (horizon_scale + spec.epsilon))
    )
    day_counts = tuple(
        int(len(group)) for _, group in target.groupby("trading_day", sort=False)
    )
    output: dict[str, object] = {
        "case_key": case.case_key,
        "target_contract_id": case.target_contract_id,
        "product": case.product,
        "origin_timestamp": case.origin_timestamp,
        "origin_trading_day": case.origin_trading_day,
        "target_start_day": case.target_days[0],
        "target_end_day": case.target_end_day,
        "target_days": list(case.target_days),
        "target_bar_count": case.pred_len,
        "target_day_bar_counts": list(day_counts),
        "long_mae": long_excursion / horizon_scale,
        "short_mae": short_excursion / horizon_scale,
        "future_realized_scale": future_scale,
        "future_vol_ratio": future_vol_ratio,
        "risk_label_rule_version": spec.version,
        "data_fingerprint": case.data_fingerprint,
        "integrity_passed": True,
    }
    output.update(context)
    numeric = np.asarray(
        [output["long_mae"], output["short_mae"], future_vol_ratio],
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all():
        raise RiskLabelError(f"{case.case_key} produced a non-finite V6 outcome")
    return output


def risk_outcomes_frame(
    cases: Sequence[TargetOnlyCase], spec: RiskLabelSpec
) -> pd.DataFrame:
    """Build one unique continuous-outcome row per case or fail explicitly."""

    if not cases:
        raise RiskLabelError("V6 risk outcomes require at least one case")
    rows = [risk_outcome(case, spec) for case in cases]
    frame = pd.DataFrame(rows)
    if frame["case_key"].duplicated().any():
        raise RiskLabelError("V6 risk outcomes contain duplicate case keys")
    return frame.sort_values(
        ["target_end_day", "target_contract_id", "origin_timestamp"], kind="stable"
    ).reset_index(drop=True)


def fit_tail_thresholds(
    fit_records: pd.DataFrame,
    *,
    fold_id: str,
    fit_start_day: pd.Timestamp,
    fit_end_day: pd.Timestamp,
    primary_products: Sequence[str],
    quantile: float,
    quantile_method: str,
) -> TailThresholds:
    """Fit a fold threshold while rejecting non-fit, transfer, or out-of-range rows."""

    required = {
        "case_key",
        "product",
        "target_end_day",
        "long_mae",
        "short_mae",
        "data_fingerprint",
    }
    missing = sorted(required.difference(fit_records.columns))
    if missing:
        raise RiskLabelError(f"Fit records miss threshold columns: {missing!r}")
    if fit_records.empty:
        raise RiskLabelError(f"{fold_id} has no primary fit records")
    if fit_records["case_key"].duplicated().any():
        raise RiskLabelError(f"{fold_id} fit threshold records contain duplicate case keys")
    if "split" in fit_records and set(fit_records["split"].astype(str)) != {"fit"}:
        raise RiskLabelError(f"{fold_id} threshold source contains a non-fit split")
    if "fold_id" in fit_records and set(fit_records["fold_id"].astype(str)) != {str(fold_id)}:
        raise RiskLabelError(f"{fold_id} threshold source contains another fold")
    allowed_products = tuple(str(value) for value in primary_products)
    observed_products = set(fit_records["product"].astype(str))
    if observed_products != set(allowed_products):
        raise RiskLabelError(
            f"{fold_id} threshold source must contain every primary product and no transfer product"
        )
    days = pd.to_datetime(fit_records["target_end_day"]).dt.normalize()
    start = pd.Timestamp(fit_start_day).normalize()
    end = pd.Timestamp(fit_end_day).normalize()
    if bool(((days < start) | (days > end)).any()):
        raise RiskLabelError(f"{fold_id} threshold source escapes its fit boundary")
    values = fit_records[["long_mae", "short_mae"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RiskLabelError(f"{fold_id} threshold source contains non-finite labels")
    fingerprints = tuple(sorted(set(fit_records["data_fingerprint"].astype(str))))
    if len(fingerprints) != 1:
        raise RiskLabelError(f"{fold_id} threshold source mixes data fingerprints")
    if quantile_method != "linear" or not 0.0 < float(quantile) < 1.0:
        raise RiskLabelError("V6 threshold quantile protocol is invalid")
    return TailThresholds(
        fold_id=str(fold_id),
        fit_start_day=start,
        fit_end_day=end,
        primary_products=allowed_products,
        quantile=float(quantile),
        quantile_method=quantile_method,
        long_tail_threshold=float(
            np.quantile(values[:, 0], quantile, method=quantile_method)
        ),
        short_tail_threshold=float(
            np.quantile(values[:, 1], quantile, method=quantile_method)
        ),
        fit_case_count=int(len(fit_records)),
        fit_case_keys_sha256=_sha256_case_keys(fit_records["case_key"].astype(str).tolist()),
        data_fingerprint=fingerprints[0],
    )


def apply_tail_thresholds(
    records: pd.DataFrame,
    thresholds: TailThresholds,
    *,
    split: str,
) -> pd.DataFrame:
    """Apply already-frozen thresholds without refitting on the destination split."""

    if split not in {"fit", "inner_validation", "evaluation"}:
        raise ValueError(f"Unsupported V6 split: {split!r}")
    required = {"long_mae", "short_mae", "data_fingerprint", "case_key"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise RiskLabelError(f"Threshold destination records miss columns: {missing!r}")
    if records["case_key"].duplicated().any():
        raise RiskLabelError("Threshold destination contains duplicate case keys")
    if not records.empty:
        fingerprints = set(records["data_fingerprint"].astype(str))
        if fingerprints != {thresholds.data_fingerprint}:
            raise RiskLabelError("Threshold destination data fingerprint does not match fit")
    output = records.copy()
    output["fold_id"] = thresholds.fold_id
    output["split"] = split
    output["long_tail_threshold"] = thresholds.long_tail_threshold
    output["short_tail_threshold"] = thresholds.short_tail_threshold
    output["long_tail_event"] = (
        output["long_mae"].to_numpy(dtype=np.float64) >= thresholds.long_tail_threshold
    )
    output["short_tail_event"] = (
        output["short_mae"].to_numpy(dtype=np.float64) >= thresholds.short_tail_threshold
    )
    output["threshold_fit_case_keys_sha256"] = thresholds.fit_case_keys_sha256
    return output


def future_mutation_context_leakage_checks(
    case: TargetOnlyCase, spec: RiskLabelSpec
) -> tuple[dict[str, object], ...]:
    """Perturb each future OHLCVA field and prove past-only diagnostics are unchanged."""

    baseline = context_diagnostics(case, spec)
    checks: list[dict[str, object]] = []
    for feature_index, feature in enumerate(MODEL_FEATURES, start=1):
        frame = case.target_contract.frame.copy(deep=True)
        positions = np.arange(case.target_start, case.target_end_exclusive, dtype=np.int64)
        column_index = int(frame.columns.get_loc(feature))
        original = frame.iloc[positions, column_index].to_numpy(dtype=np.float64)
        perturbation = np.maximum(np.abs(original) * (0.01 * feature_index), 1.0)
        if np.issubdtype(frame.dtypes.iloc[column_index], np.integer):
            mutated = original + np.ceil(perturbation)
            mutated = mutated.astype(frame.dtypes.iloc[column_index], copy=False)
        else:
            mutated = original + perturbation
        frame.iloc[positions, column_index] = mutated
        mutated_contract = replace(case.target_contract, frame=frame)
        mutated_case = replace(case, target_contract=mutated_contract)
        observed = context_diagnostics(mutated_case, spec)
        changed = not np.array_equal(
            original,
            frame.iloc[positions, column_index].to_numpy(dtype=np.float64),
        )
        equal_fields = {
            key: observed[key] == baseline[key]
            for key in baseline
        }
        checks.append(
            {
                "case_key": case.case_key,
                "product": case.product,
                "mutated_future_feature": feature,
                "future_values_changed": changed,
                "context_fields_equal": equal_fields,
                "passed": bool(changed and all(equal_fields.values())),
            }
        )
    return tuple(checks)


__all__ = [
    "RiskLabelError",
    "RiskLabelSpec",
    "TailThresholds",
    "apply_tail_thresholds",
    "case_integrity_issues",
    "context_diagnostics",
    "fit_tail_thresholds",
    "future_mutation_context_leakage_checks",
    "risk_outcome",
    "risk_outcomes_frame",
]
