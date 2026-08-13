"""Frozen V7 P1 risk baselines and validation-only selection metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from csj.v5.target_data import TargetOnlyCase


class V7BaselineError(RuntimeError):
    """A baseline would violate V7's fit/validation/evaluation contract."""


SIDES = ("long", "short")
LABEL_COLUMNS = {"long": "long_tail_event", "short": "short_tail_event"}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    if not len(probabilities):
        return None
    return float(np.mean(np.square(probabilities - labels)))


def _pr_auc(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    if not len(probabilities) or labels.sum() == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    positives = np.cumsum(sorted_labels)
    precision = positives / np.arange(1, len(sorted_labels) + 1)
    return float(np.sum(precision * sorted_labels) / labels.sum())


def _finite_matrix(records: pd.DataFrame, columns: Sequence[str], *, label: str) -> np.ndarray:
    missing = sorted(set(columns).difference(records.columns))
    if missing:
        raise V7BaselineError(f"{label} records miss columns: {missing!r}")
    try:
        values = records[list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V7BaselineError(f"{label} records have nonnumeric features") from exc
    if not np.isfinite(values).all():
        raise V7BaselineError(f"{label} records have non-finite features")
    return values


def _require_split(records: pd.DataFrame, split: str, *, label: str) -> None:
    if records.empty:
        raise V7BaselineError(f"{label} records are empty")
    if "split" not in records or set(records["split"].astype(str)) != {str(split)}:
        raise V7BaselineError(f"{label} records must be exactly split={split!r}")
    if records["case_key"].astype(str).duplicated().any():
        raise V7BaselineError(f"{label} records contain duplicate case keys")


def _context_features(case: TargetOnlyCase) -> dict[str, float]:
    """Compute only the fixed pre-origin V7 context feature set."""

    frame = case.target_context
    required = {"open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise V7BaselineError(f"{case.case_key} context misses {missing!r}")
    values = frame[["open", "high", "low", "close", "volume", "amount"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all() or (values[:, :4] <= 0.0).any() or (values[:, 4:] < 0.0).any():
        raise V7BaselineError(f"{case.case_key} context has invalid OHLCVA")
    close = values[:, 3]
    log_returns = np.diff(np.log(close))
    if len(log_returns) < 60:
        raise V7BaselineError(f"{case.case_key} has too little context for EWMA60")
    ewma_volatility = float(
        pd.Series(log_returns[-60:]).ewm(halflife=20, adjust=False).std(bias=False).iloc[-1]
    )
    atr_start = max(1, len(frame) - 20)
    high = values[:, 1]
    low = values[:, 2]
    previous_close = close[:-1]
    true_ranges = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - previous_close),
            np.abs(low[1:] - previous_close),
        ]
    )
    atr20_over_price = float(np.mean(true_ranges[-(len(frame) - atr_start) :]) / close[-1])
    recent_range = float(
        (np.max(high[-20:]) - np.min(low[-20:])) / close[-1]
    )
    groups = list(frame.groupby("trading_day", sort=False))
    completed = [group for _, group in groups if len(group) in {5, 7}]
    last_return_values: list[float] = []
    for group in completed[-3:]:
        first_close = float(group["close"].iloc[0])
        last_close = float(group["close"].iloc[-1])
        last_return_values.append(abs(math.log(last_close / first_close)))
    if len(last_return_values) != 3:
        raise V7BaselineError(f"{case.case_key} lacks three full context days")
    volume = np.log1p(values[:, 4])
    amount = np.log1p(values[:, 5])
    vol_mean, vol_std = float(volume.mean()), float(volume.std())
    amt_mean, amt_std = float(amount.mean()), float(amount.std())
    return {
        "ewma_volatility": ewma_volatility,
        "atr20_over_price": atr20_over_price,
        "range20_over_price": recent_range,
        "absolute_return_day_minus_3": last_return_values[0],
        "absolute_return_day_minus_2": last_return_values[1],
        "absolute_return_day_minus_1": last_return_values[2],
        "volume_zscore": float((volume[-1] - vol_mean) / (vol_std + 1e-8)),
        "amount_zscore": float((amount[-1] - amt_mean) / (amt_std + 1e-8)),
    }


CONTEXT_FEATURE_COLUMNS = (
    "ewma_volatility",
    "atr20_over_price",
    "range20_over_price",
    "absolute_return_day_minus_3",
    "absolute_return_day_minus_2",
    "absolute_return_day_minus_1",
    "volume_zscore",
    "amount_zscore",
    "context_clip_fraction",
)


def attach_context_features(
    records: pd.DataFrame, *, cases: Sequence[TargetOnlyCase]
) -> pd.DataFrame:
    """Attach fixed history-only features to P0 records keyed by case."""

    if records.empty:
        raise V7BaselineError("Cannot attach context features to empty records")
    by_key = {case.case_key: case for case in cases}
    keys = records["case_key"].astype(str).tolist()
    unique_keys = tuple(dict.fromkeys(keys))
    missing = sorted(set(unique_keys).difference(by_key))
    if missing:
        raise V7BaselineError(f"Context feature cases are missing: {missing[:5]!r}")
    clip_by_key: dict[str, float] = {}
    if "context_clip_fraction" in records:
        for key, values in records.groupby(records["case_key"].astype(str), sort=False)[
            "context_clip_fraction"
        ]:
            unique = values.to_numpy(dtype=np.float64)
            if not np.isfinite(unique).all() or not np.allclose(unique, unique[0]):
                raise V7BaselineError(f"{key} has inconsistent context clip fractions")
            clip_by_key[str(key)] = float(unique[0])
    rows: list[dict[str, float | str]] = []
    for key in unique_keys:
        row = _context_features(by_key[key])
        row["context_clip_fraction"] = clip_by_key.get(key, float("nan"))
        row["case_key"] = key
        rows.append(row)
    features = pd.DataFrame(rows)
    if not np.isfinite(features[list(CONTEXT_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise V7BaselineError("Fixed V7 context features are non-finite")
    overlap = set(CONTEXT_FEATURE_COLUMNS).intersection(records.columns)
    if overlap:
        records = records.drop(columns=sorted(overlap))
    return records.copy().merge(features, on="case_key", how="left", validate="many_to_one")


def _macro_brier(
    records: pd.DataFrame,
    *,
    probability_columns: Mapping[str, str],
    products: Sequence[str],
) -> tuple[float | None, list[dict[str, object]]]:
    cells: list[float] = []
    missing_cells: list[dict[str, object]] = []
    for product in products:
        product_records = records.loc[records["product"].astype(str) == str(product)]
        for side in SIDES:
            if product_records.empty:
                missing_cells.append({"product": str(product), "side": side})
                continue
            probabilities = product_records[probability_columns[side]].to_numpy(dtype=np.float64)
            labels = product_records[LABEL_COLUMNS[side]].astype(float).to_numpy()
            if not np.isfinite(probabilities).all():
                raise V7BaselineError("Baseline probabilities are non-finite")
            score = _brier(probabilities, labels)
            if score is not None:
                cells.append(score)
    return (float(np.mean(cells)) if cells else None), missing_cells


def selection_brier(
    records: pd.DataFrame,
    *,
    probability_columns: Mapping[str, str],
    gate_products: Sequence[str],
    all_products: Sequence[str],
) -> dict[str, object]:
    """Compute the frozen 50/50 core and all-product macro Brier selector."""

    core, core_missing = _macro_brier(
        records,
        probability_columns=probability_columns,
        products=gate_products,
    )
    all_macro, all_missing = _macro_brier(
        records,
        probability_columns=probability_columns,
        products=all_products,
    )
    score = None if core is None or all_macro is None else float(0.5 * core + 0.5 * all_macro)
    return {
        "selection_brier": score,
        "core_macro_brier": core,
        "all_product_macro_brier": all_macro,
        "missing_core_product_side_cells": core_missing,
        "missing_all_product_side_cells": all_missing,
    }


@dataclass(frozen=True)
class EventRateModel:
    global_rates: Mapping[str, float]
    product_rates: Mapping[str, Mapping[str, float]]
    fallback_products: tuple[str, ...]


def fit_event_rates(fit_records: pd.DataFrame) -> EventRateModel:
    _require_split(fit_records, "fit", label="event-rate fit")
    global_rates = {
        side: float(fit_records[LABEL_COLUMNS[side]].astype(float).mean()) for side in SIDES
    }
    product_rates: dict[str, dict[str, float]] = {}
    for product, group in fit_records.groupby("product", sort=True):
        product_rates[str(product)] = {
            side: float(group[LABEL_COLUMNS[side]].astype(float).mean()) for side in SIDES
        }
    return EventRateModel(global_rates, product_rates, ())


def apply_event_rates(
    records: pd.DataFrame, model: EventRateModel, *, product_specific: bool
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    output = records.copy()
    fallback: set[str] = set()
    for side in SIDES:
        values: list[float] = []
        for product in output["product"].astype(str):
            if product_specific and product in model.product_rates:
                values.append(float(model.product_rates[product][side]))
            else:
                values.append(float(model.global_rates[side]))
                if product_specific:
                    fallback.add(product)
        output[f"p_{side}"] = np.asarray(values, dtype=np.float64)
    return output, tuple(sorted(fallback))


@dataclass(frozen=True)
class RankProbabilityModel:
    feature: str
    quantiles: np.ndarray
    probabilities: Mapping[str, np.ndarray]


def fit_rank_probability(
    fit_records: pd.DataFrame, *, feature: str, bin_count: int
) -> RankProbabilityModel:
    _require_split(fit_records, "fit", label=f"rank fit/{feature}")
    values = _finite_matrix(fit_records, (feature,), label=f"rank fit/{feature}")[:, 0]
    if bin_count < 2:
        raise V7BaselineError("V7 rank probability needs at least two bins")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1), method="linear")
    # Repeated values are legitimate; searchsorted handles zero-width bins.
    indices = np.clip(np.searchsorted(quantiles[1:-1], values, side="right"), 0, bin_count - 1)
    probability: dict[str, np.ndarray] = {}
    for side in SIDES:
        labels = fit_records[LABEL_COLUMNS[side]].astype(float).to_numpy(dtype=np.float64)
        rates = np.full(bin_count, float(labels.mean()), dtype=np.float64)
        for index in range(bin_count):
            selected = labels[indices == index]
            if len(selected):
                rates[index] = float(selected.mean())
        probability[side] = rates
    return RankProbabilityModel(feature, quantiles, probability)


def apply_rank_probability(records: pd.DataFrame, model: RankProbabilityModel) -> pd.DataFrame:
    output = records.copy()
    values = _finite_matrix(output, (model.feature,), label=f"rank apply/{model.feature}")[:, 0]
    bin_count = len(model.quantiles) - 1
    indices = np.clip(
        np.searchsorted(model.quantiles[1:-1], values, side="right"), 0, bin_count - 1
    )
    for side in SIDES:
        output[f"p_{side}"] = model.probabilities[side][indices]
    return output


@dataclass(frozen=True)
class LogisticSideModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray


def _fit_logistic_side(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2_penalty: float,
    maximum_iterations: int,
) -> LogisticSideModel:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.r_[0.0, np.full(design.shape[1] - 1, float(l2_penalty))]
    for _ in range(int(maximum_iterations)):
        probability = _sigmoid(design @ coefficients)
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (probability - labels) + penalty * coefficients
        hessian = (design.T * weights) @ design + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise V7BaselineError("Fixed logistic Hessian is singular") from exc
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    if not np.isfinite(coefficients).all():
        raise V7BaselineError("Fixed logistic fit produced non-finite coefficients")
    return LogisticSideModel(mean, scale, coefficients)


def _apply_logistic_side(features: np.ndarray, model: LogisticSideModel) -> np.ndarray:
    standardized = (features - model.mean) / model.scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return _sigmoid(design @ model.coefficients)


@dataclass(frozen=True)
class FixedContextLogistic:
    features: tuple[str, ...]
    by_side: Mapping[str, LogisticSideModel]


def fit_fixed_context_logistic(
    fit_records: pd.DataFrame,
    *,
    features: Sequence[str],
    l2_penalty: float,
    maximum_iterations: int,
) -> FixedContextLogistic:
    _require_split(fit_records, "fit", label="fixed-context logistic fit")
    values = _finite_matrix(fit_records, features, label="fixed-context logistic fit")
    by_side = {
        side: _fit_logistic_side(
            values,
            fit_records[LABEL_COLUMNS[side]].astype(float).to_numpy(dtype=np.float64),
            l2_penalty=l2_penalty,
            maximum_iterations=maximum_iterations,
        )
        for side in SIDES
    }
    return FixedContextLogistic(tuple(features), by_side)


def apply_fixed_context_logistic(
    records: pd.DataFrame, model: FixedContextLogistic
) -> pd.DataFrame:
    output = records.copy()
    values = _finite_matrix(output, model.features, label="fixed-context logistic apply")
    for side in SIDES:
        output[f"p_{side}"] = _apply_logistic_side(values, model.by_side[side])
    return output


def attach_zero_shot_path_risk(
    records: pd.DataFrame,
    *,
    path_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join fit-independent raw-path crossing fractions by fold and case."""

    required = {"fold_id", "case_key", "p_long", "p_short"}
    missing = sorted(required.difference(path_features.columns))
    if missing:
        raise V7BaselineError(f"Path-risk features miss columns: {missing!r}")
    output = records.drop(columns=["p_long", "p_short"], errors="ignore")
    joined = output.merge(
        path_features[list(required)], on=["fold_id", "case_key"], how="left", validate="one_to_one"
    )
    if joined[["p_long", "p_short"]].isna().any().any():
        raise V7BaselineError("Raw-path risk baseline does not cover requested records")
    return joined


def fit_and_predict_baselines(
    *,
    fit_records: pd.DataFrame,
    destination_records: pd.DataFrame,
    config: Mapping[str, Any],
    path_features: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Fit every P1 baseline on fit only, then score one validation/eval split."""

    _require_split(fit_records, "fit", label="baseline fit")
    if destination_records.empty:
        raise V7BaselineError("Cannot score empty V7 baseline destination")
    bin_count = int(config["baselines"]["rank_probability_bins"])
    logistic_config = config["baselines"]["fixed_context_logistic"]
    event_rates = fit_event_rates(fit_records)
    result: dict[str, pd.DataFrame] = {}
    parameters: dict[str, object] = {
        "event_rates": {
            "global": dict(event_rates.global_rates),
            "product": {key: dict(value) for key, value in event_rates.product_rates.items()},
        }
    }
    global_records, _ = apply_event_rates(destination_records, event_rates, product_specific=False)
    result["fit_global_event_rate"] = global_records
    product_records, fallbacks = apply_event_rates(destination_records, event_rates, product_specific=True)
    result["fit_product_event_rate"] = product_records
    parameters["fit_product_event_rate"] = {"fallback_to_global_products": list(fallbacks)}
    for name, feature in (
        ("ewma_volatility_rank", "ewma_volatility"),
        ("atr20_rank", "atr20_over_price"),
    ):
        model = fit_rank_probability(fit_records, feature=feature, bin_count=bin_count)
        result[name] = apply_rank_probability(destination_records, model)
        parameters[name] = {
            "feature": feature,
            "quantiles": model.quantiles.tolist(),
            "probabilities": {side: values.tolist() for side, values in model.probabilities.items()},
        }
    logistic = fit_fixed_context_logistic(
        fit_records,
        features=CONTEXT_FEATURE_COLUMNS,
        l2_penalty=float(logistic_config["l2_penalty"]),
        maximum_iterations=int(logistic_config["maximum_iterations"]),
    )
    result["fixed_context_logistic"] = apply_fixed_context_logistic(destination_records, logistic)
    parameters["fixed_context_logistic"] = {
        "features": list(logistic.features),
        "by_side": {
            side: {
                "mean": model.mean.tolist(),
                "scale": model.scale.tolist(),
                "coefficients": model.coefficients.tolist(),
            }
            for side, model in logistic.by_side.items()
        },
    }
    result["zero_shot_path_risk"] = attach_zero_shot_path_risk(
        destination_records, path_features=path_features
    )
    return result, parameters


def choose_baseline(
    validation_by_name: Mapping[str, pd.DataFrame],
    *,
    selection_order: Sequence[str],
    gate_products: Sequence[str],
    all_products: Sequence[str],
) -> tuple[str, dict[str, object]]:
    """Select exactly one baseline solely from inner-validation Brier."""

    expected = tuple(str(value) for value in selection_order)
    if set(validation_by_name) != set(expected):
        raise V7BaselineError("V7 baseline candidates differ from frozen selection order")
    scores: dict[str, dict[str, object]] = {}
    for name in expected:
        records = validation_by_name[name]
        columns = {"long": "p_long", "short": "p_short"}
        scores[name] = selection_brier(
            records,
            probability_columns=columns,
            gate_products=gate_products,
            all_products=all_products,
        )
    usable = [
        name
        for name in expected
        if scores[name]["selection_brier"] is not None
        and np.isfinite(float(scores[name]["selection_brier"]))
    ]
    if not usable:
        raise V7BaselineError("No V7 baseline has a valid validation selection Brier")
    best = min(usable, key=lambda name: (float(scores[name]["selection_brier"]), expected.index(name)))
    return best, {
        "selected_baseline": best,
        "selection_metric": "0.5 * core_macro_brier + 0.5 * all_product_macro_brier",
        "validation_scores": scores,
        "selection_order": list(expected),
    }


def classification_metrics(records: pd.DataFrame) -> dict[str, object]:
    """Report probabilistic metrics for both risk sides, without selection."""

    output: dict[str, object] = {}
    for side in SIDES:
        probabilities = records[f"p_{side}"].to_numpy(dtype=np.float64)
        labels = records[LABEL_COLUMNS[side]].astype(float).to_numpy(dtype=np.float64)
        output[side] = {
            "cases": int(len(records)),
            "event_count": int(labels.sum()),
            "event_rate": float(labels.mean()) if len(labels) else None,
            "brier": _brier(probabilities, labels),
            "pr_auc": _pr_auc(probabilities, labels),
            "probability_minimum": float(np.min(probabilities)) if len(probabilities) else None,
            "probability_maximum": float(np.max(probabilities)) if len(probabilities) else None,
            "probability_standard_deviation": float(np.std(probabilities)) if len(probabilities) else None,
        }
    return output


__all__ = [
    "CONTEXT_FEATURE_COLUMNS",
    "V7BaselineError",
    "attach_context_features",
    "choose_baseline",
    "classification_metrics",
    "fit_and_predict_baselines",
    "selection_brier",
]
