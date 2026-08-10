"""P0 target-only CE baseline for concrete-contract V3 experiments.

This module contains no panel or neighbour feature.  It is deliberately a
target-contract-only baseline so a later P2 path generator can be compared on
the identical target case keys and context length.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from csj.utils.tool import MODEL_FEATURES
from csj.v3.panel_data import (
    TIME_FEATURES,
    ConcreteContract,
    PanelCase,
    add_time_features,
    case_arrays,
)
from model.kronos import auto_regressive_inference


class P0Error(RuntimeError):
    """A target-only baseline violates its fixed V3 training protocol."""


class ProductDenseWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Same-product 21-bar CE windows with per-window target-only normalization."""

    def __init__(
        self,
        contracts: Sequence[ConcreteContract],
        *,
        product: str,
        fit_end_day: pd.Timestamp,
        lookback: int,
        horizon: int = 21,
        clip: float = 5.0,
        epsilon: float = 1e-5,
    ) -> None:
        if lookback < 1 or horizon != 21:
            raise ValueError("P0 requires a positive lookback and a fixed 21-bar CE horizon")
        product_contracts = [contract for contract in contracts if contract.product == product]
        if not product_contracts:
            raise P0Error(f"No contracts supplied for product {product!r}")
        self.product = product
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self.indices: list[tuple[int, int]] = []
        cutoff = pd.Timestamp(fit_end_day).normalize()
        for contract in sorted(product_contracts, key=lambda item: item.contract_id):
            frame = add_time_features(contract.frame).sort_values("timestamps", kind="stable").reset_index(drop=True)
            features = frame[MODEL_FEATURES].to_numpy(dtype=np.float32)
            stamps = frame[list(TIME_FEATURES)].to_numpy(dtype=np.float32)
            days = frame["trading_day"].to_numpy(dtype="datetime64[ns]")
            series_index = len(self.series)
            self.series.append((contract.contract_id, features, stamps, days))
            for forecast_start in range(self.lookback, len(frame) - self.horizon + 1):
                target_end_day = pd.Timestamp(days[forecast_start + self.horizon - 1]).normalize()
                if target_end_day <= cutoff:
                    self.indices.append((series_index, forecast_start))
        if not self.indices:
            raise P0Error(
                f"No P0 dense training windows for {product} through {cutoff.date()}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_index, forecast_start = self.indices[index]
        _, features, stamps, _ = self.series[series_index]
        start = forecast_start - self.lookback
        end = forecast_start + self.horizon
        window = features[start:end].astype(np.float64)
        # The target needs the same context-fitted coordinate system; avoid
        # refitting on the history by using its statistics explicitly.
        context = window[: self.lookback]
        mean = context.mean(axis=0)
        std = context.std(axis=0)
        normalized_window = np.clip(
            (window - mean) / (std + self.epsilon),
            -self.clip,
            self.clip,
        ).astype(np.float32)
        return (
            torch.from_numpy(normalized_window),
            torch.from_numpy(stamps[start:end].copy()),
        )


def future_token_loss(
    predictor: torch.nn.Module,
    logits_s1: torch.Tensor,
    logits_s2: torch.Tensor,
    target_s1: torch.Tensor,
    target_s2: torch.Tensor,
    *,
    lookback: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply CE only from the first target bar, never to context reconstruction."""

    target_start = lookback - 1
    if target_start >= logits_s1.shape[1]:
        raise P0Error("P0 token sequence contains no future target bars")
    return predictor.head.compute_loss(
        logits_s1[:, target_start:, :],
        logits_s2[:, target_start:, :],
        target_s1[:, target_start:],
        target_s2[:, target_start:],
    )


def _normalization_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0)


def _binary_balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    positive = actual == 1
    negative = actual == -1
    if not positive.any() or not negative.any():
        return float("nan")
    return float(
        0.5
        * (
            np.mean(predicted[positive] == 1)
            + np.mean(predicted[negative] == -1)
        )
    )


def _z_normalized_dtw(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or len(actual) != len(predicted):
        return float("nan")
    actual_std = float(np.std(actual))
    predicted_std = float(np.std(predicted))
    if actual_std == 0.0 or predicted_std == 0.0:
        return float("nan")
    a = (actual - np.mean(actual)) / actual_std
    b = (predicted - np.mean(predicted)) / predicted_std
    distance = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    distance[0, 0] = 0.0
    for row in range(1, len(a) + 1):
        for column in range(1, len(b) + 1):
            distance[row, column] = abs(a[row - 1] - b[column - 1]) + min(
                distance[row - 1, column],
                distance[row, column - 1],
                distance[row - 1, column - 1],
            )
    return float(distance[-1, -1] / max(len(a), 1))


def evaluate_target_paths(
    tokenizer: torch.nn.Module,
    predictor: torch.nn.Module,
    cases: Sequence[PanelCase],
    *,
    device: torch.device,
    max_context: int,
    clip: float,
    epsilon: float,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    batch_size: int,
    seed: int,
    model_name: str,
) -> pd.DataFrame:
    """Generate target-only paths and retain the fixed P0/P2 comparison keys."""

    if not cases:
        raise P0Error("Cannot evaluate zero target-only cases")
    if sample_count < 1 or batch_size < 1:
        raise ValueError("P0 sample count and batch size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = tokenizer.to(device)
    predictor = predictor.to(device)
    tokenizer.eval()
    predictor.eval()
    rows: list[dict[str, object]] = []
    by_length: dict[int, list[PanelCase]] = {}
    for case in cases:
        by_length.setdefault(case.pred_len, []).append(case)
    for pred_len, same_length_cases in sorted(by_length.items()):
        for offset in range(0, len(same_length_cases), batch_size):
            batch_cases = same_length_cases[offset : offset + batch_size]
            normalized_contexts: list[np.ndarray] = []
            context_stamps: list[np.ndarray] = []
            target_stamps: list[np.ndarray] = []
            statistics: list[tuple[np.ndarray, np.ndarray]] = []
            for case in batch_cases:
                values, stamps, _, _ = case_arrays(case, include_neighbor=False)
                mean, std = _normalization_statistics(values)
                normalized_contexts.append(
                    np.clip((values - mean) / (std + epsilon), -clip, clip).astype(np.float32)
                )
                context_stamps.append(stamps)
                target_frame = add_time_features(case.target)
                target_stamps.append(target_frame[list(TIME_FEATURES)].to_numpy(dtype=np.float32))
                statistics.append((mean, std))
            x = torch.from_numpy(np.stack(normalized_contexts)).to(device)
            x_stamp = torch.from_numpy(np.stack(context_stamps)).to(device)
            y_stamp = torch.from_numpy(np.stack(target_stamps)).to(device)
            normalized_samples = auto_regressive_inference(
                tokenizer,
                predictor,
                x,
                x_stamp,
                y_stamp,
                max_context=max_context,
                pred_len=pred_len,
                clip=clip,
                T=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
                return_samples=True,
            )[:, :, -pred_len:, :]
            for index, case in enumerate(batch_cases):
                mean, std = statistics[index]
                samples = normalized_samples[index] * (std + epsilon) + mean
                samples[..., 1] = np.maximum.reduce(
                    [samples[..., 1], samples[..., 0], samples[..., 3]]
                )
                samples[..., 2] = np.minimum.reduce(
                    [samples[..., 2], samples[..., 0], samples[..., 3]]
                )
                samples[..., 4:] = np.maximum(samples[..., 4:], 0.0)
                predicted = np.median(samples, axis=0)
                actual = case.target[MODEL_FEATURES].to_numpy(dtype=np.float64)
                origin_close = float(case.target_context["close"].iloc[-1])
                actual_return = float(actual[case.day_end_indices[-1], 3] / origin_close - 1.0)
                predicted_return = float(
                    predicted[case.day_end_indices[-1], 3] / origin_close - 1.0
                )
                actual_direction = 1 if actual_return > 0.0 else (-1 if actual_return < 0.0 else 0)
                predicted_direction = 1 if predicted_return >= 0.0 else -1
                actual_path_returns = actual[:, 3] / origin_close - 1.0
                predicted_path_returns = predicted[:, 3] / origin_close - 1.0
                correlation = (
                    float(np.corrcoef(actual_path_returns, predicted_path_returns)[0, 1])
                    if len(actual) > 1
                    and np.std(actual_path_returns) > 0.0
                    and np.std(predicted_path_returns) > 0.0
                    else float("nan")
                )
                rows.append(
                    {
                        "case_key": case.case_key,
                        "model": model_name,
                        "target_contract_id": case.target_contract_id,
                        "product": case.product,
                        "target_end_day": case.target_end_day,
                        "origin_timestamp": case.origin_timestamp,
                        "pred_len": pred_len,
                        "day_end_indices": list(case.day_end_indices),
                        "origin_close": origin_close,
                        "actual_path": actual.tolist(),
                        "predicted_path": predicted.tolist(),
                        "sample_paths": samples.tolist(),
                        "day3_actual_return": actual_return,
                        "day3_predicted_return": predicted_return,
                        "day3_actual_direction": actual_direction,
                        "day3_predicted_direction": predicted_direction,
                        "path_return_correlation": correlation,
                        "z_normalized_dtw": _z_normalized_dtw(
                            actual_path_returns,
                            predicted_path_returns,
                        ),
                    }
                )
            del x, x_stamp, y_stamp, normalized_samples
    return pd.DataFrame(rows).sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def target_path_metrics(records: pd.DataFrame) -> dict[str, object]:
    """The fixed P0 path metric tuple used for selection and later P2 comparison."""

    if records.empty:
        raise P0Error("Cannot calculate P0 metrics from zero records")
    valid = records.loc[records["day3_actual_direction"] != 0]
    actual = valid["day3_actual_direction"].to_numpy(dtype=np.int8)
    predicted = valid["day3_predicted_direction"].to_numpy(dtype=np.int8)
    correlations = records["path_return_correlation"].to_numpy(dtype=np.float64)
    dtw = records["z_normalized_dtw"].to_numpy(dtype=np.float64)
    output: dict[str, object] = {
        "samples": int(len(records)),
        "direction_samples": int(len(valid)),
        "day3_path_balanced_accuracy": _binary_balanced_accuracy(actual, predicted),
        "day3_return_mae": float(
            np.mean(
                np.abs(
                    records["day3_predicted_return"].to_numpy(dtype=np.float64)
                    - records["day3_actual_return"].to_numpy(dtype=np.float64)
                )
            )
        ),
        "mean_return_path_correlation": float(np.nanmean(correlations)),
        "mean_z_normalized_dtw": float(np.nanmean(dtw)),
    }
    output["by_product"] = {
        str(product): target_path_metrics(group)
        for product, group in records.groupby("product", sort=True)
    } if len(records["product"].unique()) > 1 else {}
    return output


@dataclass(frozen=True)
class P0TrainingConfig:
    learning_rate: float = 3e-6
    batch_size: int = 32
    max_epochs: int = 15
    early_stopping_patience: int = 3
    weight_decay: float = 0.1
    gradient_clip: float = 3.0
    warmup_ratio: float = 0.05
    num_workers: int = 0
    seed: int = 42


@dataclass(frozen=True)
class P0TrainingResult:
    checkpoint_path: Path
    best_epoch: int
    best_metrics: dict[str, object]
    history: list[dict[str, object]]
    elapsed_seconds: float


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def train_p0_ce(
    predictor: torch.nn.Module,
    tokenizer: torch.nn.Module,
    train_dataset: ProductDenseWindowDataset,
    validation_cases: Sequence[PanelCase],
    *,
    config: P0TrainingConfig,
    device: torch.device,
    output_dir: str | Path,
    max_context: int,
    clip: float,
    epsilon: float,
    validation_sample_count: int,
    validation_temperature: float,
    validation_top_k: int,
    validation_top_p: float,
    validation_batch_size: int,
) -> P0TrainingResult:
    """Fine-tune a target-only predictor using CE and fixed path-metric selection."""

    if not validation_cases:
        raise P0Error("P0 requires target-only inner-validation cases")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    predictor.to(device)
    tokenizer.to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=config.num_workers,
    )
    total_steps = max(len(loader) * config.max_epochs, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_model.pt"
    history_path = destination / "history.json"
    history: list[dict[str, object]] = []
    best_rank = (-float("inf"), -float("inf"), -float("inf"))
    best_epoch = 0
    best_metrics: dict[str, object] = {}
    stale_epochs = 0
    started = time.monotonic()

    for epoch in range(1, config.max_epochs + 1):
        predictor.train()
        tokenizer.eval()
        loss_total = 0.0
        batches = 0
        for values, stamps in loader:
            values = values.to(device)
            stamps = stamps.to(device)
            with torch.no_grad():
                token_s1, token_s2 = tokenizer.encode(values, half=True)
            logits_s1, logits_s2 = predictor(
                token_s1[:, :-1],
                token_s2[:, :-1],
                stamps[:, :-1, :],
            )
            loss, _, _ = future_token_loss(
                predictor,
                logits_s1,
                logits_s2,
                token_s1[:, 1:],
                token_s2[:, 1:],
                lookback=train_dataset.lookback,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), config.gradient_clip
            )
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise P0Error("Non-finite P0 CE loss or gradient norm")
            optimizer.step()
            scheduler.step()
            loss_total += float(loss.detach().cpu())
            batches += 1
        records = evaluate_target_paths(
            tokenizer,
            predictor,
            validation_cases,
            device=device,
            max_context=max_context,
            clip=clip,
            epsilon=epsilon,
            sample_count=validation_sample_count,
            temperature=validation_temperature,
            top_k=validation_top_k,
            top_p=validation_top_p,
            batch_size=validation_batch_size,
            seed=config.seed,
            model_name="p0_ce_only",
        )
        metrics = target_path_metrics(records)
        balanced_accuracy = float(metrics["day3_path_balanced_accuracy"])
        return_mae = float(metrics["day3_return_mae"])
        dtw = float(metrics["mean_z_normalized_dtw"])
        rank = (
            balanced_accuracy if math.isfinite(balanced_accuracy) else -float("inf"),
            -return_mae if math.isfinite(return_mae) else -float("inf"),
            -dtw if math.isfinite(dtw) else -float("inf"),
        )
        improved = rank > best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            best_metrics = metrics
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": 1,
                    "predictor_state": _cpu_state_dict(predictor),
                    "epoch": epoch,
                    "metrics": metrics,
                    "training_config": config.__dict__,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_token_ce": loss_total / max(batches, 1),
                "validation": metrics,
                "improved": improved,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        if stale_epochs >= config.early_stopping_patience:
            break
    if not checkpoint_path.exists():
        raise P0Error("P0 training completed without a usable checkpoint")
    return P0TrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        history=history,
        elapsed_seconds=time.monotonic() - started,
    )


def load_p0_checkpoint(predictor: torch.nn.Module, path: str | Path) -> dict[str, object]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("predictor_state"), dict):
        raise P0Error(f"Invalid P0 checkpoint: {path}")
    predictor.load_state_dict(checkpoint["predictor_state"])
    return checkpoint
