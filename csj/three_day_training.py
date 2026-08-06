from __future__ import annotations

import gc
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from csj.futures_data import (
    TIME_FEATURES,
    DenseInstrumentWindowDataset,
    ThreeDayDirectionDataset,
    ThreeTradingDayCase,
    fit_context_normalization,
)
from csj.metrics import (
    balanced_direction_accuracy,
    direction_label,
    three_day_endpoint_returns,
    three_day_metrics_with_instruments,
)
from csj.three_day_evaluation import predict_three_day_cases
from csj.trend_model import KronosTrendWrapper
from csj.utils.tool import MODEL_FEATURES


def slice_future_token_tensors(
    logits_s1: torch.Tensor,
    logits_s2: torch.Tensor,
    target_s1: torch.Tensor,
    target_s2: torch.Tensor,
    *,
    lookback: int,
    padding_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Slice next-token tensors so loss starts at the first future bar."""

    if lookback < 1:
        raise ValueError("lookback must be positive")
    sequence_length = logits_s1.shape[1]
    if logits_s2.shape[1] != sequence_length:
        raise ValueError("S1 and S2 logits must have the same sequence length")
    if (
        target_s1.shape[1] != sequence_length
        or target_s2.shape[1] != sequence_length
    ):
        raise ValueError("Targets and logits must have the same sequence length")
    if padding_mask is not None and padding_mask.shape[:2] != target_s1.shape[:2]:
        raise ValueError("padding_mask must align with targets")

    target_start = lookback - 1
    if target_start >= sequence_length:
        raise ValueError("No future target tokens remain after the context")
    future_mask = None if padding_mask is None else padding_mask[:, target_start:]
    return (
        logits_s1[:, target_start:, :],
        logits_s2[:, target_start:, :],
        target_s1[:, target_start:],
        target_s2[:, target_start:],
        future_mask,
    )


def compute_future_token_loss(
    predictor: torch.nn.Module,
    logits_s1: torch.Tensor,
    logits_s2: torch.Tensor,
    target_s1: torch.Tensor,
    target_s2: torch.Tensor,
    *,
    lookback: int,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    future = slice_future_token_tensors(
        logits_s1,
        logits_s2,
        target_s1,
        target_s2,
        lookback=lookback,
        padding_mask=padding_mask,
    )
    return predictor.head.compute_loss(
        future[0],
        future[1],
        future[2],
        future[3],
        padding_mask=future[4],
    )


def direction_logits_from_context(
    wrapper: KronosTrendWrapper,
    tokenizer: torch.nn.Module,
    normalized_window: torch.Tensor,
    stamp_window: torch.Tensor,
    *,
    lookback: int,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tokenize only the context and feed only those tokens to the trend head."""

    if normalized_window.ndim != 3 or stamp_window.ndim != 3:
        raise ValueError("Window tensors must have [batch, time, feature] shapes")
    if normalized_window.shape[:2] != stamp_window.shape[:2]:
        raise ValueError("Feature and timestamp windows must align")
    if lookback < 1 or lookback > normalized_window.shape[1]:
        raise ValueError("lookback falls outside the supplied window")

    context = normalized_window[:, :lookback, :]
    context_stamp = stamp_window[:, :lookback, :]
    context_mask = None if padding_mask is None else padding_mask[:, :lookback]
    with torch.no_grad():
        context_s1, context_s2 = tokenizer.encode(context, half=True)
    return wrapper.direction_logits(
        context_s1,
        context_s2,
        context_stamp,
        context_mask,
    )


def direction_targets_from_cases(
    cases: Sequence[ThreeTradingDayCase],
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels: list[list[float]] = []
    valid: list[list[bool]] = []
    for case in cases:
        origin_close = float(case.context["close"].iloc[-1])
        returns = three_day_endpoint_returns(
            case.target["close"].to_numpy(dtype=np.float64),
            origin_close,
            case.day_end_indices,
        )
        labels.append([float(value > 0) for value in returns])
        valid.append([value != 0 for value in returns])
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )


def masked_direction_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != labels.shape or logits.shape != valid_mask.shape:
        raise ValueError("Direction logits, labels, and mask must have the same shape")
    if not torch.any(valid_mask):
        raise ValueError("Direction batch contains no non-zero endpoint returns")
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return losses[valid_mask].mean()


@dataclass(frozen=True)
class ThreeDayTrainingResult:
    checkpoint_path: Path
    best_epoch: int
    best_day3_balanced_accuracy: float
    best_day3_return_mae: float
    best_z_normalized_dtw: float
    history: list[dict[str, Any]]
    elapsed_seconds: float


@dataclass(frozen=True)
class ThreeDayDirectionTrainingResult:
    checkpoint_path: Path
    best_epoch: int
    best_day3_balanced_accuracy: float
    best_day3_return_mae: float
    best_z_normalized_dtw: float
    best_auxiliary_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    elapsed_seconds: float


def _cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    denominator = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device).cpu()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        return torch.mps.get_rng_state().cpu()
    return None


def _restore_device_rng_state(
    device: torch.device,
    state: torch.Tensor | None,
) -> None:
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps" and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state)


def _finite_or(value: float, fallback: float) -> float:
    return value if np.isfinite(value) else fallback


def predict_auxiliary_direction_cases(
    wrapper: KronosTrendWrapper,
    tokenizer: torch.nn.Module,
    cases: Sequence[ThreeTradingDayCase],
    *,
    device: torch.device,
    clip: float,
    normalization_epsilon: float,
    batch_size: int,
) -> pd.DataFrame:
    """Evaluate the context-only direction head without generating target bars."""

    if not cases:
        raise ValueError("At least one case is required for auxiliary evaluation")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    wrapper = wrapper.to(device)
    tokenizer = tokenizer.to(device)
    wrapper.eval()
    tokenizer.eval()
    rows: list[dict[str, object]] = []

    for offset in range(0, len(cases), batch_size):
        batch_cases = cases[offset : offset + batch_size]
        normalized_contexts: list[np.ndarray] = []
        context_stamps: list[np.ndarray] = []
        for case in batch_cases:
            stats = fit_context_normalization(
                case.context,
                clip=clip,
                epsilon=normalization_epsilon,
            )
            context = case.context[MODEL_FEATURES].to_numpy(dtype=np.float64)
            normalized_contexts.append(stats.transform(context).astype(np.float32))
            context_stamps.append(
                case.context[TIME_FEATURES].to_numpy(dtype=np.float32)
            )

        context_tensor = torch.from_numpy(np.stack(normalized_contexts)).to(device)
        stamp_tensor = torch.from_numpy(np.stack(context_stamps)).to(device)
        labels, valid_masks = direction_targets_from_cases(batch_cases, device=device)
        with torch.no_grad():
            logits = direction_logits_from_context(
                wrapper,
                tokenizer,
                context_tensor,
                stamp_tensor,
                lookback=len(normalized_contexts[0]),
            )
            probabilities = torch.sigmoid(logits)
            losses = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
        logits_array = logits.detach().cpu().numpy()
        probabilities_array = probabilities.detach().cpu().numpy()
        losses_array = losses.detach().cpu().numpy()
        valid_array = valid_masks.detach().cpu().numpy()

        for case_index, case in enumerate(batch_cases):
            origin_close = float(case.context["close"].iloc[-1])
            returns = three_day_endpoint_returns(
                case.target["close"].to_numpy(dtype=np.float64),
                origin_close,
                case.day_end_indices,
            )
            row: dict[str, object] = {
                "instrument": case.instrument,
                "fold_id": case.fold_id,
                "split": case.split,
                "target_day": case.target_days[0],
                "target_days": case.target_days,
                "origin_timestamp": case.origin_timestamp,
                "origin_trading_day": case.origin_trading_day,
            }
            for day_index, endpoint_return in enumerate(returns, start=1):
                probability = float(probabilities_array[case_index, day_index - 1])
                valid = bool(valid_array[case_index, day_index - 1])
                row.update(
                    {
                        f"day{day_index}_actual_return": float(endpoint_return),
                        f"day{day_index}_actual_direction": direction_label(
                            endpoint_return
                        ),
                        f"aux_day{day_index}_logit": float(
                            logits_array[case_index, day_index - 1]
                        ),
                        f"aux_day{day_index}_up_probability": probability,
                        f"aux_day{day_index}_direction": (
                            1 if probability >= 0.5 else -1
                        ),
                        f"aux_day{day_index}_bce": (
                            float(losses_array[case_index, day_index - 1])
                            if valid
                            else float("nan")
                        ),
                    }
                )
            rows.append(row)
        del (
            context_tensor,
            stamp_tensor,
            labels,
            valid_masks,
            logits,
            probabilities,
            losses,
        )

    records = pd.DataFrame(rows).sort_values(
        ["target_day", "instrument"], kind="stable"
    ).reset_index(drop=True)
    records["inference_device"] = str(device)
    return records


def compute_auxiliary_direction_metrics(records: pd.DataFrame) -> dict[str, object]:
    """Compute direction-head metrics while keeping them distinct from path metrics."""

    if records.empty:
        raise ValueError("Cannot compute auxiliary metrics from an empty table")
    output: dict[str, object] = {"samples": int(len(records)), "endpoints": {}}
    endpoints = output["endpoints"]
    assert isinstance(endpoints, dict)
    for day_number in (1, 2, 3):
        actual = records[f"day{day_number}_actual_direction"].to_numpy(
            dtype=np.int8
        )
        predicted = records[f"aux_day{day_number}_direction"].to_numpy(
            dtype=np.int8
        )
        valid = actual != 0
        probability = records[f"aux_day{day_number}_up_probability"].to_numpy(
            dtype=np.float64
        )
        bce = records[f"aux_day{day_number}_bce"].to_numpy(dtype=np.float64)
        endpoints[f"day{day_number}"] = {
            "direction_samples": int(valid.sum()),
            "excluded_zero_actual_returns": int((~valid).sum()),
            "aux_direction_balanced_accuracy": balanced_direction_accuracy(
                actual, predicted
            ),
            "aux_direction_accuracy": float(np.mean(actual[valid] == predicted[valid]))
            if valid.any()
            else float("nan"),
            "actual_direction_counts": {
                str(label): int(np.sum(actual == label)) for label in (-1, 0, 1)
            },
            "predicted_direction_counts": {
                str(label): int(np.sum(predicted == label)) for label in (-1, 0, 1)
            },
            "confusion_matrix": {
                str(actual_label): {
                    str(predicted_label): int(
                        np.sum(
                            (actual == actual_label)
                            & (predicted == predicted_label)
                        )
                    )
                    for predicted_label in (-1, 0, 1)
                }
                for actual_label in (-1, 1)
            },
            "mean_up_probability": float(np.mean(probability)),
            "mean_bce": float(np.nanmean(bce)) if valid.any() else float("nan"),
        }
    return output


def auxiliary_direction_metrics_with_instruments(
    records: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    output = {"pooled": compute_auxiliary_direction_metrics(records)}
    for instrument, group in records.groupby("instrument", sort=True):
        output[str(instrument)] = compute_auxiliary_direction_metrics(group)
    return output


def train_ce_only_predictor(
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    train_dataset: DenseInstrumentWindowDataset,
    validation_cases: Sequence[ThreeTradingDayCase],
    *,
    device: torch.device,
    output_dir: str | Path,
    learning_rate: float,
    seed: int,
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    weight_decay: float,
    gradient_clip: float,
    warmup_ratio: float,
    evaluation_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    validation_device: torch.device | None = None,
    max_train_batches: int | None = None,
) -> ThreeDayTrainingResult:
    if not validation_cases:
        raise ValueError("CE-only training requires three-day validation cases")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_model.pt"
    latest_state_path = output_path / "latest_training_state.pt"
    history_path = output_path / "history.json"
    summary_path = output_path / "summary.json"
    validation_device = validation_device or device

    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    tokenizer.to(device)
    model.to(device)
    model.train()

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    batches_per_epoch = len(loader)
    if max_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, max_train_batches)
    total_steps = max(max_epochs * batches_per_epoch, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_with_warmup(
            step, total_steps, warmup_steps
        ),
    )

    best_rank = (-float("inf"), -float("inf"), -float("inf"))
    best_epoch = 0
    best_day3_balanced_accuracy = -float("inf")
    best_day3_return_mae = float("inf")
    best_z_normalized_dtw = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    elapsed_before_resume = 0.0
    run_signature = {
        "learning_rate": learning_rate,
        "seed": seed,
        "max_epochs": max_epochs,
        "batches_per_epoch": batches_per_epoch,
        "dense_windows": len(train_dataset),
        "validation_cases": len(validation_cases),
        "training_device_type": device.type,
        "validation_device_type": validation_device.type,
    }
    if latest_state_path.exists() and not summary_path.exists():
        latest_state = torch.load(
            latest_state_path,
            map_location="cpu",
            weights_only=False,
        )
        if latest_state.get("run_signature") != run_signature:
            raise RuntimeError(
                f"Incomplete training state does not match this run: "
                f"{latest_state_path}"
            )
        model.load_state_dict(latest_state["predictor_state"])
        optimizer.load_state_dict(latest_state["optimizer_state"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(latest_state["scheduler_state"])
        generator.set_state(latest_state["data_generator_state"])
        torch.set_rng_state(latest_state["torch_rng_state"])
        np.random.set_state(latest_state["numpy_rng_state"])
        _restore_device_rng_state(device, latest_state["device_rng_state"])
        best_rank = tuple(latest_state["best_rank"])
        best_epoch = int(latest_state["best_epoch"])
        best_day3_balanced_accuracy = float(
            latest_state["best_day3_balanced_accuracy"]
        )
        best_day3_return_mae = float(
            latest_state["best_day3_return_mae"]
        )
        best_z_normalized_dtw = float(
            latest_state["best_z_normalized_dtw"]
        )
        epochs_without_improvement = int(
            latest_state["epochs_without_improvement"]
        )
        history = list(latest_state["history"])
        start_epoch = int(latest_state["epoch"]) + 1
        elapsed_before_resume = float(latest_state["elapsed_seconds"])
        if epochs_without_improvement >= early_stopping_patience:
            start_epoch = max_epochs + 1
        print(
            f"resuming CE-only training at epoch={start_epoch} "
            f"from={latest_state_path}"
        )
    started_at = time.monotonic()
    print(
        f"ce_only_train instrument={train_dataset.instrument} "
        f"fold={train_dataset.period.fold_id} lr={learning_rate:.1e} seed={seed} "
        f"lambda_dir=0 windows={len(train_dataset)} "
        f"batches_per_epoch={batches_per_epoch} device={device}"
    )

    for epoch in range(start_epoch, max_epochs + 1):
        model.to(device)
        tokenizer.to(device)
        model.train()
        epoch_started = time.monotonic()
        training_started = time.monotonic()
        loss_totals = torch.zeros(3, dtype=torch.float32, device=device)
        finite_train_state = torch.ones((), dtype=torch.bool, device=device)
        processed_batches = 0
        for batch_index, (batch_x, batch_stamp) in enumerate(loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            batch_x = batch_x.to(device)
            batch_stamp = batch_stamp.to(device)
            with torch.no_grad():
                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_in_s1 = token_s1[:, :-1]
            token_in_s2 = token_s2[:, :-1]
            target_s1 = token_s1[:, 1:]
            target_s2 = token_s2[:, 1:]
            logits_s1, logits_s2 = model(
                token_in_s1,
                token_in_s2,
                batch_stamp[:, :-1, :],
            )
            loss, loss_s1, loss_s2 = compute_future_token_loss(
                model,
                logits_s1,
                logits_s2,
                target_s1,
                target_s2,
                lookback=train_dataset.lookback,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip
            )
            finite_train_state.logical_and_(
                torch.isfinite(loss) & torch.isfinite(gradient_norm)
            )
            optimizer.step()
            scheduler.step()

            loss_totals.add_(
                torch.stack((loss, loss_s1, loss_s2)).detach().to(torch.float32)
            )
            processed_batches += 1
            del (
                batch_x,
                batch_stamp,
                token_s1,
                token_s2,
                logits_s1,
                logits_s2,
                loss,
                loss_s1,
                loss_s2,
            )

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        if not bool(finite_train_state.detach().cpu()):
            raise RuntimeError("Non-finite loss or gradient norm in CE-only training")
        loss_values = loss_totals.detach().cpu().numpy() / max(
            processed_batches, 1
        )
        training_seconds = time.monotonic() - training_started
        optimizer.zero_grad(set_to_none=True)
        del loss_totals, finite_train_state, gradient_norm
        model.to(validation_device)
        tokenizer.to(validation_device)
        if device.type == "mps" and validation_device.type != "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda" and validation_device.type != "cuda":
            torch.cuda.empty_cache()
        print(
            f"epoch={epoch} training_complete token_ce={loss_values[0]:.6f} "
            f"seconds={training_seconds:.2f} "
            f"validation_device={validation_device}"
        )
        validation_started = time.monotonic()
        validation_records = predict_three_day_cases(
            model,
            tokenizer,
            validation_cases,
            device=validation_device,
            max_context=int(evaluation_config["max_context"]),
            clip=float(evaluation_config["clip"]),
            normalization_epsilon=float(
                evaluation_config["normalization_epsilon"]
            ),
            sample_count=int(evaluation_config["sample_count"]),
            temperature=float(evaluation_config["temperature"]),
            top_k=int(evaluation_config["top_k"]),
            top_p=float(evaluation_config["top_p"]),
            batch_size=int(evaluation_config["inference_batch_size"]),
            seed=int(evaluation_config["random_seed"]),
            point_estimate=str(evaluation_config["path_point_estimate"]),
            turning_point_threshold=float(
                evaluation_config["turning_point_return_threshold"]
            ),
            model_name="ce_only",
        )
        validation_metrics = three_day_metrics_with_instruments(
            validation_records
        )["pooled"]
        day3_metrics = validation_metrics["endpoints"]["day3"]
        day3_balanced_accuracy = float(
            day3_metrics["path_direction_balanced_accuracy"]
        )
        day3_return_mae = float(day3_metrics["endpoint_return_mae"])
        z_normalized_dtw = float(
            validation_metrics["path"]["mean_z_normalized_dtw_distance"]
        )
        validation_seconds = time.monotonic() - validation_started
        rank = (
            _finite_or(day3_balanced_accuracy, -float("inf")),
            -_finite_or(day3_return_mae, float("inf")),
            -_finite_or(z_normalized_dtw, float("inf")),
        )
        improved = rank > best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            best_day3_balanced_accuracy = day3_balanced_accuracy
            best_day3_return_mae = day3_return_mae
            best_z_normalized_dtw = z_normalized_dtw
            epochs_without_improvement = 0
            torch.save(
                {
                    "predictor_state": _cpu_state_dict(model),
                    "trend_head_state": None,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "lambda_dir": 0.0,
                    "validation_metrics": validation_metrics,
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train_token_ce": float(loss_values[0]),
            "train_s1_ce": float(loss_values[1]),
            "train_s2_ce": float(loss_values[2]),
            "processed_batches": processed_batches,
            "validation": validation_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "training_device": str(device),
            "validation_device": str(validation_device),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(epoch_record)
        history_path.write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
                allow_nan=True,
                default=str,
            ),
            encoding="utf-8",
        )
        torch.save(
            {
                "run_signature": run_signature,
                "predictor_state": _cpu_state_dict(model),
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
                "scheduler_state": scheduler.state_dict(),
                "data_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "device_rng_state": _device_rng_state(device),
                "epoch": epoch,
                "best_rank": best_rank,
                "best_epoch": best_epoch,
                "best_day3_balanced_accuracy": (
                    best_day3_balanced_accuracy
                ),
                "best_day3_return_mae": best_day3_return_mae,
                "best_z_normalized_dtw": best_z_normalized_dtw,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "elapsed_seconds": elapsed_before_resume
                + time.monotonic()
                - started_at,
            },
            latest_state_path,
        )
        print(
            f"epoch={epoch} token_ce={epoch_record['train_token_ce']:.6f} "
            f"day3_bal_acc={day3_balanced_accuracy:.4f} "
            f"day3_mae={day3_return_mae:.6f} z_dtw={z_normalized_dtw:.6f} "
            f"best_epoch={best_epoch} train_seconds={training_seconds:.2f} "
            f"validation_seconds={validation_seconds:.2f} "
            f"seconds={epoch_record['epoch_seconds']:.2f}"
        )
        if epochs_without_improvement >= early_stopping_patience:
            break

    elapsed_seconds = elapsed_before_resume + time.monotonic() - started_at
    if not checkpoint_path.exists():
        raise RuntimeError("CE-only training produced no checkpoint")
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_day3_balanced_accuracy": best_day3_balanced_accuracy,
        "best_day3_return_mae": best_day3_return_mae,
        "best_z_normalized_dtw": best_z_normalized_dtw,
        "elapsed_seconds": elapsed_seconds,
        "learning_rate": learning_rate,
        "seed": seed,
        "lambda_dir": 0.0,
        "metadata": checkpoint_metadata,
    }
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
            default=str,
        ),
        encoding="utf-8",
    )
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return ThreeDayTrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_day3_balanced_accuracy=best_day3_balanced_accuracy,
        best_day3_return_mae=best_day3_return_mae,
        best_z_normalized_dtw=best_z_normalized_dtw,
        history=history,
        elapsed_seconds=elapsed_seconds,
    )


def load_ce_only_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["predictor_state"])
    return checkpoint


def load_ce_only_training_result(
    output_dir: str | Path,
) -> ThreeDayTrainingResult | None:
    output_path = Path(output_dir)
    checkpoint_path = output_path / "best_model.pt"
    history_path = output_path / "history.json"
    summary_path = output_path / "summary.json"
    if not (
        checkpoint_path.exists()
        and history_path.exists()
        and summary_path.exists()
    ):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return ThreeDayTrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=int(summary["best_epoch"]),
        best_day3_balanced_accuracy=float(
            summary["best_day3_balanced_accuracy"]
        ),
        best_day3_return_mae=float(summary["best_day3_return_mae"]),
        best_z_normalized_dtw=float(summary["best_z_normalized_dtw"]),
        history=history,
        elapsed_seconds=float(summary["elapsed_seconds"]),
    )


def train_direction_predictor(
    wrapper: KronosTrendWrapper,
    tokenizer: torch.nn.Module,
    dense_dataset: DenseInstrumentWindowDataset,
    direction_dataset: ThreeDayDirectionDataset,
    validation_cases: Sequence[ThreeTradingDayCase],
    *,
    device: torch.device,
    output_dir: str | Path,
    learning_rate: float,
    seed: int,
    lambda_dir: float,
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    direction_batch_size: int,
    dense_batches_per_direction_batch: int,
    num_workers: int,
    weight_decay: float,
    gradient_clip: float,
    warmup_ratio: float,
    evaluation_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    validation_device: torch.device | None = None,
    max_train_batches: int | None = None,
) -> ThreeDayDirectionTrainingResult:
    """Fine-tune a predictor and context-only direction head with two streams."""

    if not validation_cases:
        raise ValueError("Direction training requires three-day validation cases")
    if lambda_dir <= 0:
        raise ValueError("Direction training requires a positive lambda_dir")
    if dense_dataset.lookback != direction_dataset.lookback:
        raise ValueError("Dense and direction streams must use the same lookback")
    if batch_size < 1 or direction_batch_size < 1:
        raise ValueError("Both batch sizes must be positive")
    if dense_batches_per_direction_batch < 1:
        raise ValueError("dense_batches_per_direction_batch must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_model.pt"
    latest_state_path = output_path / "latest_training_state.pt"
    history_path = output_path / "history.json"
    summary_path = output_path / "summary.json"
    validation_device = validation_device or device

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    tokenizer.to(device)
    wrapper.to(device)
    wrapper.train()

    dense_generator = torch.Generator().manual_seed(seed)
    direction_generator = torch.Generator().manual_seed(seed + 1)
    dense_loader = DataLoader(
        dense_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=dense_generator,
    )
    direction_loader = DataLoader(
        direction_dataset,
        batch_size=direction_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=direction_generator,
    )
    batches_per_epoch = len(dense_loader)
    if max_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, max_train_batches)
    total_steps = max(max_epochs * batches_per_epoch, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = torch.optim.AdamW(
        wrapper.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_with_warmup(
            step, total_steps, warmup_steps
        ),
    )

    best_rank = (-float("inf"), -float("inf"), -float("inf"))
    best_epoch = 0
    best_day3_balanced_accuracy = -float("inf")
    best_day3_return_mae = float("inf")
    best_z_normalized_dtw = float("inf")
    best_auxiliary_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    elapsed_before_resume = 0.0
    run_signature = {
        "learning_rate": learning_rate,
        "seed": seed,
        "lambda_dir": lambda_dir,
        "max_epochs": max_epochs,
        "batches_per_epoch": batches_per_epoch,
        "dense_windows": len(dense_dataset),
        "direction_cases": len(direction_dataset),
        "direction_batch_size": direction_batch_size,
        "dense_batches_per_direction_batch": dense_batches_per_direction_batch,
        "validation_cases": len(validation_cases),
        "training_device_type": device.type,
        "validation_device_type": validation_device.type,
    }
    if latest_state_path.exists() and not summary_path.exists():
        latest_state = torch.load(
            latest_state_path,
            map_location="cpu",
            weights_only=False,
        )
        if latest_state.get("run_signature") != run_signature:
            raise RuntimeError(
                f"Incomplete direction state does not match this run: "
                f"{latest_state_path}"
            )
        wrapper.load_state_dict(latest_state["wrapper_state"])
        optimizer.load_state_dict(latest_state["optimizer_state"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(latest_state["scheduler_state"])
        dense_generator.set_state(latest_state["dense_data_generator_state"])
        direction_generator.set_state(latest_state["direction_data_generator_state"])
        torch.set_rng_state(latest_state["torch_rng_state"])
        np.random.set_state(latest_state["numpy_rng_state"])
        random.setstate(latest_state["python_rng_state"])
        _restore_device_rng_state(device, latest_state["device_rng_state"])
        best_rank = tuple(latest_state["best_rank"])
        best_epoch = int(latest_state["best_epoch"])
        best_day3_balanced_accuracy = float(
            latest_state["best_day3_balanced_accuracy"]
        )
        best_day3_return_mae = float(latest_state["best_day3_return_mae"])
        best_z_normalized_dtw = float(latest_state["best_z_normalized_dtw"])
        best_auxiliary_metrics = dict(
            latest_state.get("best_auxiliary_metrics", {})
        )
        epochs_without_improvement = int(
            latest_state["epochs_without_improvement"]
        )
        history = list(latest_state["history"])
        start_epoch = int(latest_state["epoch"]) + 1
        elapsed_before_resume = float(latest_state["elapsed_seconds"])
        if epochs_without_improvement >= early_stopping_patience:
            start_epoch = max_epochs + 1
        print(
            f"resuming direction training at epoch={start_epoch} "
            f"from={latest_state_path}"
        )

    started_at = time.monotonic()
    print(
        f"direction_train instrument={dense_dataset.instrument} "
        f"fold={dense_dataset.period.fold_id} lr={learning_rate:.1e} seed={seed} "
        f"lambda_dir={lambda_dir:.3f} dense_windows={len(dense_dataset)} "
        f"direction_cases={len(direction_dataset)} "
        f"batches_per_epoch={batches_per_epoch} "
        f"dense_batches_per_direction_batch={dense_batches_per_direction_batch} "
        f"device={device}"
    )

    for epoch in range(start_epoch, max_epochs + 1):
        wrapper.to(device)
        tokenizer.to(device)
        wrapper.train()
        tokenizer.eval()
        epoch_started = time.monotonic()
        training_started = time.monotonic()
        loss_totals = torch.zeros(5, dtype=torch.float32, device=device)
        gradient_norm_total = torch.zeros((), dtype=torch.float32, device=device)
        finite_train_state = torch.ones((), dtype=torch.bool, device=device)
        processed_batches = 0
        direction_batches = 0
        direction_iterator = iter(direction_loader)

        for batch_index, (batch_x, batch_stamp) in enumerate(dense_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            batch_x = batch_x.to(device)
            batch_stamp = batch_stamp.to(device)
            with torch.no_grad():
                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_in_s1 = token_s1[:, :-1]
            token_in_s2 = token_s2[:, :-1]
            target_s1 = token_s1[:, 1:]
            target_s2 = token_s2[:, 1:]
            logits_s1, logits_s2 = wrapper(
                token_in_s1,
                token_in_s2,
                batch_stamp[:, :-1, :],
            )
            token_loss, loss_s1, loss_s2 = compute_future_token_loss(
                wrapper.predictor,
                logits_s1,
                logits_s2,
                target_s1,
                target_s2,
                lookback=dense_dataset.lookback,
            )
            direction_loss = torch.zeros((), dtype=token_loss.dtype, device=device)
            if batch_index % dense_batches_per_direction_batch == 0:
                try:
                    direction_batch = next(direction_iterator)
                except StopIteration:
                    direction_iterator = iter(direction_loader)
                    direction_batch = next(direction_iterator)
                direction_x, direction_stamp, labels, valid_mask = direction_batch
                direction_x = direction_x.to(device)
                direction_stamp = direction_stamp.to(device)
                labels = labels.to(device)
                valid_mask = valid_mask.to(device)
                direction_logits = direction_logits_from_context(
                    wrapper,
                    tokenizer,
                    direction_x,
                    direction_stamp,
                    lookback=direction_dataset.lookback,
                )
                direction_loss = masked_direction_bce_loss(
                    direction_logits,
                    labels,
                    valid_mask,
                )
                direction_batches += 1

            total_loss = token_loss + lambda_dir * direction_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                wrapper.parameters(), max_norm=gradient_clip
            )
            finite_train_state.logical_and_(
                torch.isfinite(total_loss) & torch.isfinite(gradient_norm)
            )
            optimizer.step()
            scheduler.step()

            loss_totals.add_(
                torch.stack(
                    (total_loss, token_loss, loss_s1, loss_s2, direction_loss)
                )
                .detach()
                .to(torch.float32)
            )
            gradient_norm_total.add_(gradient_norm.detach().to(torch.float32))
            processed_batches += 1
            del (
                batch_x,
                batch_stamp,
                token_s1,
                token_s2,
                token_in_s1,
                token_in_s2,
                target_s1,
                target_s2,
                logits_s1,
                logits_s2,
                token_loss,
                loss_s1,
                loss_s2,
                direction_loss,
                total_loss,
                gradient_norm,
            )

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        if processed_batches == 0:
            raise RuntimeError("Direction training processed no dense batches")
        if direction_batches == 0:
            raise RuntimeError("Direction training processed no direction batches")
        if not bool(finite_train_state.detach().cpu()):
            raise RuntimeError("Non-finite loss or gradient norm in direction training")
        loss_values = loss_totals.detach().cpu().numpy() / processed_batches
        mean_gradient_norm = float(
            gradient_norm_total.detach().cpu() / processed_batches
        )
        training_seconds = time.monotonic() - training_started
        optimizer.zero_grad(set_to_none=True)
        del loss_totals, gradient_norm_total, finite_train_state

        wrapper.to(validation_device)
        tokenizer.to(validation_device)
        if device.type == "mps" and validation_device.type != "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda" and validation_device.type != "cuda":
            torch.cuda.empty_cache()
        print(
            f"epoch={epoch} training_complete token_ce={loss_values[1]:.6f} "
            f"direction_bce={loss_values[4]:.6f} "
            f"total_loss={loss_values[0]:.6f} seconds={training_seconds:.2f} "
            f"validation_device={validation_device}"
        )
        validation_started = time.monotonic()
        validation_records = predict_three_day_cases(
            wrapper.predictor,
            tokenizer,
            validation_cases,
            device=validation_device,
            max_context=int(evaluation_config["max_context"]),
            clip=float(evaluation_config["clip"]),
            normalization_epsilon=float(
                evaluation_config["normalization_epsilon"]
            ),
            sample_count=int(evaluation_config["sample_count"]),
            temperature=float(evaluation_config["temperature"]),
            top_k=int(evaluation_config["top_k"]),
            top_p=float(evaluation_config["top_p"]),
            batch_size=int(evaluation_config["inference_batch_size"]),
            seed=int(evaluation_config["random_seed"]),
            point_estimate=str(evaluation_config["path_point_estimate"]),
            turning_point_threshold=float(
                evaluation_config["turning_point_return_threshold"]
            ),
            model_name="ce_direction",
        )
        validation_metrics = three_day_metrics_with_instruments(
            validation_records
        )["pooled"]
        auxiliary_validation_records = predict_auxiliary_direction_cases(
            wrapper,
            tokenizer,
            validation_cases,
            device=validation_device,
            clip=float(evaluation_config["clip"]),
            normalization_epsilon=float(
                evaluation_config["normalization_epsilon"]
            ),
            batch_size=int(evaluation_config["inference_batch_size"]),
        )
        auxiliary_validation_metrics = auxiliary_direction_metrics_with_instruments(
            auxiliary_validation_records
        )["pooled"]
        day3_metrics = validation_metrics["endpoints"]["day3"]
        day3_balanced_accuracy = float(
            day3_metrics["path_direction_balanced_accuracy"]
        )
        day3_return_mae = float(day3_metrics["endpoint_return_mae"])
        z_normalized_dtw = float(
            validation_metrics["path"]["mean_z_normalized_dtw_distance"]
        )
        validation_seconds = time.monotonic() - validation_started
        rank = (
            _finite_or(day3_balanced_accuracy, -float("inf")),
            -_finite_or(day3_return_mae, float("inf")),
            -_finite_or(z_normalized_dtw, float("inf")),
        )
        improved = rank > best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            best_day3_balanced_accuracy = day3_balanced_accuracy
            best_day3_return_mae = day3_return_mae
            best_z_normalized_dtw = z_normalized_dtw
            best_auxiliary_metrics = dict(auxiliary_validation_metrics)
            epochs_without_improvement = 0
            torch.save(
                {
                    "predictor_state": _cpu_state_dict(wrapper.predictor),
                    "trend_head_state": _cpu_state_dict(wrapper.trend_head),
                    "wrapper_state": _cpu_state_dict(wrapper),
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "lambda_dir": lambda_dir,
                    "validation_metrics": validation_metrics,
                    "auxiliary_validation_metrics": auxiliary_validation_metrics,
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train_total_loss": float(loss_values[0]),
            "train_token_ce": float(loss_values[1]),
            "train_s1_ce": float(loss_values[2]),
            "train_s2_ce": float(loss_values[3]),
            "train_direction_bce": float(loss_values[4]),
            "mean_gradient_norm": mean_gradient_norm,
            "processed_dense_batches": processed_batches,
            "processed_direction_batches": direction_batches,
            "validation": validation_metrics,
            "auxiliary_validation": auxiliary_validation_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "training_device": str(device),
            "validation_device": str(validation_device),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(epoch_record)
        history_path.write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
                allow_nan=True,
                default=str,
            ),
            encoding="utf-8",
        )
        torch.save(
            {
                "run_signature": run_signature,
                "wrapper_state": _cpu_state_dict(wrapper),
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
                "scheduler_state": scheduler.state_dict(),
                "dense_data_generator_state": dense_generator.get_state(),
                "direction_data_generator_state": direction_generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "device_rng_state": _device_rng_state(device),
                "epoch": epoch,
                "best_rank": best_rank,
                "best_epoch": best_epoch,
                "best_day3_balanced_accuracy": best_day3_balanced_accuracy,
                "best_day3_return_mae": best_day3_return_mae,
                "best_z_normalized_dtw": best_z_normalized_dtw,
                "best_auxiliary_metrics": best_auxiliary_metrics,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "elapsed_seconds": elapsed_before_resume
                + time.monotonic()
                - started_at,
            },
            latest_state_path,
        )
        print(
            f"epoch={epoch} token_ce={epoch_record['train_token_ce']:.6f} "
            f"direction_bce={epoch_record['train_direction_bce']:.6f} "
            f"grad_norm={mean_gradient_norm:.4f} "
            f"path_day3_bal_acc={day3_balanced_accuracy:.4f} "
            f"day3_mae={day3_return_mae:.6f} z_dtw={z_normalized_dtw:.6f} "
            f"best_epoch={best_epoch} train_seconds={training_seconds:.2f} "
            f"validation_seconds={validation_seconds:.2f} "
            f"seconds={epoch_record['epoch_seconds']:.2f}"
        )
        if epochs_without_improvement >= early_stopping_patience:
            break

    elapsed_seconds = elapsed_before_resume + time.monotonic() - started_at
    if not checkpoint_path.exists():
        raise RuntimeError("Direction training produced no checkpoint")
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_day3_balanced_accuracy": best_day3_balanced_accuracy,
        "best_day3_return_mae": best_day3_return_mae,
        "best_z_normalized_dtw": best_z_normalized_dtw,
        "best_auxiliary_metrics": best_auxiliary_metrics,
        "elapsed_seconds": elapsed_seconds,
        "learning_rate": learning_rate,
        "seed": seed,
        "lambda_dir": lambda_dir,
        "metadata": checkpoint_metadata,
    }
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
            default=str,
        ),
        encoding="utf-8",
    )
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return ThreeDayDirectionTrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_day3_balanced_accuracy=best_day3_balanced_accuracy,
        best_day3_return_mae=best_day3_return_mae,
        best_z_normalized_dtw=best_z_normalized_dtw,
        best_auxiliary_metrics=best_auxiliary_metrics,
        history=history,
        elapsed_seconds=elapsed_seconds,
    )


def load_direction_checkpoint(
    wrapper: KronosTrendWrapper,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "wrapper_state" in checkpoint:
        wrapper.load_state_dict(checkpoint["wrapper_state"])
    else:
        trend_head_state = checkpoint.get("trend_head_state")
        if trend_head_state is None:
            raise ValueError("Checkpoint does not contain a Phase 3 trend head")
        wrapper.predictor.load_state_dict(checkpoint["predictor_state"])
        wrapper.trend_head.load_state_dict(trend_head_state)
    return checkpoint


def load_direction_training_result(
    output_dir: str | Path,
) -> ThreeDayDirectionTrainingResult | None:
    output_path = Path(output_dir)
    checkpoint_path = output_path / "best_model.pt"
    history_path = output_path / "history.json"
    summary_path = output_path / "summary.json"
    if not (
        checkpoint_path.exists()
        and history_path.exists()
        and summary_path.exists()
    ):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return ThreeDayDirectionTrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=int(summary["best_epoch"]),
        best_day3_balanced_accuracy=float(
            summary["best_day3_balanced_accuracy"]
        ),
        best_day3_return_mae=float(summary["best_day3_return_mae"]),
        best_z_normalized_dtw=float(summary["best_z_normalized_dtw"]),
        best_auxiliary_metrics=dict(summary.get("best_auxiliary_metrics", {})),
        history=history,
        elapsed_seconds=float(summary["elapsed_seconds"]),
    )
