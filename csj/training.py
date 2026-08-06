from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from csj.evaluation import predict_cases, set_seed
from csj.futures_data import ForecastCase, MultiContractWindowDataset
from csj.metrics import compute_metrics


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    best_epoch: int
    best_score: float
    best_return_mae: float
    history: list[dict[str, Any]]
    elapsed_seconds: float


def _cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    denominator = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def train_predictor(
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    train_dataset: MultiContractWindowDataset,
    validation_cases: list[ForecastCase],
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
    max_train_batches: int | None = None,
) -> TrainingResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_model.pt"
    history_path = output_path / "history.json"
    summary_path = output_path / "summary.json"

    set_seed(seed)
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    model.train()
    tokenizer.to(device)
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)
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
        lr_lambda=lambda step: _cosine_with_warmup(step, total_steps, warmup_steps),
    )

    best_score = -float("inf")
    best_return_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    started_at = time.monotonic()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        processed_batches = 0
        epoch_started = time.monotonic()

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

            target_start = train_dataset.lookback - 1
            loss, _, _ = model.head.compute_loss(
                logits_s1[:, target_start:, :],
                logits_s2[:, target_start:, :],
                target_s1[:, target_start:],
                target_s2[:, target_start:],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.detach().cpu())
            processed_batches += 1
            del batch_x, batch_stamp, token_s1, token_s2, logits_s1, logits_s2, loss

        validation_records = predict_cases(
            model,
            tokenizer,
            validation_cases,
            device=device,
            max_context=int(evaluation_config["max_context"]),
            clip=float(evaluation_config["clip"]),
            sample_count=int(evaluation_config["sample_count"]),
            temperature=float(evaluation_config["temperature"]),
            top_k=int(evaluation_config["top_k"]),
            top_p=float(evaluation_config["top_p"]),
            batch_size=int(evaluation_config["inference_batch_size"]),
            seed=int(evaluation_config["random_seed"]),
        )
        validation_metrics = compute_metrics(validation_records)
        score = float(validation_metrics["direction_balanced_accuracy"])
        return_mae = float(validation_metrics["return_mae"])
        average_loss = total_loss / max(processed_batches, 1)

        improved = score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and return_mae < best_return_mae
        )
        if improved:
            best_score = score
            best_return_mae = return_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": _cpu_state_dict(model),
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": average_loss,
            "validation": validation_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(epoch_record)
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        print(
            f"epoch={epoch} loss={average_loss:.6f} "
            f"val_bal_acc={score:.4f} val_return_mae={return_mae:.6f} "
            f"best_epoch={best_epoch}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            break

    elapsed_seconds = time.monotonic() - started_at
    if not checkpoint_path.exists():
        raise RuntimeError("Training finished without producing a checkpoint")

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_return_mae": best_return_mae,
        "elapsed_seconds": elapsed_seconds,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_score=best_score,
        best_return_mae=best_return_mae,
        history=history,
        elapsed_seconds=elapsed_seconds,
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
