"""Frozen-backbone, target-only direction probe for V5.

This module intentionally contains no neighbour tensor, mask, term structure,
or pair eligibility check.  It is numerically equivalent to V4's target-only
``PairProbe`` branch when its target head state is loaded, while making an
accidental neighbour access structurally impossible in the V5 data path.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from csj.v3.panel_data import add_time_features
from csj.utils.tool import MODEL_FEATURES
from csj.v5.target_data import TargetOnlyCase


class TargetOnlyProbeError(RuntimeError):
    """The V5 target-only probe cannot satisfy its frozen protocol."""


class TargetOnlyProbeDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    """Context-only inputs with target-contract-local normalization only."""

    def __init__(
        self,
        cases: Sequence[TargetOnlyCase],
        *,
        clip: float = 5.0,
        epsilon: float = 1e-5,
    ) -> None:
        if not cases:
            raise TargetOnlyProbeError("V5 P1 requires at least one target case")
        lengths = {len(case.target_context) for case in cases}
        if len(lengths) != 1:
            raise TargetOnlyProbeError("V5 P1 contexts must use one fixed lookback")
        valid_count = sum(case.day3_return != 0.0 for case in cases)
        if valid_count == 0:
            raise TargetOnlyProbeError("V5 P1 contains no non-zero Day3 labels")
        self.cases = tuple(cases)
        self.lookback = next(iter(lengths))
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.valid_direction_count = int(valid_count)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        case = self.cases[index]
        context = add_time_features(case.target_context)
        values = context[MODEL_FEATURES].to_numpy(dtype=np.float64)
        stamps = context[["minute", "hour", "weekday", "day", "month"]].to_numpy(
            dtype=np.float32
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        normalized = np.clip(
            (values - mean) / (std + self.epsilon), -self.clip, self.clip
        ).astype(np.float32)
        day3_return = case.day3_return
        return (
            torch.from_numpy(normalized),
            torch.from_numpy(stamps),
            torch.tensor(float(day3_return > 0.0), dtype=torch.float32),
            torch.tensor(day3_return != 0.0, dtype=torch.bool),
            torch.tensor(index, dtype=torch.long),
        )


class TargetOnlyProbe(nn.Module):
    """The V4 target-only fusion-head path, factored into a single stream."""

    def __init__(
        self,
        tokenizer: nn.Module,
        predictor: nn.Module,
        *,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(predictor, "d_model"):
            raise ValueError("Kronos predictor must expose d_model")
        if fusion_hidden_dim < 1:
            raise ValueError("fusion_hidden_dim must be positive")
        self.tokenizer = tokenizer
        self.predictor = predictor
        self.d_model = int(predictor.d_model)
        # These names/shapes intentionally match PairProbe so a V4 target-only
        # head can be transferred without changing its target branch semantics.
        self.role_embeddings = nn.Embedding(2, self.d_model)
        self.state_projection = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(fusion_hidden_dim, 1),
        )
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.tokenizer.requires_grad_(False)
        self.predictor.requires_grad_(False)
        self.tokenizer.eval()
        self.predictor.eval()

    def train(self, mode: bool = True) -> "TargetOnlyProbe":
        super().train(mode)
        self.tokenizer.eval()
        self.predictor.eval()
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if not name.startswith("tokenizer.") and not name.startswith("predictor.")
        }

    def load_head_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        expected = {
            name
            for name in current
            if not name.startswith("tokenizer.") and not name.startswith("predictor.")
        }
        if set(state) != expected:
            raise TargetOnlyProbeError("V5 probe head checkpoint does not match this architecture")
        with torch.no_grad():
            for name, value in state.items():
                current[name].copy_(value)

    def _encode_context(self, values: torch.Tensor, stamps: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or stamps.ndim != 3 or values.shape[:2] != stamps.shape[:2]:
            raise ValueError("V5 P1 values and timestamps must align as [batch, time, feature]")
        with torch.no_grad():
            token_s1, token_s2 = self.tokenizer.encode(values, half=True)
            _, hidden = self.predictor.decode_s1(token_s1, token_s2, stamps)
        return hidden[:, -1, :]

    def forward(
        self,
        target_values: torch.Tensor,
        target_stamps: torch.Tensor,
    ) -> torch.Tensor:
        """Run V4's target-only branch from target-contract inputs only.

        The zero neighbour representation and zero state are created inside
        this one-stream module.  They exist only because V4's target branch
        had that fixed control representation; V5 callers cannot provide,
        load, validate, or encode any neighbour input.
        """
        target_hidden = self._encode_context(target_values, target_stamps)
        batch_size = target_hidden.shape[0]
        target_hidden = target_hidden + self.role_embeddings.weight[0].expand(batch_size, -1)
        neighbor_hidden = torch.zeros_like(target_hidden)
        fixed_state = torch.zeros(
            (batch_size, 4), dtype=target_hidden.dtype, device=target_hidden.device
        )
        state_hidden = self.state_projection(fixed_state)
        return self.fusion(
            torch.cat((target_hidden, neighbor_hidden, state_hidden), dim=-1)
        ).squeeze(-1)


@dataclass(frozen=True)
class TargetProbeTrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 5
    weight_decay: float = 0.01
    gradient_clip: float = 3.0
    num_workers: int = 0
    seed: int = 42
    sampling_strategy: str = "prediction_day_product_uniform"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("V5 P1 learning rate, batch size, and epochs must be positive")
        if self.early_stopping_patience < 1 or self.gradient_clip <= 0.0:
            raise ValueError("V5 P1 early stopping and gradient clip must be positive")
        if self.sampling_strategy != "prediction_day_product_uniform":
            raise ValueError("V5 only permits prediction_day_product_uniform sampling")


@dataclass(frozen=True)
class TargetProbeTrainingResult:
    checkpoint_path: Path
    best_epoch: int
    best_balanced_accuracy: float
    history: list[dict[str, object]]
    sampling_summary: dict[str, object]
    elapsed_seconds: float


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    if labels.shape != predictions.shape:
        raise ValueError("V5 P1 labels and predictions must have matching shapes")
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    return float(
        0.5 * (np.mean(predictions[positive] == 1) + np.mean(predictions[negative] == 0))
    )


def prediction_day_product_uniform_weights(cases: Sequence[TargetOnlyCase]) -> torch.Tensor:
    """Give each prediction-day × primary-product group equal sampling mass."""

    if not cases:
        raise TargetOnlyProbeError("Cannot build V5 P1 weights from zero cases")
    groups = [
        (pd.Timestamp(case.origin_trading_day).normalize(), str(case.product))
        for case in cases
    ]
    counts = Counter(groups)
    return torch.tensor([1.0 / float(counts[group]) for group in groups], dtype=torch.double)


def prediction_day_sampling_summary(cases: Sequence[TargetOnlyCase]) -> dict[str, object]:
    if not cases:
        raise TargetOnlyProbeError("Cannot summarize V5 P1 sampling from zero cases")
    days = [pd.Timestamp(case.origin_trading_day).normalize() for case in cases]
    groups = [(day, str(case.product)) for day, case in zip(days, cases, strict=True)]
    day_counts = pd.Series(days).value_counts()
    group_counts = pd.Series(groups).value_counts()
    weights = prediction_day_product_uniform_weights(cases)
    return {
        "strategy": "prediction_day_product_uniform",
        "cases": int(len(cases)),
        "unique_prediction_days": int(len(day_counts)),
        "min_cases_per_prediction_day": int(day_counts.min()),
        "max_cases_per_prediction_day": int(day_counts.max()),
        "unique_prediction_day_product_groups": int(len(group_counts)),
        "per_prediction_day_product_sampling_mass": float(
            weights.sum().item() / len(group_counts)
        ),
    }


def _loader(
    dataset: TargetOnlyProbeDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    if shuffle:
        sampler = WeightedRandomSampler(
            prediction_day_product_uniform_weights(dataset.cases),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in batch)


def evaluate_target_probe(
    probe: TargetOnlyProbe,
    cases: Sequence[TargetOnlyCase],
    *,
    device: torch.device,
    batch_size: int,
    clip: float = 5.0,
    epsilon: float = 1e-5,
) -> pd.DataFrame:
    """Evaluate a target-only probe with its fixed 0.5 decision threshold."""

    dataset = TargetOnlyProbeDataset(cases, clip=clip, epsilon=epsilon)
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=0)
    probe.to(device)
    probe.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            (
                values,
                stamps,
                labels,
                valid,
                indexes,
            ) = _move_batch(batch, device)
            probabilities = torch.sigmoid(probe(values, stamps)).detach().cpu().numpy()
            label_values = labels.detach().cpu().numpy()
            valid_values = valid.detach().cpu().numpy()
            for item_index, probability, label, is_valid in zip(
                indexes.detach().cpu().tolist(),
                probabilities.tolist(),
                label_values.tolist(),
                valid_values.tolist(),
                strict=True,
            ):
                case = dataset.cases[int(item_index)]
                rows.append(
                    {
                        "case_key": case.case_key,
                        "target_contract_id": case.target_contract_id,
                        "product": case.product,
                        "target_end_day": case.target_end_day,
                        "actual_label": int(label),
                        "actual_direction": 1 if bool(label) else -1,
                        "valid_direction": bool(is_valid),
                        "probability_up": float(probability),
                        "predicted_label": int(probability >= 0.5),
                        "predicted_direction": 1 if probability >= 0.5 else -1,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["target_end_day", "target_contract_id", "case_key"], kind="stable"
    ).reset_index(drop=True)


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Tie-aware Mann-Whitney ROC-AUC without a scikit-learn dependency."""

    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = pd.Series(probabilities).rank(method="average").to_numpy(dtype=np.float64)
    return float((ranks[positive].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def probe_metrics(records: pd.DataFrame) -> dict[str, object]:
    """V5 directional classification metrics, including calibration metrics."""

    if records.empty:
        raise TargetOnlyProbeError("Cannot calculate V5 P1 metrics from zero records")
    valid = records.loc[records["valid_direction"].astype(bool)].copy()
    labels = valid["actual_label"].to_numpy(dtype=np.int8)
    predicted = valid["predicted_label"].to_numpy(dtype=np.int8)
    probabilities = valid["probability_up"].to_numpy(dtype=np.float64)
    brier = float(np.mean((probabilities - labels) ** 2)) if len(labels) else float("nan")
    return {
        "samples": int(len(valid)),
        "excluded_zero_day3_returns": int(len(records) - len(valid)),
        "balanced_accuracy": _balanced_accuracy(labels, predicted),
        "accuracy": float(np.mean(labels == predicted)) if len(labels) else float("nan"),
        "roc_auc": _roc_auc(labels, probabilities),
        "brier_score": brier,
        "positive_labels": int(labels.sum()),
        "negative_labels": int((labels == 0).sum()),
        "by_product": {
            str(product): probe_metrics(group)
            for product, group in valid.groupby("product", sort=True)
        }
        if len(valid["product"].unique()) > 1
        else {},
    }


def train_target_probe(
    probe: TargetOnlyProbe,
    train_cases: Sequence[TargetOnlyCase],
    validation_cases: Sequence[TargetOnlyCase],
    *,
    config: TargetProbeTrainingConfig,
    device: torch.device,
    output_dir: str | Path,
    clip: float = 5.0,
    epsilon: float = 1e-5,
) -> TargetProbeTrainingResult:
    """Train only the V5 target-only head and save its best validation BA head."""

    train_dataset = TargetOnlyProbeDataset(train_cases, clip=clip, epsilon=epsilon)
    # Validate the inner split before any expensive epoch starts.
    TargetOnlyProbeDataset(validation_cases, clip=clip, epsilon=epsilon)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    probe.to(device)
    probe.train()
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not trainable:
        raise TargetOnlyProbeError("V5 P1 target-only head has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = _loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_probe_head.pt"
    history_path = destination / "history.json"
    sampling_summary = prediction_day_sampling_summary(train_dataset.cases)
    history: list[dict[str, object]] = []
    best_epoch = 0
    best_score = -float("inf")
    epochs_without_improvement = 0
    started_at = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        probe.train()
        loss_total = 0.0
        batches = 0
        for batch in loader:
            (
                values,
                stamps,
                labels,
                valid,
                _,
            ) = _move_batch(batch, device)
            if not bool(valid.any()):
                continue
            logits = probe(values, stamps)
            loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise TargetOnlyProbeError("Non-finite V5 P1 loss or gradient norm")
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            batches += 1
        if batches == 0:
            raise TargetOnlyProbeError("Every V5 P1 training batch had only zero-return labels")
        validation_records = evaluate_target_probe(
            probe,
            validation_cases,
            device=device,
            batch_size=config.batch_size,
            clip=clip,
            epsilon=epsilon,
        )
        validation = probe_metrics(validation_records)
        score = float(validation["balanced_accuracy"])
        finite_score = score if math.isfinite(score) else -float("inf")
        improved = finite_score > best_score
        if improved:
            best_score = finite_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "schema_version": 1,
                    "mode": "target_only_probe",
                    "epoch": epoch,
                    "head_state": probe.head_state_dict(),
                    "validation": validation,
                    "config": config.__dict__,
                    "sampling": sampling_summary,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train_bce": loss_total / batches,
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "improved": improved,
            }
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    if not checkpoint_path.is_file():
        raise TargetOnlyProbeError("V5 P1 did not produce a valid checkpoint")
    # Persist one compact training trace only after the chosen checkpoint is
    # known.  Rewriting it every epoch produces no additional recoverable
    # information and is unnecessary on the CUDA run volume.
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    return TargetProbeTrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_balanced_accuracy=best_score,
        history=history,
        sampling_summary=sampling_summary,
        elapsed_seconds=time.monotonic() - started_at,
    )


def load_target_probe_head(probe: TargetOnlyProbe, path: str | Path) -> dict[str, object]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("head_state"), dict):
        raise TargetOnlyProbeError(f"Invalid V5 P1 head checkpoint: {path}")
    probe.load_head_state_dict(checkpoint["head_state"])
    return checkpoint


__all__ = [
    "TargetOnlyProbe",
    "TargetOnlyProbeDataset",
    "TargetOnlyProbeError",
    "TargetProbeTrainingConfig",
    "TargetProbeTrainingResult",
    "evaluate_target_probe",
    "load_target_probe_head",
    "prediction_day_product_uniform_weights",
    "prediction_day_sampling_summary",
    "probe_metrics",
    "train_target_probe",
]
