"""Frozen-backbone P1 Pair Probe for V3 concrete-contract panels.

P1 deliberately answers a narrower question than path generation: does one
nearest delivery-month neighbour add Day3 directional information beyond the
same target context?  The tokenizer and Kronos trunk remain frozen.  Both arms
share an identical fusion-head shape and initialization; the target-only arm
sets its neighbour branch to a fixed zero representation and never tokenizes a
neighbour stream.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from csj.v3.panel_data import PanelCase, case_arrays, normalize_context


ProbeMode = Literal["target_only_probe", "pair_probe"]


class PairProbeError(RuntimeError):
    """P1 inputs do not preserve the required paired-comparison invariants."""


def _valid_direction_cases(cases: Sequence[PanelCase]) -> tuple[PanelCase, ...]:
    return tuple(case for case in cases if case.day3_return != 0.0)


def assert_same_case_keys(
    target_only_cases: Sequence[PanelCase],
    pair_cases: Sequence[PanelCase],
) -> None:
    """Enforce the P1 rule that both arms use exactly the same case keys."""

    target_keys = [case.case_key for case in target_only_cases]
    pair_keys = [case.case_key for case in pair_cases]
    if len(set(target_keys)) != len(target_keys) or len(set(pair_keys)) != len(pair_keys):
        raise PairProbeError("P1 case keys must be unique within each arm")
    if set(target_keys) != set(pair_keys):
        missing = sorted(set(target_keys).difference(pair_keys))
        extra = sorted(set(pair_keys).difference(target_keys))
        raise PairProbeError(
            "P1 target-only/pair case keys differ "
            f"(missing_pair={len(missing)}, extra_pair={len(extra)})"
        )


class PanelProbeDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    """Context-only P1 records with independent per-contract normalization."""

    def __init__(
        self,
        cases: Sequence[PanelCase],
        *,
        mode: ProbeMode,
        clip: float = 5.0,
        epsilon: float = 1e-5,
    ) -> None:
        if mode not in ("target_only_probe", "pair_probe"):
            raise ValueError(f"Unsupported P1 mode: {mode}")
        if not cases:
            raise PairProbeError("P1 requires at least one paired case")
        if any(not case.has_pair for case in cases):
            raise PairProbeError("P1 cases must all have one valid nearest neighbour")
        lengths = {len(case.target_context) for case in cases}
        if len(lengths) != 1:
            raise PairProbeError("P1 contexts must use one fixed lookback")
        self.cases = tuple(cases)
        self.mode = mode
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.lookback = next(iter(lengths))
        self.valid_direction_count = sum(case.day3_return != 0.0 for case in self.cases)
        if self.valid_direction_count == 0:
            raise PairProbeError("P1 contains no non-zero Day3 direction labels")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        case = self.cases[index]
        target_values, target_stamps, neighbor_values, neighbor_stamps = case_arrays(
            case,
            include_neighbor=self.mode == "pair_probe",
        )
        target_normalized = normalize_context(
            target_values,
            clip=self.clip,
            epsilon=self.epsilon,
        )
        if self.mode == "pair_probe":
            assert neighbor_values is not None and neighbor_stamps is not None
            neighbor_normalized = normalize_context(
                neighbor_values,
                clip=self.clip,
                epsilon=self.epsilon,
            )
            assert case.term_structure is not None
            state = case.term_structure.as_array()
            neighbor_mask = 1.0
        else:
            # Shape-compatible zeros are intentionally never passed through the
            # neighbour tokenizer; PairProbe checks the mask before encoding.
            neighbor_normalized = np.zeros_like(target_normalized)
            neighbor_stamps = np.zeros_like(target_stamps)
            state = np.zeros(3, dtype=np.float32)
            neighbor_mask = 0.0
        day3_return = case.day3_return
        return (
            torch.from_numpy(target_normalized),
            torch.from_numpy(target_stamps),
            torch.from_numpy(neighbor_normalized),
            torch.from_numpy(neighbor_stamps),
            torch.from_numpy(state),
            torch.tensor(neighbor_mask, dtype=torch.float32),
            torch.tensor(float(day3_return > 0.0), dtype=torch.float32),
            torch.tensor(day3_return != 0.0, dtype=torch.bool),
            torch.tensor(index, dtype=torch.long),
        )


class PairProbe(nn.Module):
    """Shared frozen encoder plus a small role/state-aware P1 fusion head."""

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

    def train(self, mode: bool = True) -> "PairProbe":
        super().train(mode)
        # Frozen encoders must not acquire training-time dropout variance between
        # the two P1 arms.
        self.tokenizer.eval()
        self.predictor.eval()
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize only the lightweight trainable P1 head, not Kronos weights."""

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
            raise PairProbeError("P1 head checkpoint does not match this fusion architecture")
        with torch.no_grad():
            for name, value in state.items():
                current[name].copy_(value)

    def _encode_context(
        self,
        values: torch.Tensor,
        stamps: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3 or stamps.ndim != 3 or values.shape[:2] != stamps.shape[:2]:
            raise ValueError("P1 values and timestamps must align as [batch, time, feature]")
        with torch.no_grad():
            token_s1, token_s2 = self.tokenizer.encode(values, half=True)
            _, hidden = self.predictor.decode_s1(token_s1, token_s2, stamps)
        return hidden[:, -1, :]

    def forward(
        self,
        target_values: torch.Tensor,
        target_stamps: torch.Tensor,
        neighbor_values: torch.Tensor,
        neighbor_stamps: torch.Tensor,
        term_state: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one Day3-up logit per case.

        ``neighbor_mask == 0`` produces a fixed all-zero neighbour representation
        and skips the neighbour encoder entirely.  That property is required for
        the target-only paired control, not merely an optimization.
        """

        if term_state.ndim != 2 or term_state.shape[-1] != 3:
            raise ValueError("P1 term structure must have shape [batch, 3]")
        if neighbor_mask.ndim == 2 and neighbor_mask.shape[-1] == 1:
            neighbor_mask = neighbor_mask[:, 0]
        if neighbor_mask.ndim != 1 or neighbor_mask.shape[0] != target_values.shape[0]:
            raise ValueError("P1 neighbour mask must have shape [batch]")
        if torch.any((neighbor_mask != 0) & (neighbor_mask != 1)):
            raise ValueError("P1 neighbour mask must be binary")

        target_hidden = self._encode_context(target_values, target_stamps)
        batch_size = target_hidden.shape[0]
        neighbor_hidden = torch.zeros_like(target_hidden)
        present = torch.nonzero(neighbor_mask.to(dtype=torch.bool), as_tuple=False).flatten()
        if len(present):
            encoded_neighbor = self._encode_context(
                neighbor_values.index_select(0, present),
                neighbor_stamps.index_select(0, present),
            )
            neighbor_hidden.index_copy_(0, present, encoded_neighbor)
            neighbor_hidden = neighbor_hidden + (
                neighbor_mask[:, None]
                * self.role_embeddings.weight[1].expand(batch_size, -1)
            )
        target_hidden = target_hidden + self.role_embeddings.weight[0].expand(
            batch_size, -1
        )
        state_with_mask = torch.cat((term_state, neighbor_mask[:, None]), dim=-1)
        state_hidden = self.state_projection(state_with_mask)
        return self.fusion(torch.cat((target_hidden, neighbor_hidden, state_hidden), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class ProbeTrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 5
    weight_decay: float = 0.01
    gradient_clip: float = 3.0
    num_workers: int = 0
    seed: int = 42
    sampling_strategy: str = "prediction_day_uniform"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("P1 learning rate, batch size, and epochs must be positive")
        if self.early_stopping_patience < 1 or self.gradient_clip <= 0:
            raise ValueError("P1 early stopping and gradient clip must be positive")
        if self.sampling_strategy not in {
            "case_uniform",
            "prediction_day_uniform",
            "prediction_day_product_uniform",
        }:
            raise ValueError(
                "P1 sampling_strategy must be case_uniform, prediction_day_uniform, "
                "or prediction_day_product_uniform"
            )


@dataclass(frozen=True)
class ProbeTrainingResult:
    checkpoint_path: Path
    best_epoch: int
    best_balanced_accuracy: float
    history: list[dict[str, object]]
    sampling_summary: dict[str, object]
    elapsed_seconds: float


def _balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    if labels.shape != predictions.shape:
        raise ValueError("P1 labels and predictions must have the same shape")
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    return float(
        0.5
        * (
            np.mean(predictions[positive] == 1)
            + np.mean(predictions[negative] == 0)
        )
    )


def prediction_day_uniform_weights(cases: Sequence[PanelCase]) -> torch.Tensor:
    """Return inverse-day-count weights so each prediction day has equal mass.

    Multiple target contracts share a market shock on a prediction day.  Plain
    case-level shuffling therefore overweights days with more active contracts.
    This helper implements the V3 strategy's day-grouped sampling rule while
    preserving equal treatment of cases within any one day.
    """

    if not cases:
        raise PairProbeError("Cannot build P1 day-balanced weights from zero cases")
    days = [pd.Timestamp(case.origin_trading_day).normalize() for case in cases]
    counts = pd.Series(days).value_counts()
    weights = [1.0 / float(counts.loc[day]) for day in days]
    return torch.tensor(weights, dtype=torch.double)


def prediction_day_product_uniform_weights(cases: Sequence[PanelCase]) -> torch.Tensor:
    """Return inverse day×product-count weights for V4 shared probe training."""

    if not cases:
        raise PairProbeError("Cannot build P1 day-product-balanced weights from zero cases")
    groups = [
        (pd.Timestamp(case.origin_trading_day).normalize(), str(case.product))
        for case in cases
    ]
    counts = Counter(groups)
    weights = [1.0 / float(counts[group]) for group in groups]
    return torch.tensor(weights, dtype=torch.double)


def prediction_day_sampling_summary(
    cases: Sequence[PanelCase],
    *,
    strategy: str,
) -> dict[str, object]:
    """Persist enough sampler provenance to reproduce a P1 arm exactly."""

    if strategy not in {
        "case_uniform",
        "prediction_day_uniform",
        "prediction_day_product_uniform",
    }:
        raise PairProbeError(f"Unsupported P1 sampling strategy: {strategy!r}")
    if not cases:
        raise PairProbeError("Cannot summarize P1 sampling from zero cases")
    counts = pd.Series(
        [pd.Timestamp(case.origin_trading_day).normalize() for case in cases]
    ).value_counts()
    summary: dict[str, object] = {
        "strategy": strategy,
        "cases": int(len(cases)),
        "unique_prediction_days": int(len(counts)),
        "min_cases_per_prediction_day": int(counts.min()),
        "max_cases_per_prediction_day": int(counts.max()),
    }
    if strategy == "prediction_day_uniform":
        weights = prediction_day_uniform_weights(cases)
        day_mass = float(weights.sum().item() / len(counts))
        summary["per_prediction_day_sampling_mass"] = day_mass
    elif strategy == "prediction_day_product_uniform":
        groups = [
            (pd.Timestamp(case.origin_trading_day).normalize(), str(case.product))
            for case in cases
        ]
        group_counts = pd.Series(groups).value_counts()
        weights = prediction_day_product_uniform_weights(cases)
        summary["unique_prediction_day_product_groups"] = int(len(group_counts))
        summary["per_prediction_day_product_sampling_mass"] = float(
            weights.sum().item() / len(group_counts)
        )
    return summary


def _loader(
    dataset: PanelProbeDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    sampling_strategy: str = "case_uniform",
) -> DataLoader[Any]:
    if sampling_strategy not in {
        "case_uniform",
        "prediction_day_uniform",
        "prediction_day_product_uniform",
    }:
        raise PairProbeError(f"Unsupported P1 sampling strategy: {sampling_strategy!r}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    if shuffle and sampling_strategy in {
        "prediction_day_uniform",
        "prediction_day_product_uniform",
    }:
        weights = (
            prediction_day_uniform_weights(dataset.cases)
            if sampling_strategy == "prediction_day_uniform"
            else prediction_day_product_uniform_weights(dataset.cases)
        )
        sampler = WeightedRandomSampler(
            weights,
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
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )


def _move_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in batch)


def evaluate_probe(
    probe: PairProbe,
    cases: Sequence[PanelCase],
    *,
    mode: ProbeMode,
    device: torch.device,
    batch_size: int,
    clip: float = 5.0,
    epsilon: float = 1e-5,
) -> pd.DataFrame:
    """Evaluate a P1 arm and retain per-case records for strict pairing/bootstrap."""

    dataset = PanelProbeDataset(cases, mode=mode, clip=clip, epsilon=epsilon)
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=0)
    probe.to(device)
    probe.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            (
                target_values,
                target_stamps,
                neighbor_values,
                neighbor_stamps,
                term_state,
                neighbor_mask,
                labels,
                valid,
                indexes,
            ) = _move_batch(batch, device)
            logits = probe(
                target_values,
                target_stamps,
                neighbor_values,
                neighbor_stamps,
                term_state,
                neighbor_mask,
            )
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
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
                        "neighbor_contract_id": case.nearest_neighbor_id,
                        "neighbor_direction": (
                            "later"
                            if (case.neighbor_delta_month or 0) > 0
                            else "earlier"
                        ),
                        "actual_label": int(label),
                        "actual_direction": 1 if bool(label) else -1,
                        "valid_direction": bool(is_valid),
                        "probability_up": float(probability),
                        "predicted_label": int(probability >= 0.5),
                        "predicted_direction": 1 if probability >= 0.5 else -1,
                        "mode": mode,
                    }
                )
    records = pd.DataFrame(rows)
    return records.sort_values(["target_end_day", "target_contract_id", "case_key"], kind="stable").reset_index(drop=True)


def probe_metrics(records: pd.DataFrame) -> dict[str, object]:
    """Report Day3 classification metrics without treating zero returns as a class."""

    if records.empty:
        raise PairProbeError("Cannot calculate P1 metrics from zero records")
    valid = records.loc[records["valid_direction"]].copy()
    labels = valid["actual_label"].to_numpy(dtype=np.int8)
    predictions = valid["predicted_label"].to_numpy(dtype=np.int8)
    return {
        "samples": int(len(valid)),
        "excluded_zero_day3_returns": int(len(records) - len(valid)),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "accuracy": float(np.mean(labels == predictions)) if len(labels) else float("nan"),
        "positive_labels": int(labels.sum()),
        "negative_labels": int((labels == 0).sum()),
        "by_product": {
            str(product): probe_metrics(group.drop(columns=[], errors="ignore"))
            for product, group in valid.groupby("product", sort=True)
        }
        if len(valid["product"].unique()) > 1
        else {},
        "by_neighbor_direction": {
            str(direction): {
                "samples": int(len(group)),
                "balanced_accuracy": _balanced_accuracy(
                    group["actual_label"].to_numpy(dtype=np.int8),
                    group["predicted_label"].to_numpy(dtype=np.int8),
                ),
            }
            for direction, group in valid.groupby("neighbor_direction", sort=True)
        },
    }


def train_probe(
    probe: PairProbe,
    train_cases: Sequence[PanelCase],
    validation_cases: Sequence[PanelCase],
    *,
    mode: ProbeMode,
    config: ProbeTrainingConfig,
    device: torch.device,
    output_dir: str | Path,
    clip: float = 5.0,
    epsilon: float = 1e-5,
) -> ProbeTrainingResult:
    """Train only the P1 fusion head and checkpoint the best inner-validation arm."""

    train_dataset = PanelProbeDataset(train_cases, mode=mode, clip=clip, epsilon=epsilon)
    validation_dataset = PanelProbeDataset(
        validation_cases,
        mode=mode,
        clip=clip,
        epsilon=epsilon,
    )
    del validation_dataset  # evaluate_probe recreates this small deterministic view.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    probe.to(device)
    probe.train()
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    if not trainable:
        raise PairProbeError("P1 fusion head has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
        sampling_strategy=config.sampling_strategy,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_probe_head.pt"
    history_path = destination / "history.json"
    sampling_summary = prediction_day_sampling_summary(
        train_dataset.cases,
        strategy=config.sampling_strategy,
    )
    (destination / "sampling.json").write_text(
        json.dumps(sampling_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
                target_values,
                target_stamps,
                neighbor_values,
                neighbor_stamps,
                term_state,
                neighbor_mask,
                labels,
                valid,
                _,
            ) = _move_batch(batch, device)
            if not bool(valid.any()):
                continue
            logits = probe(
                target_values,
                target_stamps,
                neighbor_values,
                neighbor_stamps,
                term_state,
                neighbor_mask,
            )
            loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise PairProbeError("Non-finite P1 loss or gradient norm")
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            batches += 1
        if batches == 0:
            raise PairProbeError("Every P1 training batch had only zero-return labels")
        validation_records = evaluate_probe(
            probe,
            validation_cases,
            mode=mode,
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
                    "mode": mode,
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
                "validation": validation,
                "improved": improved,
            }
        )
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    if not checkpoint_path.exists():
        raise PairProbeError("P1 did not produce a valid checkpoint")
    return ProbeTrainingResult(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_balanced_accuracy=best_score,
        history=history,
        sampling_summary=sampling_summary,
        elapsed_seconds=time.monotonic() - started_at,
    )


def load_probe_head(probe: PairProbe, path: str | Path) -> dict[str, object]:
    """Load a lightweight P1 head checkpoint onto an already loaded frozen backbone."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("head_state"), dict):
        raise PairProbeError(f"Invalid P1 head checkpoint: {path}")
    probe.load_head_state_dict(checkpoint["head_state"])
    return checkpoint


def paired_block_bootstrap(
    pair_records: pd.DataFrame,
    target_only_records: pd.DataFrame,
    *,
    block_days: int,
    iterations: int = 2_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Day-block bootstrap of pair-minus-target-only balanced accuracy."""

    required = {"case_key", "target_end_day", "actual_label", "predicted_label", "valid_direction"}
    for records in (pair_records, target_only_records):
        missing = required.difference(records.columns)
        if missing:
            raise PairProbeError(f"P1 records miss columns: {sorted(missing)!r}")
    pair = pair_records.loc[pair_records["valid_direction"]].set_index("case_key", verify_integrity=True)
    target = target_only_records.loc[target_only_records["valid_direction"]].set_index("case_key", verify_integrity=True)
    if set(pair.index) != set(target.index):
        raise PairProbeError("Cannot bootstrap unpaired P1 records")
    pair = pair.sort_index()
    target = target.loc[pair.index]
    if not np.array_equal(
        pair["actual_label"].to_numpy(dtype=np.int8),
        target["actual_label"].to_numpy(dtype=np.int8),
    ):
        raise PairProbeError("P1 paired records disagree on actual Day3 labels")
    days = sorted(pd.Timestamp(day).normalize() for day in pair["target_end_day"].unique())
    if block_days < 1 or block_days > len(days):
        raise PairProbeError("Invalid P1 moving-bootstrap block length")
    pair_by_day = {
        day: pair.loc[pd.to_datetime(pair["target_end_day"]).dt.normalize() == day]
        for day in days
    }
    target_by_day = {
        day: target.loc[pd.to_datetime(target["target_end_day"]).dt.normalize() == day]
        for day in days
    }
    rng = np.random.default_rng(seed)
    blocks_needed = int(math.ceil(len(days) / block_days))
    improvements = np.full(iterations, np.nan, dtype=np.float64)
    for iteration in range(iterations):
        sampled: list[pd.Timestamp] = []
        for start in rng.integers(0, len(days), size=blocks_needed):
            sampled.extend(days[(int(start) + offset) % len(days)] for offset in range(block_days))
        sampled = sampled[: len(days)]
        labels: list[int] = []
        pair_predictions: list[int] = []
        target_predictions: list[int] = []
        for day in sampled:
            pair_day = pair_by_day[day]
            target_day = target_by_day[day].loc[pair_day.index]
            labels.extend(pair_day["actual_label"].astype(int).tolist())
            pair_predictions.extend(pair_day["predicted_label"].astype(int).tolist())
            target_predictions.extend(target_day["predicted_label"].astype(int).tolist())
        improvements[iteration] = _balanced_accuracy(
            np.asarray(labels), np.asarray(pair_predictions)
        ) - _balanced_accuracy(np.asarray(labels), np.asarray(target_predictions))
    point = _balanced_accuracy(
        pair["actual_label"].to_numpy(dtype=np.int8),
        pair["predicted_label"].to_numpy(dtype=np.int8),
    ) - _balanced_accuracy(
        pair["actual_label"].to_numpy(dtype=np.int8),
        target["predicted_label"].to_numpy(dtype=np.int8),
    )
    finite = improvements[np.isfinite(improvements)]
    if not len(finite):
        raise PairProbeError("P1 bootstrap samples never contained both direction classes")
    return {
        "samples": int(len(pair)),
        "unique_days": int(len(days)),
        "iterations": int(iterations),
        "block_days": int(block_days),
        "point_estimate": float(point),
        "ci_lower_95": float(np.quantile(finite, 0.025)),
        "ci_upper_95": float(np.quantile(finite, 0.975)),
        "probability_improvement_positive": float(np.mean(finite > 0.0)),
    }
