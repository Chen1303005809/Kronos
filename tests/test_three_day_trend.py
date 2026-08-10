from __future__ import annotations

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import csj.three_day_evaluation as three_day_evaluation
import csj.three_day_training as three_day_training
from csj.config import load_config
from csj.three_day_experiment import (
    _phase3_bootstrap,
    _reference_path_records_are_compatible,
)
from csj.futures_data import (
    CasePeriod,
    DenseInstrumentWindowDataset,
    ThreeDayDirectionDataset,
    ThreeTradingDayCase,
    build_expanding_walk_forward_folds,
    build_three_trading_day_cases,
    day_end_indices_from_bar_counts,
    fit_context_normalization,
    group_three_day_cases_by_length,
)
from csj.metrics import (
    compute_three_day_record_metrics,
    compute_three_day_path_metrics,
    return_path_correlation,
    return_space_dtw_distance,
    slope_sign_agreement,
    turning_point_similarity,
    z_normalized_dtw_distance,
)
from csj.three_day_evaluation import (
    make_three_day_baselines,
    predict_three_day_cases,
)
from csj.three_day_training import (
    auxiliary_direction_metrics_with_instruments,
    compute_future_token_loss,
    direction_logits_from_context,
    load_direction_checkpoint,
    train_ce_only_predictor,
    train_direction_predictor,
)
from csj.trend_model import KronosTrendWrapper
from csj.utils.tool import MODEL_FEATURES
from model.kronos import top_k_top_p_filtering
from model.module import RotaryPositionalEmbedding


def _synthetic_frame(instrument: str, bar_counts: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_number = 0
    for day, bar_count in zip(
        pd.bdate_range("2024-01-02", periods=len(bar_counts)),
        bar_counts,
        strict=True,
    ):
        for bar_number in range(bar_count):
            close = 100.0 + row_number * 0.1
            rows.append(
                {
                    "instrument": instrument,
                    "timestamps": day.replace(hour=int(9 + bar_number)),
                    "trading_day": day,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000.0 + row_number,
                    "amount": 10_000.0 + row_number * 10,
                }
            )
            row_number += 1
    return pd.DataFrame(rows)


def _case_period(frame: pd.DataFrame, start: int, end: int) -> CasePeriod:
    days = list(frame["trading_day"].drop_duplicates())
    return CasePeriod(
        split="evaluation",
        start_day=pd.Timestamp(days[start]),
        end_day=pd.Timestamp(days[end]),
        fold_id="fold_test",
    )


def test_three_day_bar_combinations_have_locked_lengths_and_endpoints() -> None:
    lengths: set[int] = set()
    for counts in product((5, 7), repeat=3):
        indices = day_end_indices_from_bar_counts(counts)
        expected = tuple(np.cumsum(counts) - 1)
        assert indices == expected
        assert indices[-1] == sum(counts) - 1
        lengths.add(sum(counts))
    assert lengths == {15, 17, 19, 21}


def test_sampling_filters_use_functional_mask_with_same_kept_tokens() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    top_k = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)
    top_p = top_k_top_p_filtering(logits, top_k=0, top_p=0.8)

    expected = torch.tensor([[-torch.inf, 4.0, -torch.inf, 3.0]])
    torch.testing.assert_close(top_k, expected)
    torch.testing.assert_close(top_p, expected)


def test_rotary_cache_tracks_module_dtype_during_device_style_migration() -> None:
    rotary = RotaryPositionalEmbedding(4)
    query = torch.ones(1, 1, 3, 4)

    rotary(query, query)
    assert rotary.cos_cached is not None
    assert "cos_cached" in dict(rotary.named_buffers())

    rotary.to(dtype=torch.float64)
    migrated_query = query.to(dtype=torch.float64)
    output_query, output_key = rotary(migrated_query, migrated_query)

    assert output_query.dtype == torch.float64
    assert output_key.dtype == torch.float64
    assert rotary.cos_cached is not None
    assert rotary.cos_cached.dtype == torch.float64


def test_ce_training_resumes_from_complete_epoch_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ToyDataset(torch.utils.data.Dataset):
        instrument = "toy"
        period = SimpleNamespace(fold_id="fold_test")
        lookback = 2

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            values = torch.zeros(3, 6)
            values[:, 0] = torch.tensor([0.0, 1.0, float(index % 2)])
            return values, torch.zeros(3, 5)

    class ToyTokenizer(nn.Module):
        def encode(
            self,
            values: torch.Tensor,
            half: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del half
            token = (values[..., 0] > 0).to(torch.long)
            return token, token

    class ToyHead(nn.Module):
        def compute_loss(
            self,
            logits_s1: torch.Tensor,
            logits_s2: torch.Tensor,
            target_s1: torch.Tensor,
            target_s2: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del padding_mask
            loss_s1 = nn.functional.cross_entropy(
                logits_s1.reshape(-1, 2), target_s1.reshape(-1)
            )
            loss_s2 = nn.functional.cross_entropy(
                logits_s2.reshape(-1, 2), target_s2.reshape(-1)
            )
            return loss_s1 + loss_s2, loss_s1, loss_s2

    class ToyPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection_s1 = nn.Linear(2, 2)
            self.projection_s2 = nn.Linear(2, 2)
            self.head = ToyHead()

        def forward(
            self,
            token_s1: torch.Tensor,
            token_s2: torch.Tensor,
            stamp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del stamp
            encoded_s1 = nn.functional.one_hot(token_s1, 2).to(torch.float32)
            encoded_s2 = nn.functional.one_hot(token_s2, 2).to(torch.float32)
            return self.projection_s1(encoded_s1), self.projection_s2(encoded_s2)

    validation_calls = 0
    fail_second_validation = True

    def fake_predict(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal validation_calls
        del args, kwargs
        validation_calls += 1
        if fail_second_validation and validation_calls == 2:
            raise RuntimeError("intentional validation interruption")
        return pd.DataFrame({"placeholder": [1]})

    def fake_metrics(records: pd.DataFrame) -> dict[str, object]:
        assert len(records) == 1
        return {
            "pooled": {
                "endpoints": {
                    "day3": {
                        "path_direction_balanced_accuracy": 0.5,
                        "endpoint_return_mae": 0.1,
                    }
                },
                "path": {"mean_z_normalized_dtw_distance": 0.2},
            }
        }

    monkeypatch.setattr(three_day_training, "predict_three_day_cases", fake_predict)
    monkeypatch.setattr(
        three_day_training,
        "three_day_metrics_with_instruments",
        fake_metrics,
    )
    output_dir = tmp_path / "resume"
    arguments = {
        "device": torch.device("cpu"),
        "output_dir": output_dir,
        "learning_rate": 1e-3,
        "seed": 42,
        "max_epochs": 2,
        "early_stopping_patience": 3,
        "batch_size": 2,
        "num_workers": 0,
        "weight_decay": 0.0,
        "gradient_clip": 3.0,
        "warmup_ratio": 0.0,
        "evaluation_config": {
            "max_context": 8,
            "clip": 5.0,
            "normalization_epsilon": 1e-5,
            "sample_count": 1,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.9,
            "inference_batch_size": 1,
            "random_seed": 7,
            "path_point_estimate": "median",
            "turning_point_return_threshold": 0.0005,
        },
        "checkpoint_metadata": {"test": True},
        "validation_device": torch.device("cpu"),
    }

    with pytest.raises(RuntimeError, match="intentional validation interruption"):
        train_ce_only_predictor(
            ToyPredictor(),
            ToyTokenizer(),
            ToyDataset(),
            [object()],
            **arguments,
        )

    assert (output_dir / "latest_training_state.pt").exists()
    assert not (output_dir / "summary.json").exists()
    fail_second_validation = False
    result = train_ce_only_predictor(
        ToyPredictor(),
        ToyTokenizer(),
        ToyDataset(),
        [object()],
        **arguments,
    )

    assert [record["epoch"] for record in result.history] == [1, 2]
    assert (output_dir / "summary.json").exists()


def test_three_day_builder_keeps_exact_context_and_complete_day_end_indices() -> None:
    frame = _synthetic_frame("alpha", [7] * 37 + [5, 7, 5, 7])
    cases = build_three_trading_day_cases(
        {"alpha": frame}, _case_period(frame, 37, 40), lookback=256
    )

    assert [case.pred_len for case in cases] == [17, 19]
    assert cases[0].day_end_indices == (4, 11, 16)
    assert cases[1].day_end_indices == (6, 11, 18)
    assert set(group_three_day_cases_by_length(cases)) == {17, 19}
    for case in cases:
        assert len(case.context) == 256
        assert case.context["timestamps"].max() < case.target["timestamps"].min()
        assert case.day_end_indices[-1] == len(case.target) - 1


def test_three_day_builder_never_crosses_instrument() -> None:
    counts = [7] * 37 + [5, 7, 5]
    alpha = _synthetic_frame("alpha", counts)
    beta = _synthetic_frame("beta", counts)
    beta.loc[:, MODEL_FEATURES] = beta[MODEL_FEATURES] + 10_000.0
    period = _case_period(alpha, 37, 39)

    cases = build_three_trading_day_cases(
        {"alpha": alpha, "beta": beta}, period, lookback=256
    )

    assert {case.instrument for case in cases} == {"alpha", "beta"}
    for case in cases:
        assert set(case.context["instrument"]) == {case.instrument}
        assert set(case.target["instrument"]) == {case.instrument}
        if case.instrument == "alpha":
            assert float(case.target["close"].max()) < 10_000.0
        else:
            assert float(case.target["close"].min()) > 10_000.0


def test_three_day_builder_never_crosses_split_or_fold() -> None:
    frame = _synthetic_frame("alpha", [7] * 45)
    period = _case_period(frame, 37, 41)

    cases = build_three_trading_day_cases({"alpha": frame}, period, lookback=256)

    assert len(cases) == 3
    for case in cases:
        assert case.split == "evaluation"
        assert case.fold_id == "fold_test"
        assert all(period.contains(day) for day in case.target_days)
        assert case.target_days[-1] <= period.end_day


def test_walk_forward_folds_are_complete_and_evaluation_cases_stay_inside() -> None:
    counts = [7] * 50
    frames = {
        "alpha": _synthetic_frame("alpha", counts),
        "beta": _synthetic_frame("beta", counts),
    }
    folds = build_expanding_walk_forward_folds(
        frames,
        minimum_train_days=20,
        evaluation_days=10,
        step_days=10,
        inner_validation_days=5,
    )

    assert len(folds) == 3
    for fold in folds:
        assert fold.fit_period.end_day < fold.inner_validation_period.start_day
        assert fold.inner_validation_period.end_day < fold.evaluation_period.start_day
        cases = build_three_trading_day_cases(
            frames, fold.evaluation_period, lookback=100
        )
        assert cases
        assert all(case.fold_id == fold.fold_id for case in cases)
        assert all(
            fold.evaluation_period.contains(day)
            for case in cases
            for day in case.target_days
        )


def test_dense_instrument_stream_uses_21_targets_without_crossing_fold() -> None:
    frame = _synthetic_frame("alpha", [7] * 45)
    period = _case_period(frame, 37, 41)
    dataset = DenseInstrumentWindowDataset(
        frame,
        period,
        instrument="alpha",
        lookback=256,
        horizon=21,
        clip=5.0,
    )

    normalized, stamps = dataset[0]
    assert normalized.shape == (277, 6)
    assert stamps.shape == (277, 5)
    assert np.max(np.abs(normalized.numpy())) <= 5.0
    for forecast_start in dataset.forecast_starts:
        target_days = dataset.trading_days[forecast_start : forecast_start + 21]
        assert period.contains(pd.Timestamp(target_days[0]))
        assert period.contains(pd.Timestamp(target_days[-1]))


def test_normalization_statistics_use_context_only() -> None:
    frame = _synthetic_frame("alpha", [7] * 37 + [5, 7, 5])
    case = build_three_trading_day_cases(
        {"alpha": frame}, _case_period(frame, 37, 39), lookback=256
    )[0]
    stats = fit_context_normalization(case.context, clip=5.0)
    modified_target = case.target[MODEL_FEATURES].to_numpy(dtype=np.float64) * 1_000
    repeated_stats = fit_context_normalization(case.context, clip=5.0)

    np.testing.assert_array_equal(stats.mean, repeated_stats.mean)
    np.testing.assert_array_equal(stats.std, repeated_stats.std)
    assert stats.clipping_mask(modified_target).any()


def test_inverse_normalization_is_consistent_for_longest_21_bar_target() -> None:
    rng = np.random.default_rng(7)
    context = rng.normal(size=(256, 6)) * np.array([2, 3, 4, 5, 20, 100])
    normalized_target = rng.uniform(-4.5, 4.5, size=(21, 6))
    stats = fit_context_normalization(context, clip=5.0)

    restored = stats.inverse(normalized_target)
    roundtrip = stats.transform(restored, apply_clip=False)

    assert restored.shape == (21, 6)
    np.testing.assert_allclose(roundtrip, normalized_target, rtol=1e-12, atol=1e-12)


def test_direction_dataset_exposes_only_context_and_precomputed_labels() -> None:
    frame = _synthetic_frame("alpha", [7] * 37 + [5, 7, 5])
    case = build_three_trading_day_cases(
        {"alpha": frame}, _case_period(frame, 37, 39), lookback=256
    )[0]
    dataset = ThreeDayDirectionDataset([case], lookback=256, clip=5.0)

    context, stamps, labels, valid_mask = dataset[0]
    original_context = context.clone()
    original_labels = labels.clone()
    case.target.loc[:, MODEL_FEATURES] = case.target[MODEL_FEATURES] * 100_000.0
    changed_context, changed_stamps, changed_labels, changed_mask = dataset[0]

    assert context.shape == (256, 6)
    assert stamps.shape == (256, 5)
    assert labels.shape == (3,)
    assert valid_mask.dtype == torch.bool
    torch.testing.assert_close(changed_context, original_context)
    torch.testing.assert_close(changed_stamps, stamps)
    torch.testing.assert_close(changed_labels, original_labels)
    torch.testing.assert_close(changed_mask, valid_mask)


def test_auxiliary_direction_metrics_keep_head_results_separate() -> None:
    records = pd.DataFrame(
        {
            "instrument": ["alpha", "alpha", "beta", "beta"],
            "day1_actual_direction": [1, -1, 1, -1],
            "aux_day1_direction": [1, -1, -1, -1],
            "aux_day1_up_probability": [0.8, 0.2, 0.3, 0.1],
            "aux_day1_bce": [0.2, 0.2, 1.2, 0.1],
            "day2_actual_direction": [1, -1, 1, -1],
            "aux_day2_direction": [1, -1, 1, -1],
            "aux_day2_up_probability": [0.8, 0.2, 0.7, 0.1],
            "aux_day2_bce": [0.2, 0.2, 0.3, 0.1],
            "day3_actual_direction": [1, -1, 1, -1],
            "aux_day3_direction": [1, -1, -1, -1],
            "aux_day3_up_probability": [0.8, 0.2, 0.3, 0.1],
            "aux_day3_bce": [0.2, 0.2, 1.2, 0.1],
        }
    )

    metrics = auxiliary_direction_metrics_with_instruments(records)

    assert metrics["pooled"]["endpoints"]["day1"][
        "aux_direction_balanced_accuracy"
    ] == pytest.approx(0.75)
    assert metrics["pooled"]["endpoints"]["day3"][
        "aux_direction_accuracy"
    ] == pytest.approx(0.75)
    assert metrics["alpha"]["endpoints"]["day2"][
        "aux_direction_balanced_accuracy"
    ] == pytest.approx(1.0)


def test_phase3_reference_cache_requires_path_fields_and_matching_cases() -> None:
    target_day = pd.Timestamp("2024-04-01")
    row: dict[str, object] = {"instrument": "alpha", "target_day": target_day}
    for day_number in (1, 2, 3):
        row.update(
            {
                f"day{day_number}_actual_return": 0.01,
                f"day{day_number}_actual_direction": 1,
                f"day{day_number}_predicted_return": 0.02,
                f"day{day_number}_path_direction": 1,
                f"day{day_number}_up_probability": 0.7,
            }
        )
    records = pd.DataFrame([row])

    compatible, reason = _reference_path_records_are_compatible(
        records,
        expected_keys={("alpha", target_day)},
    )
    assert compatible, reason

    compatible, reason = _reference_path_records_are_compatible(
        records,
        expected_keys={("alpha", pd.Timestamp("2024-04-02"))},
    )
    assert not compatible
    assert "case keys differ" in reason

    compatible, reason = _reference_path_records_are_compatible(
        records.drop(columns=["day3_up_probability"]),
        expected_keys={("alpha", target_day)},
    )
    assert not compatible
    assert "day3_up_probability" in reason


def test_phase3_bootstrap_accepts_direction_only_day3_records() -> None:
    target_days = pd.date_range("2024-04-01", periods=4, freq="B")
    actual = [-1, 1, -1, 1]
    phase3 = pd.DataFrame(
        {
            "instrument": ["alpha"] * len(target_days),
            "target_day": target_days,
            "day3_actual_direction": actual,
            "day3_path_direction": actual,
        }
    )
    phase2 = phase3.copy()
    phase2["day3_path_direction"] = [-value for value in actual]

    bootstrap = _phase3_bootstrap(
        phase3,
        phase2,
        iterations=20,
        block_days=[2],
        seed=7,
    )

    result = bootstrap["block_2"]
    assert result["point_estimate"] == pytest.approx(1.0)
    assert result["ci_lower_95"] == pytest.approx(1.0)
    assert result["ci_upper_95"] == pytest.approx(1.0)


def test_direction_training_saves_predictor_and_trend_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ToyDenseDataset(torch.utils.data.Dataset):
        instrument = "toy"
        period = SimpleNamespace(fold_id="fold_test")
        lookback = 2

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            del index
            values = torch.zeros((3, 6), dtype=torch.float32)
            values[:, 0] = torch.tensor([0.0, 1.0, 1.0])
            values[:, 1] = torch.tensor([1.0, 0.0, 1.0])
            return values, torch.zeros((3, 5), dtype=torch.float32)

    class ToyDirectionDataset(torch.utils.data.Dataset):
        lookback = 2

        def __len__(self) -> int:
            return 2

        def __getitem__(
            self, index: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            del index
            values = torch.zeros((2, 6), dtype=torch.float32)
            values[:, 0] = torch.tensor([0.0, 1.0])
            values[:, 1] = torch.tensor([1.0, 0.0])
            return (
                values,
                torch.zeros((2, 5), dtype=torch.float32),
                torch.tensor([1.0, 0.0, 1.0]),
                torch.tensor([True, True, True]),
            )

    class ToyTokenizer(nn.Module):
        def encode(
            self, values: torch.Tensor, half: bool = True
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del half
            return (
                (values[..., 0] > 0.5).to(torch.long),
                (values[..., 1] > 0.5).to(torch.long),
            )

    class ToyHead(nn.Module):
        def compute_loss(
            self,
            logits_s1: torch.Tensor,
            logits_s2: torch.Tensor,
            target_s1: torch.Tensor,
            target_s2: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del padding_mask
            loss_s1 = nn.functional.cross_entropy(
                logits_s1.reshape(-1, 2), target_s1.reshape(-1)
            )
            loss_s2 = nn.functional.cross_entropy(
                logits_s2.reshape(-1, 2), target_s2.reshape(-1)
            )
            return loss_s1 + loss_s2, loss_s1, loss_s2

    class ToyPredictor(nn.Module):
        d_model = 2

        def __init__(self) -> None:
            super().__init__()
            self.projection_s1 = nn.Linear(2, 2)
            self.projection_s2 = nn.Linear(2, 2)
            self.context_projection = nn.Linear(2, 2)
            self.head = ToyHead()

        def forward(
            self,
            token_s1: torch.Tensor,
            token_s2: torch.Tensor,
            stamp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del stamp
            s1 = nn.functional.one_hot(token_s1, 2).to(torch.float32)
            s2 = nn.functional.one_hot(token_s2, 2).to(torch.float32)
            return self.projection_s1(s1), self.projection_s2(s2)

        def decode_s1(
            self,
            token_s1: torch.Tensor,
            token_s2: torch.Tensor,
            stamp: torch.Tensor | None,
            padding_mask: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del token_s2, stamp, padding_mask
            hidden = self.context_projection(
                nn.functional.one_hot(token_s1, 2).to(torch.float32)
            )
            return torch.zeros((*token_s1.shape, 2)), hidden

    def fake_path_prediction(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        return pd.DataFrame({"placeholder": [1]})

    def fake_path_metrics(records: pd.DataFrame) -> dict[str, object]:
        assert len(records) == 1
        return {
            "pooled": {
                "endpoints": {
                    "day3": {
                        "path_direction_balanced_accuracy": 0.6,
                        "endpoint_return_mae": 0.1,
                    }
                },
                "path": {"mean_z_normalized_dtw_distance": 0.2},
            }
        }

    def fake_auxiliary_prediction(
        *args: object, **kwargs: object
    ) -> pd.DataFrame:
        del args, kwargs
        return pd.DataFrame({"placeholder": [1]})

    def fake_auxiliary_metrics(records: pd.DataFrame) -> dict[str, object]:
        assert len(records) == 1
        return {
            "pooled": {
                "endpoints": {
                    "day3": {"aux_direction_balanced_accuracy": 0.7}
                }
            }
        }

    monkeypatch.setattr(
        three_day_training, "predict_three_day_cases", fake_path_prediction
    )
    monkeypatch.setattr(
        three_day_training,
        "three_day_metrics_with_instruments",
        fake_path_metrics,
    )
    monkeypatch.setattr(
        three_day_training,
        "predict_auxiliary_direction_cases",
        fake_auxiliary_prediction,
    )
    monkeypatch.setattr(
        three_day_training,
        "auxiliary_direction_metrics_with_instruments",
        fake_auxiliary_metrics,
    )

    output_dir = tmp_path / "direction"
    wrapper = KronosTrendWrapper(ToyPredictor())
    result = train_direction_predictor(
        wrapper,
        ToyTokenizer(),
        ToyDenseDataset(),
        ToyDirectionDataset(),
        [object()],
        device=torch.device("cpu"),
        output_dir=output_dir,
        learning_rate=1e-3,
        seed=42,
        lambda_dir=0.2,
        max_epochs=1,
        early_stopping_patience=1,
        batch_size=2,
        direction_batch_size=2,
        dense_batches_per_direction_batch=1,
        num_workers=0,
        weight_decay=0.0,
        gradient_clip=3.0,
        warmup_ratio=0.0,
        evaluation_config={
            "max_context": 8,
            "clip": 5.0,
            "normalization_epsilon": 1e-5,
            "sample_count": 1,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.9,
            "inference_batch_size": 1,
            "random_seed": 7,
            "path_point_estimate": "median",
            "turning_point_return_threshold": 0.0005,
        },
        checkpoint_metadata={"test": True},
        validation_device=torch.device("cpu"),
    )

    assert result.best_epoch == 1
    assert result.history[0]["processed_direction_batches"] == 1
    assert result.history[0]["train_direction_bce"] > 0
    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["lambda_dir"] == pytest.approx(0.2)
    assert checkpoint["trend_head_state"]
    restored = KronosTrendWrapper(ToyPredictor())
    load_direction_checkpoint(restored, output_dir / "best_model.pt")
    for expected, observed in zip(
        wrapper.trend_head.parameters(), restored.trend_head.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, observed)


class _FakeTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_lengths: list[int] = []

    def encode(
        self, values: torch.Tensor, *, half: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert half
        self.seen_lengths.append(values.shape[1])
        return values[..., 0].round().long(), values[..., 1].round().long()


class _FakePredictor(nn.Module):
    d_model = 4

    def decode_s1(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del padding_mask
        assert stamp is not None
        hidden = torch.stack(
            (
                s1_ids.float(),
                s2_ids.float(),
                stamp[..., 0].float(),
                torch.ones_like(s1_ids, dtype=torch.float32),
            ),
            dim=-1,
        )
        logits = torch.zeros((*s1_ids.shape, 2), dtype=torch.float32)
        return logits, hidden

    def forward(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError


def test_direction_head_is_invariant_to_constructive_target_changes() -> None:
    generator = torch.Generator().manual_seed(9)
    original = torch.randn((2, 277, 6), generator=generator)
    modified = original.clone()
    modified[:, 256:, :] += 100_000.0
    stamps = torch.randn((2, 277, 5), generator=generator)
    tokenizer = _FakeTokenizer()
    wrapper = KronosTrendWrapper(_FakePredictor()).eval()

    original_logits = direction_logits_from_context(
        wrapper, tokenizer, original, stamps, lookback=256
    )
    modified_logits = direction_logits_from_context(
        wrapper, tokenizer, modified, stamps, lookback=256
    )

    torch.testing.assert_close(original_logits, modified_logits)
    assert tokenizer.seen_lengths == [256, 256]


class _RecordingHead:
    def __init__(self) -> None:
        self.seen_lengths: list[int] = []

    def compute_loss(
        self,
        logits_s1: torch.Tensor,
        logits_s2: torch.Tensor,
        target_s1: torch.Tensor,
        target_s2: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del padding_mask
        self.seen_lengths.append(logits_s1.shape[1])
        first = logits_s1.sum() + target_s1.float().sum()
        second = logits_s2.sum() + target_s2.float().sum()
        return (first + second) / 2, first, second


class _LossPredictor:
    def __init__(self) -> None:
        self.head = _RecordingHead()


def test_token_loss_covers_only_the_21_future_tokens() -> None:
    predictor = _LossPredictor()
    logits_s1 = torch.zeros((1, 276, 2))
    logits_s2 = torch.zeros((1, 276, 2))
    target_s1 = torch.zeros((1, 276), dtype=torch.long)
    target_s2 = torch.zeros((1, 276), dtype=torch.long)
    baseline = compute_future_token_loss(
        predictor,
        logits_s1,
        logits_s2,
        target_s1,
        target_s2,
        lookback=256,
    )[0]

    context_changed = logits_s1.clone()
    context_changed[:, :255, :] = 10_000.0
    unchanged = compute_future_token_loss(
        predictor,
        context_changed,
        logits_s2,
        target_s1,
        target_s2,
        lookback=256,
    )[0]
    future_changed = logits_s1.clone()
    future_changed[:, 255:, :] = 1.0
    changed = compute_future_token_loss(
        predictor,
        future_changed,
        logits_s2,
        target_s1,
        target_s2,
        lookback=256,
    )[0]

    assert baseline == unchanged
    assert changed != baseline
    assert predictor.head.seen_lengths == [21, 21, 21]


def test_synthetic_return_path_metrics_are_hand_checkable() -> None:
    actual = [100.0, 200.0]
    predicted = [100.0, 300.0]

    assert return_path_correlation(actual, predicted, 100.0) == pytest.approx(1.0)
    assert return_space_dtw_distance(actual, predicted, 100.0) == pytest.approx(0.5)
    assert z_normalized_dtw_distance(actual, predicted, 100.0) == pytest.approx(0.0)
    assert slope_sign_agreement(actual, predicted, 100.0) == pytest.approx(1.0)
    assert turning_point_similarity(
        [100.0, 110.0, 100.0],
        [100.0, 110.0, 100.0],
        100.0,
        threshold=0.05,
    ) == pytest.approx(1.0)
    assert turning_point_similarity(
        [100.0, 110.0, 100.0],
        [100.0, 90.0, 100.0],
        100.0,
        threshold=0.05,
    ) == pytest.approx(0.0)

    complete = compute_three_day_path_metrics(
        actual_close=[101, 102, 101, 103, 102, 104],
        predicted_close=[101, 102, 101, 103, 102, 104],
        origin_close=100.0,
        day_end_indices=(1, 3, 5),
        turning_point_threshold=0.005,
    )
    assert complete["day3_actual_return"] == pytest.approx(0.04)
    assert complete["day3_path_direction"] == 1
    assert complete["return_space_dtw_distance"] == pytest.approx(0.0)


def test_v2_config_locks_independent_roots_and_phase_parameters() -> None:
    config = load_config("csj/configs/futures_3day_trend.yaml")

    assert config["data"]["lookback"] == 256
    assert config["data"]["possible_pred_lengths"] == [15, 17, 19, 21]
    assert config["training"]["learning_rates"] == [0.000003, 0.00001]
    assert config["training"]["direction_smoke_lambda"] == 0.2
    assert config["evaluation"]["device"] == "auto"
    assert config["evaluation"]["device"] == config["training"]["device"]
    assert config["output"]["root"].endswith("csj/runs/futures_3day_trend")
    assert config["output"]["results_root"].endswith(
        "csj/results/futures_3day_trend"
    )


def test_three_day_prediction_preserves_samples_quantiles_and_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_frame("alpha", [7] * 37 + [5, 7, 5])
    case = build_three_trading_day_cases(
        {"alpha": frame}, _case_period(frame, 37, 39), lookback=256
    )[0]

    def fake_inference(
        tokenizer: nn.Module,
        model: nn.Module,
        x: torch.Tensor,
        x_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
        max_context: int,
        pred_len: int,
        **kwargs: object,
    ) -> np.ndarray:
        del tokenizer, model, x_stamp, y_stamp, max_context
        sample_count = int(kwargs["sample_count"])
        values = np.zeros((x.shape[0], sample_count, pred_len, 6), dtype=np.float64)
        for sample_index in range(sample_count):
            values[:, sample_index, :, :] = sample_index * 0.1
        return values

    monkeypatch.setattr(
        three_day_evaluation,
        "auto_regressive_inference",
        fake_inference,
    )
    records = predict_three_day_cases(
        nn.Identity(),
        nn.Identity(),
        [case],
        device=torch.device("cpu"),
        max_context=512,
        clip=5.0,
        normalization_epsilon=1e-5,
        sample_count=3,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        batch_size=1,
        seed=3,
        point_estimate="median",
        turning_point_threshold=0.0005,
    )

    assert len(records) == 1
    assert len(records.loc[0, "sample_paths"]) == 3
    assert len(records.loc[0, "raw_sample_paths"]) == 3
    assert np.asarray(records.loc[0, "sample_paths"]).shape == (3, 17, 6)
    np.testing.assert_allclose(
        np.asarray(records.loc[0, "predicted_path"]),
        np.asarray(records.loc[0, "median_path"]),
    )
    assert len(records.loc[0, "sample_endpoint_returns"]) == 3
    assert records.loc[0, "inference_device"] == "cpu"
    assert records.loc[0, "day3_path_direction"] in (-1, 0, 1)


def test_three_day_baselines_and_metric_aggregation_share_case_contract() -> None:
    frame = _synthetic_frame("alpha", [7] * 37 + [5, 7, 5, 7])
    cases = build_three_trading_day_cases(
        {"alpha": frame}, _case_period(frame, 37, 40), lookback=256
    )

    baselines = make_three_day_baselines(
        cases[:1],
        cases[1:],
        point_estimate="median",
        turning_point_threshold=0.0005,
    )

    assert set(baselines) == {"majority", "momentum", "persistence"}
    assert all(len(records) == 1 for records in baselines.values())
    assert baselines["persistence"].loc[0, "day3_path_direction"] == 0
    majority_metrics = compute_three_day_record_metrics(baselines["majority"])
    assert majority_metrics["samples"] == 1
    assert set(majority_metrics["endpoints"]) == {"day1", "day2", "day3"}
    assert "mean_return_space_dtw_distance" in majority_metrics["path"]
