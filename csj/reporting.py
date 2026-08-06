from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csj.metrics import compute_metrics, metrics_with_instruments


def plot_metric_comparison(
    record_sets: Mapping[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    labels = list(record_sets)
    instruments = sorted(
        {
            str(instrument)
            for records in record_sets.values()
            for instrument in records["instrument"].unique()
        }
    )
    groups = ["pooled", *instruments]
    values = np.zeros((len(labels), len(groups)), dtype=np.float64)
    for row, label in enumerate(labels):
        metrics = metrics_with_instruments(record_sets[label])
        for column, group in enumerate(groups):
            values[row, column] = float(
                metrics[group]["direction_balanced_accuracy"]
            )

    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, label in enumerate(labels):
        ax.bar(
            x - 0.4 + width / 2 + index * width,
            values[index],
            width,
            label=label,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x, groups)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced direction accuracy")
    ax.set_title("Sealed-test direction comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_histories(
    histories: Mapping[str, Sequence[dict[str, Any]]],
    output_path: str | Path,
) -> None:
    fig, (loss_ax, score_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for label, history in histories.items():
        epochs = [int(item["epoch"]) for item in history]
        losses = [float(item["train_loss"]) for item in history]
        scores = [
            float(item["validation"]["direction_balanced_accuracy"])
            for item in history
        ]
        loss_ax.plot(epochs, losses, marker="o", markersize=3, label=label)
        score_ax.plot(epochs, scores, marker="o", markersize=3, label=label)
    loss_ax.set_ylabel("Target token CE")
    loss_ax.set_title("Training loss")
    loss_ax.grid(alpha=0.25)
    score_ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    score_ax.set_xlabel("Epoch")
    score_ax.set_ylabel("Validation balanced accuracy")
    score_ax.set_title("Daily validation direction")
    score_ax.grid(alpha=0.25)
    score_ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_examples(records: pd.DataFrame, output_path: str | Path) -> None:
    selections: list[pd.Series] = []
    for _, instrument_records in records.groupby("instrument", sort=True):
        ordered = instrument_records.sort_values("target_day").reset_index(drop=True)
        indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
        selections.extend(ordered.iloc[index] for index in indices)

    fig, axes = plt.subplots(
        len(selections),
        1,
        figsize=(9, max(3 * len(selections), 4)),
        squeeze=False,
    )
    for axis, row in zip(axes[:, 0], selections, strict=True):
        actual = np.asarray(row["actual_close_path"], dtype=np.float64)
        predicted = np.asarray(row["predicted_close_path"], dtype=np.float64)
        steps = np.arange(1, len(actual) + 1)
        axis.plot(steps, actual, marker="o", label="actual close")
        axis.plot(steps, predicted, marker="o", label="predicted close")
        axis.set_title(f"{row['instrument']} — {pd.Timestamp(row['target_day']).date()}")
        axis.set_xlabel("Hourly bar in target trading day")
        axis.set_ylabel("Price")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _metric_table(metrics: Mapping[str, Mapping[str, float | int]]) -> str:
    lines = [
        "| Model | Samples | Balanced accuracy | Accuracy | Return MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics.items():
        lines.append(
            "| {label} | {samples} | {balanced} | {accuracy} | {mae:.6f} |".format(
                label=label,
                samples=int(values["samples"]),
                balanced=_percentage(float(values["direction_balanced_accuracy"])),
                accuracy=_percentage(float(values["direction_accuracy"])),
                mae=float(values["return_mae"]),
            )
        )
    return "\n".join(lines)


def write_report(
    output_path: str | Path,
    *,
    audits: Sequence[dict[str, object]],
    split_summary: dict[str, object],
    selected_learning_rate: float,
    training_summaries: Sequence[dict[str, Any]],
    validation_records: Mapping[str, pd.DataFrame],
    test_records: Mapping[str, pd.DataFrame],
    strongest_baseline: str,
    bootstrap: dict[str, float | int],
    sensitivity_metrics: dict[str, dict[str, float | int]],
    normalization_consistency: Mapping[str, Any],
) -> None:
    output = Path(output_path)
    validation_metrics = {
        label: compute_metrics(records) for label, records in validation_records.items()
    }
    test_metrics = {
        label: compute_metrics(records) for label, records in test_records.items()
    }
    per_instrument = metrics_with_instruments(test_records["fine_tuned_ensemble"])
    test_ensemble_metrics = test_metrics["fine_tuned_ensemble"]
    test_zero_shot_metrics = test_metrics["zero_shot"]
    mae_improvement = 1.0 - (
        float(test_ensemble_metrics["return_mae"])
        / float(test_zero_shot_metrics["return_mae"])
    )
    direction_improvement = (
        float(test_ensemble_metrics["direction_balanced_accuracy"])
        - float(test_zero_shot_metrics["direction_balanced_accuracy"])
    )

    lower = float(bootstrap["ci_lower_95"])
    point = float(bootstrap["point_estimate"])
    if lower > 0:
        verdict = "可靠提升：相对验证阶段选定的最强基线，95% 分块 bootstrap 区间下界高于 0。"
    elif point > 0:
        verdict = "有希望但证据不足：点估计优于基线，但 95% 区间仍跨过 0。"
    else:
        verdict = "未证明有效：封存测试集上的点估计没有优于最强基线。"

    audit_lines = "\n".join(
        f"- {audit['instrument']}: {audit['raw_bars']} → {audit['clean_bars']} bars; "
        f"removed days: {json.dumps(audit['removed_days'], ensure_ascii=False)}"
        for audit in audits
    )
    training_lines = "\n".join(
        f"- seed={item['seed']}, lr={item['learning_rate']:.1e}, "
        f"best epoch={item['best_epoch']}, validation balanced accuracy="
        f"{_percentage(float(item['best_score']))}"
        for item in training_summaries
    )
    instrument_lines = "\n".join(
        f"- {instrument}: balanced accuracy "
        f"{_percentage(float(values['direction_balanced_accuracy']))}, "
        f"return MAE {float(values['return_mae']):.6f}"
        for instrument, values in per_instrument.items()
        if instrument != "pooled"
    )

    report = f"""# Kronos 小时期货微调实验报告

## 结论

{verdict}

本报告衡量的是连续合约历史序列上的预测 edge，不是可执行合约的真实 PnL 回测。

## 数据审计

{audit_lines}

时间切分：`{split_summary['boundaries']['first_day']}` 至 `{split_summary['boundaries']['last_day']}`；训练截止 `{split_summary['boundaries']['train_end']}`，验证截止 `{split_summary['boundaries']['val_end']}`。日边界评估案例数：`{json.dumps(split_summary['forecast_cases'], ensure_ascii=False)}`。

## 短期预测与测试口径

这里的测试集 `218` 条不是一次预测未来 218 日，而是 `109` 个历史交易日 × `2` 个合约的 walk-forward 短期预测。每个案例只输入当时以前的 `256` 根小时 K，输出紧接着一个完整交易日的 `5` 或 `7` 根小时 K；随后向前滚动到下一个交易日重复测试。因此预测 horizon 始终是下一交易日。

## 标准化与尺度一致性

每个合约、每个预测窗口都独立按过去 256 根的六个特征统计量执行 `(x - mean) / (std + 1e-5)`，目标区间使用同一坐标系；统计量不跨合约共享，也不使用未来目标数据。Kronos 输出仍在标准化空间，随后只用该窗口原统计量反归一化。

与原生 `KronosPredictor.predict()` 对照时，六字段反归一化最大相对误差为 `{float(normalization_consistency['raw_all_features_max_rel_diff']):.3e}`，close 最大相对误差为 `{float(normalization_consistency['max_rel_close_diff']):.3e}`，标准化往返最大绝对误差为 `{float(normalization_consistency['normalization_roundtrip_max_abs']):.3e}`。因此没有把标准化输出当成原始价格，也没有套用其他合约的统计量。

## 训练选择

验证集选出的学习率：`{selected_learning_rate:.1e}`。

{training_lines}

单 seed 的验证方向看似最高达到 53.21%，但三 seed 等权集成只有 {_percentage(float(validation_metrics['fine_tuned_ensemble']['direction_balanced_accuracy']))}；这表明逐案例预测存在 seed 分歧，不能用最佳单 seed 代表稳定表现。

## 验证集

{_metric_table(validation_metrics)}

## 封存测试集

{_metric_table(test_metrics)}

相对 zero-shot，微调集成的测试 balanced accuracy 变化为 `{_percentage(direction_improvement)}`，return MAE 改善 `{_percentage(mae_improvement)}`。方向没有改善，且收益相关性更差，因此 MAE 的下降不能解释为已获得稳定交易 edge。

微调集成模型分合约结果：

{instrument_lines}

## 配对分块 Bootstrap

验证阶段选定的最强基线：`{strongest_baseline}`。

- Balanced accuracy 提升点估计：{_percentage(point)}
- 95% 区间：[{_percentage(float(bootstrap['ci_lower_95']))}, {_percentage(float(bootstrap['ci_upper_95']))}]
- Bootstrap 提升为正的比例：{_percentage(float(bootstrap['probability_improvement_positive']))}
- 配对预测：{bootstrap['samples']} 条，{bootstrap['unique_days']} 个交易日

## 大跳空敏感性

排除目标交易日开盘跳空绝对值不小于 3% 的样本后：

{_metric_table(sensitivity_metrics)}

## 图表

![测试集指标](test_metrics.png)

![训练曲线](training_curves.png)

![预测路径示例](forecast_examples.png)
"""
    output.write_text(report, encoding="utf-8")
