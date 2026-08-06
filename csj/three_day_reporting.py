from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _direction_text(value: int) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def plot_three_day_return_examples(
    records: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    if records.empty:
        raise ValueError("Cannot plot an empty record table")
    instruments = sorted(records["instrument"].astype(str).unique())
    examples: dict[str, pd.DataFrame] = {}
    max_examples = 0
    for instrument in instruments:
        group = records.loc[records["instrument"] == instrument].sort_values(
            "target_day", kind="stable"
        )
        candidate_indices = sorted({0, len(group) // 2, len(group) - 1})
        examples[instrument] = group.iloc[candidate_indices]
        max_examples = max(max_examples, len(candidate_indices))

    fig, axes = plt.subplots(
        len(instruments),
        max_examples,
        figsize=(5.4 * max_examples, 3.8 * len(instruments)),
        squeeze=False,
        sharey=False,
    )
    for row_index, instrument in enumerate(instruments):
        instrument_examples = examples[instrument]
        for column_index in range(max_examples):
            axis = axes[row_index, column_index]
            if column_index >= len(instrument_examples):
                axis.axis("off")
                continue
            record = instrument_examples.iloc[column_index]
            origin_close = float(record["origin_close"])
            actual = np.asarray(record["actual_path"], dtype=np.float64)[:, 3]
            median = np.asarray(record["median_path"], dtype=np.float64)[:, 3]
            q10 = np.asarray(record["q10_path"], dtype=np.float64)[:, 3]
            q90 = np.asarray(record["q90_path"], dtype=np.float64)[:, 3]
            x_values = np.arange(1, len(actual) + 1)
            actual_returns = actual / origin_close - 1.0
            median_returns = median / origin_close - 1.0
            q10_returns = q10 / origin_close - 1.0
            q90_returns = q90 / origin_close - 1.0

            axis.plot(x_values, actual_returns, color="black", label="actual", lw=2)
            axis.plot(
                x_values,
                median_returns,
                color="#1f77b4",
                label="predicted median",
                lw=2,
            )
            axis.fill_between(
                x_values,
                q10_returns,
                q90_returns,
                color="#1f77b4",
                alpha=0.2,
                label="10–90%",
            )
            for day_end in list(record["day_end_indices"])[:-1]:
                axis.axvline(float(day_end) + 1.5, color="gray", ls="--", lw=1)
            axis.axhline(0.0, color="gray", lw=0.8)
            actual_direction = int(record["day3_actual_direction"])
            predicted_direction = int(record["day3_path_direction"])
            target_day = pd.Timestamp(record["target_day"]).strftime("%Y-%m-%d")
            axis.set_title(
                f"{instrument} {target_day}\n"
                f"day3 actual={_direction_text(actual_direction)}, "
                f"pred={_direction_text(predicted_direction)}"
            )
            axis.set_xlabel("Target hourly bar")
            axis.set_ylabel("Cumulative return from origin close")
            axis.grid(alpha=0.2)
            if row_index == 0 and column_index == 0:
                axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def _percent(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 6) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_phase1_report(
    metrics: Mapping[str, Any],
    *,
    by_fold_metrics: Mapping[str, Any],
    output_path: str | Path,
    elapsed_seconds: float,
    record_counts: Mapping[str, int],
    figure_name: str,
    metrics_name: str,
    smoke: bool = False,
) -> Path:
    models = ["zero_shot", "majority", "momentum", "persistence"]
    lines = [
        "# V2 Phase 1 Smoke：三交易日 Zero-shot 与数据基线"
        if smoke
        else "# V2 Phase 1：三交易日 Zero-shot 与数据基线",
        "",
        "本报告仅验证 4 个 smoke 案例，不能用于判断预测效果。"
        if smoke
        else "本报告覆盖预先锁定的全部完整 walk-forward folds。",
        "",
        "本阶段是已观察历史上的 expanding walk-forward 开发比较，不是新的封存测试。",
        "所有路径方向指标均由完整生成路径的 sampled median path 推导；第三日末方向是主指标。",
        "",
        "## Pooled 三日端点指标",
        "",
        "| Model | Cases | Day1 bal. acc. | Day2 bal. acc. | Day3 bal. acc. | Day3 return MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in models:
        pooled = metrics[model_name]["pooled"]
        endpoints = pooled["endpoints"]
        lines.append(
            "| {model} | {cases} | {day1} | {day2} | {day3} | {mae} |".format(
                model=model_name,
                cases=record_counts[model_name],
                day1=_percent(endpoints["day1"]["path_direction_balanced_accuracy"]),
                day2=_percent(endpoints["day2"]["path_direction_balanced_accuracy"]),
                day3=_percent(endpoints["day3"]["path_direction_balanced_accuracy"]),
                mae=_number(endpoints["day3"]["endpoint_return_mae"]),
            )
        )

    lines.extend(
        [
            "",
            "## Zero-shot 分合约第三日指标",
            "",
            "| Instrument | Cases | Day3 bal. acc. | Day3 accuracy | Return MAE | Return corr. |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for instrument in sorted(
        key for key in metrics["zero_shot"] if key != "pooled"
    ):
        instrument_metrics = metrics["zero_shot"][instrument]
        day3 = instrument_metrics["endpoints"]["day3"]
        lines.append(
            "| {instrument} | {cases} | {balanced} | {accuracy} | {mae} | {correlation} |".format(
                instrument=instrument,
                cases=instrument_metrics["samples"],
                balanced=_percent(day3["path_direction_balanced_accuracy"]),
                accuracy=_percent(day3["path_direction_accuracy"]),
                mae=_number(day3["endpoint_return_mae"]),
                correlation=_number(day3["endpoint_return_correlation"], 4),
            )
        )

    zero_path = metrics["zero_shot"]["pooled"]["path"]
    diagnostics = metrics["zero_shot"]["pooled"]["generation_diagnostics"]
    lines.extend(
        [
            "",
            "## Zero-shot 路径与生成审计",
            "",
            f"- Mean return-path correlation: `{_number(zero_path.get('mean_return_path_correlation'), 4)}`",
            f"- Mean z-normalized DTW: `{_number(zero_path.get('mean_z_normalized_dtw_distance'), 6)}`",
            f"- Mean return-space DTW: `{_number(zero_path.get('mean_return_space_dtw_distance'), 6)}`",
            f"- Mean slope-sign agreement: `{_percent(zero_path.get('mean_slope_sign_agreement'))}`",
            f"- Raw non-finite value rate: `{_percent(diagnostics.get('mean_raw_nonfinite_rate'))}`",
            f"- Raw OHLC violation rate: `{_percent(diagnostics.get('mean_raw_ohlc_violation_rate'))}`",
            f"- Raw negative volume/amount rate: `{_percent(diagnostics.get('mean_raw_negative_flow_rate'))}`",
            "",
            "## Zero-shot 逐 fold 稳定性",
            "",
            "| Fold | Cases | Day3 bal. acc. | Return-path corr. | Z-normalized DTW |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold_id, fold_metrics in sorted(by_fold_metrics.items()):
        pooled = fold_metrics["zero_shot"]["pooled"]
        lines.append(
            "| {fold} | {cases} | {balanced} | {correlation} | {dtw} |".format(
                fold=fold_id,
                cases=pooled["samples"],
                balanced=_percent(
                    pooled["endpoints"]["day3"][
                        "path_direction_balanced_accuracy"
                    ]
                ),
                correlation=_number(
                    pooled["path"]["mean_return_path_correlation"], 4
                ),
                dtw=_number(
                    pooled["path"]["mean_z_normalized_dtw_distance"], 6
                ),
            )
        )
    lines.extend(
        [
            "",
            f"![三交易日 return-space 路径示例]({figure_name})",
            "",
            "## 运行信息",
            "",
            f"- Elapsed seconds: `{elapsed_seconds:.2f}`",
            "- 原始及 OHLC/非负流量修复后的逐 sample 路径均保存在对应 run 目录。",
            "- 报告没有把均值曲线当作唯一输出；主指标固定使用 sampled median path。",
            "- 15/17 根组合在当前清洗后真实历史中未出现；构造性单元测试仍覆盖全部 5/7 三日组合。",
            "",
            f"完整机器可读指标：`{metrics_name}`。",
        ]
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def write_phase3_report(
    path_metrics: Mapping[str, Any],
    *,
    auxiliary_metrics: Mapping[str, Any],
    by_fold_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    output_path: str | Path,
    elapsed_seconds: float,
    record_counts: Mapping[str, int],
    figure_name: str,
    metrics_name: str,
    pilot: bool = False,
) -> Path:
    """Write the Phase 3 ablation report without conflating head and path metrics."""

    lines = [
        "# V2 Phase 3 Pilot：方向辅助损失消融"
        if pilot
        else "# V2 Phase 3：方向辅助损失消融",
        "",
        "本阶段固定 `lambda_dir=0.2`、seed 42；checkpoint 仍由完整生成路径的第三日指标选择。",
        "方向头指标与生成路径指标分开报告，方向头本身变好不等于 K 线路径方向变好。",
        "",
        "## Pooled 生成路径对照",
        "",
        "| Model | Cases | Day1 bal. acc. | Day2 bal. acc. | Day3 bal. acc. | Day3 return MAE | Return-path corr. | Z-DTW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in ("ce_direction", "phase2_ce_only", "zero_shot"):
        pooled = path_metrics[model_name]["pooled"]
        endpoints = pooled["endpoints"]
        path = pooled["path"]
        lines.append(
            "| {model} | {cases} | {day1} | {day2} | {day3} | {mae} | {corr} | {dtw} |".format(
                model=model_name,
                cases=record_counts[model_name],
                day1=_percent(endpoints["day1"]["path_direction_balanced_accuracy"]),
                day2=_percent(endpoints["day2"]["path_direction_balanced_accuracy"]),
                day3=_percent(endpoints["day3"]["path_direction_balanced_accuracy"]),
                mae=_number(endpoints["day3"]["endpoint_return_mae"]),
                corr=_number(path.get("mean_return_path_correlation"), 4),
                dtw=_number(path.get("mean_z_normalized_dtw_distance"), 6),
            )
        )

    lines.extend(
        [
            "",
            "## 辅助方向头（非主结果）",
            "",
            "| Scope | Cases | Day1 aux bal. acc. | Day2 aux bal. acc. | Day3 aux bal. acc. | Day3 BCE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scope in ["pooled", *sorted(key for key in auxiliary_metrics if key != "pooled")]:
        metrics = auxiliary_metrics[scope]
        endpoints = metrics["endpoints"]
        lines.append(
            "| {scope} | {cases} | {day1} | {day2} | {day3} | {bce} |".format(
                scope=scope,
                cases=metrics["samples"],
                day1=_percent(
                    endpoints["day1"]["aux_direction_balanced_accuracy"]
                ),
                day2=_percent(
                    endpoints["day2"]["aux_direction_balanced_accuracy"]
                ),
                day3=_percent(
                    endpoints["day3"]["aux_direction_balanced_accuracy"]
                ),
                bce=_number(endpoints["day3"]["mean_bce"]),
            )
        )

    lines.extend(
        [
            "",
            "## 分合约第三日生成路径",
            "",
            "| Instrument | CE + direction | Phase 2 CE-only | Zero-shot |",
            "|---|---:|---:|---:|",
        ]
    )
    instruments = sorted(
        key for key in path_metrics["ce_direction"] if key != "pooled"
    )
    for instrument in instruments:
        lines.append(
            "| {instrument} | {phase3} | {phase2} | {zero} |".format(
                instrument=instrument,
                phase3=_percent(
                    path_metrics["ce_direction"][instrument]["endpoints"][
                        "day3"
                    ]["path_direction_balanced_accuracy"]
                ),
                phase2=_percent(
                    path_metrics["phase2_ce_only"][instrument]["endpoints"][
                        "day3"
                    ]["path_direction_balanced_accuracy"]
                ),
                zero=_percent(
                    path_metrics["zero_shot"][instrument]["endpoints"]["day3"][
                        "path_direction_balanced_accuracy"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## 配对 moving-block bootstrap（day3 generated-path direction）",
            "",
            "| Comparison | Block days | Point estimate | 95% CI | P(improvement > 0) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison, blocks in bootstrap.items():
        for block_name, result in blocks.items():
            lines.append(
                "| {comparison} | {days} | {point} | [{lower}, {upper}] | {positive} |".format(
                    comparison=comparison,
                    days=result["block_days"],
                    point=_percent(result["point_estimate"]),
                    lower=_percent(result["ci_lower_95"]),
                    upper=_percent(result["ci_upper_95"]),
                    positive=_percent(result["probability_improvement_positive"]),
                )
            )

    lines.extend(
        [
            "",
            "## 逐 fold 路径方向变化",
            "",
            "| Fold | CE + direction | Phase 2 CE-only | Zero-shot | Δ vs Phase 2 | Δ vs zero-shot |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_id, fold_metrics in sorted(by_fold_metrics.items()):
        phase3_value = fold_metrics["ce_direction"]["pooled"]["endpoints"][
            "day3"
        ]["path_direction_balanced_accuracy"]
        phase2_value = fold_metrics["phase2_ce_only"]["pooled"]["endpoints"][
            "day3"
        ]["path_direction_balanced_accuracy"]
        zero_value = fold_metrics["zero_shot"]["pooled"]["endpoints"]["day3"][
            "path_direction_balanced_accuracy"
        ]
        lines.append(
            "| {fold} | {phase3} | {phase2} | {zero} | {delta_phase2} | {delta_zero} |".format(
                fold=fold_id,
                phase3=_percent(phase3_value),
                phase2=_percent(phase2_value),
                zero=_percent(zero_value),
                delta_phase2=_percent(phase3_value - phase2_value),
                delta_zero=_percent(phase3_value - zero_value),
            )
        )

    lines.extend(
        [
            "",
            f"![Phase 3 return-space 路径示例]({figure_name})",
            "",
            "## 运行信息",
            "",
            f"- Elapsed seconds: `{elapsed_seconds:.2f}`",
            "- 只有生成路径的第三日方向、路径相关性和 DTW 同时改善，才可考虑后续 lambda 或多 seed。",
            "- 结果仍是已观察历史上的 walk-forward 开发证据；不得作为独立前瞻或可交易结论。",
            "",
            f"完整机器可读指标：`{metrics_name}`。",
        ]
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
