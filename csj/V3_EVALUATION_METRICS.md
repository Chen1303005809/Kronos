# V3 固化评估与绘图契约

V3 的模型评估只能通过 `csj.v3.evaluation_plotter` 输出。它是模型阶段的强制步骤：P0 zero-shot、P0 CE-only 和 P1 Pair Probe 在写出阶段指标前都会同步生成图表；绘图、指标契约校验或文件落盘失败会使该阶段失败。

禁止在实验代码中直接调用 `matplotlib` 绘制临时指标图，也不要只同步 checkpoint 或单一总分数。

## 固定指标

| 阶段 | 固定指标 | 优化方向 |
| --- | --- | --- |
| P0 / 后续目标路径模型 | Day3 path balanced accuracy | 高 |
| P0 / 后续目标路径模型 | Day3 return MAE | 低 |
| P0 / 后续目标路径模型 | Mean return-path correlation | 高 |
| P0 / 后续目标路径模型 | Mean z-normalized DTW | 低 |
| P1 Pair Probe | target-only Day3 balanced accuracy | 高 |
| P1 Pair Probe | pair-probe Day3 balanced accuracy | 高 |
| P1 Pair Probe | pair − target-only balanced accuracy | 高 |
| P1 Pair Probe | 5-day block `P(improvement > 0)` | 高 |
| P1 Pair Probe | 10-day block `P(improvement > 0)` | 高 |

指标定义版本固定为 `v3-evaluation-v1`。JSON 中的 `metric_contract` 是机器可读的唯一口径，PNG 只是对同一份聚合结果的可视化。

## 每次模型运行的必备产物

P0：

```text
csj/runs/.../p0_zero_shot/evaluation/evaluation_report.json
csj/runs/.../p0_zero_shot/evaluation/fixed_metrics_overview.png
csj/runs/.../p0_zero_shot/evaluation/fixed_metrics_by_product.png
```

P1：

```text
csj/runs/.../p1/evaluation/evaluation_report.json
csj/runs/.../p1/evaluation/fixed_metrics_overview.png
csj/runs/.../p1/evaluation/fixed_metrics_strata.png
```

`p0_ce_only` 使用与 P0 相同的三份文件。阶段 `metrics.json` 会包含 `evaluation_artifacts`，便于同步回来后自动定位图表。

当前 partial-panel 配置的报告会保留 `result_scope: exploratory_partial_panel`；图表用于探索比较，不能改变 P1 到 P2 的正式门禁。

## 后续模型扩展

新增 V3 模型阶段时，必须复用本模块：目标路径模型调用 `render_p0_evaluation_report`，配对分类模型调用 `render_p1_evaluation_report`。若确实需要新的任务类型，先在此文档和 `evaluation_plotter.py` 中版本化定义指标契约、JSON schema 与图表，再接入训练入口。
