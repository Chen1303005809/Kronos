# V6 主动风控微调

V6 把研究目标从“预测上涨或下跌”改为“估计未来三个完整交易日内，对当前持仓方向不利的尾部风险”。这是一次主目标变更，因此作为独立 V6 管线存在，不覆盖 V5，也不复用 V5 的 gate 结论。

当前状态：**P0 标签、五折审计、gate 与可视化已实施；`v6_p0_20260813` gate 未通过，因此 P1-P5 保持锁死，不能用于生产。**

权威设计文档是 [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)，预注册配置是 [`../configs/risk_control_v6.yaml`](../configs/risk_control_v6.yaml)。若文档与未来代码发生冲突，必须先更新方案版本与配置，再运行新的实验。

V6 最终计划对外只暴露两个深模块接口：

```text
RiskForecaster(context, future_timestamps) -> RiskForecast
RiskOverlay(base_position, RiskForecast)   -> RiskOverlayDecision
```

`RiskForecaster` 只输出经过校准的多头/空头不利风险与不确定性，不产生订单；`RiskOverlay` 只能缩小已有基础仓位，不能开新仓、反向或提高杠杆。止损、保证金、流动性、涨跌停和人工熔断继续由模型之外的硬风控处理。

源码结构：

```text
csj/v6/
  config.py             # 已实现：固定协议校验
  risk_labels.py        # 已实现：风险结果与 fit-only 阈值
  audit.py              # 已实现：五折支持度与防泄漏 P0 gate
  plotting.py           # 已实现：P0 标签分布、支持度与阈值图
  experiment.py         # 已实现 P0；P1-P5 是 gate-guarded 入口
  path_features.py      # 未实施：冻结 Kronos 路径库及风险摘要
  risk_head.py          # 未实施：冻结主干的多任务风险头
  calibration.py        # 未实施：概率校准、alert threshold、abstain
  overlay.py            # 未实施：只减仓的策略 adapter
```

运行 P0：

```bash
.venv/bin/python -m csj.v6.experiment audit --run-id v6_p0_20260813
```

当前证据位于 [`../results/risk_control_v6/`](../results/risk_control_v6/)。P0 不加载 Kronos、不要求 CUDA。P0 gate 未通过是研究结论，不是命令运行失败；`p1-path-bank` 会拒绝继续。

后续方案已递增为 V7。实现 V7 时阅读 [`../v7/IMPLEMENTATION_PLAN.md`](../v7/IMPLEMENTATION_PLAN.md)，不要解锁或续跑 V6 P1-P5，也不要用 V7 的通过结果覆盖本目录的失败证据。
