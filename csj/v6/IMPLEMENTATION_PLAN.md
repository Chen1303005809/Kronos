# V6 目标单流主动风控微调实施方案

> 状态：P0 已实施并于 2026-08-13 运行；gate 未通过，P1-P5 保持锁死
> 日期：2026-08-13
> 方案版本：V6
> 当前 phase：P0 failed
> 结果范围：`retrospective_observed_contracts`
> 生产资格：`production_eligible: false`

## 1. 决策摘要

V6 不再把“预测第三日涨跌”作为最终目标。它回答的问题是：

> 仅使用目标具体合约在预测原点前的历史，以及冻结 Kronos 生成的未来路径，能否提前识别未来三个完整交易日内的显著不利波动，并在不改变基础策略方向的情况下减少尾部损失？

V6 是新的研究目标，不是用另一组指标挽救 V5。V5 的数据、模型选择和 evaluation 已经被研究者观察过，因此 V6 的历史结果仍只算回看证据；最终有效性必须来自冻结后的新增数据。

固定阶段为：

```text
P0 风险标签、样本支持度与防泄漏审计
  ↓ gate
P1 冻结 Kronos 的路径库与简单风险基线
  ↓ gate
P2 冻结主干的多任务风险头微调
  ↓ gate
P3 概率校准、告警预算与 abstain 冻结
  ↓ gate
P4 只减仓风险 overlay 回放
  ↓ gate
P5 多 seed 冻结与新增数据 shadow evaluation
```

任一 gate 失败即停止后续阶段。V6 首轮不解冻 tokenizer 或 Kronos 主干，不通过扩大模型、搜索标签阈值或增加特征来挽救失败结果。

### 1.1 当前 P0 决策

`v6_p0_20260813` 已完成 3071 个 case、5 个 outer fold 的风险标签、支持度与防泄漏审计。数据完整性失败为 0，但 P0 gate 未通过：

- fold_00/01 的早期 fit 尾部事件不足；fold_00 evaluation 的多头事件只有 5 个，fold_01 inner validation 的多头事件只有 19 个。
- fold_04 的 `2026-05-13` 预测原点跨 fit/inner-validation：多数合约的三日标签在 `2026-05-18` 完成，而稀疏的 `j2610` 在 `2026-05-26` 才完成。沿用 V5 的 target-completion-day 切分因此不满足“同一预测日全部案例进入同一 split”。

按预注册停止条件，不实施 P1 路径库，不降低正例门槛，也不静默修改 split。证据保存在 `csj/results/risk_control_v6/`。

后续处置：V6 保持冻结失败状态。2026-08-13 新增跨品种未交割合约数据后，结构性变更（多品种产品池、预测原点日切分）已递增为独立 V7；不得用 V7 的通过结果覆盖或改写 V6 gate。

## 2. 研究边界

### 2.1 输入与预测范围

- 一个具体交割月目标合约自己的最近 `256` 根小时 OHLCVA。
- 不输入邻居合约、连续合约、主力拼接、OI、新闻、宏观或跨品种行情。
- 风险 horizon 固定为紧随预测原点的三个完整交易日，目标长度仍为 `15/17/19/21` 根。
- 主要训练与门禁品种固定为 `i/jm/rb`。
- `j` 只作为未见品种迁移报告，不参与训练、校准、阈值选择或 gate。

### 2.2 V6 输出

风险预测模块只返回：

```text
RiskForecast:
  case_key
  p_long_adverse       # 当前为多头时发生尾部不利波动的校准概率
  p_short_adverse      # 当前为空头时发生尾部不利波动的校准概率
  expected_long_mae    # 标准化的多头最大不利波动严重度
  expected_short_mae   # 标准化的空头最大不利波动严重度
  future_vol_ratio     # 未来实现波动相对当前波动的预测
  abstain
  abstain_reason
  model_version
  calibration_version
```

它不返回 `BUY/SELL`，也不接收当前策略收益、未来价格或真实标签。

### 2.3 模块 seam

V6 只建立两个外部 seam：

1. `RiskForecaster(context, future_timestamps) -> RiskForecast`：隐藏归一化、路径采样、风险特征、模型集成和概率校准。
2. `RiskOverlay(base_position, forecast) -> RiskOverlayDecision`：把风险概率映射为仓位乘数。

`RiskOverlay` 必须满足：

```text
sign(target_position) == sign(base_position)，除非 base_position == 0
abs(target_position) <= abs(base_position)
base_position == 0 时 target_position == 0
```

也就是说，模型只能减少已有风险，不能开仓、反向或增加杠杆。硬止损、保证金、流动性和人工熔断不属于这两个模块。

## 3. 风险标签

### 3.1 只用过去确定尺度

令预测原点收盘价为 `C0`，目标路径第 `u` 根的最低价和最高价分别为 `L_u`、`H_u`。从 context 最后 `60` 根 close-to-close 对数收益计算 EWMA 波动率：

```text
sigma_t = max(EWMA_STD(context_log_returns, halflife=20), 1e-5)
scale_H = sigma_t * sqrt(target_bar_count)
```

`sigma_t` 只使用预测原点前的数据。未来目标长度虽然已知，但未来 OHLCVA 不参与尺度计算。

统计实现固定为 `adjust=false`、`bias=false`，避免 Pandas EWMA 语义漂移。

### 3.2 多头与空头最大不利波动

```text
long_mae  = max(0, -min_u(log(L_u / C0))) / scale_H
short_mae = max(0,  max_u(log(H_u / C0))) / scale_H
```

这里的 `MAE` 是 Maximum Adverse Excursion，不是平均绝对误差。

每个 outer fold 分别只用该 fold 的 primary fit cases 固定：

```text
long_tail_threshold  = fit-period long_mae  的 80% 分位数
short_tail_threshold = fit-period short_mae 的 80% 分位数
```

分位数算法固定为 `linear`，事件比较固定为 `>=`。

随后得到：

```text
long_tail_event  = long_mae  >= long_tail_threshold
short_tail_event = short_mae >= short_tail_threshold
```

阈值不能读取 inner validation 或 evaluation。最终冻结模型使用最终训练窗口产生的一组固定阈值。

### 3.3 未来波动辅助标签

目标 close 收益包含从 `C0` 到第一根目标 close 的变化：

```text
future_realized_scale = sqrt(sum(target_log_returns ** 2))
future_vol_ratio = log((future_realized_scale + 1e-5) / (scale_H + 1e-5))
```

该标签只用于辅助训练和诊断，不直接定义减仓动作。

## 4. 数据与切分

- 复用 V5 的 immutable observed-contract snapshot loader 和 target-only case universe。
- V6 自己记录 `risk_label_rule_version`、case keys、snapshot SHA、模型 revision 和 data fingerprint。
- 保持五折 expanding walk-forward：`60` 日 minimum fit、`20` 日 inner validation、`20` 日 evaluation、`20` 日 step、`3` 日 purge。
- 同一预测日的全部具体合约和品种必须进入同一 split。
- 三日标签必须完整落在一个 split 内。
- 所有模型、基线、校准器和仓位阈值只能读取 fit 与 inner validation；evaluation 每个正式候选只运行一次。
- bootstrap 以交易日块为单位，不把同日多个合约当作独立样本。

当前快照仍存在 survivor/cohort selection 限制，因此所有历史结果固定为 `retrospective_observed_contracts`，不能表述为生产活跃合约或可交易证据。

数据来源硬约束：当前 K 线服务不能查询已交割合约，因此不得通过枚举旧交割月补齐 V6 历史样本。只有两类数据可进入后续研究：合约仍活跃时已经形成、带 manifest 与哈希的不可变快照；以及从当时确认仍活跃的合约向前采集的新快照。数据不足只能通过未来持续积累解决，不能把查询失败的已交割合约当作缺失数据重试。

## 5. P0：标签与支持度审计

P0 不加载模型，先生成：

- 每个 fold/split/product/contract 的案例数。
- `long_mae`、`short_mae`、`future_vol_ratio` 分布和异常值。
- 每个 fold 的 fit-only 尾部阈值与正例率。
- context clipping 比例、缺根、NaN、OHLC 关系异常和被拒案例。
- 修改预测原点后数据是否会改变 context scale 的泄漏测试。

P0 gate：

- 每个 fold 的 fit 集合在多头和空头两侧都至少有 `100` 个尾部正例。
- 每个 fold 的 inner validation 和 evaluation 在两侧都至少有 `20` 个尾部正例。
- 五折 evaluation 汇总后，`i/jm/rb` 每个品种在两侧都至少有 `20` 个尾部正例。
- 任一阈值、归一化统计量或输入特征读取 evaluation 标签时立即失败。

支持度不足时停止，不降低分位数、不延长 evaluation，也不合并未来数据补样本。

## 6. P1：冻结路径库与基线

### 6.1 Kronos 路径库

冻结 V5 相同 tokenizer/predictor revision，对 fit、inner validation 和 evaluation 的案例各生成一次可复用路径库：

```text
sample_count = 64
temperature  = 1.0
top_k        = 0
top_p        = 0.9
```

每个 case×sample 使用由 case key 派生的确定性 seed。必须保存原始 64 条 OHLCVA 路径，而不是只保存均值。

从每个路径库提取：

- 多头/空头尾部阈值穿越比例。
- 多头/空头不利波动的 median、P80、P95。
- 未来波动率的 median、P90。
- 路径间 IQR、方向熵和无效路径比例。

P1 路径质量 gate：

- primary cases 的有限、满足基本 OHLC 约束的路径比例至少 `99%`。
- 每个 case 至少保留 `60/64` 条有效路径，否则该 case 必须 `abstain`，不能偷偷补采样。
- 多头和空头原始尾部概率在 evaluation universe 中均具有非零方差，不能退化为全 0 或全 1。

### 6.2 预声明基线

在 inner validation 预先选择一个最强风险基线：

1. `fit_global_event_rate`
2. `fit_product_event_rate`
3. `ewma_volatility_rank`
4. `atr20_rank`
5. `fixed_context_logistic`：只使用固定的过去风险特征
6. `zero_shot_path_risk`：直接使用 64 条路径的阈值穿越比例

固定 context 特征仅包括过去可计算的 EWMA 波动、ATR、近期振幅、1/3 日收益绝对值、volume/amount z-score 和 clipping fraction。不得在看 evaluation 后增删特征。

## 7. P2：冻结主干的多任务风险头

### 7.1 结构

Tokenizer 和 Kronos 主干全部冻结。模型不使用 product ID，以保留跨品种迁移可能性。

```text
Kronos 最后一个 context hidden
  -> LayerNorm -> Linear(d_model, 128) -> SiLU

固定 context 风险特征 + P1 path summaries
  -> LayerNorm -> Linear(feature_dim, 64) -> SiLU

concat
  -> Linear(192, 128) -> SiLU -> Linear(128, 5)
```

五个输出依次为：

```text
long_tail_logit
short_tail_logit
log1p_expected_long_mae
log1p_expected_short_mae
future_vol_ratio
```

首轮 `dropout=0`，隐藏维度固定，不搜索网络宽度。

同时训练一个 `context_feature_control`：它保留同一批固定 context 风险特征、五个输出、loss、采样、seed 和训练协议，但把 Kronos hidden 与 path summaries 全部置零。候选必须同时战胜这个 matched control，才能把增量归因于 Kronos 表示或生成路径，而不是普通历史波动特征。

### 7.2 损失与训练

```text
loss = BCE(long_tail)
     + BCE(short_tail)
     + 0.25 * SmoothL1(log1p(long_mae))
     + 0.25 * SmoothL1(log1p(short_mae))
     + 0.25 * SmoothL1(future_vol_ratio)
```

- 不使用 class weight 或 focal loss；80% fit 分位标签已经避免极端稀有正例。
- 使用 `prediction_day_product_uniform` 采样，避免同日多合约和多案例品种获得过大权重。
- AdamW，LR `3e-4`，weight decay `0.01`，batch size `64`，最多 `30` epochs，patience `5`，gradient clip `3.0`。
- checkpoint 只按 inner-validation 两侧未校准 macro Brier 最小值选择。
- seeds 固定为 `42/43/44`，保留单 seed 和等权概率 ensemble。

### 7.3 P2 gate

相对每 fold 在 inner validation 预选的最强风险基线，evaluation 必须同时满足：

- 两侧平均相对 Brier 改善至少 `2%`。
- 两侧平均 PR-AUC 至少提高 `0.02`。
- 在固定 `20%` alert budget 下，两侧平均尾部事件 recall 至少提高 `5pp`。
- 至少 `3/5` fold 的 macro Brier 改善。
- `i/jm/rb` 任一品种 macro Brier 的绝对恶化不超过 `0.02`。
- 5 日和 10 日配对 moving-block bootstrap 的 `P(Brier 改善 > 0)` 均至少 `80%`。
- long/short severity 的 pooled Spearman 相关均为正，且至少一侧达到 `0.10`。
- 三个 seed 至少 `2/3` 相对基线改善，seed 间 macro Brier 标准差不超过 `0.02`。

相对同训练协议的 `context_feature_control` 还必须满足：

- 两侧平均相对 Brier 改善至少 `1%`。
- 两侧平均 PR-AUC 不低于 control，且至少一侧提高 `0.01`。
- 至少 `3/5` fold 的 macro Brier 改善。

P2 失败即停止。不得通过解冻主干、修改 80% 标签分位数、增加 seed 或搜索 loss 权重挽救。

## 8. P3：校准、告警预算与 abstain

- 对三 seed ensemble 的 long/short logits 分别做共享品种的 affine Platt calibration。
- 校准器只用 inner validation，形式和优化目标预先固定为 Bernoulli NLL。
- 不训练 product-specific calibrator；`j` 继续只作迁移报告。
- soft alert 和 hard alert 分别固定为 inner-validation 校准概率的 P80 与 P95；evaluation 不重新计算。
- 输入缺失、context clipping fraction 超限、有效路径少于 `60` 或不支持的品种必须返回 `abstain`。

P3 gate：

- 校准后的 macro Brier 不差于未校准 ensemble，也不差于预选基线。
- pooled long/short ECE 均不高于 `0.05`；各主要品种均不高于 `0.10`。
- primary evaluation 覆盖率至少 `95%`；所有 abstain 原因完整记录。
- 校准曲线、PR 曲线、预测概率与真实不利波动严重度图必须生成。

## 9. P4：主动风控 overlay 回放

### 9.1 固定仓位映射

根据基础仓位方向选择风险概率：

```text
base_position > 0  -> p_adverse = p_long_adverse
base_position < 0  -> p_adverse = p_short_adverse
base_position == 0 -> target_position = 0
```

仓位乘数固定为：

```text
p <= soft_threshold:
    multiplier = 1.00
soft_threshold < p < hard_threshold:
    multiplier 从 1.00 线性降到 0.50
p >= hard_threshold:
    multiplier = 0.25
abstain:
    multiplier = 1.00，并把决定交给模型外硬风控
```

V6 每个预测日只使用最新一份风险预测，不叠加未来三日重叠预测。交易只能在信号产生后的下一根可成交 bar 执行。

### 9.2 正式回放前提

正式 P4 必须有逐时点的外部 `base_position`、可交易具体合约映射以及下列参数：

- 合约乘数、最小变动价位、保证金和仓位上限。
- 到期、移仓换月和不可交易处理。
- 手续费、买卖价差、滑点、涨跌停和成交容量。

如果这些数据缺失，只能运行 `unit_long/unit_short` 风险诊断，不能生成 P4 gate 或声称 PnL 改善。

### 9.3 P4 gate

相对完全相同的基础仓位，扣除新增换手成本后必须同时满足：

- 5% downside CVaR 至少改善 `5%`。
- 最大回撤至少改善 `5%`。
- 净收益保留率至少 `85%`。
- 至少 `3/5` fold 的 downside CVaR 改善。
- 改善不能只来自单一品种，任一主要品种 downside CVaR 不得恶化超过 `2%`。
- 5 日和 10 日交易日块 bootstrap 的 `P(CVaR 改善 > 0)` 均至少 `80%`。
- 资金曲线、回撤曲线、风险概率与真实不利波动关系图必须生成。

此外必须与一个 `exposure_matched_random_overlay` 配对比较：它在每个 fold、品种和基础仓位方向内保持与 V6 相同的平均仓位乘数和调仓次数，但随机打乱减仓日期。V6 的 downside CVaR 必须优于该 control 至少 `2%`，且 5/10 日块 bootstrap 的 `P(改善 > 0)` 均至少 `80%`。这用于排除“只因平均仓位下降而自然减少回撤”的解释。

这里验证的是风险 overlay 的增量，不是 Kronos 单独产生交易 alpha。

## 10. P5：冻结与新增数据

P4 通过后冻结：

- tokenizer、predictor、三 seed 风险头与校准器。
- 风险标签规则、fit 阈值、soft/hard alert 阈值。
- path sampling 参数、abstain 规则和仓位映射。
- 数据规则、可交易合约规则、费用与滑点模型。

冻结后：

- 前 `20` 个新增交易日只做 shadow checkpoint，不调参。
- 至少 `60` 个新增交易日后形成第一段正式前瞻窗口。
- 标签结果至少滞后三个完整交易日后才能更新监控指标。
- 每 `20` 日报告 Brier、PR-AUC、ECE、alert rate、tail recall、CVaR、回撤和收益保留率。
- 连续两个 20 日窗口出现 `Brier skill <= 0`、`ECE > 0.10` 或数据漂移告警时，停用模型 overlay 并启动新版本评审；不得原地静默重训。

历史 gate 全部通过也不自动把 `production_eligible` 改为 true。

## 11. 测试与可视化合同

必须测试：

- 修改预测原点后的任意 OHLCVA 不改变 context、路径输入或风险预测。
- context scale 只读取历史；tail threshold 只读取 outer fit 标签。
- 同一预测日全部案例进入同一 fold，purge 不少于三交易日。
- 路径库在相同 case key/revision/seed 下确定性一致。
- 风险头训练时 tokenizer/predictor 参数没有梯度和权重变化。
- calibration、alert threshold 和 baseline selection 不读取 evaluation。
- `RiskOverlay` 永远不增加绝对仓位、不反向、不从零开仓。
- abstain case 不进入“模型成功覆盖”的分母，并完整报告原因。

每个正式 fold 至少生成：

- long/short PR 曲线。
- long/short reliability diagram。
- 预测概率分箱与实际 MAE/事件率图。
- 预测严重度与真实严重度散点/分位图。
- P4 的基础/overlay 资金曲线和回撤图。

任一必需图表缺失使对应阶段失败。

## 12. 计划入口与产物

计划入口：

```bash
.venv/bin/python -m csj.v6.experiment audit
.venv/bin/python -m csj.v6.experiment p1-path-bank
.venv/bin/python -m csj.v6.experiment p2-risk-head
.venv/bin/python -m csj.v6.experiment p3-calibrate
.venv/bin/python -m csj.v6.experiment p4-overlay --base-positions <path>
.venv/bin/python -m csj.v6.experiment p5-freeze
```

计划产物：

```text
csj/runs/risk_control_v6/<run_id>/
  resolved_config.json
  data_audit.json
  p1_path_bank/
  p2_risk_head/
  p3_calibration/
  p4_overlay/
  p5_freeze/

csj/results/risk_control_v6/
  v6_data_audit.json
  p1_path_baselines.json
  p2_risk_head_gate.json
  p3_calibration_gate.json
  p4_overlay_gate.json
  REPORT.md
```

每个结果必须带有 `strategy_version`、`phase`、`result_scope`、`production_eligible`、`data_fingerprint`、风险标签版本、模型/tokenizer revision 以及上游 gate 的路径和 SHA-256。

## 13. 明确停止条件

- 简单 EWMA/ATR/logistic 基线已经同样好时，不把复杂度当作进展。
- P2 风险预测没有增量时，不实施校准和 overlay 来包装结果。
- P3 概率无法校准时，不把 raw score 当作风险概率。
- 没有真实基础仓位和执行约束时，不生成正式 P4 结论。
- 不因回测收益难看而调整风险标签、alert budget 或仓位乘数。
- 不把降低仓位天然带来的回撤下降归功于模型；必须同时报告收益保留率，并与相同平均风险暴露的机械减仓基线比较。
- 不把回看 observed contracts 的结果写成生产结论。

## 14. 建议提交边界

1. `docs(v6): register target-only active-risk-control strategy`
2. `feat(v6): add risk labels and leakage audit`
3. `feat(v6): add frozen kronos path-risk bank and baselines`
4. `feat(v6): add frozen-backbone multitask risk head`
5. 同步 CUDA P2 结果并审阅 gate；未通过即停止。
6. `feat(v6): add probability calibration and abstention`
7. `feat(v6): add position-reducing risk overlay replay`
8. P4 通过后再实现多 seed 冻结和 forward shadow 监控。
