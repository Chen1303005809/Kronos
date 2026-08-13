# V7 多品种目标单流主动风控实施与交接方案

> 状态：P0 已实施并通过；P1-P5 尚未实施
> 更新日期：2026-08-13
> 方案版本：V7
> 当前 phase：P0 passed / P1 unlocked
> 权威 P0 运行：`v7_p0_20260813_verified`
> P0 代码与证据检查点：`9e8d675`
> 结果范围：`retrospective_observed_diversified_contracts`
> 生产资格：`production_eligible: false`

## 0. 给下一会话的直接指令

下一会话从本文件开始，不要重新设计 V7，不要重跑数据搜索，也不要从其他 `v7_p0_*` 目录挑选更好结果。

执行顺序固定为：

1. 阅读 `PROJECT_PATH_GUIDE.md`、`AGENTS.md`、本文件和 `csj/configs/risk_control_v7.yaml`。
2. 校验第 2 节列出的快照、P0 gate 和 SHA-256；只承认 `v7_p0_20260813_verified`。
3. 先把本文件中已经冻结的 P1-P5 参数迁入 V7 配置及严格校验，不根据 evaluation 结果修改。
4. 只实现 P1，完成测试、smoke、全量可恢复路径库和 P1 gate。
5. P1 通过后才实现 P2；其后每个 phase 也必须由上游持久化 gate 解锁。
6. 正式 evaluation 必须在候选、基线、阈值、代码提交和 checkpoint 哈希全部冻结后运行一次。
7. 没有真实基础仓位与执行约束时，P4 只能做 unit-long/unit-short 风险诊断，不能生成正式 P4 通过结论。

当前 `csj/v7/experiment.py` 只有 `audit` 命令。本文后续出现的 P1-P5 命令均是待实现接口，不是现有能力。

## 1. V6 到 V7 的迁移决策

风险目标没有变化：V7 仍然估计未来三个完整交易日内，对多头或空头持仓不利的尾部风险；它不预测交易方向，也不能自主开仓。

V7 只替换了被 V6 P0 证实不可行的两个数据协议前提，并保留 V6 的失败证据：

| 维度 | V6 | V7 冻结决定 | 原因 |
|---|---|---|---|
| 产品池 | `i/jm/rb` 训练，`j` 迁移 | 21 品种训练/诊断，`i/jm/rb` 正式门禁 | V6 早期 fold 的尾部事件不足 |
| 产品选择 | 固定少量品种 | 只按历史覆盖度选择 | 不用标签或 evaluation 表现选品种 |
| 切分键 | target completion day | `origin_trading_day` | 保证同一预测原点日原子进入一个 split |
| 研究起点 | 由旧案例日历隐式决定 | `2025-10-15` | 避免少量长历史品种主导稀疏前段 |
| 标签 | 三日波动尺度化 MAE/P80 | 不变 | 不靠降低风险标准补样本 |
| 风控 seam | 预测与执行分离 | 不变 | 模型只提供风险依据，overlay 只减仓 |

固定阶段：

```text
P0 数据、风险标签、支持度与防泄漏审计（已通过）
  ↓
P1 冻结 Kronos 路径/hidden 缓存、简单风险基线与工程质量 gate
  ↓
P2 冻结主干的多任务风险头与增量 edge gate
  ↓
P3 概率校准、告警预算与 abstain gate
  ├─→ 预测概率 forward shadow（P3 通过即可开始积累）
  ↓
P4 只减仓 overlay 回放（必须有真实基础仓位与执行数据）
  ↓
P5 模型/策略冻结与新增数据正式 shadow evaluation
```

P0 通过只说明“数据足够研究”，不说明 Kronos 已经有稳定 edge。稳定 edge 至少需要 P2 的增量预测证据、P3 的概率可信度和 P5 的冻结前瞻证据；若要用于仓位，还必须有 P4。

## 2. 已冻结的 P0 事实

### 2.1 权威输入与哈希

只允许使用下列 P0 证据作为后续输入：

```text
snapshot:
  csj/data/active_contract_snapshots/20260813T105916+0800/
manifest sha256:
  5239b20176002a615f0c50dd7527962b511020620ad2c146e4d4210ff5b430a3
data fingerprint:
  2387ff95a1f7bedc7a96377d3af0a45a4bf02f707c2edab1d3f47731d6e0ea1c
P0 audit:
  csj/runs/risk_control_v7/v7_p0_20260813_verified/data_audit.json
P0 audit sha256:
  45bc5866e9dadbf897c07fc688809e0906a763076c1002b111c3709f6fabd32b
P0 gate:
  csj/runs/risk_control_v7/v7_p0_20260813_verified/p0_gate.json
P0 gate sha256:
  7e606cc45acbf2cfaa99e92bba8373e8e26e2b19fc6df2fc80e8859a950887bf
```

快照是 `partial_panel`：共请求 63 个未交割具体合约，50 个成功、13 个失败。它是当前观察 cohort，不是完整历史活跃合约面板；失败合约不能被当作零行情，也不能通过查询已交割合约补齐。

验证命令：

```bash
shasum -a 256 \
  csj/data/active_contract_snapshots/20260813T105916+0800/manifest.json \
  csj/runs/risk_control_v7/v7_p0_20260813_verified/data_audit.json \
  csj/runs/risk_control_v7/v7_p0_20260813_verified/p0_gate.json
```

任一哈希不符时停止，不自动重跑并覆盖权威运行。

### 2.2 产品角色与案例范围

训练/诊断池固定为：

```text
a, b, br, bu, c, eb, eg, fu, hc, i, jm, l, m, p, pg, pp, rb, ru, sp, v, y
```

其中：

- `i/jm/rb` 是 `gate_products`，承担正式业务门禁和逐品种退化约束。
- 其余 18 个品种是 auxiliary products，用于增加训练覆盖和报告跨品种稳定性；它们不能在看到 evaluation 后被逐个剔除。
- 首轮风险头不输入 product ID，不训练 product-specific head 或 calibrator。
- 非 `i/jm/rb` 的结果在 V7 只能作为辅助诊断，不能自动扩展生产适用范围。

产品进入池的规则是“当前统一快照中至少一个成功具体合约满足 `bars >= 1300` 且首根 bar 不晚于 `2025-10-01`”。产品入池后，案例构造器仍可使用该产品下其他能够独立形成 `256` 根 context 和完整三日目标的 observed contracts。不能误写成“每个入池合约都有 1300 根 bar”。

权威 P0 共形成：

- 5703 个完整带标签案例、200 个预测原点日；
- 五折记录 17263 条，去重后实际被五折使用的 case key 为 5154 个；
- 正式 outer evaluation 的 case key 在五折间无重复；
- integrity、leakage、protocol failure 均为 0。

P1 原始路径和 hidden 应按 5154 个唯一 case 缓存一次，不能按 17263 条 fold 记录重复推理。fold-specific 阈值穿越特征再按 fold 派生。

### 2.3 五折支持度

| fold | fit cases / long / short | validation cases / long / short | evaluation cases / long / short |
|---|---:|---:|---:|
| 00 | 890 / 178 / 178 | 558 / 98 / 102 | 613 / 45 / 198 |
| 01 | 1446 / 290 / 290 | 601 / 49 / 200 | 666 / 147 / 106 |
| 02 | 2037 / 408 / 408 | 660 / 189 / 80 | 714 / 207 / 122 |
| 03 | 2692 / 539 / 539 | 707 / 135 / 132 | 756 / 240 / 62 |
| 04 | 3387 / 678 / 678 | 748 / 249 / 97 | 788 / 136 / 150 |

五折 pooled evaluation 的核心产品支持度：

| product | cases | long events | short events |
|---|---:|---:|---:|
| i | 596 | 151 | 91 |
| jm | 602 | 111 | 83 |
| rb | 539 | 131 | 95 |

其他 `v7_p0_20260813*` 运行是实现过程中的探索记录，不得替代 `verified`，也不得把多个运行中最有利的统计拼接到一起。

P0 为验证支持度已经物化并检查过 evaluation 标签及事件计数，所以 V7 的 outer evaluation 是“协议隔离的回看 holdout”，不是研究者完全未见的 blind set。它仍可检验尚未生成的模型预测，但稳定性最终必须由 P5 的冻结后新增数据确认。

## 3. 研究边界与对外接口

### 3.1 输入边界

- 每个案例只输入一个具体交割月目标合约自己的最近 `256` 根小时 OHLCVA 和对应时间戳。
- 不输入邻居、连续合约、主力拼接、持仓量、新闻、宏观、其他品种行情或未来可得性标记。
- horizon 是预测原点后的三个完整交易日；每日只接受 5 或 7 根小时 bar，因此总长度只能是 `15/17/19/21`。
- 产品池扩展发生在样本层，不是把 21 个品种同时拼成多变量输入。

### 3.2 风险预测接口

计划只公开一个深接口：

```text
RiskForecaster(context, future_timestamps) -> RiskForecast

RiskForecast:
  case_key
  p_long_adverse
  p_short_adverse
  expected_long_mae
  expected_short_mae
  future_vol_ratio
  abstain
  abstain_reason
  model_version
  calibration_version
  data_fingerprint
```

`RiskForecaster` 不返回 `BUY/SELL`，不读取基础策略收益、当前盈亏或真实未来标签。

### 3.3 风险执行接口

```text
RiskOverlay(base_position, RiskForecast) -> RiskOverlayDecision
```

必须始终满足：

```text
base_position == 0       => target_position == 0
base_position != 0       => sign(target_position) == sign(base_position)
abs(target_position)     <= abs(base_position)
```

模型不能开仓、反向、增加杠杆，也不能替代硬止损、保证金、流动性、涨跌停和人工熔断。

## 4. 冻结的数据、标签与切分协议

### 4.1 风险标签

令预测原点 close 为 `C0`。仅用 context 最后 60 根 close-to-close 对数收益计算：

```text
sigma_t = max(EWMA_STD(log_return, halflife=20, adjust=false, bias=false), 1e-5)
scale_H = sigma_t * sqrt(target_bar_count)

long_mae  = max(0, -min_u(log(L_u / C0))) / scale_H
short_mae = max(0,  max_u(log(H_u / C0))) / scale_H
```

每个 outer fold 只使用该 fold 的全部 21 品种 fit cases 计算全局 P80：

```text
long_tail_event  = long_mae  >= fit_long_P80
short_tail_event = short_mae >= fit_short_P80
```

这不是 product-specific threshold。不得在 P1-P5 改成逐品种分位数、降低 P80、改变 `>=`、改变 EWMA 定义或用 evaluation 重算阈值。未来波动辅助标签仍沿用 V6 的 `future_vol_ratio` 定义。

### 4.2 切分与原子性

- `split_key = origin_trading_day`。
- 五折 expanding walk-forward：60 日 minimum fit、20 日 validation、20 日 evaluation、20 日 step、3 日 purge。
- 同一 `origin_trading_day` 的所有品种和具体合约必须在同一个 split。
- 三日标签不能跨 split 边界。
- 基线、checkpoint、校准器、alert threshold 和仓位阈值只能读取 fit/validation。
- bootstrap 以交易日块重采样，不把同日多个品种或合约当成独立样本。

### 4.3 数据服务限制

K 线服务不能查询已交割合约。V7 后续若积累新增数据，只能：

1. 读取已经在合约活跃时落盘且带 manifest/哈希的不可变快照；或
2. 对当时确认仍活跃的具体合约向前采集，并创建新的不可变快照。

不得枚举旧交割月、持续重试已交割合约、覆盖旧快照，或用连续合约替代具体合约。

## 5. 实施状态与目标文件

已存在：

```text
csj/v7/config.py          # P0 配置校验
csj/v7/audit.py           # 覆盖度、origin-day folds、标签与 gate
csj/v7/experiment.py      # 仅 audit
tests/test_v7_risk_control.py
```

下一会话计划新增或扩展：

```text
csj/v7/path_bank.py       # 唯一 case 的 raw path/hidden 缓存与 fold 摘要
csj/v7/baselines.py       # 固定风险基线及 validation 选择
csj/v7/risk_head.py       # 冻结主干的多任务 head 与 matched control
csj/v7/calibration.py     # Platt、alert threshold、abstain
csj/v7/overlay.py         # position-reduction-only adapter
csj/v7/evaluation.py      # 指标、交易日块 bootstrap、正式 evaluation ledger
csj/v7/reporting.py       # JSON/Markdown/必需图表
csj/v7/experiment.py      # phase-gated CLI 编排
tests/test_v7_path_bank.py
tests/test_v7_risk_head.py
tests/test_v7_calibration.py
tests/test_v7_overlay.py
tests/test_v7_evaluation.py
```

优先把复杂实现隐藏在这些模块内，不把 P1-P5 继续堆入 CLI 文件。

## 6. P1：冻结路径、hidden 与简单基线

### 6.1 模型与采样参数

P1 实现前，将下列 V6 已预注册参数原样迁入 V7 配置并加入严格校验：

```text
tokenizer_id       = NeoQuasar/Kronos-Tokenizer-base
tokenizer_revision = 0e0117387f39004a9016484a186a908917e22426
predictor_id       = NeoQuasar/Kronos-small
predictor_revision = 901c26c1332695a2a8f243eb2f37243a37bea320
max_context        = 512
freeze_tokenizer   = true
freeze_predictor   = true

sample_count       = 64
temperature        = 1.0
top_k              = 0
top_p              = 0.9
minimum_valid      = 60/64 per case
global_valid_rate  >= 0.99
```

不得使用未固定 revision 的 `from_pretrained`。首轮不解冻 tokenizer/Kronos，不搜索采样参数。

### 6.2 唯一 case 缓存

缓存 key 必须包含：

```text
strategy_version + data_fingerprint + case_key
+ tokenizer_revision + predictor_revision
+ sample_count + temperature + top_k + top_p
```

seed 使用上述 key 的 SHA-256 派生，禁止使用 Python `hash()`。一个 case 的 64 条路径作为原子任务生成；同一设备、依赖版本和 seed 下重跑必须一致。运行记录必须保存 Python、PyTorch、CUDA/MPS 和设备信息，不能宣称跨设备 bitwise 一致。

核心 `KronosPredictor.predict()` 当前只返回样本均值。V7 应使用自己的适配器调用支持 `return_samples=True` 的底层推理，不得把 64 条路径先平均后再计算尾部风险。若修改 `model/kronos.py` 公共接口，必须先补核心回归测试。

每个唯一 case 保存：

- 64 条去归一化后的原始 OHLCVA 路径和目标时间戳；
- context 归一化统计、`C0`、`scale_H`、clip fraction；
- 冻结 Kronos 最后一个 context token 的 hidden；
- 模型 revision、seed、输入签名和输出 SHA-256。

大数组放在已忽略的 `csj/artifacts/v7_path_bank/<cache_key>/`，按 target length 分组并以压缩 NumPy shard 保存；`csj/runs/risk_control_v7/<run_id>/p1_path_bank/manifest.json` 只保存相对路径、case keys、shape、dtype 和每个 shard 的 SHA-256。不得生成 5154 个巨型 JSON，也不得把无校验的本地 cache 当成正式证据。

必须支持：原子 shard 写入、`--resume`、已完成 shard 哈希复验、失败 case 单独记录、smoke 与 full 产物隔离。smoke 可用少量 case 和 4 条路径验证工程逻辑，但所有结果必须标记 `smoke: true`，不能触发正式 gate。

### 6.3 路径有效性与 fold 派生

路径不得静默修复。以下任一条件使该 sample 无效：

- 任意 OHLCVA 非有限；
- `high < max(open, close)` 或 `low > min(open, close)`；
- `high < low`；
- volume/amount 明显非法，或时间戳数量/顺序不符。

每个 case 至少保留 60 条有效路径，否则输出 `abstain=path_quality`。不能为失败 case 额外补抽样。

raw path、hidden 和不依赖阈值的路径 MAE 只算一次。以下特征必须按 `fold_id` 使用该 fold 的 fit-only P80 再派生：

- long/short threshold crossing fraction；
- long/short adverse MAE 的 median/P80/P95；
- future volatility 的 median/P90；
- 路径 IQR、方向熵和无效路径比例。

同一 case 在不同 fold 中可能有不同的 threshold crossing fraction，这是正确行为；复制某一 fold 的阈值到所有 fold 是泄漏/协议错误。

### 6.4 冻结的基线

每个 fold 只在 fit 拟合，在 inner validation 选择一个最强基线：

1. `fit_global_event_rate`
2. `fit_product_event_rate`
3. `ewma_volatility_rank`
4. `atr20_rank`
5. `fixed_context_logistic`
6. `zero_shot_path_risk`

固定 context 特征只包括过去可计算的 EWMA 波动、ATR20/价格、近 20 根振幅、最近 1/3 个完整交易日绝对收益、volume/amount 历史 z-score 和 clipping fraction。除单独的 `fit_product_event_rate` 外，不给 logistic 或 P2 head 输入 product ID。

rank score 必须只用 fit 拟合单变量概率映射；product event rate 要记录无样本回退到 global rate 的规则。不得在 evaluation 后新增技术指标。

基线选择分数固定为：

```text
selection_brier = 0.5 * macro_brier(i/jm/rb, long/short)
                + 0.5 * macro_brier(all 21 products, long/short)
```

macro 先对有案例的 `product × side` 单元等权，再计算上式；没有案例的单元必须列出，不能填零。这样既不让样本多的核心合约淹没其他品种，也不让 auxiliary products 改写正式业务目标。每 fold 选择一个基线，选择记录和参数必须冻结。

### 6.5 P1 gate

P1 只判断工程与信号非退化，不把 evaluation 标签用于调参：

- 全量有限且满足 OHLC 约束的路径比例至少 99%；
- 每个非 abstain case 至少 60/64 条有效路径；
- fit/validation 的 long/short raw path risk 都不是全 0、全 1 或零方差；
- 相同 cache key 重跑抽检结果一致，所有 shard 哈希通过；
- evaluation context 可生成路径，但 P1 不输出其预测性能指标；
- path bank manifest、失败表、覆盖图、有效路径分布图和风险概率分布图齐全。

基线选择属于 validation 模型评估，因此还必须为被选基线生成 validation 的 long/short PR 曲线、reliability diagram 和预测概率分箱实际事件率图；不得用 evaluation 标签补这些图。

P1 不要求 zero-shot Kronos 必须优于基线；它的结果会作为 P2 的固定输入和竞争基线。P1 失败时停止，不调 `top_p`、不增加 sample count、不丢弃表现差的品种。

## 7. P2：冻结主干的多任务风险头

### 7.1 模型结构

Tokenizer 和 Kronos 全部冻结。hidden 取最后一个 context token，维度从模型配置读取：

```text
Kronos hidden
  -> LayerNorm -> Linear(d_model, 128) -> SiLU

固定 context 特征 + fold-specific path summaries
  -> LayerNorm -> Linear(feature_dim, 64) -> SiLU

concat
  -> Linear(192, 128) -> SiLU -> Linear(128, 5)
```

五个输出：

```text
long_tail_logit
short_tail_logit
log1p_expected_long_mae
log1p_expected_short_mae
future_vol_ratio
```

首轮 `dropout=0`，不搜索宽度。matched control 使用完全相同的特征、输出、损失、采样和训练协议，但将 Kronos hidden 与 path summaries 置零，只保留固定 context 特征。

### 7.2 训练协议

```text
loss = BCE(long_tail) + BCE(short_tail)
     + 0.25 * SmoothL1(log1p(long_mae))
     + 0.25 * SmoothL1(log1p(short_mae))
     + 0.25 * SmoothL1(future_vol_ratio)
```

- seeds：`42/43/44`；保留单 seed 和等权概率 ensemble。
- AdamW：LR `3e-4`、weight decay `0.01`、batch size `64`、最多 30 epochs、patience `5`、gradient clip `3.0`。
- 不使用 class weight/focal loss，不搜索 loss 权重。
- sampler 固定为 `prediction_day_product_uniform`：先均匀选预测日，再均匀选该日产品，再均匀选具体 case，防止多合约产品和高密度日期主导训练。
- 每个 epoch 有放回抽取 `len(fit_records)` 个样本；fold/seed 使用独立、可复现的随机生成器。
- checkpoint 只按 inner validation 的 `selection_brier` 选择。
- 一个 fold 的 fit/validation/evaluation 只能使用该 fold 对应的标签阈值和 path summaries。

### 7.3 正式 evaluation 封存

训练与选择阶段不读取 evaluation metrics。首次正式评估前写入不可覆盖的 `candidate_freeze.json`：

- Git commit、完整 resolved config SHA；
- P0 gate SHA、data fingerprint；
- path/hidden manifest SHA；
- 三 seed candidate/control checkpoint SHA；
- 每 fold 预选基线及参数；
- metric/gate 版本和图表合同。

`p2-evaluate` 必须创建 `evaluation_ledger.json`，同一 candidate hash 再次正式执行要拒绝或明确标记为复算，不能挑最好一次。评估程序可以读取已经物化的 P0 evaluation labels，但实现者不得用其结果回头改模型或 gate；若要改，必须递增主版本。

### 7.4 P2 gate

相对每 fold 在 inner validation 预选的最强基线，核心 `i/jm/rb` evaluation 必须同时满足：

- long/short 平均相对 Brier 改善至少 2%；
- long/short 平均 PR-AUC 提高至少 0.02；
- 使用各模型 inner-validation 固定 20% alert budget 阈值时，平均 tail recall 提高至少 5pp；
- 至少 3/5 fold 的 macro Brier 改善；
- `i/jm/rb` 任一产品 macro Brier 的绝对恶化不超过 0.02；
- 5 日和 10 日 moving-block bootstrap 的 `P(Brier 改善 > 0)` 都至少 80%；
- long/short severity 的 pooled Spearman 都为正，至少一侧达到 0.10；
- 三个 seed 至少 2/3 相对基线改善，seed 间 macro Brier 标准差不超过 0.02。

其中相对 Brier 改善固定为 `(baseline_brier - candidate_brier) / baseline_brier`；核心 macro 对 `i/jm/rb × long/short` 六个单元等权。bootstrap 对 paired per-case Brier loss difference 按 `origin_trading_day` 做 5/10 日 moving blocks，同一天的全部产品与合约必须一起重采样；severity Spearman 只在核心产品 pooled evaluation 上计算。

相对 matched control 还必须满足：

- long/short 平均相对 Brier 改善至少 1%；
- 平均 PR-AUC 不低于 control，且至少一侧提高 0.01；
- 至少 3/5 fold 的 macro Brier 改善。

21 品种等权 macro、逐品种指标和辅助品种分布必须完整报告，但 V7 正式通过与否仍由 `i/jm/rb` gate 决定。不得在看到结果后删除拖后腿的 auxiliary product 再重跑。

P2 失败即停止风险头路线；不能通过解冻主干、搜索标签、增加 seed、换品种池或调 loss 挽救同一 V7。

## 8. P3：概率校准、告警预算与 abstain

- 对三 seed ensemble 的 long/short logits 分别拟合共享 affine Platt calibration。
- 只汇总 `i/jm/rb` inner validation，目标固定为 Bernoulli NLL；不训练 product-specific calibrator。
- soft/hard alert 固定为 `i/jm/rb` inner-validation 校准概率 P80/P95，evaluation 不重算。
- context 缺失、clip fraction 超过 5%、有效路径少于 60、数据指纹或产品不支持时必须 abstain。
- `i/jm/rb` 是正式可用域；其他 18 品种即使输出研究概率，也要标记 `research_only_product=true`。

P3 gate：

- 校准后的核心 macro Brier 不差于未校准 ensemble，也不差于预选基线；
- pooled long/short ECE 均不高于 0.05，`i/jm/rb` 各产品均不高于 0.10；
- 核心 evaluation 非 abstain 覆盖率至少 95%；
- 所有 abstain 原因可枚举且计数完整；
- PR 曲线、reliability diagram、概率分箱实际事件率、概率与真实 MAE 图全部生成。

P3 通过后可以立即冻结模型并开始“只记录不执行”的 forward probability shadow，避免等待 P4 外部仓位数据才开始积累前瞻证据；这不授予 overlay 生产资格。

## 9. P4：只减仓 overlay 回放

### 9.1 固定映射

```text
base_position > 0  -> 使用 p_long_adverse
base_position < 0  -> 使用 p_short_adverse
base_position == 0 -> target_position = 0

p <= soft:          multiplier = 1.00
soft < p < hard:    multiplier 从 1.00 线性下降到 0.50
p >= hard:          multiplier = 0.25
abstain:            multiplier = 1.00，并交给模型外硬风控
```

每个预测日只使用最新预测，下一根可成交 bar 执行，不叠加重叠三日预测。

### 9.2 正式前提

必须提供逐时点外部 `base_position`、具体可交易合约映射，以及合约乘数、tick、保证金、仓位上限、移仓、手续费、价差、滑点、涨跌停和成交容量。

缺任一关键项时：

- 可以实现 overlay 不变量和 unit-long/unit-short 风险诊断；
- 必须写 `formal_gate_available: false` 和缺失项；
- 不得输出正式 PnL/CVaR 改善结论，也不得解锁仓位生产使用。

### 9.3 P4 gate

相对完全相同的基础仓位，扣除新增换手成本后：

- 5% downside CVaR 至少改善 5%；
- 最大回撤至少改善 5%；
- 净收益保留率至少 85%；
- 至少 3/5 fold 的 downside CVaR 改善；
- `i/jm/rb` 任一产品 downside CVaR 不恶化超过 2%；
- 5/10 日块 bootstrap 的 `P(CVaR 改善 > 0)` 均至少 80%。

还必须战胜 `exposure_matched_random_overlay`：按 fold、产品和持仓方向保持相同平均 multiplier 与调仓次数，只随机打乱减仓日期。V7 downside CVaR 至少优于该 control 2%，且 5/10 日块概率均至少 80%。

必须生成基础/overlay 资金曲线、回撤曲线、风险概率与真实不利波动关系图。这里检验的是风险 overlay 的增量，不是交易 alpha。

## 10. P5：冻结与新增数据

### 10.1 冻结对象

P3 通过后冻结预测组件；P4 通过后再冻结执行组件：

- tokenizer/predictor revision、三 seed 风险头和校准器；
- 标签规则、fit P80、soft/hard alert threshold；
- path sampling、abstain、产品适用域；
- overlay multiplier、基础策略接口、合约映射和成本模型。

### 10.2 前瞻采集

- 每个采集时点只查询仍活跃合约并创建新快照，不覆盖历史快照。
- 预测先落盘并哈希，至少三个完整交易日后标签成熟再评估。
- 前 20 个新增交易日只做 shadow checkpoint，不调参。
- 至少 60 个新增交易日形成第一段正式 forward 窗口。
- 每 20 日报告 Brier、PR-AUC、ECE、alert rate、tail recall、覆盖率、数据漂移；有正式 overlay 时再报告 CVaR、回撤和收益保留率。
- 连续两个 20 日窗口出现 `Brier skill <= 0`、`ECE > 0.10` 或输入漂移告警时停用模型 overlay，启动新版本评审，不原地静默重训。

历史数据没有“持续有效到某个固定日期”的保证。P0-P4 只能证明冻结历史窗口内的回看行为；模型是否继续有效，要由 P5 的新增数据持续验证。即使全部历史 gate 通过，`production_eligible` 也不能自动改为 true。

## 11. 测试、泄漏与可视化合同

至少测试：

- 修改预测原点后的 OHLCVA 不改变 context、hidden、path 输入或风险预测；
- context scale 只读历史，tail threshold 只读当前 fold fit；
- 同一 origin day 原子切分、三日 purge、split case key 不重叠；
- path cache 不因 fold、遍历顺序或 resume 重复生成；
- 同一 case 在不同 fold 使用各自阈值；
- 64 条 raw paths 没有在提取风险前被平均；
- tokenizer/Kronos 训练前后参数与 hash 不变且无梯度；
- sampler 实现真正的 day→product→case 均衡；
- baseline/checkpoint/calibration/alert threshold 不读取 evaluation；
- formal evaluation ledger 拒绝未冻结候选和静默覆盖；
- overlay 永不增仓、反向或从零开仓；
- abstain 不计入模型成功覆盖，并完整报告原因。

每个正式模型评估必须生成图表。最低集合：

- long/short PR 曲线与 reliability diagram；
- 预测概率分箱对实际事件率与真实 MAE；
- 预测/真实 severity 散点或分位图；
- fold、核心产品、21 品种 macro 的对比图；
- P4 的资金与回撤曲线。

任一必需图表缺失使对应模型评估不完整，不能把 gate 标记为通过。

## 12. 计划 CLI、产物与 provenance

待实现 CLI：

```bash
.venv/bin/python -m csj.v7.experiment audit --run-id <run_id>
.venv/bin/python -m csj.v7.experiment p1-path-bank --run-id <run_id> --resume
.venv/bin/python -m csj.v7.experiment p1-baselines --run-id <run_id>
.venv/bin/python -m csj.v7.experiment p2-train --run-id <run_id>
.venv/bin/python -m csj.v7.experiment p2-evaluate --run-id <run_id>
.venv/bin/python -m csj.v7.experiment p3-calibrate --run-id <run_id>
.venv/bin/python -m csj.v7.experiment p4-overlay --run-id <run_id> --base-positions <path>
.venv/bin/python -m csj.v7.experiment p5-freeze --run-id <run_id>
```

目标产物：

```text
csj/runs/risk_control_v7/<run_id>/
  resolved_config.json
  upstream_p0.json
  p1_path_bank/manifest.json
  p1_path_bank/failures.json
  p1_baselines/selection.json
  p1_gate.json
  p2_risk_head/candidate_freeze.json
  p2_risk_head/evaluation_ledger.json
  p2_gate.json
  p3_calibration/
  p3_gate.json
  p4_overlay/
  p4_gate.json
  p5_freeze/

csj/results/risk_control_v7/
  p0_gate.json
  p1_gate.json
  p2_gate.json
  p3_gate.json
  p4_gate.json
  REPORT.md
  figures/
```

所有正式 JSON 必须包含：`strategy_version`、`phase`、`run_id`、`result_scope`、`production_eligible`、Git commit、resolved config SHA、data fingerprint、模型/tokenizer revision、风险标签版本、上游 gate 路径与 SHA，以及 `smoke` 标记。

每个 phase 要先验证上游 `allows_next_phase: true`、版本、scope、fingerprint 和 SHA，再运行；不得只判断文件存在。

## 13. 下一会话的实施批次

### Batch A：协议迁移与守卫

- 扩展 `risk_control_v7.yaml` 的 model/path/risk-head/calibration/overlay/evaluation/runtime 段；值采用本文冻结值。
- 扩展 `validate_v7_config`，测试任一关键值漂移都拒绝。
- 实现通用 gate provenance 校验和 planned CLI guard。

完成条件：只跑单元测试，不调用模型，不产生正式 evaluation。

### Batch B：P1 可恢复缓存

- 先写 path validity、cache key、seed、shard、resume、fold threshold 派生测试。
- 实现 8 cases × 4 paths smoke，人工检查 raw path shape、时间戳和图。
- 实现 5154 unique cases × 64 full path/hidden bank，生成 P1 gate。

完成条件：P1 gate 真实通过；否则停止。

### Batch C：P2 训练与一次正式评估

- 先实现固定特征、均衡 sampler、冻结主干和 matched control 测试。
- 每 fold/seed 训练 candidate/control，只用 validation 选 checkpoint 和 baseline。
- 写 `candidate_freeze.json` 后才运行一次 `p2-evaluate`。

完成条件：P2 gate 通过且图表齐全；失败则不实现 P3/P4 来包装结果。

### Batch D：校准与风险接口

- 实现 shared Platt、alert budget、abstain 和 `RiskForecaster`。
- 冻结 P3 后开始 forward probability shadow。

完成条件：P3 gate 通过，否则 raw score 不能称为概率。

### Batch E：overlay 与 forward

- 先确认真实 `base_position` 和执行元数据是否齐全。
- 齐全才做 P4；否则只交付 diagnostic 并明确 formal gate unavailable。
- P5 只评估冻结后的新数据，不回改 V7。

## 14. 停止条件与完成定义

立即停止当前 phase 的情形：

- 上游 gate、fingerprint、revision 或 SHA 不匹配；
- 需要改变标签、切分、产品池、正式指标或 gate 才能继续；
- 简单历史特征基线与 matched control 已经同样好；
- 概率无法校准，或改进只来自平均仓位下降；
- 缺少模型评估图表；
- 缺少真实基础仓位却准备声称策略回测有效；
- 把 observed-contract 回看结果写成生产结论。

V7 的真正完成不是“训练跑完”，而是形成一条可审计结论：

```text
P0 数据足够
+ P1 路径工程有效
+ P2 对强基线与 matched control 有增量
+ P3 概率可信
+ P4 在真实基础仓位上有净风险收益（若要用于仓位）
+ P5 冻结后新增数据仍成立
```

其中任何一项失败，都应保留失败证据，而不是继续调参直到看起来好看。

## 15. 建议提交边界

1. `docs(v7): freeze diversified risk-control implementation protocol`
2. `test(v7): register P1 cache and provenance contracts`
3. `feat(v7): add resumable frozen Kronos path bank`
4. 同步 P1 full 结果并审阅 gate；失败即停止。
5. `test(v7): register balanced risk-head evaluation contract`
6. `feat(v7): add frozen-backbone risk head and matched control`
7. 同步一次正式 P2 evaluation；失败即停止。
8. `feat(v7): add shared calibration and abstention`
9. `feat(v7): add position-reducing overlay replay`
10. P4 通过后再实现 forward freeze/monitor。
