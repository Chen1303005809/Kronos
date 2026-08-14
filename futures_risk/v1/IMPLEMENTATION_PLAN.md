# Futures Risk V1：冻结 Kronos 接口的期货原生双分支风控方案

> 状态：方案冻结，P0-P5 尚未实施  
> 更新日期：2026-08-14  
> 方案版本：V1  
> 当前 phase：design frozen / P0 pending  
> 生产资格：`production_eligible: false`

## 0. 给实施会话的直接指令

本目录是与 `csj/` 同级的新项目，不在 `csj/v7` 上继续打补丁，也不修改 Kronos 的预训练输入接口。

实施顺序固定为：

1. 阅读根目录 `PROJECT_PATH_GUIDE.md`、`AGENTS.md` 和本文件。
2. 先实现 P0 数据能力与合约生命周期审计；缺少最低生命周期元数据时，P1 之后的训练全部锁定。
3. P1 只能封装并校验冻结 Kronos，不得改变 OHLCVA、时间戳语义、tokenizer、predictor 或 checkpoint。
4. 先完成期货原生基线 `F0`，再训练含 Kronos 表示的 `KF`；不得只训练候选而没有匹配对照。
5. `KF` 只有在预注册的增量门禁上胜过 `F0` 和 `KF-shuffle`，才允许保留 Kronos 依赖。
6. V1 不生成未来 OHLCVA 路径，不把路径合法率作为训练前提，也不使用 V7 的失败路径库。
7. evaluation 只能在接口、特征、标签、切分、候选、checkpoint 选择规则和门禁全部冻结后执行一次。
8. 任一 gate 失败即停止对应下游 phase；不得通过删品种、改阈值、调报警预算或重跑挑最好结果解锁。

## 1. V1 要回答的唯一核心问题

V1 不预设 Kronos 对期货有用，而是检验：

> 在一个已经能够读取期货交易时段和合约生命周期的原生风险模型上，冻结 Kronos 表示是否提供稳定、可复现、可校准的增量风险信息？

这拆成两个独立命题：

1. `F0` 是否比只使用事件率或简单历史波动的基线更有用；
2. `KF` 是否比参数匹配的 `F0` 更有用。

只有两个命题都通过，才能说“使用 Kronos 的期货风控模型有用”。如果 `F0` 有用而 `KF` 无增益，正确结论是交付期货原生模型并删除 Kronos，而不是继续用 LoRA 修补。

## 2. 冻结的架构决策

### 2.1 seam 的位置

模型 seam 固定放在 Kronos hidden 输出之后，而不是 Kronos 输入之前：

```text
OHLCVA + 原始五维时间戳
          │
          ▼
FrozenKronosEncoder ───────► z_kronos ───────┐
                                              │
历史期货状态 + 合约生命周期                    ├─► RiskFusionHead ─► RiskForecast
          │                                   │
          ▼                                   │
FuturesStateEncoder ───────► z_futures ──────┘
```

V1 有三个公开 module interface：

```text
FrozenKronosEncoder(KronosContext) -> KronosRepresentation
FuturesStateEncoder(FuturesContext) -> FuturesRepresentation
RiskFusionHead(KronosRepresentation | NullKronosSlot,
               FuturesRepresentation) -> RiskForecast
```

复杂的数据校验、归一化、日内分组、生命周期计算、缺失能力判断和门禁都必须封装在对应 module 内。调用方不能自行拼接特征或绕过 gate。

### 2.2 Kronos 接口不可变

`KronosContext` 严格保持预训练接口：

```text
values:
  open, high, low, close, volume, amount
timestamps:
  minute, hour, weekday, day, month
context_length:
  256 hourly bars
```

以下行为在 V1 禁止：

- 把持仓量、距到期日、合约编号或主力排名追加到六维输入；
- 把五维时间嵌入中的任一字段改成期货生命周期字段；
- 扩展 tokenizer 输入宽度；
- 增加 prefix token、cross-attention 或新的条件通道；
- 对 tokenizer、predictor 或 temporal embedding 做 LoRA、全量微调或梯度更新；
- 通过修改归一化来适配期货；
- 把未来期货特征送入历史 hidden 提取。

冻结 checkpoint、tokenizer checkpoint、归一化定义、clip、上下文长度和 hidden 提取位置必须写入 provenance 并带 SHA-256。

V1 的 `KronosRepresentation` 固定为最后一根历史 token 的 predictor hidden：

```text
z_kronos = stop_gradient(hidden[:, -1, :])
```

任何 full-sequence pooling、跨层拼接或可训练 Kronos pooler 都属于新候选，需要新 major version。

### 2.3 期货特征不进入 Kronos

所有期货专用信息只进入 `FuturesStateEncoder`。这些信息可以与 Kronos 表示在 `RiskFusionHead` 中交互，但不能回写或调制 Kronos 内部层。

V1 的交互方式固定为输出后门控：

```text
k = LinearNoBias(LayerNorm(z_kronos))        # d_k -> 64
f = FuturesStateEncoder(...)                 # -> 64
gamma, beta = MLP(f)                         # -> 64, 64
conditioned_k = gamma * k + beta
joint = MLP(concat(f, conditioned_k))         # -> 64
```

`F0` 使用同样形状、同样参数量和同样融合实现，只把 `z_kronos` 替换为确定性的全零 `NullKronosSlot`。因此候选与对照的区别只有 Kronos 信息本身，而不是 head 容量。

## 3. 任务边界与输出

### 3.1 风控任务

V1 估计预测原点后未来三个完整交易日内，对既有多头或空头仓位不利的尾部风险。它不预测开仓方向，不返回交易信号，也不生成未来 K 线路径。

公开输出固定为：

```text
RiskForecast:
  case_key
  p_long_adverse
  p_short_adverse
  expected_long_mae
  expected_short_mae
  future_vol_ratio
  abstain
  abstain_reason
  model_arm
  model_version
  calibration_version
  data_fingerprint
```

风险头的五个数值输出与模型输入 seam 相互独立。增加期货原生分支不会改变标签定义。

### 3.2 明确不做的事情

V1 不包含：

- 自动开仓、反向或加杠杆；
- 连续合约拼接和未经记录的换月复权；
- 生命周期条件化的 AR 路径生成；
- LoRA 或任何形式的 Kronos 域适配；
- 用 evaluation 选择产品、特征、模型或报警阈值；
- 把绝对日期字符串、合约完整代码或样本行号当作可学习 ID；
- 在没有真实基础仓位和执行约束时宣称可用于生产仓位管理。

若后续目标变为“让生命周期直接影响每一步路径生成”，必须建立 V2，修改模型结构并重新预训练/微调；不能把该变化伪装成 V1 的小改动。

## 4. P0 数据能力与生命周期审计

### 4.1 输入来源

首轮可以只读复用现有不可变行情快照：

```text
csj/data/active_contract_snapshots/20260813T105916+0800/
```

V1 必须自己生成 data fingerprint 和审计产物，不能把 V7 P0 通过自动视为 V1 P0 通过。新采集数据统一写入 `futures_risk/data/`，不得覆盖 `csj/data/`。

### 4.2 最低必需字段

每个案例至少必须能够因果地获得：

```text
bar-level:
  timestamp
  trading_day
  open, high, low, close, volume, amount

contract-level:
  full_contract_id
  product_id
  delivery_year_month
  listing_date
  last_trading_date 或 authoritative expiry date
```

`listing_date` 和 `last_trading_date` 必须来自带版本和来源记录的合约元数据，不能从当前快照的首尾 bar 猜测。缺少这两个字段时，P0 的 `lifecycle_metadata_gate` 失败，P1-P5 锁定。

下列字段属于增强能力，不作为首轮 P0 硬前提，但必须报告覆盖率：

```text
open_interest
contract_multiplier
margin_ratio
same-product contract volume/open-interest rank
main/secondary contract flag
neighbor maturity prices
basis / term structure
holiday calendar
```

增强字段缺失时不能填零并伪装成真实观测。只能全局禁用该特征，或同时提供显式 availability mask；使用哪种方式必须在看到 validation/evaluation 前冻结。

### 4.3 P0 必做审计

P0 至少输出：

```text
futures_risk/runs/v1/<run_id>/
  data_capability.json
  contract_metadata_audit.json
  bar_integrity_audit.json
  lifecycle_coverage.json
  split_audit.json
  p0_gate.json
  provenance.json
```

审计内容：

- OHLCVA 基本市场约束；
- 时间戳严格递增、交易日映射、每交易日 5/7 根时 K；
- 同一具体合约不拼接其他交割月；
- 上市日、交割月、最后交易日之间的逻辑关系；
- `days_since_listing >= 0`、`days_to_last_trade >= 0`；
- 预测目标不能越过该合约最后交易日；
- 每个 fold、品种和生命周期桶的案例数及尾部事件数；
- 同一预测原点日的 split 原子性；
- 数据、元数据、标签和 split 的 SHA-256。

生命周期桶在看模型结果前冻结为：

```text
early:       days_since_listing <= 20 trading days
active:      非 early 且 days_to_last_trade > 20 trading days
near_expiry: days_to_last_trade <= 20 trading days
```

如果元数据支持更可靠的主力迁移状态，可额外报告 `pre_roll/roll/post_roll`，但不得用 evaluation 表现重新划桶。

## 5. 期货原生输入与编码

### 5.1 时间特征原则

自然日期本身不作为连续数值或唯一类别输入。期货分支只使用具有市场语义、在预测原点可得的时间状态：

- `bar_position_in_trading_day`；
- `bars_in_trading_day`（5 或 7）；
- 夜盘/上午/下午 session；
- 是否为交易日首根/末根；
- 与上一根 K 的真实时间间隔；
- weekday、节前/节后标记（有权威交易日历时）；
- `days_since_listing`；
- `days_to_last_trade`；
- delivery month 的周期编码。

Kronos 分支仍使用原始 `minute/hour/weekday/day/month`，不做上述替换。

### 5.2 因果市场特征

每根历史 bar 可派生：

- close-to-close、open-to-close、high-low 和 gap 对数收益；
- candle body、上下影线、真实波幅；
- 仅用历史计算的 EWMA 波动、偏度和尾部幅度；
- volume/amount 的历史 rank、变化率和异常度；
- open interest 变化及量仓关系（仅在真实存在时）；
- 同品种流动性排名与换月迁移（仅在同一时点合约面板完整时）。

所有 rolling scaler、分位数、缺失策略和 winsorize 阈值只在 outer fold 的 fit 区间拟合。

### 5.3 FuturesStateEncoder

V1 使用参数受限的层次结构，不使用大规模 AR 生成器：

```text
per-bar feature MLP:       feature_dim -> 32
within-day masked pooling: mean + max + last, 支持 5/7 根
across-day GRU:            input 96, hidden 64, 单层
contract-state MLP:        lifecycle/meta -> 32
final projection:          concat(64, 32) -> 64
```

256 根上下文约对应 36 个交易日。日内 pooling 显式保留交易日 seam，GRU 建模跨日状态；绝对日期不承担“第几天”的功能。

完整 contract ID 不输入模型。`product_id` 可以作为低维类别特征，但 `F0`、`KF` 和 `KF-shuffle` 必须完全相同，并额外报告留一品种诊断；不得通过 product embedding 记忆具体合约。

## 6. 标签与切分

V1 首轮沿用三日主动风控标签语义：

```text
sigma_t = max(EWMA_STD(context log returns,
                       halflife=20,
                       adjust=false,
                       bias=false), 1e-5)
scale_H = sigma_t * sqrt(target_bar_count)

long_mae  = max(0, -min(log(target_low / origin_close))) / scale_H
short_mae = max(0,  max(log(target_high / origin_close))) / scale_H
```

每个 outer fold 只用 fit cases 的全产品 P80 定义：

```text
long_tail_event  = long_mae  >= fit_long_P80
short_tail_event = short_mae >= fit_short_P80
```

切分固定为按 `origin_trading_day` 原子化的 expanding walk-forward，并保留 target horizon purge。若复用 V7 的五折边界，V1 必须重新计算并逐项证明 case fingerprint、标签和 split 一致，不能只引用旧 JSON。

任何标准化器、特征阈值、类别编码器、checkpoint、校准器和报警阈值只能读取 fit/validation。evaluation 标签不能进入模型选择日志。

## 7. 固定实验臂

### 7.1 简单基线

```text
B0-prevalence:
  fold-fit 尾部事件率常数预测

B1-volatility:
  只使用历史波动、振幅、量能和生命周期标量的正则化模型
```

### 7.2 神经模型臂

```text
F0:
  FuturesStateEncoder + NullKronosSlot + RiskFusionHead

K0-diagnostic:
  FrozenKronosEncoder + 最小 head，不输入期货分支

KF:
  FuturesStateEncoder + 真实冻结 Kronos hidden + RiskFusionHead

KF-shuffle:
  FuturesStateEncoder + split 内确定性错配的 Kronos hidden + RiskFusionHead
```

`KF-shuffle` 的排列由 `case_key + global_seed + fold_id` 确定，不读取标签；fit、validation、evaluation 各自在自身 split 内排列，禁止跨 split 取 hidden。

所有神经模型使用相同训练预算、batch 顺序、优化器、早停规则和预注册种子集合。不得给 `KF` 更多 epoch 或额外调参机会。

### 7.3 多任务风险头

共享表示后接五个独立 head：

```text
classification:
  p_long_adverse
  p_short_adverse

regression:
  expected_long_mae
  expected_short_mae
  future_vol_ratio
```

分类训练损失使用 BCE-with-logits，最终模型选择指标使用校准后的 Brier。连续标签只用 fit 统计做尺度化，使用 Huber loss。损失权重、优化器、学习率、最大 epoch 和早停 patience 必须在 P2 evaluation 解锁前写入配置并冻结。

## 8. Phase 与 gate

### P0：数据与生命周期能力

通过条件：

- 所有必需审计通过；
- 生命周期最低字段完整且来源可追溯；
- 无 split 泄漏或 target 越过最后交易日；
- 支持度达到配置中的预注册阈值；
- 产物和哈希完整。

失败动作：停止。不能降级成“从第一根观测 bar 猜上市日”的伪生命周期模型。

### P1：冻结 Kronos 接口与表示库

必须验证：

- 官方 checkpoint 与配置哈希；
- 输入严格为六维 OHLCVA 和原五维时间戳；
- tokenizer encode/decode 全为有限值；
- hidden 维度、dtype、设备和 case 对齐；
- 相同 case 重跑得到相同 hidden；
- 表示库不包含 target 数据；
- V7 raw paths 不被读取。

P1 输出：

```text
kronos_representation_manifest.json
kronos_representation_gate.json
representations/<case_key>.npz
```

P1 只判断工程完整性，不宣称 Kronos 有预测 edge。

### P2：期货原生基线

先训练 `B0/B1/F0`。只有 `F0` 的 selection Brier 合法且在 validation 上至少不弱于最佳简单基线，才允许进入 P3。

若 `F0` 无法胜过简单基线，停止并判定当前数据/标签/模型不足；不得借助 Kronos 掩盖一个无效的期货风险基线。

### P3：Kronos 增量实验

训练 `K0-diagnostic/KF/KF-shuffle`，冻结所有结果后执行增量 gate。主要比较：

```text
delta_primary = mean(
  Brier_long(KF)  - Brier_long(F0),
  Brier_short(KF) - Brier_short(F0)
)
```

bootstrap 以 `origin_trading_day` 为块，保持同日多品种相关性。

Kronos 增量 gate 同时要求：

1. `delta_primary` 的 95% block-bootstrap 上界 `< 0`；
2. long、short 两侧 Brier 点估计均不劣于 `F0`；
3. `KF` 对 `KF-shuffle` 的 primary delta 的 95% 上界 `< 0`；
4. 固定报警预算下，long/short 尾部召回均不低于 `F0`；
5. 在有足够支持度的 `early/active/near_expiry` 桶中，不出现预注册的实质性退化；
6. 结果不是由单一品种或单一 fold 驱动。

任一条件失败，V1 结论固定为 `kronos_incremental_value: false`，后续模型删除 Kronos 依赖并以 `F0` 为候选。

### P4：校准、abstain 与正式 evaluation

只对 P3 选定的唯一模型臂进行概率校准。校准器和报警阈值只读取 validation，evaluation 只执行一次。

必须输出：

- Brier、Brier skill score、ECE、log loss；
- long/short MAE、future vol ratio MAE；
- 固定报警率下的 precision/recall；
- 分 fold、品种、生命周期和 session 的结果；
- 交易日块 bootstrap 区间；
- abstain 覆盖率及原因；
- 与 `B0/B1/F0/KF-shuffle` 的配对差异。

abstain 至少覆盖：

- 生命周期元数据缺失或矛盾；
- 上下文不足或交易日形状非法；
- 非有限特征；
- 超出训练支持的生命周期范围；
- 模型或数据 fingerprint 不匹配。

### P5：冻结后 shadow

P4 通过只允许进入 forward shadow，不允许自动控制真实仓位。P5 必须使用 checkpoint 冻结之后新增、不可变、带 manifest 的数据，验证漂移、校准、生命周期稳定性和告警预算。

生产资格至少还需要独立的执行/仓位 overlay 方案；它不属于本 V1 模型实施范围。

## 9. 日期和 Transformer 的验证方式

不得通过篡改冻结 Kronos 的 timestamp 来判断日期是否有用，因为把 timestamp 清零或乱序本身就是对预训练模型制造 OOD 输入。

日期增量只在 `FuturesStateEncoder` 内做训练对照：

```text
T0: 日内位置 + session + elapsed gap
T1: T0 + weekday/holiday
T2: T1 + day/month 周期特征
T3: T1 + contract lifecycle
```

该消融必须在 P2、evaluation 解锁前完成并冻结。预期决策原则不是“日期变化频率”，而是 validation 上是否存在稳定增量；绝对日期永不进入模型。

V1 不检验“AR 能否生成合法期货路径”，而检验冻结 AR hidden 是否给直接风险头提供增量信息。这避免把生成能力和风险判别能力混成同一个门禁。

## 10. 预期代码与产物结构

计划结构：

```text
futures_risk/
  README.md
  configs/
    risk_v1.yaml
  data/
    contract_metadata/
    snapshots/
  runs/
    v1/
  v1/
    IMPLEMENTATION_PLAN.md
    __init__.py
    config.py
    contracts.py
    data.py
    audit.py
    labels.py
    splits.py
    kronos_encoder.py
    futures_features.py
    futures_encoder.py
    fusion_head.py
    baselines.py
    training.py
    calibration.py
    evaluation.py
    plotting.py
    provenance.py
    experiment.py

tests/
  test_futures_risk_v1_contracts.py
  test_futures_risk_v1_data.py
  test_futures_risk_v1_kronos_interface.py
  test_futures_risk_v1_features.py
  test_futures_risk_v1_models.py
  test_futures_risk_v1_gates.py
```

CLI 目标接口：

```bash
python -m futures_risk.v1.experiment audit --config futures_risk/configs/risk_v1.yaml
python -m futures_risk.v1.experiment cache-kronos --config futures_risk/configs/risk_v1.yaml
python -m futures_risk.v1.experiment train-baselines --config futures_risk/configs/risk_v1.yaml
python -m futures_risk.v1.experiment train-candidates --config futures_risk/configs/risk_v1.yaml
python -m futures_risk.v1.experiment evaluate --config futures_risk/configs/risk_v1.yaml
python -m futures_risk.v1.experiment report --config futures_risk/configs/risk_v1.yaml
```

每条命令必须读取上游持久化 gate，不能只靠调用顺序假设 phase 已通过。

## 11. 必需测试与图表

### 11.1 测试

最低测试集合：

- Kronos 输入列、顺序、时间字段和 hidden 提取的回归测试；
- 确认冻结模块无可训练参数、无梯度和无权重变化；
- 5/7 根时 K 的日内分组与 mask；
- 夜盘交易日映射和真实 elapsed gap；
- 生命周期元数据因果性及最后交易日边界；
- fit-only scaler/threshold/calibrator；
- `F0/KF/KF-shuffle` 参数量和训练预算匹配；
- shuffle 确定性、split 内部性和标签独立性；
- evaluation 不参与选择；
- gate 失败后下游命令拒绝运行；
- provenance、SHA-256 和原子写入/恢复。

### 11.2 图表

正式报告至少包含：

- long/short reliability diagram；
- 各模型臂 Brier 与置信区间；
- `KF - F0` 的 fold/品种/lifecycle forest plot；
- 固定报警预算下的 precision-recall 对照；
- lifecycle 桶内风险预测与真实事件率；
- abstain 覆盖率；
- 训练曲线和 seed 稳定性。

没有图表的 evaluation 报告视为不完整。

## 12. 失败分支与最终结论模板

```text
P0 fails:
  数据不能支持生命周期感知研究；停止，不训练。

P0 passes, F0 fails:
  当前期货原生风险信号不足；停止，不使用 Kronos。

F0 passes, KF fails incremental gate:
  Kronos 对该期货风控任务无可证实增量；保留 F0，删除 Kronos 依赖。

KF passes incremental gate, P4 fails calibration:
  有预测增量但概率不可用于风险预算；只保留研究结论。

KF and P4 pass:
  冻结候选进入 forward shadow；仍非生产模型。
```

严禁把“模型成功训练”“loss 下降”“某个 fold 更好”或“Kronos 曾用期货预训练”写成有用性结论。

## 13. Definition of Done

V1 只有在以下条件全部满足时才算实施完成：

1. P0-P4 的代码、测试、配置和持久化 gate 完整；
2. Kronos 原接口与 checkpoint 全程未变；
3. `F0/KF/KF-shuffle` 是匹配对照；
4. 正式 evaluation 只运行一次且 provenance 可复现；
5. 必需指标、bootstrap、生命周期分桶和图表齐全；
6. 结论明确区分“期货模型有用”与“Kronos 有增量”；
7. 失败结论也被保存，且未通过事后调参覆盖；
8. README 明确模型只输出风险度，不产生开仓方向。

在 Definition of Done 前，统一标记：

```text
research_only: true
production_eligible: false
kronos_incremental_value: unknown
```
