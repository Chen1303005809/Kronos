# V5 目标单流方向引导路径实施计划

> 状态：P0/P1 已实施，等待 CUDA 正式运行及 P1 gate 审阅；P2–P4 按门禁阻断  
> 日期：2026-08-12  
> 方案版本：V5  
> 结果范围：`retrospective_observed_contracts`  
> 生产资格：`production_eligible: false`

## 1. 决策摘要

V4 在 P1 已经完成预注册的 `shared/per-product × target-only/pair` 消融。两种粒度的邻居增量均未通过门禁，因此：

- 冻结 V4，不实施或运行 V4 P2/P3。
- V5 不再输入邻居合约，不保留期限结构状态，也不重新搜索邻居数量或邻居选择规则。
- V5 使用目标合约自己的 256 根小时 K 线，继续预测未来三个完整交易日的完整 OHLCVA 路径。
- 先在不要求存在邻居的全量 `target_cases` 上复验方向信号。
- 方向信号通过后，先做不训练生成器的“同一采样路径库方向重加权”桥接实验。
- 只有重加权能改善由路径本身导出的 Day3 方向、且不破坏路径质量，才训练方向条件路径适配器。

V5 的核心问题不是“分类头能不能预测涨跌”，而是：

> 只使用目标合约历史得到的方向概率，能否稳定改善目标合约的完整三日生成路径？

阶段顺序固定为：

```text
P0 全量目标案例审计与路径基线
  ↓
P1 全量 target-only 方向信号复验
  ↓ gate
P2 同路径库方向概率重加权
  ↓ gate
P3 target-only 方向条件路径适配器
  ↓ gate
P4 三 seed 稳定性与冻结
  ↓
新增数据上的前瞻 shadow evaluation
```

任一 gate 未通过即停止后续阶段。不得把 P1 分类指标当成路径生成结果，也不得绕过 P2 直接训练 P3。

## 2. 为什么升级为 V5

V4 的正式 gate 文件为 [`results/observed_contract_cohort_v4/p1_granularity_gate.json`](results/observed_contract_cohort_v4/p1_granularity_gate.json)。主要结果如下：

| 粒度 | target-only ensemble BA | pair ensemble BA | pair − target-only | 5 日 P(改善>0) | 10 日 P(改善>0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared | 57.50% | 52.57% | -4.92pp | 10.20% | 7.30% |
| per-product | 55.51% | 53.61% | -1.89pp | 24.35% | 25.85% |

两种邻居方案都没有稳定增量，V4 的 `allows_p2` 为 `false`。这否证了当前数据、最近月份邻居和现有融合表示组合下的邻居假设。

同时，V4 留下了值得单独复验的 target-only 信号：

- 在相同的 1,494 个有效方向案例上，shared target-only ensemble BA 为 57.50%。
- 同 case 的 V4 P0 zero-shot/majority BA 为 47.99%，差值为 +9.50pp。
- 5 日和 10 日 moving-block bootstrap 的 `P(改善 > 0)` 分别为 99.0% 和 99.5%。
- shared target-only 比 per-product target-only 高 1.99pp，但两者差异的 5/10 日 bootstrap 概率仅约 69.1%/67.0%，置信区间跨 0，不能据此认定 per-product 的额外复杂度有价值。

以上 target-only 对比是 V4 结果出来后的诊断，只覆盖“存在邻居”的配对案例子集，不是预注册的 V5 结论。V5 P1 必须在所有合格目标案例上重新运行并重新过 gate。

V2 还提供了一个明确的负面边界：简单加入 `lambda_dir=0.2` 的方向辅助 loss 后，生成路径 Day3 BA 相对 CE-only 下降 0.52pp，路径相关性和 DTW 也没有改善。因此 V5 不重复“CE + 方向辅助头”路线，而是先验证生成分布是否具备可被方向信号利用的路径支持。

## 3. 固定任务与研究边界

### 3.1 预测任务

- 输入：一个具体交割月目标合约在预测原点前的 256 根小时 K 线。
- 特征：`open/high/low/close/volume/amount` 六维，不增加第七个 tokenizer 字段。
- 输出：紧随其后的三个完整期货交易日，长度为 `15/17/19/21` 根的完整 OHLCVA 路径。
- 主方向：第三个目标交易日最后一根 close 相对预测原点 close 的符号。
- 主指标：由最终生成路径计算的 Day3 balanced accuracy。
- 辅助路径指标：Day3 return MAE、return-path correlation、z-normalized DTW。

### 3.2 数据范围

V5 继续从当前冻结的具体合约数据中构造案例，但不再要求目标案例具有邻居：

- 使用 V4 `lookback=256` 的 `target_cases`，禁止使用 `pair_cases` 过滤。
- V4 审计中共有 3,071 个 target cases；五个 evaluation fold 合计预期 2,123 个 target cases。V5 audit 必须重新计算并固化实际数量。
- 主要门禁品种固定为 `i/jm/rb`。
- `j` 只作为 shared 模型的未见品种迁移报告：从 probe/adapter 的 fit、inner validation、checkpoint 选择和全部 gate 中排除，仅在该 fold 确有合格案例时评估，不伪造独立支持。
- `j` 的预测记录与主要品种隔离保存，不纳入 pooled BA/AUC/Brier、bootstrap 或任何基线/候选选择。
- 数据仍来自当前可观测的合约集合，存在 survivor/cohort selection 限制，因此结果范围改记为 `retrospective_observed_contracts`，不能描述为历史生产面板或可交易结论。

### 3.3 固定不再搜索的变量

- lookback 固定为 256，不重新比较 512。
- 模型粒度固定为 shared，不再运行 shared/per-product 大消融。
- shared 训练仅使用 `i/jm/rb`，并继续使用 `prediction_day_product_uniform`，使每个预测日×主要品种具有相同采样质量。
- Tokenizer 与 Kronos revision 沿用 V4 的固定版本。
- P0/P1 不解冻 Kronos 主干。
- 正式阶段使用固定 seeds `42/43/44`；smoke 结果不参与 gate。
- 不根据 evaluation 调整分类阈值，P1 主结果固定使用 `P(up) >= 0.5`。

shared 是复杂度默认选择，不是已经证明绝对更优。V5 必须继续按品种报告结果；任一主要品种触发退化保护时停止，而不是临时切回独立模型。

## 4. 数据切分与防泄漏

继续使用 V4 的五折 expanding walk-forward：

```text
minimum_fit_days      = 60
inner_validation_days = 20
evaluation_days       = 20
step_days             = 20
purge_days            = 3
fold_count            = 5
```

固定约束：

- 同一预测日的全部合约和品种必须进入同一 split/fold。
- 三日标签必须完整落在一个区间内。
- fit、inner validation 和 evaluation 之间至少 purge 三个交易日。
- 所有模型、阈值、P3a/P3b 选择只看 fit 与 inner validation。
- evaluation 每个 fold、每个正式候选只执行一次。
- 同一天的多个具体合约共享市场冲击，bootstrap 继续以交易日块为重采样单位。
- 归一化统计量只使用目标合约自己的历史 context，不使用目标区间或其他合约。
- 每个阶段必须记录 cohort/data fingerprint、case keys 和上一阶段 gate hash，防止阶段间案例静默变化。

## 5. Phase 0：全量目标案例与路径基线

### 5.1 数据审计

P0 首先生成 V5 独立审计，至少记录：

- 全量 target case 数及 fit/validation/evaluation 分布。
- fold、品种、合约、目标长度和正负/零方向覆盖。
- 相比 V4 pair-only evaluation 新增加的案例数量与原因。
- snapshot manifest、原始 payload、模型和 tokenizer revision 的 SHA/revision。
- 所有被拒绝案例及拒绝原因。

若代码路径仍隐式检查 `has_pair`、`nearest_neighbor_id` 或期限结构状态，P0 必须失败。

### 5.2 固定方向基线

在完全相同的 evaluation case keys 上保留四类预声明基线：

1. `zero_shot_mean_path`：Kronos 原始采样路径均值的 Day3 方向。
2. `zero_shot_sample_vote`：同一 sample bank 中 Day3 上涨路径的比例，以 0.5 为阈值。
3. `fit_product_majority`：只由当前 fold fit 期各品种方向比例得到。
4. `context_3day_momentum`：目标 context 内最近三个完整交易日端点收益的方向。

每个 fold 只根据 inner validation 选择一个 `selected_direction_baseline`，并在查看 evaluation 前写入 `p0_baseline_selection.json`。evaluation 同时报告全部基线，但 P1 的主增益只与预先选定基线比较。

### 5.3 固定路径采样库

正式 P0 对 inner validation 和 evaluation 各生成一次可复用的 zero-shot 路径库：

```text
sample_count = 32
temperature  = 1.0
top_k        = 0
top_p        = 0.9
random_seed  = 20260812
```

- smoke 可以使用较少样本，但不得生成 gate。
- P0 与 P2 必须读取同一份 sample bank，不能重新采样。
- 保存每条原始路径，而不只保存均值。
- 记录每个案例的上涨、下跌和零收益路径数量。
- 额外计算“真实方向在 sample bank 中是否存在”的 oracle support，仅作生成分布诊断，严禁用于候选路径选择。

每个 fold 只生成两张路径图：actual/zero-shot 的 Day3 收盘价涨跌幅对比，及 actual/zero-shot 收盘价路径对比。分类概率折线图不生成。

## 6. Phase 1：全量 shared target-only Probe

### 6.1 目的

回答 V4 的 target-only 信号能否扩展到不要求邻居存在的全部目标案例。P1 仍是方向信息探针，不是最终路径模型。

### 6.2 模型与训练

- 冻结 tokenizer 与 Kronos 主干。
- 实现纯 target-only dataset，不加载、编码或校验邻居。
- Probe 头与 V4 `target_only_probe` 保持数值等价的结构和初始化语义，以便把主要变化限制在案例覆盖。
- 新实现必须提供回归测试：在同一 paired case 和相同 head 权重下，V5 target-only logits 与 V4 target-only logits 数值一致。
- 固定 `fusion_hidden_dim=256`、`dropout=0`、LR `3e-4`、batch size `64`、最多 30 epochs、patience `5`、weight decay `0.01`、gradient clip `3.0`。
- seeds 固定为 `42/43/44`。
- 每个 seed 独立按 inner validation BA 选择 checkpoint；evaluation 只运行一次。
- 三 seed 的概率做等权 ensemble，保留每个 seed 的原始预测。

### 6.3 P1 gate

P1 必须同时满足：

- 三个 seed 相对 `selected_direction_baseline` 的 BA 增益中位数至少 `+2pp`。
- 至少 `2/3` seed 的增益为正。
- ensemble 至少 `3/5` fold 改善。
- `i/jm/rb` 任一品种的 ensemble BA 退化不超过 `1pp`。
- 5 日和 10 日 paired moving-block bootstrap 的 `P(improvement > 0)` 均至少 `80%`。
- ensemble pooled ROC-AUC 至少 `0.55`。
- ensemble pooled Brier score 不高于 `0.25`，且不高于只使用 fit-period 品种上涨率的概率基线。

`j` 单独报告 BA/AUC/Brier 和案例覆盖，不参与 gate。

P1 未通过时停止 V5：保留 target-only probe 作为研究诊断，不运行 P2/P3，也不通过增加 epoch、seed、阈值搜索或 per-product 模型挽救结果。

## 7. Phase 2：同路径库方向概率重加权

### 7.1 目的

在不训练或修改生成器的前提下，验证 P1 的方向概率能否从 Kronos 已经生成的路径分布中选出更好的完整路径。这是分类与路径生成之间的桥接实验。

### 7.2 固定重加权规则

对每个案例，令三 seed ensemble 的方向概率为 `q = P(up)`。把同一 P0 sample bank 中的非零 Day3 路径分为上涨集合 `U` 和下跌集合 `D`：

```text
如果 U、D 均非空：
    每条上涨路径权重 = q / |U|
    每条下跌路径权重 = (1 - q) / |D|
    零收益路径权重   = 0
否则：
    回退到全部 32 条路径的普通均值
```

最终 `probe_weighted_path` 是各时间步、各 OHLCVA 特征的凸组合。候选与基线使用完全相同的原始路径，唯一变量是权重；最终方向必须从加权后路径的 Day3 close 重新计算，禁止直接复制 probe 分类结果。

不搜索温度、阈值、top-p、sample count 或多种重排公式。若固定公式失败，P2 失败。

### 7.3 支持度检查

在评估质量前，sample bank 必须满足：

- 同时存在上涨与下跌路径、因而可以实际重加权的 pooled 双侧支持率至少 `90%`。
- `i/jm/rb` 每个品种的双侧支持率至少 `80%`。
- probe 预测方向在路径库中的 pooled 可用率至少 `95%`。
- 回退比例、按 fold/品种的单双方向路径分布必须完整报告。

现有 V4 `K=10` 配对子集诊断中，probe 预测方向可用率约为 87.2%；因此 V5 正式固定 `K=32`，但该覆盖率仍必须由 P0 在全量案例上重新确认。

### 7.4 P2 gate

相对同 sample bank 的 `zero_shot_mean_path`，必须同时满足：

- Day3 path BA 至少提高 `2pp`。
- 至少 `3/5` fold 改善。
- 任一主要品种 Day3 BA 退化不超过 `1pp`。
- 5 日和 10 日 paired block bootstrap 均满足 `P(improvement > 0) >= 80%`。
- Day3 return MAE 恶化不超过 `2%`。
- pooled mean return-path correlation 的绝对下降不超过 `0.01`。
- 任一主要品种 mean return-path correlation 的绝对下降不超过 `0.02`。
- pooled mean z-normalized DTW 恶化不超过 `2%`。

P2 通过只说明方向信号可以利用当前生成分布，不代表学习型路径模型已经成功。P2 未通过时停止，不训练 P3。

每个 fold 只生成 actual/probe-weighted/zero-shot-mean 的 Day3 收盘价涨跌幅对比与 close 路径图，并在图中标记 fallback 案例；不生成分类概率图。

## 8. Phase 3：方向条件目标路径适配器

### 8.1 固定模型结构

P3 只使用目标 context 和冻结的 P1 ensemble 输出：

1. 在预测原点计算一次 `q=P(up)`，并在完整自回归生成期间复用。
2. 构造条件向量：

   ```text
   [1, logit(clip(q, 0.01, 0.99)), 2 * abs(q - 0.5)]
   ```

3. 首轮只使用一种残差适配器：`Linear(3, 128) -> SiLU -> Linear(128, d_model)`，不另做 FiLM/门控分支。
4. 最后一个 `Linear` 的 weight 和 bias 均零初始化；其余层使用 PyTorch 默认初始化，dropout 固定为 `0`。
5. MLP 输出广播到全序列，在目标 token embedding 加入 temporal embedding 后、进入第一个 Transformer block 前做加法注入。
6. 零初始化时的 logits 必须与未条件化 Kronos 数值一致。
7. condition 只能读取预测原点前的目标 context；teacher-forcing 的未来 token 不得进入 probe 或 condition encoder。

matched control 使用完全相同的模型、参数量、初始化、batch 顺序、优化器和随机 seed，但每个案例固定令 `q=0.5`，即条件向量为 `[1,0,0]`。这样 control 可以学习全局适配，而候选额外获得逐案例方向信息。

### 8.2 训练协议

- 只使用预测原点对齐、未来为三个完整交易日的 target cases，不把 probe 外推到任意小时起点的 dense window。
- adapter 训练与 inner-validation 选择只使用 `i/jm/rb`；`j` 仅在正式 evaluation 中作额外迁移报告。
- tokenizer 始终冻结。
- P1 probe checkpoint 始终冻结，方向概率可以预计算缓存。
- P3 训练所用的方向概率必须按时间做 cross-fitting：对 outer-fold fit 区间使用 `20` 交易日 warm-up，之后每 `20` 交易日产生一个 expanding 预测块，块前保留 `3` 交易日 purge。每个适配器训练案例的 `q` 必须来自一个训练截止时间早于该案例、且从未见过该案例标签的 probe。
- outer-fold fit 的前 `23` 交易日（warm-up + purge）从 P3 adapter fit 中排除并记录，禁止使用 in-sample probe 概率补齐。V5 audit 必须在训练前按 outer fold 固化可用的 cross-fitted 天数和案例数；任一 fold 少于 `20` 个可用交易日或 `100` 个主要品种案例时，P3 不可训练。
- outer inner validation 和 evaluation 使用固定的 condition probe。该 probe 只能在 outer fit 内训练并用 outer fit 内的嵌套 validation 选择 checkpoint；不得使用 outer inner-validation/evaluation 标签选择 condition probe。
- cross-fitted cache key 必须包含 case key、probe fold/cutoff、seed、模型 revision、data fingerprint 和 normalization 版本。
- loss 固定为未来 target token CE；不加入 V2 已失败的方向辅助 loss。
- P3 首轮适配器 seed 固定为 `42`；只有该轮过 gate 后，P4 才追加 `43/44`。
- 候选与 matched control 的 optimizer 均固定为 AdamW（`betas=(0.9, 0.95)`、weight decay `0.1`），batch size `32`，最多 `15` epochs，patience `3`，gradient clip `3.0`，cosine schedule 与 `5%` warmup。
- adapter LR 固定为 `3e-4`；P3b 最后一个 Transformer block 的 LR 固定为 `3e-6`。不做 LR、hidden dim、注入位置或 epoch 搜索。
- early stopping 只使用 inner-validation 的 future-token CE；每臂独立按相同规则选 checkpoint。选定 checkpoint 后才在 inner validation 生成固定路径，用于 P3a/P3b 选择。
- inner-validation 和 evaluation 生成都固定 `sample_count=32`、`temperature=1.0`、`top_k=0`、`top_p=0.9`；分别使用 `20260813` 和 `20260814` 作为起始 seed。每个 case×sample 的 seed 由 case key 确定性派生，候选/control 使用共同随机数。
- `P3a`：冻结 Kronos 主干，只训练条件适配器。
- 只有 inner validation 上候选相对 matched control 的路径 BA 至少 `+1pp`、MAE 恶化不超过 `2%`、DTW 恶化不超过 `2%`，才允许产生 `P3b` 候选。
- `P3b`：候选与 control 都只解冻最后一个 Transformer block；两臂从相同的基础权重和相同零初始化 adapter 重新开始，主干 LR 与 adapter LR 分组配置，不解冻更多层。
- P3a/P3b 以严格配对的“候选/control”组合参与选择，不能拿 P3b 候选与 P3a control 比较。只根据 inner validation 的固定顺序选择组合：先最大化 candidate − control 的 Day3 path BA，再最小化候选相对 control 的 MAE 恶化，最后最小化 DTW 恶化。
- 正式 evaluation 执行一次，并保留逐样本路径。

每个 fold 在 evaluation 前，必须根据 inner validation 在以下路径基线中选择并持久化一个 `selected_path_baseline`：

- `zero_shot_mean_path`
- `probe_weighted_path`
- `matched_q_0.5_control`

不得根据 evaluation 结果切换基线。

### 8.3 P3 gate

相对 `selected_path_baseline` 必须同时满足：

- Day3 path BA 至少提高 `2pp`。
- Day3 return MAE 恶化不超过 `2%`。
- pooled mean return-path correlation 至少 `0.10`，且 `i/jm/rb` 均为正。
- mean z-normalized DTW 至少改善 `5%`。
- 任一主要品种 Day3 BA 退化不超过 `1pp`。
- 至少 `3/5` fold 改善。
- 5 日和 10 日 paired block bootstrap 均满足 `P(improvement > 0) >= 80%`。

此外，为隔离逐案例方向条件本身的贡献，`direction_conditioned_path` 相对同结构的 `matched_q_0.5_control` 还必须满足：

- Day3 path BA 至少提高 `1pp`。
- 至少 `3/5` fold 改善。
- 任一主要品种退化不超过 `1pp`。
- 5 日和 10 日 paired block bootstrap 均满足 `P(improvement > 0) >= 80%`。

只有 P3 全部通过，才把 V5 记为“target-only 方向信号成功迁移到路径生成”。只提升 probe 或只提升路径方向但破坏 MAE/相关性/DTW，均不能得到该结论。

## 9. Phase 4：多 seed 稳定性与冻结

P3 通过后运行 seeds `42/43/44`：

- 保存每个 seed 的 adapter/最后 block checkpoint、原始路径、逐案例记录和逐 fold 图。
- 报告单 seed、三 seed 概率/路径 ensemble，不得只保留 ensemble。
- 三个 seed 相对固定基线至少 `2/3` 为正，且任一 seed 不得低于基线超过 `1pp`。
- seed 间 Day3 path BA 标准差不超过 `2pp`。
- ensemble 仍须满足 P3 的全部路径 gate。

通过后冻结：

- 模型、probe、adapter checkpoint。
- 分类阈值与 sample 参数。
- 数据规则、归一化和合约范围。
- 评估代码 revision 与图表契约。

当前回看数据即使全部通过，仍保持 `production_eligible: false`。真正前瞻结论需要冻结后等待新增、未参与任何选择的数据；建议先做 20 个新增交易日的 shadow checkpoint，再以至少 60 个新增交易日作为首个正式前瞻窗口。

## 10. 固定输出结构

计划新增：

```text
csj/configs/target_only_path_v5.yaml

csj/v5/
  config.py
  target_probe.py
  path_bridge.py
  path_adapter.py
  experiment.py

csj/scripts/run_v5_cuda.sh
```

运行产物：

```text
csj/runs/target_only_path_v5/<run_id>/
  resolved_config.json
  data_audit.json
  p0/
  p1_signal/
  p2_path_bridge/
  p3_adapter/
  p4_stability/
```

汇总产物：

```text
csj/results/target_only_path_v5/
  v5_data_audit.json
  p0_baselines.json
  p1_signal_metrics.json
  p1_signal_gate.json
  p2_path_bridge_metrics.json
  p2_path_bridge_gate.json
  p3_adapter_metrics.json
  p3_adapter_gate.json
  p4_stability_metrics.json
  REPORT.md
```

每份配置、run、metrics 和 gate 必须包含：

```text
strategy_version: 5
phase
result_scope: retrospective_observed_contracts
production_eligible: false
data_fingerprint
evaluation_contract_version
model/tokenizer revisions
upstream_gate_path
upstream_gate_sha256
```

## 11. 测试与验收

### 11.1 数据与 split

- 无邻居的 target case 能进入 V5。
- 修改/删除邻居数据不改变 V5 case、输入或输出。
- 同一预测日所有案例进入同一 split。
- purge 不少于三交易日，标签不跨边界。
- 目标长度与三个 day-end index 正确。
- 修改预测原点后的 K 线不改变 context、probe probability 或 condition。

### 11.2 Probe

- V5 target-only 与 V4 target-only 在相同 paired case、相同权重下 logits 数值一致。
- target-only 路径从不调用邻居 tokenizer/encoder。
- prediction-day×product 采样质量相同。
- 三 seed checkpoint 只由 inner validation 选择。
- evaluation 阈值固定 0.5，不能由 evaluation 反推。

### 11.3 Path bridge

- baseline 与候选读取完全相同的 sample bank 和 case keys。
- 上下两类路径均存在时，权重和严格为 1，组内权重相等。
- 缺一类方向时确定性回退，不访问真实标签。
- oracle support 不可进入候选路径计算。
- 凸组合后 OHLC 关系、非负 volume/amount 和路径长度保持有效。

### 11.4 Adapter

- 零初始化 adapter 时与基础 Kronos logits 数值一致。
- `q=0.5` control 与候选结构/初始化相同。
- condition 仅由 context 产生，修改未来 target 不改变 condition。
- 自回归每一步复用同一 condition。
- P3a 只更新 adapter；P3b 只额外更新最后一个 Transformer block。
- 方向辅助 loss 始终不存在。
- adapter fit 使用的每个 q 都有可审计的时间 cross-fit 来源；in-sample q 必须使测试失败。
- cross-fit 固定为 20 日 warm-up、20 日预测块和 3 日 purge；不足最低日数/案例数时拒绝训练。

### 11.5 评估与可视化

- 每个 fold 必须生成非空的 Day3 收盘价涨跌幅图和 close 路径图。
- actual/candidate/baseline case keys 完全一致。
- 图表按品种分面并标明样本数、BA、accuracy、fallback/support 状态。
- JSON 同时包含方向、MAE、相关性、DTW、fold/product/seed 分层和 bootstrap。
- 任一必需图表生成失败必须使阶段失败。

## 12. 运行入口与执行顺序

计划入口：

```bash
.venv/bin/python -m csj.v5.experiment audit
.venv/bin/python -m csj.v5.experiment p0
.venv/bin/python -m csj.v5.experiment p1-signal
.venv/bin/python -m csj.v5.experiment p2-path-bridge
.venv/bin/python -m csj.v5.experiment p3-adapter
.venv/bin/python -m csj.v5.experiment p4-stability
```

CUDA 包装器计划对应：

```bash
RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh check
RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh p0-smoke
RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh p0
RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh p1
RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh full
```

同步正式 P1 结果并人工核对 gate 后，才实现 P2；同步并确认 P2 gate 后，才实现 P3。runner 必须从持久化 gate 文件读取 `allows_next_phase`，不能使用命令行开关绕过。

## 13. 明确不做与停止条件

- 不实施 V4 邻居 P2，也不把邻居字段留作隐藏输入。
- 不重新搜索 shared/per-product、256/512 或邻居数量。
- 不使用连续合约、主力拼接或换月序列。
- 不加入 OI、技术指标、宏观、新闻或跨品种信息。
- 不搜索 evaluation 阈值、采样温度、top-p 或多种路径重排公式。
- 不重复 V2 的方向辅助 loss。
- 不在一个阶段同时扩大 adapter、解冻多层和改变 loss。
- P1 失败时不通过更多 seed/epoch 挽救；P2 失败时不训练 P3；P3 失败时不运行 P4。
- 不把 retrospective observed contracts 的结果写成生产、交易或正式历史活跃合约证据。

## 14. 建议提交边界

1. `docs(v5): freeze target-only path strategy after v4 gate failure`
2. `feat(v5): add target-only audit and full-case baselines`
3. `feat(v5): add full-coverage shared target-only probe gate`
4. 同步 CUDA P1 结果并记录 gate；未通过则停止。
5. `feat(v5): add fixed-bank direction reweighting bridge`
6. 同步 CUDA P2 结果并记录 gate；未通过则停止。
7. `feat(v5): add direction-conditioned target path adapter`
8. P3 通过后再增加多 seed 稳定性与冻结流程。

任何阶段的耗时 checkpoint、原始 sample bank 和逐案例记录都必须同步完整 run 目录，不能只复制 results 汇总 JSON。
