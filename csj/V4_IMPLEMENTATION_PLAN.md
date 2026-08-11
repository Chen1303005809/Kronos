# V4 可观测合约 Cohort 与逐 Fold 涨跌图实施计划

> 状态：方案已确认，尚未实施  
> 日期：2026-08-11  
> 结果范围：`retrospective_observed_cohort`，仅作研究证据，不作生产有效性或可交易性声明

## 1. 交接摘要

V3 冻结在 P1，不进入 V3 P2。最新 CUDA P1 使用预测日均匀采样后，`pair_probe` 相对 `target_only_probe` 的 Day3 balanced accuracy 提升 `8.93pp`，5 日与 10 日 block bootstrap 的 `P(improvement > 0)` 分别为 `99.2%` 和 `99.25%`。但是全部配对案例都来自单个事后快照：

- `strict_pair_cases = 0`
- `partial_pair_cases = 1590`
- `formal_p1_to_p2_eligible = false`
- `passes_p1_to_p2_gate = false`

因此 V3 只能保留“邻居可能包含增量信息”的探索性结论。V4 的核心变更是放弃“重建历史活跃合约面板”的声明，改为在当前冻结数据 cohort 中，仅根据预测时刻已经存在且上下文完整的数据选择邻居。

实施前必须阅读仓库根目录 `AGENTS.md`。Python 默认使用 `.venv/bin/python`，版本为 Python 3.12；训练入口以 CUDA 为主。

## 2. 已确认的产品与实验决策

### 2.1 逐 fold 图表

每个 fold 完成最终 evaluation 后，必须立即生成一张预测与实际涨跌的对比图：

- 一个 fold 一张 PNG；2×2 消融阶段仍保持一张 PNG，在图内分区展示两种粒度。
- 图内按品种分面。
- 每个比较区固定展示三条轨道：真实方向、候选模型、严格配对基线。
- 红色表示上涨，绿色表示下跌，灰色表示零收益或无效方向。
- 错误预测需有醒目标记。
- 案例按 `target_end_day`、`target_contract_id`、`case_key` 排序。
- 每个品种标题显示样本数、balanced accuracy 和 accuracy。
- 图表生成失败必须使当前 fold/阶段失败，不能只保留 checkpoint 或 JSON。

现有聚合柱状图不再是必备图。聚合指标继续保留在 JSON 中，新的强制可视化以逐案例的 fold 涨跌对比图为准。

### 2.2 V4 数据定位

V4 使用“现有数据研究版”：

- 数据范围明确标记为 `retrospective_observed_cohort`。
- 不使用快照时间判断历史时刻的活跃面板成员。
- 不把当前 cohort 上的历史回看结果描述成正式历史活跃面板结论。
- 所有报告固定写入 `production_eligible: false`。

### 2.3 共享训练还是独立训练

历史结果不能直接裁决模型粒度：

| 版本 | 实际训练粒度 | 可比结果 |
| --- | --- | --- |
| V1 | `i8888 + rb8888` 共享训练 | 封存测试 pooled BA 相对 zero-shot 为 `0.00pp`；i `-1.16pp`，rb `+1.35pp` |
| V2 | 按合约独立训练 | 三日路径 pooled BA 相对 zero-shot `-1.53pp`；5 个 fold 仅 1 个改善 |
| V3 | 按品种独立 Probe | pair 相对 target-only `+8.93pp`，但 target-only BA 仅 `44.81%` |

因此 V4 Phase 1 不预设共享或独立，而是运行严格的：

```text
共享 / 独立 × target-only / pair
```

2×2 消融。选择粒度后才允许进入完整路径生成。

## 3. 固定绘图工具改造

### 3.1 公共接口

将跨版本绘图入口放到 `csj/evaluation_plotter.py`。V3 保留兼容包装，V4 不依赖 `csj.v3.evaluation_plotter`。

新增统一接口，语义如下：

```python
render_fold_direction_comparison(
    records_by_model,
    *,
    fold_id,
    candidate_model,
    baseline_model,
    output_dir,
    stage,
    metadata,
) -> EvaluationArtifacts
```

标准化记录至少包含：

```text
case_key
fold_id
product
target_end_day
target_contract_id
actual_direction
predicted_direction
probability_up
model
```

绘图器必须验证：

- 候选与基线的 case key 完全一致。
- 同一 case 的日期、品种、合约和真实方向一致。
- 不存在重复 case key。
- fold 不为空，且至少包含一个有效方向案例。
- 所有必需图表和 JSON 已成功落盘且非空。

### 3.2 固定产物

每个 fold 输出：

```text
evaluation/fold_XX/prediction_vs_actual.png
evaluation/fold_XX/prediction_vs_actual.json
```

阶段总报告记录全部 fold 的产物路径、绘图契约版本、模型名称和案例覆盖。P0、P1、P2 的固定对照分别为：

| 阶段 | 候选 | 基线 |
| --- | --- | --- |
| P0 | zero-shot | majority |
| P1 | pair probe | target-only probe |
| P2 | pair-conditioned path | matched target-only path |

P1 2×2 消融图在同一 PNG 内并排显示 shared 与 per-product 两个比较区，每个区仍只包含 actual/candidate/baseline 三条轨道。

### 3.3 V3 图表回填

绘图器完成后，先读取以下现有记录回填 5 个 fold 图，不重新训练：

```text
csj/runs/active_contract_panel_v3_partial/
  v3_partial_day_balanced_cuda/p1/fold_*/**/*_records.json
```

回填只增加新的 evaluation 产物，不修改旧指标、预测记录或 checkpoint。

## 4. V4 数据构建

### 4.1 Cohort 冻结

使用当前 snapshot manifest 中成功加载的具体合约构建冻结 cohort，并记录：

- manifest SHA-256。
- 每个原始 payload SHA-256。
- 模型与 tokenizer 的 id/revision。
- 数据规则版本、lookback、clip 和 normalization epsilon。
- 合约成功/失败状态。

失败合约不进入候选集，但必须保留在审计报告中。

### 4.2 邻居选择规则

对每个目标案例，在预测原点执行以下顺序：

1. 从冻结 cohort 中取同品种且非目标自身的合约。
2. 只保留在预测原点拥有完整 `lookback` 上下文的合约。
3. 邻居上下文必须与目标上下文时间戳逐根一致。
4. 完整交易日只能包含允许的 5/7 根；不填充、不截断、不使用未来数据。
5. 最近五个完整交易日必须可计算成交量比状态。
6. 先完成上述可用性过滤，再按 `abs(delta_month)` 选择最近邻。
7. 等距时优先选择交割月份更晚的一侧，再以合约代码作确定性 tie-break。

这与 V3 的关键区别是：V3 先选名义最近邻，若其上下文不可用就丢弃案例；V4 从预测时刻实际可用的邻居中选择最近者。

每个案例额外记录：

```text
candidate_count
selected_neighbor_id
signed_month_distance
selection_rule_version
cohort_fingerprint
neighbor_context_available_at_origin
```

### 4.3 当前数据的只读审计基准

已使用现有数据进行只读可行性审计，预期结果如下：

| Lookback | Target cases | Pair cases | Later | Earlier |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 3071 | 2269 | 1603 | 666 |
| 512 | 2057 | 1438 | 958 | 480 |

512 相对 256 的 pair 保留率约为 `63.4%`，低于既定 80% 门槛，因此 V4 固定 `lookback = 256`。

256 长度下各 evaluation fold 的预期 pair 数为：

```text
fold_00 = 190
fold_01 = 265
fold_02 = 328
fold_03 = 402
fold_04 = 408
```

五个 evaluation fold 均同时具有 earlier 与 later 邻居。`j` 只在后期 fold 出现，没有足够的早期单品种训练历史，因此只参与共享模型的迁移能力辅助报告，不参与共享/独立粒度的主选择门禁。

## 5. V4 训练阶段

V4 入口计划为：

```text
python -m csj.v4.experiment audit
python -m csj.v4.experiment p0
python -m csj.v4.experiment p1-ablation
python -m csj.v4.experiment p2
python -m csj.v4.experiment p3-stability
```

CUDA shell runner统一放在 `csj/scripts/run_v4_cuda.sh`。`p2` 和 `p3-stability` 必须读取上一阶段 gate 文件；门禁未通过时直接拒绝运行。

### Phase 0：数据与基线

- 生成 cohort、拒绝原因、月份距离、品种、邻居方向和 fold 覆盖审计。
- 固定 5 个 expanding walk-forward folds。
- 保持 20 日 inner validation、20 日 final evaluation、3 日 purge。
- 在 V4 pair 共同案例上运行 zero-shot 与 majority 基线。
- 每个 fold 结束后强制生成实际/zero-shot/majority 涨跌图。

### Phase 1：共享/独立 2×2 Pair Probe

冻结 tokenizer 与 Kronos 主干。为降低重复编码开销，允许缓存目标和邻居的上下文隐藏状态；缓存键必须包含 case key、模型 revision、cohort hash、lookback 和 normalization 配置。

固定 seeds：

```text
42, 43, 44
```

四个 arm：

```text
shared_target_only
shared_pair
per_product_target_only
per_product_pair
```

约束：

- 同一粒度下 target-only 与 pair 使用相同 case、相同初始化、相同 seed 和相同训练协议。
- shared 使用预测日×品种等权采样，防止案例多的日期或品种支配训练。
- per-product 在每个品种内部使用预测日等权采样。
- 共享与独立模型使用相同 Probe 头结构、优化器和 checkpoint 选择规则。
- 模型选择只看 inner validation；evaluation 每个 fold 只执行一次。

每种粒度自身的 P1 门槛：

- 三个 seed 的中位 `pair - target-only` BA 至少 `+2pp`。
- 至少 2/3 seed 的增益为正。
- 至少 3/5 fold 改善。
- i/jm/rb 任一品种的中位退化不超过 `1pp`。
- earlier/later 任一方向的中位退化不超过 `1pp`。
- 对 seed 平均概率进行 5 日和 10 日 paired moving-block bootstrap，两者 `P(improvement > 0) >= 80%`。

粒度选择规则：

1. 只有一种粒度通过自身门槛：选择该粒度。
2. 两种均未通过：停止 V4，不实施 P2。
3. 两种均通过：在 i/jm/rb 的完全相同 evaluation case 上比较 pair 的绝对 BA。
4. 只有领先至少 `1pp`、paired bootstrap `>= 80%` 且任一品种不退化超过 `1pp` 时才判定该粒度明确胜出。
5. 差异不显著时默认选择 shared，以减少模型数量并保留 j 的迁移评估能力。
6. 若 shared 触发品种退化保护，则选择 per-product；此时 j 标记为 unsupported，不伪造独立模型。

阶段输出 `p1_granularity_gate.json`，写明 selected granularity、每个 arm 的指标、seed/fold/product/direction 分层结果及是否允许 P2。

### Phase 2：单邻居条件路径生成

仅在 Phase 1 门禁通过后实施。

模型对照：

- `matched_target_only_path`：与候选相同参数结构和训练协议，但邻居 mask 固定为 0。
- `pair_conditioned_path`：使用目标上下文、一个合格邻居和期限结构状态。

两者必须从相同基础权重、相同零初始化适配器和相同 case 开始训练。

条件结构：

1. 邻居上下文通过共享 Kronos 编码器得到最后一个有效隐藏状态。
2. 隐藏状态与 `log_close_ratio`、`signed_month_distance`、`log_volume_ratio_5d` 融合为 `neighbor_condition`。
3. 适配器产生 FiLM/残差条件，在目标 token embedding 加入 temporal embedding 后、进入 Transformer 前注入。
4. 最后一层条件投影零初始化，确保关闭条件时数值回归到 target-only。
5. 自回归生成期间，整个预测路径复用预测原点计算出的同一 `neighbor_condition`。
6. teacher forcing 的未来目标 token 不允许进入邻居条件编码器。

训练顺序：

1. P2a 冻结 tokenizer 和 Kronos 主干，只训练融合与条件适配器。
2. 若 inner validation 上 candidate BA 至少比匹配基线提高 `1pp`，且 MAE 恶化不超过 `2%`，才增加 P2b 候选：只解冻最后一个 Transformer block。
3. P2a/P2b 只根据 inner validation 选择，随后对 evaluation 执行一次最终评估。
4. 不加入方向辅助 loss，不同时搜索第二个邻居、上下文长度或新 lambda。

P2 研究门槛：

- Day3 path balanced accuracy 相对 matched target-only 至少 `+2pp`。
- Day3 return MAE 恶化不超过 `2%`。
- mean return-path correlation 至少 `0.10`，且 i/jm/rb 均为正。
- z-normalized DTW 至少改善 `5%`。
- 任一主要品种 BA 退化不超过 `1pp`。
- 至少 3/5 fold 改善。
- 5 日与 10 日 paired block bootstrap 均 `>= 80%`。

每个 fold 强制生成 actual/pair-conditioned/matched-target-only 涨跌图。

### Phase 3：多 seed 稳定性

只有 P2 通过后才运行 seeds `42/43/44`：

- 保存每个 seed 的 checkpoint、原始路径、逐案例预测和 fold 图。
- 生成 seed ensemble，但不得只保留 ensemble。
- 报告 seed 间均值、标准差、fold/product/direction 分层和 bootstrap。
- 即使所有研究门槛通过，结果仍保持 `production_eligible: false`。

## 6. 输出目录与版本记录

计划新增配置：

```text
csj/configs/observed_contract_cohort_v4.yaml
```

计划输出：

```text
csj/runs/observed_contract_cohort_v4/<run_id>/
  resolved_config.json
  data_audit.json
  p0/
  p1_ablation/
  p2/
  p3_stability/

csj/results/observed_contract_cohort_v4/
  v4_data_audit.json
  p0_metrics.json
  p1_ablation_metrics.json
  p1_granularity_gate.json
  p2_metrics.json
  p2_gate.json
  p3_stability_metrics.json
```

每份配置、run、metrics 和 report 必须写入：

```text
strategy_version: 4
phase
result_scope: retrospective_observed_cohort
production_eligible: false
cohort_fingerprint
evaluation_contract_version
model/tokenizer revisions
```

## 7. 测试与验收

### 数据测试

- 可用性过滤发生在月份排序之前。
- 名义最近邻不可用时选择下一个合格邻居。
- 等距时选择晚月。
- 跨年月份距离正确。
- 目标/邻居上下文时间戳完全一致。
- 缺根、异常交易日、五日状态不足时正确拒绝。
- 修改预测原点后的数据不改变案例或期限结构状态。
- cohort 与 payload 哈希稳定。

### Split 与采样测试

- 同一预测日的全部案例进入同一 split/fold。
- 三日标签完整落在一个区间。
- purge 不少于三个交易日。
- shared 的每个预测日×品种采样质量相同。
- per-product 的每个预测日采样质量相同。
- 四个消融 arm 的共同 evaluation case key 完全一致。

### 条件模型测试

- 零初始化适配器且 condition 关闭时，与 target-only logits 数值一致。
- neighbor mask 为 0 时不调用邻居 tokenizer/encoder。
- 修改邻居未来数据不改变条件。
- 推理每一步复用同一条件。
- 无有效邻居时路由到 target-only，不构造伪邻居。

### 绘图测试

- 每个 fold 恰好产生非空 PNG 和 JSON。
- actual/candidate/baseline 顺序固定。
- 品种分面、case 排序和颜色映射稳定。
- 零方向以灰色显示且不进入 BA。
- 缺列、重复 key、两臂 key 不一致或写图失败时阶段失败。
- 阶段总报告包含全部 fold 图路径。

### 运行验收

1. `.venv/bin/python -m pytest tests -q` 全量通过。
2. 本地 CPU smoke 完成一个最小 fold，并生成规定图表。
3. CUDA 机器先运行单 fold smoke，再运行完整 Phase 1。
4. Phase 1 结果同步回本机并人工确认 gate 后，才开始实现/运行 P2。

## 8. 明确不做

- 不把 partial/retrospective cohort 偷换成正式历史活跃面板。
- 不在 V4 Phase 1 同时改变上下文长度、Probe 结构或分类阈值。
- 不在 P1 通过前实施 P2。
- 不在首轮 P2 加第二个邻居、方向辅助 loss、OI、宏观或技术指标。
- 不将当前结果描述为可交易或生产有效。

## 9. 建议实施顺序与提交边界

1. `feat(eval): add mandatory per-fold direction comparison plots`
   - 建立共享绘图模块、更新契约与测试。
   - 回填当前 V3 五个 fold 图。
2. `feat(v4): add observed-cohort data audit and case builder`
   - 实现 cohort、邻居规则、审计和 split 测试。
3. `feat(v4): add shared-vs-per-product P1 ablation`
   - 实现特征缓存、2×2 arm、三 seed、门禁和 CUDA runner。
4. 同步 CUDA Phase 1 结果并分析。
5. 只有 gate 通过后，再单独提交 P2 条件路径实现。

## 10. 新窗口交接检查清单

新窗口开始实施时按以下顺序确认：

1. 阅读根目录 `AGENTS.md` 与本文档。
2. 检查 `git status --short`，保留用户已有改动和未跟踪 runs。
3. 确认 `.venv/bin/python --version` 为 Python 3.12。
4. 阅读现有 `csj/v3/evaluation_plotter.py`、`csj/v3/experiment.py` 和 `csj/v3/panel_data.py`，复用已有指标、防泄漏与对齐逻辑。
5. 不修改或重写已有 V3 runs/results；图表回填只能新增产物。
6. 先完成绘图工具和 V3 回填，再开始 V4 数据层。
7. V4 首轮实施止于 P1 2×2 消融；没有同步结果和 gate 结论时不得提前实现 P2。

