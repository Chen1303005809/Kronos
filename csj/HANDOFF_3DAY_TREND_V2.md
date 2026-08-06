# Kronos 小时期货趋势预测：现状与 V2 三交易日方案交接

> 更新时间：2026-08-06（Asia/Shanghai）
> 仓库：`/Users/eurus/Code/kronos/Kronos`
> 用途：清空会话历史后，用本文件恢复完整上下文并继续实施。
> 本文件把“已经完成并验证的事实”和“下一步拟实施方案”分开记录；不要把方案描述误当成已完成代码。

## 0. 清空历史后先做什么

恢复工作时按以下顺序阅读：

1. 本文件：`csj/HANDOFF_3DAY_TREND_V2.md`
2. V1 最终报告：`csj/results/futures_hourly/REPORT.md`
3. V1 运行说明：`csj/README.md`
4. V1 配置：`csj/configs/futures_hourly.yaml`

建议在新会话中直接发送下面这句话：

> 读取 `csj/HANDOFF_3DAY_TREND_V2.md`，先核对 git status、V1 报告和现有 checkpoint，不覆盖或删除 V1；然后从文档中的 V2 Phase 0 开始继续实施。

恢复后的第一组只读命令：

```bash
cd /Users/eurus/Code/kronos/Kronos
git status --short
git log -1 --oneline
.venv/bin/python -V
find csj/runs/futures_hourly/full/training -name best_model.pt -print | sort
```

重要保护事项：

- 当前 V1 代码、测试、报告都是工作区中的**未提交改动**。不要执行 `git reset --hard`、`git clean` 或覆盖式 checkout。
- `csj/runs/` 和 `csj/artifacts/` 已被 `.gitignore` 忽略，但里面有耗时数小时生成的 checkpoint、预测缓存和模型缓存；不要删除。
- V2 必须使用新的配置名、run id 和结果目录，不能覆盖 `futures_hourly/full` 或 `csj/results/futures_hourly/`。
- 项目约定 Python 3.12 虚拟环境。当前环境为 `.venv/bin/python`，已核对版本 `Python 3.12.13`。
- 模型已缓存到 `csj/artifacts/hf_cache`。当前受限执行环境不能依赖 Hugging Face 默认缓存或网络；原生 regression 要显式设置 `HF_HUB_CACHE` 和 `HF_HUB_OFFLINE=1`。
- 当前分支为 `master`，工作区基础提交为 `49251d37a31addd311b70630da89aa6205aa18e4`（`add. k线数据与部分工具函数`）。

截至本文件写入时的 `git status --short` 快照：

```text
 M .gitignore
 M csj/utils/tool.py
 M model/kronos.py
?? csj/HANDOFF_3DAY_TREND_V2.md
?? csj/README.md
?? csj/__init__.py
?? csj/config.py
?? csj/configs/
?? csj/evaluation.py
?? csj/experiment.py
?? csj/futures_data.py
?? csj/metrics.py
?? csj/reporting.py
?? csj/results/
?? csj/training.py
?? csj/utils/__init__.py
?? tests/test_futures_metrics.py
?? tests/test_futures_pipeline.py
```

以上不是待清理垃圾，而是当前 V1 实现与本交接文件本身。恢复时应保留并在其上继续。

---

## 1. 用户目标与已经确认的决策

### 1.1 总目标

使用 Kronos 原生 K 线模型，在提供的螺纹钢 `rb8888` 和铁矿石 `i8888` 连续合约小时 K 线上微调，预测短期未来走势。

用户关心的是：

- 输入是**小时 K**，不是把小时 K 聚合成日 K 后训练。
- 模型应输出未来的完整小时 K 路径，而不只是一个涨跌分类标签。
- 不要求预测价格逐点完全一致；更看重趋势方向与路径形态是否相似。
- 由于单个交易日只有 5 或 7 根小时 K，V1 的曲线太短，V2 改成预测未来 3 个完整交易日。
- 先使用已有历史数据把方案和效果跑通；暂不要求接入实时数据。
- 不同合约报价、成交量和成交额量级不同，必须正确处理标准化与反归一化。

### 1.2 V2 已确认的任务定义

- 历史输入：预测起点之前的 `256` 根小时 K。
- 预测目标：紧接着的 **3 个完整交易日**。
- 每个交易日有 5 或 7 根小时 K，因此三日目标长度只可能是 `15 / 17 / 19 / 21` 根，整体范围是 15–21 根。
- 这里的“三日”是 3 个期货交易日分组，不是 72 个自然小时；周末和休市日自然跳过。
- 仍生成完整 `[open, high, low, close, volume, amount]` 路径。
- 主方向定义为第三个目标交易日最后一根 close 相对预测起点 close 的涨跌：

```text
origin_close = context 最后一根小时 K 的 close
day3_close   = 第三个目标交易日最后一根小时 K 的 close
r_3d         = day3_close / origin_close - 1
direction_3d = sign(r_3d)
```

- 这不代表只预测第三日。模型仍预测三天内全部 15–21 根 K；只是把第三日末方向设为主要业务指标。
- 同时报告第 1、2、3 个交易日末相对同一 `origin_close` 的方向与收益。
- V2 优先分别训练 `rb8888` 与 `i8888`，不再一开始就强制共享一个 predictor。V1 分合约结果显示共享模型可能存在负迁移。
- 路径生成得到的第三日方向仍是主结果；新增的方向辅助头只作为训练正则与单独的辅助信号，不能悄悄替代 K 线路径结果。

### 1.3 关于“修改指标是否会修改损失函数”的结论

- 单纯修改评估指标或画图逻辑，**不会自动修改训练损失**。
- V1 即使增加 balanced accuracy、路径相关性、区间误差等指标，训练损失仍是 Kronos 的 token 交叉熵。
- V2 若要让模型真正重视三日末方向，需要显式增加方向辅助损失。计划为：

```text
L_total = L_token_CE + lambda_dir * L_direction
```

- 初版 `L_direction` 使用 day1/day2/day3 三个端点的 `BCEWithLogitsLoss`；`L_token_CE` 继续保证完整 K 线路径建模。

---

## 2. V1 已完成的实现

V1 的目标是预测**下一个完整交易日**，目标交易日通常有 7 根小时 K，无夜盘时为 5 根。

### 2.1 模型与固定版本

- Tokenizer：`NeoQuasar/Kronos-Tokenizer-base`
- Tokenizer revision：`0e0117387f39004a9016484a186a908917e22426`
- Predictor：`NeoQuasar/Kronos-small`
- Predictor revision：`901c26c1332695a2a8f243eb2f37243a37bea320`
- Tokenizer 冻结，仅微调 predictor。
- `model/kronos.py` 已增加保留原始采样路径的能力，用于多 seed 和路径指标；默认调用行为保持兼容。
- 原生 Kronos regression test 已通过，说明默认 API 与原先预测结果没有被破坏。

### 2.2 特征与原始字段映射

训练和推理统一使用六个特征：

```text
[open, high, low, close, volume, amount]
```

字段处理：

- `volume` 来自原始字段 `VD`，表示单根 K 的成交量。
- `amount` 来自原始累计成交额 `A`，在每个 `TiD` 交易日内部做差分，转换成单根 K 成交额。
- 持仓量 `OI` 当前没有输入 Kronos，因为原生 tokenizer 使用六特征接口。
- 真正的时间戳必须用 `TeD + T`。
- `TiD` 只用于标识期货交易日分组。
- 原工具曾用 `TiD + T` 生成时间戳，会在夜盘处制造约 700 次/合约的时间倒序；当前实现已修正。

### 2.3 数据审计

用户最初认为每个合约约有 10,000 根，但实际 JSON 文件各有 4,999 根：

| 合约 | 原始 bars | 清洗后 bars | 清洗后交易日 | 5-bar 日 | 7-bar 日 |
|---|---:|---:|---:|---:|---:|
| rb8888 | 4,999 | 4,988 | 718 | 19 | 699 |
| i8888 | 4,999 | 4,993 | 719 | 20 | 699 |

只删除结构上不是 5/7 根的异常交易日，不删除真实大涨大跌：

- `rb8888`：删除 `2023-08-11`（4 根）、`2024-04-04`（1 根）、`2024-04-30`（6 根）。
- `i8888`：删除 `2024-04-30`（6 根）。

详细审计文件：`csj/results/futures_hourly/data_audit.json`。

### 2.4 时间切分

两个合约使用共同日期边界，按时间顺序约 70% / 15% / 15% 切分：

```text
数据首日：  2023-08-11
训练截止：  2025-09-08
验证截止：  2026-02-24
数据末日：  2026-08-03
共同交易日：train 503 / val 107 / test 109
```

日边界预测案例：

| split | rb8888 | i8888 | 合计 |
|---|---:|---:|---:|
| train | 465 | 465 | 930 |
| val | 107 | 107 | 214 |
| test | 109 | 109 | 218 |

这里的测试集 `218` 条是：

```text
109 个历史目标交易日 × 2 个合约 = 218 次独立的下一交易日预测
```

它不是一次性向未来预测 218 天，也不是使用 218 天作为单个输入。每次仍只看之前 256 根小时 K，预测紧接着的 5/7 根，然后滚动到下一个历史日期重新预测。

V1 训练不是只有 930 个日边界案例，而是使用所有不跨 split 的小时滑窗，共 `6,457` 个训练窗口；日边界案例用于业务对齐的验证与测试。

详细切分文件：`csj/results/futures_hourly/split_summary.json`。

### 2.5 标准化与反归一化

不同合约绝对价格和成交量不同，V1 已经做了逐窗口、逐合约、逐特征标准化：

```text
mean, std  = 该样本过去 256 根 context 的逐特征统计量
x_norm     = clip((x - mean) / (std + 1e-5), -5, 5)
target_norm 使用同一组 context mean/std
model_output 仍处于标准化空间
x_raw_pred = model_output * (std + 1e-5) + mean
```

关键约束：

- 统计量只来自该合约当前预测窗口的历史 context。
- 不跨合约共享统计量。
- 不使用目标区间或任何未来数据计算统计量。
- 目标使用与 context 相同的坐标系。
- Kronos 输出先按标准化值理解，再用当前窗口原统计量反归一化。

已经做过原生一致性测试，将自定义推理与仓库原生 `KronosPredictor.predict()` 对照：

| 检查项 | 最大误差 |
|---|---:|
| 六特征反归一化最大相对误差 | `1.0548434547903508e-07` |
| close 最大相对误差 | `3.745430457805629e-08` |
| 标准化 round-trip 最大绝对误差 | `2.384185791015625e-07` |

`amount` 的最大绝对差约为 211，但其原值为十亿级，最大相对差只有 `1.055e-7`，属于 float32 舍入，不是错用标准化或错用合约尺度。

一致性结果：`csj/runs/futures_hourly/full/normalization_consistency.json`。

### 2.6 V1 训练目标和参数

- 输入 lookback：256 根小时 K。
- 训练 horizon：固定 7 根小时 K。
- 验证/测试 horizon：目标完整交易日，动态为 5 或 7 根。
- token loss 只计算目标部分，不计算历史 context 重构；切片起点为 `lookback - 1 = 255`。
- 优化器：AdamW。
- weight decay：0.1。
- gradient clip：3.0。
- warmup：5%。
- scheduler：cosine。
- batch size：32。
- 每次最多 15 epochs，验证 balanced accuracy 连续 3 次不提升则早停；同分时以 return MAE 决胜。
- 正式训练在 Apple MPS 上完成。

学习率搜索：

```text
seed 42: 1e-6 / 3e-6 / 1e-5
胜出 1e-5 后: seeds 42 / 43 / 44
```

正式五次 run 共完成 36 个 epoch；实测每个 epoch 通常约 8–12 分钟，因此 V2 不应未经阶段门槛就直接重复完整网格。

### 2.7 V1 推理、集成和统计检验

- 每个模型每个案例采样 10 条路径。
- `temperature=1.0`、`top_p=0.9`、`top_k=0`。
- inference batch size：2。
- 三个 seed 等权集成。
- 原始 per-seed、per-sample 路径保留，不只保留均值。
- 基线：zero-shot Kronos、majority、momentum。
- 配对 moving/circular block bootstrap：2,000 次、5 交易日 block。
- 同一历史日期的两个合约作为配对单位处理，共 109 个唯一测试日期。

近期已修正的实现细节：

- naive baseline 与模型记录现在共享 opening-gap 字段，敏感性分析口径一致。
- ensemble 的路径相关性与区间误差由集成后的路径重新计算，不再沿用某个 seed 的二级指标。
- `sample_final_returns` 现在拼接三 seed 共 30 条采样结果，不再只取 seed 42。
- JSON 中非有限浮点数会写成 `null`，避免非标准 `NaN/Infinity`。
- 如果完整 consistency 文件缺失，正式流程会自动补跑一致性检查。

### 2.8 当前文件职责

| 文件 | 职责 |
|---|---|
| `csj/utils/tool.py` | 原始 JSON 转 DataFrame、字段转换、时间处理和辅助绘图 |
| `csj/futures_data.py` | 合约加载、结构清洗、时间切分、小时滑窗和日边界案例 |
| `csj/evaluation.py` | 标准化推理、反归一化、采样路径和预测记录 |
| `csj/training.py` | V1 token CE 训练、早停和 checkpoint |
| `csj/metrics.py` | 方向/收益/路径指标、基线、集成和 bootstrap |
| `csj/experiment.py` | audit、consistency、smoke、baseline、full 流程与断点复用 |
| `csj/reporting.py` | 指标图、训练曲线、预测样例和 Markdown 报告 |
| `csj/config.py` | YAML 配置加载与校验 |
| `model/kronos.py` | Kronos 原生模型；当前扩展了采样路径返回，默认行为保持兼容 |
| `tests/test_futures_pipeline.py` | 数据、时间、标准化和 pipeline 测试 |
| `tests/test_futures_metrics.py` | 指标、集成和 bootstrap 测试 |

---

## 3. V1 实验结果与真实结论

### 3.1 验证选择

验证集选择出的学习率为 `1e-5`：

| seed | best epoch | val balanced accuracy | val return MAE |
|---:|---:|---:|---:|
| 42 | 5 | 53.21% | 0.007054 |
| 43 | 1 | 53.16% | 0.007079 |
| 44 | 5 | 51.80% | 0.007067 |

单 seed 看起来能达到约 53.2%，但三 seed 等权集成验证 balanced accuracy 只有 51.75%，低于 zero-shot 的 52.08%。这说明不同 seed 对同一案例的方向判断不稳定。

### 3.2 原封存测试集

| 模型 | pooled balanced accuracy | accuracy | return MAE | return correlation |
|---|---:|---:|---:|---:|
| zero-shot | 51.38% | 53.52% | 0.007138 | -0.0241 |
| fine-tuned ensemble | 51.38% | 53.52% | 0.006788 | -0.0625 |

相对 zero-shot：

- balanced accuracy 改善：`0.00` 个百分点。
- return MAE 改善：约 `4.91%`。
- return correlation 反而更差。
- 平均 close 路径相关性：`0.0104538`，几乎没有路径形态相关性。
- 平均振幅相对误差：`0.332431`。

分合约：

| 合约 | zero-shot bal. acc. | fine-tuned bal. acc. | fine-tuned MAE | fine-tuned return corr. |
|---|---:|---:|---:|---:|
| i8888 | 58.27% | 57.11% | 0.008720 | -0.0336 |
| rb8888 | 44.21% | 45.56% | 0.004855 | -0.1555 |

含义：

- i8888 的方向准确率绝对值较高，但微调比 zero-shot 退化。
- rb8888 微调略高于 zero-shot，但仍低于随机方向的 50%，且收益相关性明显为负。
- pooled 指标掩盖了合约间差异，这正是 V2 优先拆分合约模型的依据。

### 3.3 Bootstrap 与敏感性

fine-tuned ensemble 相对 zero-shot 的 balanced accuracy 配对分块 bootstrap：

```text
点估计：       0.0000
95% 区间：    [-0.06363, 0.06951]
提升为正概率： 0.475
案例数：       218
唯一日期：     109
```

排除唯一一个目标开盘跳空绝对值不小于 3% 的案例后：

```text
zero-shot balanced accuracy： 51.02%
fine-tuned balanced accuracy：51.55%
案例数：217
```

敏感性结果稍有变化，但不足以推翻总体结论。

### 3.4 V1 最终结论

**V1 没有证明微调获得了稳定的下一交易日方向 edge，也没有获得可接受的路径相似性。**

不能因为 return MAE 降低 4.91% 就宣称模型有效，因为：

- 方向 balanced accuracy 与 zero-shot 完全相同。
- 收益相关性更差。
- 路径相关性接近 0。
- seed 分歧大。
- 两个合约的行为相反。

完整结果：`csj/results/futures_hourly/REPORT.md` 和 `csj/results/futures_hourly/metrics.json`。

预测图选择的是每个合约按时间排列后的首/中/末案例，不是挑选最好看的案例。图上预测与真实不相似，属于真实实验结果，不是绘图错误。

---

## 4. 清空上下文后最容易犯的错误

1. **误以为模型在预测日 K。** 实际输入和输出都仍是小时 K；“交易日”只定义预测边界和评价端点。
2. **误以为 218 是预测 horizon。** 它是 109 个日期 × 2 个合约的独立案例；V1 单次 horizon 只有 5/7 根。
3. **误以为没有做标准化。** 已按每个预测窗口的历史 256 根、逐合约、逐特征标准化，并通过原生一致性测试。
4. **误把 Kronos 输出直接当原始价格。** Kronos 输出在标准化空间，必须用同一窗口 context 的 mean/std 反归一化；现有代码已经这样做。
5. **误以为改了指标就改了损失。** V1 指标扩展没有改变 token CE；V2 的方向 loss 必须显式实现。
6. **用 V1 测试集再次宣称“封存测试”。** `2026-02-25` 至 `2026-08-03` 的结果已经被查看，并且 V2 方案受到这些结果影响；它从现在起只能算已观察历史/开发比较集。
7. **把连续 8888 合约结果当作可交易 PnL。** 当前没有真实合约映射、换月规则、手续费和滑点，只能评价历史预测 edge。
8. **只看 pooled 指标。** 必须同时报告 rb8888、i8888，避免一个合约掩盖另一个合约失败。
9. **只展示平均预测曲线。** 10 条采样路径再跨 3 seed 求均值会显著变平；V2 必须展示中位数、10–90% 区间，必要时展示单路径分布。
10. **覆盖 V1 产物。** V2 必须新建 run/result 目录，保留可复现对照。

---

## 5. V2 三交易日方案：预先锁定的定义

### 5.1 预测案例

建议新增 `ThreeTradingDayCase`（名称可调整，但语义不可变），至少保存：

```text
instrument
origin_timestamp
origin_trading_day
context                         # 恰好 256 根
target                          # 后续 3 个完整交易日拼接
target_days                     # 长度 3
day_end_indices                 # target 内 day1/day2/day3 最后一根的位置
pred_len                        # 15/17/19/21
split/fold_id
```

案例构建规则：

- 按每个 instrument 独立构建，绝不跨合约连接窗口。
- context 最后一根必须严格早于 target 第一根。
- target 必须由接下来的 3 个完整 `TiD` 分组组成。
- 三个目标交易日必须全部落在同一个 split/fold 内，不能从 train 跨入 validation，也不能从 validation 跨入 evaluation。
- 目标时间戳仍用 `TeD + T`；`TiD` 只负责交易日分组。
- 保存每个日末索引，避免用固定 `7/14/21` 猜端点。
- 按 `pred_len` 分组或用 padding mask 组 batch；不能因长度不同而截断 7-bar 夜盘或填入下一交易日。

### 5.2 标准化

V2 继续沿用已经验证的 context-only 标准化，不重新发明尺度处理：

```text
stats = 当前案例 context 的 256 根、逐特征 mean/std
context 与完整三日 target 都用这组 stats 标准化
所有生成路径都用同一组 stats 反归一化
```

新增审计：

- 分合约、分特征报告 context 和 target 被 `clip(-5, 5)` 的比例。
- 特别检查三日目标比一日目标更容易发生 clipping 的问题。
- 初版不同时引入 log-volume、log-amount 等额外变换，以便先隔离“更长 horizon”和“拆分合约”的影响；如 clipping 明显，再做独立 ablation。
- V2 consistency 必须覆盖最长 21 根输出，并继续与原生 predictor 对照。

### 5.3 输出与方向口径

对每个案例保存：

- 完整实际 OHLCVA 路径。
- 每个 seed、每个 sample 的完整预测路径。
- seed 内的均值/中位数，跨 seed 的集成路径。
- 10%、50%、90% 分位路径。
- day1/day2/day3 实际 endpoint return。
- day1/day2/day3 路径生成的 endpoint return。
- day1/day2/day3 辅助方向头的 logit/probability（若启用）。

主方向必须由**生成路径第三日最后 close**得出：

```text
predicted_r_3d = predicted_day3_close / origin_close - 1
```

辅助头方向单独命名，例如 `aux_direction_3d`，不能覆盖 `path_direction_3d`。

### 5.4 路径显示坐标

主要图不再只画不同绝对价格尺度的 raw close。使用以预测起点为 0 的累计收益路径：

```text
return_path[t] = close[t] / origin_close - 1
```

每张图至少显示：

- 真实累计收益路径。
- 预测中位数路径。
- 预测 10–90% 区间。
- day1/day2/day3 分界线。
- 第三日末实际与预测方向。

raw price 图可以作为补充，但不能替代 return-space 图。

---

## 6. V2 训练设计

### 6.1 为什么不只把 `train_horizon` 从 7 改成 21

简单改成固定 21 根可以快速验证更长 autoregressive horizon，但它和业务目标仍不完全一致：

- 三个完整交易日可能是 15/17/19/21 根。
- 固定 21 根滑窗可能在短交易日组合后进入第四个交易日。
- 纯日边界三日案例每个合约只有约 460 个训练起点，直接只用这些案例又会大幅减少 token 训练样本。

因此推荐“双流训练”，兼顾样本量和业务对齐。

### 6.2 双流训练

#### A. Dense token stream

- 每个合约独立的小时滑窗。
- context 256，固定 horizon 21。
- 使用所有不跨训练边界的小时起点。
- 继续只对目标 token 计算 Kronos CE。
- 用途：保留较多训练样本，让生成器学习长一些的局部 K 线路径。

#### B. Trading-day direction stream

- 只使用日边界 `ThreeTradingDayCase`。
- target 恰好是后续 3 个完整交易日。
- 从**仅包含 context token 的 forward**取得最后一个 context hidden state，输入方向辅助头。
- 方向标签为 day1/day2/day3 endpoint 相对同一 origin close 的正负。
- 严禁把 target token 送入辅助头的 context encoder；即使模型内部注意力不是严格因果，也不会产生目标泄漏。

训练时按可复现的比例交替两个 stream，例如每个 dense batch 后取一个循环的 direction batch。具体比例必须写入 config 和 checkpoint metadata。

### 6.3 方向辅助头

推荐把训练专用 wrapper 放在 `csj/` 下，不改变 `Kronos.forward()` 的默认返回接口：

```text
KronosTrendWrapper
  predictor: Kronos
  trend_head: Linear(d_model, 3)
```

实现思路：

1. 对 context-only token 调用 `Kronos.decode_s1()`。
2. 读取返回的 transformer context，形状 `[B, T, d_model]`。
3. 取最后一个有效 context token 的 hidden state。
4. `Linear(d_model, 3)` 输出 day1/day2/day3 logits。
5. 用三个二分类标签计算 `BCEWithLogitsLoss`。

checkpoint 必须同时保存：

- predictor state。
- trend head state。
- instrument。
- fold、日期边界。
- normalization/clip 参数。
- `lambda_dir`。
- 数据和配置摘要。

### 6.4 总损失

```text
L_token = Kronos 两级 token CE，仅覆盖 dense stream 的未来 21 根
L_dir   = mean(BCE(day1), BCE(day2), BCE(day3))
L_total = L_token + lambda_dir * L_dir
```

推荐执行顺序：

1. 先跑 per-contract、21-horizon、纯 CE，令 `lambda_dir=0`。
2. 再加入方向头，首个 smoke 使用固定 `lambda_dir=0.2`。
3. 只有 smoke 正确且方向/路径有改善迹象时，才比较 `lambda_dir=0.1` 与 `0.3`。
4. 不要在已观察结果上无限扫 lambda；所有候选必须在正式 run 前写入配置。

如果后续发现 BCE 提高了辅助分类但没有提高生成路径方向，应明确报告“分类头有效、路径生成无效”，不能合并口径。

### 6.5 参数更新范围

V2 第一版建议：

- tokenizer 继续冻结。
- predictor 全量微调，与 V1 保持一致，便于对照。
- trend head 参与优化。
- rb8888 与 i8888 各自从同一个官方 Kronos-small 初始权重出发，不能让第二个合约从第一个合约的微调权重继续训练。

后续若样本过少或过拟合，再单独比较：

- 冻结底层、只训练高层 + trend head。
- LoRA/adapter。
- 先共享预热、再分合约微调。

这些不是 V2 第一阶段的默认动作，避免一次改变太多因素后无法归因。

---

## 7. V2 评估与实验纪律

### 7.1 当前测试集已经不再封存

V1 的 `2026-02-25` 至 `2026-08-03` 测试结果已经被查看；V2 的设计正是因为看到了这些结果才调整。因此：

- V2 可以在这段历史上做透明的 retrospective/walk-forward 开发比较。
- 不能再把它写成“完全未见的 sealed test”。
- 真正新的最终结论需要获取 `2026-08-03` 之后的新增数据，或者未来先冻结模型再等待新数据。

### 7.2 建议的历史 walk-forward 口径

为了先用现有历史评估效果，预先固定 expanding walk-forward：

```text
共同交易日序列
minimum_train_days = 360
evaluation_days    = 60
step_days          = 60
使用能够生成的全部完整 folds
每个三日 target 必须完整落在对应 evaluation fold 内
```

每个 fold：

- 训练只用 fold 截止日前的数据。
- normalization 仍逐案例只用 context。
- checkpoint 选择只能用该 fold 训练尾部再划出的 inner validation，不能查看 fold evaluation 标签。
- rb/i 使用完全相同的日期 fold，结果既分合约报告，也按日期配对汇总。
- 最终报告 mean、standard deviation、每个 fold 明细，不能只挑最好 fold。

这个 walk-forward 仍属于“已观察历史上的开发证据”，不是新的完全独立样本，但比只盯单个原测试段更能暴露 regime 和合约差异。

### 7.3 基线

V2 至少比较：

- 官方 zero-shot Kronos，直接预测相同三日路径。
- majority direction。
- context momentum（口径需预先固定，例如最近一个完整交易日或最近 7 bars）。
- persistence/flat-return 路径。
- V2 per-contract CE-only。
- V2 per-contract CE + direction auxiliary loss。

同一案例、同一采样预算、同一方向阈值下比较。

### 7.4 主要方向指标

主指标：

- `day3_path_direction_balanced_accuracy`。

必须同时报告：

- day1/day2/day3 path-derived balanced accuracy。
- day1/day2/day3 ordinary accuracy、类别分布和混淆矩阵。
- day1/day2/day3 endpoint return MAE 与 bias。
- 实际收益和预测收益相关性。
- 辅助头 day1/day2/day3 balanced accuracy，单独成表。
- rb8888、i8888、pooled 三套结果。

若实际 endpoint return 恰好为 0，则与 V1 一样不计入二元方向 balanced accuracy，并单独报告排除数量。

### 7.5 路径形态指标

至少实现和报告：

1. **Return-path Pearson correlation**：以 origin close 锚定后的累计收益路径相关性。
2. **Z-normalized DTW distance**：衡量忽略绝对振幅后的形态相似度；越低越好。
3. **Return-space DTW distance**：同时保留方向和振幅信息；越低越好。
4. **Step/slope sign agreement**：相邻小时 close 变化方向的一致率。
5. **Turning-point similarity**：对小噪声设置预先固定阈值后比较局部转折；阈值不得看结果后修改。
6. **Range relative error**：预测三日 high-low 范围与真实范围的相对误差。
7. **Endpoint MAE**：day1/day2/day3 端点收益误差。

不要只用 raw price correlation：价格序列的共同绝对水平会产生虚高相关，也无法跨合约公平比较。

### 7.6 不确定性与统计检验

- 保留每个 seed 的结果，报告 seed dispersion。
- 采样路径展示 median 和 10–90% band。
- 对方向 improvement 做按日期配对的 moving-block bootstrap；同日 rb/i 一起抽样。
- block 长度先固定为 5 交易日，并增加 10 日作为敏感性分析。
- 报告点估计、95% CI 和 improvement > 0 的 bootstrap 比例。
- 如果做多个 lambda/LR 候选，最终比较需明确选择过程，避免把选择集表现当无偏测试表现。

---

## 8. 建议的阶段执行计划

### Phase 0：保护现场与正确性测试

目标：在任何长时间训练前证明三日案例和 loss 没有泄漏。

任务：

- 新建 `csj/configs/futures_3day_trend.yaml`。
- 新建独立 output root，例如 `csj/runs/futures_3day_trend`。
- 新建独立 result root，例如 `csj/results/futures_3day_trend`。
- 实现三交易日 case builder、day-end index 和长度分组。
- 实现 path-return 与三日端点指标。
- 增加以下单元测试：
  - 三天 5/7 组合对应 `15/17/19/21` 长度。
  - 三个 day-end index 正确。
  - 不跨 instrument。
  - 不跨 split/fold。
  - context 恰好 256 且严格早于 target。
  - normalization stats 只由 context 计算。
  - 反归一化最长 21 根的一致性。
  - direction head 只收到 context token，构造性修改 target 不改变 head logits。
  - token loss 仍只覆盖未来目标 token。
  - 合成路径上的 correlation/DTW/slope/turning-point 指标结果可手算。
- 跑原有全部测试和原生 regression，确保 V1 不回归。

Phase 0 完成条件：全部测试通过、数据审计无泄漏、V1 报告和 checkpoint 未改变。

### Phase 1：三日 zero-shot 与数据基线

目标：不训练，先建立三日任务本身的难度和可视化基准。

任务：

- 在锁定的 walk-forward cases 上跑官方 zero-shot 三日预测。
- 跑 majority、momentum、flat/persistence 基线。
- 输出分合约的 day1/day2/day3 指标和路径指标。
- 画 return-space 路径、中位数和 10–90% band。
- 审计各特征 clipping rate 和 autoregressive horizon 越长后的异常路径比例。

Phase 1 完成条件：指标可复现，案例数、日期和每个 pred_len 分布都写入 JSON/报告。

### Phase 2：拆分合约、纯 CE 的 21-horizon 基线

目标：隔离两个变化的效果——更长 horizon、每个合约单独微调；暂不加入方向 loss。

任务：

- rb8888 和 i8888 分别从官方 predictor 初始化。
- dense stream 固定 21 根，`lambda_dir=0`。
- 初始 LR 只比较 `[3e-6, 1e-5]`，先用 seed 42。
- 每个合约先做 1–2 batch 的 MPS smoke，再开始完整 epoch。
- checkpoint 选择使用 inner validation 的预先固定复合规则：主看 day3 path balanced accuracy，同分看 day3 return MAE，再看 path DTW。
- 与相同 cases 上的 zero-shot 比较。

Phase 2 的意义：如果纯 CE 已经让三日路径显著好转，就能把改善归因于 horizon/拆分模型，而不是辅助头。

### Phase 3：加入方向辅助 loss

目标：验证显式方向监督是否能改善生成路径的第三日方向，而不只是改善分类头。

任务：

- 首跑固定 `lambda_dir=0.2`、seed 42。
- 报告 path-derived 和 auxiliary-head 两套方向指标。
- 只有相对 Phase 2 有一致改善，才追加 `lambda_dir=0.1/0.3` 小范围比较。
- 检查 token CE、方向 BCE、梯度范数和路径质量是否出现此消彼长。

Phase 3 必须回答：

1. 辅助头自身方向是否改善？
2. 完整生成路径的 day3 方向是否改善？
3. 路径相关性/DTW 是否同时改善，还是分类与生成脱节？

### Phase 4：多 seed 稳定性

只在 Phase 2 或 Phase 3 达到预先门槛后进行：

- 选定每个合约的 LR 和 lambda。
- 运行 seeds `42/43/44`。
- 保存所有原始路径，不能只保留 ensemble。
- 比较单 seed、均值集成、中位数集成。
- 报告 seed 标准差和逐案例分歧。

### Phase 5：报告与是否值得继续

生成新的：

```text
csj/results/futures_3day_trend/REPORT.md
csj/results/futures_3day_trend/metrics.json
csj/results/futures_3day_trend/data_audit.json
csj/results/futures_3day_trend/fold_summary.json
csj/results/futures_3day_trend/path_examples.png
csj/results/futures_3day_trend/training_curves.png
```

报告必须明确分成：

- 历史开发证据。
- 尚未验证的假设。
- 是否达到继续收集新数据、冻结模型做真正前瞻测试的门槛。

---

## 9. 预先建议的验收门槛

下面是建议在正式 V2 结果出现前锁定的工程门槛，避免结果出来后移动标准。若要调整，应在首次正式训练前修改本文件或配置并记录原因。

### 9.1 正确性门槛（必须全部满足）

- 三日案例无日期、split、合约泄漏。
- 目标长度和三个日末索引 100% 正确。
- 标准化/反归一化一致性与 V1 同量级，六特征最大相对差不高于 `1e-6`。
- 默认 Kronos regression test 继续通过。
- 方向头 target-leakage 测试通过。
- V1 的 11 个 futures pipeline/metrics tests 继续通过。

### 9.2 是否值得进入多 seed 的开发门槛

推荐同时满足：

- pooled day3 path balanced accuracy 至少比同口径 zero-shot 高 `2` 个百分点。
- rb8888 和 i8888 都不低于 50%，且任一合约相对 zero-shot 的退化不超过 `1` 个百分点。
- 平均 return-path correlation 至少达到 `0.10`，并且两个合约都为正。
- z-normalized DTW 相对 zero-shot 至少改善 `5%`。
- 改善不只来自某一个 fold；至少多数 folds 同方向改善。

这些是开发筛选门槛，不代表统计显著或可交易。

### 9.3 真正“有效”的最终证据门槛

需要在 `2026-08-03` 之后的新数据上、冻结任何参数和阈值后评估，并满足：

- day3 path direction 相对预先选定最强基线的配对 bootstrap 95% CI 下界大于 0。
- 两个合约分别不出现明显负 edge。
- 路径指标和方向指标同时改善，而不是只有 MAE 或辅助分类头改善。
- 多 seed 方向 balanced accuracy 标准差建议不超过 2 个百分点。
- 若要宣称可交易，还必须另加真实主力/可执行合约映射、换月、手续费、滑点和下单约束回测。

---

## 10. 已知风险和需要诚实报告的限制

- 每个合约只有约 5,000 根小时 K、约 719 个交易日，深度模型样本量偏小。
- 拆分合约能减少负迁移，但也进一步减少每个模型的数据量。
- 15–21 根 autoregressive 输出会比 5/7 根更容易误差累积。
- 三日路径看起来更长，不等于模型自然就会有更高形态相关性。
- 方向辅助头可能只学会分类，未必改善 autoregressive K 线生成。
- 均值集成会把波动和转折抹平，所以必须报告中位数、区间和单样本分布。
- 连续合约包含拼接和换月效应，不能直接等价为某个可成交合约价格过程。
- 路径指标很多，存在多重比较和选择性报告风险；主指标必须预先固定为 day3 path balanced accuracy。
- 已有 V1 测试区间被观察过，后续任何在同一历史上的提升都可能含有研究者适应性偏差。
- 如果 V2 仍无改善，应接受“当前数据规模/模型/目标下没有可靠 edge”，而不是无限扩大超参数搜索。

---

## 11. 当前产物与 checkpoint

V1 报告目录：

```text
csj/results/futures_hourly/REPORT.md
csj/results/futures_hourly/metrics.json
csj/results/futures_hourly/data_audit.json
csj/results/futures_hourly/split_summary.json
csj/results/futures_hourly/test_metrics.png
csj/results/futures_hourly/training_curves.png
csj/results/futures_hourly/forecast_examples.png
```

V1 正式 checkpoint：

```text
csj/runs/futures_hourly/full/training/lr_1e-06_seed_42/best_model.pt
csj/runs/futures_hourly/full/training/lr_3e-06_seed_42/best_model.pt
csj/runs/futures_hourly/full/training/lr_1e-05_seed_42/best_model.pt
csj/runs/futures_hourly/full/training/lr_1e-05_seed_43/best_model.pt
csj/runs/futures_hourly/full/training/lr_1e-05_seed_44/best_model.pt
```

最终集成使用后三个 `1e-5` checkpoint。

当前测试状态（最近一次执行）：

```text
11 个 futures pipeline/metrics tests：通过
1 个 Kronos native regression（256 context）：使用项目本地离线模型缓存后通过
git diff --check：通过
```

补充说明：不设置缓存变量直接跑原生 regression 时，受限环境会去 Hugging Face 默认路径找模型，并在下载/读取 config 阶段失败；这不是数值回归断言失败。项目本地缓存中已同时存在固定 revision 的 tokenizer 和 predictor，离线复跑通过。

恢复后仍需重新执行一次，确认清空会话或后续改动没有改变工作区：

```bash
.venv/bin/python -m pytest tests/test_futures_pipeline.py tests/test_futures_metrics.py -q
HF_HUB_CACHE=/Users/eurus/Code/kronos/Kronos/csj/artifacts/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python -m pytest 'tests/test_kronos_regression.py::test_kronos_predictor_regression[256]' -q
git diff --check
```

---

## 12. 下一位执行者的第一批具体动作

1. 只读检查 git status、Python 版本、V1 报告和 checkpoint。
2. 不重新跑 V1 full，不重新下载已有模型，优先复用 cache。
3. 建立 V2 独立配置与目录，锁定 run id，例如 `v2_seed42_ce_only`。
4. 先实现 `ThreeTradingDayCase` 和 builder，并立即写长度、端点、边界测试。
5. 运行数据审计，记录：总 case 数、分 split/fold 数、分合约数、`15/17/19/21` 分布。
6. 实现 return-space 三日端点与路径指标，并用合成数据测试。
7. 跑 zero-shot Phase 1，先确认长 horizon 的基础表现与推理耗时。
8. 再实现 per-contract CE-only Phase 2；不要一开始就加入方向头，否则无法判断改进来自哪里。
9. Phase 2 完成并出对照表后，再加入 context-only trend head 和显式方向 loss。
10. 每个耗时训练前打印模型来源、instrument、fold、LR、seed、lambda、样本数和预计 batches，防止跑错配置。

建议的新文件边界：

```text
csj/configs/futures_3day_trend.yaml  # V2 配置
csj/trend_model.py                   # 训练专用 wrapper / trend head
csj/three_day_training.py            # 双流 trainer，避免继续膨胀 V1 trainer
csj/three_day_experiment.py          # V2 CLI 与阶段编排
```

数据与通用指标可以扩展现有 `futures_data.py`、`evaluation.py`、`metrics.py`，但必须保持 V1 接口和测试可用。

---

## 13. 一句话状态

V1 已完整实现并验证了尺度处理，但实测仅降低了收益 MAE，没有改善方向，预测路径相关性接近 0；下一步不是继续美化 5/7 根曲线，而是保留 V1、拆分合约、把任务改成未来 3 个完整交易日的 15–21 根小时 K，并用“长 horizon CE 基线 → 显式三端点方向辅助 loss → 多 seed”的阶段实验，透明地在已观察历史上做 walk-forward 开发验证，最终等待 2026-08-03 之后的新数据做真正独立检验。
