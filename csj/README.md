# 小时期货微调实验

这套实验把 Kronos-small 微调到 `rb8888` 与 `i8888` 的小时 K 线上，目标是预测下一个完整交易日（5 或 7 根小时线）。训练样本使用连续 7 根小时线作为目标；验证和测试只在完整交易日边界发起预测。

## 尺度约定

Kronos 的预测器输出不是合约原始价格，而是标准化空间中的
`[open, high, low, close, volume, amount]`。每个样本独立执行：

```text
mean, std = 过去 256 根小时线逐特征统计量
model_input = clip((context - mean) / (std + 1e-5), -5, 5)
raw_prediction = model(model_input)          # 仍在标准化空间
prediction = raw_prediction * (std + 1e-5) + mean
```

`mean/std` 只由该合约、该预测时点之前的 256 根线计算，既不跨合约共享，也不使用目标区间数据。因此不同报价单位和成交量级不会直接混在同一绝对数值空间里。训练目标也使用同一上下文统计量标准化。

`consistency` 阶段会把自定义推理路径与仓库原生 `KronosPredictor.predict()` 对照，检查六个特征的反归一化结果；超过相对误差阈值会直接失败。

## 数据与切分

- `TeD + T` 是真实时间戳，`TiD` 是期货交易日。
- `VD` 作为单根成交量；累计成交额 `A` 在各交易日内差分成单根成交额。
- 只剔除不是 5/7 根线的结构异常交易日，不删除真实的大波动。
- 全市场共同按时间做 70%/15%/15% 切分，测试集在学习率和随机种子选择完成前保持封存。
- tokenizer 冻结，只微调 predictor；损失仍是 Kronos token 交叉熵，并且只计算目标 7 根线对应的 token。

## 运行

项目约定使用 Python 3.12 虚拟环境：

```bash
.venv/bin/python -m csj.experiment audit --run-id full
.venv/bin/python -m csj.experiment consistency --run-id full
.venv/bin/python -m csj.experiment smoke --run-id full
.venv/bin/python -m csj.experiment baseline --run-id full
.venv/bin/python -m csj.experiment full --run-id full
```

正式实验依次搜索学习率 `1e-6 / 3e-6 / 1e-5`（seed 42），再用胜出学习率训练 seeds `42/43/44`。每次最多 15 epochs，验证 balanced direction accuracy 连续 3 次不提升即早停。最终报告写入 `csj/results/futures_hourly/`。

连续合约只能用来测历史预测 edge；没有真实可交易合约映射、换月规则、手续费和滑点时，报告不会把结果称作可执行 PnL。
