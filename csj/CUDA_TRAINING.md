# V2 CUDA 训练启动说明

## 1. 准备环境

把当前工作区同步到 CUDA 机器，至少要包含：

- `csj/`、`model/`、`tests/` 和 `requirements.txt`
- `csj/data/kline_rb8888.json`
- `csj/data/kline_i8888.json`

使用 Python 3.12 虚拟环境，并安装与机器驱动匹配的 CUDA 版 PyTorch：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最后一项必须显示 `True` 和正确的 GPU 名称。

## 2. 先跑单折 pilot

新机器没有模型缓存时，首次运行允许下载固定版本的模型：

```bash
ALLOW_MODEL_DOWNLOAD=1 RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh pilot
```

如果已经复制了 `csj/artifacts/hf_cache`，直接运行：

```bash
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh pilot
```

`pilot` 会依次执行：

1. CPU 回归测试；
2. CUDA 数据审计和 21 根一致性检查；
3. fold 00 的 CUDA zero-shot 基线；
4. 两个合约的 2-batch CUDA smoke；
5. fold 00 的两个学习率训练和评估。

## 3. 继续全量训练

pilot 正常后，保持同一个 `RUN_ID`：

```bash
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh full
```

全量运行会补齐 5 个 folds。已完成的 Phase 2 训练、最佳模型和完整 epoch 状态会自动复用。

训练中断后直接恢复：

```bash
# 单折 pilot 中断
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh resume-pilot

# 全量 Phase 2 中断
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh resume
```

恢复点包含当前模型、AdamW、学习率计划、随机状态、数据顺序和早停计数；不会从头重跑已完成 epoch。如果中断发生在 Phase 1，重新执行 `pilot` 或 `full` 即可复用已经保存的预测分片。

## 4. 常用单独命令

```bash
# 只检查环境、测试和数据
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh check

# 只跑完整 CUDA zero-shot 基线
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh phase1

# 已有对应 Phase 1 缓存时，只跑 Phase 2
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh phase2
```

## 5. 运行 Phase 3 方向辅助损失

Phase 3 固定 `lambda_dir=0.2`、seed 42，保留与 Phase 2 相同的两个
learning-rate 候选、walk-forward folds 和 checkpoint 选择规则。它要求同一
`RUN_ID` 下已经存在对应的 Phase 1 zero-shot 与 Phase 2 逐案例预测缓存，
以便输出严格配对的比较与 moving-block bootstrap。

只复制 `csj/results/futures_3day_trend/phase*_cuda_metrics.json` **不够**：
这些是汇总，不能重建逐案例的配对比较。若 Phase 1/2 在另一台机器完成，先把
该机器同一 `RUN_ID` 的整个
`csj/runs/futures_3day_trend/<RUN_ID>/` 目录同步过来（至少保留
`phase1_cuda/zero_shot/` 与 `phase2/evaluation/`）。Phase 3 会在加载模型、
开始训练前检查这两类缓存，缺失时直接报出待复制路径。

```bash
# 先确认 context-only direction head、双流 loss 和 checkpoint 都能跑通
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh phase3-smoke

# 只跑 fold 00；建议先看报告再启动全量
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh phase3-pilot

# 全部 5 个 folds
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh phase3

# 中断后复用 completed run，并从完整 epoch 状态继续
RUN_ID=cuda_v2 bash csj/scripts/run_v2_cuda.sh resume-phase3
```

`phase3-pilot` 对应的是之前 `pilot` 产生的 `phase1_pilot_cuda/` 缓存；如果你
已经完成的是完整 `full` / `phase1` / `phase2`，请直接使用 `phase3`，不要把
full 的缓存与 pilot 命令混用。

Phase 3 会分别保存：

- `csj/runs/futures_3day_trend/<RUN_ID>/phase3/training/`：predictor、trend
  head、AdamW、scheduler、随机状态和每 epoch 的 token CE / direction BCE /
  gradient norm；
- `phase3/evaluation/`：完整生成路径记录；
- `phase3/auxiliary_evaluation/`：context-only 辅助头记录；
- `csj/results/futures_3day_trend/phase3_cuda_metrics.json` 与
  `PHASE3_CUDA_REPORT.md`：路径指标、辅助头指标、与 Phase 2/zero-shot 的
  paired moving-block bootstrap。

训练、内部验证和最终推理都会由 `--device cuda` 强制放在 CUDA 上。实际设备会写入：

```text
csj/runs/futures_3day_trend/<RUN_ID>/resolved_config.json
```

耗时产物位于 `csj/runs/futures_3day_trend/<RUN_ID>/`，汇总结果位于 `csj/results/futures_3day_trend/`。V1 目录不会被读取、覆盖或删除。

## 6. 注意事项

- `RUN_ID` 不要中途更换，否则自动恢复找不到原 checkpoint。
- 首次下载完成后不要再设置 `ALLOW_MODEL_DOWNLOAD=1`，脚本会默认离线读取缓存。
- 当前脚本使用单张 CUDA GPU；可用 `CUDA_VISIBLE_DEVICES=0` 指定设备。
- 不要随意改学习率、batch size、fold 或采样数，否则不再属于当前锁定实验。
- Phase 3 的主结果仍是生成路径的 day3 direction；辅助头分类指标不能替代它。
- `full` 结果仍是已观察历史上的开发验证，不应写成真正独立的新数据结论。
