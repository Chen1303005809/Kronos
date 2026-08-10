# V3 CUDA 训练与同步说明

V3 使用具体交割月合约和活跃合约面板。正式训练入口是：

```bash
bash csj/scripts/run_v3_cuda.sh audit
```

它与 V1/V2 训练目录隔离，详细策略见 `csj/ACTIVE_CONTRACT_PANEL_TRAINING_STRATEGY.md`，旧入口分类见 `csj/legacy/README.md`。

当前归档数据只有 `partial_panel` 快照。用户已明确允许使用它进行探索性训练，因此使用独立配置
`csj/configs/active_contract_panel_v3_partial.yaml`；严格完整面板配置
`active_contract_panel_v3.yaml` 仍保留给未来正式验证。两者的 runs/results 目录彼此隔离。

## CUDA 机器准备

同步整个 `Kronos/` 目录最稳妥；至少须包含 `csj/`、`model/`、`tests/`、`requirements.txt` 和 V3 快照目录。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最后一条必须显示 CUDA 可用及正确 GPU。若 `pip install -r requirements.txt` 安装的是 CPU PyTorch，请先按 CUDA 机器的驱动和官方 PyTorch 指引安装匹配的 CUDA wheel，再安装其余依赖。

首次没有模型缓存时允许下载固定 revision：

```bash
ALLOW_MODEL_DOWNLOAD=1 RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh audit
```

已有 `csj/artifacts/hf_cache` 时，不设置 `ALLOW_MODEL_DOWNLOAD`，脚本会默认离线读取缓存。

## 固定运行顺序

```bash
# 环境、V3 单元测试、数据审计；必须先完成
RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh check

# 仅重跑数据和 256/512 覆盖审计
RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh audit

# 当前数据的探索性训练：必须显式指定 partial 配置
CONFIG=csj/configs/active_contract_panel_v3_partial.yaml RUN_ID=v3_partial_cuda \
  bash csj/scripts/run_v3_cuda.sh check

# P0 的 zero-shot 目标单流路径基线
CONFIG=csj/configs/active_contract_panel_v3_partial.yaml RUN_ID=v3_partial_cuda \
  bash csj/scripts/run_v3_cuda.sh p0-zero-shot

# P0 的 CE-only 目标单流微调
CONFIG=csj/configs/active_contract_panel_v3_partial.yaml RUN_ID=v3_partial_cuda \
  bash csj/scripts/run_v3_cuda.sh p0

# 当前数据的探索性 Pair Probe
CONFIG=csj/configs/active_contract_panel_v3_partial.yaml RUN_ID=v3_partial_cuda \
  bash csj/scripts/run_v3_cuda.sh p1
```

`full` 依次运行上述阶段。严格配置下，P1 若因数据门槛退出是正确行为：当前唯一快照是 `partial_panel`，且其活跃清单晚于大部分历史预测原点。partial 配置会运行 P1，但每份 P0/P1 输出都会带有 `result_scope: exploratory_partial_panel`；即使统计门槛通过，`passes_p1_to_p2_gate` 也会保持 `false`，不得据此实施 P2。

P1 通过所有预注册门槛前，不要实现或运行 P2 条件路径生成：

- Day3 balanced accuracy 相对 target-only Probe 至少提升 2pp；
- 多数 walk-forward folds 改善；
- 单一品种退化不得超过 1pp；
- 5 日和 10 日配对 moving-block bootstrap 的正改进概率均至少 80%；
- 更早与更晚交割月份邻居均须存在。

## 日常快照归档

在每个交易日持续保存完整清单和原始 K 线；该命令每个合约使用独立进程、socket 超时和硬超时，不做无界重试：

```bash
.venv/bin/python -m csj.active_contract_data \
  --output-root csj/data/active_contract_snapshots \
  --socket-timeout 15 \
  --process-timeout 30
```

收集失败会写入 manifest，而不是通过未来成交量或主力规则替换邻居。需要先补齐多个交易日的完整 `panel_completeness: complete` 快照，P1 才能得到正式的历史配对样本；在此之前，partial 配置的训练结果仅用于探索和后续数据对比。

## 将 CUDA 结果同步回本机

每个 `RUN_ID` 都要完整同步，而不只是汇总 JSON：

```text
csj/runs/active_contract_panel_v3_partial/<RUN_ID>/
csj/results/active_contract_panel_v3_partial/
```

其中 run 目录含固定配置、P0 checkpoint、P1 head checkpoint、逐案例预测和训练历史；results 目录含数据审计与阶段汇总。同步完成后，我可以基于同一组 case key 检查 P0/P1 的配对指标、fold 退化和 bootstrap，而不是只读一份总分数。
