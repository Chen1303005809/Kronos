# V1 / V2 兼容归档

这里记录 V3 之前的训练资产，避免新实验误用连续合约或 V2 的方向辅助损失。
为保护已有 checkpoint、报告和外部调用，旧模块暂保留原路径作为兼容入口；V3 不应导入或覆盖它们。

| 分类 | 保留入口 | 数据口径 | 状态 |
| --- | --- | --- | --- |
| V1 | `csj.experiment`、`csj/training.py`、`csj/evaluation.py` | `rb8888` / `i8888` 连续合约，下一交易日 | 归档，仅复现实验 |
| V2 | `csj.three_day_experiment`、`csj/three_day_*.py`、`csj/scripts/run_v2_cuda.sh` | 连续合约，未来三个交易日 | 归档，仅复现实验 |
| V3 | `csj.v3.experiment`、`csj/v3/`、`csj/scripts/run_v3_cuda.sh` | 具体交割月合约、可变邻居面板 | 当前实现 |

旧配置保持原位以便已有命令可复现：

- V1：`csj/configs/futures_hourly.yaml`
- V2：`csj/configs/futures_3day_trend.yaml`
- V3 正式完整面板：`csj/configs/active_contract_panel_v3.yaml`
- V3 当前探索性 partial-panel：`csj/configs/active_contract_panel_v3_partial.yaml`

V3 的输出只写入 `csj/runs/active_contract_panel_v3/` 与
`csj/results/active_contract_panel_v3/`，不会读取、删除或覆盖 V1/V2 的 runs、results 或模型。
