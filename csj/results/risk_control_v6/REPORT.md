# V6 P0 风险标签与支持度审计

- 结论：**P0 未通过**。
- P1 路径库：**保持锁死**。
- 结果范围：`retrospective_observed_contracts`。
- 生产资格：`production_eligible: false`。
- 数据指纹：`921f843b1fcdaec8874235d0108319a90271dda1a352e11b6162d8ca892e6476`。
- 风险标签规则：`adverse-excursion-vol-scaled-v1`。

## 五折主要品种事件支持度

| fold | split | cases | long events | short events |
|---|---:|---:|---:|---:|
| fold_00 | fit | 237 | 48 | 48 |
| fold_00 | inner_validation | 203 | 64 | 18 |
| fold_00 | evaluation | 253 | 9 | 57 |
| fold_01 | fit | 436 | 88 | 88 |
| fold_01 | inner_validation | 242 | 19 | 48 |
| fold_01 | evaluation | 306 | 83 | 51 |
| fold_02 | fit | 670 | 134 | 134 |
| fold_02 | inner_validation | 300 | 104 | 53 |
| fold_02 | evaluation | 354 | 103 | 92 |
| fold_03 | fit | 965 | 193 | 193 |
| fold_03 | inner_validation | 347 | 57 | 77 |
| fold_03 | evaluation | 396 | 124 | 42 |
| fold_04 | fit | 1300 | 260 | 260 |
| fold_04 | inner_validation | 388 | 122 | 73 |
| fold_04 | evaluation | 444 | 89 | 61 |

## 五折 evaluation 汇总

| product | cases | long events | short events |
|---|---:|---:|---:|
| i | 603 | 156 | 100 |
| jm | 602 | 114 | 92 |
| rb | 548 | 138 | 111 |

## Gate 失败项

- `fold_00:fit:long_event_support`
- `fold_00:fit:short_event_support`
- `fold_00:inner_validation:short_event_support`
- `fold_00:evaluation:long_event_support`
- `fold_01:fit:long_event_support`
- `fold_01:fit:short_event_support`
- `fold_01:inner_validation:long_event_support`

## 防泄漏与切分审计

- 所有 past-only 与 split 检查均通过。
- 未来 OHLCVA 扰动检查：60 个，失败 0 个。

支持度不足不会通过降低 80% 分位标签、延长 evaluation 或并入迁移品种来补救。

## 证据产物

- 数据审计：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_expanded_20260813_104537/data_audit.json`
- P0 gate：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_expanded_20260813_104537/p0_gate.json`
- 标签分布图：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_expanded_20260813_104537/p0/figures/risk_label_distributions.png`
- 事件支持度图：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_expanded_20260813_104537/p0/figures/fold_event_support.png`
- fit-only 阈值图：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_expanded_20260813_104537/p0/figures/fold_tail_thresholds.png`

图表的分布轴仅为可读性裁剪；所有审计统计和 gate 均使用完整数值。
