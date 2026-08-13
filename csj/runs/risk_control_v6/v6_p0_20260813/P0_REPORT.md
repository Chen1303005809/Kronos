# V6 P0 风险标签与支持度审计

- 结论：**P0 未通过**。
- P1 路径库：**保持锁死**。
- 结果范围：`retrospective_observed_contracts`。
- 生产资格：`production_eligible: false`。
- 数据指纹：`e08be7ee29b3abfcec6b22426595934804fc48b5d5525f8ca6d1565ec8648fc0`。
- 风险标签规则：`adverse-excursion-vol-scaled-v1`。

## 五折主要品种事件支持度

| fold | split | cases | long events | short events |
|---|---:|---:|---:|---:|
| fold_00 | fit | 232 | 47 | 47 |
| fold_00 | inner_validation | 204 | 51 | 21 |
| fold_00 | evaluation | 270 | 5 | 63 |
| fold_01 | fit | 429 | 86 | 86 |
| fold_01 | inner_validation | 257 | 19 | 50 |
| fold_01 | evaluation | 322 | 93 | 52 |
| fold_02 | fit | 680 | 136 | 136 |
| fold_02 | inner_validation | 317 | 108 | 54 |
| fold_02 | evaluation | 386 | 105 | 89 |
| fold_03 | fit | 991 | 199 | 199 |
| fold_03 | inner_validation | 378 | 57 | 94 |
| fold_03 | evaluation | 427 | 136 | 69 |
| fold_04 | fit | 1352 | 271 | 271 |
| fold_04 | inner_validation | 418 | 122 | 81 |
| fold_04 | evaluation | 479 | 97 | 68 |

## 五折 evaluation 汇总

| product | cases | long events | short events |
|---|---:|---:|---:|
| i | 620 | 154 | 113 |
| jm | 698 | 132 | 113 |
| rb | 566 | 150 | 115 |

## Gate 失败项

- `fold_00:fit:long_event_support`
- `fold_00:fit:short_event_support`
- `fold_00:evaluation:long_event_support`
- `fold_01:fit:long_event_support`
- `fold_01:fit:short_event_support`
- `fold_01:inner_validation:long_event_support`
- `past_only_and_split_leakage_failures`

## 防泄漏与切分审计

- `fold_04:prediction_day_atomic` 未通过；违规预测日：2026-05-13。
- 未来 OHLCVA 扰动检查：66 个，失败 0 个。

支持度不足不会通过降低 80% 分位标签、延长 evaluation 或并入迁移品种来补救。

## 证据产物

- 数据审计：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_20260813/data_audit.json`
- P0 gate：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_20260813/p0_gate.json`
- 标签分布图：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_20260813/p0/figures/risk_label_distributions.png`
- 事件支持度图：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_20260813/p0/figures/fold_event_support.png`
- fit-only 阈值图：`/Users/eurus/Code/kronos/Kronos/csj/runs/risk_control_v6/v6_p0_20260813/p0/figures/fold_tail_thresholds.png`

图表的分布轴仅为可读性裁剪；所有审计统计和 gate 均使用完整数值。
