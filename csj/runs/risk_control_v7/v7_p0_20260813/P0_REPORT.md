# V7 P0 多品种主动风控数据门控

- 结论：**通过**
- 快照：`20260813T105916+0800`
- 产品数：21
- 案例数：5703
- 预测原点日：200

## Fold 支持度

| fold | split | cases | long | short |
|---|---:|---:|---:|---:|
| fold_00 | fit | 890 | 178 | 178 |
| fold_00 | inner_validation | 558 | 98 | 102 |
| fold_00 | evaluation | 613 | 45 | 198 |
| fold_01 | fit | 1446 | 290 | 290 |
| fold_01 | inner_validation | 601 | 49 | 200 |
| fold_01 | evaluation | 666 | 147 | 106 |
| fold_02 | fit | 2037 | 408 | 408 |
| fold_02 | inner_validation | 660 | 189 | 80 |
| fold_02 | evaluation | 714 | 207 | 122 |
| fold_03 | fit | 2692 | 539 | 539 |
| fold_03 | inner_validation | 707 | 135 | 132 |
| fold_03 | evaluation | 756 | 240 | 62 |
| fold_04 | fit | 3387 | 678 | 678 |
| fold_04 | inner_validation | 748 | 249 | 97 |
| fold_04 | evaluation | 788 | 136 | 150 |

## 失败项

- 无。

本结果仍是回看式 observed-contract 证据，不能直接宣称生产有效。
产品池只按上市时间和 bar 数冻结，未使用风险标签或 evaluation 表现筛选。
