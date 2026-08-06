# V2 Phase 1：三交易日 Zero-shot 与数据基线

本报告覆盖预先锁定的全部完整 walk-forward folds。

本阶段是已观察历史上的 expanding walk-forward 开发比较，不是新的封存测试。
所有路径方向指标均由完整生成路径的 sampled median path 推导；第三日末方向是主指标。

## Pooled 三日端点指标

| Model | Cases | Day1 bal. acc. | Day2 bal. acc. | Day3 bal. acc. | Day3 return MAE |
|---|---:|---:|---:|---:|---:|
| zero_shot | 580 | 53.26% | 52.75% | 56.09% | 0.016673 |
| majority | 580 | 50.07% | 51.92% | 52.31% | 0.013358 |
| momentum | 580 | 47.91% | 47.51% | 49.99% | 0.026325 |
| persistence | 580 | 0.00% | 0.00% | 0.00% | 0.013358 |

## Zero-shot 分合约第三日指标

| Instrument | Cases | Day3 bal. acc. | Day3 accuracy | Return MAE | Return corr. |
|---|---:|---:|---:|---:|---:|
| i8888 | 290 | 59.07% | 56.89% | 0.020547 | 0.0237 |
| rb8888 | 290 | 53.17% | 52.94% | 0.012800 | 0.0945 |

## Zero-shot 路径与生成审计

- Mean return-path correlation: `0.0258`
- Mean z-normalized DTW: `0.656709`
- Mean return-space DTW: `0.008498`
- Mean slope-sign agreement: `46.14%`
- Raw non-finite value rate: `0.00%`
- Raw OHLC violation rate: `3.84%`
- Raw negative volume/amount rate: `0.00%`

## Zero-shot 逐 fold 稳定性

| Fold | Cases | Day3 bal. acc. | Return-path corr. | Z-normalized DTW |
|---|---:|---:|---:|---:|
| fold_00 | 116 | 71.05% | 0.2055 | 0.540661 |
| fold_01 | 116 | 54.26% | -0.0529 | 0.724659 |
| fold_02 | 116 | 49.69% | 0.0288 | 0.639523 |
| fold_03 | 116 | 55.17% | -0.0340 | 0.669849 |
| fold_04 | 116 | 54.71% | -0.0182 | 0.708853 |

![三交易日 return-space 路径示例](phase1_cuda_path_examples.png)

## 运行信息

- Elapsed seconds: `102.06`
- 原始及 OHLC/非负流量修复后的逐 sample 路径均保存在对应 run 目录。
- 报告没有把均值曲线当作唯一输出；主指标固定使用 sampled median path。
- 15/17 根组合在当前清洗后真实历史中未出现；构造性单元测试仍覆盖全部 5/7 三日组合。

完整机器可读指标：`phase1_cuda_metrics.json`。
