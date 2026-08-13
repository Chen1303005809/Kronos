# V7 P1 冻结路径与简单风险基线门控

- 结论：**未通过**
- 原始路径缓存：5154 个唯一 case
- 全局有效路径比例：37.6482%
- path-quality abstain：5082 个 case
- P1 的基线只使用 inner validation 选择；未输出 outer evaluation 预测性能指标。

## 每折选中的基线

| fold | selected baseline | validation cases |
|---|---|---:|
| fold_00 | unavailable: no_finite_validation_selection_brier | 4 |
| fold_01 | fit_product_event_rate | 18 |
| fold_02 | fit_global_event_rate | 10 |
| fold_03 | fit_global_event_rate | 22 |
| fold_04 | unavailable: no_path_quality_eligible_inner_validation_cases | 0 |

## 失败项

- `global_finite_ohlc_path_validity`
- `validation_baseline_selection_available_by_fold`
- `fold_02:inner_validation:short_raw_path_risk_non_degenerate`
- `fold_04:inner_validation:long_raw_path_risk_non_degenerate`
- `fold_04:inner_validation:short_raw_path_risk_non_degenerate`

本结果是 observed-contract 回看研究证据，`production_eligible: false`。
