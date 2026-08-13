# V7 多品种主动风控 P0

当前状态：`v7_p0_20260813_verified` 的五折 P0 gate 已通过；P1 尚未实施。

V7 是 V6 数据门控失败后的新主版本。V6 的失败证据保持不变；V7 不通过降低事件阈值或偷看 evaluation 修复它，而是改变两个已证实不可行的协议前提：

- 数据从 `i/jm/rb` 扩展到按纯覆盖度规则冻结的多品种具体合约池；
- walk-forward 改为按 `origin_trading_day` 原子切分，并把研究起点固定为原 V6 的 `2025-10-15`，避免单个长历史品种把稀疏前段误计入全局日历。

产品进入池的规则固定为：当前统一快照内至少有一个成功具体合约，小时 bar 数不少于 `1300`，首根 bar 不晚于 `2025-10-01`，且能按冻结的 5/7 小时完整交易日协议产生三日案例。规则不读取风险标签、validation 或 evaluation 表现。最终训练池为 21 个品种；`i/jm/rb` 继续承担每品种 pooled evaluation 门控。

V7 P0 保留 V6 的三日不利波动标签、fit-only P80 阈值、每折 fit 每侧 100 事件、validation/evaluation 每侧 20 事件要求；`i/jm/rb` 五折 evaluation 汇总门槛也保持每品种每侧 20 事件。

运行：

```bash
.venv/bin/python -m csj.v7.experiment audit --run-id v7_p0_20260813
```

V7 仍是 `production_eligible: false` 的回看式 observed-contract 研究。P0 通过只表示数据与切分足以进入路径风险研究，不表示模型已经存在稳定 edge。
