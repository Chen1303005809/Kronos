# V7 多品种主动风控

当前状态：`v7_p0_20260813_verified` 的五折 P0 gate 已通过；P1 CUDA 路径库与验证基线已实施，等待正式 CUDA 运行结果。

P1-P5 的唯一权威实施与跨会话交接文档是 [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)。当前 CLI 实现了 `audit`、`p1-path-bank` 和 `p1-baselines`；P2-P5 仍是受上游 gate 保护的待实现接口。

V7 是 V6 数据门控失败后的新主版本。V6 的失败证据保持不变；V7 不通过降低事件阈值或偷看 evaluation 修复它，而是改变两个已证实不可行的协议前提：

- 数据从 `i/jm/rb` 扩展到按纯覆盖度规则冻结的多品种具体合约池；
- walk-forward 改为按 `origin_trading_day` 原子切分，并把研究起点固定为原 V6 的 `2025-10-15`，避免单个长历史品种把稀疏前段误计入全局日历。

产品进入池的规则固定为：当前统一快照内至少有一个成功具体合约，小时 bar 数不少于 `1300`，首根 bar 不晚于 `2025-10-01`，且能按冻结的 5/7 小时完整交易日协议产生三日案例。规则不读取风险标签、validation 或 evaluation 表现。最终训练池为 21 个品种；`i/jm/rb` 继续承担每品种 pooled evaluation 门控。

V7 P0 保留 V6 的三日不利波动标签、fit-only P80 阈值、每折 fit 每侧 100 事件、validation/evaluation 每侧 20 事件要求；`i/jm/rb` 五折 evaluation 汇总门槛也保持每品种每侧 20 事件。

CUDA 运行顺序：

```bash
RUN_ID=v7_p1_cuda bash csj/scripts/run_v7_cuda.sh check
RUN_ID=v7_p1_cuda bash csj/scripts/run_v7_cuda.sh p1-smoke
RUN_ID=v7_p1_cuda bash csj/scripts/run_v7_cuda.sh p1
```

`p1-smoke` 固定为 8 个 case × 4 条路径，只验证模型加载、原始 path/hidden 的形状、原子 shard、resume 和确定性抽检。它会生成 `p1_smoke_gate.json`，但该 gate 固定不解锁 P2。正式 `p1` 固定缓存 5154 个唯一 case × 64 条路径，按 target length 分 shard；随后仅用 inner validation 选择每折简单风险基线，并产生 cache 覆盖、有效路径、风险概率、PR、reliability 与概率分箱事件率图。

正式 P1 只承认冻结的 `v7_p0_20260813_verified` 输入及其 SHA-256。若 P1 gate 未通过，停止，不调采样参数或后续风险头。

V7 仍是 `production_eligible: false` 的回看式 observed-contract 研究。P0 通过只表示数据与切分足以进入路径风险研究，不表示模型已经存在稳定 edge。
