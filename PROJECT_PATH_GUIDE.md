# Kronos 项目目录路径指引

> 目的：给新会话提供一张“先看哪里、不要看哪里、不同实验版本如何区分”的地图。除非任务明确要求，不要从 `csj/runs/`、`csj/results/` 或 `webui/prediction_results/` 开始递归扫描。

## 0. 项目根目录与新会话启动顺序

本仓库的 Git 根目录是：

```text
/Users/eurus/Code/kronos/Kronos
```

外层 `/Users/eurus/Code/kronos` 是工作区容器，不是项目根目录。进入新会话后按以下顺序恢复上下文：

1. 先阅读本文件，确定任务属于哪条路径。
2. 阅读 [`AGENTS.md`](AGENTS.md)，遵守项目级实验与评估约束。
3. 涉及期货具体合约、邻居、面板或生产有效性的任务，再阅读 [`CONTEXT.md`](CONTEXT.md)。
4. 只打开对应路径下的入口文件、配置和报告；不要先遍历所有运行产物。
5. 修改代码后优先运行对应测试文件；模型评估相关任务必须检查图表产物是否生成。

最小恢复命令：

```bash
cd /Users/eurus/Code/kronos/Kronos
sed -n '1,240p' PROJECT_PATH_GUIDE.md
sed -n '1,200p' AGENTS.md
```

## 1. 总体结构

```text
Kronos/
├── model/                     # Kronos 核心模型、Tokenizer、Predictor
├── examples/                  # 面向使用者的预测、回测和数据示例
├── figures/                   # README 与文档使用的静态图片
├── finetune/                  # Qlib/A 股示例微调流水线
├── finetune_csv/              # 自定义 CSV 数据的独立微调流水线
├── csj/                       # 期货研究、训练、评估和版本化实验体系
├── webui/                     # Flask Web UI 与其预测结果
├── tests/                     # 回归测试、期货管线测试和版本测试
├── README.md                  # 项目对外介绍与基础预测/微调说明
├── CONTEXT.md                 # 具体合约期货领域术语与边界
├── AGENTS.md                  # 本项目必须遵守的工作约束
├── requirements.txt           # 主项目依赖
└── .venv/                     # Python 3.12 虚拟环境，不是源码
```

## 2. 顶层目录说明

### `/model`：核心模型库（最重要的源码入口）

这里是所有预测和微调流程共同依赖的模型实现，不按实验版本复制。

- `model/kronos.py`：`KronosTokenizer`、`Kronos`、`KronosPredictor`、自回归推理和归一化相关核心逻辑。
- `model/module.py`：Transformer、层归一化、层级 token embedding、二值球面量化等底层模块。
- `model/__init__.py`：公开导出和模型类查找入口。

修改模型结构、token、推理、输入输出形状时，从这里开始；先看 [`tests/test_kronos_regression.py`](tests/test_kronos_regression.py)。不要把 `csj/v3` 或 `csj/v4` 的实验融合头误认为基础模型本体。

### `/examples`：可运行示例，不是公共库

用于展示如何加载模型、预测、批量预测、接入 AkShare、做回测或运行 GUI 示例。常用入口：

- `prediction_example.py`：最小预测与绘图示例。
- `prediction_wo_vol_example.py`：没有 volume/amount 时的预测示例。
- `prediction_batch_example.py`：批量预测。
- `prediction_cn_markets_day.py`、`prediction_akshare_2024-2025.py`：中国市场数据获取与预测。
- `run_backtest_kronos.py`：Qlib 回测入口。
- `examples/yuce/`：历史预测/回测输出、报告 JSON 和 PNG；属于示例产物，不是模型实现。

### `/figures`：静态展示资源

存放 README、论文式说明或项目介绍使用的图片，例如模型概览、预测示例和回测示例。除非任务是更新文档图片，否则不要从这里寻找业务逻辑。

### `/finetune`：原始 Qlib/A 股微调路径

这是仓库原始的、面向 Qlib 数据的微调示例，与 `finetune_csv/` 分开维护。

- `finetune/config.py`：数据路径、窗口、模型路径、训练和回测配置；运行前通常需要按本机环境修改。
- `finetune/qlib_data_preprocess.py`：Qlib 数据预处理、切分和 pickle 数据集生成。
- `finetune/dataset.py`：微调数据集定义。
- `finetune/train_tokenizer.py`：Tokenizer 微调。
- `finetune/train_predictor.py`：Predictor 微调。
- `finetune/qlib_test.py`：加载微调结果并进行 Qlib 预测/回测验证。
- `finetune/utils/`：训练辅助工具。

任务关键词是“Qlib、A 股、pickle、backtest、原始微调流程”时走这条路径。先读 [`README.md`](README.md) 的微调章节，再读 `finetune/config.py`。

### `/finetune_csv`：自定义 CSV 微调路径

这是另一套相对独立的 CSV 训练实现，包含 Tokenizer 微调和基础 Predictor 微调，并支持顺序训练/分布式环境。

- `finetune_csv/config_loader.py`：配置加载和实验参数。
- `finetune_csv/finetune_tokenizer.py`：CSV 数据上的 Tokenizer 训练。
- `finetune_csv/finetune_base_model.py`：CSV 数据上的基础模型训练。
- `finetune_csv/train_sequential.py`：先 Tokenizer、后基础模型的顺序入口。
- `finetune_csv/configs/`：CSV 微调 YAML 配置。
- `finetune_csv/data/`：示例 CSV 原始数据；数据可能较大，先看配置引用的文件名。
- `finetune_csv/examples/`：由 CSV 数据生成的示例图。
- `finetune_csv/README.md`、`README_CN.md`：该流水线的使用说明，中文任务优先读 `README_CN.md`。

任务是“自有 CSV、K 线 CSV、Tokenizer/基础模型顺序微调”时只看这条路径，不要混入 `finetune/` 的 Qlib 数据假设。

### `/csj`：期货研究与实验主目录

`csj` 是当前开发最密集的研究体系。它同时保留历史 V1/V2、具体合约面板 V3、观察 cohort V4、目标单流路径 V5、三品种主动风控 V6 和多品种主动风控 V7，因此必须先按版本分流，不能把所有脚本当作同一条管线。

#### `/csj` 目录下的通用源码

- `config.py`：通用期货实验配置读取、路径解析和约束校验。
- `futures_data.py`：连续合约/多合约数据载入、窗口、案例构造和时间切分。
- `training.py`：Predictor 训练、checkpoint 和训练状态处理。
- `evaluation.py`：预测案例、设备解析和评估推理。
- `metrics.py`：方向指标、基线、配对比较、bootstrap 等指标逻辑。
- `reporting.py`：通用报告和图表输出。
- `experiment.py`：V1/V2 通用小时级实验命令入口。
- `three_day_experiment.py`、`three_day_training.py`、`three_day_evaluation.py`、`three_day_reporting.py`：三交易日预测路径的拆分入口；其固定协议见 V2 文档。
- `trend_model.py`：三日方向辅助头/趋势相关模型逻辑。
- `evaluation_plotter.py`：跨版本共享的评估绘图入口；模型评估任务优先检查它及对应测试。
- `active_contract_data.py`：按快照保存具体活跃合约清单和原始 K 线。
- `__init__.py`：包入口。

#### `/csj/configs`：实验配置，而不是结果

每个 YAML 都决定数据、模型 revision、窗口、fold、训练协议和产物根目录。先根据任务选配置，再追代码：

- `futures_hourly.yaml`：V1 单日/小时级连续合约路径。
- `futures_3day_trend.yaml`：V2 三日趋势/路径生成，使用 `three_day_experiment`。
- `active_contract_panel_v3.yaml`：V3 严格完整面板配置，正式结论要求完整历史面板。
- `active_contract_panel_v3_partial.yaml`：V3 partial panel 探索配置；结果只能作探索性证据。
- `observed_contract_cohort_v4.yaml`：V4 当前观察 cohort 研究配置，固定 `production_eligible: false`。
- `target_only_path_v5.yaml`：V5 计划配置，尚未实施；正式实现前以 `V5_IMPLEMENTATION_PLAN.md` 为准。
- `risk_control_v6.yaml`：V6 主动风控预注册配置；P0 标签与审计 runner 已实施，P1-P5 仍受 gate 保护。
- `risk_control_v7.yaml`：V7 多品种主动风控配置；固定覆盖度筛选、预测原点日切分及五折 P0 gate。

如果任务涉及训练协议、设备、学习率、窗口或输出位置，先读对应 YAML 和版本文档，不要从已有 JSON 结果反推配置。

#### `/csj/data`：输入数据与不可变快照

- `kline_rb8888.json`、`kline_i8888.json`：连续合约小时 K 线，主要供 V1/V2 历史实验使用；它们不是具体交割月合约生产面板。
- `active_contract_snapshots/`：具体合约的快照归档。每个时间目录通常包含 `manifest.json`、原始 payload 和处理后的 K 线；V3/V4 的面板/cohort 数据从这里构建。
- 快照目录是数据证据，不要覆盖旧快照。要判断某次 V3/V4 实验使用了什么数据，先看其 `manifest.json`、`snapshot_id`、哈希和配置。
- K 线服务不能查询已交割合约。补充数据时只能从当时确认仍活跃的具体合约向前采集并生成新快照；禁止枚举旧交割月做历史回填。合约在活跃期已经归档的不可变快照仍可按其 provenance 使用。

#### `/csj/v3`：V3 活跃具体合约面板

V3 使用具体交割月合约，不使用连续合约换月逻辑。主要文件：

- `panel_data.py`：具体合约、面板案例、时间对齐、归一化和 walk-forward fold。
- `config.py`：V3 配置和运行约束。
- `p0.py`：目标单流 P0 基线。
- `pair_probe.py`：冻结主干的 target-only/pair P1 probe、采样和配对评估。
- `experiment.py`：V3 的 `audit`、`p0-zero-shot`、`p0`、`p1` 等命令入口。
- `evaluation_plotter.py`：V3 兼容绘图包装；通用绘图契约仍要对照 `csj/evaluation_plotter.py`。
- `backfill_direction_plots.py`：为已有记录补生成方向图。

配套文档按顺序看：`V3_TRAINING.md` → `V3_EVALUATION_METRICS.md` → `ACTIVE_CONTRACT_PANEL_TRAINING_STRATEGY.md`。V3 的 `partial` 结果不能写成正式历史活跃面板结论；P1 gate 通过前不要实现/运行 P2。

#### `/csj/v4`：V4 observed cohort

V4 不声称重建历史活跃合约面板，而是在当前冻结 cohort 中选择预测时刻可用且上下文完整的最近邻，因此结果范围是 `retrospective_observed_cohort`，且 `production_eligible: false`。

- `cohort_data.py`：冻结 manifest、记录 SHA-256、构建 observed cohort、选择最近可用邻居和构造案例。
- `config.py`：V4 配置校验和路径规则。
- `experiment.py`：V4 audit、P0、P1 ablation、粒度 gate 等实验入口；P2/P3 受持久化 gate 保护。

配套设计文档是 `V4_IMPLEMENTATION_PLAN.md`。如果问题是“为什么 V4 不等同于历史生产面板”，先读该文档和 `cohort_data.py` 的模块注释。

V4 已于 P1 冻结：shared 与 per-product 的 pair arm 均未通过邻居增量 gate，`allows_p2: false`。不得实现或运行 V4 P2/P3。

#### V5：target-only 方向引导路径（计划阶段）

V5 去除邻居输入，使用不要求 `has_pair` 的全量目标案例。它先复验 shared target-only 方向信号，再以固定路径库重加权作为分类到完整路径的桥接 gate，最后才允许训练方向条件路径适配器。

- 当前唯一权威设计文档：`csj/V5_IMPLEMENTATION_PLAN.md`。
- 计划源码目录：`csj/v5/`；当前尚未实施，不能把计划接口写成已完成能力。
- 计划配置：`csj/configs/target_only_path_v5.yaml`。
- 计划结果：`csj/results/target_only_path_v5/`，范围固定为 `retrospective_observed_contracts`、`production_eligible: false`。

#### V6：target-only 主动风控（P0 已实施，gate 未通过）

V6 不再以第三日涨跌为最终目标，而是从目标具体合约的历史和冻结 Kronos 路径中估计未来三日多头/空头最大不利波动风险。风险预测与仓位执行是两个独立 seam；overlay 只能缩小外部基础仓位，不能开仓、反向或增加杠杆。

- 当前权威设计：`csj/v6/IMPLEMENTATION_PLAN.md`。
- 目录状态：`csj/v6/` 已实现配置校验、风险标签、五折 P0 审计、gate、报告与图表；P1-P5 尚未实施。
- 预注册配置：`csj/configs/risk_control_v6.yaml`。
- P0 结果：`csj/results/risk_control_v6/`。当前支持度/预测日原子性 gate 未通过，范围固定为 `retrospective_observed_contracts`、`production_eligible: false`。

#### V7：多品种 target-only 主动风控（P0 已通过）

V7 保留 V6 的三日不利波动标签和门槛，但用纯数据覆盖度规则扩展到 21 个商品品种，并将 walk-forward 切分键改为 `origin_trading_day`。V6 的失败证据不被覆盖。

- 入口与设计：`csj/v7/README.md`、`csj/v7/experiment.py`。
- 配置：`csj/configs/risk_control_v7.yaml`。
- P0 结果：`csj/results/risk_control_v7/`，逐运行证据在 `csj/runs/risk_control_v7/`。
- 当前权威运行 `v7_p0_20260813_verified` 的五折 gate 已通过，但仍是 `production_eligible: false` 的回看证据；P1 尚未实施，不能把 P0 通过写成稳定 edge。

#### `/csj/utils`：外部数据服务与小工具

- `kline_client.py`：K 线服务客户端、请求和超时边界。
- `tool.py`：共享特征常量和通用小工具。

修改数据采集、socket/进程超时或原始响应处理时先看这里和 `active_contract_data.py`，不要直接改快照产物。

#### `/csj/scripts`：CUDA 运行包装器

- `run_v2_cuda.sh`：V2 三日实验；支持 check、pilot、full、phase1/2/3 和 resume。
- `run_v3_cuda.sh`：V3 面板实验；支持 audit、P0、P1。
- `run_v4_cuda.sh`：V4 observed cohort 实验；支持 audit、P0、P1 ablation。当前 gate 已固定拒绝 P2/P3，不得绕过。

V5 runner 尚未实施；实施后使用独立的 `run_v5_cuda.sh`，不得复用 V4 run/results 目录。

脚本只是编排器，真正的业务逻辑在相应 Python 模块。运行前先确认 `PYTHON_BIN`、`CONFIG`、`RUN_ID` 和 CUDA 环境；不要把 V2、V3、V4 的 run 目录混用。

#### `/csj/legacy`：历史入口说明

只存放旧路径分类和迁移说明。目前应先读 `csj/legacy/README.md` 判断旧脚本属于 V1/V2 哪一支，再决定是否需要打开历史代码。不要把 legacy 文档当作当前实验协议。

#### `/csj/artifacts`：模型缓存/运行依赖产物

通常包含 Hugging Face 模型缓存（例如 `hf_cache/`），并在 `.gitignore` 中排除。它不是源码，也不是实验结论；只有遇到模型加载、revision、离线运行或缓存损坏时才检查。

#### `/csj/runs`：逐次运行的详细产物（默认跳过）

这里保存耗时运行的 resolved config、checkpoint、训练历史、逐案例预测、fold 目录和日志，文件量很大。典型结构：

```text
csj/runs/<strategy>/<run_id>/
├── resolved_config.json      # 该次运行实际解析后的配置
├── fold_00/ ...              # walk-forward 折
├── phase1/ phase2/ phase3/   # V2/V3/V4 阶段产物（依版本而定）
└── evaluation/ ...           # 逐案例评估、图表和 JSON
```

定位已有实验时，先从 `csj/results/<strategy>/` 的报告/汇总 JSON 得到 `run_id`，再只打开对应 run 的 `resolved_config.json` 和需要的 fold/phase。不要递归读取整个 `runs` 目录，也不要仅复制汇总 JSON 后声称可以恢复训练；恢复通常需要完整 run 目录。

#### `/csj/results`：实验汇总、报告和图表

这是比 `runs` 更适合新会话优先阅读的结果层。子目录与策略对应：

- `futures_hourly/`：V1 小时级结果。
- `futures_3day_trend/`：V2 三日趋势结果和阶段报告。
- `active_contract_panel_v3/`：V3 严格配置结果。
- `active_contract_panel_v3_partial/`：V3 partial 探索结果。
- `observed_contract_cohort_v4/`：V4 observed cohort 结果。
- `target_only_path_v5/`：V5 计划结果目录；尚未产生正式结果。

先读 `REPORT.md`、`*_REPORT.md`、`metrics.json`、`*_metrics.json`、`data_audit.json`、`split_summary.json`；只有需要追溯案例、checkpoint 或绘图时才进入对应 `runs`。

### `/webui`：Flask 预测界面

这是面向交互式预测的独立 Web UI，不是 `csj` 研究实验入口。

- `app.py`：Flask 路由、数据加载、模型加载、预测、Plotly 图表和结果保存。
- `run.py`：依赖检查、启动 Flask、自动打开浏览器。
- `start.sh`：Shell 启动方式；默认使用系统 `python3`，如需遵守项目 Python 3.12 约定，优先手动使用 `.venv/bin/python` 启动。
- `templates/index.html`：前端页面模板。
- `requirements.txt`：Web UI 额外依赖。
- `prediction_results/`：每次 Web UI 预测保存的 JSON；是可追溯产物，不是源码，默认跳过批量扫描。

若任务是网页布局、API、交互预测或 Plotly 图表，先看 `webui/README.md`、`app.py` 和 `templates/index.html`；不要把 Web UI 的数据目录假设套到 `csj/data`。

### `/tests`：验证边界与回归保护

- `test_kronos_regression.py`：核心模型数值/回归行为。
- `test_futures_pipeline.py`、`test_futures_metrics.py`、`test_three_day_trend.py`：通用/V2 期货数据、指标和三日路径。
- `test_active_contract_data.py`、`test_v3_panel.py`：快照与 V3 面板数据不泄漏约束。
- `test_v3_evaluation_plotter.py`、`test_v4_evaluation_plotter.py`：评估图表契约。
- `test_v4_experiment.py`、`test_v4_observed_cohort.py`：V4 gate、cohort 和实验隔离。
- `tests/data/`：回归输入、期望输出和小型测试数据；不是生产数据。

测试选择原则：改 `model/` 先跑核心回归；改 `csj/v3` 或 `csj/v4` 先跑对应版本测试；改绘图器先跑 v3/v4 plotter 测试；不要因为已有运行产物存在而跳过测试。

## 3. 版本与路径分流

| 任务/关键词 | 首先阅读 | 主要源码 | 主要结果 |
|---|---|---|---|
| 基础模型、Tokenizer、Predictor | `README.md`、`model/` | `model/kronos.py`、`model/module.py` | `tests/test_kronos_regression.py` |
| 普通示例预测/回测 | `README.md` | `examples/` | `examples/yuce/` 或生成的本地结果 |
| Qlib/A 股微调 | README 微调章节、`finetune/config.py` | `finetune/` | 由配置中的路径决定 |
| 自定义 CSV 微调 | `finetune_csv/README_CN.md` | `finetune_csv/` | 由 CSV 配置中的 save path 决定 |
| V1 小时级连续合约 | `csj/README.md`、`csj/configs/futures_hourly.yaml` | `csj/experiment.py` 及通用模块 | `csj/results/futures_hourly/` |
| V2 三日趋势 | `csj/HANDOFF_3DAY_TREND_V2.md`、`csj/V3_TRAINING.md`、`futures_3day_trend.yaml` | `csj/three_day_*.py`、`trend_model.py` | `csj/results/futures_3day_trend/` |
| V3 具体合约面板 | `csj/V3_TRAINING.md`、`V3_EVALUATION_METRICS.md` | `csj/v3/`、`csj/active_contract_data.py` | `csj/results/active_contract_panel_v3*/` |
| V4 observed cohort | `csj/V4_IMPLEMENTATION_PLAN.md` | `csj/v4/` | `csj/results/observed_contract_cohort_v4/` |
| V5 target-only 路径 | `csj/V5_IMPLEMENTATION_PLAN.md` | `csj/v5/`（计划） | `csj/results/target_only_path_v5/`（计划） |
| V6 主动风控 | `csj/v6/IMPLEMENTATION_PLAN.md` | `csj/v6/`（P0） | `csj/results/risk_control_v6/`（P0 failed） |
| Web UI | `webui/README.md` | `webui/app.py`、`webui/templates/` | `webui/prediction_results/` |

注意：V2 文档文件名中仍可能出现 `V3`（例如三日趋势阶段的历史命名），应以入口模块和 YAML 的 `experiment.version` 为准，不要只按文件名判断版本。

## 4. 哪些目录默认不要扫描

以下目录不是新会话恢复上下文的首选入口：

- `.git/`：Git 内部对象和日志；只在需要提交历史、分支或变更审计时使用。
- `.venv/`：Python 3.12 虚拟环境；不读源码，不提交。
- `.pytest_cache/`、任意 `__pycache__/`：测试/字节码缓存。
- `csj/artifacts/`：模型缓存；只在模型加载问题时查。
- `csj/runs/`：大量逐运行产物；先用 results 汇总定位具体 run。
- `webui/prediction_results/`：历史 Web UI 输出；只按时间或文件名定位。
- 任何 `*.pt`、`*.pth`、`*.ckpt`、`*.bin`、大型 CSV/JSON：先看配置、manifest 或报告，不要批量读入上下文。

## 5. 产物目录的判读规则

### `runs` 与 `results` 的区别

```text
源码/配置  ──运行──>  csj/runs/<strategy>/<run_id>/  ──汇总/绘图──>  csj/results/<strategy>/
                         详细证据                         新会话首选阅读
```

- `runs` 回答“这次运行具体做了什么、checkpoint 和逐案例记录是什么”。
- `results` 回答“这次实验的结论、指标、审计状态和图表是什么”。
- `resolved_config.json` 是判断实际参数的第一证据；不要只相信命令行默认值。
- `data_audit.json`、`split_summary.json` 用来判断数据覆盖、切分和泄漏边界。
- 模型评估不能只看数值 JSON；按 `AGENTS.md`，还要确认预测/真实值关系的图表产物存在。

### V3/V4 的结果范围

- `active_contract_panel_v3_partial/`：partial panel，只能支持探索性结论。
- `observed_contract_cohort_v4/`：retrospective observed cohort，固定不是生产有效性证据。
- 只有看到报告中明确的快照完整性、gate 状态和 `production_eligible`，才能判断结果能否进入下一阶段。

## 6. 最短阅读路径

### 只想了解项目

```text
README.md → model/ → examples/ → csj/README.md
```

### 修改基础预测逻辑

```text
AGENTS.md → model/kronos.py → model/module.py → examples/prediction_example.py
         → tests/test_kronos_regression.py
```

### 修改通用期货训练/评估

```text
AGENTS.md → csj/README.md → 对应 csj/configs/*.yaml
         → csj/futures_data.py / training.py / evaluation.py / metrics.py / reporting.py
         → 对应 tests/test_futures_*.py 或 test_three_day_trend.py
```

### 修改 V3/V4/V5 具体合约研究

```text
AGENTS.md → CONTEXT.md → 对应版本训练/计划文档
         → 对应 config → csj/v3/、csj/v4/ 或 csj/v5/
         → 对应 tests/ → results 汇总 → 必要时追踪 runs
```

### 排查某次实验结果

```text
对应 csj/results/<strategy>/ 报告
→ metrics/data_audit/split_summary
→ resolved_config.json
→ 具体 csj/runs/<strategy>/<run_id>/fold_*/phase*/evaluation/
```

### 修改 Web UI

```text
webui/README.md → webui/app.py → webui/templates/index.html
               → webui/run.py / start.sh → webui/prediction_results/（仅按需）
```

## 7. 维护约定

新增目录或实验版本时，同时更新本文件的：

1. 总体结构树；
2. 对应目录说明；
3. 版本与路径分流表；
4. 运行产物路径和默认跳过列表。

如果实验主方案发生重大变化，除更新路径说明外，还要按 `AGENTS.md` 的版本管理规则更新方案主版本或 phase，并在对应计划/报告中写明变更原因。
