"""Run a native Kronos probabilistic forecast from a futures JSON payload.

Example::

    .venv/bin/python examples/prediction_3day_json.py \
        --input csj/data/kline_rb8888.json \
        --output-dir outputs/rb8888_three_day \
        --sample-count 100

The input must use the provider payload format consumed by ``csj.utils.tool``:
``data`` records with ``O/H/L/C/V/VD/A/OI/TeD/TiD/T`` fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.three_day_json_forecast import (  # noqa: E402
    DEFAULT_CLIP,
    DEFAULT_LOOKBACK,
    DEFAULT_MAX_CONTEXT,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_NORMALIZATION_EPSILON,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TOKENIZER_REVISION,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TURNING_POINT_THRESHOLD,
    ThreeDayForecastConfig,
    _json_safe,
    load_native_kronos_models,
    prepare_json_three_day_forecast,
    run_native_three_day_forecast,
    write_forecast_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从期货服务 JSON 中选取三个完整交易日，使用原生 Kronos 进行多路径预测，"
            "并输出 10%%–90%% 区间、基准比较和图表。"
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="输入 JSON payload")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="产物目录；默认写入 outputs/three_day_<instrument>_<target_start>",
    )
    parser.add_argument(
        "--target-start-day",
        default=None,
        help="目标三日窗口起始 TiD，格式 YYYY-MM-DD；默认使用最后三个完整交易日",
    )
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    parser.add_argument(
        "--normalization-epsilon",
        type=float,
        default=DEFAULT_NORMALIZATION_EPSILON,
    )
    parser.add_argument(
        "--turning-point-threshold",
        type=float,
        default=DEFAULT_TURNING_POINT_THRESHOLD,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备：auto、cpu、cuda、mps 或 torch 支持的设备字符串",
    )
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "csj" / "artifacts" / "hf_cache",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="仅使用本地 Hugging Face 缓存，不下载模型",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prepared = prepare_json_three_day_forecast(
        args.input,
        target_start_day=args.target_start_day,
        lookback=args.lookback,
    )
    target_start = prepared.target_case.target_days[0].strftime("%Y%m%d")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / f"three_day_{prepared.instrument}_{target_start}"

    config = ThreeDayForecastConfig(
        lookback=args.lookback,
        sample_count=args.sample_count,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_context=args.max_context,
        clip=args.clip,
        normalization_epsilon=args.normalization_epsilon,
        turning_point_threshold=args.turning_point_threshold,
        seed=args.seed,
        device=args.device,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )

    print(
        f"Loading tokenizer={args.tokenizer_id}@{args.tokenizer_revision} "
        f"model={args.model_id}@{args.model_revision}"
    )
    model, tokenizer = load_native_kronos_models(
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        model_id=args.model_id,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    result = run_native_three_day_forecast(
        prepared,
        model,
        tokenizer,
        config=config,
    )
    paths = write_forecast_artifacts(result, output_dir)

    summary = {
        "instrument": prepared.instrument,
        "target_days": [day.strftime("%Y-%m-%d") for day in prepared.target_case.target_days],
        "pred_len": prepared.target_case.pred_len,
        "sample_count": config.sample_count,
        "device": str(result.device),
        "artifacts": paths,
        "close_mae": result.metrics["models"]["kronos"]["close_mae"],
        "interval_coverage": result.metrics["models"]["kronos"]["interval_coverage"],
    }
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
