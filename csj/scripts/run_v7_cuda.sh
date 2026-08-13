#!/usr/bin/env bash
set -euo pipefail

# V7 CUDA launcher. P1 is the only currently executable post-P0 phase.
# It uses only the verified v7_p0_20260813_verified predecessor and never
# treats smoke output as a formal gate.
#
# Usage:
#   RUN_ID=v7_cuda bash csj/scripts/run_v7_cuda.sh check
#   RUN_ID=v7_cuda bash csj/scripts/run_v7_cuda.sh p1-smoke
#   RUN_ID=v7_cuda bash csj/scripts/run_v7_cuda.sh p1
#   RUN_ID=v7_cuda bash csj/scripts/run_v7_cuda.sh p1-baselines
#   RUN_ID=v7_cuda bash csj/scripts/run_v7_cuda.sh full
#
# Environment:
#   PYTHON_BIN=.venv/bin/python   # Python 3.12 with CUDA PyTorch
#   CONFIG=csj/configs/risk_control_v7.yaml
#   RUN_ID=v7_cuda
#   CUDA_VISIBLE_DEVICES=0
#   ALLOW_MODEL_DOWNLOAD=1
#   HF_HUB_CACHE=/shared/kronos-hf-cache

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODE="${1:-help}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/RohonDev1/miniconda3/envs/kronos/bin/python}"
CONFIG="${CONFIG:-csj/configs/risk_control_v7.yaml}"
RUN_ID="${RUN_ID:-v7_cuda}"

usage() {
  sed -n '4,23s/^# \{0,1\}//p' "$0"
}

if [[ "$MODE" == "help" || "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  usage
  exit 0
fi

cd "$REPO_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/csj/artifacts/hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kronos-v7-matplotlib}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DOWNLOAD_ARGS=()
if [[ "${ALLOW_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS=(--allow-model-download)
fi

check_cuda() {
  "$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
  "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count()})'
}

run_stage() {
  local stage="$1"
  shift
  echo "[kronos-v7] stage=${stage} run_id=${RUN_ID} device=cuda"
  "$PYTHON_BIN" -u -m csj.v7.experiment "$stage" \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --device cuda \
    "${DOWNLOAD_ARGS[@]}" \
    "$@"
}

run_check() {
  check_cuda
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest tests/test_v7_risk_control.py tests/test_v7_p1.py -q
  run_stage audit
}

case "$MODE" in
  check)
    run_check
    ;;
  audit)
    check_cuda
    run_stage audit
    ;;
  p1-smoke)
    check_cuda
    run_stage p1-path-bank --smoke --resume
    ;;
  p1)
    check_cuda
    run_stage p1-path-bank --resume
    run_stage p1-baselines
    ;;
  p1-baselines)
    # Use this after updating baseline/reporting code for an already complete
    # formal path bank.  It never regenerates the 5154 × 64 raw paths.
    check_cuda
    run_stage p1-baselines
    ;;
  full)
    run_check
    run_stage p1-path-bank --smoke --resume
    run_stage p1-path-bank --resume
    run_stage p1-baselines
    ;;
  p2|p2-train|p2-evaluate|p3|p3-calibrate|p4|p4-overlay|p5|p5-freeze)
    echo "V7 ${MODE} is intentionally unavailable until the synchronized formal P1 gate is reviewed." >&2
    exit 2
    ;;
  *)
    echo "Unknown V7 mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
