#!/usr/bin/env bash
set -euo pipefail

# Kronos V3 CUDA launcher
#
# Usage:
#   RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh check
#   RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh audit
#   RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh p0-zero-shot
#   RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh p0
#   RUN_ID=v3_cuda bash csj/scripts/run_v3_cuda.sh p1
#   CONFIG=csj/configs/active_contract_panel_v3_partial.yaml RUN_ID=v3_partial_cuda \
#     bash csj/scripts/run_v3_cuda.sh full
#
# Modes:
#   check          CUDA environment + focused V3 data/probe tests + audit.
#   audit          Run only immutable-snapshot and 256/512 coverage audit.
#   p0-zero-shot   Target-only zero-shot path baseline for all V3 folds.
#   p0             Target-only CE-only baseline for all V3 folds.
#   p1             Frozen-backbone paired nearest-neighbour Probe.
#   full           check, P0 zero-shot, P0 CE-only, then P1 (P1 may correctly gate-stop).
#
# Environment:
#   PYTHON_BIN=.venv/bin/python
#   CONFIG=csj/configs/active_contract_panel_v3.yaml  # or the explicit partial config
#   RUN_ID=v3_cuda
#   CUDA_VISIBLE_DEVICES=0
#   ALLOW_MODEL_DOWNLOAD=1
#   HF_HUB_CACHE=/shared/kronos-hf-cache

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODE="${1:-help}"
PYTHON_BIN="/mnt/RohonDev1/miniconda3/envs/kronos/bin/python"
CONFIG="${CONFIG:-csj/configs/active_contract_panel_v3.yaml}"
RUN_ID="${RUN_ID:-v3_cuda}"

usage() {
  sed -n '4,24s/^# \{0,1\}//p' "$0"
}

if [[ "$MODE" == "help" || "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  usage
  exit 0
fi

cd "$REPO_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/csj/artifacts/hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kronos-matplotlib}"
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
  echo "[kronos-v3] stage=${stage} run_id=${RUN_ID} device=cuda"
  "$PYTHON_BIN" -u -m csj.v3.experiment "$stage" \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --device cuda \
    "${DOWNLOAD_ARGS[@]}"
}

run_check() {
  check_cuda
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest tests/test_v3_panel.py tests/test_active_contract_data.py -q
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
  p0-zero-shot)
    check_cuda
    run_stage p0-zero-shot
    ;;
  p0)
    check_cuda
    run_stage p0
    ;;
  p1)
    check_cuda
    run_stage p1
    ;;
  full)
    run_check
    run_stage p0-zero-shot
    run_stage p0
    run_stage p1
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
