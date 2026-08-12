#!/usr/bin/env bash
set -euo pipefail

# V5 CUDA launcher.  P2/P3/P4 intentionally remain unavailable until their
# preceding persisted gate has been synchronized and manually reviewed.
#
# Usage:
#   RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh check
#   RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh p0
#   RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh p1
#   RUN_ID=v5_cuda bash csj/scripts/run_v5_cuda.sh full
#
# Environment:
#   PYTHON_BIN=.venv/bin/python   # Python 3.12 with CUDA PyTorch
#   CONFIG=csj/configs/target_only_path_v5.yaml
#   RUN_ID=v5_cuda
#   CUDA_VISIBLE_DEVICES=0
#   ALLOW_MODEL_DOWNLOAD=1
#   HF_HUB_CACHE=/shared/kronos-hf-cache

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODE="${1:-help}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/RohonDev1/miniconda3/envs/kronos/bin/python}"
CONFIG="${CONFIG:-csj/configs/target_only_path_v5.yaml}"
RUN_ID="${RUN_ID:-v5_cuda}"

usage() {
  sed -n '4,23s/^# \{0,1\}//p' "$0"
}

if [[ "$MODE" == "help" || "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  usage
  exit 0
fi

cd "$REPO_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/csj/artifacts/hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kronos-v5-matplotlib}"
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
  echo "[kronos-v5] stage=${stage} run_id=${RUN_ID} device=cuda"
  "$PYTHON_BIN" -u -m csj.v5.experiment "$stage" \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --device cuda \
    "${DOWNLOAD_ARGS[@]}" \
    "$@"
}

run_check() {
  check_cuda
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest tests/test_v5_target_only.py tests/test_v5_plotting.py tests/test_v5_experiment.py -q
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
  p0-smoke)
    check_cuda
    run_stage p0 --fold-id fold_00 --max-cases-per-split 4
    ;;
  p0)
    check_cuda
    run_stage p0
    ;;
  p1)
    check_cuda
    run_stage p1-signal
    ;;
  full)
    run_check
    run_stage p0
    run_stage p1-signal
    ;;
  p2|p2-path-bridge)
    check_cuda
    run_stage p2-path-bridge
    ;;
  p3|p3-adapter)
    check_cuda
    run_stage p3-adapter
    ;;
  p4|p4-stability)
    check_cuda
    run_stage p4-stability
    ;;
  *)
    echo "Unknown V5 mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
