#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODE="${1:-help}"

usage() {
  sed -n '12,35s/^# \{0,1\}//p' "$0"
}

# Kronos V2 CUDA launcher
#
# Usage:
#   bash csj/scripts/run_v2_cuda.sh pilot
#   bash csj/scripts/run_v2_cuda.sh full
#
# Modes:
#   check    Run CPU regressions, CUDA data audit, and consistency check.
#   smoke    Run check plus small Phase 1/2 CUDA smoke tests.
#   pilot    Recommended first run: check, fold-00 baseline, smoke, fold-00 train.
#   full     Run check, all-fold baseline, smoke, and all-fold Phase 2 train.
#   phase1  Run only the all-fold CUDA zero-shot baseline.
#   phase2  Run only all-fold Phase 2; requires matching Phase 1 cache.
#   phase3-smoke  Run the two-batch CE + direction-head smoke test.
#   phase3-pilot  Run fold-00 Phase 3; requires matching Phase 1/2 caches.
#   phase3  Run all-fold Phase 3; requires matching Phase 1/2 caches.
#   resume-pilot  Resume an interrupted fold-00 Phase 2 pilot.
#   resume  Resume an interrupted all-fold Phase 2 run.
#   resume-phase3  Resume an interrupted all-fold Phase 3 run.
#   help     Show this message.
#
# Environment:
#   RUN_ID=cuda_v2                 Shared run id; keep it unchanged for resume/full.
#   PYTHON_BIN=.venv/bin/python    Python 3.12 virtualenv interpreter.
#   CONFIG=csj/configs/futures_3day_trend.yaml
#   ALLOW_MODEL_DOWNLOAD=1         Allow first-run download of pinned model revisions.
#   HF_HUB_CACHE=/path/to/cache    Optional shared Hugging Face cache.
#   CUDA_VISIBLE_DEVICES=0         Optional GPU selection.

if [[ "$MODE" == "help" || "$MODE" == "--help" || "$MODE" == "-h" ]]; then
  usage
  exit 0
fi

cd "$REPO_ROOT"

PYTHON_BIN="/mnt/RohonDev1/miniconda3/envs/kronos/bin/python"
CONFIG="${CONFIG:-csj/configs/futures_3day_trend.yaml}"
RUN_ID="${RUN_ID:-cuda_v2}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/csj/artifacts/hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kronos-matplotlib}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"



"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count()})'

run_stage() {
  local stage="$1"
  echo "[kronos-v2] stage=${stage} run_id=${RUN_ID} device=cuda"
  "$PYTHON_BIN" -u -m csj.three_day_experiment \
    "$stage" \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --device cuda \
    "${DOWNLOAD_ARGS[@]}"
}

run_tests() {
  echo "[kronos-v2] CPU regression suite"
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m pytest tests -q
}

run_check() {
  run_tests
  run_stage phase0
}

case "$MODE" in
  check)
    run_check
    ;;
  smoke)
    run_check
    run_stage phase1_smoke
    run_stage phase2_smoke
    ;;
  pilot)
    run_check
    run_stage phase1_pilot
    run_stage phase2_smoke
    run_stage phase2_pilot
    ;;
  full)
    run_check
    run_stage phase1
    run_stage phase2_smoke
    run_stage phase2
    ;;
  phase1)
    run_stage phase1
    ;;
  phase2|resume)
    run_stage phase2
    ;;
  phase3-smoke)
    run_stage phase3_smoke
    ;;
  phase3-pilot)
    run_stage phase3_pilot
    ;;
  phase3|resume-phase3)
    run_stage phase3
    ;;
  resume-pilot)
    run_stage phase2_pilot
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
