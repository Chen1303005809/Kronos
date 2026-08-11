#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   csj/scripts/run_v4_cuda.sh audit --run-id v4_cuda_audit
#   csj/scripts/run_v4_cuda.sh p0 --run-id v4_cuda_p0
#   csj/scripts/run_v4_cuda.sh p1-ablation --run-id v4_cuda_p1_smoke --fold-id fold_00
#   csj/scripts/run_v4_cuda.sh p1-ablation --run-id v4_cuda_p1
#
# P2 and P3 remain guarded by the persisted Phase 1 gate and intentionally do
# not contain an implementation in this first V4 delivery.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

STAGE="${1:-p1-ablation}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec .venv/bin/python -m csj.v4.experiment "$STAGE" --device cuda "$@"
