#!/usr/bin/env bash
# GPU 4 — seed replication (primary statistical band)
# Usage (from anywhere):
#   source /path/to/repo/gpu_block.sh 4
#   bash gecco/scripts/overnight/gpu4_two_runs.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_gecco gpu4_seed0 "${OVERNIGHT_BASE[@]}" --seed 0 --init-scale 0.1 \
  --out "$GECCO_ROOT/figures/overnight/gpu4_seed0.png"

run_gecco gpu4_seed1 "${OVERNIGHT_BASE[@]}" --seed 1 --init-scale 0.1 \
  --out "$GECCO_ROOT/figures/overnight/gpu4_seed1.png"

echo "Done GPU4 runs. Logs: $RUN_DIR"
