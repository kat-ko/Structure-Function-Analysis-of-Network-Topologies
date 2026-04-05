#!/usr/bin/env bash
# GPU 5 — seed replication (continued)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_gecco gpu5_seed2 "${OVERNIGHT_BASE[@]}" --seed 2 --init-scale 0.1 \
  --out "$GECCO_ROOT/figures/overnight/gpu5_seed2.png"

run_gecco gpu5_seed3 "${OVERNIGHT_BASE[@]}" --seed 3 --init-scale 0.1 \
  --out "$GECCO_ROOT/figures/overnight/gpu5_seed3.png"

echo "Done GPU5 runs. Logs: $RUN_DIR"
