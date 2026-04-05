#!/usr/bin/env bash
# GPU 7 — (1) add Same condition  (2) longer cyclic pattern + more trials/segment
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_gecco gpu7_near_far_same "${OVERNIGHT_BASE[@]}" --seed 0 --init-scale 0.1 \
  --conditions near far same \
  --out "$GECCO_ROOT/figures/overnight/gpu7_near_far_same.png"

run_gecco gpu7_longpattern "${OVERNIGHT_BASE[@]}" --seed 0 --init-scale 0.1 \
  --cyclic-pattern ABABABAB \
  --max-trials-per-segment 36 \
  --train-steps 200 \
  --out "$GECCO_ROOT/figures/overnight/gpu7_longpattern_ABx4.png"

echo "Done GPU7 runs. Logs: $RUN_DIR"
