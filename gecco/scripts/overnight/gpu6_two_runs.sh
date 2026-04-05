#!/usr/bin/env bash
# GPU 6 — init-scale axis (rich vs lazy evolved pathway), fixed seed
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_gecco gpu6_init0.001 "${OVERNIGHT_BASE[@]}" --seed 0 --init-scale 0.001 \
  --out "$GECCO_ROOT/figures/overnight/gpu6_init0.001.png"

run_gecco gpu6_init2 "${OVERNIGHT_BASE[@]}" --seed 0 --init-scale 2.0 \
  --out "$GECCO_ROOT/figures/overnight/gpu6_init2.png"

echo "Done GPU6 runs. Logs: $RUN_DIR"
