#!/usr/bin/env bash
# Sourced by gpu*_two_runs.sh — do not run alone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GECCO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$GECCO_ROOT/.." && pwd)"
export PYTHONUNBUFFERED=1

# Default trial_df (override with TRIAL_DF=... before sourcing if needed)
TRIAL_DF="${TRIAL_DF:-$REPO_ROOT/transfer-interference/data/participants/trial_df.csv}"

RUN_DIR="${RUN_DIR:-$GECCO_ROOT/runs/overnight/$(date +%Y%m%d_%H%M)}"
mkdir -p "$RUN_DIR" "$GECCO_ROOT/figures/overnight"

run_gecco() {
  local name="$1"
  shift
  local log="$RUN_DIR/${name}.log"
  echo "=============================================="
  echo "$(date -Is) START $name"
  echo "LOG $log"
  echo "=============================================="
  {
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi -L 2>/dev/null || true
    cd "$GECCO_ROOT"
    python -m gecco.experiments.run_lba_figure "$@"
  } 2>&1 | tee "$log"
  echo "$(date -Is) END $name"
}

# “Big” overnight preset: stronger than notebook defaults
OVERNIGHT_BASE=(
  --trial-df "$TRIAL_DF"
  --population 44
  --generations 24
  --train-steps 180
  --participants-per-condition 3
  --max-trials-per-segment 28
  --batch-size 32
  --lr 0.02
  --hidden 32
  --plot-all-population
)
