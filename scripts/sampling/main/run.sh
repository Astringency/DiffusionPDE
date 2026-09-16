#!/usr/bin/env bash
set -euo pipefail
# Smooth comparisons at the two reported inference budgets.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a STEPS <<< "${STEP_LIST:-100 1000}"
for steps in "${STEPS[@]}"; do
  NUM_STEPS="${steps}" OUTPUT_DIR="${OUTPUT_ROOT:-outputs/main}/steps_${steps}" \
    bash "${HERE}/run_sweep.sh"
done
