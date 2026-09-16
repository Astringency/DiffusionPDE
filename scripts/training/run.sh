#!/usr/bin/env bash
set -euo pipefail
# Existing EDM training recipe. DATA_ROOT contains the physical PDE training shards.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PDES <<< "${PDE_LIST:-${PDE:-poisson helmholtz darcy nsnonbounded burger}}"
for pde in "${PDES[@]}"; do
  case "${pde}" in poisson|helmholtz|darcy|nsnonbounded|burger) ;; *) echo "Unknown PDE: ${pde}" >&2; exit 2;; esac
  key="TRAIN_DATA_${pde^^}"
  dataset="${!key:-${DATA_ROOT:-${ROOT}/datasets}}"
  cmd=("${PYTHON_BIN}" -m torch.distributed.run --standalone
       --nproc_per_node="${NPROC_PER_NODE:-1}" train.py --pde="${pde}"
       --outdir="${OUTDIR:-outputs/training}/${pde}" --data="${dataset}"
       --cond=0 --arch=ddpmpp --batch="${BATCH:-64}" --batch-gpu="${BATCH_GPU:-32}"
       --tick=10 --snap=50 --dump=100 --duration="${DURATION:-20}" --ema=0.05)
  if [[ -n "${RESUME:-}" ]]; then
    [[ ${#PDES[@]} -eq 1 ]] || { echo 'RESUME requires exactly one PDE.' >&2; exit 2; }
    cmd+=(--resume="${RESUME}")
  fi
  if [[ "${DRY_RUN:-false}" == true ]]; then
    printf ' %q' "${cmd[@]}"; printf '\n'
  else
    [[ -d "${dataset}" ]] || { echo "Missing PDE training data: ${dataset}" >&2; exit 2; }
    "${cmd[@]}"
  fi
done
