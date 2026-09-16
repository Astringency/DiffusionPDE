#!/usr/bin/env bash
set -euo pipefail
# The paper's DiffusionPDE sensitivity study is its paired error/time trajectory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FM_ROOT="${FM_ROOT:-${ROOT}/../FM4PDE}"
[[ -f "${FM_ROOT}/scripts/sampling/ablations/run_traces.sh" ]] || { echo "Set FM_ROOT to the FM4PDE repository." >&2; exit 2; }
export DIFFUSION_ROOT="${ROOT}"
exec bash "${FM_ROOT}/scripts/sampling/ablations/run_traces.sh" "$@"
