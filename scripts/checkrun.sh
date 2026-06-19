#!/usr/bin/env bash
set -euo pipefail

# End-to-end DiffusionPDE check run:
#   1. load each PDE dataset through data.load.PDEloader
#   2. train a short diffusion run and save network-snapshot-*.pkl
#   3. sample with generate_pde.py using a temporary config override
#   4. evaluate the generated result and aggregate metrics
#
# Default command:
#   bash scripts/checkrun.sh
#
# Common overrides:
#   PDE=heat bash scripts/checkrun.sh
#   PDE_LIST="heat wave advection_diffusion steady_heat_conduction" bash scripts/checkrun.sh
#   RUN_TRAIN=0 CHECKPOINT_ROOT=output/pretrained bash scripts/checkrun.sh
#   DATA_ROOT=/large_storage/zhangxf/PDEdata OUTPUT_ROOT=outputs/checkrun SAMPLE_STEPS=10 bash scripts/checkrun.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_DATA_ROOT="/large_storage/zhangxf/PDEdata"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
DATA_ROOT="${DATA_ROOT%/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/checkrun}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${OUTPUT_ROOT}/train}"
SAMPLE_OUTPUT_DIR="${SAMPLE_OUTPUT_DIR:-${OUTPUT_ROOT}/samples}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/metrics}"
CONFIG_OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-${OUTPUT_ROOT}/configs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-output/pretrained}"

DEFAULT_PDE_LIST=(
  darcy
  poisson
  helmholtz
  nsnonbounded
  burger
  reaction_diffusion
  shallow_water
  heat
  wave
  advection_diffusion
  steady_heat_conduction
)

if [[ -z "${PDE_LIST:-}" ]]; then
  if [[ -n "${PDE:-}" ]]; then
    PDE_LIST="${PDE}"
  else
    PDE_LIST="${DEFAULT_PDE_LIST[*]}"
  fi
fi
read -r -a PDES <<< "${PDE_LIST}"
if (( ${#PDES[@]} == 0 )); then
  echo "PDE_LIST resolved to an empty list." >&2
  exit 2
fi

RUN_LOAD="${RUN_LOAD:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

LOAD_SIZE="${LOAD_SIZE:-1}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_ROOT}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-12400}"
ARCH="${ARCH:-ddpmpp}"
PRECOND="${PRECOND:-edm}"
TRAIN_BATCH="${TRAIN_BATCH:-1}"
TRAIN_BATCH_GPU="${TRAIN_BATCH_GPU:-1}"
DURATION="${DURATION:-0.001}"
EMA="${EMA:-0.05}"
LR="${LR:-0.001}"
TICK="${TICK:-1}"
SNAP="${SNAP:-1}"
DUMP="${DUMP:-50}"
WORKERS="${WORKERS:-1}"
SEED="${SEED:-0}"

PROBLEM_LIST="${PROBLEM_LIST:-both}"
SAMPLE_BATCH="${SAMPLE_BATCH:-1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-1}"
SAMPLE_DEVICE="${SAMPLE_DEVICE:-cuda}"
OBS_SIZE="${OBS_SIZE:-500}"
OFFSET="${OFFSET:-0}"

CONDA_ENV="${CONDA_ENV:-diffusionpde}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_PREFIX_FOR_ENV="$(conda env list | awk -v env="${CONDA_ENV}" '$1 == env {print $NF; exit}')"
    if [[ -n "${CONDA_PREFIX_FOR_ENV}" && -x "${CONDA_PREFIX_FOR_ENV}/bin/python" ]]; then
      PYTHON_BIN="${CONDA_PREFIX_FOR_ENV}/bin/python"
    fi
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUTPUT_ROOT}" "${TRAIN_OUTPUT_ROOT}" "${SAMPLE_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}" "${CONFIG_OUTPUT_DIR}" "${LOG_DIR}"

echo "== DiffusionPDE check run =="
echo "repo: ${ROOT_DIR}"
echo "python: ${PYTHON_BIN}"
echo "pdes: ${PDES[*]}"
echo "data_root: ${DATA_ROOT}"
echo "output_root: ${OUTPUT_ROOT}"
echo "phases: load=${RUN_LOAD} train=${RUN_TRAIN} sample=${RUN_SAMPLE} eval=${RUN_EVAL}"

upper_name() {
  local value="$1"
  value="${value^^}"
  printf '%s\n' "${value//[^A-Z0-9]/_}"
}

config_name_for_pde() {
  case "$1" in
    burger) printf '%s\n' "burgers" ;;
    nsnonbounded) printf '%s\n' "ns-nonbounded" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

display_name_for_pde() {
  case "$1" in
    burger) printf '%s\n' "Burgers" ;;
    darcy) printf '%s\n' "Darcy" ;;
    poisson) printf '%s\n' "Poisson" ;;
    helmholtz) printf '%s\n' "Helmholtz" ;;
    nsnonbounded) printf '%s\n' "NS-NonBounded" ;;
    reaction_diffusion) printf '%s\n' "Reaction-Diffusion" ;;
    shallow_water) printf '%s\n' "Shallow-water" ;;
    heat) printf '%s\n' "Heat" ;;
    wave) printf '%s\n' "Wave" ;;
    advection_diffusion) printf '%s\n' "Advection-Diffusion" ;;
    steady_heat_conduction) printf '%s\n' "Steady-Heat-Conduction" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

config_for_pde() {
  local pde="$1"
  local key
  key="$(config_name_for_pde "${pde}")"
  printf '%s\n' "${CONFIG_DIR}/${key}.yaml"
}

pretrained_name_for_pde() {
  case "$1" in
    burger) printf '%s\n' "pretrained-burgers.pkl" ;;
    nsnonbounded) printf '%s\n' "pretrained-ns-nonbounded.pkl" ;;
    *) printf 'pretrained-%s.pkl\n' "$1" ;;
  esac
}

pde_data_dirs() {
  local pde="$1"
  printf '%s\n' "${DATA_ROOT}/${pde}"
  case "${pde}" in
    burger) printf '%s\n' "${DATA_ROOT}/burgers" ;;
    nsnonbounded) printf '%s\n' "${DATA_ROOT}/ns-nonbounded" ;;
    shallow_water) printf '%s\n' "${DATA_ROOT}/shallow-water" ;;
    reaction_diffusion) printf '%s\n' "${DATA_ROOT}/reaction-diffusion" ;;
  esac
  printf '%s\n' "${DATA_ROOT}/test1125"
  printf '%s\n' "${DATA_ROOT}"
}

test_data_patterns_for_pde() {
  case "$1" in
    darcy)
      printf '%s\n' "darcy_test_*.mat" "*darcy*test*.mat"
      ;;
    poisson)
      printf '%s\n' "poisson_test_*.mat" "*poisson*test*.mat"
      ;;
    helmholtz)
      printf '%s\n' "helmholtz_test_*.mat" "*helmholtz*test*.mat"
      ;;
    nsnonbounded)
      printf '%s\n' "nsnonbounded_test_*.mat" "*nsnonbounded*test*.mat" "nsnonbounded_*-*-*-*.mat"
      ;;
    burger)
      printf '%s\n' "burger_test_*.mat" "burgers_test_*.mat" "burger_*.mat" "*burger*test*.mat"
      ;;
    reaction_diffusion)
      printf '%s\n' "reaction_diffusion_test_*.h5" "*reaction_diffusion*test*.h5"
      ;;
    shallow_water)
      printf '%s\n' "shallow_water_test_*.h5" "2d_swe_test_*.h5" "*shallow*water*test*.h5" "*swe*test*.h5"
      ;;
    heat)
      printf '%s\n' "heat_test_*.h5" "*heat*test*.h5"
      ;;
    wave)
      printf '%s\n' "wave_test_*.h5" "*wave*test*.h5"
      ;;
    advection_diffusion)
      printf '%s\n' "advection_diffusion_test_*.h5" "*advection*diffusion*test*.h5"
      ;;
    steady_heat_conduction)
      printf '%s\n' "steady_heat_conduction_test_*.h5" "*steady*heat*conduction*test*.h5"
      ;;
    *)
      printf '%s\n' "*${1}*test*"
      ;;
  esac
}

find_test_data_for_pde() {
  local pde="$1"
  local pde_dir
  local pattern
  local found
  while IFS= read -r pde_dir; do
    [[ -d "${pde_dir}" ]] || continue
    while IFS= read -r pattern; do
      found="$(
        find "${pde_dir}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' \
          | sort -nr \
          | head -n 1 \
          | cut -d' ' -f2-
      )"
      if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return
      fi
    done < <(test_data_patterns_for_pde "${pde}")
  done < <(pde_data_dirs "${pde}")
  printf '\n'
}

yaml_value() {
  local path="$1"
  local dotted_key="$2"
  "${PYTHON_BIN}" - "${path}" "${dotted_key}" <<'PY'
import sys
import yaml
from pathlib import Path

data = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = data
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

test_data_for_pde() {
  local pde="$1"
  local config="$2"
  local per_pde_key
  local discovered
  local config_path
  per_pde_key="TEST_DATA_$(upper_name "${pde}")"
  if [[ -n "${!per_pde_key:-}" ]]; then
    printf '%s\n' "${!per_pde_key}"
    return
  fi
  if [[ -n "${TEST_DATA_PATH:-}" ]]; then
    printf '%s\n' "${TEST_DATA_PATH}"
    return
  fi
  discovered="$(find_test_data_for_pde "${pde}")"
  if [[ -n "${discovered}" ]]; then
    printf '%s\n' "${discovered}"
    return
  fi
  config_path="$(yaml_value "${config}" "data.datapath")"
  if [[ -n "${config_path}" && "${DATA_ROOT}" != "${DEFAULT_DATA_ROOT}" && "${config_path}" == "${DEFAULT_DATA_ROOT}"* ]]; then
    printf '%s\n' "${config_path/${DEFAULT_DATA_ROOT}/${DATA_ROOT}}"
  else
    printf '%s\n' "${config_path}"
  fi
}

latest_snapshot() {
  local train_dir="$1"
  find "${train_dir}" -type f -name "network-snapshot-*.pkl" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

checkpoint_for_pde() {
  local pde="$1"
  local train_dir="$2"
  local config="$3"
  local per_pde_key
  local snapshot
  local fallback
  per_pde_key="CHECKPOINT_$(upper_name "${pde}")"
  if [[ -n "${!per_pde_key:-}" ]]; then
    printf '%s\n' "${!per_pde_key}"
    return
  fi
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    printf '%s\n' "${CHECKPOINT_PATH}"
    return
  fi
  snapshot="$(latest_snapshot "${train_dir}")"
  if [[ -n "${snapshot}" ]]; then
    printf '%s\n' "${snapshot}"
    return
  fi
  fallback="${CHECKPOINT_ROOT}/$(pretrained_name_for_pde "${pde}")"
  if [[ -f "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
    return
  fi
  yaml_value "${config}" "test.pre-trained"
}

write_run_config() {
  local base_config="$1"
  local out_config="$2"
  local pde="$3"
  local data_path="$4"
  local checkpoint="$5"
  "${PYTHON_BIN}" - "${base_config}" "${out_config}" "${pde}" "${data_path}" "${checkpoint}" \
    "${SAMPLE_OUTPUT_DIR}" "${SAMPLE_DEVICE}" "${SAMPLE_STEPS}" "${SAMPLE_BATCH}" "${OFFSET}" "${OBS_SIZE}" <<'PY'
import sys
from pathlib import Path
import yaml

base, out, pde, data_path, checkpoint, output_dir, device, steps, batch, offset, obs_size = sys.argv[1:]
config = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
config.setdefault("data", {})
config.setdefault("test", {})
config.setdefault("generate", {})
config.setdefault("output", {})
config["data"]["datapath"] = data_path
config["data"]["offset"] = int(offset)
config["data"]["obs_size"] = int(obs_size)
config["test"]["pre-trained"] = checkpoint
config["test"]["iterations"] = int(steps)
config["generate"]["device"] = device
config["generate"]["batch_size"] = int(batch)
config["output"]["file_path"] = output_dir
config["output"]["save"] = True
config["output"]["return"] = False
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

run_load_check() {
  local pde="$1"
  local log_path="${LOG_DIR}/load_${pde}.log"
  echo "== Loading data for ${pde} =="
  "${PYTHON_BIN}" - "${pde}" "${TRAIN_DATA_PATH}" "${LOAD_SIZE}" 2>&1 <<'PY' | tee "${log_path}"
import sys
from data.load import PDEloader
from data.transform import PDEtransform

pde, data_path, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
loader = PDEloader(pde)
data, labels, metadata = loader.load_data(data_path, size=size, return_metadata=True)
transformed = PDEtransform(data).transform()
print({
    "pde": pde,
    "data_shape": tuple(data.shape),
    "transformed_shape": tuple(transformed.shape),
    "labels_shape": tuple(labels.shape),
    "metadata_keys": sorted(metadata),
})
PY
}

run_train() {
  local pde="$1"
  local train_dir="$2"
  local port="$3"
  mkdir -p "${train_dir}"
  echo "== Training ${pde} =="
  torchrun --master_addr=127.0.0.1 --master_port="${port}" --nproc_per_node="${NPROC_PER_NODE}" \
    train.py \
    --pde="${pde}" \
    --outdir="${train_dir}" \
    --data="${TRAIN_DATA_PATH}" \
    --cond=0 \
    --arch="${ARCH}" \
    --precond="${PRECOND}" \
    --batch="${TRAIN_BATCH}" \
    --batch-gpu="${TRAIN_BATCH_GPU}" \
    --tick="${TICK}" \
    --snap="${SNAP}" \
    --dump="${DUMP}" \
    --duration="${DURATION}" \
    --ema="${EMA}" \
    --lr="${LR}" \
    --workers="${WORKERS}" \
    --seed="${SEED}" \
    --nosubdir 2>&1 | tee "${LOG_DIR}/train_${pde}.log"
}

find_new_result() {
  local marker="$1"
  find "${SAMPLE_OUTPUT_DIR}" -type f -newer "${marker}" \( -name "*_results.pkl" -o -name "*_results.mat" -o -name "ns_nonbounded_results.mat" \) -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

run_sampling_and_eval() {
  local pde="$1"
  local run_config="$2"
  local problem="$3"
  local marker
  local result_path
  mkdir -p "${SAMPLE_OUTPUT_DIR}" "${SAMPLE_OUTPUT_DIR}/${problem}" "${EVAL_OUTPUT_DIR}/${pde}_${problem}"
  marker="${OUTPUT_ROOT}/.sample_marker_${pde}_${problem}"
  touch "${marker}"

  echo "== Sampling ${pde} / ${problem} =="
  "${PYTHON_BIN}" -u generate_pde.py \
    --config="${run_config}" \
    --problem="${problem}" \
    --batch="${SAMPLE_BATCH}" \
    --step_size="${SAMPLE_STEPS}" 2>&1 | tee "${LOG_DIR}/sample_${pde}_${problem}.log"

  result_path="$(find_new_result "${marker}")"
  if [[ -z "${result_path}" || ! -f "${result_path}" ]]; then
    echo "No sampling result was created for ${pde}/${problem} under ${SAMPLE_OUTPUT_DIR}" >&2
    exit 1
  fi
  echo "result: ${result_path}"

  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "== Evaluating ${pde} / ${problem} =="
    "${PYTHON_BIN}" -u evaluate_results.py \
      --config="${run_config}" \
      --result="${result_path}" \
      --problem="${problem}" \
      --output-dir="${EVAL_OUTPUT_DIR}/${pde}_${problem}" 2>&1 | tee "${LOG_DIR}/eval_${pde}_${problem}.log"
  fi
}

run_one_pde() {
  local pde="$1"
  local index="$2"
  local config
  local train_dir
  local data_path
  local checkpoint
  local run_config
  local problem

  config="$(config_for_pde "${pde}")"
  if [[ ! -f "${config}" ]]; then
    echo "Missing config for ${pde}: ${config}" >&2
    exit 2
  fi

  echo
  echo "============================================================"
  echo "PDE: ${pde}"
  echo "base_config: ${config}"
  echo "============================================================"

  if [[ "${RUN_LOAD}" == "1" ]]; then
    run_load_check "${pde}"
  fi

  train_dir="${TRAIN_OUTPUT_ROOT}/${pde}"
  if [[ "${RUN_TRAIN}" == "1" ]]; then
    run_train "${pde}" "${train_dir}" "$((MASTER_PORT_BASE + index))"
  fi

  if [[ "${RUN_SAMPLE}" == "1" ]]; then
    data_path="$(test_data_for_pde "${pde}" "${config}")"
    if [[ ! -e "${data_path}" ]]; then
      echo "Test data path does not exist for ${pde}: ${data_path}" >&2
      exit 2
    fi
    checkpoint="$(checkpoint_for_pde "${pde}" "${train_dir}" "${config}")"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Checkpoint does not exist for ${pde}: ${checkpoint}" >&2
      exit 2
    fi
    run_config="${CONFIG_OUTPUT_DIR}/$(config_name_for_pde "${pde}").yaml"
    write_run_config "${config}" "${run_config}" "${pde}" "${data_path}" "${checkpoint}"

    read -r -a problems <<< "${PROBLEM_LIST}"
    for problem in "${problems[@]}"; do
      run_sampling_and_eval "${pde}" "${run_config}" "${problem}"
    done
  fi
}

idx=0
for pde in "${PDES[@]}"; do
  run_one_pde "${pde}" "${idx}"
  idx=$((idx + 1))
done

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo
  echo "== Aggregating metrics =="
  "${PYTHON_BIN}" -u aggregate_eval.py "${EVAL_OUTPUT_DIR}" --output-dir="${OUTPUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/aggregate.log"
fi

echo
echo "== Done =="
echo "output_root: ${OUTPUT_ROOT}"
echo "logs: ${LOG_DIR}"
