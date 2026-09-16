#!/usr/bin/env bash
set -euo pipefail

# DiffusionPDE MAIN sampling sweep.
#
# Defaults:
#   - Poisson, Helmholtz, Darcy, and non-bounded Navier-Stokes
#     run forward, inverse, and both tasks.
#   - Burgers runs both with 500 random space-time points and five complete
#     time slices, saved as separate jobs over the full trajectory.
#   - Each PDE/task job samples offsets [0, NUM_SAMPLES).
#
# Examples:
#   bash scripts/sampling/main/run_sweep.sh
#   PLAN_ONLY=true bash scripts/sampling/main/run_sweep.sh
#   NUM_SAMPLES=10 NUM_STEPS=20 DEVICE=cuda:0 \
#     bash scripts/sampling/main/run_sweep.sh
#   DEVICE_LIST="cuda:0 cuda:1" PARALLEL=true MAX_PARALLEL_TASKS=2 \
#     bash scripts/sampling/main/run_sweep.sh
#   DATA_ROOT=/path/to/PDEdata CHECKPOINT_ROOT=/path/to/pretrained \
#     OUTPUT_DIR=outputs/MAIN1000_100 bash scripts/sampling/main/run_sweep.sh
#
# Per-PDE data/checkpoint overrides are also supported, for example:
#   DATA_HELMHOLTZ=/path/to/helmholtz_test.mat
#   CHECKPOINT_HELMHOLTZ=/path/to/pretrained-helmholtz.pkl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
BURGER_SENSOR_MODES="${BURGER_SENSOR_MODES:-random time_slices}"
BURGER_TIME_SLICES="${BURGER_TIME_SLICES:-5}"
BURGER_MASK_SEED="${BURGER_MASK_SEED:-1}"
BURGER_SAMPLE_SEED="${BURGER_SAMPLE_SEED:-${SAMPLE_SEED:-20260913}}"
PDE_LIST="${PDE_LIST:-poisson helmholtz darcy nsnonbounded burger}"
TASK_LIST="${TASK_LIST:-forward inverse both}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/MAIN1000_100}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/output/pretrained}"
DEVICE="${DEVICE:-cuda}"
DEVICE_LIST="${DEVICE_LIST:-${DEVICE}}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
CONDA_ENV="${CONDA_ENV:-fm4pdebaseline}"
PYTHON_BIN="${PYTHON_BIN:-}"
PARALLEL="${PARALLEL:-false}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
RESUME="${RESUME:-true}"
EVALUATE="${EVALUATE:-true}"
AGGREGATE="${AGGREGATE:-true}"
PLAN_ONLY="${PLAN_ONLY:-false}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if python -c 'import h5py, scipy, torch, yaml' >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_PREFIX_FOR_ENV="$(conda env list | awk -v env="${CONDA_ENV}" '$1 == env {print $NF; exit}')"
        if [[ -n "${CONDA_PREFIX_FOR_ENV}" && -x "${CONDA_PREFIX_FOR_ENV}/bin/python" ]]; then
            PYTHON_BIN="${CONDA_PREFIX_FOR_ENV}/bin/python"
        fi
    fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_CONFIG_DIR="${OUTPUT_DIR}/.sample_sweep/configs"
LOG_DIR="${OUTPUT_DIR}/logs"
METRICS_DIR="${OUTPUT_DIR}/metrics"

is_true() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        false|0|no|off) return 1 ;;
        *) echo "Invalid boolean value: $1" >&2; exit 2 ;;
    esac
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
}

upper_name() {
    local value="${1^^}"
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
        poisson) printf '%s\n' "Poisson" ;;
        helmholtz) printf '%s\n' "Helmholtz" ;;
        darcy) printf '%s\n' "Darcy" ;;
        nsnonbounded) printf '%s\n' "NS-NonBounded" ;;
        burger) printf '%s\n' "Burgers" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

default_data_path() {
    local pde="$1"
    [[ -n "${DATA_ROOT}" ]] || return 0
    case "${pde}" in
        poisson) printf '%s\n' "${DATA_ROOT%/}/poisson/poisson_test_10000-128-128_smooth.mat" ;;
        helmholtz) printf '%s\n' "${DATA_ROOT%/}/helmholtz/helmholtz_test_10000-128-128_smooth.mat" ;;
        darcy) printf '%s\n' "${DATA_ROOT%/}/darcy/darcy_test_10000-128-128_smooth.mat" ;;
        nsnonbounded) printf '%s\n' "${DATA_ROOT%/}/nsnonbounded/nsnonbounded_test_10000-128-128-10_smooth.mat" ;;
        burger) printf '%s\n' "${DATA_ROOT%/}/burgers/burger_test_10000-128-128_smooth.mat" ;;
    esac
}

default_checkpoint_path() {
    local pde="$1"
    [[ -n "${CHECKPOINT_ROOT}" ]] || return 0
    case "${pde}" in
        nsnonbounded) printf '%s\n' "${CHECKPOINT_ROOT%/}/pretrained-ns-nonbounded.pkl" ;;
        burger) printf '%s\n' "${CHECKPOINT_ROOT%/}/pretrained-burgers.pkl" ;;
        *) printf '%s\n' "${CHECKPOINT_ROOT%/}/pretrained-${pde}.pkl" ;;
    esac
}

value_override_for_pde() {
    local prefix="$1"
    local pde="$2"
    local variable_name
    variable_name="${prefix}_$(upper_name "${pde}")"
    printf '%s\n' "${!variable_name:-}"
}

prepare_config() {
    local pde="$1"
    local task="$2"
    local device="$3"
    local base_config="$4"
    local run_config="$5"
    local sample_dir="$6"
    local sensor_mode="$7"
    local data_path
    local checkpoint

    data_path="$(value_override_for_pde DATA "${pde}")"
    checkpoint="$(value_override_for_pde CHECKPOINT "${pde}")"
    [[ -n "${data_path}" ]] || data_path="$(default_data_path "${pde}")"
    [[ -n "${checkpoint}" ]] || checkpoint="$(default_checkpoint_path "${pde}")"

    "${PYTHON_BIN}" - "${base_config}" "${run_config}" "${task}" "${sample_dir}" \
        "${device}" "${NUM_STEPS}" "${NUM_OBS}" "${SAMPLE_SEED}" \
        "${data_path}" "${checkpoint}" "${sensor_mode}" \
        "${BURGER_TIME_SLICES}" "${BURGER_MASK_SEED}" "${BURGER_SAMPLE_SEED}" <<'PY'
import sys
from pathlib import Path

import yaml

base, out, task, sample_dir, device, steps, obs, seed, data_path, checkpoint, sensor_mode, time_slices, mask_seed, burger_seed = sys.argv[1:]
config = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
config.setdefault("data", {})
config.setdefault("test", {})
config.setdefault("generate", {})
config.setdefault("output", {})
if data_path:
    config["data"]["datapath"] = data_path
if checkpoint:
    config["test"]["pre-trained"] = checkpoint
config["data"]["obs_size"] = int(obs)
config["test"]["iterations"] = int(steps)
config["generate"]["device"] = device
config["generate"]["batch_size"] = 1
config["generate"]["seed"] = int(seed)
if sensor_mode:
    config["data"]["sensor_mode"] = sensor_mode
    config["data"]["num_time_slices"] = int(time_slices)
    config["data"]["mask_seed"] = int(mask_seed)
    config["generate"]["seed"] = int(burger_seed)
    config["generate"]["seed_per_sample"] = True
config["generate"]["problem"] = task
config["output"]["file_path"] = sample_dir
config["output"]["save"] = True
config["output"]["return"] = False
out_path = Path(out)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

config_value() {
    local config="$1"
    local dotted_key="$2"
    "${PYTHON_BIN}" - "${config}" "${dotted_key}" <<'PY'
import sys
from pathlib import Path

import yaml

value = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

validate_assets() {
    local config="$1"
    local data_path
    local checkpoint
    data_path="$(config_value "${config}" data.datapath)"
    checkpoint="$(config_value "${config}" test.pre-trained)"
    if [[ ! -f "${data_path}" ]]; then
        echo "Missing test data: ${data_path}" >&2
        return 2
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        return 2
    fi
}

result_count() {
    local pde="$1"
    local result_dir="$2"
    local display_name
    display_name="$(display_name_for_pde "${pde}")"
    find "${result_dir}" -maxdepth 1 -type f -name "${display_name}_*_results.pkl" -printf '.\n' 2>/dev/null | wc -l
}

run_job() {
    local pde="$1"
    local task="$2"
    local device="$3"
    local config="$4"
    local sample_root="$5"
    local sensor_mode="$6"
    local result_dir="${sample_root}/${task}"
    local metrics_dir="${METRICS_DIR}/${pde}${sensor_mode:+/${sensor_mode}}/${task}"
    local log_path="${LOG_DIR}/${pde}${sensor_mode:+_${sensor_mode}}_${task}.log"
    local existing=0
    local start_offset=0
    local sample_count="${NUM_SAMPLES}"

    mkdir -p "${result_dir}" "${metrics_dir}" "${LOG_DIR}"
    if is_true "${RESUME}"; then
        existing="$(result_count "${pde}" "${result_dir}")"
    fi

    if ! validate_assets "${config}"; then
        return 2
    fi

    {
        echo "[${pde}/${task}] device=${device} samples=${NUM_SAMPLES} steps=${NUM_STEPS} layout=${sensor_mode:-random}"
        if is_true "${RESUME}" && (( existing >= NUM_SAMPLES )); then
            echo "[${pde}/${task}] sampling skipped: found ${existing} result files"
        else
            if is_true "${RESUME}" && (( existing > 0 )); then
                start_offset="${existing}"
                sample_count="$((NUM_SAMPLES - existing))"
                echo "[${pde}/${task}] sampling resumed: found ${existing} result files; starting at offset ${start_offset}"
            fi
            if ! "${PYTHON_BIN}" -u generate_pde.py \
                --config="${config}" \
                --problem="${task}" \
                --batch="${sample_count}" \
                --start_offset="${start_offset}" \
                --step_size="${NUM_STEPS}"; then
                echo "[${pde}/${task}] sampling failed" >&2
                exit 1
            fi
        fi

        if is_true "${EVALUATE}"; then
            if ! "${PYTHON_BIN}" -u evaluate_results.py \
                --config="${config}" \
                --result="${result_dir}" \
                --problem="${task}" \
                --output-dir="${metrics_dir}"; then
                echo "[${pde}/${task}] evaluation failed" >&2
                exit 1
            fi
        fi
    } 2>&1 | tee "${log_path}"
}

require_positive_integer NUM_SAMPLES "${NUM_SAMPLES}"
require_positive_integer NUM_STEPS "${NUM_STEPS}"
require_positive_integer NUM_OBS "${NUM_OBS}"
require_positive_integer BURGER_TIME_SLICES "${BURGER_TIME_SLICES}"
require_positive_integer MAX_PARALLEL_TASKS "${MAX_PARALLEL_TASKS}"
if (( NUM_STEPS < 2 )); then
    echo "NUM_STEPS must be at least 2 for the EDM time-step schedule." >&2
    exit 2
fi

read -r -a PDES <<< "${PDE_LIST}"
read -r -a TASKS <<< "${TASK_LIST}"
read -r -a DEVICES <<< "${DEVICE_LIST}"
if (( ${#PDES[@]} == 0 || ${#TASKS[@]} == 0 || ${#DEVICES[@]} == 0 )); then
    echo "PDE_LIST, TASK_LIST, and DEVICE_LIST must not be empty." >&2
    exit 2
fi

mkdir -p "${RUN_CONFIG_DIR}" "${LOG_DIR}" "${METRICS_DIR}"

JOB_PDES=()
JOB_TASKS=()
JOB_DEVICES=()
JOB_CONFIGS=()
JOB_SAMPLE_ROOTS=()
JOB_SENSOR_MODES=()
job_index=0

for pde in "${PDES[@]}"; do
    case "${pde}" in
        poisson|helmholtz|darcy|nsnonbounded|burger) ;;
        *) echo "Unsupported MAIN PDE: ${pde}" >&2; exit 2 ;;
    esac

    base_config="${CONFIG_DIR}/$(config_name_for_pde "${pde}").yaml"
    if [[ ! -f "${base_config}" ]]; then
        echo "Missing base config: ${base_config}" >&2
        exit 2
    fi

    pde_tasks=("${TASKS[@]}")
    pde_modes=("")
    if [[ "${pde}" == "burger" ]]; then
        pde_tasks=(both)
        read -r -a pde_modes <<< "${BURGER_SENSOR_MODES}"
        if (( ${#pde_modes[@]} == 0 )); then
            echo "BURGER_SENSOR_MODES must not be empty." >&2
            exit 2
        fi
        for sensor_mode in "${pde_modes[@]}"; do
            case "${sensor_mode}" in
                random|time_slices|sensor_columns) ;;
                *) echo "Unsupported Burgers sensor mode: ${sensor_mode}" >&2; exit 2 ;;
            esac
        done
    fi

    for task in "${pde_tasks[@]}"; do
        case "${task}" in
            forward|inverse|both) ;;
            *) echo "Unsupported task: ${task}" >&2; exit 2 ;;
        esac
        for sensor_mode in "${pde_modes[@]}"; do
            device="${DEVICES[$((job_index % ${#DEVICES[@]}))]}"
            run_config="${RUN_CONFIG_DIR}/${pde}${sensor_mode:+_${sensor_mode}}_${task}.yaml"
            sample_root="${OUTPUT_DIR}/samples/${pde}${sensor_mode:+/${sensor_mode}}"
            prepare_config "${pde}" "${task}" "${device}" "${base_config}" "${run_config}" "${sample_root}" "${sensor_mode}"
            JOB_PDES+=("${pde}")
            JOB_TASKS+=("${task}")
            JOB_DEVICES+=("${device}")
            JOB_CONFIGS+=("${run_config}")
            JOB_SAMPLE_ROOTS+=("${sample_root}")
            JOB_SENSOR_MODES+=("${sensor_mode}")
            job_index=$((job_index + 1))
        done
    done
done

echo "DiffusionPDE MAIN sampling sweep"
echo "  jobs: ${#JOB_PDES[@]}"
echo "  samples/job: ${NUM_SAMPLES}"
echo "  steps: ${NUM_STEPS}"
echo "  observations: ${NUM_OBS}"
echo "  output: ${OUTPUT_DIR}"
for ((i = 0; i < ${#JOB_PDES[@]}; i++)); do
    printf '  [%02d] %-14s %-7s %-14s %s\n' "$((i + 1))" "${JOB_PDES[i]}" "${JOB_TASKS[i]}" "${JOB_SENSOR_MODES[i]}" "${JOB_DEVICES[i]}"
done

if is_true "${PLAN_ONLY}"; then
    echo "Plan only; no sampling was started."
    exit 0
fi

failures=0
if is_true "${PARALLEL}"; then
    active_pids=()
    active_names=()
    for ((i = 0; i < ${#JOB_PDES[@]}; i++)); do
        run_job "${JOB_PDES[i]}" "${JOB_TASKS[i]}" "${JOB_DEVICES[i]}" \
            "${JOB_CONFIGS[i]}" "${JOB_SAMPLE_ROOTS[i]}" "${JOB_SENSOR_MODES[i]}" &
        active_pids+=("$!")
        active_names+=("${JOB_PDES[i]}/${JOB_SENSOR_MODES[i]}/${JOB_TASKS[i]}")
        if (( ${#active_pids[@]} >= MAX_PARALLEL_TASKS )); then
            if ! wait "${active_pids[0]}"; then
                echo "Job failed: ${active_names[0]}" >&2
                failures=$((failures + 1))
            fi
            active_pids=("${active_pids[@]:1}")
            active_names=("${active_names[@]:1}")
        fi
    done
    for ((i = 0; i < ${#active_pids[@]}; i++)); do
        if ! wait "${active_pids[i]}"; then
            echo "Job failed: ${active_names[i]}" >&2
            failures=$((failures + 1))
        fi
    done
else
    for ((i = 0; i < ${#JOB_PDES[@]}; i++)); do
        if ! run_job "${JOB_PDES[i]}" "${JOB_TASKS[i]}" "${JOB_DEVICES[i]}" \
            "${JOB_CONFIGS[i]}" "${JOB_SAMPLE_ROOTS[i]}" "${JOB_SENSOR_MODES[i]}"; then
            echo "Job failed: ${JOB_PDES[i]}/${JOB_SENSOR_MODES[i]}/${JOB_TASKS[i]}" >&2
            failures=$((failures + 1))
        fi
    done
fi

if (( failures > 0 )); then
    echo "Sampling sweep finished with ${failures} failed job(s)." >&2
    exit 1
fi

if is_true "${EVALUATE}" && is_true "${AGGREGATE}"; then
    "${PYTHON_BIN}" -u aggregate_eval.py "${METRICS_DIR}" --output-dir="${OUTPUT_DIR}"
fi

echo "Sampling sweep completed: ${OUTPUT_DIR}"
