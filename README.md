# DiffusionPDE Experiments for FM4PDE

Adapted DiffusionPDE training and PDE-guided sampling for the comparisons in
**Guided Flow Matching for Forward and Inverse PDE Problems with Sparse
Observations: Algorithm and Theory**. The retained experiments cover Poisson,
Helmholtz, Darcy, Navier–Stokes, and Burgers.

## Training

Run commands from the repository root in Bash (Linux or WSL). Install
[requirements.txt](requirements.txt) in a suitable PyTorch environment.
Datasets and weights are external assets.

```bash
python -m pip install -r requirements.txt
export DATA_ROOT=/path/to/PDEdata
export CHECKPOINT_ROOT=/path/to/pretrained
export PYTHON_BIN=python
```

[train.py](train.py) calls `training_loop()` in
[training/training_loop.py](training/training_loop.py).
The adapted data loader reads five physical MATLAB/HDF5 shards per equation
through [data/load.py](data/load.py); [data/transform.py](data/transform.py)
provides the training normalization.

Launch a single-PDE training run directly:

```bash
python train.py --pde=poisson --outdir=outputs/training/poisson \
  --data="$DATA_ROOT" --cond=0 --arch=ddpmpp \
  --batch=64 --batch-gpu=32 \
  --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
```

Or use the Bash launcher:

```bash
PDE=poisson DRY_RUN=true bash scripts/training/run.sh
PDE=poisson NPROC_PER_NODE=2 bash scripts/training/run.sh
PDE_LIST="poisson helmholtz darcy nsnonbounded burger" \
  bash scripts/training/run.sh
```

The launcher uses the DDPM++/EDM recipe, a global batch of 64, and a budget of
20 million training images. `DURATION`, `BATCH`, `BATCH_GPU`, and `OUTDIR`
override these settings. `TRAIN_DATA_<PDE>` selects another data root;
`RESUME=/path/to/training-state.pt` resumes a single selected PDE.

## Main Sampling

[generate_pde.py](generate_pde.py) dispatches to the equation-specific samplers
in [scripts](scripts), such as [scripts/generate_poisson.py](scripts/generate_poisson.py).
The YAML files in [configs](configs) specify the observations, weights, noise
schedule, guidance, and outputs.

For direct Python commands, first edit the selected YAML's `data.datapath`,
`test.pre-trained`, `generate.device`, and `output.file_path` to match your
installation. The root environment variables above are applied by the Bash
sweep; `generate_pde.py` reads the YAML paths directly.

```bash
python generate_pde.py --config configs/poisson.yaml \
  --problem forward --batch 10 --start_offset 0 --step_size 100
python generate_pde.py --config configs/poisson.yaml \
  --problem inverse --batch 10 --start_offset 0 --step_size 100
python generate_pde.py --config configs/poisson.yaml \
  --problem both --batch 10 --start_offset 0 --step_size 1000
python generate_pde.py --config configs/burgers.yaml \
  --problem both --batch 10 --start_offset 0 --step_size 1000
```

`--step_size` sets the number of denoising steps and overrides the YAML value.
`--batch` is the number of input cases processed by this entry.

The main Bash workflow runs Smooth comparisons at 100 and 1,000 steps:

```bash
PLAN_ONLY=true bash scripts/sampling/main/run.sh
DEVICE_LIST="cuda:0 cuda:1" PARALLEL=true \
  bash scripts/sampling/main/run.sh

# One equation, one task, and one step budget.
PDE_LIST=poisson TASK_LIST=both NUM_SAMPLES=100 NUM_STEPS=100 \
  OUTPUT_DIR=outputs/main_subset \
  bash scripts/sampling/main/run_sweep.sh

# Burgers: random points and complete time slices, at both budgets.
PDE_LIST=burger bash scripts/sampling/main/run.sh
```

Default weights under `CHECKPOINT_ROOT` are `pretrained-poisson.pkl`,
`pretrained-helmholtz.pkl`, `pretrained-darcy.pkl`,
`pretrained-ns-nonbounded.pkl`, and `pretrained-burgers.pkl`.
`DATA_<PDE>` and `CHECKPOINT_<PDE>` override individual files
(use `BURGER` for Burgers). `OUTPUT_ROOT` changes `outputs/main`;
`STEP_LIST` changes the two budgets. Completed results can be resumed.

Each comparison uses 1,000 inputs and 500 observations per active field.
Burgers uses either 500 random space–time points or five complete physical
time levels (640 values on a 128 × 128 trajectory), with the full trajectory
as the evaluation target. Select layouts with
`BURGER_SENSOR_MODES="random time_slices"`; `BURGER_TIME_SLICES` changes the
number of observed time levels. Fixed observation and latent seeds preserve
input-level reproducibility across batching and resume order.

## Ablations

The paired error–time trajectory study is implemented in the companion
FM4PDE repository's `experiments/trajectories/`. This repository provides
[scripts/sampling/ablations/run.sh](scripts/sampling/ablations/run.sh) as its
launcher. Set `FM_ROOT` explicitly when the checkout is named `FM4PDEdebug`.

```bash
FM_ROOT=/path/to/FM4PDEdebug \
  bash scripts/sampling/ablations/run.sh prepare --help
FM_ROOT=/path/to/FM4PDEdebug \
  bash scripts/sampling/ablations/run.sh run --root /path/to/trace_study
FM_ROOT=/path/to/FM4PDEdebug \
  bash scripts/sampling/ablations/run.sh plot \
    --root /path/to/trace_study --output /path/to/figures
```

The run and plot examples require prepared inputs, fixed observations, and both
methods' weights. The study records reconstruction error and a common physical
residual; DiffusionPDE's native NS guidance retains its own residual definition.

## Baseline and other info

Related repositories: [FM4PDE](https://github.com/Astringency/FM4PDEdebug.git),
[RecFNO and other baselines](https://github.com/Astringency/FM4PDEbaseline.git),
and [CoCoGen comparisons](https://github.com/Astringency/CoCoGen.git).

Our code is modified and adapted from the official
[DiffusionPDE implementation](https://github.com/jhhuangchloe/DiffusionPDE)
by Huang et al. for the FM4PDE datasets, observation protocols, and evaluation.
The training infrastructure, `dnnlib`, and `torch_utils` build on
[NVIDIA EDM](https://github.com/NVlabs/edm) by Karras et al.
The bundled resizing utility credits
[Assaf Shocher's resizer](https://github.com/assafshocher/resizer).
Please also acknowledge these upstream projects and retain their source notices.
