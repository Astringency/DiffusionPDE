# DiffusionPDE comparisons

The existing EDM training and PDE-guided samplers used for the Poisson, Helmholtz,
Darcy, Navier–Stokes and Burgers comparisons. Upstream license notices remain in the source files. Sampling equations and numerical guidance have been retained.

## Setup and data

Install the dependencies in `requirements.txt` in a suitable PyTorch environment.
The customized training loop uses `data/load.py` to read five physical MATLAB/HDF5
training shards per equation and `data/transform.py` for its original normalization.
Sampling reads physical test fields and retains each sampler's normalization.

```bash
export DATA_ROOT=/path/to/PDEdata
export CHECKPOINT_ROOT=/path/to/pretrained
export PYTHON_BIN=python
```

Expected sampling weights are `pretrained-poisson.pkl`, `pretrained-helmholtz.pkl`,
`pretrained-darcy.pkl`, `pretrained-ns-nonbounded.pkl` and `pretrained-burgers.pkl`.
Per-equation `DATA_<PDE>` and `CHECKPOINT_<PDE>` overrides select explicit files.

## Training

```bash
PDE=poisson DATA_ROOT=/path/to/PDEdata DRY_RUN=true bash scripts/training/run.sh
PDE=poisson DATA_ROOT=/path/to/PDEdata NPROC_PER_NODE=2 bash scripts/training/run.sh
```

The launcher retains the DDPM++/EDM recipe, 64-example global batch and 20 M-image
budget. Set `DURATION`, `BATCH`, `BATCH_GPU`, `PDE_LIST`, `OUTDIR`, or a single-PDE
`RESUME` state as needed. `TRAIN_DATA_<PDE>` overrides the physical training-data directory.

## Main sampling

```bash
PLAN_ONLY=true bash scripts/sampling/main/run.sh
DEVICE_LIST="cuda:0 cuda:1" PARALLEL=true bash scripts/sampling/main/run.sh
```

For Poisson, Helmholtz, Darcy and Navier–Stokes, this runs the Smooth comparisons
at 100 and 1,000 steps, 1,000 inputs per task, and 500 observations per active field.
The Burgers entry still uses five spatial sensor columns across all time levels.
The paper's 500 random space–time points and five complete time slices were run
through a separate study driver and are not yet integrated into this entry.
Outputs are separated by step budget under `outputs/main`; `OUTPUT_ROOT`
changes that location. `run_sweep.sh` exposes one budget and supports explicit
per-PDE data/checkpoint overrides. Completed results can be resumed.

## Error–time trajectories

```bash
FM_ROOT=/path/to/FM4PDE bash scripts/sampling/ablations/run.sh --help
```

The paired trajectory implementation resides in FM4PDE and consumes prepared
inputs with fixed observations. It records reconstruction error and a common
physical-residual evaluation for both methods. The native DiffusionPDE NS sampling
guidance is distinct from FM4PDE's endpoint-secant evaluation. Both original
definitions are retained.

Earlier exploratory samplers, unused PDE extensions and development figures are
preserved in ignored `bak/`. Existing datasets, model weights and results are untouched.
