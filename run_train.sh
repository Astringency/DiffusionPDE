#!/bin/bash

OUTDIR=${OUTDIR:-/data0/zhangxf/Models/}
DATA_ROOT=${DATA_ROOT:-/data0/zhangxf/PDEdata/}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES

# torchrun --master_addr=127.0.0.1 --master_port=12351 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=darcy --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12352 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=poisson --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12353 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=helmholtz --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12354 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=nsnonbounded --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12355 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=shallow_water --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
torchrun --master_addr=127.0.0.1 --master_port=12356 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=reaction_diffusion --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12357 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=heat --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12358 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=wave --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12359 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=advection_diffusion --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# torchrun --master_addr=127.0.0.1 --master_port=12360 --nproc_per_node="${NPROC_PER_NODE}" train.py --pde=steady_heat_conduction --outdir="${OUTDIR}" --data="${DATA_ROOT}" --cond=0 --arch=ddpmpp --batch=64 --batch-gpu=32 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
