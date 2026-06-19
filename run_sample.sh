#!/bin/bash

problem_vals=("forward" "inverse")
# problem_vals=("both" "forward" "inverse")
# pde_vals=("darcy" "poisson" "helmholtz" "ns-nonbounded")
pde_vals=("heat" "wave" "advection_diffusion" "steady_heat_conduction")
# pde_vals=("burgers" "darcy" "poisson" "helmholtz" "shallow_water" "ns-nonbounded" "heat" "wave" "advection_diffusion" "steady_heat_conduction")
# pde_vals=("ns-nonbounded" "shallow_water")

steps_vals=(100 200 500 1000 2000)

for prob in "${problem_vals[@]}"; do
    for pde in "${pde_vals[@]}"; do
        for step in "${steps_vals[@]}"; do
            python ./generate_pde.py --config="./configs/$pde.yaml" --problem=$prob --batch=1 --step_size=$step
        done
    done
done

wait

echo "All Done."

# python ./generate_pde.py --config="./configs/poisson.yaml" --problem="both" --batch=1
