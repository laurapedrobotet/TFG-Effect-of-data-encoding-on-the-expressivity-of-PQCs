#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/laurap/pinn_qml/scripts
#SBATCH --output=logs/NLE_1D_%A_%a.out
#SBATCH --job-name=1D_NL_PQC
#SBATCH --array=0-239  # 6 layers * 10 seeds * 4 times 
#SBATCH --mem=16G

layers=(1 2 4 6 8 10)
seeds=($(seq 0 9))
times=(0 0.3 0.5 1)

total_seeds=${#seeds[@]}
total_layers=${#layers[@]}
total_times=${#times[@]}

layer_index=$((SLURM_ARRAY_TASK_ID / (total_seeds * total_times)))
seed_index=$(((SLURM_ARRAY_TASK_ID / total_times) % total_seeds))
time_index=$((SLURM_ARRAY_TASK_ID % total_times))

nlayers=${layers[$layer_index]}
random_seed=${seeds[$seed_index]}
time=${times[$time_index]}

echo "Running job $SLURM_ARRAY_TASK_ID with time=$time, nlayers=$nlayers, seed=$random_seed on host $(hostname)"

python3 run_1D_PQC_NLE.py \
    --time "$time" \
    --nlayers "$nlayers" \
    --epochs 1000 \
    --lr 0.1 \
    --random_seed "$random_seed"
