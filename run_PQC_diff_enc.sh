#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/laurap/pinn_qml/scripts
#SBATCH --output=logs/nonlinearDifEnc_%A_%a.out
#SBATCH --job-name=nonlinear_PQC_diff
#SBATCH --array=0-59
#SBATCH --mem=16G

layers=(1 2 4 6 8 10)
seeds=($(seq 0 9))

layer_index=$((SLURM_ARRAY_TASK_ID / 10))
seed_index=$((SLURM_ARRAY_TASK_ID % 10))

nlayers=${layers[$layer_index]}
random_seed=${seeds[$seed_index]}

echo "Running job $SLURM_ARRAY_TASK_ID with nlayers=$nlayers and seed=$random_seed on host $(hostname)"

python3 run_PQC_diff_enc.py \
    --encoding nonlinear \
    --nlayers "$nlayers" \
    --npoints 1000 \
    --epochs 1000 \
    --lr 0.1 \
    --random_seed "$random_seed"
