#!/bin/bash

#SBATCH --job-name=VICReg-GridSearch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --partition=dev
#SBATCH --signal=B:CONT@60    
#SBATCH --requeue
#SBATCH --output=/checkpoint/amirbar/experiments/eb_jepa/logs/%A_%a.out
#SBATCH --error=/checkpoint/amirbar/experiments/eb_jepa/logs/%A_%a.err
#SBATCH --array=0-17

# Grid search parameters
BATCH_SIZES=(128 256 512)           # batch size
EPOCHS=(50 100 1000)               # number of epochs
USE_PROJECTOR=(true false)          # whether to use projector or not

chmod a+x ~/.bashrc
PS1='$ '
source ~/.bashrc
cd "/private/home/amirbar/projects/eb_jepa_internal"

# Generate all combinations
combinations=()
for bs in "${BATCH_SIZES[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
        for use_proj in "${USE_PROJECTOR[@]}"; do
            combinations+=("($bs, $epochs, $use_proj)")
        done
    done
done

# Get the combination for this array task
combination="${combinations[$SLURM_ARRAY_TASK_ID]}"
bs=$(echo "$combination" | awk -F '[(), ]+' '{print $2}')
epochs=$(echo "$combination" | awk -F '[(), ]+' '{print $3}')
use_proj=$(echo "$combination" | awk -F '[(), ]+' '{print $4}')

echo "Running VICReg grid search experiment:"
echo "  batch_size=$bs"
echo "  epochs=$epochs"
echo "  use_projector=$use_proj"

# Create run name for this specific configuration
run_name="vicreg-cifar10-bs${bs}-ep${epochs}-proj${use_proj}"

/private/home/amirbar/projects/eb_jepa_internal/.venv/bin/python -m examples.image_jepa.main \
    --batch_size=${bs} \
    --epochs=${epochs} \
    --use_projector=${use_proj} \
    --run_name=${run_name} \
    --project_name="vicreg-gridsearch"