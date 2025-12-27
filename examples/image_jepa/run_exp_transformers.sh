#!/bin/bash

#SBATCH --job-name=VICReg-GridSearch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --account fair_amaia_cw_video
#SBATCH --qos dev
#SBATCH --signal=B:CONT@60    
#SBATCH --requeue
#SBATCH --output=/checkpoint/amirbar/experiments/eb_jepa/logs/%A_%a.out
#SBATCH --error=/checkpoint/amirbar/experiments/eb_jepa/logs/%A_%a.err
#SBATCH --array=0-17

# Grid search parameters
BATCH_SIZES=(512)           # batch size
EPOCHS=(50 100 1000)               # number of epochs
USE_PROJECTOR=(1 0)                 # whether to use projector or not (1=true, 0=false)
MODEL_TYPE=(resnet vit_s vit_b)
PATCH_SIZE=(2)

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
uv run python -m examples.image_jepa.main \
    --model_type=${MODEL_TYPE} \
    --patch_size=${PATCH_SIZE} \
    --batch_size=${bs} \
    --num_workers=8 \
    --epochs=${epochs} \
    --use_projector=${use_proj} \
    --project_name="vicreg-gridsearch" \
    --data_dir /checkpoint/amaia/video/davidfan/data/CIFAR10 \
    --use_amp