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
#SBATCH --array=1-587

# Grid search parameters
LMBD=(10.0 1.0 100.0)                # bcs loss weight
EPOCHS=(300)               # number of epochs
BATCH_SIZES=(256 512)           # batch size
USE_PROJECTOR=(0 1)          # whether to use projector or not
proj_hidden_dim=(64 128 256 512 1024 2048 4096)
proj_output_dim=(64 128 256 512 1024 2048 4096)

chmod a+x ~/.bashrc
PS1='$ '
source ~/.bashrc
cd "/private/home/amirbar/projects/eb_jepa_release"

# Generate all combinations
combinations=()
for bs in "${BATCH_SIZES[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
        for lmbd in "${LMBD[@]}"; do
            for use_proj in "${USE_PROJECTOR[@]}"; do
                for proj_hidden_dim in "${proj_hidden_dim[@]}"; do
                    for proj_output_dim in "${proj_output_dim[@]}"; do
                        combinations+=("($bs, $epochs, $lmbd, $use_proj, $proj_hidden_dim, $proj_output_dim)")
                    done
                done
            done
        done
    done
done

# Get the combination for this array task
combination="${combinations[$SLURM_ARRAY_TASK_ID]}"
bs=$(echo "$combination" | awk -F '[(), ]+' '{print $2}')
epochs=$(echo "$combination" | awk -F '[(), ]+' '{print $3}')
lmbd=$(echo "$combination" | awk -F '[(), ]+' '{print $4}')
use_proj=$(echo "$combination" | awk -F '[(), ]+' '{print $5}')
proj_hidden_dim=$(echo "$combination" | awk -F '[(), ]+' '{print $6}')
proj_output_dim=$(echo "$combination" | awk -F '[(), ]+' '{print $7}')

echo "Running VICReg grid search experiment:"
echo "  batch_size=$bs"
echo "  epochs=$epochs"
echo "  lmbd=$lmbd"
echo "  use_projector=$use_proj"
echo "  proj_hidden_dim=$proj_hidden_dim"
echo "  proj_output_dim=$proj_output_dim"

/private/home/amirbar/projects/eb_jepa_release/.venv/bin/python examples/image_jepa/main.py \
    --batch_size=${bs} \
    --epochs=${epochs} \
    --lmbd=${lmbd} \
    --use_projector=${use_proj} \
    --proj_hidden_dim=${proj_hidden_dim} \
    --proj_output_dim=${proj_output_dim} \
    --project_name="lejepa-gridsearch" \
    --loss_type="bcs" \
    --model_type="resnet"
    