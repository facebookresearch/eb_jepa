#!/bin/bash

#SBATCH --job-name=NWM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --partition=dev
#SBATCH --signal=B:CONT@60    
#SBATCH --requeue
#SBATCH --output=/checkpoint/amirbar/experiments/eb_jepa/logs/%A_%a.out
#SBATCH --error=/checkpoint/amirbar/experiments/eb_jepa/logs/%A_%a.err
#SBATCH --array=0-17

# BATCH_SIZES=(8 16 32 64)
BATCH_SIZES=(512 1024)
LRS=(1e-3 1e-4 5e-4)
STD=(1 10 100 200)
COV=(1 10 100 200)
# COV=(100)

chmod a+x ~/.bashrc
PS1='$ '
source ~/.bashrc
cd "/private/home/amirbar/projects/eb_jepa_internal"
echo ${exp_list[$SLURM_ARRAY_TASK_ID]}

triplets=()

for item1 in "${BATCH_SIZES[@]}"; do
    for item2 in "${LRS[@]}"; do
        for item3 in "${STD[@]}"; do
            for item4 in "${COV[@]}"; do
                triplets+=("($item1, $item2, $item3, $item4)")
            done
        done
    done
done


triplet="${triplets[$SLURM_ARRAY_TASK_ID]}"
bs=$(echo "$triplet" | awk -F '[(), ]+' '{print $2}')
lr=$(echo "$triplet" | awk -F '[(), ]+' '{print $3}')
std=$(echo "$triplet" | awk -F '[(), ]+' '{print $4}')
cov=$(echo "$triplet" | awk -F '[(), ]+' '{print $5}')

/private/home/amirbar/projects/eb_jepa_internal/.venv/bin/python -m examples.image_jepa.main \
    --batch_size=${bs} \
    --lr=${lr} \
    --std_coeff=${std} \
    --cov_coeff=${cov} \
    --epochs=100