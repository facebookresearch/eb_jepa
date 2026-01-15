#!/bin/bash

#SBATCH --job-name=AC-JEPA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH --time=24:00:00
#SBATCH --partition=learn
#SBATCH --signal=B:CONT@60
#SBATCH --requeue
#SBATCH --output=logs/ac_jepa/%A_%a.out
#SBATCH --error=logs/ac_jepa/%A_%a.err
#SBATCH --array=0-71

# =============================================================================
# Action-Conditioned Video JEPA - Hyperparameter Sweep
# =============================================================================
# This script launches a grid search over regularization coefficients.
# Adjust the arrays below to customize your sweep.
#
# Usage:
#   sbatch examples/ac_video_jepa/run_exp.sh
#
# Single run (without SLURM):
#   python -m examples.ac_video_jepa.main --model.regularizer.cov_coeff 8
# =============================================================================

# Grid search parameters
COV_COEFFS=(8 12)
STD_COEFFS=(8 16)
SIM_COEFFS=(8 12 16)
IDM_COEFFS=(1 2)
SEEDS=(1 1000 10000)

# Setup environment
chmod a+x ~/.bashrc
PS1='$ '
source ~/.bashrc

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p logs/ac_jepa

# Generate all parameter combinations
combinations=()
for cov in "${COV_COEFFS[@]}"; do
    for std in "${STD_COEFFS[@]}"; do
        for sim in "${SIM_COEFFS[@]}"; do
            for idm in "${IDM_COEFFS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    combinations+=("($cov, $std, $sim, $idm, $seed)")
                done
            done
        done
    done
done

# Get the combination for this array task
combination="${combinations[$SLURM_ARRAY_TASK_ID]}"
cov=$(echo "$combination" | awk -F '[(), ]+' '{print $2}')
std=$(echo "$combination" | awk -F '[(), ]+' '{print $3}')
sim=$(echo "$combination" | awk -F '[(), ]+' '{print $4}')
idm=$(echo "$combination" | awk -F '[(), ]+' '{print $5}')
seed=$(echo "$combination" | awk -F '[(), ]+' '{print $6}')

echo "=============================================="
echo "Running AC Video JEPA experiment:"
echo "  cov_coeff=$cov"
echo "  std_coeff=$std"
echo "  sim_coeff_t=$sim"
echo "  idm_coeff=$idm"
echo "  seed=$seed"
echo "  task_id=$SLURM_ARRAY_TASK_ID"
echo "=============================================="

# Run the experiment
python -m examples.ac_video_jepa.main \
    --model.regularizer.cov_coeff=${cov} \
    --model.regularizer.std_coeff=${std} \
    --model.regularizer.sim_coeff_t=${sim} \
    --model.regularizer.idm_coeff=${idm} \
    --meta.seed=${seed}
