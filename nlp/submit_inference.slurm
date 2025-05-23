#!/bin/bash
#SBATCH --account=p32859              # Your account number
#SBATCH --partition=gengpu            # Use gengpu or short/gpu depending on time
#SBATCH --time=06:00:00               # Walltime (adjust based on expected runtime)
#SBATCH --job-name=LLM_Inference      # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node (GPUs)
#SBATCH --cpus-per-task=8             # CPUs per task
#SBATCH --mem=32G                     # Memory per node
#SBATCH --gres=gpu:p100:1             # GPU type and count (can be h100, t4, etc.)
#SBATCH --mail-type=ALL               # Email notifications
#SBATCH --mail-user=narasimhajwalapuram2026@u.northwestern.edu

# --- Output directories ---
SCRATCH_BASE="/scratch/wnn7240/inference_runs"
PROJECT_NAME="financial_llm_inference_$(date +%Y%m%d_%H%M%S)"

# Create paths
PROJECT_OUTPUT_DIR="${SCRATCH_BASE}/${PROJECT_NAME}"
mkdir -p "${PROJECT_OUTPUT_DIR}/logs" "${PROJECT_OUTPUT_DIR}/results"

LOGS_DIR="${PROJECT_OUTPUT_DIR}/logs"
OUTPUT_CSV="${PROJECT_OUTPUT_DIR}/results/inference_results.csv"

# Log job info
echo "=== Financial LLM Inference Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Output directory: ${PROJECT_OUTPUT_DIR}"
echo "=================================="

# --- Environment Setup ---
module purge
module load python/anaconda3

# Activate conda
source /home/wnn7240/miniconda/etc/profile.d/conda.sh
conda activate base  # Replace 'base' if using a different environment

# Navigate to your working directory
cd /home/wnn7240/projects/financial-inference || { echo "Failed to enter project directory"; exit 1; }

# --- Run your Python inference script ---
echo "Running inference at $(date)"
python run_inference.py

# Move output CSV to results folder
if [ -f "inference_results.csv" ]; then
    mv inference_results.csv "${OUTPUT_CSV}"
    echo "Results saved to: ${OUTPUT_CSV}"
else
    echo "Error: inference_results.csv not found!"
fi

echo "Inference completed at $(date)"