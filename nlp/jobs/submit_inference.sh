#!/bin/bash
#SBATCH --account=p32859              # Your account number
#SBATCH --partition=gengpu            # Use gengpu or short/gpu depending on time
#SBATCH --time=06:00:00               # Walltime (adjust based on expected runtime)
#SBATCH --job-name=jnk_test           # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node (GPUs)
#SBATCH --cpus-per-task=8             # CPUs per task
#SBATCH --mem=256G                     # Memory per node
#SBATCH --gres=gpu:h100:1             # GPU type and count (can be h100, t4, etc.)
#SBATCH --mail-type=ALL               # Email notifications
#SBATCH --constraint=sxm
#SBATCH --mail-user=narasimhajwalapuram2026@u.northwestern.edu

# --- Output directories ---
SCRATCH_BASE="/scratch/wnn7240/JNK/NLPF/inference_runs"
PROJECT_NAME="financial_llm_inference_full_v1"

# Create paths
PROJECT_OUTPUT_DIR="${SCRATCH_BASE}/${PROJECT_NAME}"
PROJECT_RESULTS_DIR="${PROJECT_OUTPUT_DIR}/results" # Define results directory
PROJECT_LOGS_DIR="${PROJECT_OUTPUT_DIR}/logs"     # Define logs directory

mkdir -p "${PROJECT_LOGS_DIR}" "${PROJECT_RESULTS_DIR}" # Create both directories

# Export the results directory for the Python script to use
export INFERENCE_RESULTS_DIR="${PROJECT_RESULTS_DIR}"
echo "Exported INFERENCE_RESULTS_DIR=${INFERENCE_RESULTS_DIR}"

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

# Activate conda
source /home/wnn7240/miniconda/etc/profile.d/conda.sh
conda activate base

# Navigate to your working directory
# This path should be the directory containing inference.py and its associated config/data subdirectories
cd /home/wnn7240/JNK/NLP/nlp/nlp || { echo "Failed to enter project directory: /home/wnn7240/JNK/NLP/nlp/nlp"; exit 1; }

# --- Run your Python inference script ---
echo "Running inference at $(date)"
python inference.py # Removed '1' as it's not used by the script

# Define the expected final output file from the Python script
# The Python script will now write directly to INFERENCE_RESULTS_DIR
FINAL_PREDICTIONS_FILE="${INFERENCE_RESULTS_DIR}/predictions.json"

# Check if the Python script successfully created the output file in the expected location
if [ -f "${FINAL_PREDICTIONS_FILE}" ]; then
    echo "Results successfully saved by Python script to: ${FINAL_PREDICTIONS_FILE}"
else
    echo "Error: Python output file ${FINAL_PREDICTIONS_FILE} not found!"
    echo "Please check Python script logs in ${PROJECT_LOGS_DIR} and Slurm output for errors."
fi

echo "Inference completed at $(date)"
