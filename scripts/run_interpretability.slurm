#!/bin/bash
#SBATCH --account=p32275
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --job-name=financial_interpretability
#SBATCH --output=logs/interpretability_%j.out
#SBATCH --error=logs/interpretability_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@northwestern.edu

# Load necessary modules
module purge
module load python/3.9.0
module load cuda/11.8

# Navigate to project directory
cd /projects/p32275/nlp_project/nlp

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment
echo "Setting up Python environment..."
source venv/bin/activate || {
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r ../requirements_interpretability.txt
}

# Print system information
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "================================================================"

# Check GPU availability
nvidia-smi
echo "CUDA Version: $(nvcc --version)"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/projects/p32275/cache/transformers
export HF_HOME=/projects/p32275/cache/huggingface

# Create cache directories
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME

# Print Python environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run mechanistic interpretability analysis
echo "================================================================"
echo "Starting Mechanistic Interpretability Analysis"
echo "================================================================"

python ../scripts/mechanistic_interpretability.py \
    --config config/interpretability_config.yaml \
    --predictions outputs/predictions_checkpoint_1800.json \
    --max_samples 30

echo "================================================================"
echo "Analysis completed at: $(date)"
echo "================================================================"

# Print output directory contents
echo "Generated files:"
ls -la results/interpretability/

# Deactivate environment
deactivate

echo "Job completed successfully!"