#!/bin/bash
#SBATCH --job-name=pip-install-ark
#SBATCH --output=pip_install_%j.out
#SBATCH --error=pip_install_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=general

# Set number of cores for compilation
export CMAKE_BUILD_PARALLEL_LEVEL=8
export MAKEFLAGS="-j8"

# Load any required modules (adjust as needed for your cluster)
# module load python/3.6

echo "Starting pip install at $(date)"
echo "Using $SLURM_CPUS_PER_TASK cores for compilation"

# Install requirements
pip install --user -r Ark_MICCAI2023/requirements.txt

echo "Installation completed at $(date)"
echo "Exit code: $?"