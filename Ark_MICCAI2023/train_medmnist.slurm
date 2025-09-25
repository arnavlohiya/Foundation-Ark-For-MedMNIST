#!/bin/bash
#SBATCH --job-name=ark_medmnist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ===================================================================
# ARK MedMNIST Training Script for ASU SOL Supercomputer
# Trains ARK framework on MedMNIST 2D datasets
# ===================================================================

echo "🚀 Starting ARK MedMNIST Training Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Load required modules
echo "📦 Loading modules..."
module load cuda/11.8
module load python/3.9

# Check loaded modules
echo "✅ Loaded modules:"
module list

# Set environment variables
export DATA_PATH="/scratch/$USER/data"
export CUDA_VISIBLE_DEVICES=0

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Verify GPU availability
echo "🖥️  GPU Information:"
nvidia-smi

# Check data directory
echo "📂 Data directory: $DATA_PATH"
ls -la $DATA_PATH/medmnist/ 2>/dev/null || echo "⚠️  MedMNIST data not found. Run setup_medmnist.py first!"

# Set training parameters
DATASETS="PathMNIST DermaMNIST PneumoniaMNIST"  # Start with 3 datasets for testing
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=50
EXP_NAME="medmnist_3datasets_test"

echo "🎯 Training Configuration:"
echo "  Datasets: $DATASETS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Experiment Name: $EXP_NAME"
echo "========================================"

# Update datasets_config.yaml to use correct data path
echo "🔧 Updating dataset configuration..."
sed -i "s|<PATH_TO_DATASET>|$DATA_PATH|g" datasets_config.yaml

# Run training
echo "🎯 Starting ARK training..."
python main_ark.py \
    --dataset_list $DATASETS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --pretrain_epochs $EPOCHS \
    --exp_name $EXP_NAME \
    --normalization imagenet \
    --crop_size 224 \
    --resize 256 \
    --workers 4 \
    --val_loss_metric average \
    --test_epoch 5 \
    --device cuda

echo "========================================"
echo "🏁 Training completed!"
echo "End time: $(date)"
echo "========================================"

# Optional: Copy results to home directory for backup
echo "💾 Backing up results..."
cp -r logs/ $HOME/ark_results_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "No logs directory found"

echo "✅ Job finished successfully!"