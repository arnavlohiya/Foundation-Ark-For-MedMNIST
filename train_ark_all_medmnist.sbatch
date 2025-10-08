#!/bin/bash
#SBATCH --job-name=ark-12-medmnist
#SBATCH --output=logs/ark_training_%j.out
#SBATCH --error=logs/ark_training_%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

module load mamba/latest
# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Ark unified training on all 12 MedMNIST datasets at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

cd Ark_MICCAI2023

# Train one unified model on all 12 MedMNIST datasets simultaneously
python main_ark.py \
    --dataset_list PneumoniaMNIST BreastMNIST PathMNIST DermaMNIST OCTMNIST BloodMNIST TissueMNIST OrganAMNIST OrganCMNIST OrganSMNIST RetinaMNIST ChestMNIST \
    --model swin_base \
    --init ImageNet_1k \
    --epochs 200 \
    --batch_size 32 \
    --img_size 224 \
    --workers 8 \
    --device cuda \
    --exp_name "all_medmnist_unified" \
    --normalization imagenet \
    --lr 1e-3 \
    --weight-decay 0.01 \
    --warmup-epochs 10 \
    --test_epoch 1 \
    --print_freq 50

echo "Ark training completed at $(date)"
echo "Exit code: $?"