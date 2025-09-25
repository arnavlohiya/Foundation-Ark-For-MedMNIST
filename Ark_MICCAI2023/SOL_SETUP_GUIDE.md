# 🚀 ARK MedMNIST Setup Guide for ASU SOL

This guide walks you through setting up and running ARK with MedMNIST datasets on ASU's SOL supercomputer.

## 📋 Quick Start Checklist

### 1. Initial Setup on SOL
```bash
# Clone repository
git clone https://github.com/yourusername/ark-medmnist.git
cd ark-medmnist

# Load Python module
module load python/3.9

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download MedMNIST Datasets
```bash
# Create data directory in scratch space
mkdir -p /scratch/$USER/data

# Download all 12 MedMNIST datasets (takes 5-10 minutes)
python setup_medmnist.py --data_path /scratch/$USER/data

# Verify downloads
python setup_medmnist.py --data_path /scratch/$USER/data --verify-only
```

### 3. Quick Test (Optional)
```bash
# Test with small dataset for 1 epoch
python main_ark.py --dataset_list PneumoniaMNIST --pretrain_epochs 1 --batch_size 8
```

### 4. Submit Training Job
```bash
# Submit SLURM job
sbatch train_medmnist.slurm

# Monitor job
squeue -u $USER
tail -f slurm-*.out
```

## 🔧 Configuration Options

### Modify Training Parameters
Edit `train_medmnist.slurm` to change:
- `DATASETS`: Which MedMNIST datasets to train on
- `BATCH_SIZE`: Batch size (adjust based on GPU memory)
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate

### Available MedMNIST Datasets
```bash
# Binary classification
PneumoniaMNIST BreastMNIST

# Multi-class classification  
PathMNIST DermaMNIST OCTMNIST BloodMNIST TissueMNIST OrganAMNIST OrganCMNIST OrganSMNIST

# Multi-label classification
ChestMNIST

# Ordinal regression
RetinaMNIST
```

### Example Training Commands
```bash
# Single dataset
python main_ark.py --dataset_list PathMNIST --pretrain_epochs 100

# Multiple datasets
python main_ark.py --dataset_list PathMNIST DermaMNIST PneumoniaMNIST --pretrain_epochs 50

# All 12 datasets
python main_ark.py --dataset_list PathMNIST ChestMNIST DermaMNIST OCTMNIST PneumoniaMNIST RetinaMNIST BreastMNIST BloodMNIST TissueMNIST OrganAMNIST OrganCMNIST OrganSMNIST --pretrain_epochs 100
```

## 📊 Monitoring Training

### Check Job Status
```bash
squeue -u $USER                    # Job queue status
scontrol show job <jobid>          # Detailed job info
scancel <jobid>                    # Cancel job if needed
```

### View Training Progress
```bash
tail -f slurm-*.out               # Live training output
less slurm-*.out                  # Browse full output
grep "ACCURACY\|AUC" slurm-*.out  # View performance metrics
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `BATCH_SIZE` in `train_medmnist.slurm`
- Request GPU with more memory: `#SBATCH --gres=gpu:a100:1`

**2. Dataset Not Found**
- Verify data path: `ls /scratch/$USER/data/medmnist/`
- Re-run: `python setup_medmnist.py --data_path /scratch/$USER/data`

**3. Module Not Found**
- Activate environment: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**4. Job Killed (Time Limit)**
- Increase time: `#SBATCH --time=48:00:00`
- Or reduce epochs for initial testing

### Performance Tips
- Start with 2-3 datasets for initial testing
- Use `--test_epoch 10` for less frequent evaluation
- Monitor GPU utilization: `nvidia-smi -l 1`

## 📁 File Structure
```
ark-medmnist/
├── datasets_config.yaml          # Dataset configurations
├── dataloader.py                 # MedMNIST data loading
├── engine.py                     # Training engine with multi-loss support
├── trainer.py                    # Training/testing functions
├── main_ark.py                   # Main training script
├── requirements.txt              # Python dependencies
├── setup_medmnist.py             # Dataset download script
├── train_medmnist.slurm          # SLURM job script
└── SOL_SETUP_GUIDE.md            # This guide
```

## 🎯 Expected Results

After successful training, you should see:
- Validation losses for each dataset
- AUROC scores for binary/multi-label tasks
- Accuracy scores for multi-class tasks
- Model checkpoints saved in `logs/` directory

Good luck with your training! 🚀