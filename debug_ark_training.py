#!/usr/bin/env python3
"""
Interactive debugging script for Ark training on MedMNIST datasets.
This script trains on only 3 datasets for 1 epoch to enable fast debugging.
"""

import os
import sys
import argparse
import torch
import torchvision
import numpy as np
import medmnist
import albumentations
import timm
import sklearn
import yaml
from tqdm import tqdm

# Add the Ark_MICCAI2023 directory to Python path
sys.path.append('./Ark_MICCAI2023')

def create_debug_args():
    """Create arguments for debugging with minimal datasets and epochs"""
    
    class DebugArgs:
        def __init__(self):
            # Minimal dataset selection for debugging
            self.dataset_list = ['PneumoniaMNIST', 'BreastMNIST', 'PathMNIST']
            
            # Model configuration
            self.model_name = 'swin_base'
            self.init = 'ImageNet_1k'
            self.pretrained_weights = None
            self.from_checkpoint = False
            
            # Training parameters - minimal for debugging
            self.epochs = 1
            self.pretrain_epochs = 1
            self.batch_size = 8  # Smaller batch size for debugging
            self.img_size = 224
            self.img_depth = 3
            
            # Data parameters
            self.normalization = 'imagenet'
            self.anno_percent = 100
            self.test_augment = True
            
            # Optimization parameters
            self.lr = 1e-3
            self.weight_decay = 0.01
            self.opt = 'momentum'
            self.opt_eps = 1e-8
            self.opt_betas = None
            self.momentum = 0.9
            self.clip_grad = None
            
            # Scheduler parameters
            self.sched = 'cosine'
            self.lr_noise = None
            self.lr_noise_pct = 0.67
            self.lr_noise_std = 1.0
            self.warmup_lr = 1e-6
            self.min_lr = 1e-5
            self.decay_epochs = 30
            self.warmup_epochs = 0
            self.cooldown_epochs = 10
            self.decay_rate = 0.5
            self.patience_epochs = 10
            
            # System parameters
            self.device = 'cuda'
            self.workers = 2  # Fewer workers for debugging
            self.GPU = None
            
            # Experiment parameters
            self.exp_name = 'debug_test'
            self.print_freq = 5  # Print more frequently for debugging
            self.test_epoch = 1
            self.val_loss_metric = 'average'
            
            # Model-specific parameters
            self.projector_features = 1376
            self.use_mlp = False
            self.ema_mode = 'epoch'
            self.momentum_teacher = 0.9
            
            # Activation and labels
            self.activate = 'Sigmoid'
            self.uncertain_label = 'LSR-Ones'
            self.unknown_label = 0
            
            # Resume
            self.resume = False
    
    return DebugArgs()

def main():
    """Main debugging function"""
    print("="*50)
    print("Ark Training Debug Script")
    print("Training on 3 datasets for 1 epoch")
    print("="*50)
    
    try:
        # Import main training modules
        from main_ark import main
        
        # Create debug arguments
        args = create_debug_args()
        
        print(f"Debug configuration:")
        print(f"  Datasets: {args.dataset_list}")
        print(f"  Model: {args.model_name}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Workers: {args.workers}")
        print(f"  Device: {args.device}")
        print(f"  Experiment name: {args.exp_name}")
        print()
        
        # Check if we're in the right directory
        if not os.path.exists('./Ark_MICCAI2023/main_ark.py'):
            print("ERROR: Please run this script from the Foundation-Ark-For-MedMNIST root directory")
            print("Current directory:", os.getcwd())
            return 1
        
        # Change to Ark directory for training
        original_dir = os.getcwd()
        os.chdir('./Ark_MICCAI2023')
        
        try:
            print("Starting debug training...")
            main(args)
            print("\n" + "="*50)
            print("Debug training completed successfully!")
            print("="*50)
            
        except Exception as e:
            print(f"\nERROR during training: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return 1
        
        finally:
            # Change back to original directory
            os.chdir(original_dir)
    
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the correct directory and all dependencies are installed")
        return 1
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Quick check for CUDA availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
    except ImportError:
        print("WARNING: PyTorch not found!")
    
    exit_code = main()
    sys.exit(exit_code if exit_code else 0)