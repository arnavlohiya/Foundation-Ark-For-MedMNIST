#!/usr/bin/env python3
"""
MedMNIST Dataset Download Script for ARK Framework
Downloads all 12 2D MedMNIST datasets and organizes them for ARK training

Usage:
    python setup_medmnist.py
    python setup_medmnist.py --data_path /custom/path
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path


def download_file(url, filepath, dataset_name, split):
    """Download a single NPZ file with progress tracking"""
    try:
        print(f"  Downloading {dataset_name}_{split}.npz...")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                bar_length = 30
                filled_length = (bar_length * percent) // 100
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"\r    [{bar}] {percent}% ({downloaded//1024//1024}MB)", end='')
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n    ✓ Downloaded {dataset_name}_{split}.npz")
        return True
        
    except urllib.error.URLError as e:
        print(f"\n    ✗ Failed to download {dataset_name}_{split}.npz: {e}")
        return False
    except Exception as e:
        print(f"\n    ✗ Error downloading {dataset_name}_{split}.npz: {e}")
        return False


def download_medmnist_datasets(data_path):
    """
    Download all 12 MedMNIST 2D datasets for ARK training
    
    Creates directory structure:
    data_path/medmnist/
    ├── pathmnist/pathmnist_train.npz, pathmnist_val.npz, pathmnist_test.npz
    ├── chestmnist/chestmnist_train.npz, chestmnist_val.npz, chestmnist_test.npz
    └── ... (for all 12 datasets)
    """
    
    # All 12 MedMNIST 2D datasets
    datasets = [
        'pathmnist',     # 9-class pathology tissue classification
        'chestmnist',    # 14-label chest X-ray multi-label classification
        'dermamnist',    # 7-class dermatology classification
        'octmnist',      # 4-class OCT image classification
        'pneumoniamnist',# Binary pneumonia classification
        'retinamnist',   # 5-level diabetic retinopathy ordinal regression
        'breastmnist',   # Binary breast cancer classification
        'bloodmnist',    # 8-class blood cell classification
        'tissuemnist',   # 8-class kidney tissue classification
        'organamnist',   # 11-class abdominal organ classification
        'organcmnist',   # 11-class coronary organ classification
        'organsmnist'    # 11-class sagittal organ classification
    ]
    
    # MedMNIST v2 base URL (Zenodo repository)
    base_url = "https://zenodo.org/record/10519652/files/"
    
    print("=" * 60)
    print("🔬 MedMNIST Dataset Download for ARK Framework")
    print("=" * 60)
    print(f"📂 Download path: {data_path}")
    print(f"📊 Datasets to download: {len(datasets)}")
    print("=" * 60)
    
    # Create base directory
    medmnist_path = Path(data_path) / 'medmnist'
    medmnist_path.mkdir(parents=True, exist_ok=True)
    
    successful_downloads = 0
    total_downloads = len(datasets) * 3  # 3 splits per dataset
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n📥 [{i}/{len(datasets)}] Processing {dataset.upper()}")
        
        # Create dataset directory
        dataset_dir = medmnist_path / dataset
        dataset_dir.mkdir(exist_ok=True)
        
        dataset_success = 0
        for split in ['train', 'val', 'test']:
            filename = f"{dataset}_{split}.npz"
            filepath = dataset_dir / filename
            
            # Skip if file already exists
            if filepath.exists():
                print(f"  ✓ {filename} already exists, skipping...")
                successful_downloads += 1
                dataset_success += 1
                continue
            
            # Download file
            url = f"{base_url}{filename}"
            if download_file(url, str(filepath), dataset, split):
                successful_downloads += 1
                dataset_success += 1
        
        # Dataset summary
        if dataset_success == 3:
            print(f"  🎉 {dataset.upper()} complete!")
        else:
            print(f"  ⚠️  {dataset.upper()} incomplete ({dataset_success}/3 files)")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful_downloads}/{total_downloads} files")
    print(f"❌ Failed: {total_downloads - successful_downloads}/{total_downloads} files")
    
    if successful_downloads == total_downloads:
        print("🎉 All MedMNIST datasets downloaded successfully!")
        print("\n🚀 You can now run ARK training with:")
        print("   python main_ark.py --dataset_list PathMNIST DermaMNIST PneumoniaMNIST")
    else:
        print("⚠️  Some downloads failed. Please check your internet connection and try again.")
        return False
    
    print("=" * 60)
    return True


def verify_datasets(data_path):
    """Verify that all datasets are properly downloaded"""
    datasets = [
        'pathmnist', 'chestmnist', 'dermamnist', 'octmnist',
        'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist',
        'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist'
    ]
    
    medmnist_path = Path(data_path) / 'medmnist'
    
    print("\n🔍 Verifying downloaded datasets...")
    missing_files = []
    
    for dataset in datasets:
        for split in ['train', 'val', 'test']:
            filepath = medmnist_path / dataset / f"{dataset}_{split}.npz"
            if not filepath.exists():
                missing_files.append(str(filepath))
    
    if not missing_files:
        print("✅ All datasets verified successfully!")
        return True
    else:
        print(f"❌ Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"  - {file}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download MedMNIST datasets for ARK framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_medmnist.py
  python setup_medmnist.py --data_path /scratch/username/data
  python setup_medmnist.py --verify-only
        """
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        default=os.path.expanduser('~/data'),
        help='Path to store MedMNIST datasets (default: ~/data)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets without downloading'
    )
    
    args = parser.parse_args()
    
    # Expand environment variables and user paths
    data_path = os.path.expandvars(os.path.expanduser(args.data_path))
    
    if args.verify_only:
        success = verify_datasets(data_path)
        sys.exit(0 if success else 1)
    
    # Download datasets
    try:
        success = download_medmnist_datasets(data_path)
        
        # Verify after download
        if success:
            verify_datasets(data_path)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()