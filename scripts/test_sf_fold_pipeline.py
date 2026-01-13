#!/usr/bin/env python3
"""
Validation test script for sf_fold dataset and pipeline.
Run this before training to verify:
1. Dataset class loads correctly
2. Config loads without errors
3. Action normalization works
4. Output shapes are correct

Usage:
    python scripts/test_sf_fold_pipeline.py --data_dir /path/to/sf_fold_npz
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dataset(data_dir):
    """Test 1: Load dataset class with a few samples."""
    print("\n" + "=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)

    from lvdm.data.sf_fold_vid import SFFoldVid

    # Check for required files
    action_stats_path = os.path.join(data_dir, 'action_stats.json')
    val_file_list_path = os.path.join(data_dir, 'val_file_list.json')

    if not os.path.exists(action_stats_path):
        print(f"WARNING: action_stats.json not found at {action_stats_path}")
        print("Creating dummy action stats for testing...")
        import json
        dummy_stats = {
            'mean': [0.0] * 20,
            'std': [1.0] * 20
        }
        with open(action_stats_path, 'w') as f:
            json.dump(dummy_stats, f)

    if not os.path.exists(val_file_list_path):
        print(f"WARNING: val_file_list.json not found at {val_file_list_path}")
        print("Creating dummy validation split for testing...")
        import json
        # Get first 10% of files as validation
        npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        npz_files.sort()
        val_count = max(1, len(npz_files) // 10)
        val_files = npz_files[:val_count]
        with open(val_file_list_path, 'w') as f:
            json.dump(val_files, f)

    # Load training dataset
    print("\nLoading training dataset...")
    ds_train = SFFoldVid(
        data_dir=data_dir,
        resolution=[320, 512],
        video_length=16,
        spatial_transform='resize_center_crop',
        mode='training',
        augment_data=True,
        subsample=True,  # Only load 16 samples for testing
        val_file_list_path=val_file_list_path,
        action_stats_path=action_stats_path,
    )
    print(f"Training dataset size: {len(ds_train)}")

    # Load validation dataset
    print("\nLoading validation dataset...")
    ds_val = SFFoldVid(
        data_dir=data_dir,
        resolution=[320, 512],
        video_length=16,
        spatial_transform='resize_center_crop',
        mode='validation',
        augment_data=False,
        subsample=True,
        val_file_list_path=val_file_list_path,
        action_stats_path=action_stats_path,
    )
    print(f"Validation dataset size: {len(ds_val)}")

    # Get a sample
    print("\nLoading sample from training dataset...")
    sample = ds_train[0]

    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Video shape: {sample['video'].shape}")  # Expected: [3, 16, 320, 512]
    print(f"Action shape: {sample['action'].shape}")  # Expected: [16, 20]
    print(f"Caption: '{sample['caption']}'")
    print(f"FPS: {sample['fps']}")
    print(f"Frame stride: {sample['frame_stride']}")
    print(f"Path: {sample['path']}")

    # Verify shapes
    assert sample['video'].shape == (3, 16, 320, 512), \
        f"Video shape mismatch: {sample['video'].shape} != (3, 16, 320, 512)"
    assert sample['action'].shape == (16, 20), \
        f"Action shape mismatch: {sample['action'].shape} != (16, 20)"

    # Verify normalization ranges
    video_min, video_max = sample['video'].min().item(), sample['video'].max().item()
    action_min, action_max = sample['action'].min().item(), sample['action'].max().item()

    print(f"\nVideo range: [{video_min:.3f}, {video_max:.3f}]")  # Should be ~[-1, 1]
    print(f"Action range: [{action_min:.3f}, {action_max:.3f}]")  # Should be ~[-3, 3]

    assert -1.1 <= video_min <= 0.0, f"Video min out of range: {video_min}"
    assert 0.0 <= video_max <= 1.1, f"Video max out of range: {video_max}"
    assert -3.1 <= action_min <= 3.1, f"Action min out of range: {action_min}"
    assert -3.1 <= action_max <= 3.1, f"Action max out of range: {action_max}"

    print("\n[PASS] Dataset test passed!")
    return True


def test_config():
    """Test 2: Load config file."""
    print("\n" + "=" * 60)
    print("TEST 2: Config Loading")
    print("=" * 60)

    from omegaconf import OmegaConf

    config_path = 'configs/manipulation/config_sf_fold_train.yaml'
    print(f"\nLoading config from: {config_path}")

    config = OmegaConf.load(config_path)

    # Verify key parameters
    action_dim = config.model.params.unet_config.params.action_dim
    print(f"action_dim: {action_dim}")
    assert action_dim == 20, f"action_dim should be 20, got {action_dim}"

    batch_size = config.data.params.batch_size
    print(f"batch_size: {batch_size}")
    assert batch_size == 3, f"batch_size should be 3, got {batch_size}"

    lr = config.model.base_learning_rate
    print(f"base_learning_rate: {lr}")
    assert lr == 5.0e-06, f"base_learning_rate should be 5.0e-06, got {lr}"

    max_steps = config.lightning.trainer.max_steps
    print(f"max_steps: {max_steps}")
    assert max_steps == 50000, f"max_steps should be 50000, got {max_steps}"

    precision = config.lightning.precision
    print(f"precision: {precision}")
    assert precision == 'bf16-mixed', f"precision should be bf16-mixed, got {precision}"

    action_dropout = config.model.params.action_dropout_prob
    print(f"action_dropout_prob: {action_dropout}")
    assert action_dropout == 0.15, f"action_dropout_prob should be 0.15, got {action_dropout}"

    data_target = config.data.params.train.target
    print(f"data target: {data_target}")
    assert 'sf_fold_vid.SFFoldVid' in data_target, f"data target should contain SFFoldVid"

    print("\n[PASS] Config test passed!")
    return True


def test_dataloader(data_dir):
    """Test 3: Test DataLoader integration."""
    print("\n" + "=" * 60)
    print("TEST 3: DataLoader Integration")
    print("=" * 60)

    import torch
    from torch.utils.data import DataLoader
    from lvdm.data.sf_fold_vid import SFFoldVid

    action_stats_path = os.path.join(data_dir, 'action_stats.json')
    val_file_list_path = os.path.join(data_dir, 'val_file_list.json')

    ds = SFFoldVid(
        data_dir=data_dir,
        resolution=[320, 512],
        video_length=16,
        spatial_transform='resize_center_crop',
        mode='training',
        augment_data=True,
        subsample=True,
        val_file_list_path=val_file_list_path,
        action_stats_path=action_stats_path,
    )

    loader = DataLoader(ds, batch_size=2, num_workers=0, shuffle=True)

    print("\nLoading batch from DataLoader...")
    batch = next(iter(loader))

    print(f"Batch video shape: {batch['video'].shape}")  # [2, 3, 16, 320, 512]
    print(f"Batch action shape: {batch['action'].shape}")  # [2, 16, 20]

    assert batch['video'].shape == (2, 3, 16, 320, 512), \
        f"Batch video shape mismatch: {batch['video'].shape}"
    assert batch['action'].shape == (2, 16, 20), \
        f"Batch action shape mismatch: {batch['action'].shape}"

    print("\n[PASS] DataLoader test passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test sf_fold pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to sf_fold NPZ directory')
    parser.add_argument('--skip_dataset', action='store_true',
                        help='Skip dataset tests (for config-only verification)')
    args = parser.parse_args()

    print("=" * 60)
    print("SF_FOLD PIPELINE VALIDATION TEST")
    print("=" * 60)

    all_passed = True

    # Test 1: Dataset
    if not args.skip_dataset:
        try:
            test_dataset(args.data_dir)
        except Exception as e:
            print(f"\n[FAIL] Dataset test failed: {e}")
            all_passed = False

    # Test 2: Config
    try:
        test_config()
    except Exception as e:
        print(f"\n[FAIL] Config test failed: {e}")
        all_passed = False

    # Test 3: DataLoader
    if not args.skip_dataset:
        try:
            test_dataloader(args.data_dir)
        except Exception as e:
            print(f"\n[FAIL] DataLoader test failed: {e}")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Update config placeholders with actual paths:")
        print("   - |<your_data_dir>| -> actual data directory")
        print("   - |<your_save_dir>| -> actual save directory")
        print("\n2. Run single-GPU forward pass test:")
        print("   python main/trainer.py \\")
        print("       --base configs/manipulation/config_sf_fold_train.yaml \\")
        print("       --train --devices 1 --name test_run --logdir /tmp/test")
        print("   # Kill after first batch succeeds")
        print("\n3. Launch full training (8 GPUs):")
        print("   python3 -m torch.distributed.launch \\")
        print("       --nproc_per_node=8 --nnodes=1 \\")
        print("       ./main/trainer.py \\")
        print("       --base configs/manipulation/config_sf_fold_train.yaml \\")
        print("       --train --name sf_fold_v1 --logdir /path/to/logs --devices 8")
    else:
        print("SOME TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
