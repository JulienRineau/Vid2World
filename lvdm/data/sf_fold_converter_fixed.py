#!/usr/bin/env python3
"""
SF-Fold Dataset Converter (FIXED): LeRobot v3 → NPZ Format

This is the CORRECTED version that properly accumulates deltas when subsampling.

CRITICAL FIX:
When subsampling from 30fps to 3fps (taking every 10th frame), the original
converter incorrectly took every 10th action (per-frame delta). This version
correctly ACCUMULATES the deltas across the stride window:
- Position deltas: summed across 10 frames
- Rotation deltas: multiplied as rotation matrices, converted back to 6D
- Gripper deltas: summed across 10 frames

Source dataset: trossen_folds_v30_256_fk_delta_relrot_ee6d20
- Delta pose format with 6D rotation representation
- 20D action space: [right_arm(10), left_arm(10)]
- Per arm: [delta_pos(3), delta_rot6d(6), delta_gripper(1)]

Usage:
    python sf_fold_converter_fixed.py --input_dir /path/to/lerobot_dataset \
                                      --output_dir /path/to/sf_fold_npz_cumulative \
                                      --num_workers 8
"""

import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, List, Dict, Any
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Video decoding
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# 6D ROTATION UTILITIES
# =============================================================================

def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    The 6D representation consists of the first two columns of the rotation
    matrix. The third column is computed as the cross product.

    Reference: Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks" (CVPR 2019)

    Args:
        rot6d: Array of shape [..., 6] containing [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]

    Returns:
        Array of shape [..., 3, 3] rotation matrix
    """
    # Extract and reshape
    shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)

    # First column (a1) and second column (a2) - raw vectors
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    # Gram-Schmidt orthonormalization
    # b1 = normalize(a1)
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)

    # b2 = normalize(a2 - (b1·a2)*b1)
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)

    # b3 = b1 × b2
    b3 = np.cross(b1, b2)

    # Construct rotation matrix
    R = np.stack([b1, b2, b3], axis=-1)  # [..., 3, 3]

    return R.reshape(*shape, 3, 3)


def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation.

    Args:
        R: Array of shape [..., 3, 3] rotation matrix

    Returns:
        Array of shape [..., 6] containing first two columns of R
    """
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    # Take first two columns
    rot6d = R[:, :, :2].reshape(-1, 6)  # Flatten [col1, col2]

    # Reorder to [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
    rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)

    return rot6d.reshape(*shape, 6)


def accumulate_rotation_deltas(rot6d_deltas: np.ndarray) -> np.ndarray:
    """Accumulate rotation deltas via matrix multiplication.

    For a sequence of delta rotations [R1, R2, ..., Rn], computes
    R_total = R1 @ R2 @ ... @ Rn

    Args:
        rot6d_deltas: Array of shape [N, 6] delta rotations in 6D format

    Returns:
        Array of shape [6] accumulated rotation in 6D format
    """
    if len(rot6d_deltas) == 0:
        # Identity rotation in 6D: [1,0,0, 0,1,0]
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    if len(rot6d_deltas) == 1:
        return rot6d_deltas[0]

    # Convert all deltas to rotation matrices
    R_deltas = rotation_6d_to_matrix(rot6d_deltas)  # [N, 3, 3]

    # Multiply in sequence: R_total = R_0 @ R_1 @ ... @ R_{n-1}
    # This represents applying rotations in order
    R_accumulated = R_deltas[0]
    for i in range(1, len(R_deltas)):
        R_accumulated = R_accumulated @ R_deltas[i]

    # Convert back to 6D
    return matrix_to_rotation_6d(R_accumulated.reshape(1, 3, 3))[0]


def accumulate_action_window(actions: np.ndarray, start_idx: int, stride: int) -> np.ndarray:
    """Accumulate action deltas across a stride window.

    Action format (20D total):
    - Right arm [0:10]: [delta_pos(3), delta_rot6d(6), delta_gripper(1)]
    - Left arm [10:20]: [delta_pos(3), delta_rot6d(6), delta_gripper(1)]

    Args:
        actions: Full action array of shape [T, 20]
        start_idx: Starting frame index
        stride: Number of frames to accumulate

    Returns:
        Accumulated action of shape [20]
    """
    end_idx = min(start_idx + stride, len(actions))
    window = actions[start_idx:end_idx]

    if len(window) == 0:
        return np.zeros(20, dtype=np.float32)

    if len(window) == 1:
        return window[0].copy()

    # Initialize accumulated action
    accumulated = np.zeros(20, dtype=np.float32)

    # Process each arm
    for arm_offset in [0, 10]:
        # Position delta (indices 0-2): sum
        pos_start = arm_offset
        pos_end = arm_offset + 3
        accumulated[pos_start:pos_end] = window[:, pos_start:pos_end].sum(axis=0)

        # Rotation delta (indices 3-8): accumulate via matrix multiplication
        rot_start = arm_offset + 3
        rot_end = arm_offset + 9
        accumulated[rot_start:rot_end] = accumulate_rotation_deltas(
            window[:, rot_start:rot_end]
        )

        # Gripper delta (index 9): sum
        grip_idx = arm_offset + 9
        accumulated[grip_idx] = window[:, grip_idx].sum()

    return accumulated


def subsample_actions_with_accumulation(actions: np.ndarray, stride: int) -> np.ndarray:
    """Subsample actions with proper delta accumulation.

    Args:
        actions: Full action array of shape [T, 20]
        stride: Subsample stride (e.g., 10 for 30fps → 3fps)

    Returns:
        Subsampled actions with accumulated deltas, shape [T//stride, 20]
    """
    n_output_frames = len(actions) // stride
    subsampled = np.zeros((n_output_frames, 20), dtype=np.float32)

    for i in range(n_output_frames):
        start_idx = i * stride
        subsampled[i] = accumulate_action_window(actions, start_idx, stride)

    return subsampled


# =============================================================================
# VIDEO LOADING (unchanged from original)
# =============================================================================

def load_episode_frames_av(
    video_path: str,
    from_timestamp: float,
    to_timestamp: float,
    fps: int,
    subsample_rate: int,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """Load and subsample frames from video segment using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]

    start_pts = int(from_timestamp / stream.time_base)
    container.seek(start_pts, stream=stream)

    frames = []
    frame_count = 0

    for frame in container.decode(stream):
        ts = float(frame.pts * stream.time_base)

        if ts < from_timestamp:
            continue
        if ts >= to_timestamp:
            break

        if frame_count % subsample_rate == 0:
            img = frame.to_image()
            if img.size != (target_size[1], target_size[0]):
                img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            frames.append(np.array(img))

        frame_count += 1

    container.close()
    return np.array(frames) if frames else np.array([])


def load_episode_frames_cv2(
    video_path: str,
    from_timestamp: float,
    to_timestamp: float,
    fps: int,
    subsample_rate: int,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """Load and subsample frames from video segment using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    start_frame = int(from_timestamp * fps)
    end_frame = int(to_timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_count = 0

    while cap.isOpened():
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % subsample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[:2] != target_size:
                frame = cv2.resize(frame, (target_size[1], target_size[0]),
                                   interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        frame_count += 1

    cap.release()
    return np.array(frames) if frames else np.array([])


def load_episode_frames(
    video_path: str,
    from_timestamp: float,
    to_timestamp: float,
    fps: int,
    subsample_rate: int,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """Load frames using best available decoder."""
    if HAS_AV:
        try:
            return load_episode_frames_av(
                video_path, from_timestamp, to_timestamp,
                fps, subsample_rate, target_size
            )
        except Exception as e:
            if HAS_CV2:
                warnings.warn(f"PyAV failed, falling back to OpenCV: {e}")
            else:
                raise

    if HAS_CV2:
        return load_episode_frames_cv2(
            video_path, from_timestamp, to_timestamp,
            fps, subsample_rate, target_size
        )

    raise ImportError("Neither PyAV nor OpenCV available for video decoding")


# =============================================================================
# EPISODE PROCESSING
# =============================================================================

def process_episode(args: Tuple) -> Optional[Dict[str, Any]]:
    """Process a single episode with CORRECTED cumulative delta accumulation."""
    episode_info, input_dir, output_dir, config = args
    episode_idx = episode_info['episode_index']

    output_file = os.path.join(output_dir, f"train_eps_{episode_idx:08d}.npz")

    if config.get('resume', True) and os.path.exists(output_file):
        return {'status': 'skipped', 'episode': episode_idx}

    try:
        # Load action data from parquet
        data_chunk = episode_info['data_chunk']
        data_file = episode_info['data_file']
        parquet_path = os.path.join(
            input_dir, 'data', f'chunk-{data_chunk:03d}', f'file-{data_file:03d}.parquet'
        )

        df = pd.read_parquet(parquet_path)
        episode_df = df[df['episode_index'] == episode_idx].copy()
        episode_df = episode_df.sort_values('frame_index')

        if len(episode_df) == 0:
            return {'status': 'error', 'episode': episode_idx, 'reason': 'no_data'}

        # Get actions
        actions = np.stack([np.array(a) for a in episode_df['action'].values], axis=0)

        # FIXED: Accumulate deltas across stride window instead of just subsampling
        subsample_rate = config.get('subsample_rate', 10)
        subsampled_actions = subsample_actions_with_accumulation(actions, subsample_rate)

        # Minimum sequence length check
        min_length = config.get('min_sequence_length', 16)
        if len(subsampled_actions) < min_length:
            return {'status': 'skipped', 'episode': episode_idx, 'reason': 'too_short'}

        # Load video frames
        camera_key = config.get('camera_key', 'observation.images.cam_high')
        video_chunk = episode_info[f'{camera_key}/chunk']
        video_file = episode_info[f'{camera_key}/file']
        from_ts = episode_info[f'{camera_key}/from_ts']
        to_ts = episode_info[f'{camera_key}/to_ts']

        video_path = os.path.join(
            input_dir, 'videos', camera_key,
            f'chunk-{video_chunk:03d}', f'file-{video_file:03d}.mp4'
        )

        if not os.path.exists(video_path):
            return {'status': 'error', 'episode': episode_idx, 'reason': 'video_not_found'}

        fps = config.get('fps', 30)
        target_size = tuple(config.get('target_size', [320, 512]))

        frames = load_episode_frames(
            video_path, from_ts, to_ts, fps, subsample_rate, target_size
        )

        if len(frames) == 0:
            return {'status': 'error', 'episode': episode_idx, 'reason': 'no_frames'}

        # Align lengths
        min_len = min(len(frames), len(subsampled_actions))
        frames = frames[:min_len]
        subsampled_actions = subsampled_actions[:min_len]

        if len(frames) < min_length:
            return {'status': 'skipped', 'episode': episode_idx, 'reason': 'too_short_after_align'}

        # Save NPZ
        np.savez_compressed(
            output_file,
            image=frames,
            action=subsampled_actions
        )

        return {
            'status': 'success',
            'episode': episode_idx,
            'frames': len(frames),
            'file': os.path.basename(output_file)
        }

    except Exception as e:
        return {'status': 'error', 'episode': episode_idx, 'reason': str(e)}


def load_episode_info(input_dir: str, camera_key: str = 'observation.images.cam_high') -> List[Dict]:
    """Load episode information from LeRobot v3 dataset."""
    episodes_dir = os.path.join(input_dir, 'meta', 'episodes')
    chunk_dirs = sorted([d for d in os.listdir(episodes_dir)
                        if os.path.isdir(os.path.join(episodes_dir, d))])

    episode_list = []

    for chunk_dir in chunk_dirs:
        chunk_path = os.path.join(episodes_dir, chunk_dir)
        parquet_files = sorted([f for f in os.listdir(chunk_path) if f.endswith('.parquet')])

        for pf in parquet_files:
            df = pd.read_parquet(os.path.join(chunk_path, pf))

            for _, row in df.iterrows():
                episode_list.append({
                    'episode_index': int(row['episode_index']),
                    'length': int(row['length']),
                    'data_chunk': int(row['data/chunk_index']),
                    'data_file': int(row['data/file_index']),
                    f'{camera_key}/chunk': int(row[f'videos/{camera_key}/chunk_index']),
                    f'{camera_key}/file': int(row[f'videos/{camera_key}/file_index']),
                    f'{camera_key}/from_ts': float(row[f'videos/{camera_key}/from_timestamp']),
                    f'{camera_key}/to_ts': float(row[f'videos/{camera_key}/to_timestamp']),
                })

    return episode_list


def compute_action_stats(output_dir: str, files: List[str]) -> Dict:
    """Compute mean and std for action normalization."""
    all_actions = []

    for f in tqdm(files, desc="Computing action stats", leave=False):
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            with np.load(path) as data:
                all_actions.append(data['action'])

    if not all_actions:
        raise ValueError("No valid NPZ files found for computing stats")

    all_actions = np.concatenate(all_actions, axis=0)

    return {
        'mean': all_actions.mean(axis=0).tolist(),
        'std': all_actions.std(axis=0).tolist()
    }


def create_val_split(files: List[str], val_ratio: float = 0.1, seed: int = 42) -> List[str]:
    """Create validation split."""
    np.random.seed(seed)
    files = sorted(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_indices = np.random.choice(len(files), size=n_val, replace=False)
    return [files[i] for i in sorted(val_indices)]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_cumulative_deltas(input_dir: str, output_dir: str, stride: int = 10,
                                num_samples: int = 3) -> Dict:
    """Validate that cumulative deltas are computed correctly.

    Compares position/rotation magnitudes between per-frame and cumulative.
    Expected: cumulative magnitudes should be roughly stride times larger.

    Args:
        input_dir: LeRobot dataset directory
        output_dir: NPZ output directory
        stride: Subsample stride
        num_samples: Number of episodes to validate

    Returns:
        Validation results dict
    """
    print("\n" + "=" * 60)
    print("VALIDATION: Cumulative Delta Correctness")
    print("=" * 60)

    # Find episodes to validate
    npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npz')])[:num_samples]

    if not npz_files:
        return {'valid': False, 'reason': 'no_npz_files'}

    results = {
        'valid': True,
        'samples': [],
        'summary': {}
    }

    pos_ratios = []
    rot_ratios = []
    grip_ratios = []

    for npz_file in npz_files:
        # Load cumulative actions from new dataset
        npz_path = os.path.join(output_dir, npz_file)
        with np.load(npz_path) as data:
            cumulative_actions = data['action']

        # Extract episode index
        episode_idx = int(npz_file.replace('train_eps_', '').replace('.npz', ''))

        # Load original per-frame actions from source
        episode_info = load_episode_info(input_dir)[episode_idx]
        data_chunk = episode_info['data_chunk']
        data_file = episode_info['data_file']
        parquet_path = os.path.join(
            input_dir, 'data', f'chunk-{data_chunk:03d}', f'file-{data_file:03d}.parquet'
        )

        df = pd.read_parquet(parquet_path)
        episode_df = df[df['episode_index'] == episode_idx].sort_values('frame_index')
        original_actions = np.stack([np.array(a) for a in episode_df['action'].values], axis=0)

        # Compare magnitudes for first 5 transitions
        sample_info = {
            'episode': episode_idx,
            'transitions': []
        }

        for i in range(min(5, len(cumulative_actions))):
            # Get cumulative action
            cum_action = cumulative_actions[i]

            # Get corresponding per-frame actions
            start_idx = i * stride
            end_idx = min(start_idx + stride, len(original_actions))
            per_frame_window = original_actions[start_idx:end_idx]

            if len(per_frame_window) == 0:
                continue

            # Compare position magnitudes (right arm)
            cum_pos_mag = np.linalg.norm(cum_action[0:3])
            per_frame_pos_mag = np.mean([np.linalg.norm(pf[0:3]) for pf in per_frame_window])
            pos_ratio = cum_pos_mag / (per_frame_pos_mag + 1e-8)

            # Compare gripper magnitudes (right arm)
            cum_grip = abs(cum_action[9])
            per_frame_grip = np.mean([abs(pf[9]) for pf in per_frame_window])
            grip_ratio = cum_grip / (per_frame_grip + 1e-8)

            if i < 3:  # First 3 transitions for display
                sample_info['transitions'].append({
                    'frame': i,
                    'cum_pos_mag': float(cum_pos_mag),
                    'per_frame_pos_mag': float(per_frame_pos_mag),
                    'pos_ratio': float(pos_ratio),
                    'cum_grip': float(cum_grip),
                    'per_frame_grip': float(per_frame_grip),
                    'grip_ratio': float(grip_ratio)
                })

            if per_frame_pos_mag > 1e-6:  # Only count non-zero movements
                pos_ratios.append(pos_ratio)
            if per_frame_grip > 1e-6:
                grip_ratios.append(grip_ratio)

        results['samples'].append(sample_info)

        # Print sample results
        print(f"\nEpisode {episode_idx}:")
        for t in sample_info['transitions']:
            print(f"  Frame {t['frame']}: pos_ratio={t['pos_ratio']:.2f}x, "
                  f"grip_ratio={t['grip_ratio']:.2f}x")

    # Compute summary statistics
    if pos_ratios:
        results['summary']['pos_ratio_mean'] = float(np.mean(pos_ratios))
        results['summary']['pos_ratio_std'] = float(np.std(pos_ratios))
    if grip_ratios:
        results['summary']['grip_ratio_mean'] = float(np.mean(grip_ratios))
        results['summary']['grip_ratio_std'] = float(np.std(grip_ratios))

    # Validation criteria: ratios should be approximately equal to stride
    expected_ratio = stride * 0.8  # Allow 20% tolerance

    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    if pos_ratios:
        mean_pos_ratio = np.mean(pos_ratios)
        print(f"Position magnitude ratio: {mean_pos_ratio:.2f}x (expected ~{stride}x)")
        if mean_pos_ratio < expected_ratio:
            results['valid'] = False
            print("  ⚠️  WARNING: Ratio too low - accumulation may be incorrect")
        else:
            print("  ✅ Position accumulation looks correct")

    if grip_ratios:
        mean_grip_ratio = np.mean(grip_ratios)
        print(f"Gripper magnitude ratio: {mean_grip_ratio:.2f}x (expected ~{stride}x)")
        if mean_grip_ratio < expected_ratio:
            results['valid'] = False
            print("  ⚠️  WARNING: Ratio too low - accumulation may be incorrect")
        else:
            print("  ✅ Gripper accumulation looks correct")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert SF-Fold dataset to NPZ format with CUMULATIVE DELTAS'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to LeRobot v3 dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for NPZ files')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum episodes to convert (for testing)')
    parser.add_argument('--subsample_rate', type=int, default=10,
                        help='Subsample rate (30fps/rate = output fps, default: 10 → 3fps)')
    parser.add_argument('--target_height', type=int, default=320,
                        help='Target image height')
    parser.add_argument('--target_width', type=int, default=512,
                        help='Target image width')
    parser.add_argument('--camera', type=str, default='observation.images.cam_high',
                        help='Camera key (observation.images.cam_high, etc.)')
    parser.add_argument('--min_length', type=int, default=16,
                        help='Minimum sequence length after subsampling')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--fps', type=int, default=30,
                        help='Source video FPS')
    parser.add_argument('--no_resume', action='store_true',
                        help='Disable resume (re-process existing files)')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip validation step')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        'subsample_rate': args.subsample_rate,
        'target_size': [args.target_height, args.target_width],
        'camera_key': args.camera,
        'min_sequence_length': args.min_length,
        'fps': args.fps,
        'resume': not args.no_resume,
    }

    print("=" * 60)
    print("SF-FOLD DATASET CONVERTER (FIXED - CUMULATIVE DELTAS)")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Camera: {args.camera}")
    print(f"Target size: {args.target_height}x{args.target_width}")
    print(f"Subsample: {args.fps}fps → {args.fps // args.subsample_rate}fps")
    print(f"Delta accumulation: ENABLED (stride={args.subsample_rate})")
    print(f"Workers: {args.num_workers}")
    print(f"Decoder: {'PyAV' if HAS_AV else 'OpenCV' if HAS_CV2 else 'NONE'}")
    print("=" * 60)

    print("\nLoading episode metadata...")
    episodes = load_episode_info(args.input_dir, args.camera)
    print(f"Found {len(episodes)} episodes")

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"Limited to {len(episodes)} episodes")

    worker_args = [
        (ep, args.input_dir, args.output_dir, config)
        for ep in episodes
    ]

    print(f"\nConverting episodes with cumulative delta accumulation...")
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_episode, worker_args),
                total=len(worker_args),
                desc="Converting"
            ))
    else:
        results = [process_episode(arg) for arg in tqdm(worker_args, desc="Converting")]

    successful = [r for r in results if r and r['status'] == 'success']
    skipped = [r for r in results if r and r['status'] == 'skipped']
    errors = [r for r in results if r and r['status'] == 'error']

    print(f"\n{'=' * 60}")
    print("CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successful: {len(successful)}")
    print(f"Skipped:    {len(skipped)}")
    print(f"Errors:     {len(errors)}")

    if errors:
        print("\nError details:")
        for e in errors[:10]:
            print(f"  Episode {e['episode']}: {e.get('reason', 'unknown')}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    converted_files = [r['file'] for r in successful if 'file' in r]

    if not converted_files:
        print("\nNo files converted. Check paths and video codec support.")
        return

    # Compute action stats
    print("\nComputing action statistics...")
    stats = compute_action_stats(args.output_dir, converted_files)
    stats_path = os.path.join(args.output_dir, 'action_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    print("\nAction statistics (first 5 dims):")
    print(f"  Mean: {[f'{x:.6f}' for x in stats['mean'][:5]]}")
    print(f"  Std:  {[f'{x:.6f}' for x in stats['std'][:5]]}")

    # Create validation split
    print("\nCreating validation split...")
    val_files = create_val_split(converted_files, args.val_ratio)
    val_path = os.path.join(args.output_dir, 'val_file_list.json')
    with open(val_path, 'w') as f:
        json.dump(val_files, f, indent=2)
    print(f"Saved: {val_path}")

    # Validation
    if not args.skip_validation:
        validation_results = validate_cumulative_deltas(
            args.input_dir, args.output_dir, args.subsample_rate
        )

        validation_path = os.path.join(args.output_dir, 'validation_results.json')
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\nValidation results saved: {validation_path}")

    print(f"\n{'=' * 60}")
    print("CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes:      {len(converted_files)}")
    print(f"Training episodes:   {len(converted_files) - len(val_files)}")
    print(f"Validation episodes: {len(val_files)}")
    print(f"Output directory:    {args.output_dir}")
    print(f"\nKey fix applied: Actions now use CUMULATIVE deltas")
    print(f"                 (summed over {args.subsample_rate} frames)")


if __name__ == '__main__':
    main()
