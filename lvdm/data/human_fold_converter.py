#!/usr/bin/env python3
"""
Human T-Shirt Folding Dataset Converter: LeRobot v3 → NPZ Format

Converts the ZeroShot human t-shirt folding dataset from LeRobot v3 format
to NPZ format compatible with Vid2World training.

Source: human_folds_npz/ (LeRobot v3 with absolute quaternion poses)
Target: human_fold_npz/ (NPZ with delta 6D actions)

Key conversions:
- Absolute quaternion poses → frame-to-frame delta poses
- Quaternion rotations → 6D rotation representation
- Absolute gripper widths → delta gripper changes
- 30 FPS → 3 FPS with cumulative delta accumulation

Action format (20D):
    Left Arm [0:10]:
        [0:3]   delta_position (meters)
        [3:9]   delta_rotation_6d (first two columns of rotation matrix)
        [9]     delta_gripper_width
    Right Arm [10:20]:
        [10:13] delta_position (meters)
        [13:19] delta_rotation_6d
        [19]    delta_gripper_width

Usage:
    python human_fold_converter.py --input_dir /path/to/human_folds_npz \
                                   --output_dir /path/to/human_fold_npz \
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
# QUATERNION UTILITIES
# =============================================================================

def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute conjugate (inverse for unit quaternions) of quaternion.

    Args:
        q: Quaternion array [..., 4] in format [qx, qy, qz, qw]

    Returns:
        Conjugate quaternion [-qx, -qy, -qz, qw]
    """
    q_conj = q.copy()
    q_conj[..., :3] = -q_conj[..., :3]  # Negate xyz components
    return q_conj


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions using Hamilton product.

    Args:
        q1, q2: Quaternion arrays [..., 4] in format [qx, qy, qz, qw]

    Returns:
        Product quaternion q1 * q2
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.stack([x, y, z, w], axis=-1)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion array [..., 4] in format [qx, qy, qz, qw]

    Returns:
        Rotation matrix [..., 3, 3]
    """
    # Normalize quaternion
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Rotation matrix from quaternion
    R = np.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype)

    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - z*w)
    R[..., 0, 2] = 2*(x*z + y*w)

    R[..., 1, 0] = 2*(x*y + z*w)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - x*w)

    R[..., 2, 0] = 2*(x*z - y*w)
    R[..., 2, 1] = 2*(y*z + x*w)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)

    return R


# =============================================================================
# 6D ROTATION UTILITIES
# =============================================================================

def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    The 6D representation consists of the first two columns of the rotation
    matrix. The third column is computed via Gram-Schmidt orthonormalization.

    Reference: Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks" (CVPR 2019)

    Args:
        rot6d: Array of shape [..., 6] containing [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]

    Returns:
        Array of shape [..., 3, 3] rotation matrix
    """
    shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)

    # First column (a1) and second column (a2) - raw vectors
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    # Gram-Schmidt orthonormalization
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)

    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)

    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=-1)
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

    # Take first two columns and concatenate
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
    R_deltas = rotation_6d_to_matrix(rot6d_deltas)

    # Multiply in sequence
    R_accumulated = R_deltas[0]
    for i in range(1, len(R_deltas)):
        R_accumulated = R_accumulated @ R_deltas[i]

    return matrix_to_rotation_6d(R_accumulated.reshape(1, 3, 3))[0]


# =============================================================================
# DELTA COMPUTATION
# =============================================================================

def compute_pose_deltas(poses: np.ndarray, gripper_widths: np.ndarray) -> np.ndarray:
    """Compute per-frame deltas from absolute poses.

    Args:
        poses: [T, 7] absolute poses [x, y, z, qx, qy, qz, qw]
        gripper_widths: [T] absolute gripper widths in meters

    Returns:
        deltas: [T, 10] per-frame deltas [delta_pos(3), delta_rot6d(6), delta_grip(1)]
    """
    T = len(poses)
    deltas = np.zeros((T, 10), dtype=np.float32)

    for t in range(T):
        if t == 0:
            # First frame: zero delta (no previous frame)
            deltas[t, :3] = 0.0  # Position delta
            deltas[t, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Identity rotation in 6D
            deltas[t, 9] = 0.0  # Gripper delta
        else:
            # Position delta: p[t] - p[t-1]
            deltas[t, :3] = poses[t, :3] - poses[t-1, :3]

            # Rotation delta: q[t] * inv(q[t-1])
            q_curr = poses[t, 3:7]
            q_prev = poses[t-1, 3:7]
            q_prev_inv = quaternion_conjugate(q_prev)
            q_delta = quaternion_multiply(q_curr, q_prev_inv)

            # Convert quaternion delta to 6D
            R_delta = quaternion_to_matrix(q_delta)
            deltas[t, 3:9] = matrix_to_rotation_6d(R_delta.reshape(1, 3, 3))[0]

            # Gripper delta
            deltas[t, 9] = gripper_widths[t] - gripper_widths[t-1]

    return deltas


def accumulate_action_window(deltas: np.ndarray, start_idx: int, stride: int) -> np.ndarray:
    """Accumulate action deltas across a stride window.

    Action format (10D per arm):
    - [0:3]: delta_position (sum)
    - [3:9]: delta_rotation_6d (multiply matrices)
    - [9]: delta_gripper (sum)

    Args:
        deltas: Full delta array of shape [T, 10]
        start_idx: Starting frame index
        stride: Number of frames to accumulate

    Returns:
        Accumulated action of shape [10]
    """
    end_idx = min(start_idx + stride, len(deltas))
    window = deltas[start_idx:end_idx]

    if len(window) == 0:
        return np.zeros(10, dtype=np.float32)

    if len(window) == 1:
        return window[0].copy()

    accumulated = np.zeros(10, dtype=np.float32)

    # Position delta: sum
    accumulated[:3] = window[:, :3].sum(axis=0)

    # Rotation delta: accumulate via matrix multiplication
    accumulated[3:9] = accumulate_rotation_deltas(window[:, 3:9])

    # Gripper delta: sum
    accumulated[9] = window[:, 9].sum()

    return accumulated


def subsample_deltas_with_accumulation(deltas: np.ndarray, stride: int) -> np.ndarray:
    """Subsample deltas with proper accumulation.

    Args:
        deltas: Full delta array of shape [T, 10]
        stride: Subsample stride (e.g., 10 for 30fps → 3fps)

    Returns:
        Subsampled deltas with accumulated values, shape [T//stride, 10]
    """
    n_output_frames = len(deltas) // stride
    subsampled = np.zeros((n_output_frames, 10), dtype=np.float32)

    for i in range(n_output_frames):
        start_idx = i * stride
        subsampled[i] = accumulate_action_window(deltas, start_idx, stride)

    return subsampled


# =============================================================================
# VIDEO LOADING
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
    """Process a single episode: compute deltas, subsample, save NPZ."""
    episode_info, input_dir, output_dir, config = args
    episode_idx = episode_info['episode_index']

    output_file = os.path.join(output_dir, f"train_eps_{episode_idx:08d}.npz")

    if config.get('resume', True) and os.path.exists(output_file):
        return {'status': 'skipped', 'episode': episode_idx}

    try:
        # Load pose data from parquet
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

        # Extract poses and gripper widths
        left_poses = np.stack([np.array(p) for p in episode_df['left_fingertip_pose'].values])
        right_poses = np.stack([np.array(p) for p in episode_df['right_fingertip_pose'].values])
        gripper_widths = np.stack([np.array(g) for g in episode_df['gripper_width'].values])

        # Compute per-frame deltas
        left_deltas = compute_pose_deltas(left_poses, gripper_widths[:, 0])
        right_deltas = compute_pose_deltas(right_poses, gripper_widths[:, 1])

        # Combine into 20D action vector [left(10), right(10)]
        full_deltas = np.concatenate([left_deltas, right_deltas], axis=1)

        # Subsample with accumulation
        subsample_rate = config.get('subsample_rate', 10)
        subsampled_actions = np.zeros((len(full_deltas) // subsample_rate, 20), dtype=np.float32)

        for i in range(len(subsampled_actions)):
            start_idx = i * subsample_rate
            # Left arm
            subsampled_actions[i, :10] = accumulate_action_window(left_deltas, start_idx, subsample_rate)
            # Right arm
            subsampled_actions[i, 10:] = accumulate_action_window(right_deltas, start_idx, subsample_rate)

        # Minimum sequence length check
        min_length = config.get('min_sequence_length', 16)
        if len(subsampled_actions) < min_length:
            return {'status': 'skipped', 'episode': episode_idx, 'reason': 'too_short'}

        # Load video frames
        camera_key = config.get('camera_key', 'observation.images.cam_ego')
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


def load_episode_info(input_dir: str, camera_key: str = 'observation.images.cam_ego') -> List[Dict]:
    """Load episode information from LeRobot v3 dataset."""
    episodes_dir = os.path.join(input_dir, 'meta', 'episodes')

    if not os.path.exists(episodes_dir):
        raise ValueError(f"Episodes directory not found: {episodes_dir}")

    chunk_dirs = sorted([d for d in os.listdir(episodes_dir)
                        if os.path.isdir(os.path.join(episodes_dir, d))])

    episode_list = []

    for chunk_dir in chunk_dirs:
        chunk_path = os.path.join(episodes_dir, chunk_dir)
        parquet_files = sorted([f for f in os.listdir(chunk_path) if f.endswith('.parquet')])

        for pf in parquet_files:
            df = pd.read_parquet(os.path.join(chunk_path, pf))

            for _, row in df.iterrows():
                try:
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
                except KeyError as e:
                    warnings.warn(f"Missing column {e} for episode {row.get('episode_index', 'unknown')}")
                    continue

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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert Human T-Shirt Folding dataset to NPZ format'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to LeRobot v3 dataset directory (human_folds_npz)')
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
    parser.add_argument('--camera', type=str, default='observation.images.cam_ego',
                        help='Camera key to use')
    parser.add_argument('--min_length', type=int, default=16,
                        help='Minimum sequence length after subsampling')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--fps', type=int, default=30,
                        help='Source video FPS')
    parser.add_argument('--no_resume', action='store_true',
                        help='Disable resume (re-process existing files)')
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
    print("HUMAN T-SHIRT FOLDING DATASET CONVERTER")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Camera: {args.camera}")
    print(f"Target size: {args.target_height}x{args.target_width}")
    print(f"Subsample: {args.fps}fps → {args.fps // args.subsample_rate}fps")
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

    print(f"\nConverting episodes...")
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

    print(f"\n{'=' * 60}")
    print("CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes:      {len(converted_files)}")
    print(f"Training episodes:   {len(converted_files) - len(val_files)}")
    print(f"Validation episodes: {len(val_files)}")
    print(f"Output directory:    {args.output_dir}")


if __name__ == '__main__':
    main()
