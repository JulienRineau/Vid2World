#!/usr/bin/env python3
"""
SF-Fold Dataset Converter: LeRobot v3 → NPZ Format

Converts the Trossen folding dataset (LeRobot v3 format) to NPZ files
compatible with the Vid2World training pipeline.

Source dataset: trossen_folds_v30_256_fk_delta_relrot_ee6d20
- Already in delta pose format with 6D rotation representation
- 20D action space: [right_arm(10), left_arm(10)]
- Per arm: [delta_pos(3), delta_rot6d(6), delta_gripper(1)]

LeRobot v3 format:
- Multiple episodes concatenated in each video file
- Episode boundaries defined by timestamps in meta/episodes/*.parquet
- Actions stored in data/chunk-*/file-*.parquet

Output format:
- NPZ files with 'image': [T, H, W, C] and 'action': [T, 20]
- Subsampled to 3 fps from 30 fps source
- Images resized to 320x512

Usage:
    python sf_fold_converter.py --input_dir /path/to/lerobot_dataset \
                                --output_dir /path/to/sf_fold_npz \
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


def load_episode_frames_av(
    video_path: str,
    from_timestamp: float,
    to_timestamp: float,
    fps: int,
    subsample_rate: int,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """Load and subsample frames from video segment using PyAV.

    Args:
        video_path: Path to MP4 video file
        from_timestamp: Start timestamp in seconds
        to_timestamp: End timestamp in seconds
        fps: Source video FPS
        subsample_rate: Take every nth frame
        target_size: (height, width) for resizing

    Returns:
        Array of shape [T, H, W, 3] where T = num_frames // subsample_rate
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Seek to start timestamp
    start_pts = int(from_timestamp / stream.time_base)
    container.seek(start_pts, stream=stream)

    # Calculate frame range
    start_frame = int(from_timestamp * fps)
    end_frame = int(to_timestamp * fps)
    total_frames = end_frame - start_frame

    frames = []
    frame_count = 0

    for frame in container.decode(stream):
        # Get timestamp in seconds
        ts = float(frame.pts * stream.time_base)

        # Skip frames before our range
        if ts < from_timestamp:
            continue

        # Stop after our range
        if ts >= to_timestamp:
            break

        # Subsample
        if frame_count % subsample_rate == 0:
            img = frame.to_image()  # PIL Image
            if img.size != (target_size[1], target_size[0]):  # PIL uses (W, H)
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
    """Load and subsample frames from video segment using OpenCV.

    Args:
        video_path: Path to MP4 video file
        from_timestamp: Start timestamp in seconds
        to_timestamp: End timestamp in seconds
        fps: Source video FPS
        subsample_rate: Take every nth frame
        target_size: (height, width) for resizing

    Returns:
        Array of shape [T, H, W, 3] where T = num_frames // subsample_rate
    """
    cap = cv2.VideoCapture(video_path)

    # Seek to start frame
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

        # Subsample
        if frame_count % subsample_rate == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize if needed
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


def process_episode(args: Tuple) -> Optional[Dict[str, Any]]:
    """Process a single episode: extract frames and actions, save as NPZ.

    Args:
        args: Tuple of (episode_info, input_dir, output_dir, config)

    Returns:
        Dict with stats if successful, None if failed or skipped
    """
    episode_info, input_dir, output_dir, config = args
    episode_idx = episode_info['episode_index']

    output_file = os.path.join(output_dir, f"train_eps_{episode_idx:08d}.npz")

    # Skip if already exists and resume mode is on
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

        # Subsample actions
        subsample_rate = config.get('subsample_rate', 10)
        subsampled_actions = actions[::subsample_rate]

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

        # Construct video path
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

        # Align lengths (may differ by 1-2 frames due to rounding)
        min_len = min(len(frames), len(subsampled_actions))
        frames = frames[:min_len]
        subsampled_actions = subsampled_actions[:min_len]

        if len(frames) < min_length:
            return {'status': 'skipped', 'episode': episode_idx, 'reason': 'too_short_after_align'}

        # Save NPZ
        np.savez_compressed(
            output_file,
            image=frames,              # [T, H, W, C]
            action=subsampled_actions  # [T, 20]
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
    """Load episode information from LeRobot v3 dataset.

    Args:
        input_dir: Path to LeRobot dataset directory
        camera_key: Camera to use for video extraction

    Returns:
        List of episode info dicts
    """
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
    """Compute mean and std for action normalization.

    Args:
        output_dir: Directory containing NPZ files
        files: List of NPZ filenames

    Returns:
        Dict with 'mean' and 'std' arrays of shape [20]
    """
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
    """Create validation split.

    Args:
        files: List of all NPZ filenames
        val_ratio: Fraction of files for validation
        seed: Random seed for reproducibility

    Returns:
        List of validation filenames
    """
    np.random.seed(seed)
    files = sorted(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_indices = np.random.choice(len(files), size=n_val, replace=False)
    return [files[i] for i in sorted(val_indices)]


def main():
    parser = argparse.ArgumentParser(description='Convert SF-Fold dataset to NPZ format')
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
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Config for workers
    config = {
        'subsample_rate': args.subsample_rate,
        'target_size': [args.target_height, args.target_width],
        'camera_key': args.camera,
        'min_sequence_length': args.min_length,
        'fps': args.fps,
        'resume': not args.no_resume,
    }

    print("=" * 60)
    print("SF-FOLD DATASET CONVERTER")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Camera: {args.camera}")
    print(f"Target size: {args.target_height}x{args.target_width}")
    print(f"Subsample: {args.fps}fps → {args.fps // args.subsample_rate}fps")
    print(f"Workers: {args.num_workers}")
    print(f"Decoder: {'PyAV' if HAS_AV else 'OpenCV' if HAS_CV2 else 'NONE'}")
    print("=" * 60)

    # Load episode info
    print("\nLoading episode metadata...")
    episodes = load_episode_info(args.input_dir, args.camera)
    print(f"Found {len(episodes)} episodes")

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"Limited to {len(episodes)} episodes")

    # Prepare worker arguments
    worker_args = [
        (ep, args.input_dir, args.output_dir, config)
        for ep in episodes
    ]

    # Process episodes
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

    # Summarize results
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

    # Print stats summary
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

    # Final summary
    print(f"\n{'=' * 60}")
    print("CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes:      {len(converted_files)}")
    print(f"Training episodes:   {len(converted_files) - len(val_files)}")
    print(f"Validation episodes: {len(val_files)}")
    print(f"Output directory:    {args.output_dir}")


if __name__ == '__main__':
    main()
