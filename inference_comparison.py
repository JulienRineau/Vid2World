#!/usr/bin/env python
"""
inference_comparison.py - Generate side-by-side comparison video

Creates videos showing: Initial Frame | Ground Truth | Model Prediction
For use as the opening "wow effect" video in the blog post.

Usage:
    python inference_comparison.py \
        --checkpoint logs/human_fold_v1/checkpoints/epoch=449-step=44500.ckpt \
        --config configs/manipulation/config_human_fold_test.yaml \
        --output comparison_video.mp4 \
        --num_samples 4
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
from einops import repeat
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import instantiate_from_config
from lvdm.data.sf_fold_vid import SFFoldVid


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load trained model from checkpoint."""
    print(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)

    print(f"Instantiating model...")
    model = instantiate_from_config(config.model)

    print(f"Loading checkpoint from {checkpoint_path}")
    pl_sd = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    elif "module" in pl_sd:
        from collections import OrderedDict
        sd = OrderedDict()
        for key in pl_sd["module"].keys():
            sd[key[16:]] = pl_sd["module"][key]
    else:
        sd = pl_sd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys")

    model = model.to(device)
    model.eval()
    return model


def load_dataset(config_path: str, mode: str = "training") -> SFFoldVid:
    """Load dataset from config."""
    config = OmegaConf.load(config_path)
    data_config = config.data.params.validation
    data_config.params.mode = mode

    dataset = instantiate_from_config(data_config)
    print(f"Loaded {mode} dataset with {len(dataset)} samples")
    return dataset


def prepare_single_batch(sample: dict, device: str = "cuda") -> dict:
    """Prepare batch from a single sample."""
    batch = {
        'video': sample['video'].unsqueeze(0).to(device),
        'action': sample['action'].unsqueeze(0).to(device),
        'caption': [sample.get('caption', '')],
        'fps': torch.tensor([sample.get('fps', 3)]).to(device),
        'frame_stride': torch.tensor([sample.get('frame_stride', 1)]).to(device),
    }
    return batch


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    """Normalize video from [-1, 1] to [0, 1]."""
    video = video.detach().cpu()
    video = torch.clamp(video, -1, 1)
    return (video + 1.0) / 2.0


def add_text_to_frame(frame: np.ndarray, texts: list, positions: list, font_size: int = 20) -> np.ndarray:
    """Add text labels to a frame.

    Args:
        frame: RGB frame as numpy array [H, W, C]
        texts: List of text strings to add
        positions: List of (x, y) positions for each text
        font_size: Font size for text

    Returns:
        Frame with text added
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to use a nicer font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for text, (x, y) in zip(texts, positions):
        # Draw text with black outline for visibility
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def create_comparison_video(
    all_initial: list,
    all_gt: list,
    all_pred: list,
    output_path: str,
    fps: int = 4
) -> None:
    """Create side-by-side comparison video with labels.

    Args:
        all_initial: List of initial frame videos [C, T, H, W] each
        all_gt: List of ground truth videos [C, T, H, W] each
        all_pred: List of prediction videos [C, T, H, W] each
        output_path: Output video path
        fps: Frames per second
    """
    num_samples = len(all_gt)
    C, T, H, W = all_gt[0].shape

    # Calculate label positions (centered in each column)
    col_width = W
    label_y = 5
    labels = ["Initial Frame", "Ground Truth", "Prediction"]

    all_frames = []

    for t in range(T):
        rows = []
        for i in range(num_samples):
            # Get frame t from each video
            init_frame = all_initial[i][:, t, :, :]  # [C, H, W]
            gt_frame = all_gt[i][:, t, :, :]
            pred_frame = all_pred[i][:, t, :, :]

            # Concatenate horizontally: [C, H, W*3]
            row = torch.cat([init_frame, gt_frame, pred_frame], dim=-1)

            # Convert to numpy [H, W*3, C]
            row_np = (row.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Add labels only on first frame of each sample
            if t == 0 or True:  # Always show labels
                positions = [
                    (col_width // 2 - 50, label_y),
                    (col_width + col_width // 2 - 55, label_y),
                    (2 * col_width + col_width // 2 - 40, label_y)
                ]
                row_np = add_text_to_frame(row_np, labels, positions, font_size=18)

            rows.append(row_np)

        # Stack all rows vertically
        frame = np.concatenate(rows, axis=0)  # [num_samples*H, W*3, C]
        all_frames.append(frame)

    # Convert to tensor for video writing
    video = np.stack(all_frames, axis=0)  # [T, H_total, W_total, C]
    video = torch.from_numpy(video)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    torchvision.io.write_video(
        output_path,
        video,
        fps=fps,
        video_codec='h264',
        options={'crf': '18'}
    )
    print(f"Saved comparison video to {output_path}")
    print(f"  Resolution: {video.shape[2]}x{video.shape[1]} (WxH)")
    print(f"  Samples: {num_samples}, Frames: {T}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison video: Initial | GT | Prediction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--output", type=str, default="outputs/comparison_video.mp4", help="Output video path")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to include")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if not set)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fps", type=int, default=4, help="Output video FPS")
    parser.add_argument("--ar", action="store_true", help="Use autoregressive generation")
    parser.add_argument("--start_idx", type=int, default=None, help="Starting index in dataset (default: random)")
    parser.add_argument("--mode", type=str, default="training", choices=["training", "validation"], help="Dataset mode")
    parser.add_argument("--random_samples", action="store_true", help="Randomly select samples from dataset")
    parser.add_argument("--num_candidates", type=int, default=1, help="Generate N candidates per sample for cherry-picking")
    args = parser.parse_args()

    # Set seed (random if not provided)
    if args.seed is None:
        args.seed = np.random.randint(0, 2**31)
    print(f"Using seed: {args.seed}  (use --seed {args.seed} to reproduce)")
    seed_everything(args.seed)

    # Load model
    model = load_model(args.config, args.checkpoint, args.device)

    # Load dataset
    dataset = load_dataset(args.config, mode=args.mode)

    # Select sample indices
    if args.random_samples or args.start_idx is None:
        # Random selection
        indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False).tolist()
        print(f"Randomly selected indices: {indices}")
    else:
        # Sequential from start_idx
        indices = list(range(args.start_idx, min(args.start_idx + args.num_samples, len(dataset))))
        if len(indices) < args.num_samples:
            print(f"Warning: Only {len(indices)} samples available starting from index {args.start_idx}")

    print(f"\nGenerating predictions for {len(indices)} samples...")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Autoregressive: {args.ar}")

    all_initial = []
    all_gt = []
    all_pred = []

    # Process each sample individually
    for i, idx in enumerate(tqdm(indices, desc="Processing samples")):
        sample = dataset[idx]
        batch = prepare_single_batch(sample, args.device)

        # Run inference
        with torch.no_grad():
            with model.ema_scope("Inference"):
                batch_logs = model.log_images(
                    batch,
                    sample=True,
                    ddim_steps=args.ddim_steps,
                    ddim_eta=1.0,
                    unconditional_guidance_scale=args.guidance_scale,
                    ar=args.ar,
                    cond_frame=1 if args.ar else None,
                )

        # Extract and normalize
        pred = normalize_video(batch_logs['samples'][0])  # [C, T, H, W]
        gt = normalize_video(batch['video'][0])  # [C, T, H, W]

        # Create initial frames (first frame repeated)
        T = gt.shape[1]
        init = gt[:, 0:1, :, :].repeat(1, T, 1, 1)  # [C, T, H, W]

        all_initial.append(init)
        all_gt.append(gt)
        all_pred.append(pred)

    # Create comparison video
    create_comparison_video(
        all_initial,
        all_gt,
        all_pred,
        args.output,
        fps=args.fps
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
