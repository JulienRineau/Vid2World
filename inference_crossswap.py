#!/usr/bin/env python
"""
inference_crossswap.py - Generate 2x2 action cross-swap grid

Demonstrates action-conditioned generation by swapping action sequences
between two episodes. The diagonal shows expected behavior (matching actions),
while off-diagonal shows cross-swap (proves model follows actions, not memorizing).

Grid layout:
    ┌─────────────────┬─────────────────┐
    │   Actions A     │   Actions B     │
    ├─────────────────┼─────────────────┤
    │ Frame A₀        │ Frame A₀        │
    │ (Expected)      │ (Cross-swap)    │
    ├─────────────────┼─────────────────┤
    │ Frame B₀        │ Frame B₀        │
    │ (Cross-swap)    │ (Expected)      │
    └─────────────────┴─────────────────┘

Usage:
    python inference_crossswap.py \
        --checkpoint logs/human_fold_v1/checkpoints/epoch=449-step=44500.ckpt \
        --config configs/manipulation/config_human_fold_test.yaml \
        --output crossswap_grid.mp4 \
        --episode_a_idx 0 \
        --episode_b_idx 10
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


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    """Normalize video from [-1, 1] to [0, 1]."""
    video = video.detach().cpu()
    video = torch.clamp(video, -1, 1)
    return (video + 1.0) / 2.0


def add_text_to_frame(frame: np.ndarray, texts: list, positions: list,
                      font_size: int = 20, colors: list = None) -> np.ndarray:
    """Add text labels to a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    if colors is None:
        colors = [(255, 255, 255)] * len(texts)

    for text, (x, y), color in zip(texts, positions, colors):
        # Draw text with black outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=color, font=font)

    return np.array(img)


def run_single_inference(model, video, action, device, ddim_steps, guidance_scale, ar):
    """Run inference for a single video-action pair."""
    batch = {
        'video': video.unsqueeze(0).to(device),
        'action': action.unsqueeze(0).to(device),
        'caption': [''],
        'fps': torch.tensor([3]).to(device),
        'frame_stride': torch.tensor([1]).to(device),
    }

    with torch.no_grad():
        with model.ema_scope("Inference"):
            batch_logs = model.log_images(
                batch,
                sample=True,
                ddim_steps=ddim_steps,
                ddim_eta=1.0,
                unconditional_guidance_scale=guidance_scale,
                ar=ar,
                cond_frame=1 if ar else None,
            )

    return normalize_video(batch_logs['samples'][0])  # [C, T, H, W]


def create_grid_video(
    pred_aa: torch.Tensor,
    pred_ab: torch.Tensor,
    pred_ba: torch.Tensor,
    pred_bb: torch.Tensor,
    output_path: str,
    fps: int = 4
) -> None:
    """Create 2x2 grid video with labels.

    Grid layout:
        |  Actions A  |  Actions B  |
    ----|-------------|-------------|
    A   |   Expected  | Cross-swap  |
    ----|-------------|-------------|
    B   | Cross-swap  |  Expected   |
    """
    C, T, H, W = pred_aa.shape

    # Labels for each cell - clear description of what's shown
    cell_labels = [
        ["Frame A + Actions A", "(GT)"],       # top-left: ground truth
        ["Frame A + Actions B", "(Swapped)"],  # top-right: swapped
        ["Frame B + Actions A", "(Swapped)"],  # bottom-left: swapped
        ["Frame B + Actions B", "(GT)"],       # bottom-right: ground truth
    ]

    # Colors: green for GT (diagonal), cyan for swapped (off-diagonal)
    cell_colors = [
        [(255, 255, 255), (0, 255, 0)],    # GT - green
        [(255, 255, 255), (0, 255, 255)],  # Swapped - cyan
        [(255, 255, 255), (0, 255, 255)],  # Swapped - cyan
        [(255, 255, 255), (0, 255, 0)],    # GT - green
    ]

    preds = [pred_aa, pred_ab, pred_ba, pred_bb]

    all_frames = []

    for t in range(T):
        # Get frame t from each prediction
        frames = [p[:, t, :, :] for p in preds]  # Each is [C, H, W]

        # Convert to numpy
        frames_np = []
        for i, f in enumerate(frames):
            f_np = (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Add labels - both on top
            labels = cell_labels[i]
            positions = [
                (W // 2 - 75, 5),   # Main label (top line)
                (W // 2 - 35, 25),  # GT/Swapped (second line)
            ]
            f_np = add_text_to_frame(f_np, labels, positions, font_size=16, colors=cell_colors[i])
            frames_np.append(f_np)

        # Arrange in 2x2 grid
        row1 = np.concatenate([frames_np[0], frames_np[1]], axis=1)  # [H, W*2, C]
        row2 = np.concatenate([frames_np[2], frames_np[3]], axis=1)
        grid = np.concatenate([row1, row2], axis=0)  # [H*2, W*2, C]

        all_frames.append(grid)

    # Convert to tensor
    video = np.stack(all_frames, axis=0)  # [T, H*2, W*2, C]
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
    print(f"Saved cross-swap grid video to {output_path}")
    print(f"  Resolution: {video.shape[2]}x{video.shape[1]} (WxH)")


def main():
    parser = argparse.ArgumentParser(description="Generate 2x2 action cross-swap grid")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--output", type=str, default="outputs/crossswap_grid.mp4", help="Output video path")
    parser.add_argument("--episode_a_idx", type=int, default=None, help="Index of first episode (default: random)")
    parser.add_argument("--episode_b_idx", type=int, default=None, help="Index of second episode (default: random)")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if not set)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fps", type=int, default=4, help="Output video FPS")
    parser.add_argument("--ar", action="store_true", help="Use autoregressive generation")
    parser.add_argument("--mode", type=str, default="training", choices=["training", "validation"], help="Dataset mode")
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

    # Select episodes (random if not specified)
    if args.episode_a_idx is None:
        args.episode_a_idx = np.random.randint(0, len(dataset))
    if args.episode_b_idx is None:
        args.episode_b_idx = np.random.randint(0, len(dataset))
        while args.episode_b_idx == args.episode_a_idx:
            args.episode_b_idx = np.random.randint(0, len(dataset))

    if args.episode_a_idx == args.episode_b_idx:
        print("Warning: episode_a_idx and episode_b_idx are the same. Using different episodes is recommended.")

    if args.episode_a_idx >= len(dataset) or args.episode_b_idx >= len(dataset):
        raise ValueError(f"Episode indices must be < {len(dataset)}")

    # Get samples
    sample_a = dataset[args.episode_a_idx]
    sample_b = dataset[args.episode_b_idx]

    print(f"\nSelected episodes:")
    print(f"  Episode A (idx {args.episode_a_idx})")
    print(f"  Episode B (idx {args.episode_b_idx})")

    # Extract videos and actions
    video_a = sample_a['video']  # [C, T, H, W]
    video_b = sample_b['video']
    action_a = sample_a['action']  # [T, 20]
    action_b = sample_b['action']

    print(f"\nGenerating 2x2 grid (2 swaps + 2 ground truths)...")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Autoregressive: {args.ar}")

    # Ground truth videos (diagonal) - no inference needed
    gt_aa = normalize_video(video_a)  # [C, T, H, W]
    gt_bb = normalize_video(video_b)

    # Generate only the swapped predictions (off-diagonal)
    print("\n[1/2] Frame A + Actions B (Swapped)")
    pred_ab = run_single_inference(model, video_a, action_b, args.device,
                                    args.ddim_steps, args.guidance_scale, args.ar)

    print("[2/2] Frame B + Actions A (Swapped)")
    pred_ba = run_single_inference(model, video_b, action_a, args.device,
                                    args.ddim_steps, args.guidance_scale, args.ar)

    # Create grid video: GT on diagonal, swapped on off-diagonal
    create_grid_video(gt_aa, pred_ab, pred_ba, gt_bb, args.output, fps=args.fps)

    print("\nDone!")
    print("\nGrid interpretation:")
    print("  Top-left:     Frame A + Actions A → Ground Truth (green)")
    print("  Top-right:    Frame A + Actions B → Swapped prediction (cyan)")
    print("  Bottom-left:  Frame B + Actions A → Swapped prediction (cyan)")
    print("  Bottom-right: Frame B + Actions B → Ground Truth (green)")


if __name__ == "__main__":
    main()
