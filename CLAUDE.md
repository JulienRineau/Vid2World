# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vid2World transforms internet-scale pretrained video diffusion models into interactive world models. It converts non-causal video diffusion backbones into autoregressive, temporally causal architectures with frame-level action conditioning, enabling high-fidelity, action-conditioned video simulation.

## Common Commands

### Environment Setup
```bash
conda create -n v2w python=3.8 -y
conda activate v2w
pip install -r requirements.txt
```

### Training (4 GPUs)
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/<domain>/config_<name>_train.yaml --train --name <experiment_name> --logdir <log_dir> --devices 4 lightning.trainer.num_nodes=1
```

### Inference/Validation (4 GPUs)
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/<domain>/config_<name>_test.yaml --val --name <experiment_name> --logdir <log_dir> --devices 4 lightning.trainer.num_nodes=1
```

### Evaluation
```bash
python eval.py --exp_folder <log_image_dir> --env <rt1|csgo|recon_time|recon_rollout>
```

## Architecture Overview

### Core Model Stack
```
lvdm/models/ddpm3d.py
├── DDPM                    # Base diffusion model with Gaussian diffusion
├── LatentDiffusion         # Main class - latent space diffusion
├── LatentVisualDiffusion   # Image-conditioned visual diffusion (primary model)
└── DiffusionWrapper        # Wraps UNet with conditioning logic
```

### Model Components
- **UNet Backbone** (`lvdm/modules/networks/openaimodel3d.py`): 3D UNet with spatial/temporal attention
- **Attention** (`lvdm/modules/attention.py`): `SpatialTransformer` and `TemporalTransformer` with causal attention support
- **Samplers** (`lvdm/models/samplers/`): DDIM sampler with KV-cache for autoregressive generation
- **Autoencoder** (`lvdm/models/autoencoder.py`): VAE for latent space encoding/decoding
- **Condition Encoders** (`lvdm/modules/encoders/`): CLIP text/image encoders, Resampler for image projection

### Conditioning Flow
The model uses hybrid conditioning (`conditioning_key: hybrid`):
1. **c_concat**: Latent frame concatenation (spatial conditioning)
2. **c_crossattn**: Text + Image embeddings via cross-attention
3. **c_action**: Frame-level action embeddings (for interactive generation)

### Key Configuration Parameters
- `use_causal_attention`: Enables autoregressive temporal processing
- `better_weight_transfer`: Weight initialization strategy (`extrapolative`, `masked`, `shift`)
- `action_emb`: Enable action conditioning
- `df_loss`: Per-frame diffusion loss (different timesteps per frame)
- `rescale_betas_zero_snr`: Zero terminal SNR for v-prediction

### Data Pipeline
```
lvdm/data/
├── rtvid.py          # RT-1 robot manipulation dataset
├── csgovid.py        # CS:GO game frames
├── reconvid.py       # RECON navigation dataset
└── oxe_data_converter.py  # Data preprocessing script
```

Dataset format: `.npz` files with `image: [T, H, W, C]` and `action: [T, M]` arrays.

### Entry Points
- **Training/Inference**: `main/trainer.py` - PyTorch Lightning trainer
- **Evaluation**: `eval.py` - Computes MSE, PSNR, SSIM, LPIPS, DreamSim, FVD, FID
- **Callbacks**: `main/callbacks.py` - Image logging, metrics logging

### Required Checkpoints
```
checkpoints/
├── dynamicrafter_512_v1/model.ckpt  # Base video model
└── i3d/i3d_torchscript.pt           # For FVD evaluation
```

## Config Structure

Configs in `configs/<domain>/` follow this structure:
- `model`: Model architecture and hyperparameters
- `data`: Dataset paths and preprocessing
- `lightning`: Training settings (precision, gradient clipping, checkpointing)

Key placeholders to replace in configs:
- `|<your_data_dir>|`: Local dataset path
- `|<your_pretrained_checkpoint>|`: Model checkpoint path
- `|<your_log_dir>|`: Output directory
- `|<your_save_dir>|`: Metrics save directory

## Domain-Specific Notes

- **RT-1 (manipulation)**: 16-frame sequences, 320x512 resolution, 7-DoF actions
- **CS:GO (game)**: 12-frame evaluation, preprocessing includes 275x512 center crop → 150x280 resize
- **RECON (navigation)**: Single-step and rollout evaluation modes, 640x480 preprocessing resolution
