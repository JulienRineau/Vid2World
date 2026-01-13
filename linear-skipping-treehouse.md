# Vid2World Training Plan for sf_fold Robotics Dataset

## Executive Summary

This plan adapts Vid2World (DynamiCrafter-based action-conditioned video diffusion) to the sf_fold dataset (100 hours of bimanual t-shirt folding with UMI-style tracking and 200° fisheye cameras) within a 24-hour training window on 8×H100 GPUs.

---

## Critical Design Decisions

### 1. Fisheye Strategy: Train on Raw Fisheye (Recommended)

| Aspect | Raw Fisheye | Rectified |
|--------|-------------|-----------|
| FOV Preserved | Full 200° | ~90-110° |
| Data Sufficiency | 100hrs adequate for adaptation | Same |
| Preprocessing | Simple resize | Complex calibration |
| Manipulation Performance | Better (per UMI ablations) | Degraded |

**Decision**: Train directly on raw fisheye frames. The pretrained DynamiCrafter backbone learned motion priors and temporal coherence, not strict pinhole geometry. The U-Net will adapt spatial priors during fine-tuning with 100 hours of data.

**Contingency**: If FVD degrades >50% after 10k steps, implement progressive distortion augmentation (start mild, increase to full fisheye).

### 2. Coordinate System: Frame-to-Frame Relative Poses

**Problem**: Floating origin (each trajectory's origin = right gripper at t=0) makes absolute poses incomparable across trajectories.

**Solution**: Convert to frame-to-frame delta poses with 6D rotation representation.

```
delta_position[t] = position[t] - position[t-1]  # 3D
delta_rotation[t] = R[t] @ R[t-1].T → 6D repr   # 6D (continuous, no discontinuities)
gripper_state[t] = continuous value (e.g., 0.0-1.0 or mm opening)  # 1D
```

**Rationale**:
- Translation invariant (independent of trajectory origin)
- Compatible with action-conditioned generation ("move +5cm" vs "move to position X")
- Bounded range (limited by robot velocity)
- 6D rotation avoids quaternion discontinuities

### 3. Action Space: 20-Dimensional Bimanual

```
Left Gripper (10D):
  [0:3]   delta_position (x, y, z) in meters
  [3:9]   delta_rotation_6d (first two columns of rotation matrix)
  [9]     gripper_width (continuous, e.g., 0.0-1.0 normalized or raw mm)

Right Gripper (10D):
  [10:13] delta_position (x, y, z)
  [13:19] delta_rotation_6d
  [19]    gripper_width (continuous)
```

**Note**: Gripper values are continuous (not binary open/close). Normalize to consistent range during preprocessing.

**Normalization**: Per-dimension standardization (subtract mean, divide by std), clamp to [-3, 3].

---

## Phase 1: Data Preparation (Estimated: 3-4 hours)

### 1.1 LeRobot v3 Dataset Analysis

The sf_fold dataset is already in **LeRobot v3 format**, which uses:
- **Parquet files** for episode metadata and actions
- **Video files** (MP4/WebM) or image sequences for observations
- **Standardized schema** with `observation.images.*`, `action`, `episode_index`, etc.

| Task | Description |
|------|-------------|
| Load dataset | `from lerobot.common.datasets import LeRobotDataset` |
| Inspect schema | Verify action columns, image keys, fps |
| Identify pose columns | Find left/right gripper poses and gripper states |
| Validate integrity | Check episode counts, frame alignment |

### 1.2 LeRobot to NPZ Conversion Pipeline

**Create**: `lvdm/data/sf_fold_converter.py`

**Conversion Steps**:
1. Load LeRobot dataset via HuggingFace datasets API
2. Iterate episodes using LeRobot's episode indexing
3. Extract video frames (resize fisheye to 320×512)
4. Extract pose columns for both grippers
5. Convert absolute poses → relative delta poses with 6D rotation
6. Subsample to 3 fps if source fps is higher
7. Save as NPZ: `{'image': [T, 320, 512, 3], 'action': [T, 20]}`

```python
# LeRobot loading pattern
from datasets import load_dataset

dataset = load_dataset("zeroshotdata/sf_fold", split="train")

# LeRobot v3 expected columns (verify against actual schema):
# - observation.images.cam_fisheye: video frames
# - observation.state: robot proprioception (may include poses)
# - action: robot actions (structure TBD from dataset inspection)
# - episode_index: episode boundaries
# - frame_index: frame within episode
# - timestamp: time in seconds
```

**Helper Functions Needed**:
- `quat_to_rotation_matrix()`: Quaternion to 3×3 rotation
- `rotation_matrix_to_6d()`: Extract first two columns, flatten to 6D
- `compute_delta_poses()`: Frame-to-frame relative transformation
- `extract_bimanual_poses()`: Parse LeRobot action format to left/right poses

### 1.3 LeRobot Action Schema Mapping

**Expected LeRobot action structure** (verify against actual dataset):
```
action: [left_pos(3), left_quat(4), left_gripper_width(1),
         right_pos(3), right_quat(4), right_gripper_width(1)]
# Total: 16D absolute poses per frame
# gripper_width: continuous value (NOT binary)
```

**Conversion to Vid2World format**:
```
vid2world_action: [left_delta_pos(3), left_delta_rot6d(6), left_gripper_width(1),
                   right_delta_pos(3), right_delta_rot6d(6), right_gripper_width(1)]
# Total: 20D relative poses per frame
# gripper_width: keep as continuous, normalize with mean/std
```

### 1.4 Dataset Statistics

| Metric | Expected Value |
|--------|----------------|
| Total hours | 100 |
| Total frames (at 20Hz source) | ~7.2M |
| Subsampled frames (at 3Hz) | ~1.08M |
| 16-frame episodes | ~67,500 |
| Train split (90%) | ~60,750 episodes |
| Val split (10%) | ~6,750 episodes |

### 1.5 Output Structure

```
sf_fold_npz/
├── train_eps_00000000.npz
├── train_eps_00000001.npz
├── ...
├── val_file_list.json
└── action_stats.json  # mean, std for normalization
```

### 1.6 Alternative: Direct LeRobot DataLoader

**Option B** (if NPZ conversion is bottleneck): Create a LeRobot-native dataset class that loads directly from HuggingFace format without NPZ intermediate:

```python
class SFFoldLeRobot(Dataset):
    def __init__(self, hf_dataset_name="zeroshotdata/sf_fold", ...):
        self.dataset = load_dataset(hf_dataset_name, split=split)
        # Build episode index for random access
        # Apply transforms on-the-fly
```

**Trade-off**: Simpler pipeline but slower training I/O. Recommend NPZ for 8×H100 throughput.

---

## Phase 2: Model Architecture Setup (Estimated: 2-3 hours)

### 2.1 Create Dataset Class

**Create**: `lvdm/data/sf_fold_vid.py`

Based on `rtvid.py` template with modifications:

```python
class SFFoldVid(Dataset):
    def __init__(self, data_dir, video_length=16, resolution=[320, 512],
                 mode='training', augment_data=True, action_stats_path=None):
        # Load action normalization stats
        self.action_mean, self.action_std = load_action_stats(action_stats_path)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        frames = data['image']   # [T, H, W, C]
        actions = data['action'] # [T, 20]

        # Normalize actions
        actions = (actions - self.action_mean) / self.action_std
        actions = np.clip(actions, -3, 3)

        # Sample 16 consecutive frames
        # Apply augmentation (color jitter, random crop)
        # Normalize frames to [-1, 1]

        return {
            'video': frames_tensor,    # [C, T, H, W]
            'action': actions_tensor,  # [T, 20]
            'caption': "",
            'path': path,
            'fps': 3,
            'frame_stride': 1
        }
```

### 2.2 Create Training Configuration

**Create**: `configs/manipulation/config_sf_fold_train.yaml`

**Key Modifications from RT-1 config**:

| Parameter | RT-1 Value | sf_fold Value | Rationale |
|-----------|------------|---------------|-----------|
| `action_dim` | 13 | 20 | Bimanual action space |
| `base_learning_rate` | 1.0e-5 | 5.0e-6 | Domain shift requires gentler LR |
| `batch_size` | 2 | 3 | H100 has more memory |
| `max_steps` | 100000 | 50000 | Compute budget constraint |
| `accumulate_grad_batches` | 2 | 2 | Same |
| `precision` | 16 | bf16-mixed | H100 native bf16 |
| `action_dropout_prob` | 0.2 | 0.15 | Bimanual complexity |

**Effective Batch Size**: 3 × 8 × 2 = 48

### 2.3 Verify Action Embedding Path

**File**: `lvdm/modules/networks/openaimodel3d.py`

Confirm `action_dim` flows correctly through:
- `action_emb_preprocess`: Linear(20 → 320)
- `action_emb_proj`: MLP(320 → 1280)
- Addition to timestep embedding in forward pass

No code changes needed - architecture supports arbitrary action_dim via config.

---

## Phase 3: Training Execution (Estimated: 20-21 hours)

### 3.1 Training Command

```bash
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    ./main/trainer.py \
    --base configs/manipulation/config_sf_fold_train.yaml \
    --train \
    --name sf_fold_v1 \
    --logdir /path/to/logs \
    --devices 8
```

### 3.2 Training Schedule

| Phase | Steps | Duration | Description |
|-------|-------|----------|-------------|
| Warm-up | 0-5,000 | ~2.5 hrs | Train temporal + action layers only |
| Full Fine-tune | 5,000-50,000 | ~17.5 hrs | All layers (except frozen VAE/CLIP) |
| **Total** | 50,000 | ~20 hrs | Under 24hr budget |

**Time Estimation**:
- ~1.5 seconds per step on 8×H100 (based on RT-1 benchmarks)
- 50,000 × 1.5s = 75,000s = 20.8 hours

### 3.3 Curriculum Implementation (Optional but Recommended)

**Approach**: Two-phase training with selective freezing

**Phase 1 (Steps 0-5,000)**: Warm-up
- Freeze: Spatial transformer layers (preserve pretrained visual priors)
- Train: Temporal transformers + action embedding layers
- Purpose: Let model learn temporal dynamics and action conditioning first

**Phase 2 (Steps 5,000-50,000)**: Full Fine-tuning
- Unfreeze all (except frozen VAE and CLIP encoders)
- Cosine LR decay from 5e-6 to 1e-6
- Purpose: Adapt spatial features to fisheye domain

### 3.4 Monitoring Checkpoints

| Checkpoint | Steps | Purpose |
|------------|-------|---------|
| Early validation | 5,000 | Verify training is progressing |
| Warm-up complete | 10,000 | Check fisheye adaptation |
| Mid-training | 25,000 | Full evaluation suite |
| Final | 50,000 | Production model |

### 3.5 Key Metrics to Monitor

- **Training Loss**: Should decrease steadily
- **Validation FVD**: Primary quality metric (lower is better)
- **LPIPS**: Perceptual quality per frame
- **Action Embedding Gradient Norm**: Ensure action conditioning is learning

---

## Phase 4: Evaluation Strategy (Estimated: 2-3 hours)

### 4.1 Quantitative Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| FVD | Fréchet Video Distance | < 150 (RT-1 baseline ~100-120) |
| LPIPS | Perceptual similarity | < 0.3 |
| SSIM | Structural similarity | > 0.7 |
| PSNR | Peak signal-to-noise | > 20 dB |
| Action Controllability | Video difference / action difference | > 0.5 |

### 4.2 Qualitative Evaluation

1. **Visual Inspection**: Generate 50 random videos, manually assess quality
2. **Action Response**: Perturb actions, verify visually different outputs
3. **Bimanual Coordination**: Test coordinated vs independent hand movements
4. **Long Rollout**: Generate 64+ frames autoregressively, check for drift

### 4.3 Evaluation Protocol

```bash
# Standard evaluation (1024 videos, 16 frames each)
python eval.py --exp_folder /path/to/sf_fold_v1 --env sf_fold

# Action controllability test
python eval_action_control.py --checkpoint /path/to/model.ckpt
```

### 4.4 Ablation Studies (Post-Training, If Time Permits)

| Ablation | Purpose |
|----------|---------|
| Relative vs Absolute poses | Validate coordinate system choice |
| 6D vs Quaternion rotation | Validate rotation representation |
| Fisheye vs Rectified | Validate fisheye decision |
| 10D vs 7D per gripper | Validate action dimensionality |

---

## Phase 5: Timeline and Resource Allocation

### Full 24-Hour Schedule

| Hour | Activity | Resources |
|------|----------|-----------|
| 0-1 | LeRobot dataset download and schema inspection | CPU |
| 1-4 | LeRobot → NPZ conversion (parallel processing) | CPU (multicore) |
| 4-5 | Dataset class implementation and testing | - |
| 5-6 | Config creation and single-GPU validation | 1 GPU test |
| 6-7 | Multi-GPU training launch and monitoring | 8×H100 |
| 7-27 | Training (50k steps, ~20 hours) | 8×H100 |
| 27-29 | Evaluation suite (FVD + action controllability) | 2 GPUs |
| 29-30 | Analysis, visualization, documentation | - |

**Note**: Total wall-clock is ~30 hours but fits within constraints since data prep can run before GPU allocation.

### Resource Utilization

| Resource | Allocation |
|----------|------------|
| 8×H100 Training | 20 hours |
| CPU Data Prep | 4-6 hours (parallel) |
| GPU Evaluation | 2-3 hours |
| Total GPU-hours | 192 H100-hours (within budget) |

---

## Phase 6: Risk Mitigation

### Risk 1: Fisheye Adaptation Failure
**Symptom**: FVD > 200 after 10k steps
**Mitigation**:
- Implement progressive distortion augmentation
- Fallback: Rectify dataset and retrain

### Risk 2: Bimanual Action Collapse
**Symptom**: One gripper dominates, other produces noise
**Mitigation**:
- Monitor per-gripper loss separately
- Add balancing loss term: `loss += 0.1 * abs(left_loss - right_loss)`

### Risk 3: Compute Budget Overrun
**Symptom**: Training slower than expected
**Mitigation**:
- Reduce to `batch_size: 2, max_steps: 35000`
- Skip warm-up phase (direct full fine-tuning)

### Risk 4: Relative Pose Drift
**Symptom**: Long rollouts accumulate error
**Mitigation**:
- Add trajectory-level augmentation (random rotation/translation of entire trajectory)
- Consider hybrid representation (relative + periodic absolute anchors)

### Risk 5: Data Quality Issues
**Symptom**: Many corrupted/missing frames in dataset
**Mitigation**:
- Implement robust loading with skip-on-error
- Pre-validate all NPZ files before training

---

## Critical Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `lvdm/data/sf_fold_converter.py` | CREATE | Data conversion pipeline |
| `lvdm/data/sf_fold_vid.py` | CREATE | Dataset class |
| `configs/manipulation/config_sf_fold_train.yaml` | CREATE | Training config |
| `eval_sf_fold.py` | CREATE | Evaluation script |
| `lvdm/data/val_file_list_sf_fold.json` | CREATE | Validation split |

**No modifications needed to core model files** - architecture supports arbitrary action_dim via config.

---

## Verification Checklist

Before training:
- [ ] All NPZ files validate (can be loaded, correct shapes)
- [ ] Action statistics computed and saved
- [ ] Dataset class returns correct tensor shapes
- [ ] Config loads without errors
- [ ] Single forward pass succeeds on 1 GPU
- [ ] Multi-GPU training starts without errors

During training:
- [ ] Loss decreasing after 1000 steps
- [ ] No NaN/Inf in gradients
- [ ] Checkpoints saving correctly
- [ ] Validation running at intervals

After training:
- [ ] FVD < 150 on validation set
- [ ] Action controllability test passes
- [ ] Qualitative samples look reasonable
- [ ] Model generates coherent 64-frame rollouts

---

## Summary

This plan enables training an action-conditioned world model for bimanual cloth manipulation within 24 hours on 8×H100 GPUs by:

1. **Preserving fisheye FOV** for optimal manipulation coverage
2. **Using relative poses** to handle floating origin coordinates
3. **20D bimanual action space** with 6D rotation representation
4. **50k training steps** with curriculum (warm-up → full fine-tune)
5. **Comprehensive evaluation** including action controllability tests

The approach builds directly on Vid2World's proven RT-1 pipeline, requiring only data conversion and config changes rather than architectural modifications.
