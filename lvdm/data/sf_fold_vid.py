import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import json
from torch import Tensor
from typing import Optional, List, Tuple
import math


class SFFoldVid(Dataset):
    """
    SF Fold Dataset for bimanual cloth manipulation.

    Assumes sf_fold data is structured as follows:
    data_dir/
        train_eps_00000000.npz
        train_eps_00000001.npz
        ...
        action_stats.json  # Contains mean and std for action normalization
        val_file_list.json # List of validation file names

    Each .npz file contains:
        - image: shape [T, H, W, C] (fisheye frames, 320x512)
        - action: shape [T, 20] (bimanual: 10D per gripper)

    Action format (20D bimanual):
        Left Gripper [0:10]:
            [0:3]   delta_position (x, y, z) in meters
            [3:9]   delta_rotation_6d (first two columns of rotation matrix)
            [9]     gripper_width (continuous, normalized)
        Right Gripper [10:20]:
            [10:13] delta_position (x, y, z)
            [13:19] delta_rotation_6d
            [19]    gripper_width (continuous)
    """

    def __init__(self,
                 data_dir,
                 resolution,
                 video_length=16,
                 spatial_transform=None,
                 crop_resolution=None,
                 subsample=False,
                 augment_data=True,
                 mode='training',  # training or validation
                 strong_augmentation=False,
                 val_file_list_path=None,
                 action_stats_path=None,
                 max_samples=None,  # Limit dataset size for overfit testing
                 ):
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.subsample = subsample
        self.mode = mode
        self.augment_data = augment_data
        self.strong_augmentation = strong_augmentation

        # Augmentation parameters
        if self.strong_augmentation:
            self.brightness = [0.6, 1.4]
            self.contrast = [0.6, 1.4]
            self.saturation = [0.6, 1.4]
            self.hue = [-0.5, 0.5]
            self.random_resized_crop_scale = (0.6, 1.0)
            self.random_resized_crop_ratio = (0.75, 1.3333)
        else:
            self.brightness = [0.9, 1.1]
            self.contrast = [0.9, 1.1]
            self.saturation = [0.9, 1.1]
            self.hue = [-0.05, 0.05]
            self.random_resized_crop_scale = (0.8, 1.0)
            self.random_resized_crop_ratio = (0.9, 1.1)

        # Load action normalization statistics
        self.action_stats_path = action_stats_path
        self.action_mean = None
        self.action_std = None
        self._load_action_stats()

        # Initialize val_file_set before _load_metadata
        self.val_file_list_path = val_file_list_path
        self.val_file_set = set()
        if self.val_file_list_path and os.path.exists(self.val_file_list_path):
            with open(self.val_file_list_path, 'r') as f:
                self.val_file_set = set(json.load(f))

        # Load all npz files
        self._load_metadata()

        # Apply max_samples limit for overfit testing
        self.max_samples = max_samples
        if self.max_samples is not None and self.max_samples > 0:
            self.file_list = self.file_list[:self.max_samples]

        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                ])
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

        if self.mode == 'validation':
            assert self.augment_data == False, "validation mode, augment_data should be set False"

    def _load_action_stats(self):
        """Load action normalization statistics from JSON file."""
        if self.action_stats_path and os.path.exists(self.action_stats_path):
            with open(self.action_stats_path, 'r') as f:
                stats = json.load(f)
                self.action_mean = np.array(stats['mean'], dtype=np.float32)
                self.action_std = np.array(stats['std'], dtype=np.float32)
                # Prevent division by zero
                self.action_std = np.maximum(self.action_std, 1e-6)
        else:
            # Default: no normalization (identity transform)
            self.action_mean = np.zeros(20, dtype=np.float32)
            self.action_std = np.ones(20, dtype=np.float32)

    def _load_metadata(self):
        """Get all npz files in the directory, split by train/val."""
        self.file_list = []
        for file in os.listdir(self.data_dir):
            if not file.endswith('.npz'):
                continue
            if self.mode == 'training':
                if file not in self.val_file_set:
                    self.file_list.append(os.path.join(self.data_dir, file))
            elif self.mode == 'validation':
                if file in self.val_file_set:
                    self.file_list.append(os.path.join(self.data_dir, file))

        self.file_list.sort()  # Sort files for consistency

        if self.mode == 'validation':
            if len(self.file_list) > 1024:
                # Random sample from valid files for evaluation efficiency
                rand_state = random.getstate()
                random.seed(0)
                valid_files = []
                for file_path in self.file_list:
                    with np.load(file_path) as data:
                        frames = data['action']
                        if len(frames) >= self.video_length:
                            valid_files.append((file_path, len(frames)))
                valid_file_paths = [x[0] for x in valid_files]
                assert len(valid_file_paths) >= 1024, f"Only {len(valid_file_paths)} files with enough frames"
                self.file_list = random.sample(valid_file_paths, 1024)
                random.setstate(rand_state)

        if self.subsample:
            self.file_list = self.file_list[:16]  # Use 16 videos for quick testing


    def __getitem__(self, index):
        # Find which file to load based on index
        file_idx = index % len(self.file_list)
        file_path = self.file_list[file_idx]

        # Load the npz file
        with np.load(file_path) as data:
            frames = data['image']   # [T, H, W, C]
            actions = data['action']  # [T, 20]

            if len(frames) < self.video_length:
                # Skip files with insufficient frames
                index += 1
                return self.__getitem__(index)
            else:
                random_range = len(frames) - self.video_length
                if self.mode == 'validation':
                    # Use deterministic start index for validation
                    random_state = random.getstate()
                    file_hash = hash(file_path.split('/')[-1])
                    random.seed(file_hash)
                    start_idx = random.randint(0, random_range) if random_range > 0 else 0
                    selected_frames = frames[start_idx:start_idx + self.video_length]
                    selected_actions = actions[start_idx:start_idx + self.video_length]
                    random.setstate(random_state)
                else:
                    start_idx = random.randint(0, random_range) if random_range > 0 else 0
                    selected_frames = frames[start_idx:start_idx + self.video_length]
                    selected_actions = actions[start_idx:start_idx + self.video_length]

            # Convert to torch tensor and adjust dimensions
            frames_tensor = torch.tensor(selected_frames).permute(3, 0, 1, 2).float()  # [C, T, H, W]
            actions_tensor = torch.tensor(selected_actions).float()  # [T, 20]

            # Normalize actions: (action - mean) / std, clip to [-3, 3]
            actions_tensor = (actions_tensor - torch.tensor(self.action_mean)) / torch.tensor(self.action_std)
            actions_tensor = torch.clamp(actions_tensor, -3.0, 3.0)

            if self.spatial_transform is not None:
                frames_tensor = self.spatial_transform(frames_tensor)

            if self.augment_data:
                frames_tensor = self.augment_video_clip(frames_tensor)

            if self.resolution is not None:
                assert (frames_tensor.shape[2], frames_tensor.shape[3]) == (self.resolution[0], self.resolution[1]), \
                    f'frames={frames_tensor.shape}, self.resolution={self.resolution}'

            # Normalize frames to [-1, 1]
            frames_tensor = (frames_tensor / 255 - 0.5) * 2

            data = {
                'video': frames_tensor,
                'caption': "",
                'action': actions_tensor,
                'path': file_path,
                'fps': 3,  # sf_fold uses 3 fps (same as RT-1)
                'frame_stride': 1
            }
            return data

    def __len__(self):
        return len(self.file_list)

    def augment_video_clip(self, video):
        """Apply consistent augmentation across all frames in a video clip."""
        # video: [C, T, H, W]
        C, T, H, W = video.shape
        assert C == 3, f"Expected 3 channels, got {C} channels"

        new_frames = []
        transform = transforms.ToPILImage()

        # Sample augmentation parameters once for the entire clip
        brightness = random.uniform(*self.brightness)
        contrast = random.uniform(*self.contrast)
        saturation = random.uniform(*self.saturation)
        hue = random.uniform(*self.hue)
        padding = 2
        i = random.randint(0, 2 * padding)
        j = random.randint(0, 2 * padding)

        for t in range(T):
            # Get frame t and keep it as [C, H, W]
            frame = video[:, t, :, :]  # [C, H, W]

            frame = F.pad(frame, [padding] * 4, padding_mode='reflect')  # Pad all sides
            frame = frame[:, i:i + H, j:j + W]  # Random crop back to original size

            # Convert to PIL for color augmentation
            frame = transform(frame / 255)  # Convert to PIL Image
            tensor = transforms.ToTensor()

            # Color augmentation
            for fn_id in range(4):
                if fn_id == 0:
                    frame = F.adjust_brightness(frame, brightness)
                elif fn_id == 1:
                    frame = F.adjust_contrast(frame, contrast)
                elif fn_id == 2:
                    frame = F.adjust_saturation(frame, saturation)
                elif fn_id == 3:
                    frame = F.adjust_hue(frame, hue)

            # Convert back to Tensor
            frame = tensor(frame)
            frame = frame * 255.0
            new_frames.append(frame)

        return torch.stack(new_frames, dim=1)  # [C, T, H, W]

    @staticmethod
    def get_crop_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop."""
        _, height, width = F.get_dimensions(img)
        area = min(height, width) ** 2

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    @staticmethod
    def get_jittor_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image."""
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h
