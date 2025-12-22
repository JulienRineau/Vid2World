import torch
import numpy as np
import scipy
import argparse
from einops import rearrange
from utils.metrics import Evaluator
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm
from pathlib import Path
from torchvision import io
import torch.nn.functional as F

def cal_step_metric(evaluator, real_video, fake_video, device):
    # real_video: [t h w c], (-1, 1), tensor, fp32
    real_video = rearrange(real_video, 't h w c -> t c h w').to(device)
    fake_video = rearrange(fake_video, 't h w c -> t c h w').to(device)
    mse = evaluator.compute_mse(real_video, fake_video)  # [t]
    psnr = evaluator.compute_psnr(real_video, fake_video)  # [t]
    ssim = evaluator.compute_ssim(real_video, fake_video)  # [t]
    lpips = evaluator.compute_lpips(real_video, fake_video)  # [t]
    result = {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips,
        'dreamsim': evaluator.compute_dreamsim(real_video, fake_video),
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate video metrics (MSE, PSNR, SSIM, LPIPS, DreamSim, FVD, FID)')
    parser.add_argument('--exp_folder', type=str, required=True,
                        help='Path to the experiment folder containing train_eps_* subdirectories')
    parser.add_argument('--env', type=str, required=True,
                        help='Environment name: rt1, csgo, recon_time, recon_rollout')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation (default: cuda:0)')
    parser.add_argument('--i3d_model_path', type=str,
                        default='checkpoints/i3d/i3d_torchscript.pt',
                        help='Path to I3D model checkpoint')
    parser.add_argument('--override_sample_number', type=int, default=None,
                        help='Override the sample number for the evaluation')
    
    args = parser.parse_args()
    
    exp_folder = Path(args.exp_folder)

    env = args.env
    if env == "rt1":
        num_video = 1024
        metric_frame = 16
        batchsize = 64
        max_batchsize = 16
    elif env == "csgo":
        num_video = 500
        metric_frame = 12
        batchsize = 50
        max_batchsize = 10
    elif env == "recon_time":
        num_video = 500
        metric_frame = 16
        batchsize = 50
        max_batchsize = 10
    elif env == "recon_rollout":
        num_video = 150
        metric_frame = 16
        batchsize = 50
        max_batchsize = 10
    else:
        raise ValueError(f"Environment {env} not supported")
    if args.override_sample_number is not None:
        num_video = args.override_sample_number
    device = args.device
    i3d_mode_path = args.i3d_model_path

    evaluator = Evaluator(
        i3d_model_path=i3d_mode_path,
        max_batchsize=max_batchsize,
        device=device,
        env=env,
    )
    
    val_metrics_buffer = {}
    val_step_metrics_buffer = {}
    buffer_fake_video = []
    buffer_real_video = []
    count = 0
    solve_num = 0
    
    # Get video paths based on environment
    if env == "rt1":
        video_num = len(list(exp_folder.glob("train_eps_*")))
        video_paths = sorted(exp_folder.glob("train_eps_*"))
    elif env == "csgo":
        video_num = len(list(exp_folder.glob("hdf5_dm_*")))
        video_paths = sorted(exp_folder.glob("hdf5_dm_*"))
    elif env in ["recon_time", "recon_rollout"]:
        # For recon_time and recon_rollout, videos are in the same directory with _x_gt.mp4 suffix
        gt_video_paths = sorted(exp_folder.glob("*_x_gt.mp4"))
        video_num = len(gt_video_paths)
        video_paths = gt_video_paths  # Store gt paths, we'll derive pred paths from them
    else:
        raise ValueError(f"Environment {env} not supported")
    
    assert video_num == num_video, f"Number of videos {video_num} does not equal {num_video}, please check"
    if video_num % batchsize != 0:
        raise ValueError(f"Number of videos {video_num} cannot be divided by {batchsize}, please check")
    
    for video_path in tqdm(video_paths, desc="Processing videos", total=video_num):
        # Determine gt_path and pred_path based on environment
        if env in ["rt1", "csgo"]:
            # Old structure: subdirectories with gt_video.mp4 and pred_video.mp4
            gt_path = video_path / "gt_video.mp4"
            pred_path = video_path / "pred_video.mp4"
        elif env in ["recon_time", "recon_rollout"]:
            # New structure: files in same directory with _x_gt.mp4 and _x_pred.mp4 suffixes
            gt_path = video_path  # video_path is already the _x_gt.mp4 file
            # Replace _x_gt.mp4 with _x_pred.mp4
            pred_path = Path(str(gt_path).replace("_x_gt.mp4", "_x_pred.mp4"))
        else:
            raise ValueError(f"Environment {env} not supported")
        
        if not (gt_path.exists() and pred_path.exists()):
            raise FileNotFoundError(f"Missing files: gt={gt_path}, pred={pred_path}")

        real_video, _, _ = io.read_video(str(gt_path), pts_unit='sec')
        fake_video, _, _ = io.read_video(str(pred_path), pts_unit='sec')
        real_video = real_video.float() / 255.0
        fake_video = fake_video.float() / 255.0
        real_video = (real_video - 0.5) * 2
        fake_video = (fake_video - 0.5) * 2
        fake_video = fake_video.clamp(-1, 1)
        real_video = real_video.clamp(-1, 1)

        # Preprocessing for csgo: center crop to 275x512, then resize to 150x280
        if env in ["csgo", "recon_time", "recon_rollout"]:
            # real_video and fake_video: [t, h, w, c], input size: 320x512
            h, w = real_video.shape[1], real_video.shape[2]  # h=320, w=512
            if env == "csgo":
                crop_h = 275
                crop_w = 512
            else:
                crop_h = 320
                crop_w = 320
            start_h = (h - crop_h) // 2  
            end_h = start_h + crop_h  
            start_w = (w - crop_w) // 2  
            end_w = start_w + crop_w  
            real_video = real_video[:, start_h:end_h, start_w:end_w, :] 
            fake_video = fake_video[:, start_h:end_h, start_w:end_w, :] 
            
            real_video = rearrange(real_video, 't h w c -> t c h w')
            fake_video = rearrange(fake_video, 't h w c -> t c h w')
            if env == "csgo":
                real_video = F.interpolate(real_video, size=(150, 280), mode='bilinear', align_corners=False)
                fake_video = F.interpolate(fake_video, size=(150, 280), mode='bilinear', align_corners=False)
            else:
                real_video = F.interpolate(real_video, size=(224, 224), mode='bilinear', align_corners=False)
                fake_video = F.interpolate(fake_video, size=(224, 224), mode='bilinear', align_corners=False)
            real_video = rearrange(real_video, 't c h w -> t h w c')
            fake_video = rearrange(fake_video, 't c h w -> t h w c')
        real_video = rearrange(real_video, 't h w c -> 1 c t h w')
        fake_video = rearrange(fake_video, 't h w c -> 1 c t h w')
        buffer_fake_video.append(fake_video)
        buffer_real_video.append(real_video)
        solve_num += 1

        if count >= batchsize or solve_num == video_num:
            count = 0
            buffer_real_video = torch.cat(buffer_real_video, dim=0).to(device)
            buffer_fake_video = torch.cat(buffer_fake_video, dim=0).to(device)
            
            if env == "recon_rollout":
                # For rollout: evaluate at frames 8, 12, 20 (0-indexed: 7, 11, 19)
                # No FVD computation for rollout, but need FID stats
                eval_frames = [7, 11, 19]  # Frame indices (0-indexed)
                frame_names = ["frame_8", "frame_12", "frame_20"]
                
                for frame_idx, frame_name in zip(eval_frames, frame_names):
                    # Check if frame exists
                    if buffer_fake_video.shape[2] > frame_idx:
                        # Extract single frame: [B, C, 1, H, W]
                        pred_frame = buffer_fake_video[:, :, frame_idx:frame_idx+1, :, :]
                        gt_frame = buffer_real_video[:, :, frame_idx:frame_idx+1, :, :]
                        
                        # Compute metrics for this frame (no FVD, but collect FID stats)
                        metrics = evaluator.evaluate_all(
                            pred_frame,
                            gt_frame,
                            raw=True,  # Use raw=True to get real_stats and fake_stats
                            evaluate=True,
                            compute_fvd=False
                        )
                        
                        # Store image metrics with frame name prefix
                        for key, value in metrics.items():
                            if key not in ["real_stats", "fake_stats"]:
                                full_key = f"{frame_name}_{key}"
                                if full_key not in val_metrics_buffer.keys():
                                    val_metrics_buffer[full_key] = torch.tensor(value, device=device).unsqueeze(0)
                                else:
                                    val_metrics_buffer[full_key] = torch.cat([
                                        val_metrics_buffer[full_key], 
                                        torch.tensor(value, device=device).unsqueeze(0)
                                    ], dim=0)
                        
                        # Store FID stats separately for each frame (will compute FID at the end)
                        real_stats_key = f"{frame_name}_real_stats"
                        fake_stats_key = f"{frame_name}_fake_stats"
                        tensorized_real_stats = torch.tensor(metrics['real_stats'], device=device)
                        tensorized_fake_stats = torch.tensor(metrics['fake_stats'], device=device)
                        
                        if real_stats_key not in val_metrics_buffer.keys():
                            val_metrics_buffer[real_stats_key] = tensorized_real_stats
                        else:
                            val_metrics_buffer[real_stats_key] = torch.cat([
                                val_metrics_buffer[real_stats_key], 
                                tensorized_real_stats
                            ], dim=0)
                        
                        if fake_stats_key not in val_metrics_buffer.keys():
                            val_metrics_buffer[fake_stats_key] = tensorized_fake_stats
                        else:
                            val_metrics_buffer[fake_stats_key] = torch.cat([
                                val_metrics_buffer[fake_stats_key], 
                                tensorized_fake_stats
                            ], dim=0)
                    else:
                        print(f"Warning: Frame {frame_idx+1} (index {frame_idx}) not available, video has {buffer_fake_video.shape[2]} frames")
            elif env == "recon_time":
                fvd_metrics = evaluator.evaluate_all(
                    buffer_fake_video[:, :, -metric_frame:, :, :],
                    buffer_real_video[:, :, -metric_frame:, :, :],
                    raw=True,
                    evaluate=True, 
                    compute_fvd=True
                )
                
                for key in ["raw_gt_features", "raw_pred_features"]:
                    if key in fvd_metrics:
                        tensorized_value = torch.tensor(fvd_metrics[key], device=device)
                        if key not in val_metrics_buffer.keys():
                            val_metrics_buffer[key] = tensorized_value
                        else:
                            val_metrics_buffer[key] = torch.cat([val_metrics_buffer[key], tensorized_value], dim=0)
                
                last_frame_pred = buffer_fake_video[:, :, -1:, :, :] # [B, C, 1, H, W]
                last_frame_gt = buffer_real_video[:, :, -1:, :, :]   # [B, C, 1, H, W]
                
                last_frame_metrics = evaluator.evaluate_all(
                    last_frame_pred,
                    last_frame_gt,
                    raw=True,
                    evaluate=True,
                    compute_fvd=False 
                )
                
                for key, value in last_frame_metrics.items():
                    if key in ["real_stats", "fake_stats"]:
                        tensorized_value = torch.tensor(value, device=device)
                        if key not in val_metrics_buffer.keys():
                            val_metrics_buffer[key] = tensorized_value
                        else:
                            val_metrics_buffer[key] = torch.cat([val_metrics_buffer[key], tensorized_value], dim=0)
                    
                    elif key not in ["fvd", "fid", "raw_gt_features", "raw_pred_features"]:
                        if key not in val_metrics_buffer.keys():
                            val_metrics_buffer[key] = torch.tensor(value, device=device).unsqueeze(0)
                        else:
                            val_metrics_buffer[key] = torch.cat([
                                val_metrics_buffer[key], 
                                torch.tensor(value, device=device).unsqueeze(0)
                            ], dim=0)
            else:
                # For other environments (rt1, csgo): use metric_frame and compute FVD/FID
                metrics = evaluator.evaluate_all(
                    buffer_fake_video[:, :, -metric_frame:, :, :],
                    buffer_real_video[:, :, -metric_frame:, :, :],
                    raw=True,
                )
                
                for key, value in metrics.items():
                    if key in ["raw_gt_features", "raw_pred_features", "real_stats", "fake_stats"]:
                        # Store raw features/stats for FVD/FID computation at the end
                        # value is a list of numpy arrays
                        tensorized_value = torch.tensor(value, device=device)
                        if key not in val_metrics_buffer.keys():
                            val_metrics_buffer[key] = tensorized_value
                        else:
                            val_metrics_buffer[key]= torch.cat([val_metrics_buffer[key], tensorized_value], dim=0)
                    elif key not in ["fvd", "fid"]:
                        # Store image metrics (MSE, PSNR, SSIM, LPIPS, DreamSim) for mean computation
                        # Exclude FVD and FID - they should be computed from raw features/stats, not averaged
                        if key not in val_metrics_buffer.keys():
                            val_metrics_buffer[key] = torch.tensor(value, device=device).unsqueeze(0)
                        else:
                            val_metrics_buffer[key]= torch.cat([val_metrics_buffer[key], torch.tensor(value, device=device).unsqueeze(0)], dim=0)
            
            buffer_fake_video = []
            buffer_real_video = []
        count += 1
    print(f'Processed {solve_num} videos, now computing metrics')
    ans = {}
    gather_val_metrics_buffer = val_metrics_buffer
    
    if env == "recon_rollout":
        # For rollout: compute means for image metrics, and FID from collected stats
        eval_frames = ["frame_8", "frame_12", "frame_20"]
        
        for frame_name in eval_frames:
            # Compute FID for this frame from collected stats
            real_stats_key = f"{frame_name}_real_stats"
            fake_stats_key = f"{frame_name}_fake_stats"
            
            if real_stats_key in gather_val_metrics_buffer and fake_stats_key in gather_val_metrics_buffer:
                real_stats = gather_val_metrics_buffer[real_stats_key].cpu().numpy()
                fake_stats = gather_val_metrics_buffer[fake_stats_key].cpu().numpy()
                
                mu_real = np.mean(real_stats, axis=0)
                mu_fake = np.mean(fake_stats, axis=0)
                sigma_real = np.cov(real_stats, rowvar=False)
                sigma_fake = np.cov(fake_stats, rowvar=False)
                
                fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
                ans[f"{frame_name}_fid"] = fid.item()
        
        # Compute means for other metrics
        for key, value in gather_val_metrics_buffer.items():
            if key.endswith("_real_stats") or key.endswith("_fake_stats"):
                continue  # Skip stats, already processed above
            mean_value = torch.mean(value)
            ans[key] = mean_value.item()
    else:
        # For other environments (rt1, csgo, recon_time): compute FVD and FID from collected features/stats, remember in recon_time metrics except fvd is calculated on the last frame only
        for key, value in gather_val_metrics_buffer.items():
            if key == "raw_gt_features":
                raw_gt_features = value.cpu().numpy()
                mu_true = np.mean(raw_gt_features, axis=0)
                sigma_true = np.cov(raw_gt_features, rowvar=False)
            elif key == "raw_pred_features":
                raw_pred_features = value.cpu().numpy()
                mu_pred = np.mean(raw_pred_features, axis=0)
                sigma_pred = np.cov(raw_pred_features, rowvar=False)
            elif key == "real_stats":
                real_stats = value.cpu().numpy()
                mu_real = np.mean(real_stats, axis=0)
                sigma_real = np.cov(real_stats, rowvar=False)
            elif key == "fake_stats":
                fake_stats = value.cpu().numpy()
                mu_fake = np.mean(fake_stats, axis=0)
                sigma_fake = np.cov(fake_stats, rowvar=False)
            elif key not in ["raw_gt_features", "raw_pred_features", "real_stats", "fake_stats"]:
                # Compute means for image metrics (MSE, PSNR, SSIM, LPIPS, DreamSim)
                # Exclude raw features and stats - they are used for FVD/FID computation
                mean_value = torch.mean(value)
                ans[key] = mean_value.item()
        
        # Compute FVD and FID from collected features/stats (not from averaged values)
        m = np.square(mu_pred - mu_true).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
        fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))
        fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        ans['fvd'] = fvd.item()
        ans['fid'] = fid.item()

    for key, value in ans.items():
        print(f"{key}: {value}")