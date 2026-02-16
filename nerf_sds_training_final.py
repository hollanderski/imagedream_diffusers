
"""

NeRF Training with Score Distillation Sampling (SDS)
Using LoRA-finetuned MVDream as frozen prior for EEG-conditional 3D generation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
from accelerate import Accelerator
import wandb
from pathlib import Path
import sys

# Add your existing paths
sys.path.append("G:/ninon/workspace/imagery2024/2DReconstructionGeneration/2Dreconstructioncode")
sys.path.append("G:/ninon/workspace/imagery2024/3DReconstruction/imagedream-eeg/extern/ImageDream")
sys.path.append("G:/ninon/workspace/imagery2024/3DReconstruction/imagedream-eeg/extern/ImageDream/imagedream")

from pipeline_mvdream import MVDreamPipeline
from mv_unet import get_camera
from EEGNet_Embedding_version import EEGNet_Embedding
from multiview_dataset import MultiViewEEGDataset  
import lpips

# ============================================================================
# 1. Simple NeRF Implementation
# ============================================================================

class NeRFMLP(nn.Module):
    """
    Simple MLP-based NeRF for density and RGB prediction.
    You can replace this with InstantNGP, triplane, or DMTet representations.
    """
    def __init__(
        self,
        pos_dim=3,
        dir_dim=3,
        hidden_dim=256,
        num_layers=8,
        skips=[4],
        use_viewdirs=True,
    ):
        super().__init__()
        self.pos_dim = pos_dim
        self.dir_dim = dir_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # Positional encoding (optional; can also use hash encoding)
        self.pos_encoder = PositionalEncoding(pos_dim, num_frequencies=10)
        self.dir_encoder = PositionalEncoding(dir_dim, num_frequencies=4) if use_viewdirs else None

        pos_encoded_dim = self.pos_encoder.out_dim
        dir_encoded_dim = self.dir_encoder.out_dim if use_viewdirs else 0

        # Density network
        self.density_net = []
        in_dim = pos_encoded_dim
        for i in range(num_layers):
            if i in skips and i > 0:
                in_dim = hidden_dim + pos_encoded_dim
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim + 1  # +1 for density
            self.density_net.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                self.density_net.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.density_net = nn.Sequential(*self.density_net)

        # RGB network (conditioned on view direction)
        if use_viewdirs:
            self.rgb_net = nn.Sequential(
                nn.Linear(hidden_dim + dir_encoded_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 3),
                nn.Sigmoid()  # Output in [0, 1]
            )
        else:
            self.rgb_net = nn.Sequential(
                nn.Linear(hidden_dim, 3),
                nn.Sigmoid()
            )

    def forward(self, positions, directions=None):
        """
        positions: (N, 3) - 3D positions
        directions: (N, 3) - view directions (optional)
        Returns: rgb (N, 3), density (N, 1)
        """
        pos_encoded = self.pos_encoder(positions)

        # Density network with skip connections
        h = pos_encoded
        for i, layer in enumerate(self.density_net):
            if isinstance(layer, nn.Linear):
                layer_idx = sum(1 for l in self.density_net[:i] if isinstance(l, nn.Linear))
                if layer_idx in self.skips and layer_idx > 0:
                    h = torch.cat([h, pos_encoded], dim=-1)
            h = layer(h)

        density = h[..., -1:]  # (N, 1)
        features = h[..., :-1]  # (N, hidden_dim)

        # RGB network
        if self.use_viewdirs and directions is not None:
            dir_encoded = self.dir_encoder(directions)
            rgb_input = torch.cat([features, dir_encoded], dim=-1)
        else:
            rgb_input = features

        rgb = self.rgb_net(rgb_input)  # (N, 3)

        return rgb, density


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, input_dim, num_frequencies=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.out_dim = input_dim * (1 + 2 * num_frequencies)

    def forward(self, x):
        """x: (N, input_dim)"""
        encoded = [x]
        for freq in range(self.num_frequencies):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(2.0 ** freq * np.pi * x))
        return torch.cat(encoded, dim=-1)


# ============================================================================
# 2. Differentiable Volume Renderer
# ============================================================================

class VolumeRenderer(nn.Module):
    """
    Simple volume rendering with ray marching.
    For production, use nerfacc or InstantNGP's CUDA kernels.
    """
    def __init__(
        self,
        near=0.5,
        far=2.5,
        n_samples=64,
        n_importance=0,
        white_background=True,
    ):
        super().__init__()
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.white_background = white_background

    def sample_along_rays(self, rays_o, rays_d, n_samples):
        """Stratified sampling along rays."""
        device = rays_o.device
        N_rays = rays_o.shape[0]

        t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
        z_vals = self.near * (1.0 - t_vals) + self.far * t_vals
        z_vals = z_vals.expand(N_rays, n_samples)

        # Perturb sampling (stratified)
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (N_rays, n_samples, 3)
        return pts, z_vals

    def render_rays(self, nerf, rays_o, rays_d):
        """
        Render RGB and depth for a batch of rays.
        rays_o: (N_rays, 3)
        rays_d: (N_rays, 3)
        Returns: rgb_map (N_rays, 3), depth_map (N_rays,)
        """
        N_rays = rays_o.shape[0]

        # Sample points along rays
        pts, z_vals = self.sample_along_rays(rays_o, rays_d, self.n_samples)  # (N_rays, n_samples, 3)
        pts_flat = pts.reshape(-1, 3)  # (N_rays * n_samples, 3)

        # View directions
        viewdirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
        viewdirs_flat = viewdirs[:, None, :].expand_as(pts).reshape(-1, 3)

        # Query NeRF
        rgb, density = nerf(pts_flat, viewdirs_flat)  # (N_rays * n_samples, 3), (N_rays * n_samples, 1)
        rgb = rgb.reshape(N_rays, self.n_samples, 3)
        density = density.reshape(N_rays, self.n_samples)

        # Volume rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)  # (N_rays, n_samples)

        alpha = 1.0 - torch.exp(-F.relu(density) * dists)  # (N_rays, n_samples)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], dim=-1),
            dim=-1
        )  # (N_rays, n_samples)

        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (N_rays, 3)
        depth_map = torch.sum(weights * z_vals, dim=-1)  # (N_rays,)
        acc_map = torch.sum(weights, dim=-1)  # (N_rays,)

        if self.white_background:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, depth_map, acc_map

    def render_image(self, nerf, camera_matrix, width=256, height=256):
        """
        Render full image given camera matrix.
        camera_matrix: (3, 4) or (4, 4) camera extrinsic [R | t]
        Returns: rgb (H, W, 3), depth (H, W)
        """
        device = camera_matrix.device

        # Generate rays
        rays_o, rays_d = self.get_rays(camera_matrix, width, height)  # (H*W, 3), (H*W, 3)

        # Render in chunks to avoid OOM
        chunk_size = 1024 #4096
        rgb_chunks = []
        depth_chunks = []

        for i in range(0, rays_o.shape[0], chunk_size):
            rgb_chunk, depth_chunk, _ = self.render_rays(
                nerf,
                rays_o[i:i+chunk_size],
                rays_d[i:i+chunk_size]
            )
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)

        rgb = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3)
        depth = torch.cat(depth_chunks, dim=0).reshape(height, width)

        return rgb, depth

    def get_rays(self, camera_matrix, width, height, focal=None):
        """
        Generate ray origins and directions for all pixels.
        Assumes pinhole camera with focal length = image_size / 2.
        """
        device = camera_matrix.device

        if focal is None:
            focal = width / 2.0  # Simple assumption

        # Pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing='xy'
        )

        # Camera directions (normalized device coordinates)
        dirs = torch.stack([
            (i - width * 0.5) / focal,
            -(j - height * 0.5) / focal,  # Negative for image coordinate convention
            -torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)

        # Transform to world coordinates
        c2w = camera_matrix[:3, :4] if camera_matrix.shape[0] == 4 else camera_matrix
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
        rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


# ============================================================================
# 3. Camera Utilities (reusing your get_camera format)
# ============================================================================

def sample_camera_poses(num_views, elevation_deg=15.0, radius=1.5, device='cuda'):
    """
    Sample camera poses on a sphere (azimuth varies, elevation fixed).
    Returns poses compatible with your NeRF renderer.
    """
    poses = []
    azimuths = torch.linspace(0, 360, num_views + 1)[:-1]  # Uniform azimuth

    for azimuth in azimuths:
        azimuth_rad = np.deg2rad(azimuth.item())
        elevation_rad = np.deg2rad(elevation_deg)

        # Spherical to Cartesian
        x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = radius * np.sin(elevation_rad)

        # Camera position
        camera_pos = torch.tensor([x, y, z], dtype=torch.float32)

        # Look-at matrix (looking at origin)
        forward = -camera_pos / (torch.norm(camera_pos) + 1e-8)
        up = torch.tensor([0., 0., 1.], dtype=torch.float32)
        right = torch.cross(up, forward)
        right = right / (torch.norm(right) + 1e-8)
        up = torch.cross(forward, right)

        # Build camera-to-world matrix [R | t]
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = camera_pos

        poses.append(c2w)

    poses = torch.stack(poses).to(device)  # (V, 4, 4)
    return poses


# ============================================================================
# 4. SDS Loss Implementation
# ============================================================================

class SDSLoss(nn.Module):
    """
    Score Distillation Sampling loss for NeRF optimization.
    Uses frozen MVDream as the score oracle.
    """
    def __init__(
        self,
        pipeline,
        eeg_encoder,
        eeg_projector,
        guidance_scale=7.5,
        t_range=(0.02, 0.98),
        weighting_strategy='dreamfusion',
    ):
        super().__init__()
        self.pipeline = pipeline
        self.eeg_encoder = eeg_encoder
        self.eeg_projector = eeg_projector
        self.guidance_scale = guidance_scale
        self.t_range = t_range
        self.weighting_strategy = weighting_strategy

        # Freeze all diffusion components
        for param in self.pipeline.unet.parameters():
            param.requires_grad = False
        for param in self.pipeline.vae.parameters():
            param.requires_grad = False
        for param in self.pipeline.text_encoder.parameters():
            param.requires_grad = False
        for param in self.eeg_encoder.parameters():
            param.requires_grad = False
        for param in self.eeg_projector.parameters():
            param.requires_grad = False

        self.pipeline.unet.eval()
        self.pipeline.vae.eval()
        self.eeg_encoder.eval()
        self.eeg_projector.eval()

    def encode_images_to_latents(self, images):
        """
        Encode rendered images to latent space.
        images: (B, V, 3, H, W) in [0, 1] range
        Returns: latents (B, V, 4, H//8, W//8)
        """
        B, V = images.shape[:2]
        
        # Convert [0, 1] -> [-1, 1]
        images = images * 2.0 - 1.0
        
        images_flat = images.view(-1, *images.shape[2:])  # (B*V, 3, H, W)
        
        # Encode through VAE - gradients MUST flow for SDS!
        latents = self.pipeline.vae.encode(images_flat).latent_dist.sample()
        latents = latents * self.pipeline.vae.config.scaling_factor
        
        latents = latents.view(B, V, *latents.shape[1:])  # (B, V, 4, H//8, W//8)
        return latents


    def get_eeg_context(self, eeg_signal, num_views):
        """
        Get EEG context embeddings for all views.
        eeg_signal: (B, C, T) EEG signal
        Returns: eeg_context (B*V, 77, 1024)
        """
        B = eeg_signal.shape[0]

        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg_signal, return_embedding=True)  # (B, 512)
            eeg_emb = self.eeg_projector(eeg_features)  # (B, 1024)

            # Repeat for all views
            eeg_per_view = eeg_emb.unsqueeze(1).repeat(1, num_views, 1)  # (B, V, 1024)
            eeg_context_token = eeg_per_view.unsqueeze(2).repeat(1, 1, 77, 1)  # (B, V, 77, 1024)

            # Scale to match training distribution
            current_norm = eeg_context_token.norm(dim=-1).mean()
            target_norm = 165.0
            scaling_factor = target_norm / (current_norm.item() + 1e-8)
            eeg_context_scaled = eeg_context_token * scaling_factor

            eeg_context_flat = eeg_context_scaled.view(-1, 77, 1024)  # (B*V, 77, 1024)

        return eeg_context_flat

    def forward(self, rendered_images, eeg_signal, camera_embeddings):
        """
        Compute SDS loss.

        rendered_images: (B, V, 3, H, W) in [0, 1]
        eeg_signal: (B, C, T) EEG signal
        camera_embeddings: output from get_camera() matching your UNet format

        Returns: sds_loss (scalar)
        """
        B, V = rendered_images.shape[:2]
        device = rendered_images.device

        # 1. Encode to latents
        latents = self.encode_images_to_latents(rendered_images)  # (B, V, 4, H//8, W//8)
        latents_flat = latents.view(-1, *latents.shape[2:])  # (B*V, 4, H//8, W//8)

        # 2. Get EEG context
        eeg_context = self.get_eeg_context(eeg_signal, V)  # (B*V, 77, 1024)

        # 3. Sample timestep
        t_min, t_max = self.t_range
        num_train_timesteps = self.pipeline.scheduler.config.num_train_timesteps
        t = torch.randint(
            int(t_min * num_train_timesteps),
            int(t_max * num_train_timesteps),
            (B * V,),
            device=device,
            dtype=torch.long
        )

        # 4. Add noise
        noise = torch.randn_like(latents_flat)
        noisy_latents = self.pipeline.scheduler.add_noise(latents_flat, noise, t)

        # 5. Predict noise with frozen UNet
        with torch.no_grad():
            noise_pred = self.pipeline.unet(
                x=noisy_latents,
                timesteps=t,
                context=eeg_context,
                num_frames=V,
                camera=camera_embeddings,
            )

        # 6. Compute SDS loss 
        w = 1.0

        # The key insight: we want to MAXIMIZE alignment with diffusion's denoising direction
        # grad points in direction diffusion wants to move the latent
        grad = (noise_pred.detach() - noise.detach())

        # Standard SDS trick: treat grad as a "pseudo-gradient"
        # loss.backward() will compute: ∇_θ L = -grad * ∇_θ latents
        # The negative sign is important!
        sds_loss = -(latents_flat * grad).mean()

        return sds_loss


# ============================================================================
# 5. Collate Function 
# ============================================================================

def multiview_collate_fn_grey(batch):
    """
    Custom collate function for multiview data with grey background processing.
    Reused from your original training code.
    """
    images_list = []
    
    for item in batch:
        images = item['images']  # (V, C, H, W)
        
        # If images are RGBA, convert to RGB with grey background
        if images.shape[1] == 4:  # 4 channels (RGBA)
            processed_images = []
            for view_img in images:  # Process each view
                img_chw = view_img  # (4, H, W)
                
                # Split RGB and alpha
                rgb = img_chw[:3, :, :]  # (3, H, W)
                alpha = img_chw[3:4, :, :]  # (1, H, W)
                
                # Create grey background
                grey_background = torch.full_like(rgb, 0.5)
                
                # Blend using alpha mask
                img_rgb = torch.where(alpha > 0.5, rgb, grey_background)
                
                # Ensure [-1, 1] range
                if img_rgb.max() <= 1.0 and img_rgb.min() >= 0.0:
                    img_rgb = img_rgb * 2.0 - 1.0  # [0,1] -> [-1,1]
                
                processed_images.append(img_rgb)
            
            images = torch.stack(processed_images)  # (V, 3, H, W)
        
        images_list.append(images)
    
    # Stack all samples
    images = torch.stack(images_list)  # (B, V, C, H, W)
    classes = torch.tensor([item['class'] for item in batch])
    eeg_signals = torch.stack([item['eeg'] for item in batch])
    
    # Handle camera poses if present
    camera_poses = None
    if batch[0].get('camera_poses') is not None:
        camera_poses = torch.stack([item['camera_poses'] for item in batch])
    
    return {
        'images': images,
        'classes': classes,
        'eeg': eeg_signals,
        'camera_poses': camera_poses,
        'num_views': images.shape[1],
        'batch_size': images.shape[0]
    }



# ============================================================================
# 6. Main Training Loop
# ============================================================================

class NeRFSDSTrainer:
    """Main trainer for NeRF + SDS with EEG-conditioned MVDream."""

    def __init__(
        self,
        pipeline,
        eeg_encoder,
        eeg_projector,
        train_dataloader,
        val_dataloader=None,
        learning_rate=1e-3,
        num_epochs=100,
        num_views=8,
        guidance_scale=7.5,
        output_dir='./nerf_sds_output',
        use_wandb=False,
        save_steps=500,
        validation_steps=500,
        use_gt_supervision=False,
        gt_weight=0.1,
        device='cuda',
    ):
        self.device = device
        self.num_views = num_views
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.save_steps = save_steps
        self.validation_steps = validation_steps
        self.use_gt_supervision = use_gt_supervision
        self.gt_weight = gt_weight

        os.makedirs(output_dir, exist_ok=True)

        # Initialize NeRF
        self.nerf = NeRFMLP().to(device)
        self.renderer = VolumeRenderer(n_samples=32, white_background=True) # was 64 = OOM

        # SDS loss module
        self.sds_loss = SDSLoss(
            pipeline=pipeline,
            eeg_encoder=eeg_encoder,
            eeg_projector=eeg_projector,
            guidance_scale=guidance_scale,
        ).to(device)

        # Optional LPIPS loss for GT supervision
        if use_gt_supervision:
            self.lpips_model = lpips.LPIPS(net='alex').to(device).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False

        # Optimizer (only NeRF parameters)
        self.optimizer = torch.optim.Adam(self.nerf.parameters(), lr=learning_rate)

        # Dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Camera sampling
        self.elevation = 15.0
        self.radius = 1.5

        if use_wandb:
            wandb.init(
                project='nerf-sds-eeg-mvdream',
                config={
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'num_views': num_views,
                    'guidance_scale': guidance_scale,
                    'use_gt_supervision': use_gt_supervision,
                    'gt_weight': gt_weight,
                }
            )

    def train(self):
        global_step = 0
        self.nerf.train()

        for epoch in range(self.num_epochs):
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch in pbar:
                eeg_signal = batch['eeg'].to(self.device)  # (B, C, T)
                gt_images = batch['images'].to(self.device)  # (B, V, C, H, W) in [-1, 1]
                B = eeg_signal.shape[0]
                V = batch['num_views']

                # 1. Sample camera poses
                camera_poses = sample_camera_poses(
                    V,
                    elevation_deg=self.elevation,
                    radius=self.radius,
                    device=self.device
                )  # (V, 4, 4)

                # 2. Render images with NeRF
                rendered_images = []
                for v in range(V):
                    rgb, depth = self.renderer.render_image(
                        self.nerf,
                        camera_poses[v],
                        width=128, #256,
                        height=128, #256
                    )
                    rendered_images.append(rgb.permute(2, 0, 1))  # (3, H, W)

                rendered_images = torch.stack(rendered_images, dim=0)  # (V, 3, H, W)
                rendered_images = rendered_images.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B, V, 3, H, W)

                # 3. Get camera embeddings for MVDream UNet
                camera_embeddings = get_camera(
                    num_frames=V,
                    elevation=self.elevation,
                    extra_view=False,
                    blender_coord=False
                ).to(self.device)

                # 4. Compute SDS loss
                sds_loss = self.sds_loss(rendered_images, eeg_signal, camera_embeddings)

                total_loss = sds_loss

                # 5. Optional: Add GT supervision with LPIPS
                if self.use_gt_supervision:
                    # Convert gt_images from [-1, 1] to [0, 1] for comparison
                    gt_images_01 = (gt_images + 1.0) / 2.0

                    # Flatten for LPIPS
                    rendered_flat = rendered_images.view(-1, 3, 256, 256)
                    gt_flat = gt_images_01.view(-1, 3, 256, 256)

                    # Convert to [-1, 1] for LPIPS
                    rendered_lpips = rendered_flat * 2.0 - 1.0
                    gt_lpips = gt_flat * 2.0 - 1.0

                    lpips_loss = self.lpips_model(rendered_lpips, gt_lpips).mean()
                    total_loss = sds_loss + self.gt_weight * lpips_loss

                # 6. Backprop (only updates NeRF)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Logging
                log_dict = {
                    'sds_loss': sds_loss.item(),
                    'total_loss': total_loss.item(),
                    'step': global_step
                }
                if self.use_gt_supervision:
                    log_dict['lpips_loss'] = lpips_loss.item()

                pbar.set_postfix(log_dict)

                if self.use_wandb:
                    wandb.log(log_dict)

                # Save checkpoint
                if global_step % self.save_steps == 0 and global_step > 0:
                    self.save_checkpoint(global_step)

                # Validation
                if global_step % self.validation_steps == 0 and global_step > 0:
                    self.validate(global_step)

                global_step += 1

        # Final save
        self.save_checkpoint('final')
        if self.use_wandb:
            wandb.finish()

    def validate(self, step):
        """Generate sample renderings and compare with GT."""
        if self.val_dataloader is None:
            return

        self.nerf.eval()

        with torch.no_grad():
            batch = next(iter(self.val_dataloader))
            eeg_signal = batch['eeg'][:1].to(self.device)  # Take 1 sample
            gt_images = batch['images'][:1].to(self.device)  # (1, V, C, H, W) in [-1, 1]
            V = batch['num_views']

            # Sample cameras
            camera_poses = sample_camera_poses(
                V,
                elevation_deg=self.elevation,
                radius=self.radius,
                device=self.device
            )

            # Render
            rendered_images = []
            for v in range(V):
                rgb, depth = self.renderer.render_image(
                    self.nerf,
                    camera_poses[v],
                    width=128, #256,
                    height=128, #256
                )
                rendered_images.append(rgb.cpu().numpy())

            # Convert GT to [0, 1] for visualization
            gt_images_np = ((gt_images[0].cpu().numpy() + 1.0) / 2.0).transpose(0, 2, 3, 1)  # (V, H, W, 3)

            # Save visualization
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, min(8, V), figsize=(4*min(8, V), 8))
            if V == 1:
                axes = axes[:, None]

            for i in range(min(8, V)):
                # Ground truth
                axes[0, i].imshow(gt_images_np[i])
                axes[0, i].set_title(f'GT View {i}')
                axes[0, i].axis('off')

                # Rendered
                axes[1, i].imshow(rendered_images[i])
                axes[1, i].set_title(f'Rendered View {i}')
                axes[1, i].axis('off')

            plt.tight_layout()

            save_path = os.path.join(self.output_dir, f'validation_step_{step}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            if self.use_wandb:
                wandb.log({'validation': wandb.Image(save_path), 'step': step})

        self.nerf.train()

    def save_checkpoint(self, step):
        """Save NeRF weights."""
        save_path = os.path.join(self.output_dir, f'nerf_checkpoint_{step}.pt')
        torch.save({
            'nerf_state_dict': self.nerf.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
        }, save_path)
        print(f"Saved checkpoint: {save_path}")


# ============================================================================
# 7. Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NeRF + SDS training with EEG-conditioned MVDream')

    # Paths
    parser.add_argument('--mvdream_model_path', type=str, required=True,
                        help='Path to base MVDream model')
    parser.add_argument('--lora_checkpoint', type=str, required=True,
                        help='Path to your LoRA checkpoint directory')
    parser.add_argument('--eeg_encoder_ckpt', type=str, required=True,
                        help='Path to EEG encoder checkpoint')
    parser.add_argument('--eeg_projector_ckpt', type=str, required=True,
                        help='Path to EEG projector weights')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data directory')

    # Training params
    parser.add_argument('--output_dir', type=str, default='./nerf_sds_output')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (usually 1 for NeRF+SDS)')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--validation_steps', type=int, default=500)

    # GT supervision (optional)
    parser.add_argument('--use_gt_supervision', action='store_true',
                        help='Use ground-truth images for additional LPIPS supervision')
    parser.add_argument('--gt_weight', type=float, default=0.1,
                        help='Weight for GT LPIPS loss')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("Loading MVDream pipeline with LoRA...")
    print("="*80)

    # 1. Load MVDream pipeline
    pipeline = MVDreamPipeline.from_pretrained(args.mvdream_model_path)

    # Load LoRA adapter
    from peft import PeftModel
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        args.lora_checkpoint,
        adapter_name='default',
        is_trainable=False
    )
    pipeline.unet.set_adapter('default')

    pipeline = pipeline.to(device)

    print("="*80)
    print("Loading EEG encoder and projector...")
    print("="*80)

    # 2. Load EEG encoder
    checkpoint = torch.load(args.eeg_encoder_ckpt)
    hyperparameters = checkpoint['hyper_parameters']
    fs = 512
    window_length = hyperparameters['window_time']

    eeg_encoder = EEGNet_Embedding(
        in_chans=61,
        n_classes=20,
        input_window_samples=int(fs * window_length),
        F1=hyperparameters['F1'],
        F2=hyperparameters['F1'] * hyperparameters['D'],
        D=hyperparameters['D'],
        kernel_length=hyperparameters['kernel_length'],
        depthwise_kernel_length=hyperparameters['depthwise_kernel_length'],
        lr=hyperparameters['lr'],
        epochs=hyperparameters['epochs'],
        weight_decay=hyperparameters['weight_decay'],
        drop_prob=hyperparameters['drop_prob'],
        pool_mode=hyperparameters['pool_mode'],
        separable_kernel_length=hyperparameters['separable_kernel_length'],
        momentum=hyperparameters['bn_momentum'],
        activation=hyperparameters['activation'],
        final_conv_length='auto',
    )
    eeg_encoder.load_state_dict(checkpoint['model_state_dict'], strict=True)
    eeg_encoder = eeg_encoder.to(device)
    eeg_encoder.eval()

    # 3. Load EEG projector
    eeg_projector = nn.Linear(512, 1024).to(device)
    eeg_projector.load_state_dict(torch.load(args.eeg_projector_ckpt, map_location=device))
    eeg_projector.eval()

    print("="*80)
    print("Loading datasets...")
    print("="*80)

    # 4. Create datasets using YOUR MultiViewEEGDataset
    train_dataset = MultiViewEEGDataset(
        data_dir=args.train_data,
        num_views=args.num_views,
        view_selection='uniform',
        preload_images=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multiview_collate_fn_grey,
        pin_memory=False
    )

    val_dataloader = None
    if args.val_data:
        val_dataset = MultiViewEEGDataset(
            data_dir=args.val_data,
            num_views=args.num_views,
            view_selection='uniform',
            preload_images=False
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=multiview_collate_fn_grey,
        )

    print("="*80)
    print("Initializing trainer...")
    print("="*80)

    # 5. Create trainer
    trainer = NeRFSDSTrainer(
        pipeline=pipeline,
        eeg_encoder=eeg_encoder,
        eeg_projector=eeg_projector,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_views=args.num_views,
        guidance_scale=args.guidance_scale,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        save_steps=args.save_steps,
        validation_steps=args.validation_steps,
        use_gt_supervision=args.use_gt_supervision,
        gt_weight=args.gt_weight,
        device=device,
    )

    print("="*80)
    print("Starting training...")
    print("="*80)

    # 6. Train
    trainer.train()

    print("="*80)
    print("Training complete!")
    print("="*80)


if __name__ == '__main__':
    main()
