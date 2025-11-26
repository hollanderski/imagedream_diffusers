import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from typing import Optional, List, Dict, Any
import json
from transformers import CLIPTokenizer
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
import wandb
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
import sys
import torch.nn as nn
import lpips
import shutil
from EEGNet_Embedding_version import EEGNet_Embedding


sys.path.append('G:/ninon_workspace/imagery2024/2D_Reconstruction/Generation_2D/reconstruction/code')
sys.path.append('G:/ninon_workspace/imagery2024/3D_Reconstruction/imagedream-eeg/extern/ImageDream')
sys.path.append('G:/ninon_workspace/imagery2024/3D_Reconstruction/imagedream-eeg/extern/ImageDream/imagedream')

from pipeline_mvdream import MVDreamPipeline
from mv_unet import get_camera


class MVDreamLoRATrainer:
    """LoRA trainer for MVDream finetuning with LPIPS perceptual loss"""
    
    def __init__(
        self,
        pipeline: MVDreamPipeline,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_train_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",
        output_dir: str = "./eeg_mvdream_lora_8v_fixed",
        logging_dir: str = "./logs",
        save_steps: int = 500,
        validation_steps: int = 500, #100,
        max_grad_norm: float = 1.0,
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 500,
        use_wandb: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_perceptual: bool = False,
        perceptual_weight: float = 0.1,
        perceptual_start_step: int = 500,
        lpips_net: str = 'alex',
        keep_only_latest_checkpoint: bool = True,
        max_checkpoints: int = 2,
    ):
        self.pipeline = pipeline
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.validation_steps = validation_steps
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.use_wandb = use_wandb
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        self.perceptual_start_step = perceptual_start_step
        self.keep_only_latest_checkpoint = keep_only_latest_checkpoint
        self.max_checkpoints = max_checkpoints

        # EEG conditioning 
        checkpoint_path = "G:/ninon_workspace/imagery2024/Classification/explainability/checkpoints/object/P12-new.ckpt"
        checkpoint_path = "G:/ninon_workspace/imagery2024/Classification/explainability/checkpoints/object/P10_best_model_692t6h6b_63.0208.ckpt"
        weight_path = "G:/ninon_workspace/imagedream_diffusers/eegnet_best_supcon_20251123_11-sil1.pt"
        checkpoint = torch.load(checkpoint_path)
        hyper_parameters = checkpoint['hyper_parameters']
        fs = 512
        window_length = hyper_parameters['window_time']
        
        self.eeg_encoder = EEGNet_Embedding( 
            in_chans=61,
            n_classes=6,
            input_window_samples=fs*window_length,  
            F1=hyper_parameters["F1"],  
            F2=hyper_parameters["F1"] * hyper_parameters["D"],  
            D=hyper_parameters["D"],   
            kernel_length=hyper_parameters["kernel_length"],  
            depthwise_kernel_length=hyper_parameters["depthwise_kernel_length"],  
            lr=hyper_parameters["lr"],
            epochs=hyper_parameters["epochs"],
            weight_decay=hyper_parameters["weight_decay"],
            drop_prob=hyper_parameters["drop_prob"],
            pool_mode=hyper_parameters["pool_mode"], 
            separable_kernel_length=hyper_parameters["separable_kernel_length"],
            momentum=hyper_parameters["bn_momentum"],
            activation=hyper_parameters["activation"], 
            final_conv_length="auto", 
        )

        self.eeg_encoder.load_state_dict(checkpoint['model_state_dict'], strict=True)
        eeg_features_dim = 512
        self.eeg_projector = nn.Linear(eeg_features_dim, 1024)
        
        # Initialize LPIPS model
        if self.use_perceptual:
            self.lpips_model = lpips.LPIPS(net=lpips_net).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="wandb" if use_wandb else None,
            project_dir=logging_dir,
        )
        
        # Make output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup model for training
        self.setup_training()
        
    def setup_training(self):
        """Setup LoRA model, optimizer, and scheduler"""
        
        # Freeze all components except UNet
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        
        # Setup LoRA for UNet
        print("Setting up LoRA for UNet...")
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ],
            lora_dropout=self.lora_dropout,
        )

        # Apply LoRA to UNet
        self.pipeline.unet = get_peft_model(self.pipeline.unet, lora_config)

        # Setup optimizer - only LoRA parameters
        trainable_params = [p for p in self.pipeline.unet.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-08,
        )

        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            
        # Setup scheduler
        self.lr_scheduler = get_scheduler(
                self.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=max_train_steps,
        )

        resume = True
        if resume: 
            ckpt = r"G:/ninon_workspace/imagedream_diffusers/eeg_mvdream_lora_2v_aux_P10-debug/checkpoint-2000"
            self.pipeline.unet.load_adapter(ckpt, adapter_name="default", is_trainable=True)
            self.optimizer.load_state_dict(torch.load(os.path.join(ckpt, "optimizer.bin"), map_location="cpu"))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(ckpt, "scheduler.bin"), map_location="cpu"))
            self.resume_step = 2000

        device = self.accelerator.device
        self.pipeline.text_encoder.to(device)
        self.pipeline.vae.to(device)
        self.eeg_encoder.to(device)  
        self.eeg_projector.to(device)
        if self.use_perceptual:
            self.lpips_model.to(device)
        
        # Prepare everything with accelerator
        (
            self.pipeline.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.pipeline.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        if self.test_dataloader is not None:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)


    def encode_images_to_latents(self, images):
        """Encode images to latent space using VAE"""
        batch_size, num_views = images.shape[:2]
        images_flat = images.view(-1, *images.shape[2:])
        
        with torch.no_grad():
            latents = self.pipeline.vae.encode(images_flat).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
            
        latents = latents.view(batch_size, num_views, *latents.shape[1:])
        return latents
    
    def get_text_embeddings(self, batch_size, device):
        """Generate text embeddings for dummy prompts"""
        dummy_prompts = ["a 3D object"] * batch_size
        
        with torch.no_grad():
            text_inputs = self.pipeline.tokenizer(
                dummy_prompts,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_embeddings = self.pipeline.text_encoder(
                text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )[0]
            
        return text_embeddings
    
    def compute_loss(self, batch, current_step=0, max_steps=1):
        """Compute training loss with EEG reconstruction component"""
        device = self.accelerator.device
        
        # Move batch to device
        images = batch['images'].to(device)  # [B, V, C, H, W]
        batch_size = images.shape[0]
        num_views = images.shape[1]
        

        # DEBUG START 
        eeg_signals = batch['eeg'].to(device)          # [B, ...]
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg_signals, return_embedding=True)  # [B, 512]
        eeg_embeddings = self.eeg_projector(eeg_features)                       # [B, 1024]
        eeg_embeddings_for_views = eeg_embeddings.unsqueeze(1).repeat(1, num_views, 1)  # [B, V, 1024]
        eeg_context_token = eeg_embeddings_for_views.unsqueeze(2).repeat(1, 1, 77, 1)   # [B, V, 77, 1024]


        # SCALE
        current_norm = eeg_context_token.norm(dim=-1).mean()
        target_norm = 165.0
        scaling_factor = target_norm / (current_norm.item() + 1e-8)
        eeg_context_token_scaled = eeg_context_token * scaling_factor

        eeg_context_flat = eeg_context_token_scaled.view(-1, 77, 1024) 
        
        # Encode target images
        target_latents = self.encode_images_to_latents(images)
        target_latents_flat = target_latents.view(-1, *target_latents.shape[2:])
        
        # Standard denoising loss
        noise = torch.randn_like(target_latents_flat)
        timesteps = torch.randint(0, self.pipeline.scheduler.config.num_train_timesteps, 
                                (batch_size * num_views,), device=device).long()
        noisy_latents = self.pipeline.scheduler.add_noise(target_latents_flat, noise, timesteps)
        
        # Get camera poses
        camera_batch = []
        for _ in range(batch_size):
            all_cameras = get_camera(num_views, elevation=0.0, extra_view=False, blender_coord=False)
            selected_cameras = all_cameras #[[0, 2, 4, 6]]
            camera_batch.append(selected_cameras)
        camera = torch.stack(camera_batch, dim=0).to(device, dtype=noisy_latents.dtype)
        camera_flat = camera.view(-1, camera.shape[-1])        
        # UNet prediction
        unet_inputs = {
            'x': noisy_latents,
            'timesteps': timesteps,
            'context': eeg_context_flat, # eeg_embeddings_flat,  # EEG conditioning Scaled ! 
            'num_frames': num_views,
            'camera': camera_flat,
        }
        
        predicted_noise = self.pipeline.unet(**unet_inputs)

        # Auxiliary loss 
        class_logits = self.eeg_encoder(eeg_signals)  # [B, num_classes]
        log_probs = torch.log_softmax(class_logits, dim=-1)  #  log-probs
        gt_classes = batch['classes'].to(device)             # [B] -1 ! 
        aux_loss = F.nll_loss(log_probs, gt_classes)
        aux_weight = 0.2  # Tune 
        
        #. 
        
        
        #denoising_loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        #predicted_noise = self.pipeline.unet(**unet_inputs)

        # Combined loss: denoising + EEG-image alignment
        denoising_loss = F.mse_loss(predicted_noise, noise, reduction='mean') # before return this

        total_loss = denoising_loss + aux_weight * aux_loss

        return total_loss

    def compute_loss_perceptual(self, batch, current_step=0, max_steps=1):
        """Compute training loss with LPIPS perceptual component"""
        device = self.accelerator.device
        
        # Move batch to device
        images = batch['images'].to(device)
        batch_size = images.shape[0]
        num_views = images.shape[1]
        
        # EEG embeddings
        eeg_signals = batch['eeg'].to(device)
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg_signals, return_embedding=True)
        eeg_embeddings = self.eeg_projector(eeg_features)
        eeg_embeddings = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1)
        eeg_embeddings_expanded = eeg_embeddings.unsqueeze(1).repeat(1, num_views, 1, 1)
        eeg_embeddings_flat = eeg_embeddings_expanded.view(-1, *eeg_embeddings.shape[1:])
        
        # Encode target images
        target_latents = self.encode_images_to_latents(images)
        target_latents_flat = target_latents.view(-1, *target_latents.shape[2:])
        
        # Standard denoising loss
        noise = torch.randn_like(target_latents_flat)
        timesteps = torch.randint(0, self.pipeline.scheduler.config.num_train_timesteps, 
                                (batch_size * num_views,), device=device).long()
        noisy_latents = self.pipeline.scheduler.add_noise(target_latents_flat, noise, timesteps)
        
        # Get camera poses
        camera_batch = []
        for _ in range(batch_size):
            all_cameras = get_camera(8, elevation=0.0, extra_view=False, blender_coord=False)
            selected_cameras = all_cameras #[[0, 2, 4, 6]]
            camera_batch.append(selected_cameras)
        camera = torch.stack(camera_batch, dim=0).to(device, dtype=noisy_latents.dtype)
        camera_flat = camera.view(-1, camera.shape[-1])
        
        # UNet prediction
        unet_inputs = {
            'x': noisy_latents,
            'timesteps': timesteps,
            'context': eeg_embeddings_flat,
            'num_frames': num_views,
            'camera': camera_flat,
        }
        
        predicted_noise = self.pipeline.unet(**unet_inputs)
        
        # Base denoising loss
        denoising_loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        # LPIPS perceptual loss (after warmup)
        perceptual_loss = torch.tensor(0.0, device=device)
        if self.use_perceptual and current_step > self.perceptual_start_step:
            with torch.no_grad():
                # Get clean latents estimate
                clean_latents = target_latents_flat - predicted_noise
                
                # Decode to images
                pred_images = self.pipeline.vae.decode(clean_latents / self.pipeline.vae.config.scaling_factor).sample
                target_images = self.pipeline.vae.decode(target_latents_flat / self.pipeline.vae.config.scaling_factor).sample
                
                # Clamp to [-1, 1] for LPIPS
                pred_images = torch.clamp(pred_images, -1, 1)
                target_images = torch.clamp(target_images, -1, 1)
                
            # LPIPS loss (requires gradients for backprop)
            with torch.enable_grad():
                # Randomly sample a subset of views to reduce memory usage
                num_samples = min(4, pred_images.shape[0])
                indices = torch.randperm(pred_images.shape[0])[:num_samples]
                
                lpips_loss = self.lpips_model(
                    pred_images[indices], 
                    target_images[indices]
                ).mean()
                perceptual_loss = lpips_loss * self.perceptual_weight
        
        total_loss = denoising_loss + perceptual_loss
        
        return {
            'total_loss': total_loss,
            'denoising_loss': denoising_loss,
            'perceptual_loss': perceptual_loss
        }

    def validation_step(self):
        """Perform validation"""
        if self.val_dataloader is None:
            return {}
        
        self.pipeline.unet.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self.compute_loss(batch)
                val_losses.append(loss.item())
        
        self.pipeline.unet.train()
        
        return {
            'val_loss': np.mean(val_losses),
            'val_loss_std': np.std(val_losses)
        }
    
    
    
    def _cleanup_old_checkpoints(self, current_step):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoint_dirs = []
        for d in os.listdir(self.output_dir):
            if d.startswith('checkpoint-') and d != 'checkpoint-final':
                try:
                    step_num = int(d.split('-')[1])
                    checkpoint_dirs.append((step_num, d))
                except ValueError:
                    continue
        
        # Sort by step number (newest first)
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only the latest N checkpoints
        checkpoints_to_remove = checkpoint_dirs[self.max_checkpoints:]
        
        for _, checkpoint_dir in checkpoints_to_remove:
            checkpoint_path = os.path.join(self.output_dir, checkpoint_dir)
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
    
    def save_checkpoint(self, step):
        """Save LoRA checkpoint and manage storage"""
        save_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA weights
        self.pipeline.unet.save_pretrained(save_path)
        
        # Save training state
        self.accelerator.save_state(save_path)
        
        # Cleanup old checkpoints if requested
        if self.keep_only_latest_checkpoint and str(step) != "final":
            self._cleanup_old_checkpoints(step)
    
    def train(self):
        """Main training loop"""
        # Initialize tracking
        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="eeg-mvdream-lora-finetune",
                config={
                    "learning_rate": self.learning_rate,
                    "num_train_epochs": self.num_train_epochs,
                    "train_batch_size": self.train_dataloader.batch_size,
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                }
            )
        
        # Training loop
        global_step = getattr(self, 'resume_step', 0) 
        self.pipeline.unet.train()
        
        for epoch in range(self.num_train_epochs):
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch+1}/{self.num_train_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in self.train_dataloader:
               

                with self.accelerator.accumulate(self.pipeline.unet):
                    #loss = self.compute_loss(batch)
                    if not hasattr(self, 'max_train_steps'):
                        self.max_train_steps = self.num_train_epochs * len(self.train_dataloader)
                    
                    loss = self.compute_loss(batch, global_step, self.max_train_steps)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.pipeline.unet.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    logs = {
                        "train_loss": loss.detach().item(), # TODO : add perceptual loss
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    
                    # Validation
                    if global_step % self.validation_steps == 0:
                        val_logs = self.validation_step()
                        logs.update(val_logs)
                    
                    # Log to wandb
                    if self.use_wandb and self.accelerator.is_main_process:
                        self.accelerator.log(logs, step=global_step)
                    
                    progress_bar.set_postfix(logs)
                    
                    # Save checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)
                        
                    # Add sample generation
                    # if global_step % (200) == 0: #200
                    #     self.generate_sample(global_step)
                    if global_step % (200) == 0:
                        self.diagnostic(global_step)
                    
            
            progress_bar.close()
        
        # Final save
        self.save_checkpoint("final")
        
        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()

    def diagnostic(self, step, num_samples=5):
        """Generate diagnostic context-conditional generations for quick debugging."""
        self.pipeline.unet.eval()
        datasets_to_test = [("train", self.train_dataloader)]
        if self.val_dataloader:
            datasets_to_test.append(("val", self.val_dataloader))
        if self.test_dataloader:
            datasets_to_test.append(("test", self.test_dataloader))

        for dataset_name, dataloader in datasets_to_test:
            data_iter = iter(dataloader)
            for sample_idx in range(num_samples):
                with torch.no_grad():
                    try:
                        sample_batch = next(data_iter)
                    except StopIteration:
                        break

                    device = next(self.pipeline.unet.parameters()).device
                    eeg_batch = sample_batch['eeg'].to(device)  # [B, ...]
                    batch_size = eeg_batch.shape[0]
                    num_views = sample_batch['images'].shape[1]
                    # Compute EEG embeddings for batch
                    eeg_features = self.eeg_encoder(eeg_batch, return_embedding=True)
                    eeg_embeddings = self.eeg_projector(eeg_features)
                    eeg_context = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1)  # [B, 77, ctx_dim]

                    # Scale EEG embedding
                    current_norm = eeg_context.norm(dim=-1).mean()
                    target_norm = 165.0
                    scaling_factor = target_norm / (current_norm.item() + 1e-8)
                    eeg_context_scaled = eeg_context * scaling_factor
                    print(f"EEG context scaled from mean norm {current_norm:.2f} to {eeg_context_scaled.norm(dim=-1).mean():.2f}")

                    # Prepare shuffled and noise contexts
                    perm = torch.randperm(batch_size)
                    eeg_context_shuffled = eeg_context_scaled[perm]
                    eeg_context_noise = torch.randn_like(eeg_context_scaled)

                    # Per-sample generation
                    for j in range(batch_size):
                        gt_images = sample_batch['images'][j]  # [V, C, H, W]
                        gt_images_np = [
                            (gt_images[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2 for i in range(num_views)
                        ]
                        contexts = {
                            'real': eeg_context_scaled[j:j+1],           # real EEG
                            'shuf': eeg_context_shuffled[j:j+1],         # shuffled
                            'noise': eeg_context_noise[j:j+1],           # random noise
                        }
                        for ctx_type, ctx in contexts.items():
                            # Generate latents and run diffusion
                            height, width, num_frames = 256, 256, num_views
                            latents = torch.randn(num_frames, 4, height//8, width//8, device=device, dtype=ctx.dtype)
                            self.pipeline.scheduler.set_timesteps(20, device=device)
                            for t in self.pipeline.scheduler.timesteps:
                                camera = get_camera(num_frames, elevation=0.0, extra_view=False, blender_coord=False)
                                camera = camera.to(device, dtype=latents.dtype)
                                noise_pred = self.pipeline.unet(
                                    x=latents,
                                    timesteps=torch.tensor([t] * num_frames, device=device),
                                    context=ctx.repeat(num_frames, 1, 1),
                                    num_frames=num_frames,
                                    camera=camera
                                )
                                latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                            generated_images = self.pipeline.decode_latents(latents)
                            # Save GT (row 0) and generated (row 1, per context)
                            import matplotlib.pyplot as plt
                            fig, axes = plt.subplots(2, num_views, figsize=(4*num_views, 8))
                            for i in range(num_views):
                                axes[0, i].imshow(gt_images_np[i])
                                axes[0, i].set_title(f"GT View {i} ({dataset_name})")
                                axes[0, i].axis('off')
                                axes[1, i].imshow(generated_images[i])
                                axes[1, i].set_title(f"{ctx_type.title()} Gen View {i}")
                                axes[1, i].axis('off')
                            plt.tight_layout()
                            save_path = os.path.join(
                                self.output_dir, f"{ctx_type}_context_sample_{dataset_name}_{sample_idx}_samp{j}_step_{step}.png"
                            )
                            plt.savefig(save_path, dpi=150, bbox_inches='tight')
                            plt.close()
        self.pipeline.unet.train()


    def generate_sample(self, step, num_samples=5):
        """Generate sample images for both train and validation sets, each time a new item from the dataloader."""
        self.pipeline.unet.eval()

        datasets_to_test = [("train", self.train_dataloader)]
        if self.val_dataloader:
            datasets_to_test.append(("val", self.val_dataloader))
        if self.test_dataloader:
            datasets_to_test.append(("test", self.test_dataloader))

        for dataset_name, dataloader in datasets_to_test:
            data_iter = iter(dataloader)  # create the iterator only once per dataset
            for sample_idx in range(num_samples):
                with torch.no_grad():
                    try:
                        sample_batch = next(data_iter)
                    except StopIteration:
                        break  # dataloader exhausted

                    # DEBUG SHUFFLE CONTEXT
                    eeg_batch = sample_batch['eeg']
                    batch_size = eeg_batch.shape[0]
                    eeg_features = self.eeg_encoder(eeg_batch, return_embedding=True)
                    eeg_embeddings = self.eeg_projector(eeg_features)
                    eeg_context = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1)

                    # Scaling EEG feature vector 
                    current_norm = eeg_context.norm(dim=-1).mean()  
                    target_norm = 165.0  # 100–200 
                    scaling_factor = target_norm / (current_norm.item() + 1e-8)  # non zero divide
                    eeg_context_scaled = eeg_context * scaling_factor
                    print(f"EEG context scaled from mean norm {current_norm:.2f} to {eeg_context_scaled.norm(dim=-1).mean():.2f}")

                    perm = torch.randperm(batch_size)
                    eeg_context_shuffled = eeg_context[perm]

                    # Instead of always picking index 0, cycle through the batch
                    for j in range(sample_batch['batch_size']):
                        if j >= 1:
                            break  # Only 1 sample per file, but just in case
                        eeg_signal = sample_batch['eeg'][j]

                        # Get all 8 GT views for comparison
                        gt_images = sample_batch['images'][j]  # shape: [8, C, H, W]
                        gt_images_np = [
                            (gt_images[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2 for i in range(8)
                        ]

                        # Generate EEG embeddings
                        device = next(self.pipeline.unet.parameters()).device
                        eeg_features = self.eeg_encoder(eeg_signal.unsqueeze(0).to(device), return_embedding=True)
                        eeg_embeddings = self.eeg_projector(eeg_features)
                        eeg_context = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1) ##

                        # shuffle
                        eeg_context_one = eeg_context_shuffled[j].unsqueeze(0)
                        eeg_context = eeg_context_one


                        #eeg_context = torch.randn_like(eeg_context) # NOISE

                        print(f"Sample {j}, EEG class: {sample_batch['classes'][j]}")
                        print(f"eeg_context shape: {eeg_context.shape}, mean: {eeg_context.mean().item():.4f}")


                        # Generate 8 views
                        height, width, num_frames = 256, 256, 8
                        latents = torch.randn(num_frames, 4, height//8, width//8, device=device, dtype=eeg_context.dtype)
                        self.pipeline.scheduler.set_timesteps(20, device=device)

                        for t in self.pipeline.scheduler.timesteps:
                            camera = get_camera(8, elevation=0.0, extra_view=False, blender_coord=False)
                            camera = camera.to(device, dtype=latents.dtype)

                            for v in range(num_frames):
                                print(f"View {v}: context mean {eeg_context[0].mean().item():.4f}")

                            noise_pred = self.pipeline.unet(
                                x=latents,
                                timesteps=torch.tensor([t] * num_frames, device=device),
                                context=eeg_context.repeat(num_frames, 1, 1),
                                num_frames=num_frames,
                                camera=camera
                            )
                            latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                        # Decode
                        generated_images = self.pipeline.decode_latents(latents)

                        # Save comparison: 8 GT views (row 0), 8 generated (row 1)
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(2, 8, figsize=(32, 8))

                        for i in range(8):
                            axes[0, i].imshow(gt_images_np[i])
                            axes[0, i].set_title(f"GT View {i} ({dataset_name})")
                            axes[0, i].axis('off')

                            axes[1, i].imshow(generated_images[i])
                            axes[1, i].set_title(f"Generated View {i}")
                            axes[1, i].axis('off')

                        plt.tight_layout()
                        save_path = os.path.join(
                            self.output_dir, f"eeg_contextshuffled_sample_{dataset_name}_{sample_idx}_step_{step}.png"
                        )
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
        self.pipeline.unet.train()


def multiview_collate_fn_grey(batch):
    """
    Custom collate function for multiview data with grey background processing.
    """
    # Extract and process images with grey background
    images_list = []
    for item in batch:
        images = item['images']  # [V, C, H, W] 
        
        # If images are RGBA, convert to RGB with grey background
        if images.shape[1] == 4:  # 4 channels (RGBA)
            processed_images = []
            for view_img in images:  # Process each view
                img_hwc = view_img.permute(1, 2, 0)  # CHW -> HWC
                
                # Handle RGBA to RGB with grey background
                rgb = img_hwc[:, :, :3]  # [H, W, 3]
                alpha = img_hwc[:, :, 3]   # [H, W]
                
                grey_background = torch.full_like(rgb, 0.5)
                alpha_mask = alpha.unsqueeze(-1)  # [H, W, 1]
                img_rgb = torch.where(alpha_mask > 0.5, rgb, grey_background)
                
                # Convert back to CHW and ensure [-1,1] range
                img_processed = img_rgb.permute(2, 0, 1)  # HWC -> CHW
                if img_processed.max() <= 1.0 and img_processed.min() >= 0.0:
                    img_processed = img_processed * 2.0 - 1.0  # [0,1] -> [-1,1]
                
                processed_images.append(img_processed)
            
            images = torch.stack(processed_images)  # [V, C, H, W]
        
        images_list.append(images)
    
    # Stack all samples
    images = torch.stack(images_list)  # [B, V, C, H, W]
    classes = torch.tensor([item['class'] for item in batch])
    eeg_signals = torch.stack([item['eeg'] for item in batch])
    
    # Handle camera poses
    camera_poses = None
    if batch[0]['camera_poses'] is not None:
        camera_poses = torch.stack([item['camera_poses'] for item in batch])
    
    return {
        'images': images,
        'classes': classes,
        'eeg': eeg_signals,
        'camera_poses': camera_poses,
        'num_views': images.shape[1],
        'batch_size': images.shape[0]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to EEG training data (.npy file)")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to EEG validation data (.npy file)")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to EEG test data (.npy file)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained MVDream model")
    parser.add_argument("--output_dir", type=str, default="./eeg_mvdream_lora_2v_aux_P10-debug", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views per object")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_perceptual", action="store_true", help="Use LPIPS perceptual loss")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument("--perceptual_start_step", type=int, default=500, help="Step to start perceptual loss")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS network")
    parser.add_argument("--keep_only_latest", action="store_true", help="Keep only latest checkpoints to save space")
    parser.add_argument("--max_checkpoints", type=int, default=2, help="Maximum number of checkpoints to keep")
    
    args = parser.parse_args()
    
    # Load pretrained pipeline
    pipeline = MVDreamPipeline.from_pretrained(args.model_path)
    
    # Import your dataset
    from multiview_dataset import MultiViewEEGDataset
    
    # Create dataset
    train_dataset = MultiViewEEGDataset(
        data_dir=args.data_dir,
        num_views=args.num_views,
        view_selection='uniform',
        preload_images=False
    )
    
    val_dataset = MultiViewEEGDataset(
        data_dir=args.val_dir,
        num_views=args.num_views,
        view_selection='uniform',
        preload_images=False
    )

    test_dataset = MultiViewEEGDataset(
        data_dir=args.test_dir,
        num_views=args.num_views,
        view_selection='uniform',
        preload_images=False
    )

    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, #combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multiview_collate_fn_grey,
        pin_memory=False,
    )

    from collections import Counter

    class_counts = Counter()
    for batch in train_dataloader:
        # Assuming batch['classes'] is a tensor of class labels (batch_size,)
        for cls in batch['classes']:
            class_counts[int(cls)] += 1

    print("Class distribution in dataloader:")
    for cls, count in sorted(class_counts.items()):
        print(f"Class {cls}: {count} samples")
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        collate_fn=multiview_collate_fn_grey,
        pin_memory=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        collate_fn=multiview_collate_fn_grey,
        pin_memory=False,
    )
    
    # Create trainer
    trainer = MVDreamLoRATrainer(
        pipeline=pipeline,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_perceptual=args.use_perceptual,
        perceptual_weight=args.perceptual_weight,
        perceptual_start_step=args.perceptual_start_step,
        lpips_net=args.lpips_net,
        keep_only_latest_checkpoint=args.keep_only_latest,
        max_checkpoints=args.max_checkpoints,
    )
    
    # Start training
    print("Starting LoRA training of Multiview Diffusion Model")
    trainer.train()


if __name__ == "__main__":
    main()