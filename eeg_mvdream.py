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
from EEGNet_Embedding_version import EEGNet_Embedding


sys.path.append('G:/ninon_workspace/imagery2024/2D_Reconstruction/Generation_2D/reconstruction/code')
sys.path.append('G:/ninon_workspace/imagery2024/3D_Reconstruction/imagedream-eeg/extern/ImageDream')
sys.path.append('G:/ninon_workspace/imagery2024/3D_Reconstruction/imagedream-eeg/extern/ImageDream/imagedream')

from pipeline_mvdream import MVDreamPipeline
from mv_unet import get_camera

def normalize_for_mvdream(img):
    """
    Normalize images for MVDream with grey background.
    Converts RGBA to RGB with grey background and normalizes to [-1, 1].
    """
    print(f"normalize input: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    
    # Convert to tensor first
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    
    # Handle different input formats
    if len(img.shape) == 3:
        if img.shape[-1] == 4:  # HWC with 4 channels (RGBA)
            # Extract RGB and alpha channels
            rgb = img[:, :, :3]  # [H, W, 3]
            alpha = img[:, :, 3:4]  # [H, W, 1]
            
            # Create grey background (0.5 = middle grey)
            grey_background = torch.full_like(rgb, 0.5)
            
            # Alpha blend: result = alpha * foreground + (1 - alpha) * background
            img = alpha * rgb + (1 - alpha) * grey_background
            
            # Convert to CHW
            img = img.permute(2, 0, 1)  # HWC -> CHW
            
        elif img.shape[-1] == 3:  # HWC with 3 channels (RGB)
            img = img.permute(2, 0, 1)  # HWC -> CHW
        else:
            raise ValueError(f"Unexpected number of channels: {img.shape[-1]}")
            
    elif len(img.shape) == 2:  # Grayscale
        # Convert grayscale to RGB
        img = img.unsqueeze(0)  # Add channel dim
        img = img.repeat(3, 1, 1)  # Convert to RGB
    
    # At this point img should be CHW with 3 channels
    assert img.shape[0] == 3, f"Expected 3 channels, got {img.shape[0]}"

    if img.min() >= -1.0 and img.max() <= 1.0:
    # Already normalized, don't normalize again
        print(f"normalize output: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}] (no renorm needed)")
        return img
    else:
        # Need to normalize from [0,1] to [-1,1]
        img = img * 2.0 - 1.0
        print(f"normalize output: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
        return img
    
    # print(f"normalize output: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    # return img


class MVDreamLoRATrainer:
    """LoRA trainer for MVDream finetuning"""
    
    def __init__(
        self,
        pipeline: MVDreamPipeline,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,  # Higher LR for LoRA
        num_train_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",
        output_dir: str = "./eeg_mvdream_lora",
        logging_dir: str = "./logs",
        save_steps: int = 500,
        validation_steps: int = 100,
        max_grad_norm: float = 1.0,
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 500,
        use_wandb: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        self.pipeline = pipeline
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
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

        # EEG conditioning 
        checkpoint_path =   "G:/ninon_workspace/imagery2024/Classification/explainability/checkpoints/object/P12-new.ckpt"
        checkpoint = torch.load(checkpoint_path) #, map_location=device)
        hyper_parameters = checkpoint['hyper_parameters']
        fs = 512
        window_length = hyper_parameters['window_time']
        train_overlap = 0
        val_overlap = 0
        batch_size = hyper_parameters['batch_size']

        self.eeg_encoder = EEGNet_Embedding( 
            in_chans=61,
            n_classes = 6,
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
        self.eeg_projector = nn.Linear(eeg_features_dim, 1024)  # 1024 or 1280 Match CLIP text embedding dim
        
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
        #self.load_checkpoint_if_exists() 
        
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
                "to_k", "to_q", "to_v", "to_out.0",  # Attention layers
                "proj_in", "proj_out",  # Projection layers
                "ff.net.0.proj", "ff.net.2"  # Feed-forward layers
            ],
            lora_dropout=self.lora_dropout,
        )

        #lora_config.inference_mode = False # !! important otherwise adapter is frozen

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

        resume = False

        if resume:

            ckpt = r"G:/ninon_workspace/imagedream_diffusers/eeg_mvdream_lora/checkpoint-1000"

            # Load the LoRA adapter weights
            self.pipeline.unet.load_adapter(ckpt, adapter_name="default", is_trainable=True)

            # Restore optimizer and scheduler state
            self.optimizer.load_state_dict(torch.load(os.path.join(ckpt, "optimizer.bin"), map_location="cpu"))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(ckpt, "scheduler.bin"), map_location="cpu"))
            #self.scaler.load_state_dict(torch.load(os.path.join(ckpt, "scaler.pt"), map_location="cpu"))
            self.resume_step = 1000
            #self.resume_step = self.load_checkpoint_if_exists()    

        device = self.accelerator.device
        self.pipeline.text_encoder.to(device)
        self.pipeline.vae.to(device)
        self.eeg_encoder.to(device)  
        self.eeg_projector.to(device) 
        
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


    
    def load_checkpoint_if_exists(self, checkpoint_path=None):
        """Load LoRA checkpoint if it exists"""
        if checkpoint_path is None:
            # Look for the latest checkpoint in output_dir
            checkpoint_dirs = [d for d in os.listdir(self.output_dir) 
                            if d.startswith('checkpoint-') and d != 'checkpoint-final']
            if not checkpoint_dirs:
                print("No checkpoint found, starting from scratch")
                return 0
                
            # Get the latest checkpoint by step number
            latest_step = max([int(d.split('-')[1]) for d in checkpoint_dirs])
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{latest_step}")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            
            # Load LoRA weights into UNet
            self.pipeline.unet = PeftModel.from_pretrained(
                self.pipeline.unet, 
                checkpoint_path,
                is_trainable=True
            )
            
            # Load training state (optimizer, scheduler, etc.)
            self.accelerator.load_state(checkpoint_path)
            
            # Extract step number from checkpoint path
            step = int(checkpoint_path.split('-')[-1]) if checkpoint_path.split('-')[-1].isdigit() else 0
            print(f"Resumed from step {step}")
            return step
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist")
            return 0
    

    def encode_images_to_latents(self, images):
        """Encode images to latent space using VAE"""
        batch_size, num_views = images.shape[:2]
        images_flat = images.view(-1, *images.shape[2:])  # [B*V, C, H, W]
        
        with torch.no_grad():
            latents = self.pipeline.vae.encode(images_flat).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
            
        latents = latents.view(batch_size, num_views, *latents.shape[1:])  # [B, V, C, H, W]
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
        
        # EEG embeddings (your ventral stream features)
        eeg_signals = batch['eeg'].to(device)
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg_signals, return_embedding=True)
        eeg_embeddings = self.eeg_projector(eeg_features)  # [B, 1024] - f_ventral
        eeg_embeddings = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1)  # [B, 77, 1024]
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
            selected_cameras = all_cameras[[0, 2, 4, 6]]
            camera_batch.append(selected_cameras)
        camera = torch.stack(camera_batch, dim=0).to(device, dtype=noisy_latents.dtype)
        camera_flat = camera.view(-1, camera.shape[-1])
        
        # UNet prediction
        unet_inputs = {
            'x': noisy_latents,
            'timesteps': timesteps,
            'context': eeg_embeddings_flat,  # EEG conditioning
            'num_frames': num_views,
            'camera': camera_flat,
            # 'ip': ip_embeddings.repeat_interleave(num_views, dim=0),
            # 'ip_img': ip_img_latents,
        }
        
        predicted_noise = self.pipeline.unet(**unet_inputs)
        
        # Combined loss: denoising + EEG-image alignment
        denoising_loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        predicted_noise = self.pipeline.unet(**unet_inputs)
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        # Add CLIP perceptual loss
        # if current_step > 500:  # Start after initial training
        #     with torch.no_grad():
        #         # Decode generated and target images
        #         generated_latents = target_latents_flat - predicted_noise  # Approximate clean prediction
        #         generated_images = self.pipeline.vae.decode(generated_latents / self.pipeline.vae.config.scaling_factor).sample
        #         target_images_decoded = self.pipeline.vae.decode(target_latents_flat / self.pipeline.vae.config.scaling_factor).sample
                
        #         # CLIP similarity loss
        #         gen_features = self.pipeline.image_encoder((generated_images + 1) / 2).pooler_output
        #         target_features = self.pipeline.image_encoder((target_images_decoded + 1) / 2).pooler_output
                
        #         perceptual_loss = 1 - F.cosine_similarity(gen_features, target_features, dim=-1).mean()
        #         total_loss = denoising_loss + 0.1 * perceptual_loss
        # else:
        #     total_loss = denoising_loss
        
        # return total_loss
        return loss


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
    
    def save_checkpoint(self, step):
        """Save LoRA checkpoint"""
        save_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA weights
        self.pipeline.unet.save_pretrained(save_path)
        
        # Save training state
        self.accelerator.save_state(save_path)
        
        print(f"LoRA checkpoint saved at {save_path}")
    
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
                    if global_step % (20) == 0: 
                        self.generate_sample(global_step)
            
            progress_bar.close()
        
        # Final save
        self.save_checkpoint("final")
        
        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()

    def generate_sample(self, step, num_samples=1):
        self.pipeline.unet.eval()
        
        with torch.no_grad():
            sample_batch = next(iter(self.train_dataloader))
            eeg_signal = sample_batch['eeg'][0]
            
            # Get GT for comparison
            gt_image = sample_batch['images'][0, 0]
            gt_img_np = gt_image.cpu().numpy().transpose(1, 2, 0)
            gt_img_np = (gt_img_np + 1) / 2
            
            # Generate EEG embeddings exactly like training
            device = next(self.pipeline.unet.parameters()).device
            eeg_features = self.eeg_encoder(eeg_signal.unsqueeze(0).to(device), return_embedding=True)
            eeg_embeddings = self.eeg_projector(eeg_features)
            eeg_context = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1)
            
            # Manual generation loop using your training setup
            height, width, num_frames = 256, 256, 4
            latents = torch.randn(num_frames, 4, height//8, width//8, device=device, dtype=eeg_context.dtype)
            
            # Use scheduler from training
            self.pipeline.scheduler.set_timesteps(20, device=device)
            
            for t in self.pipeline.scheduler.timesteps:
                # Same camera setup as training
                camera = get_camera(8, elevation=0.0, extra_view=False, blender_coord=False)[[0,2,4,6]]
                camera = camera.to(device, dtype=latents.dtype)
                
                # Same UNet inputs as training  
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
            
            # Save comparison
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            axes[0].imshow(gt_img_np)
            axes[0].set_title("GT")
            axes[0].axis('off')
            
            for i in range(4):
                axes[i+1].imshow(generated_images[i])
                axes[i+1].set_title(f"EEG Gen {i}")
                axes[i+1].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"eeg_sample_step_{step}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        self.pipeline.unet.train()

    def generate_sample_eeg(self, step, num_samples=10):
        self.pipeline.unet.eval()
        
        for dataset_name, dataloader in [("train", self.train_dataloader)]:
            for sample_idx in range(num_samples):
                with torch.no_grad():
                    sample_batch = next(iter(dataloader))
                    eeg_signal = sample_batch['eeg'][0]
                    
                    # Get GT image for comparison (first view)
                    gt_image = sample_batch['images'][0, 0]  # [C, H, W]
                    gt_img_np = gt_image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                    gt_img_np = (gt_img_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
                    
                    generated_images = self.pipeline.__call_eeg__(
                        conditioning_image=gt_img_np, # added for test 
                        eeg_signal=eeg_signal,
                        eeg_encoder=self.eeg_encoder,
                        eeg_projector=self.eeg_projector,
                        height=256,
                        width=256,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        num_frames=4
                    )
                    
                    # Save images - show GT + generated views
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # GT + 4 generated
                    
                    axes[0].imshow(gt_img_np)
                    axes[0].set_title(f"GT ({dataset_name})")
                    axes[0].axis('off')
                    
                    for i in range(4):
                        axes[i+1].imshow(generated_images[i])
                        axes[i+1].set_title(f"Generated View {i}")
                        axes[i+1].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(self.output_dir, f"eeg_sample_{dataset_name}_{sample_idx}_step_{step}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"EEG-generated sample {sample_idx+1}/{num_samples} saved to {save_path}")
        
        self.pipeline.unet.train()

    def generate_sample_old(self, step, num_samples=10):
        self.pipeline.unet.eval()
        
        for dataset_name, dataloader in [("train", self.train_dataloader)]:
            for sample_idx in range(num_samples):
                with torch.no_grad():
                    sample_batch = next(iter(dataloader))
                    eeg_signal = sample_batch['eeg'][0]  # Get EEG signal
                    
                    # Generate using EEG conditioning
                    generated_images = self.pipeline.__call_eeg__(
                        eeg_signal=eeg_signal,
                        eeg_encoder=self.eeg_encoder,
                        eeg_projector=self.eeg_projector,
                        height=256,
                        width=256,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        num_frames=4
                    )
                      # Save with sample index in filename
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                    axes[0].imshow(cond_img_np)
                    axes[0].set_title(f"Input ({dataset_name})")
                    axes[0].axis('off')
                
                    for i in range(4):
                        axes[i+1].imshow(generated_images[i])
                        axes[i+1].set_title(f"Generated View {i}")
                        axes[i+1].axis('off')
                
                    plt.tight_layout()
                    save_path = os.path.join(self.output_dir, f"sample_{dataset_name}_{sample_idx}_step_{step}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
                    print(f"{dataset_name.capitalize()} sample {sample_idx+1}/{num_samples} saved to {save_path}")
    
        self.pipeline.unet.train()   

    def generate_sample_img(self, step, num_samples=10):
        """Generate sample images for validation from both train and val sets"""
        self.pipeline.unet.eval()
    
        datasets_to_test = [("train", self.train_dataloader)]
        if self.val_dataloader:
            datasets_to_test.append(("val", self.val_dataloader))
    
        for dataset_name, dataloader in datasets_to_test:
            # Generate multiple samples instead of just one
            for sample_idx in range(num_samples):
                with torch.no_grad():
                    sample_batch = next(iter(dataloader))
                    conditioning_image = sample_batch['images'][0, 0]
                    cond_img_np = conditioning_image.cpu().numpy().transpose(1, 2, 0)
                    cond_img_np = (cond_img_np + 1) / 2
                
                    generated_images = self.pipeline(
                        prompt="a 3D object",
                        image=cond_img_np,
                        height=256,
                        width=256,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        num_frames=4,
                        output_type="numpy"
                    )
                
                    # Save with sample index in filename
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                    axes[0].imshow(cond_img_np)
                    axes[0].set_title(f"Input ({dataset_name})")
                    axes[0].axis('off')
                
                    for i in range(4):
                        axes[i+1].imshow(generated_images[i])
                        axes[i+1].set_title(f"Generated View {i}")
                        axes[i+1].axis('off')
                
                    plt.tight_layout()
                    save_path = os.path.join(self.output_dir, f"sample_{dataset_name}_{sample_idx}_step_{step}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
                    print(f"{dataset_name.capitalize()} sample {sample_idx+1}/{num_samples} saved to {save_path}")
    
        self.pipeline.unet.train()

    def generate_sample_train_only(self, step, num_samples=1):
        """Generate sample images for validation"""
        self.pipeline.unet.eval()
        
        with torch.no_grad():
            # Get a sample from your dataset for conditioning
            sample_batch = next(iter(self.train_dataloader))
            conditioning_image = sample_batch['images'][0, 0]  # First view of first sample
            
            # Convert to numpy for pipeline
            cond_img_np = conditioning_image.cpu().numpy().transpose(1, 2, 0)
            cond_img_np = (cond_img_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
            
            # Generate multiview images
            generated_images = self.pipeline(
                prompt="a 3D object",
                image=cond_img_np,
                height=256,
                width=256,
                num_inference_steps=20,  # Fewer steps for faster generation
                guidance_scale=7.0,
                num_frames=4,
                output_type="numpy"
            )
            
            # Save images
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Show conditioning image
            axes[0].imshow(cond_img_np)
            axes[0].set_title("Input")
            axes[0].axis('off')
            
            # Show generated views
            for i in range(4):
                axes[i+1].imshow(generated_images[i])
                axes[i+1].set_title(f"Generated View {i}")
                axes[i+1].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"sample_step_{step}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Sample saved to {save_path}")
        
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

        # if images.shape[1] == 4:  # 4 channels (RGBA)
        #     processed_images = []
        #     for view_img in images:  # Process each view
        #         img_hwc = view_img.permute(1, 2, 0)  # CHW -> HWC
                
        #         # Force RGBA to RGB conversion regardless of range
        #         if img_hwc.shape[-1] == 4:
        #             rgb = img_hwc[:, :, :3]
        #             alpha = img_hwc[:, :, 3:4]
        #             grey_bg = torch.full_like(rgb, 0.5)
        #             img_hwc = alpha * rgb + (1 - alpha) * grey_bg
                
        #         img_processed = img_hwc.permute(2, 0, 1)  # HWC -> CHW
        #         processed_images.append(img_processed)


    
    
    # Stack all samples
    images = torch.stack(images_list)  # [B, V, C, H, W]
    classes = torch.tensor([item['class'] for item in batch])
    eeg_signals = torch.stack([item['eeg'] for item in batch]) # should replicate for each image of trials?
    
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
    parser.add_argument("--output_dir", type=str, default="./eeg_mvdream_lora", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_views", type=int, default=4, help="Number of views per object")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
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
        combined_dataset, #train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        collate_fn=multiview_collate_fn_grey,
        pin_memory=False,
    )
    
    val_dataloader = DataLoader(
        test_dataset, #val_dataset,
        batch_size=args.batch_size,
        shuffle=True, #False,
        num_workers=0,  
        collate_fn=multiview_collate_fn_grey,
        pin_memory=False,
    )
    
    # Create trainer
    trainer = MVDreamLoRATrainer(
        pipeline=pipeline,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Test one batch first
    print("Testing data loading...")
    for batch in train_dataloader:
        print(f"Batch shapes:")
        print(f"  Images: {batch['images'].shape}")
        print(f"  Classes: {batch['classes'].shape}")
        if batch['camera_poses'] is not None:
            print(f"  Camera poses: {batch['camera_poses'].shape}")
        print(f"  Image range: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
        break
    
    # Start training
    print("Starting LoRA training...")
    trainer.train()


if __name__ == "__main__":
    main()