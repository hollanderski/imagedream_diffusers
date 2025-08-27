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
from peft import LoraConfig, get_peft_model
from pathlib import Path
import sys

sys.path.append('G:/ninon_workspace/imagery2024/2D_Reconstruction/Generation_2D/reconstruction/code')
sys.path.append('G:/ninon_workspace/imagery2024/3D_Reconstruction/imagedream-eeg/extern/ImageDream')
sys.path.append('G:/ninon_workspace/imagery2024/3D_Reconstruction/imagedream-eeg/extern/ImageDream/imagedream')

# Import your components
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
    
    # Normalize to [-1, 1] (MVDream expects this range)
    #img = img * 2.0 - 1.0


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
        output_dir: str = "./mvdream_lora",
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
                "to_k", "to_q", "to_v", "to_out.0",  # Attention layers
                "proj_in", "proj_out",  # Projection layers
                "ff.net.0.proj", "ff.net.2"  # Feed-forward layers
            ],
            lora_dropout=self.lora_dropout,
        )
        
        # Apply LoRA to UNet
        self.pipeline.unet = get_peft_model(self.pipeline.unet, lora_config)
        self.pipeline.unet.print_trainable_parameters()
        
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

        device = self.accelerator.device
        self.pipeline.text_encoder.to(device)
        self.pipeline.vae.to(device)
        self.pipeline.image_encoder.to(device) 
        
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
    
    def compute_loss(self, batch):
        """Compute the training loss"""
        device = self.accelerator.device
        
        # Move batch to device
        images = batch['images'].to(device)  # [B, V, C, H, W]
        batch_size = images.shape[0]
        num_views = images.shape[1]
        
        # Get text embeddings (dummy for now)
        text_embeddings = self.get_text_embeddings(batch_size, device)
        
        # Encode images to latents
        target_latents = self.encode_images_to_latents(images)  # [B, V, C, H, W]
        target_latents_flat = target_latents.view(-1, *target_latents.shape[2:])  # [B*V, C, H, W]
        
        # Sample noise and timesteps
        noise = torch.randn_like(target_latents_flat)
        timesteps = torch.randint(
            0, 
            self.pipeline.scheduler.config.num_train_timesteps,
            (batch_size * num_views,),
            device=device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.pipeline.scheduler.add_noise(target_latents_flat, noise, timesteps)
        
        # Get camera parameters
        if 'camera_poses' in batch and batch['camera_poses'] is not None:
            camera = batch['camera_poses'].to(device, dtype=noisy_latents.dtype)
            camera_flat = camera.view(-1, camera.shape[-1])  # [B*V, camera_dim]
            print("USING EXISTING CAMERA POSE")
        else:
            # Fallback to generated cameras
            print("GENERATE GT CAMERA POSES")
            camera_batch = []
            for _ in range(batch_size):
                #camera = get_camera(num_views, elevation=0.0, extra_view=False) was incorrect azimuth
                all_cameras = get_camera(8, elevation=0.0, extra_view=False, blender_coord=False) # we have 8 images in total
                selected_indices = [0, 2, 4, 6]  # Match multiview dataset uniform selection
                camera_batch = []
                for _ in range(batch_size):
                    selected_cameras = all_cameras[selected_indices]
                    camera_batch.append(selected_cameras)
                #camera_batch.append(camera)
            camera = torch.stack(camera_batch, dim=0).to(device, dtype=noisy_latents.dtype)
            camera_flat = camera.view(-1, camera.shape[-1])  # [B*V, camera_dim]
        
        # Prepare text embeddings for each view
        text_embeddings_expanded = text_embeddings.unsqueeze(1).repeat(1, num_views, 1, 1)  # [B, V, seq_len, hidden_dim]
        text_embeddings_flat = text_embeddings_expanded.view(-1, *text_embeddings.shape[1:])  # [B*V, seq_len, hidden_dim]
        
        # For ImageDream: use first view as conditioning image
        conditioning_images = images[:, 0]  # [B, C, H, W] - first view only
    
        # Encode conditioning image
        with torch.no_grad():
            # Get image embeddings (using CLIP image encoder)
            ip_embeddings = self.pipeline.encode_image(
                conditioning_images.cpu().numpy().transpose(0, 2, 3, 1),  # Convert to HWC format
                device, 
                1  # num_images_per_prompt
            )[1]  # Get positive embeddings
            
            # Get conditioning image latents --> Single img only
            ip_img_latents_list = []
            for i in range(batch_size):
                single_img = conditioning_images[i].cpu().numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
                latent = self.pipeline.encode_image_latents(single_img, device, 1)[1]
                ip_img_latents_list.append(latent)

            ip_img_latents = torch.cat(ip_img_latents_list, dim=0)  # [B, C, H, W]


        # Predict noise
        unet_inputs = {
            'x': noisy_latents,
            'timesteps': timesteps,
            'context': text_embeddings_flat,
            'num_frames': num_views,
            'camera': camera_flat,
            'ip': ip_embeddings.repeat_interleave(num_views, dim=0),
            'ip_img': ip_img_latents,  # [B, C, H, W]
        }
        
        predicted_noise = self.pipeline.unet(**unet_inputs)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
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
                project_name="mvdream-lora-finetune",
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
        global_step = 0
        self.pipeline.unet.train()
        
        for epoch in range(self.num_train_epochs):
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch+1}/{self.num_train_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.pipeline.unet):
                    loss = self.compute_loss(batch)
                    
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
                        "train_loss": loss.detach().item(),
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
                    if global_step % (200) == 0: 
                        self.generate_sample(global_step)
            
            progress_bar.close()
        
        # Final save
        self.save_checkpoint("final")
        
        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()

    def generate_sample(self, step, num_samples=1):
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
    
    # Handle camera poses
    camera_poses = None
    if batch[0]['camera_poses'] is not None:
        camera_poses = torch.stack([item['camera_poses'] for item in batch])
    
    return {
        'images': images,
        'classes': classes,
        'camera_poses': camera_poses,
        'num_views': images.shape[1],
        'batch_size': images.shape[0]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to EEG training data (.npy file)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained MVDream model")
    parser.add_argument("--output_dir", type=str, default="./mvdream_lora", help="Output directory")
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
    
    # For validation, you can use a subset or separate validation data
    # For now, we'll skip validation
    val_dataset = None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        collate_fn=multiview_collate_fn_grey,
        pin_memory=False,
    )
    
    val_dataloader = None
    
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