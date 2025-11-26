import torch
import numpy as np
from collections import defaultdict
from pipeline_mvdream import MVDreamPipeline
from EEGNet_Embedding_version import EEGNet_Embedding
from mv_unet import get_camera
import lpips

# Replace with your dataset import
from multiview_dataset import MultiViewEEGDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- CONFIGURATION ---------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_PATH = "G:/ninon_workspace/imagedream_diffusers/perception2_lr4_eeg_mvdream/checkpoint-2500"
MODEL_PATH = "./weights_mvdream"
EEG_ENCODER_PATH = "eegnet_best_supcon_003.pt"

# Path to data split you want to evaluate (val or test)
DATA_NPY = "G:/ninon_workspace/imagery2024/2D_Reconstruction/Generation_2D/Segmented_data_V2D/P12/test/data.npy"

NUM_CLASSES = 6
NUM_VIEWS = 4
#BATCH_SIZE = 2

# --------- LOAD PIPELINE AND MODELS ---------

pipeline = MVDreamPipeline.from_pretrained(MODEL_PATH)

# Restore LoRA adapter (assuming you used PEFT/LoRA)
from peft import get_peft_model, LoraConfig
lora_config = LoraConfig(
    r=16, lora_alpha=32,  # Use same params as training
    target_modules=[
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2"
    ],
    lora_dropout=0.1,
)
pipeline.unet = get_peft_model(pipeline.unet, lora_config)

pipeline.unet.load_adapter(CKPT_PATH, adapter_name="default", is_trainable=False)
pipeline.to(DEVICE)
pipeline.unet.eval()

# EEG Encoder
checkpoint_path = "G:/ninon_workspace/imagery2024/Classification/explainability/checkpoints/object/P12-new.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
hyper_parameters = checkpoint['hyper_parameters']
fs = 512
window_length = hyper_parameters['window_time']
batch_size = hyper_parameters['batch_size']

BATCH_SIZE = batch_size

eeg_encoder = EEGNet_Embedding(
        in_chans=61,
        n_classes=6,
        input_window_samples=fs*window_length,
        lr=hyper_parameters["lr"],
        epochs=hyper_parameters["epochs"],
        weight_decay=hyper_parameters["weight_decay"],
        drop_prob=hyper_parameters["drop_prob"],
        pool_mode=hyper_parameters["pool_mode"],
        F1=hyper_parameters["F1"],
        F2=None,
        D=hyper_parameters["D"],
        kernel_length=hyper_parameters["kernel_length"],
        depthwise_kernel_length=hyper_parameters["depthwise_kernel_length"], 
        separable_kernel_length=hyper_parameters["separable_kernel_length"],
        momentum=hyper_parameters["bn_momentum"],
        activation=hyper_parameters["activation"] 
).to(device)
eeg_encoder.load_state_dict(torch.load(EEG_ENCODER_PATH))
eeg_encoder.eval().to(DEVICE)

eeg_projector = torch.nn.Linear(512, 1024).to(DEVICE)  

# LPIPS metric
lpips_model = lpips.LPIPS(net='alex').to(DEVICE).eval()

# --------- LOAD DATASET ---------


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



dataset = MultiViewEEGDataset(
    data_dir=DATA_NPY,
    num_views=NUM_VIEWS,
    view_selection='uniform',
    preload_images=False
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=multiview_collate_fn_grey,
)

# --------- EVALUATION LOOP ---------
from skimage.metrics import structural_similarity as ssim

per_class_metrics = defaultdict(list)

with torch.no_grad():
    for batch in dataloader:
        eegs = batch['eeg'].to(DEVICE)
        labels = batch['classes'].cpu().numpy()
        images = batch['images'].to(DEVICE)  # [B, V, C, H, W]

        # EEG embeddings and context
        eeg_features = eeg_encoder(eegs, return_embedding=True)
        eeg_embeddings = eeg_projector(eeg_features)
        eeg_context = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1)

        num_views = images.shape[1]
    for i in range(eegs.shape[0]):  # Loop over batch
        label = labels[i]
        gt_images = images[i]  # shape: [num_views, C, H, W]

        # -- Generate 8 views for this sample, using your pipeline --
        eeg_signal = eegs[i]
        device = next(pipeline.unet.parameters()).device
        eeg_features = eeg_encoder(eeg_signal.unsqueeze(0).to(device), return_embedding=True)
        eeg_embeddings = eeg_projector(eeg_features)
        eeg_context = eeg_embeddings.unsqueeze(1).repeat(1, 77, 1)

        # Prepare input latents
        height, width, num_frames = 256, 256, num_views
        latents = torch.randn(num_frames, 4, height//8, width//8, device=device, dtype=eeg_context.dtype)
        pipeline.scheduler.set_timesteps(20, device=device)

        for t in pipeline.scheduler.timesteps:
            camera = get_camera(num_views, elevation=0.0, extra_view=False, blender_coord=False)
            camera = camera.to(device, dtype=latents.dtype)
            noise_pred = pipeline.unet(
                x=latents,
                timesteps=torch.tensor([t] * num_frames, device=device),
                context=eeg_context.repeat(num_frames, 1, 1),
                num_frames=num_frames,
                camera=camera
            )
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode all generated images (shape: [num_views, C, H, W])
        generated_images = pipeline.decode_latents(latents)

        # -- Metrics over 8 views --
        ssim_scores = []
        lpips_scores = []
        for v in range(num_views):
            # GT and GEN images: [C,H,W]
            gt_img = gt_images[v]
            gen_img = generated_images[v]
            gt_img_np = gt_img.cpu().numpy().transpose(1,2,0)
            gen_img_np = gen_img #.transpose(1,2,0)

            print("gen_img type/shape:", type(gen_img), getattr(gen_img, 'shape', None))
            print("channels ", gt_img_np.shape[2], gen_img_np.shape[2])
            # Remove alpha for LPIPS
            if gt_img_np.shape[2] > 3:
                gt_img_np = gt_img_np[:,:,:3]
            if gen_img_np.shape[2] > 3:
                gen_img_np = gen_img_np[:,:,:3]

            # Scale from [-1, 1] to [0, 1]
            gt_img_np = (gt_img_np + 1) / 2
            gen_img_np = (gen_img_np + 1) / 2

            # SSIM as grayscale (or RGB for multi-channel, but grayscale is standard)
            gt_img_gray = gt_img_np.mean(axis=2)
            gen_img_gray = gen_img_np.mean(axis=2)
            ssim_val = ssim(gt_img_gray, gen_img_gray, data_range=1.0)

            gt_tensor = torch.tensor(gt_img_np).permute(2,0,1).unsqueeze(0).to(DEVICE).float()
            gen_tensor = torch.tensor(gen_img_np).permute(2,0,1).unsqueeze(0).to(DEVICE).float()
            lpips_val = lpips_model(gt_tensor, gen_tensor).item()

            ssim_scores.append(ssim_val)
            lpips_scores.append(lpips_val)

        # Average over 8 views for this sample
        avg_ssim = np.mean(ssim_scores)
        avg_lpips = np.mean(lpips_scores)

        per_class_metrics[label].append({'ssim': avg_ssim, 'lpips': avg_lpips})


# --------- PRINT METRIC TABLE ---------
print("\nPer-class metrics (computed on split):")
for cls in range(NUM_CLASSES):
    scores = per_class_metrics[cls]
    if scores:
        ssim_scores = [x['ssim'] for x in scores]
        lpips_scores = [x['lpips'] for x in scores]
        print(f"Class {cls}: SSIM {np.mean(ssim_scores):.3f} (n={len(ssim_scores)}), LPIPS {np.mean(lpips_scores):.3f}")
    else:
        print(f"Class {cls}: No samples.")

