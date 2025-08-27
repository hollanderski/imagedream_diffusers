import numpy as np
import os
from einops import rearrange
import torch
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List, Callable
from torch.utils.data import random_split, Subset
from torch.utils.data import TensorDataset
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torchvision.transforms as transforms
from einops import rearrange

from mv_unet import get_camera
from EEGNet_Embedding_version import EEGNet_Embedding




def sliding_window_data(data, labels, fs, window_length, overlap):
    """
    Segments data into epochs using a sliding window approach.

    Parameters:
    - data: np.array, shape (num_trials, num_channels, num_samples)
    - labels: np.array, shape (num_trials,)
    - fs: int, Sampling frequency (samples per second)
    - window_length: float, Length of each window in seconds
    - overlap: float, Overlap between consecutive windows (in range [0, 1])

    Returns:
    - segmented_data: np.array, shape (num_windows_total, num_channels, samples_per_window)
    - segmented_labels: np.array, shape (num_windows_total,)
    """
    samples_per_window = int(fs * window_length)  # Number of samples per window
    step_size = int(samples_per_window * (1 - overlap))  # Step size based on overlap

    num_trials, num_channels, num_samples = data.shape
    num_windows_per_trial = (num_samples - samples_per_window) // step_size + 1
    num_windows_total = num_trials * num_windows_per_trial

    # Initialize arrays for segmented data and labels
    segmented_data = np.zeros((num_windows_total, num_channels, samples_per_window))
    segmented_labels = np.zeros(num_windows_total)

    window_index = 0
    for trial in range(num_trials):
        for window in range(num_windows_per_trial):
            start_sample = window * step_size
            end_sample = start_sample + samples_per_window
            segmented_data[window_index, :, :] = data[trial, :, start_sample:end_sample]
            segmented_labels[window_index] = labels[trial]
            window_index += 1

    return segmented_data, segmented_labels 


# Keep the original 4-channel normalize for backward compatibility if needed
def normalize(img):
    """Ensure output always has 4 channels (RGBA) - DEPRECATED"""
    print(f"normalize input: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    
    # Convert to tensor first
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    
    # Handle different input formats
    if len(img.shape) == 3:
        if img.shape[-1] == 4:  # HWC with 4 channels (RGBA)
            img = rearrange(img, 'h w c -> c h w')  # -> CHW
        elif img.shape[-1] == 3:  # HWC with 3 channels (RGB)
            img = rearrange(img, 'h w c -> c h w')  # -> CHW
            # Add alpha channel (fully opaque)
            alpha = torch.ones_like(img[0:1])
            img = torch.cat([img, alpha], dim=0)  # Now 4 channels
        else:
            raise ValueError(f"Unexpected number of channels: {img.shape[-1]}")
    elif len(img.shape) == 2:  # Grayscale
        # Convert grayscale to RGBA
        print("GRAYSCALE !!")
        img = img.unsqueeze(0)  # Add channel dim
        img = img.repeat(3, 1, 1)  # RGB
        alpha = torch.ones_like(img[0:1])
        img = torch.cat([img, alpha], dim=0)  # RGBA
    
    # At this point img should be CHW with 4 channels
    assert img.shape[0] == 4, f"Expected 4 channels, got {img.shape[0]}"
    
    # Normalize to [-1, 1] only if needed
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


def channel_last(img):
    if len(img.shape) == 4:
        return rearrange(img, 'b c h w -> b h w c')
    elif len(img.shape) == 3:
        return rearrange(img, 'c h w -> h w c')
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

class Config:
    def __init__(self):
        self.img_size = 256  # Changed from 64 to 256 for ImageDream compatibility
        self.crop_ratio = 0.2

class MultiViewEEGDataset():
    """
    Multi-view EEG Dataset for 3D-aware diffusion models.
    
    For each 8s EEG signal, loads multiple camera views of the same object.
    This enables training 3D-aware diffusion models that can generate 
    consistent multi-view images from EEG signals.
    
    Args:
        data_dir: Path to directory containing the EEG data
        num_views: Number of camera views to load per sample (default: 4)
        view_selection: Strategy for selecting views ('sequential', 'uniform', 'random', 'fixed')
        image_transform: Transform to apply to images
        camera_embeddings: Whether to include camera pose embeddings
        preload_images: Whether to pre-load all images into memory
    """
    
    def __init__(
        self, 
        data_dir: Path, 
        num_views: int = 4,
        view_selection: str = 'uniform',  # 'sequential', 'uniform', 'random', 'fixed'
        fixed_views: List[int] = None,    # For 'fixed' selection
        image_transform: Callable = None,
        camera_embeddings: bool = True,
        preload_images: bool = False,
        max_views_available: int = 8,
    ):
        
        self.num_views = num_views
        self.view_selection = view_selection
        self.fixed_views = fixed_views or [0, 2, 4, 6]  # Default: front, left, back, right
        self.camera_embeddings = camera_embeddings
        self.preload_images = preload_images
        self.max_views_available = max_views_available
        
        # Load metadata
        df = pd.read_excel("G:/ninon_workspace/imagery2024/DATA/Image3DObjects_nobg/3Dimages_Windows_imac_ordered_final.ods") 
        self.image_paths = df['images'].values
        
        # Image transform
        config = Config()
        crop_pix = int(config.crop_ratio * config.img_size)

        # REVERTED: Use original normalize for 4-channel output for latent space
        self.img_transform_train = transforms.Compose([
            normalize,  # Back to original 4-channel normalize
            transforms.Resize((config.img_size, config.img_size)),
            random_crop(config.img_size - crop_pix, p=0.5),
            transforms.Resize((config.img_size, config.img_size)),
        ])

        self.img_transform_test = transforms.Compose([
            normalize,  # Back to original 4-channel normalize
            transforms.Resize((config.img_size, config.img_size)),
        ])

        # Use the test transform as default (or train if you want augmentation)
        self.image_transform = self.img_transform_train if image_transform is None else image_transform

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config  # Store config for use in other methods
        
        # Load EEG data
        data_path = Path(data_dir)
        if data_path.is_file():
            self.data = np.load(data_path, allow_pickle=True)
            base_dir = data_path.parent
            self.labels = np.load(base_dir / "group.npy", allow_pickle=True)
            self.trials = np.load(base_dir / "trials.npy", allow_pickle=True)
        elif data_path.is_dir():
            self.data = np.load(data_path / "data.npy", allow_pickle=True)
            self.labels = np.load(data_path / "group.npy", allow_pickle=True)
            self.trials = np.load(data_path / "trials.npy", allow_pickle=True)

        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # fs = 250  
        # window_length = 8  # 8 seconds
        # overlap = 0
        
        # self.data, self.labels = sliding_window_data(
        #     self.data, self.labels, fs, window_length, overlap
        # )
        # Convert to torch tensors and reshape for EEGNet
        self.data = torch.from_numpy(self.data).type(torch.float).unsqueeze(1).to(self.device)
        self.data = self.data.permute(0, 2, 3, 1)
        #self.data = torch.from_numpy(self.data).type(torch.float32)
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor).to(self.device)
        self.labels = self.labels - 1  # 0-indexed
        
        # Group labels for mapping
        self.group_labels = df['group_label'].values        
        self.group_to_indices = {label: np.where(self.group_labels == label)[0] for label in np.unique(self.group_labels)}
        
        print(len(self.data), len(self.labels), len(self.group_labels), len(self.trials))
        # Create camera embeddings if requested
        if self.camera_embeddings:
            self.camera_poses = self._create_camera_embeddings()
        
        # Pre-load images if requested
        if self.preload_images:
            print("Pre-loading all images... This may take a while.")
            self.images = self._preload_all_images()
            print(f"Loaded {len(self.images)} images into memory.")
    
    def _create_camera_embeddings(self):
        """Create camera embeddings compatible with MVDream format"""
        # Use the original MVDream get_camera function
        camera_matrices = get_camera(
            num_frames=self.max_views_available,
            elevation=0.0,  # Keep horizontal
            azimuth_start=0,
            azimuth_span=360,
            blender_coord=False, # !!True,
            extra_view=False
        )
        
        return camera_matrices.to(self.device)  # [8, 16] - flattened 4x4 matrices

    
    def _preload_all_images(self):
        """Pre-load all images into memory - FIXED for 3 channels"""
        images = {}
        
        for i, path in enumerate(self.image_paths):
            try:
                if os.path.exists(path):
                    image_raw = Image.open(path).convert("RGB")  # FIXED: RGB not RGBA
                    image = np.array(image_raw) / 255.0
                    images[i] = self.image_transform(image)
                    print("path = ", path)
                else:
                    print(f"Warning: Image not found: {path}")
                    # FIXED: Create dummy image with 3 channels and correct size
                    images[i] = torch.zeros(3, self.config.img_size, self.config.img_size)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # FIXED: Create dummy image with 3 channels and correct size
                images[i] = torch.zeros(3, self.config.img_size, self.config.img_size)
        
        return images
    
    def _get_view_indices(self, base_idx):
        """
        Get the indices for multiple views based on the selection strategy.
        
        Args:
            base_idx: Base image index for the object/trial
            
        Returns:
            List of absolute image indices for the selected views
        """
        print(f"    _get_view_indices called with base_idx={base_idx}")
        
        if self.view_selection == 'sequential':
            # Take consecutive views starting from base_idx
            indices = [base_idx + i for i in range(self.num_views)]
        
        elif self.view_selection == 'uniform':
            # Uniformly distribute views around the 8 available views
            step = self.max_views_available // self.num_views  # step = 8//4 = 2
            indices = [base_idx + i * step for i in range(self.num_views)]
        
        elif self.view_selection == 'fixed':
            # Use pre-defined fixed view offsets from base_idx
            indices = [base_idx + view for view in self.fixed_views[:self.num_views]]
        
        elif self.view_selection == 'random':
            # Randomly sample view offsets
            available_offsets = list(range(self.max_views_available))
            selected_offsets = np.random.choice(available_offsets, size=self.num_views, replace=False)
            indices = [base_idx + offset for offset in sorted(selected_offsets)]
        
        else:
            raise ValueError(f"Unknown view_selection: {self.view_selection}")
        
        print(f"    Returning indices: {indices}")
        return indices
    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''
        ===== Explanation =====  
        Paths are ordered in the .ods file by :
        * class group (banana = group 0, strawberry = group 1) with 6 class groups in total
        * object trial within a group (banana_1, banana_2, ..., banana_n) with 13 object trials in total 
        * view of rotation (front, left, ..., back, right) with 8 views in total, 1 view corresponding to 1s segment

        The trial label is structured as follows : (group,trial,repetition) concatenated
        ex : 576 means group 5 (face), trial 7 (face_7), 6th repetition

        In order to retrieve the correct image path, given a group and trial label we need  
        path at image_idx = group*13*8 + 8*trial     
        i.e. image_idx = group_label*max_trial*max_view + max_view*trial     
        with group 0..5,  trial 0..12, max_view = 8, max_trial = 13
        '''
        
        max_view = 8
        max_trial = 13
        group_label = self.labels[idx].item()
        trial_label = self.trials[idx].item()
        
        # Parse trial label 
        my_class = int(str(trial_label)[0])
        my_trial = int(str(trial_label)[1:-1]) - 1
        
        # Use the ORIGINAL calculation with group_label (not my_class)
        base_image_idx = group_label * max_trial * max_view + max_view * my_trial
        
        print(f"DEBUG idx={idx}: group_label={group_label}, trial_label={trial_label}")
        print(f"  Parsed: my_class={my_class}, my_trial={my_trial}")
        print(f"  Base image index: {group_label}*{max_trial}*{max_view} + {max_view}*{my_trial} = {base_image_idx}")
        
        # Get indices for multiple views
        view_indices = self._get_view_indices(base_image_idx)
        print(f"  View indices: {view_indices}")
        
        # Check what objects these indices point to
        for i, view_idx in enumerate(view_indices[:2]):  # Just check first 2
            if view_idx < len(self.image_paths):
                path = self.image_paths[view_idx]
                object_name = path.split('/')[-2] if '/' in path else "unknown"
                print(f"    View {i}: {object_name} ({path.split('/')[-1]})")
        
        # Load images for all views
        images = []
        camera_poses = []
        
        for view_idx in view_indices:
            # Ensure view_idx is within bounds
            if view_idx >= len(self.image_paths):
                print(f"Warning: View index {view_idx} out of bounds, using modulo")
                view_idx = view_idx % len(self.image_paths)
            
            # Load image
            if self.preload_images:
                image = self.images[view_idx]
            else:
                image_path = self.image_paths[view_idx]
                if os.path.exists(image_path):
                    image_raw = Image.open(image_path).convert("RGBA")  # REVERTED: Back to RGBA
                    image_np = np.array(image_raw) / 255.0
                    image = self.image_transform(image_np)
                else:
                    print(f"Warning: Image not found: {image_path}")
                    # REVERTED: Create dummy with 4 channels
                    image = torch.zeros(4, self.config.img_size, self.config.img_size)
            
            images.append(image)
            
            # Get camera pose for this view
            if self.camera_embeddings:
                view_offset = view_idx % self.max_views_available
                # Get the 16D flattened camera matrix for this view
                camera_poses.append(self.camera_poses[view_offset])  # [16] vector
                # DEBUG: Extract angle from camera matrix
                camera_matrix = self.camera_poses[view_offset].view(4, 4)
                # Camera position is in the translation part
                cam_pos = camera_matrix[:3, 3]
                # Calculate azimuth from x,z coordinates  
                azimuth_rad = torch.atan2(cam_pos[0], cam_pos[2])
                azimuth_deg = torch.rad2deg(azimuth_rad) % 360
                print(f"View {len(camera_poses)-1}: image_idx={view_idx}, view_offset={view_offset}, camera_azimuth={azimuth_deg:.1f}°")

        
        # Stack images and camera poses
        images = torch.stack(images)  # [num_views, C, H, W]
        
        if self.camera_embeddings:
            camera_poses = torch.stack(camera_poses)  # [num_views, embedding_dim]
        else:
            camera_poses = None

            
        print(f"Selected view indices: {view_indices}")
        print(f"Expected angles: {[(i % 8) * 45 for i in view_indices]}")
        
        sample = {
            'eeg': self.data[idx].float(),
            'images': images,
            'camera_poses': camera_poses,
            'class': group_label,
            'metadata': {
                'my_class': my_class,
                'my_trial': my_trial,
                'trial_label': trial_label,
                'view_indices': view_indices,
                'base_image_idx': base_image_idx
            }
        }
        
        return sample
    
    def debug_multiview_sample(self, idx=0):
        """Debug a multi-view sample"""
        print(f"=== MULTI-VIEW SAMPLE DEBUG (idx={idx}) ===")
        
        sample = self[idx]
        
        print(f"EEG shape: {sample['eeg'].shape}")
        print(f"Images shape: {sample['images'].shape}")
        print(f"Class: {sample['class']}")
        
        if sample['camera_poses'] is not None:
            print(f"Camera poses shape: {sample['camera_poses'].shape}")
            print(f"Camera poses:\n{sample['camera_poses']}")
        
        metadata = sample['metadata']
        print(f"Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"View selection strategy: {self.view_selection}")
        print(f"Number of views: {self.num_views}")
        
        # Check if all images are different
        images = sample['images']
        all_same = True
        for i in range(1, len(images)):
            if not torch.equal(images[0], images[i]):
                all_same = False
                break
        
        print(f"All images identical: {all_same}")
        if all_same:
            print("WARNING: All views are identical! Check view selection logic.")
        
        return sample
    
    def visualize_sample(self, idx=0, figsize=(15, 4)):
        """Visualize all views for a given EEG sample"""
        sample = self[idx]
        
        images = sample['images']  # [num_views, C, H, W]
        metadata = sample['metadata']
        
        fig, axes = plt.subplots(1, self.num_views, figsize=figsize)
        if self.num_views == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            # Convert from tensor to displayable format
            img = images[i].cpu().numpy()
            if img.shape[0] == 3:  # If CHW format
                img = np.transpose(img, (1, 2, 0))
            
            # Denormalize if needed (assumes -1 to 1 range)
            if img.min() < 0:
                img = (img + 1) / 2
            
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            
            # Add view information
            view_angle = (metadata['view_indices'][i] % self.max_views_available) * 45
            ax.set_title(f'View {i}: {view_angle}°')
            ax.axis('off')
        
        # Add overall title with EEG info
        plt.suptitle(f'EEG Sample {idx} | Class: {sample["class"]} | Trial: {metadata["trial_label"]}', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Print detailed info
        print(f"EEG shape: {sample['eeg'].shape}")
        print(f"Class: {sample['class']} | My_class: {metadata['my_class']} | Trial: {metadata['trial_label']}")
        print(f"View indices: {metadata['view_indices']}")

    def debug_image_loading(self, idx=0):
        """Debug raw image loading to find the issue"""
        sample = self[idx]
        metadata = sample['metadata']
        
        print(f"=== IMAGE LOADING DEBUG ===")
        print(f"Sample {idx} | Class: {sample['class']} | Trial: {metadata['trial_label']}")
        
        # Get the view indices
        view_indices = metadata['view_indices']
        print(f"View indices: {view_indices}")
        
        for i, view_idx in enumerate(view_indices):
            print(f"\nView {i} (index {view_idx}):")
            
            # Check if index is in bounds
            if view_idx >= len(self.image_paths):
                print(f"  ERROR: Index {view_idx} >= {len(self.image_paths)} (out of bounds)")
                continue
                
            image_path = self.image_paths[view_idx]
            print(f"  Path: {image_path}")
            print(f"  Path exists: {os.path.exists(image_path)}")
            
            if os.path.exists(image_path):
                try:
                    # Load raw image
                    image_raw = Image.open(image_path).convert("RGB")  # FIXED: RGB
                    image_np = np.array(image_raw)
                    print(f"  Raw image shape: {image_np.shape}")
                    print(f"  Raw image range: [{image_np.min()}, {image_np.max()}]")
                    
                    # Check normalized version
                    image_norm = image_np / 255.0
                    print(f"  Normalized range: [{image_norm.min():.3f}, {image_norm.max():.3f}]")
                    
                    # Check after transform
                    if self.preload_images:
                        transformed = self.images[view_idx]
                    else:
                        transformed = self.image_transform(image_norm)
                    
                    print(f"  Transformed shape: {transformed.shape}")
                    print(f"  Transformed range: [{transformed.min():.3f}, {transformed.max():.3f}]")
                    print(f"  Transformed mean: {transformed.mean():.3f}")
                    print(f"  All zeros: {torch.allclose(transformed, torch.zeros_like(transformed))}")
                    
                except Exception as e:
                    print(f"  ERROR loading image: {e}")
            else:
                print(f"  ERROR: File does not exist!")

# Helper function to create camera-aware data loader
def multiview_collate_fn(batch):
    """
    Custom collate function to handle multi-view data.
    Moved outside to avoid pickling issues with local functions.
    """
    # Extract components
    eegs = torch.stack([item['eeg'] for item in batch])
    images = torch.stack([item['images'] for item in batch])  # [batch, views, C, H, W]
    classes = torch.tensor([item['class'] for item in batch])
    
    # Reshape images for multi-view processing
    batch_size, num_views = images.shape[:2]
    images = images.view(batch_size * num_views, *images.shape[2:])  # [batch*views, C, H, W]
    
    # Handle camera poses
    camera_poses = None
    if batch[0]['camera_poses'] is not None:
        camera_poses = torch.stack([item['camera_poses'] for item in batch])
        camera_poses = camera_poses.view(batch_size * num_views, -1)  # [batch*views, embed_dim]
    
    return {
        'eeg': eegs,  # [batch_size, ...]
        'images': images,  # [batch_size * num_views, C, H, W]
        'camera_poses': camera_poses,  # [batch_size * num_views, embed_dim] or None
        'classes': classes,  # [batch_size]
        'num_views': num_views,
        'batch_size': batch_size
    }

def create_multiview_dataloader(
    data_dir,
    batch_size=4,
    num_views=4,
    view_selection='uniform',
    num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
    shuffle=True,
    pin_memory=False
):
    """Create a DataLoader for multi-view EEG training."""
    
    dataset = MultiViewEEGDataset(
        data_dir=data_dir,
        num_views=num_views,
        view_selection=view_selection,
        preload_images=False, #True
    )

    # eeg_features_dim :  need to instantiate?
    sample_eeg = dataset[0]['eeg']  # Get one EEG sample
    eeg_features = EEGNet_Embedding(sample_eeg.unsqueeze(0))
    print(f"EEG features shape: {eeg_features.shape}")  

    # checkpoint_path =   "G:/ninon_workspace/imagery2024/P12_61ch_8s_61test.ckpt"
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     hyper_parameters = checkpoint['hyper_parameters']
    #     fs = 512
    #     window_length = hyper_parameters['window_time']
    #     train_overlap = 0
    #     val_overlap = 0
    #     batch_size = hyper_parameters['batch_size']

    #     self.eeg_encoder = EEGNet_Embedding( 
    #         in_chans=61,
    #         n_classes = 6,
    #         input_window_samples=fs*window_length,  
    #         F1=hyper_parameters["F1"],  
    #         F2=hyper_parameters["F1"] * hyper_parameters["D"],  
    #         D=hyper_parameters["D"],   
    #         kernel_length=hyper_parameters["kernel_length"],  
    #         depthwise_kernel_length=hyper_parameters["depthwise_kernel_length"],  
    #         lr=hyper_parameters["lr"],
    #         epochs=hyper_parameters["epochs"],
    #         weight_decay=hyper_parameters["weight_decay"],
    #         drop_prob=hyper_parameters["drop_prob"],
    #         pool_mode=hyper_parameters["pool_mode"], 
    #         separable_kernel_length=hyper_parameters["separable_kernel_length"],
    #         momentum=hyper_parameters["bn_momentum"],
    #         activation=hyper_parameters["activation"], 
    #         final_conv_length="auto", 
    #     )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Use 0 for Windows to avoid multiprocessing
        collate_fn=multiview_collate_fn,
        pin_memory=pin_memory#False #True if torch.cuda.is_available() else False
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the multi-view dataset
    data_dir = "G:/ninon_workspace/imagery2024/2D_Reconstruction/Generation_2D/Segmented_data_V2D/P19/train/data.npy"
    
    # Create dataset
    dataset = MultiViewEEGDataset(
        data_dir=data_dir,
        num_views=4,
        view_selection='uniform',
        preload_images=False  # Set to True if you have enough memory
    )
    
    # Debug a sample
    sample = dataset.debug_multiview_sample(idx=0)
    
    # Create data loader
    dataloader = create_multiview_dataloader(
        data_dir=data_dir,
        batch_size=2,
        num_views=4
    )
    
    # Test visualization
    print("Testing visualization...")
    dataset.visualize_sample(idx=1)
    dataset.visualize_sample(idx=20) 
    dataset.visualize_sample(idx=100)
    # Test data loader
    for batch in dataloader:
        print("Batch shapes:")
        print(f"  EEG: {batch['eeg'].shape}")
        print(f"  Images: {batch['images'].shape}")
        print(f"  Classes: {batch['classes'].shape}")
        if batch['camera_poses'] is not None:
            print(f"  Camera poses: {batch['camera_poses'].shape}")
        print(f"  Num views: {batch['num_views']}")
        print(f"  Batch size: {batch['batch_size']}")
        break
