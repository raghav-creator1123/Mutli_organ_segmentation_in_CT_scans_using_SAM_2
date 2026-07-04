'''Take a volume from the nifti data that we have, apply the preprocessing functions from week 1, then slice the volume along the z axis and convert it to png format because that is the format expected by SAM 2 model.Load all the slices from a single folder into a numpy array'''

import os
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb

def write_frames_png(nifti_path, out_dir):
    """Convert every axial slice of a CT volume to a numbered PNG file on disk."""
    # 1. Load, window, and convert to RGB
    vol, _, _ = load_volume(nifti_path)
    vol_u8 = apply_hu_window(vol)
    
    # 2. Prepare output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Write each axial slice (assuming H, W, Z structure)
    num_slices = vol_u8.shape[2]
    for z in range(num_slices):
        slice_data = vol_u8[:, :, z]
        rgb_slice = to_rgb(slice_data)
        
        # Save as 5-digit zero-padded index
        img = Image.fromarray(rgb_slice)
        img.save(out_path / f"{z:05d}.png")
        
    return num_slices

def frames_to_arrays(frames_dir):
    """Load all PNG frames from a folder into a single uint8 numpy array."""
    frames_dir = Path(frames_dir)
    # 1. Get all PNGs and sort them to ensure 00000, 00001, ... order
    frame_files = sorted(list(frames_dir.glob("*.png")))
    
    # 2. Load and stack frames
    frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        frames.append(np.array(img))
    
    # 3. Return as (Z, H, W, 3)
    return np.stack(frames, axis=0)

'''After compiling the data, we now prepare the prompt for the SAM 2 model (for PVS-Prompt Visual Segmentation). For that we take the given organ id, and find the best label file, the one with maximum area of the organ in its segmentation mask. Then we prepare the prompt by putting a bounding box around the corresponding CT image'''

import numpy as np
from scipy import ndimage

def bbox_from_mask(mask2d, pad=4):
    """Return a SAM 2-format bounding box [x0, y0, x1, y1] around the foreground."""
    # 1. Find indices of all True/nonzero pixels
    rows, cols = np.where(mask2d > 0)
    
    if len(rows) == 0:
        return None
    
    # 2. Compute tight bounding box
    y0, y1 = rows.min(), rows.max() + 1
    x0, x1 = cols.min(), cols.max() + 1
    
    # 3. Expand by pad and clamp to image bounds
    H, W = mask2d.shape
    y0 = max(0, y0 - pad)
    y1 = min(H, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(W, x1 + pad)
    
    return np.array([x0, y0, x1, y1], dtype=np.float32)

def best_start_slice(label_vol, organ_id):
    """Return the axial index (z) with the largest organ cross-section."""
    # 1. Binarize to the target organ
    organ_mask = (label_vol == organ_id)
    
    # 2. Check if organ exists
    if not np.any(organ_mask):
        raise ValueError(f"Organ ID {organ_id} not found in the provided label volume.")
    
    # 3. Compute pixel count per slice (Z axis is index 2)
    # Summing across H (0) and W (1) dimensions
    pixel_counts = np.sum(organ_mask, axis=(0, 1))
    
    # 4. Return the index with the maximum count
    return int(np.argmax(pixel_counts))

'''Use this functions now, take the whole BTCV volume and labelled volume and make slices of it, then also create the bounding boxes for each slices
We create seperate datasets for each organ id.'''

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb
from src.data.prompts import bbox_from_mask

class BTCVSliceDataset(Dataset):
    def __init__(self, cases, organ_id, image_dir, label_dir, image_size=1024):
        self.samples = []
        self.image_size = image_size

        for case in cases:
            img_path = Path(image_dir) / f"{case}.nii"
            lbl_path = Path(label_dir) / f"{case.replace('img', 'label')}.nii"
            
            # Load volumes
            vol, _, _ = load_volume(str(img_path))
            lbl, _, _ = load_volume(str(lbl_path))
            
            # HU Windowing
            vol_u8 = apply_hu_window(vol)
            
            # Iterate over slices
            for z in range(vol.shape[2]):
                # Create boolean mask for the target organ
                mask_slice = (lbl[:, :, z] == organ_id).astype(np.uint8)
                
                # If organ is not in this slice, skip
                if not np.any(mask_slice):
                    continue
                
                # Prepare Image: HU-window -> RGB -> PIL Resize (Bilinear)
                img_slice = to_rgb(vol_u8[:, :, z])
                img_pil = Image.fromarray(img_slice).resize((image_size, image_size), Image.BILINEAR)
                
                # Prepare Mask: PIL Resize (Nearest to keep binary 0/1)
                mask_pil = Image.fromarray(mask_slice).resize((image_size, image_size), Image.NEAREST)
                mask_resized = np.array(mask_pil)
                
                # Compute bbox
                bbox = bbox_from_mask(mask_resized, pad=4)
                if bbox is None:
                    continue
                    
                self.samples.append((np.array(img_pil), mask_resized, bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask, bbox = self.samples[idx]
        
        # Convert to tensors, normalize image to [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        gt_tensor = torch.from_numpy(mask).float()
        box_tensor = torch.from_numpy(bbox).float()
        
        return img_tensor, gt_tensor, box_tensor

'''finally we have a diagnostic tool to evaluate our model's performance'''

import numpy as np
from pathlib import Path
import imageio.v2 as imageio

def save_overlay_gif(vol_u8, pred3d, out_path, gt3d=None):
    """Write an animated GIF overlaying segmentation masks onto the CT volume."""
    frames = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_slices = vol_u8.shape[2]
    
    for z in range(num_slices):
        # 1. Convert grayscale slice to RGB
        slice_gray = vol_u8[:, :, z]
        rgb = np.stack([slice_gray] * 3, axis=-1)
        
        # 2. Apply GT tint (Green) - rendered beneath
        if gt3d is not None:
            mask_gt = gt3d[:, :, z] > 0
            rgb[mask_gt] = (rgb[mask_gt] * 0.5 + np.array([60, 220, 60]) * 0.5).astype(np.uint8)
            
        # 3. Apply Prediction tint (Red)
        mask_pred = pred3d[:, :, z] > 0
        rgb[mask_pred] = (rgb[mask_pred] * 0.5 + np.array([220, 60, 60]) * 0.5).astype(np.uint8)
        
        frames.append(rgb)
        
    # 4. Write to GIF
    imageio.mimsave(out_path, frames, duration=80, loop=0)
