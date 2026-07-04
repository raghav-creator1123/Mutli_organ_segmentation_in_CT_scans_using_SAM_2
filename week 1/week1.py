import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_volume(path):
    """Load a NIfTI file and return the 3D volume, affine, and voxel spacing."""
    img = nib.load(path)
    # Reorient to standard axes
    img = nib.as_closest_canonical(img)
    # Get the numpy array
    vol = img.get_fdata().astype(np.float32)
    # Read voxel spacing
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    print(f"Loaded volume with spacing: {spacing}")
    
    return vol, img.affine, spacing


def apply_hu_window(vol, lo=-150, hi=250, as_uint8=True):
    """Clip a CT volume to a soft-tissue HU window and rescale."""
    # Clip
    v = np.clip(vol, lo, hi)
    # Rescale to [0, 1]
    v = (v - lo) / (hi - lo)
    
    if as_uint8:
        return (v * 255).astype(np.uint8)
    else:
        return v.astype(np.float32)


def to_rgb(slice2d_u8):
    """Convert a (H, W) uint8 greyscale slice to (H, W, 3)."""
    # Add channel dimension and repeat
    return np.repeat(slice2d_u8[..., None], 3, axis=2)


def resample_isotropic(vol, spacing, target=1.5, order=1):
    """Resample a volume to isotropic voxel spacing using scipy zoom."""
    # Compute zoom factors
    factors = [s / target for s in spacing]
    # Apply zoom
    resampled = zoom(vol, factors, order=order)
    
    return resampled, (target, target, target)
