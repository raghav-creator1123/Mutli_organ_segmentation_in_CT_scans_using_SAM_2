import nibabel as nib
import numpy as np
from PIL import Image

def load_volume(filepath):
    """
    Loads a NIFTI volume, aligns it to anatomical standard, and extracts the array.
    """
    print(f"Loading volume from: {filepath}")
    img = nib.load(filepath)
    
    # CRITICAL STEP: Use the Affine matrix to fix the orientation.
    # This prevents the slices from being sideways, upside down, or swapped!
    canonical_img = nib.as_closest_canonical(img)
    
    # Extract the raw 3D NumPy array
    data_array = canonical_img.get_fdata()
    
    # Get the physical voxel dimensions (in millimeters)
    zooms = canonical_img.header.get_zooms()
    print(f"-> Voxel Zooms (x, y, z in mm): {zooms}")
    
    return data_array

def apply_hu_window(data, window_min=-150, window_max=250):
    """
    Clips CT data to a specific Hounsfield Unit window and rescales to 0-255 (Grayscale).
    """
    print(f"Applying HU Window: {window_min} to {window_max}")
    
    # Step 1: Clip the values. 
    # Anything below -150 becomes -150. Anything above 250 becomes 250.
    clipped_data = np.clip(data, window_min, window_max)
    
    # Step 2: Normalize the data to a 0.0 to 1.0 scale
    normalized_data = (clipped_data - window_min) / (window_max - window_min)
    
    # Step 3: Rescale to 0 to 255 for standard image saving
    rescaled_data = (normalized_data * 255).astype(np.uint8)
    
    return rescaled_data

def test_pipeline(filepath, output_filename="middle_slice.png"):
    """
    Runs the full pipeline and saves the middle axial slice.
    """
    # 1. Load the data
    data = load_volume(filepath)
    
    # 2. Apply the Soft Tissue Window
    windowed_data = apply_hu_window(data)
    
    # 3. Find the middle slice on the Z-axis
    # Because we used as_closest_canonical, the Z-axis is guaranteed to be Axial (top-down)
    z_middle_index = windowed_data.shape[2] // 2
    
    # Extract that 2D slice
    middle_slice_2d = windowed_data[:, :, z_middle_index]
    
    # Note: NiBabel arrays are often stored transposed relative to how image libraries expect them.
    # Rotating it 90 degrees ensures the patient's nose points "up" in the final image.
    middle_slice_2d = np.rot90(middle_slice_2d)
    
    # 4. Save to PNG using Pillow (PIL)
    img = Image.fromarray(middle_slice_2d)
    img.save(output_filename)
    print(f"-> Success! Saved test slice to {output_filename}")

if __name__ == "__main__":
    # Replace this with the path to your actual CT scan .nii or .nii.gz file
    # test_pipeline("path_to_your_ct_scan.nii.gz")
    
    print("Pipeline ready to run.")
