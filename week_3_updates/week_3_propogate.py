import numpy as np
from PIL import Image

def propagate_bidirectional(predictor, state, start_z, bbox, target_hw=None):
    """Run SAM 2 bidirectional propagation and return a 3-D binary mask."""
    
    # Dictionary to store masks: {frame_idx: np.ndarray}
    masks_dict = {}

    # Step 1 — Inject the bounding-box prompt
    predictor.add_new_points_or_box(
        state, frame_idx=start_z, obj_id=1,
        box=bbox  # float32 [x0, y0, x1, y1]
    )

    # Helper function to process and store masks
    def process_and_store(frame_idx, out_obj_ids, out_mask_logits):
        # out_mask_logits shape: (1, 1, h, w)
        mask = out_mask_logits[0, 0]
        # Threshold at 0 and move to CPU
        mask_bool = (mask > 0).cpu().numpy()
        
        # Resize if target_hw is provided
        if target_hw is not None and mask_bool.shape != target_hw:
            img = Image.fromarray(mask_bool)
            img = img.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
            mask_bool = np.array(img)
        
        masks_dict[frame_idx] = mask_bool

    # Step 2 — Forward pass (start_z → last slice)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            state, start_frame_idx=start_z, reverse=False):
        process_and_store(frame_idx, obj_ids, masks)

    # Step 3 — Backward pass (start_z → first slice)
    # Note: The prompt slice (start_z) is already in the dict from the forward pass
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
            state, start_frame_idx=start_z, reverse=True):
        if frame_idx != start_z:
            process_and_store(frame_idx, obj_ids, masks)

    # Step 5 — Stack all per-slice bool arrays along axis=2
    # Sort by frame index to ensure correct Z ordering
    sorted_frames = sorted(masks_dict.keys())
    stacked_mask = np.stack([masks_dict[i] for i in sorted_frames], axis=2)

    return stacked_mask
