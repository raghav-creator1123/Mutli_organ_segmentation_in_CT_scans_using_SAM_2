from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_predictor(model_cfg, ckpt_path):
    """Build a SAM 2 video predictor for bidirectional z-axis propagation."""
    return build_sam2_video_predictor(model_cfg, ckpt_path, device="cuda")


def build_image_predictor(model_cfg, ckpt_path):
    """Build a SAM 2 image predictor for single-frame prompted segmentation."""
    sam_model = build_sam2(model_cfg, ckpt_path, device="cuda")
    return SAM2ImagePredictor(sam_model)


def init_state(predictor, frames_dir):
    """Initialise the SAM 2 video memory state for a frame folder."""
    return predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,   # mandatory on 8 GB VRAM
        offload_state_to_cpu=True,   # mandatory on 8 GB VRAM
    )
