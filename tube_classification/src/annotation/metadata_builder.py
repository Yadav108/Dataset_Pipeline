import numpy as np
import datetime

from loguru import logger


def build_metadata(
    image_id: str,
    class_id: str,
    volume_ml: float,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    rgb_shape: tuple[int, int, int],
    sam_iou_score: float | None = None,
    calibration_metadata: dict | None = None,
) -> dict:
    """Build structured annotation metadata from capture data.
    
    Computes mask statistics and packages all capture information
    into a standardized dictionary structure.
    
    Args:
        image_id: Unique image identifier
        class_id: Tube class identifier
        volume_ml: Volume in milliliters
        bbox: Bounding box as (x, y, width, height)
        mask: Binary segmentation mask
        rgb_shape: RGB image shape as (height, width, channels)
        sam_iou_score: MobileSAM predicted IoU score (0.0-1.0), optional
        calibration_metadata: Distance calibration info, optional
        
    Returns:
        Dictionary with structured annotation metadata
    """
    # Compute mask statistics
    mask_area = int(np.sum(mask > 0))
    x, y, w, h = bbox
    bbox_area = w * h
    coverage_ratio = round(mask_area / bbox_area, 4) if bbox_area > 0 else 0.0
    
    metadata = {
        "image_id": image_id,
        "class_id": class_id,
        "volume_ml": volume_ml,
        "timestamp": datetime.datetime.now().isoformat(),
        "image_shape": {
            "height": rgb_shape[0],
            "width": rgb_shape[1],
            "channels": rgb_shape[2],
        },
        "bbox": {
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
        },
        "mask_area_px": mask_area,
        "bbox_area_px": bbox_area,
        "coverage_ratio": coverage_ratio,
    }
    
    # Add SAM IoU score if provided
    if sam_iou_score is not None:
        metadata["sam_iou_score"] = round(sam_iou_score, 4)
    
    # Add distance calibration info if provided
    if calibration_metadata:
        metadata.update(calibration_metadata)
    
    return metadata
