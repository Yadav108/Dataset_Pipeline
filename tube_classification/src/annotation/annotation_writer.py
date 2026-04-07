import numpy as np
import cv2
import json
from pathlib import Path

from PIL import Image
from loguru import logger
from config.parser import get_config


class AnnotationWriter:
    """Write annotation files following dataset storage schema.
    
    Writes RGB, depth, mask, bbox, and metadata files to organized
    directory structure per class and session.
    """
    
    def __init__(self):
        """Initialize writer with config."""
        self.cfg = get_config()
    
    def write(
        self,
        image_id: str,
        class_id: str,
        session_id: str,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        metadata: dict,
    ) -> None:
        """Write all annotation files for a capture.
        
        Writes 6 files: RGB PNG, depth PNG (scaled), depth NPY, mask PNG,
        bbox JSON, and metadata JSON to organized directories.
        
        Args:
            image_id: Unique image identifier
            class_id: Tube class identifier
            session_id: Capture session identifier
            rgb_frame: RGB image array
            depth_frame: Raw depth frame (uint16)
            mask: Binary segmentation mask
            bbox: Bounding box as (x, y, width, height)
            metadata: Annotation metadata dictionary
        """
        # Build directory paths
        root = Path(self.cfg.storage.root_dir)
        raw_dir = root / "raw" / class_id / session_id
        ann_dir = root / "annotations" / class_id / session_id
        
        # Create directories
        raw_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write RGB PNG
        rgb_path = raw_dir / f"{image_id}_rgb.png"
        cv2.imwrite(str(rgb_path), rgb_frame)
        
        # 2. Write depth PNG (scaled to uint8)
        depth_scaled = (depth_frame / depth_frame.max() * 255).astype(np.uint8)
        depth_png_path = raw_dir / f"{image_id}_depth.png"
        cv2.imwrite(str(depth_png_path), depth_scaled)
        
        # 3. Write depth NPY (raw)
        depth_npy_path = raw_dir / f"{image_id}_depth.npy"
        np.save(str(depth_npy_path), depth_frame)
        
        # 4. Write mask PNG
        mask_path = ann_dir / f"{image_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # 5. Write bbox JSON
        bbox_dict = {
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
        }
        bbox_path = ann_dir / f"{image_id}_bbox.json"
        with open(bbox_path, "w") as f:
            json.dump(bbox_dict, f)
        
        # 6. Write metadata JSON
        metadata_path = ann_dir / f"{image_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(
            f"Annotation written: {image_id} → class={class_id} session={session_id}"
        )
