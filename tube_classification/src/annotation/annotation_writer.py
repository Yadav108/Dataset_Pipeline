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

    @staticmethod
    def _scale_depth_for_preview(
        depth_frame: np.ndarray,
        image_id: str,
        invert: bool = False,
    ) -> np.ndarray:
        """Build an 8-bit depth preview using in-place math to limit memory spikes."""
        depth_scaled = np.zeros(depth_frame.shape, dtype=np.uint8)
        valid_mask = depth_frame > 0
        if not np.any(valid_mask):
            logger.warning(f"Depth frame has 0 valid pixels: {image_id}")
            return depth_scaled

        valid_depth = depth_frame[valid_mask].astype(np.float32, copy=True)
        low = float(np.percentile(valid_depth, 2))
        high = float(np.percentile(valid_depth, 98))
        if high <= low:
            high = low + 1.0

        np.subtract(valid_depth, low, out=valid_depth)
        valid_depth *= 255.0 / (high - low)
        np.clip(valid_depth, 0.0, 255.0, out=valid_depth)
        depth_scaled[valid_mask] = valid_depth.astype(np.uint8, copy=False)
        if invert:
            cv2.bitwise_not(depth_scaled, dst=depth_scaled)
        return depth_scaled
    
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
        
        # 2. Write depth PNG visualization (valid-pixel scaling, robust for sparse depth)
        depth_scaled = self._scale_depth_for_preview(
            depth_frame,
            image_id,
            invert=bool(getattr(self.cfg.pipeline, "depth_preview_invert", True)),
        )
        depth_png_path = raw_dir / f"{image_id}_depth.png"
        cv2.imwrite(str(depth_png_path), depth_scaled)
        depth16_png_path = raw_dir / f"{image_id}_depth16.png"
        Image.fromarray(depth_frame.astype(np.uint16), mode="I;16").save(str(depth16_png_path))
        
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
