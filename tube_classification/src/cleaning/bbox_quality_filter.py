import json
from pathlib import Path
import shutil

from loguru import logger
from config.parser import get_config


class BBoxQualityFilter:
    """Filter annotations by bbox quality using coverage_ratio threshold.
    
    Reads metadata JSON files and rejects images where the mask-to-bbox
    coverage ratio falls below the configured minimum.
    """
    
    def __init__(self):
        """Initialize filter with config."""
        self.cfg = get_config()

    @staticmethod
    def _log_reject(reason: str, value: float, threshold: float) -> None:
        logger.debug(
            f"[bbox_quality] REJECT reason={reason} "
            f"value={value:.3f} threshold={threshold:.3f}"
        )
    
    def is_low_quality(self, metadata_path: Path) -> bool:
        """Check if annotation bbox quality is below threshold.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            True if coverage_ratio < min_coverage_ratio, False otherwise
        """
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            bbox = metadata.get("bbox", {})
            w = float(bbox.get("w", 0.0))
            h = float(bbox.get("h", 0.0))
            if w <= 0 or h <= 0:
                self._log_reject("bbox_dim_invalid", min(w, h), 1.0)
                return True

            # Relaxed minimum bbox dimensions for close-range small tubes.
            min_w = 40.0
            min_h = 24.0
            if w < min_w:
                self._log_reject("bbox_width_min", w, min_w)
                return True
            if h < min_h:
                self._log_reject("bbox_height_min", h, min_h)
                return True

            # Accept IoU >= 0.60 for SAM-quality gate.
            sam_iou = metadata.get("sam_iou_score")
            min_iou = 0.60
            if sam_iou is not None and float(sam_iou) < min_iou:
                self._log_reject("sam_iou_min", float(sam_iou), min_iou)
                return True

            capture_mode = str(metadata.get("capture_mode", "")).lower()
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

            # single_side tubes may be landscape (wider than tall); only reject
            # extreme outliers. Top mode remains stricter and near-circular.
            if capture_mode in {"single_side", ""}:
                max_aspect_ratio = 4.0
                if aspect_ratio > max_aspect_ratio:
                    self._log_reject("aspect_ratio_max_single_side", aspect_ratio, max_aspect_ratio)
                    return True
            else:
                max_aspect_ratio = self.cfg.pipeline.top_max_circularity_ratio
                if aspect_ratio > max_aspect_ratio:
                    self._log_reject("aspect_ratio_max_top", aspect_ratio, float(max_aspect_ratio))
                    return True

            coverage_ratio = float(metadata.get("coverage_ratio", 0.0))
            base_cov_min = float(self.cfg.pipeline.min_coverage_ratio)
            bbox_area = w * h

            # Close-range tubes occupy less bbox area; apply relaxed floor.
            if bbox_area <= 8000:
                coverage_min = min(base_cov_min, 0.24)
            elif bbox_area <= 12000:
                coverage_min = min(base_cov_min, 0.30)
            else:
                coverage_min = base_cov_min

            if coverage_ratio < coverage_min:
                self._log_reject("coverage_ratio_min", coverage_ratio, coverage_min)
                return True

            return False
        except Exception:
            logger.warning(f"Could not read metadata: {metadata_path.name}")
            return True
    
    def filter_directory(
        self,
        raw_dir: Path,
        ann_dir: Path,
        dst_raw_dir: Path,
        dst_ann_dir: Path,
    ) -> tuple[int, int]:
        """Filter images by bbox quality and copy passing images.
        
        Reads metadata files, identifies low-quality annotations, and
        copies only passing images and their annotations to destination
        directories.
        
        Args:
            raw_dir: Source directory with raw images
            ann_dir: Source directory with annotations
            dst_raw_dir: Destination directory for raw images
            dst_ann_dir: Destination directory for annotations
            
        Returns:
            Tuple of (total_count, kept_count)
        """
        # Create destination directories
        dst_raw_dir.mkdir(parents=True, exist_ok=True)
        dst_ann_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all metadata files
        metadata_files = list(ann_dir.glob("*_metadata.json"))
        total = len(metadata_files)
        kept = 0
        
        for metadata_file in metadata_files:
            # Extract image_id from filename
            # e.g. "img_001_metadata.json" → "img_001"
            image_id = metadata_file.stem.replace("_metadata", "")
            
            # Check quality
            if self.is_low_quality(metadata_file):
                logger.debug(f"Low bbox quality rejected: {image_id}")
                continue
            
            # Copy raw files
            raw_files = [
                f"{image_id}_rgb.png",
                f"{image_id}_rgb_crop.png",
                f"{image_id}_depth.png",
                f"{image_id}_depth_color.png",
                f"{image_id}_depth.npy",
                f"{image_id}_depth_crop.npy",
                f"{image_id}_depth_crop_color.png",
                f"{image_id}_depth_crop16.png",
            ]
            for filename in raw_files:
                src_file = raw_dir / filename
                if src_file.exists():
                    shutil.copy2(str(src_file), str(dst_raw_dir / filename))
            
            # Copy annotation files
            ann_files = [
                f"{image_id}_bbox.json",
                f"{image_id}_mask.png",
                f"{image_id}_mask_crop.png",
                f"{image_id}_metadata.json",
            ]
            for filename in ann_files:
                src_file = ann_dir / filename
                if src_file.exists():
                    shutil.copy2(str(src_file), str(dst_ann_dir / filename))
            
            kept += 1
        
        logger.info(
            f"BBox quality filter: {kept}/{total} images kept in {raw_dir.name}"
        )
        
        return (total, kept)
