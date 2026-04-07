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
            
            coverage_ratio = metadata["coverage_ratio"]
            
            return coverage_ratio < self.cfg.pipeline.min_coverage_ratio
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
                f"{image_id}_depth.png",
                f"{image_id}_depth.npy",
            ]
            for filename in raw_files:
                src_file = raw_dir / filename
                if src_file.exists():
                    shutil.copy2(str(src_file), str(dst_raw_dir / filename))
            
            # Copy annotation files
            ann_files = [
                f"{image_id}_bbox.json",
                f"{image_id}_mask.png",
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
