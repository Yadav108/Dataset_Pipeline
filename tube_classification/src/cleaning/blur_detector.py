import numpy as np
import cv2
from pathlib import Path
import shutil

from loguru import logger
from config.parser import get_config


class BlurDetector:
    """Detect and filter blurry images using Laplacian variance.
    
    Computes Laplacian variance to identify blurry frames and
    filter them out during dataset cleaning.
    """
    
    def __init__(self):
        """Initialize detector with config."""
        self.cfg = get_config()
    
    def is_blurry(self, image_path: Path) -> bool:
        """Check if image is blurry using Laplacian variance.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if blurry (variance < threshold), False otherwise
        """
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return True
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance (use float32 to reduce memory usage)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        variance = float(laplacian.var())
        
        return variance < self.cfg.pipeline.blur_threshold
    
    def filter_directory(self, src_dir: Path, dst_dir: Path) -> tuple[int, int]:
        """Filter blurry images from source directory to destination.
        
        Copies non-blurry PNG images from source to destination directory,
        skipping those detected as blurry.
        
        Args:
            src_dir: Source directory containing PNG images
            dst_dir: Destination directory for non-blurry images
            
        Returns:
            Tuple of (total_count, kept_count)
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        total_count = 0
        kept_count = 0
        
        # Process all PNG files
        for image_path in src_dir.glob("*.png"):
            total_count += 1
            
            if self.is_blurry(image_path):
                logger.debug(f"Blurry rejected: {image_path.name}")
                continue
            
            # Copy non-blurry image
            shutil.copy2(str(image_path), str(dst_dir / image_path.name))
            kept_count += 1
        
        logger.info(
            f"Blur filter: {kept_count}/{total_count} images kept in {src_dir.name}"
        )
        
        return (total_count, kept_count)
