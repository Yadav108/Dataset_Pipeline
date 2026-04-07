from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from rembg import remove


class BackgroundRemover:
    """Remove image backgrounds using rembg.
    
    Processes images to remove backgrounds, running as a final
    cleaning step with lowest priority.
    """
    
    def __init__(self):
        """Initialize background remover."""
        pass
    
    def remove_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Remove background from a numpy BGR array (ROI crop).
        
        Args:
            image_array: BGR numpy array (output from cv2)
            
        Returns:
            RGBA numpy array with background removed. On error, returns original.
        """
        try:
            # Convert BGR to RGB for rembg
            rgb_array = np.flip(image_array, axis=2)  # BGR → RGB
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_array, 'RGB')
            
            # Remove background (returns RGBA PIL Image)
            output_pil = remove(pil_image)
            
            # Convert back to numpy array (RGBA)
            output_array = np.array(output_pil)
            
            return output_array
        except Exception as e:
            logger.warning(f"Background removal failed: {e}. Returning original image.")
            # Return original image with alpha channel added
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                alpha = np.full((image_array.shape[0], image_array.shape[1], 1), 255, dtype=np.uint8)
                return np.concatenate([image_array, alpha], axis=2)
            return image_array
    
    def remove_background(self, image_path: Path, output_path: Path) -> None:
        """Remove background from a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
        """
        # Open input image
        input_image = Image.open(image_path)
        
        # Remove background
        output_image = remove(input_image)
        
        # Save output as PNG
        output_image.save(output_path, "PNG")
        
        logger.debug(f"Background removed: {image_path.name}")
    
    def process_directory(self, src_dir: Path, dst_dir: Path) -> tuple[int, int]:
        """Process all images in directory to remove backgrounds.
        
        Args:
            src_dir: Source directory with PNG images
            dst_dir: Destination directory for processed images
            
        Returns:
            Tuple of (total_count, total_count)
        """
        # Create destination directory
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all PNG files
        total = 0
        for image_path in src_dir.glob("*.png"):
            output_path = dst_dir / image_path.name
            self.remove_background(image_path, output_path)
            total += 1
        
        logger.info(
            f"Background removal: {total} images processed in {src_dir.name}"
        )
        
        return (total, total)
