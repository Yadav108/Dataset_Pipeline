from pathlib import Path

from loguru import logger
import imagehash
from PIL import Image
from config.parser import get_config


class DuplicateRemover:
    """Detect and remove duplicate images using perceptual hashing.
    
    Computes perceptual hashes (phash) to identify and remove
    near-duplicate images from a directory.
    """
    
    def __init__(self):
        """Initialize remover with config."""
        self.cfg = get_config()
    
    def remove_duplicates(self, directory: Path) -> tuple[int, int]:
        """Remove duplicate images from directory using perceptual hashing.
        
        Computes phash for all PNG images and removes duplicates based on
        Hamming distance. Tubes at slightly different angles are NOT duplicates.
        
        Args:
            directory: Directory containing PNG images
            
        Returns:
            Tuple of (total_count, kept_count)
        """
        # Collect all PNG files
        files = list(directory.glob("*.png"))
        total = len(files)
        
        # Track seen hashes and removed files
        seen_hashes = {}
        removed = []
        
        for file in files:
            try:
                image = Image.open(file)
                phash = imagehash.phash(image)
            except Exception as e:
                logger.warning(f"Failed to compute hash for {file.name}: {e}")
                continue
            
            # Check against existing hashes
            duplicate_found = False
            for existing_hash, existing_path in seen_hashes.items():
                # Compute Hamming distance (number of differing bits in 64-bit hash)
                # Threshold 12 allows tubes at different angles/positions (not true duplicates)
                diff = phash - existing_hash
                
                if diff <= self.cfg.pipeline.duplicate_hash_threshold:
                    logger.debug(
                        f"Duplicate removed: {file.name} (hash_diff={diff}) "
                        f"matches {existing_path.name} (threshold={self.cfg.pipeline.duplicate_hash_threshold})"
                    )
                    file.unlink()
                    removed.append(file)
                    duplicate_found = True
                    break
            
            # If not a duplicate, add to seen hashes
            if not duplicate_found:
                seen_hashes[phash] = file
        
        # Compute kept count
        kept = total - len(removed)
        
        logger.info(
            f"Duplicate filter: {kept}/{total} images kept in {directory.name} "
            f"(threshold={self.cfg.pipeline.duplicate_hash_threshold})"
        )
        
        return (total, kept)
