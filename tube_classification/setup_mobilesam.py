#!/usr/bin/env python
"""Download and setup MobileSAM weights for the pipeline.

This script:
1. Creates the models directory
2. Downloads MobileSAM weights from GitHub
3. Verifies the download integrity
4. Tests the weights load correctly
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, format="<level>{level}</level> | {message}", level="INFO")

# Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "mobile_sam.pt"

# MobileSAM weights URLs (multiple mirrors)
MOBILESAM_URLS = [
    # Primary: HuggingFace mirror
    "https://huggingface.co/ChaoningZhang/MobileSAM/resolve/main/weights/mobile_sam.pt",
    # Alternative: Direct from ChaoningZhang repo
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    # Fallback: GitHub raw
    "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt",
]

def create_models_directory():
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Models directory ready: {MODELS_DIR}")

def download_weights():
    """Download MobileSAM weights from multiple mirror sources."""
    if WEIGHTS_PATH.exists():
        logger.info(f"✓ MobileSAM weights already exist at {WEIGHTS_PATH}")
        return True
    
    logger.info(f"Attempting to download MobileSAM weights...")
    logger.info(f"This may take a few minutes (file is ~40MB)...")
    logger.info(f"Trying multiple mirrors...")
    
    for attempt, url in enumerate(MOBILESAM_URLS, 1):
        logger.info(f"\nAttempt {attempt}/{len(MOBILESAM_URLS)}: {url}")
        
        try:
            def download_progress(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, int((downloaded / total_size) * 100))
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r  Progress: {percent}% ({mb:.1f}MB / {total_mb:.1f}MB)", end="")
            
            urllib.request.urlretrieve(url, str(WEIGHTS_PATH), download_progress)
            print()  # Newline after progress
            
            logger.info(f"✓ Download complete from mirror {attempt}")
            return True
            
        except Exception as e:
            logger.warning(f"  Mirror {attempt} failed: {type(e).__name__}: {str(e)[:60]}")
            # Clean up partial download
            if WEIGHTS_PATH.exists():
                try:
                    WEIGHTS_PATH.unlink()
                except:
                    pass
            continue
    
    logger.error(f"\n✗ All download mirrors failed!")
    logger.error("\nAlternative solutions:")
    logger.error("1. Download manually from HuggingFace:")
    logger.error("   https://huggingface.co/ChaoningZhang/MobileSAM/blob/main/weights/mobile_sam.pt")
    logger.error("2. Or clone the repo and copy weights:")
    logger.error("   git clone https://github.com/ChaoningZhang/MobileSAM.git")
    logger.error("   cp MobileSAM/weights/mobile_sam.pt tube_classification/models/")
    logger.error(f"3. Place the file at: {WEIGHTS_PATH}")
    logger.error("4. Run this script again")
    return False

def verify_weights():
    """Verify the weights file exists and can be loaded."""
    if not WEIGHTS_PATH.exists():
        logger.error(f"✗ Weights file not found: {WEIGHTS_PATH}")
        return False
    
    file_size = WEIGHTS_PATH.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    logger.info(f"✓ Weights file exists: {file_size_mb:.1f}MB")
    
    # Check file size (mobile_sam.pt is typically 38-40MB)
    if file_size < 30_000_000:  # Less than 30MB
        logger.error(f"✗ Weights file too small ({file_size_mb:.1f}MB), may be corrupted")
        return False
    
    if file_size > 50_000_000:  # More than 50MB
        logger.error(f"✗ Weights file too large ({file_size_mb:.1f}MB), may be wrong file")
        return False
    
    logger.info(f"✓ File size check passed")
    return True

def test_load_weights():
    """Test that weights can be loaded by PyTorch."""
    try:
        import torch
        logger.info("Testing weights load...")
        
        state_dict = torch.load(str(WEIGHTS_PATH), map_location="cpu")
        logger.info(f"✓ Weights loaded successfully")
        logger.info(f"  Model keys: {len(state_dict)} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load weights: {e}")
        return False

def main():
    """Main setup routine."""
    logger.info("=" * 70)
    logger.info("MobileSAM Weights Setup")
    logger.info("=" * 70)
    
    # Step 1: Create directory
    create_models_directory()
    
    # Step 2: Download weights
    if not download_weights():
        return 1
    
    # Step 3: Verify weights
    if not verify_weights():
        logger.error("Verification failed. Please download manually.")
        return 1
    
    # Step 4: Test load
    if not test_load_weights():
        logger.error("Weights load test failed. File may be corrupted.")
        return 1
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Setup Complete! MobileSAM is ready to use.")
    logger.info("=" * 70)
    logger.info("")
    logger.info("You can now run the pipeline:")
    logger.info("  python main.py")
    logger.info("")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
