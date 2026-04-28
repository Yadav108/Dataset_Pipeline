#!/usr/bin/env python
"""Quick verification that MobileSAM path is now correct."""

import sys
from pathlib import Path
import torch
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, format="<level>{level}</level> | {message}", level="INFO")

PROJECT_ROOT = Path(__file__).parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "MobileSAM" / "weights" / "mobile_sam.pt"

logger.info("=" * 70)
logger.info("MobileSAM Path Verification")
logger.info("=" * 70)

# Check 1: File exists
if not WEIGHTS_PATH.exists():
    logger.error(f"✗ File not found at: {WEIGHTS_PATH}")
    sys.exit(1)

file_size = WEIGHTS_PATH.stat().st_size
file_size_mb = file_size / (1024 * 1024)
logger.info(f"✓ File found: {WEIGHTS_PATH}")
logger.info(f"✓ File size: {file_size_mb:.1f}MB")

# Check 2: Load weights
try:
    state_dict = torch.load(str(WEIGHTS_PATH), map_location="cpu")
    logger.info(f"✓ Weights loaded successfully!")
    logger.info(f"  Model parameters: {len(state_dict)}")
except Exception as e:
    logger.error(f"✗ Failed to load weights: {e}")
    sys.exit(1)

logger.info("")
logger.info("=" * 70)
logger.info("✓ Setup Complete! Ready to run pipeline.")
logger.info("=" * 70)
logger.info("")
logger.info("Next step:")
logger.info("  python main.py")
logger.info("")

sys.exit(0)
