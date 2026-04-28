#!/usr/bin/env python
"""
Test script to verify RealSense camera startup with config fallback logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.acquisition.streamer import RealSenseStreamer

if __name__ == "__main__":
    logger.info("=== Testing RealSense Camera Startup ===")
    
    try:
        streamer = RealSenseStreamer()
        logger.info("Streamer initialized")
        
        streamer.start()
        logger.info("✅ Camera stream started successfully!")
        logger.info(f"Depth scale: {streamer.depth_scale}")
        
        # Capture a frame to verify
        frames = streamer.get_aligned_frames()
        if frames:
            color, depth = frames
            logger.info(f"✅ Successfully captured frames - Color shape: {color.shape}, Depth shape: {depth.shape}")
        else:
            logger.warning("⚠️ No frames available yet (may be normal on first call)")
        
        streamer.stop()
        logger.info("✅ All tests passed!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
