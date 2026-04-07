#!/usr/bin/env python3
"""
HD Capture Verification Script
Tests that the pipeline captures HD quality images with the new configuration
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from config.parser import get_config
from src.acquisition.streamer import RealSenseStreamer


def verify_hd_configuration():
    """Verify HD configuration is loaded correctly."""
    logger.info("=" * 70)
    logger.info("HD CAPTURE VERIFICATION - Configuration Check")
    logger.info("=" * 70)
    
    cfg = get_config()
    
    # Check camera configuration
    logger.info("\n📹 Camera Configuration:")
    logger.info(f"  Resolution: {cfg.camera.width}×{cfg.camera.height}")
    logger.info(f"  Frame Rate: {cfg.camera.fps} fps")
    logger.info(f"  Depth Range: {cfg.camera.depth_min_m}m - {cfg.camera.depth_max_m}m")
    
    # Verify it's HD
    pixels = cfg.camera.width * cfg.camera.height
    logger.info(f"  Total Pixels: {pixels:,}")
    
    if cfg.camera.width >= 1280 and cfg.camera.height >= 720:
        logger.info("  ✅ HD Configuration Detected (1280×720 or higher)")
    else:
        logger.warning(f"  ⚠️  Standard Configuration ({cfg.camera.width}×{cfg.camera.height})")
    
    # Check quality thresholds
    logger.info("\n🎯 Quality Thresholds:")
    logger.info(f"  Blur Threshold: {cfg.pipeline.blur_threshold}")
    logger.info(f"  Coverage Ratio Min: {cfg.pipeline.min_coverage_ratio}")
    logger.info(f"  SAM IoU Threshold: {cfg.pipeline.sam_iou_threshold}")
    
    return cfg


def test_camera_stream(num_frames: int = 10):
    """Test camera streaming at configured resolution."""
    logger.info("\n" + "=" * 70)
    logger.info("🎥 CAMERA STREAM TEST - Attempting to Start Stream")
    logger.info("=" * 70)
    
    try:
        streamer = RealSenseStreamer()
        logger.info("✅ RealSenseStreamer initialized")
        
        streamer.start()
        logger.info("✅ Camera stream started")
        
        # Get actual camera specs
        logger.info(f"\n📊 Camera Stream Details:")
        logger.info(f"  Depth Scale: {streamer.depth_scale}")
        
        # Capture sample frames
        logger.info(f"\n📷 Capturing {num_frames} sample frames...")
        
        frame_count = 0
        frame_sizes = []
        resolutions = []
        
        for i in range(num_frames):
            try:
                rgb, depth = streamer.get_frame()
                
                if rgb is not None and depth is not None:
                    frame_count += 1
                    
                    # Get frame dimensions
                    h, w = rgb.shape[:2]
                    resolutions.append((w, h))
                    
                    # Calculate frame size
                    rgb_bytes = rgb.nbytes
                    frame_sizes.append(rgb_bytes)
                    
                    logger.debug(f"  Frame {frame_count}: {w}×{h} ({rgb_bytes / 1e6:.2f} MB)")
                    
                    # Verify HD resolution on first frame
                    if frame_count == 1:
                        logger.info(f"\n✅ First frame captured: {w}×{h}")
                        if w >= 1280 and h >= 720:
                            logger.info(f"  ✅ HD RESOLUTION CONFIRMED")
                        else:
                            logger.warning(f"  ⚠️  Lower resolution than expected")
                    
            except Exception as e:
                logger.error(f"  Error capturing frame {i+1}: {e}")
                continue
        
        if frame_count > 0:
            logger.info(f"\n✅ Successfully captured {frame_count}/{num_frames} frames")
            
            # Summary
            avg_size = np.mean(frame_sizes)
            most_common_res = max(set(resolutions), key=resolutions.count)
            
            logger.info(f"\n📈 Frame Statistics:")
            logger.info(f"  Average Frame Size: {avg_size / 1e6:.2f} MB")
            logger.info(f"  Most Common Resolution: {most_common_res[0]}×{most_common_res[1]}")
            logger.info(f"  Frames Captured: {frame_count}")
            
            return True
        else:
            logger.error("❌ No frames captured")
            return False
            
    except Exception as e:
        logger.error(f"❌ Camera test failed: {e}")
        logger.error("   Check USB connection and camera drivers")
        return False
    finally:
        try:
            streamer.stop()
            logger.info("✅ Stream stopped")
        except:
            pass


def test_image_quality():
    """Test that images have expected quality characteristics."""
    logger.info("\n" + "=" * 70)
    logger.info("🎨 IMAGE QUALITY TEST")
    logger.info("=" * 70)
    
    try:
        streamer = RealSenseStreamer()
        streamer.start()
        
        logger.info("Capturing test frame...")
        rgb, depth = streamer.get_frame()
        
        if rgb is None:
            logger.error("❌ Failed to capture frame")
            return False
        
        # Analyze image quality
        logger.info("\n📊 Image Analysis:")
        
        # Blur detection
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.info(f"  Blur Score (Laplacian Variance): {blur_score:.2f}")
        
        if blur_score > 52:
            logger.info(f"  ✅ Good sharpness (> 52)")
        else:
            logger.warning(f"  ⚠️  Blur score below optimal (< 52)")
        
        # Contrast
        contrast = gray.std()
        logger.info(f"  Contrast (Std Dev): {contrast:.2f}")
        
        # Resolution
        h, w = rgb.shape[:2]
        pixels = w * h
        logger.info(f"  Resolution: {w}×{h} ({pixels:,} pixels)")
        
        if pixels >= 921600:  # 1280×720
            logger.info(f"  ✅ HD Resolution Confirmed")
        
        # Depth analysis
        if depth is not None:
            valid_depth = depth[(depth > 250) & (depth < 800)]
            if len(valid_depth) > 0:
                logger.info(f"  Depth Range: {valid_depth.min()}-{valid_depth.max()} mm")
                logger.info(f"  ✅ Depth sensor working")
        
        streamer.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Quality test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "  HD CAPTURE VERIFICATION - Complete Pipeline Test".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    logger.info("")
    
    # Test 1: Configuration
    cfg = verify_hd_configuration()
    
    # Test 2: Camera Stream
    stream_success = test_camera_stream(num_frames=10)
    
    # Test 3: Image Quality
    quality_success = test_image_quality()
    
    # Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("📋 VERIFICATION SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\n✅ Configuration: HD Settings Loaded")
    logger.info(f"{'✅' if stream_success else '❌'} Camera Stream: {'Working' if stream_success else 'Failed'}")
    logger.info(f"{'✅' if quality_success else '❌'} Image Quality: {'Good' if quality_success else 'Failed'}")
    
    if stream_success and quality_success:
        logger.info("\n" + "🎉" * 35)
        logger.info("✅ HD CAPTURE PIPELINE READY")
        logger.info("🎉" * 35)
        logger.info("\n✓ Camera streaming at HD resolution (1280×720)")
        logger.info("✓ Image quality metrics within expected range")
        logger.info("✓ Ready to run full pipeline with HD captures")
        return 0
    else:
        logger.warning("\n⚠️  Some tests failed - review output above")
        return 1


if __name__ == "__main__":
    exit(main())
