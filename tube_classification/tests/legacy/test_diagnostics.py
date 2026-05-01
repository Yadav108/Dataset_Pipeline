#!/usr/bin/env python
"""
Capture Diagnostics Script
Runs a test capture and provides detailed diagnostics on frame filtering.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.parser import get_config

# Enable debug logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<level>{time:HH:mm:ss.SSS}</level> | <level>{level: <8}</level> | <level>{message}</level>")

print("\n" + "="*70)
print("CAPTURE DIAGNOSTICS - Frame Filtering Analysis")
print("="*70 + "\n")

cfg = get_config()

logger.info(f"Configuration:")
logger.info(f"  Depth Range: {cfg.camera.depth_min_m:.3f}m - {cfg.camera.depth_max_m:.3f}m")
logger.info(f"  Stability Threshold: {cfg.pipeline.depth_stability_threshold}m")
logger.info(f"  Stability Frames Required: {cfg.pipeline.stability_frames}")
logger.info(f"  Min ROI Area: {cfg.pipeline.min_roi_area_px}px")
logger.info(f"  SAM IOU Threshold: {cfg.pipeline.sam_iou_threshold}")

# Test 1: Depth value range check
print("\n" + "-"*70)
print("TEST 1: Depth Sensor Range Verification")
print("-"*70 + "\n")

try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        logger.error("No RealSense device detected")
        sys.exit(1)
    
    device = devices[0]
    logger.info(f"Device: {device.get_info(rs.camera_info.name)}")
    
    # Get depth sensor
    depth_sensor = None
    for sensor in device.query_sensors():
        if sensor.get_info(rs.camera_info.name) == 'Stereo Module':
            depth_sensor = sensor
            break
    
    if depth_sensor:
        depth_scale = device.first_depth_sensor().get_depth_scale()
        logger.info(f"Depth Scale: {depth_scale} (multiply raw values by this)")
        logger.info(f"Raw depth range: {cfg.camera.depth_min_m/depth_scale:.0f} - {cfg.camera.depth_max_m/depth_scale:.0f} (in raw units)")
        
        # Check if current camera is using the right scale
        if depth_scale != 0.001:
            logger.warning(f"⚠️  Depth scale is {depth_scale}, but code assumes 0.001!")
            logger.warning(f"    This could cause ALL frames to be rejected!")
            logger.warning(f"    Recommended: Change line in roi_extractor.py from:")
            logger.warning(f"      depth_m = depth_frame.astype(np.float32) * 0.001")
            logger.warning(f"    To:")
            logger.warning(f"      depth_m = depth_frame.astype(np.float32) * {depth_scale}")
    
except Exception as e:
    logger.error(f"Failed to query depth sensor: {e}")

# Test 2: Simulate frame processing
print("\n" + "-"*70)
print("TEST 2: Simulated Frame Processing")
print("-"*70 + "\n")

def create_tube_frame():
    """Create a frame with simulated tube (depth values in valid range)."""
    frame = np.zeros((480, 848), dtype=np.uint16)
    # Add a tube-like region (400-500 depth values = 0.4-0.5m)
    frame[200:350, 350:500] = np.random.randint(400, 500, (150, 150))
    return frame

from src.acquisition.stability_detector import DepthStabilityDetector
from src.annotation.roi_extractor import DepthROIExtractor

detector = DepthStabilityDetector()
roi_extractor = DepthROIExtractor()

# Create test depth frames with tube at different distances
test_cases = [
    ("Empty frame (no tube)", np.zeros((480, 848), dtype=np.uint16)),
    ("Out of range (too close)", np.full((480, 848), dtype=np.uint16, fill_value=200)),  # 0.2m
    ("Out of range (too far)", np.full((480, 848), dtype=np.uint16, fill_value=700)),    # 0.7m
    ("In range - center", np.full((480, 848), dtype=np.uint16, fill_value=400)),         # 0.4m
    ("In range - with ROI", create_tube_frame()),
]

for test_name, test_frame in test_cases:
    logger.info(f"\n{test_name}:")
    logger.info(f"  Depth range in frame: {test_frame.min() * 0.001:.3f}m - {test_frame.max() * 0.001:.3f}m")
    
    # Test stability
    stable = detector.check(test_frame)
    logger.info(f"  Stability check: {stable}")
    
    if test_frame.max() > 0:  # Skip ROI extraction for empty frames
        # Test ROI extraction
        bbox = roi_extractor.extract(test_frame)
        logger.info(f"  ROI extraction: {bbox if bbox else 'None (no ROI found)'}")
    
    detector.reset()

# Test 3: Depth frame statistics
print("\n" + "-"*70)
print("TEST 3: Live Frame Analysis (5 frames)")
print("-"*70)
print("Analyzing actual frames from camera...\n")

try:
    from src.acquisition.streamer import RealSenseStreamer
    
    streamer = RealSenseStreamer()
    streamer.start()
    
    for i in range(5):
        frames = streamer.get_aligned_frames()
        if frames is None:
            logger.warning(f"Frame {i+1}: Failed to get frames")
            continue
        
        rgb_frame, depth_frame = frames
        
        logger.info(f"Frame {i+1}:")
        logger.info(f"  RGB shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
        logger.info(f"  Depth shape: {depth_frame.shape}, dtype: {depth_frame.dtype}")
        logger.info(f"  Depth range: {depth_frame.min() * 0.001:.3f}m - {depth_frame.max() * 0.001:.3f}m")
        
        # Count pixels in valid range
        depth_m = depth_frame.astype(np.float32) * 0.001
        valid_pixels = np.sum((depth_m >= cfg.camera.depth_min_m) & (depth_m <= cfg.camera.depth_max_m))
        total_pixels = depth_frame.shape[0] * depth_frame.shape[1]
        logger.info(f"  Valid depth pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
    
    streamer.stop()
    
except Exception as e:
    logger.error(f"Frame analysis failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70 + "\n")

print("RECOMMENDATIONS:\n")
print("If 0 images were captured, check these in order:\n")
print("1. ✅ Depth scale (TEST 1)")
print("   - Verify depth scale matches code assumptions")
print("   - Check if using 0.001 is correct for your camera\n")

print("2. ✅ Depth range (TEST 3)")
print("   - Ensure depth values fall in [0.32m, 0.56m]")
print("   - If all pixels are outside range, tube is too close/far\n")

print("3. ✅ Stability frames requirement")
print("   - Need 10 consecutive stable frames")
print("   - Keep tube still for ~0.33 seconds (at 30fps)\n")

print("4. ✅ ROI area requirement")
print("   - Tube bounding box must be ≥500 pixels")
print("   - For 3.5ml tubes, might need ≤200px (check config)\n")

print("5. ✅ SAM segmentation")
print("   - Verify SAM model can see the tube")
print("   - Check IOU threshold (0.88) isn't too strict\n")
