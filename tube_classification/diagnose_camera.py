#!/usr/bin/env python3
"""Diagnostic script to check RealSense camera capabilities."""

import pyrealsense2 as rs
from loguru import logger

logger.info("Querying RealSense D435if camera capabilities...")

try:
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        logger.error("No RealSense camera found!")
        exit(1)
    
    device = devices[0]
    logger.info(f"Camera found: {device.get_info(rs.camera_info.name)}")
    
    # Get sensor profiles
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Query all available profiles
    logger.info("\n=== COLOR STREAM PROFILES ===")
    color_sensor = device.query_sensors()[1]  # Usually color is sensor 1
    for profile in color_sensor.get_stream_profiles():
        stream = profile.stream_type()
        fmt = profile.format()
        width = profile.as_video_stream_profile().width()
        height = profile.as_video_stream_profile().height()
        fps = profile.fps()
        logger.info(f"  {stream} - {width}x{height} @ {fps}fps, format: {fmt}")
    
    logger.info("\n=== DEPTH STREAM PROFILES ===")
    depth_sensor = device.query_sensors()[0]  # Usually depth is sensor 0
    for profile in depth_sensor.get_stream_profiles():
        stream = profile.stream_type()
        fmt = profile.format()
        width = profile.as_video_stream_profile().width()
        height = profile.as_video_stream_profile().height()
        fps = profile.fps()
        logger.info(f"  {stream} - {width}x{height} @ {fps}fps, format: {fmt}")
    
    # Try the current config
    logger.info("\n=== TESTING CURRENT CONFIG (848x480 @ 30fps) ===")
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        logger.info("✓ Config 848x480@30fps WORKS!")
        pipeline.stop()
    except RuntimeError as e:
        logger.error(f"✗ Config 848x480@30fps FAILED: {e}")
        
        # Try fallback configs
        logger.info("\n=== TESTING FALLBACK CONFIGS ===")
        
        fallback_configs = [
            (640, 480, 30),
            (848, 480, 15),
            (640, 480, 15),
            (1280, 720, 15),
        ]
        
        for w, h, f in fallback_configs:
            config = rs.config()
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
            
            try:
                profile = pipeline.start(config)
                logger.info(f"✓ Config {w}x{h}@{f}fps WORKS!")
                pipeline.stop()
            except RuntimeError as e:
                logger.error(f"✗ Config {w}x{h}@{f}fps FAILED: {e}")

except Exception as e:
    logger.error(f"Error: {e}")
    exit(1)
