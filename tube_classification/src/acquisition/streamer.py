import numpy as np
import pyrealsense2 as rs

from loguru import logger
from config.parser import get_config


class RealSenseStreamer:
    """Wraps Intel RealSense D435if pipeline.
    
    Opens RGB + Depth streams, provides aligned frame pairs, handles clean shutdown.
    """
    
    def __init__(self):
        """Initialize RealSense streamer with config."""
        self.cfg = get_config()
        self.pipeline = rs.pipeline()
        self.align = None
        self.running = False
        self.depth_scale = 0.001  # Default, will be updated on start
    
    def start(self) -> None:
        """Start RGB and Depth streams with alignment to color.
        
        Enables color stream (BGR8) and depth stream (Z16) with resolution
        and frame rate from config. Captures actual depth scale from device.
        
        If primary config fails, automatically tries fallback configurations
        to find a working profile supported by the camera.
        """
        # List of fallback configs: (width, height, fps)
        # Primary config from settings, then fallbacks in order of preference
        configs_to_try = [
            (self.cfg.camera.width, self.cfg.camera.height, self.cfg.camera.fps),  # Primary: 1280×720@15
            (1024, 768, 30),   # Fallback 1: Good balance
            (848, 480, 30),    # Fallback 2: Previous config
            (640, 480, 30),    # Fallback 3: Conservative
        ]
        
        profile = None
        actual_width = None
        actual_height = None
        actual_fps = None
        
        for width, height, fps in configs_to_try:
            try:
                config = rs.config()
                
                # Enable color stream
                config.enable_stream(
                    rs.stream.color,
                    width,
                    height,
                    rs.format.bgr8,
                    fps,
                )
                
                # Enable depth stream
                config.enable_stream(
                    rs.stream.depth,
                    width,
                    height,
                    rs.format.z16,
                    fps,
                )
                
                # Try to start pipeline
                profile = self.pipeline.start(config)
                actual_width = width
                actual_height = height
                actual_fps = fps
                
                if (width, height, fps) != (self.cfg.camera.width, self.cfg.camera.height, self.cfg.camera.fps):
                    logger.warning(
                        f"Primary config ({self.cfg.camera.width}x{self.cfg.camera.height}@{self.cfg.camera.fps}fps) "
                        f"not supported. Using fallback: {width}x{height}@{fps}fps"
                    )
                break
                
            except RuntimeError as e:
                logger.debug(f"Config {width}x{height}@{fps}fps failed: {e}")
                continue
        
        if profile is None:
            logger.error(
                f"Could not resolve any camera configuration. "
                f"Tried: {configs_to_try}"
            )
            raise RuntimeError(
                "RealSense camera could not be initialized with any supported configuration. "
                "Ensure camera is connected and check USB connection."
            )
        
        # Get actual depth scale from device
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        self.running = True
        
        logger.info(
            f"RealSense stream started — {actual_width}x{actual_height} @ {actual_fps}fps"
        )
        logger.info(f"Depth scale: {self.depth_scale}")
    
    def get_aligned_frames(self, timeout_ms: int = 5000) -> tuple[np.ndarray, np.ndarray] | None:
        """Capture and return aligned RGB and depth frames.
        
        Args:
            timeout_ms: Timeout in milliseconds to wait for frames (default: 5000ms)
        
        Returns:
            Tuple of (color_array, depth_array) or None if stream not running
            or frames unavailable.
        """
        if not self.running:
            return None
        
        try:
            # Try to wait for frames with timeout
            frames = self.pipeline.wait_for_frames(timeout_ms)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                logger.warning("No color or depth frame received")
                return None
            
            color_array = np.asanyarray(color_frame.get_data())
            depth_array = np.asanyarray(depth_frame.get_data())
            
            return (color_array, depth_array)
            
        except RuntimeError as e:
            if "Frame didn't arrive" in str(e):
                logger.error(
                    f"Camera frame timeout after {timeout_ms}ms. "
                    "This usually means:\n"
                    "  1. USB connection issue (use USB 3.0 port/cable)\n"
                    "  2. Camera driver issue (update pyrealsense2)\n"
                    "  3. Unsupported resolution (1280×720 may not be supported)\n"
                    "  4. RealSense service not running\n"
                    "Try running: python diagnose_camera.py"
                )
            else:
                logger.error(f"Frame capture error: {e}")
            return None
    
    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self.running:
            self.pipeline.stop()
        
        self.running = False
        logger.info("RealSense stream stopped")
