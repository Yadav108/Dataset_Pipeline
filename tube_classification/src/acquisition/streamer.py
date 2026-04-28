import numpy as np
import pyrealsense2 as rs
from typing import Any

from loguru import logger
from config.parser import get_config


class RealSenseStreamer:
    """Wraps Intel RealSense D435i pipeline with config-driven streams/options."""
    
    def __init__(self):
        """Initialize RealSense streamer."""
        self.cfg = get_config()
        self.pipeline = rs.pipeline()
        self.align = None
        self.depth_postprocess_cfg = self.cfg.camera.depth_postprocess or {}
        self.depth_postprocess_enabled = bool(self.depth_postprocess_cfg.get("enabled", True))
        self.depth_spatial_filter = None
        self.depth_temporal_filter = None
        self.depth_hole_filling_filter = None
        self.running = False
        self.width = None       # Detected after start()
        self.height = None      # Detected after start()
        self.fps = None         # Detected after start()
        self.depth_scale = 0.001  # Default, will be updated on start
    
    def start(self) -> None:
        """Start RGB and Depth streams with alignment to color.
        
        Raises:
            RuntimeError: If camera not found or fails to initialize.
        """
        try:
            config = rs.config()

            depth_w = self.cfg.camera.depth_width or self.cfg.camera.width
            depth_h = self.cfg.camera.depth_height or self.cfg.camera.height
            depth_fps = self.cfg.camera.depth_fps or self.cfg.camera.fps
            color_w = self.cfg.camera.color_width or self.cfg.camera.width
            color_h = self.cfg.camera.color_height or self.cfg.camera.height
            color_fps = self.cfg.camera.color_fps or self.cfg.camera.fps
            config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
            config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            self._apply_sensor_options(profile)
            
            # Detect actual resolution and parameters from hardware
            self._detect_resolution_from_hardware(profile)
            
            # Align depth to color
            self.align = rs.align(rs.stream.color)
            self._setup_depth_postprocessing()
            self.running = True
            
            logger.info(
                f"RealSense stream started (RGB {color_w}x{color_h}@{color_fps}, "
                f"Depth {depth_w}x{depth_h}@{depth_fps}) — "
                f"Aligned output={self.width}x{self.height} @ {self.fps}fps"
            )
            logger.info(f"Depth scale: {self.depth_scale}")
            
        except RuntimeError as e:
            logger.error(f"Failed to start RealSense camera: {e}")
            raise RuntimeError(
                "RealSense D435i not found or failed to initialize. "
                "Check USB connection and ensure camera is connected."
            ) from e

    def _coerce_option_value(self, value: Any) -> float:
        """Convert YAML/JSON values to RealSense option numeric values."""
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip().lower()
        if text in {"true", "on", "yes", "1"}:
            return 1.0
        if text in {"false", "off", "no", "0"}:
            return 0.0
        return float(value)

    def _resolve_sensor_for_key(
        self,
        key: str,
        depth_sensor: "rs.sensor",
        color_sensor: "rs.sensor"
    ) -> tuple["rs.sensor", str] | None:
        """Map config key to (sensor, rs.option name)."""
        normalized = key.strip().lower().replace("_", "-")

        explicit_map = {
            "controls-autoexposure-auto": ("depth", "enable_auto_exposure"),
            "controls-autoexposure-manual": ("depth", "exposure"),
            "controls-depth-gain": ("depth", "gain"),
            "controls-laserpower": ("depth", "laser_power"),
            "controls-laserstate": ("depth", "emitter_enabled"),
            "controls-color-autoexposure-auto": ("color", "enable_auto_exposure"),
            "controls-color-autoexposure-manual": ("color", "exposure"),
            "controls-color-white-balance-auto": ("color", "enable_auto_white_balance"),
            "controls-color-white-balance-manual": ("color", "white_balance"),
            "controls-color-backlight-compensation": ("color", "backlight_compensation"),
            "controls-color-brightness": ("color", "brightness"),
            "controls-color-contrast": ("color", "contrast"),
            "controls-color-gain": ("color", "gain"),
            "controls-color-gamma": ("color", "gamma"),
            "controls-color-hue": ("color", "hue"),
            "controls-color-power-line-frequency": ("color", "power_line_frequency"),
            "controls-color-saturation": ("color", "saturation"),
            "controls-color-sharpness": ("color", "sharpness"),
        }
        if normalized in explicit_map:
            sensor_kind, option_name = explicit_map[normalized]
            return (depth_sensor if sensor_kind == "depth" else color_sensor, option_name)

        parts = normalized.split(".", 1)
        if len(parts) != 2:
            return None
        sensor_part, option_part = parts
        if sensor_part not in {"depth", "color"}:
            return None

        return (depth_sensor if sensor_part == "depth" else color_sensor, option_part.replace("-", "_"))

    def _apply_sensor_options(self, profile: "rs.pipeline_profile") -> None:
        """Apply optional RealSense sensor options from config.camera.sensor_options."""
        options = self.cfg.camera.sensor_options
        if not options:
            return

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        color_sensor = next(
            (s for s in device.query_sensors() if "rgb" in s.get_info(rs.camera_info.name).lower()),
            None
        )

        if color_sensor is None:
            logger.warning("Color sensor not found; skipping color sensor options")

        for key, raw_value in options.items():
            mapping = self._resolve_sensor_for_key(
                key=key,
                depth_sensor=depth_sensor,
                color_sensor=color_sensor if color_sensor is not None else depth_sensor
            )
            if mapping is None:
                logger.debug(f"Skipping unsupported sensor option key: {key}")
                continue

            sensor, option_name = mapping
            if sensor is None:
                logger.warning(f"Skipping sensor option (sensor unavailable): {key}")
                continue

            if not hasattr(rs.option, option_name):
                logger.warning(f"Skipping unknown RealSense option '{option_name}' from key '{key}'")
                continue

            option = getattr(rs.option, option_name)
            try:
                numeric_value = self._coerce_option_value(raw_value)
            except Exception as e:
                logger.warning(f"Skipping invalid sensor option value for '{key}': {raw_value} ({e})")
                continue

            if not sensor.supports(option):
                logger.warning(f"Sensor does not support option '{option_name}' (key '{key}')")
                continue

            try:
                sensor.set_option(option, numeric_value)
                logger.info(f"Applied sensor option: {key}={raw_value}")
            except RuntimeError as e:
                logger.warning(f"Failed to apply sensor option {key}={raw_value}: {e}")

    def _setup_depth_postprocessing(self) -> None:
        """Initialize RealSense depth post-processing filters from config."""
        if not self.depth_postprocess_enabled:
            return

        try:
            if bool(self.depth_postprocess_cfg.get("spatial_enabled", True)):
                self.depth_spatial_filter = rs.spatial_filter()
                self.depth_spatial_filter.set_option(
                    rs.option.filter_smooth_alpha,
                    float(self.depth_postprocess_cfg.get("spatial_alpha", 0.5))
                )
                self.depth_spatial_filter.set_option(
                    rs.option.filter_smooth_delta,
                    float(self.depth_postprocess_cfg.get("spatial_delta", 20))
                )
                self.depth_spatial_filter.set_option(
                    rs.option.filter_magnitude,
                    float(self.depth_postprocess_cfg.get("spatial_magnitude", 2))
                )
                self.depth_spatial_filter.set_option(
                    rs.option.holes_fill,
                    float(self.depth_postprocess_cfg.get("holes_fill", 2))
                )

            if bool(self.depth_postprocess_cfg.get("temporal_enabled", True)):
                self.depth_temporal_filter = rs.temporal_filter()
                self.depth_temporal_filter.set_option(
                    rs.option.filter_smooth_alpha,
                    float(self.depth_postprocess_cfg.get("temporal_alpha", 0.4))
                )
                self.depth_temporal_filter.set_option(
                    rs.option.filter_smooth_delta,
                    float(self.depth_postprocess_cfg.get("temporal_delta", 20))
                )
                persistency_value = float(self.depth_postprocess_cfg.get("persistency_control", 3))
                persistency_applied = False
                for option_name in ("persistency_index", "holes_fill"):
                    if not hasattr(rs.option, option_name):
                        continue
                    option_enum = getattr(rs.option, option_name)
                    try:
                        self.depth_temporal_filter.set_option(option_enum, persistency_value)
                        persistency_applied = True
                        break
                    except RuntimeError:
                        continue
                if not persistency_applied:
                    logger.warning(
                        "Temporal filter persistency option not supported by installed pyrealsense2; "
                        "continuing without explicit persistency control"
                    )

            if bool(self.depth_postprocess_cfg.get("hole_filling_enabled", True)):
                hole_mode = int(self.depth_postprocess_cfg.get("hole_filling_mode", 1))
                self.depth_hole_filling_filter = rs.hole_filling_filter(hole_mode)

            logger.info("Depth post-processing filters initialized")
        except Exception as e:
            logger.warning(f"Depth post-processing setup failed, continuing without filters: {e}")
            self.depth_spatial_filter = None
            self.depth_temporal_filter = None
            self.depth_hole_filling_filter = None

    def _postprocess_depth_frame(self, depth_frame: "rs.depth_frame") -> "rs.depth_frame":
        """Apply optional RealSense depth post-processing chain."""
        if not self.depth_postprocess_enabled:
            return depth_frame

        filtered = depth_frame
        try:
            if self.depth_spatial_filter is not None:
                filtered = self.depth_spatial_filter.process(filtered)
            if self.depth_temporal_filter is not None:
                filtered = self.depth_temporal_filter.process(filtered)
            if self.depth_hole_filling_filter is not None:
                filtered = self.depth_hole_filling_filter.process(filtered)
            return filtered.as_depth_frame()
        except Exception as e:
            logger.warning(f"Depth post-processing failed for frame, using unfiltered depth: {e}")
            return depth_frame
    
    def _detect_resolution_from_hardware(self, profile: "rs.pipeline_profile") -> None:
        """Detect actual resolution and parameters from hardware after start.
        
        Extracts RGB dimensions, FPS, and depth scale from the active profile.
        
        Args:
            profile: Pipeline profile returned by pipeline.start()
        """
        # Get depth sensor for depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Get RGB stream profile for resolution
        color_profile = profile.get_stream(rs.stream.color)
        color_video_profile = color_profile.as_video_stream_profile()
        intr = color_video_profile.get_intrinsics()
        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy
        self.width = intr.width
        self.height = intr.height
        
        # Get FPS from depth stream
        depth_profile = profile.get_stream(rs.stream.depth)
        self.fps = depth_profile.fps()
        
        logger.debug(
            f"Detected hardware resolution: {self.width}×{self.height} @ {self.fps}fps, "
            f"depth_scale={self.depth_scale}"
        )
    
    def get_aligned_frames(self, timeout_ms: int = 5000) -> tuple[np.ndarray, np.ndarray] | None:
        """Capture and return aligned RGB and depth frames.
        
        RGB and depth match configured stream resolution after alignment.
        
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
            if depth_frame:
                depth_frame = self._postprocess_depth_frame(depth_frame)
            
            if not color_frame or not depth_frame:
                logger.warning("No color or depth frame received")
                return None
            
            color_array = np.asanyarray(color_frame.get_data())
            depth_array = np.asanyarray(depth_frame.get_data())
            
            # RGB and depth are aligned to the same spatial resolution.
            
            return (color_array, depth_array)
            
        except RuntimeError as e:
            if "Frame didn't arrive" in str(e):
                logger.error(
                    f"Camera frame timeout after {timeout_ms}ms. "
                    "This usually means:\n"
                    "  1. USB connection issue (use USB 3.0 port/cable)\n"
                    "  2. Camera driver issue (update pyrealsense2)\n"
                    "  3. Unsupported stream profile for this camera/firmware\n"
                    "  4. RealSense service not running\n"
                    "Try running: python diagnose_camera.py"
                )
            else:
                logger.error(f"Frame capture error: {e}")
            return None

    @property
    def intrinsics(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "depth_scale": self.depth_scale
        }
    
    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self.running:
            self.pipeline.stop()
        
        self.running = False
        logger.info("RealSense stream stopped")
