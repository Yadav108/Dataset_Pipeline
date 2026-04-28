import numpy as np

from loguru import logger
from config.parser import get_config


class DepthStabilityDetector:
    """Per-frame depth analysis for tube presence and scene stillness.
    
    Detects tube presence in depth zone and monitors scene stability
    across consecutive frames.
    """
    
    def __init__(self, depth_scale: float = 0.001):
        """Initialize detector with config.
        
        Args:
            depth_scale: Depth value scale factor (default 0.001 for RealSense)
        """
        self.cfg = get_config()
        self.depth_scale = depth_scale
        self.consecutive_stable = 0
        self.prev_depth_roi = None
        self.last_roi_param = None  # Track ROI changes to detect size transitions
    
    def _extract_roi(self, depth_frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Extract ROI from depth frame for stability analysis.
        
        Priority order:
        1. If roi provided and valid (non-zero w/h): use that bbox region
        2. Otherwise: use narrow vertical center strip (20% width, full height)
           - Avoids operator's hand during tube holding
        
        Args:
            depth_frame: Full depth frame array
            roi: Optional tuple (x, y, w, h) for explicit region. If None or invalid,
                 falls back to narrow center strip.
            
        Returns:
            Cropped depth ROI
        """
        h, w = depth_frame.shape[:2]
        
        # If roi provided and valid, use it
        if roi is not None:
            x, y, box_w, box_h = roi
            if box_w > 0 and box_h > 0:
                # Clamp to frame bounds
                x_start = max(0, x)
                y_start = max(0, y)
                x_end = min(w, x + box_w)
                y_end = min(h, y + box_h)
                region_used = "explicit_roi"
                logger.debug(f"Stability: Using explicit ROI [{x_start}:{x_end}, {y_start}:{y_end}]")
                return depth_frame[y_start:y_end, x_start:x_end]
        
        # Fallback: narrow vertical center strip (20% width, full height)
        # Narrower than operator's hand (~8cm ≈ 100px), avoids grip zone
        strip_width = w // 5  # 20% of width
        strip_x_start = (w - strip_width) // 2
        strip_x_end = strip_x_start + strip_width
        region_used = "center_strip_fallback"
        logger.debug(f"Stability: Using center strip fallback [{strip_x_start}:{strip_x_end}, 0:{h}] (20% width)")
        return depth_frame[0:h, strip_x_start:strip_x_end]
    
    def _depth_array_to_meters(self, depth_roi: np.ndarray) -> np.ndarray:
        """Convert raw depth uint16 values to meters.
        
        Args:
            depth_roi: Depth array in uint16 format
            
        Returns:
            Depth array in meters (float)
        """
        return depth_roi.astype(np.float32) * self.depth_scale
    
    def is_object_present(self, depth_meters: np.ndarray, depth_min: float, depth_max: float) -> bool:
        """Check if any pixel falls within configured depth range.
        
        Args:
            depth_meters: Depth array in meters
            depth_min: Minimum depth threshold in meters
            depth_max: Maximum depth threshold in meters
            
        Returns:
            True if at least one pixel is in [depth_min, depth_max]
        """
        in_range = (
            (depth_meters >= depth_min)
            & (depth_meters <= depth_max)
        )
        present = np.any(in_range)
        if present:
            pixels_in_range = np.sum(in_range)
            avg_depth = np.mean(depth_meters[in_range])
            logger.debug(f"✓ Object present: {pixels_in_range} pixels, avg depth={avg_depth:.3f}m (range: {depth_min:.2f}-{depth_max:.2f}m)")
        else:
            logger.debug(f"✗ No object in depth range {depth_min:.2f}-{depth_max:.2f}m (depth values: {np.min(depth_meters):.3f}-{np.max(depth_meters):.3f}m)")
        return present
    
    def is_stable(self, depth_meters: np.ndarray, depth_min: float, depth_max: float) -> bool:
        """Check if scene is stable across consecutive frames.
        
        Computes stability only on pixels within the valid depth range.
        Excludes background and invalid pixels from MAD calculation.
        
        Args:
            depth_meters: Current depth array in meters
            depth_min: Minimum depth threshold in meters
            depth_max: Maximum depth threshold in meters
            
        Returns:
            True if scene has been stable for configured stability_frames
        """
        if self.prev_depth_roi is None:
            self.prev_depth_roi = depth_meters.copy()
            return False
        
        # Build mask for pixels within valid depth range
        in_range_mask = (
            (depth_meters >= depth_min)
            & (depth_meters <= depth_max)
        )
        
        # If no pixels in valid range, reset and return False
        if not np.any(in_range_mask):
            self.consecutive_stable = 0
            self.prev_depth_roi = depth_meters.copy()
            return False
        
        # Compute stability metric: median absolute deviation (MAD) on in-range pixels
        # More robust than mean when outliers are present
        current_in_range = depth_meters[in_range_mask]
        prev_in_range = self.prev_depth_roi[in_range_mask]
        
        # Only compare if both frames have in-range data
        if len(current_in_range) == 0 or len(prev_in_range) == 0:
            self.consecutive_stable = 0
            self.prev_depth_roi = depth_meters.copy()
            return False
        
        # Use median depth for robust comparison (immune to outliers)
        current_median = np.median(current_in_range)
        prev_median = np.median(prev_in_range)
        diff = np.abs(current_median - prev_median)
        
        # Store current frame for next comparison (keeping full frame for mask consistency)
        self.prev_depth_roi = depth_meters.copy()
        
        threshold = self.cfg.pipeline.depth_stability_threshold
        if diff < threshold:
            self.consecutive_stable += 1
            logger.debug(f"✓ Stable frame #{self.consecutive_stable}: median_diff={diff:.4f}m (threshold={threshold:.4f}m)")
        else:
            self.consecutive_stable = 0
            logger.debug(f"✗ Unstable frame: median_diff={diff:.4f}m > threshold={threshold:.4f}m (depth range: {depth_min:.2f}-{depth_max:.2f}m)")
        
        return self.consecutive_stable >= self.cfg.pipeline.stability_frames
    
    def check(self, depth_frame: np.ndarray, mode: str = "single_side", roi: tuple[int, int, int, int] | None = None) -> bool:
        """Perform full stability check on depth frame.
        
        Checks for object presence and scene stability. Uses mode-specific
        depth ranges and optional ROI for determining valid capture zone.
        Falls back to narrow vertical center strip if no ROI provided.
        
        Args:
            depth_frame: Raw depth frame
            mode: Capture mode ("single_side" or "single_top")
            roi: Optional bounding box (x, y, w, h) for tube region. If None,
                 uses narrow center strip to avoid hand interference.
            
        Returns:
            True if object is present and scene is stable
        """
        # Detect ROI parameter changes and reset if needed
        # (prevents comparing stability across different ROI sizes)
        if roi != self.last_roi_param:
            self.consecutive_stable = 0
            self.prev_depth_roi = None
            self.last_roi_param = roi
        
        # Select depth range based on mode
        if mode == "single_top":
            depth_min = self.cfg.pipeline.top_depth_min_m
            depth_max = self.cfg.pipeline.top_depth_max_m
        else:
            depth_min = self.cfg.camera.depth_min_m
            depth_max = self.cfg.camera.depth_max_m
        
        depth_roi = self._extract_roi(depth_frame, roi=roi)
        depth_meters = self._depth_array_to_meters(depth_roi)
        
        if not self.is_object_present(depth_meters, depth_min, depth_max):
            self.consecutive_stable = 0
            return False
        
        return self.is_stable(depth_meters, depth_min, depth_max)
    
    def compute_bbox_depth_stability(
        self,
        depth_frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> float | None:
        """Compute depth stability metric over a specific bbox region.
        
        Computes median depth change over bbox pixels, filtering to valid range.
        Useful for local depth stability checks on ROI instead of full frame.
        
        Args:
            depth_frame: Full depth frame (uint16)
            bbox: Region of interest (x, y, w, h) or None for full frame
            
        Returns:
            Median depth difference or None if insufficient valid data
        """
        # Extract depth values in bbox
        depth_m = depth_frame.astype(np.float32) * self.depth_scale
        
        if bbox is not None:
            x, y, w, h = bbox
            depth_region = depth_m[y:y+h, x:x+w]
        else:
            depth_region = depth_m
        
        # Filter to valid depth range [0.05m, 0.40m] — ignore noise/outliers
        valid_mask = (depth_region >= 0.05) & (depth_region <= 0.40)
        valid_depths = depth_region[valid_mask]
        
        if len(valid_depths) == 0:
            return None
        
        # Return median for robust comparison
        return float(np.median(valid_depths))
    
    def reset(self) -> None:
        """Reset stability detector state."""
        self.consecutive_stable = 0
        self.prev_depth_roi = None
        self.last_roi_param = None
