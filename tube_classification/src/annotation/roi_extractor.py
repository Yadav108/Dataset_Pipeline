import numpy as np
import cv2

from loguru import logger
from config.parser import get_config


class DepthROIExtractor:
    """Extract bounding box of tube from aligned depth frame.
    
    Uses depth zone masking and contour detection to identify and
    extract the tube region of interest.
    """
    
    def __init__(self, depth_scale: float = 0.001):
        """Initialize ROI extractor with config.
        
        Args:
            depth_scale: Depth value scale factor (default 0.001 for RealSense)
        """
        self.cfg = get_config()
        self._depth_scale = depth_scale
        self._debug_mask_saved = False
    
    def _create_depth_mask(
        self,
        depth_frame: np.ndarray,
        depth_min: float,
        depth_max: float
    ) -> np.ndarray:
        """Create binary mask for pixels within depth range (no morphology).
        
        Args:
            depth_frame: Raw depth frame (uint16)
            depth_min: Minimum valid depth (meters)
            depth_max: Maximum valid depth (meters)
            
        Returns:
            Binary mask (0 or 255) of pixels in range, no morphological processing
        """
        depth_m = depth_frame.astype(np.float32) * self._depth_scale
        in_range = ((depth_m >= depth_min) & (depth_m <= depth_max))
        return in_range.astype(np.uint8) * 255
    
    def _preprocess_depth(
        self,
        depth_frame: np.ndarray,
        depth_min: float,
        depth_max: float
    ) -> np.ndarray:
        """Clean depth frame and return binary mask for valid depth range.
        
        Applies depth-to-meter conversion, in-range masking, median blur
        (salt-pepper noise removal), and morphological close/open operations.
        
        Args:
            depth_frame: Raw depth frame (uint16)
            depth_min: Minimum valid depth (meters)
            depth_max: Maximum valid depth (meters)
            
        Returns:
            Binary uint8 mask (0 or 255), shape (H, W)
        """
        # Convert to meters
        depth_m = depth_frame.astype(np.float32) * self._depth_scale
        
        # Build in-range mask
        in_range = ((depth_m >= depth_min) & (depth_m <= depth_max))
        mask = in_range.astype(np.uint8) * 255
        
        # Median blur (kills salt-and-pepper depth sensor noise)
        mask = cv2.medianBlur(mask, 5)
        
        # Morphological CLOSE (fills holes inside tube body)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate to bridge small gaps between fragments
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=2)
        
        # Morphological OPEN (removes isolated noise blobs)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        if not self._debug_mask_saved:
            try:
                from pathlib import Path
                debug_path = Path("debug_mask.png")
                cv2.imwrite(str(debug_path), mask)
                
                # Also save a depth visualization
                depth_m = depth_frame.astype(np.float32) * self._depth_scale
                depth_norm = cv2.normalize(
                    depth_m, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                cv2.imwrite("debug_depth.png", depth_norm)
                
                nonzero_count = cv2.countNonZero(mask)
                logger.debug(
                    f"[DEBUG] Saved debug images: debug_mask.png, debug_depth.png "
                    f"(nonzero pixels in mask: {nonzero_count})"
                )
                self._debug_mask_saved = True
            except Exception as e:
                logger.debug(f"[DEBUG] Failed to save debug images: {e}")
        
        return mask
    
    def extract(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
        capture_mode: str | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Extract bounding box of tube from depth frame.
        
        Supports multiple capture modes:
        - "single_side": Side-view with depth-based filtering (default)
        - "single_top": Top-down view with HoughCircles fallback
        - "multi_top": Top-down with multiple tubes
        
        For "single_top" mode with RGB frame available, uses extract_top_down()
        which employs depth-first approach to locate tube cap among rack holes.
        For side-view mode or when RGB unavailable, uses standard depth contour extraction.
        
        Args:
            depth_frame: Raw depth frame in uint16 format
            rgb_frame: RGB frame (optional, used for "single_top" mode validation)
            capture_mode: Capture mode string ("single_side", "single_top", "multi_top")
                         If None, defaults to "single_side" (side-view)
            
        Returns:
            Tuple of (x, y, width, height) or None if no valid ROI found.
        """
        # Determine capture mode
        if capture_mode is None:
            capture_mode = "single_side"  # Default to side-view if not specified
        
        # Check if using new top-down depth-first approach
        if capture_mode == "single_top" and rgb_frame is not None:
            logger.debug("[extract] Using extract_top_down for single_top mode with RGB validation")
            return self.extract_top_down(rgb_frame, depth_frame)
        elif capture_mode == "single_top":
            # Fallback to extract_top if no RGB frame provided
            logger.debug("[extract] single_top mode but no RGB frame, falling back to extract_top")
            return self.extract_top(depth_frame)
        
        # ─────────────────────────────────────────────────────────────
        # DEFAULT: SINGLE-SIDE EXTRACTION (existing logic)
        # ─────────────────────────────────────────────────────────────
        
        # Step 1 — Preprocessing
        depth_min = self.cfg.camera.depth_min_m
        depth_max = self.cfg.camera.depth_max_m
        mask = self._preprocess_depth(depth_frame, depth_min, depth_max)
        
        # Step 2 — Find contours (RETR_EXTERNAL only)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found in depth frame")
            return None
        
        logger.debug(f"[extract] findContours found {len(contours)} raw contours")
        
        # Step 3 — Apply filters to each contour
        candidates = []
        for contour in contours:
            # Filter A — Area gate
            area = cv2.contourArea(contour)
            if not (self.cfg.pipeline.min_roi_area_px <= area <= self.cfg.pipeline.max_roi_area_px):
                logger.debug(
                    f"[extract] REJECT area={area:.0f}px² "
                    f"(allowed [{self.cfg.pipeline.min_roi_area_px}, "
                    f"{self.cfg.pipeline.max_roi_area_px}])"
                )
                continue
            
            # Filter B — Use minAreaRect for orientation-invariant dimensions
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            
            short_dim = min(rw, rh)
            long_dim = max(rw, rh)
            
            # Filter C — Dimension validation
            if not (self.cfg.pipeline.min_tube_dim_px <= short_dim <= self.cfg.pipeline.max_tube_dim_px):
                logger.debug(
                    f"[extract] REJECT short_dim={short_dim:.1f}px "
                    f"(allowed [{self.cfg.pipeline.min_tube_dim_px}, "
                    f"{self.cfg.pipeline.max_tube_dim_px}])"
                )
                continue
            if not (self.cfg.pipeline.min_tube_length_px <= long_dim <= self.cfg.pipeline.max_tube_length_px):
                logger.debug(
                    f"[extract] REJECT long_dim={long_dim:.1f}px "
                    f"(allowed [{self.cfg.pipeline.min_tube_length_px}, "
                    f"{self.cfg.pipeline.max_tube_length_px}])"
                )
                continue
            
            # Filter D — TRUE solidity using convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            
            if solidity < self.cfg.pipeline.min_solidity:
                logger.debug(
                    f"[extract] REJECT solidity={solidity:.3f} "
                    f"(min {self.cfg.pipeline.min_solidity})"
                )
                continue
            
            # Filter E — Border margin check
            box_pts = cv2.boxPoints(rect).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box_pts)
            m = self.cfg.pipeline.border_margin_px
            H, W = mask.shape
            if x < m or y < m or x + w > W - m or y + h > H - m:
                logger.debug(
                    f"[extract] REJECT border_margin: bbox=({x},{y},{w},{h}) "
                    f"frame=({W}x{H}) "
                    f"margin={m}"
                )
                continue
            
            logger.debug(
                f"[extract] PASS contour: bbox=({x},{y},{w},{h}) "
                f"short={short_dim:.1f} long={long_dim:.1f} "
                f"solidity={solidity:.3f} area={area:.0f}"
            )
            
            candidates.append((contour, x, y, w, h, solidity, short_dim, long_dim))
        
        # Step 4 — Select best candidate (highest true solidity)
        if not candidates:
            logger.debug(
                f"[extract] 0 candidates after all filters "
                f"from {len(contours)} raw contours"
            )
            logger.warning("No contours passed all filters")
            return None
        
        best = max(candidates, key=lambda c: c[-3])  # sort by solidity
        _, x, y, w, h, solidity, short_dim, long_dim = best
        
        # Step 5 — Return with debug log
        logger.debug(
            f"[single_side] ROI: ({x},{y},{w},{h})  solidity={solidity:.3f}  "
            f"long={long_dim:.0f}px  short={short_dim:.0f}px"
        )
        
        return (x, y, w, h)
    
    def extract_top(self, depth_frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Extract bounding box of tube from vertical side view.
        
        Strategy to isolate glass tube from yellow plastic rack:
        1. Depth-first: Filter to close range [0.05m, 0.35m] isolating tube tip
        2. Geometry: Cap is small (r 8-20px), reject area > 2000px²
        3. Expansion: 3x cap radius downward + 10px side padding
        
        Args:
            depth_frame: Raw depth frame in uint16 format
            
        Returns:
            Tuple of (x, y, width, height) or None if no valid ROI found
        """
        # DEPTH-FIRST SELECTION: Isolate tube from rack
        # Tube is closest object; rack is behind it
        # Filter to depth range [0.05m, 0.35m] to isolate tube tip/cap
        tube_depth_min = 0.05  # 5cm (closest to camera)
        tube_depth_max = 0.35  # 35cm (tube extends to here)
        
        # Create tight depth mask for tube region only
        tube_mask = self._create_depth_mask(depth_frame, tube_depth_min, tube_depth_max)
        
        # Apply light morphology to clean noise (median blur only, no closing)
        tube_mask = cv2.medianBlur(tube_mask, 5)
        
        logger.debug(
            f"[extract_top] Tube depth filter: [{tube_depth_min:.2f}m, {tube_depth_max:.2f}m] "
            f"nonzero pixels: {cv2.countNonZero(tube_mask)}"
        )
        
        # ─────────────────────────────────────────────────
        # PRIMARY PATH — HoughCircles on tube-only region
        # ─────────────────────────────────────────────────
        
        circles = cv2.HoughCircles(
            tube_mask,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=8,      # Tube cap min radius (pixels)
            maxRadius=20      # Tube cap max radius (pixels)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            candidates = []
            
            for (cx, cy, r) in circles:
                # GEOMETRY FILTER: Cap should be small (~8-20px radius)
                # Reject large circles (likely the rack background noise)
                cap_area = np.pi * r * r
                if cap_area > 2000:  # Max ~25px radius, too large for cap
                    logger.debug(f"[extract_top] REJECT circle: area={cap_area:.0f}px² > 2000 (likely rack)")
                    continue
                
                # POSITION FILTER: Accept only upper half of frame (cap location)
                frame_h = tube_mask.shape[0]
                if cy > frame_h * 0.6:  # Cap should be in top 60% of frame
                    logger.debug(f"[extract_top] REJECT circle: cy={cy} outside upper region (frame_h={frame_h})")
                    continue
                
                logger.debug(f"[extract_top] HoughCircles candidate: center=({cx},{cy}) radius={r} area={cap_area:.0f}px²")
                candidates.append((cx, cy, r))
            
            if candidates:
                # Select smallest valid circle (tube cap, not noise)
                # Smaller = more likely the actual tube cap
                best_cx, best_cy, best_r = min(candidates, key=lambda c: np.pi * c[2] * c[2])
                
                # BBOX EXPANSION: Capture full tube body from cap
                # Expand downward by 3x the cap radius to fully contain tube shaft
                x = int(best_cx - best_r)
                y = int(best_cy - best_r)  # Top of cap
                w = int(2 * best_r)
                h = int(2 * best_r + 3 * best_r)  # Original height + 3x expansion downward
                
                # Add 10px padding on left/right (less on bottom to avoid rack)
                pad_lr = 10
                x -= pad_lr
                y -= 5  # Small top padding
                w += 2 * pad_lr
                h += 5  # Small bottom padding
                
                # Clamp to image bounds
                H, W = tube_mask.shape
                x = max(0, x)
                y = max(0, y)
                w = min(w, W - x)
                h = min(h, H - y)
                
                logger.debug(
                    f"[extract_top] HoughCircles ROI (tube-focused): "
                    f"cap_center=({best_cx},{best_cy}) cap_radius={best_r} "
                    f"final_bbox=({x},{y},{w},{h})"
                )
                
                return (x, y, w, h)
        
        # ─────────────────────────────────────────────────
        # FALLBACK PATH — Contour-based on tube region
        # ─────────────────────────────────────────────────
        
        logger.debug("[extract_top] HoughCircles found nothing — falling back to contour on tube region")
        
        contours, _ = cv2.findContours(
            tube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found in tube region")
            return None
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # GEOMETRY FILTER: Accept only small contours (tube, not rack)
            if area > 2000:  # Reject large contours (likely rack)
                logger.debug(f"[extract_top] REJECT contour area={area:.0f}px² > 2000 (likely rack)")
                continue
            
            if not (self.cfg.pipeline.top_min_roi_area_px <= area <= self.cfg.pipeline.top_max_roi_area_px):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            short_dim = min(w, h)
            long_dim = max(w, h)
            
            if not (self.cfg.pipeline.top_min_tube_dim_px <= short_dim <= self.cfg.pipeline.top_max_tube_dim_px):
                continue
            
            circularity_ratio = long_dim / (short_dim + 1e-6)
            if circularity_ratio > self.cfg.pipeline.top_max_circularity_ratio:
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < self.cfg.pipeline.top_min_solidity:
                continue
            
            m = self.cfg.pipeline.top_border_margin_px
            H, W = tube_mask.shape
            if x < m or y < m or x + w > W - m or y + h > H - m:
                continue
            
            candidates.append((x, y, w, h, solidity))
        
        if not candidates:
            logger.warning("No contours passed all filters in tube region")
            return None
        
        best = max(candidates, key=lambda c: c[-1])
        x, y, w, h, _ = best
        logger.debug(f"[extract_top] Contour fallback ROI (tube-focused): ({x},{y},{w},{h})")
        return (x, y, w, h)
    
    def extract_top_down(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        """Extract bounding box of tube from top-down view using depth-first approach.
        
        Camera points straight DOWN at yellow rack with 50+ holes. Tube cap protrudes
        above rack surface, making it the closest (minimum depth) object. This method
        isolates the tube cap from rack holes using depth information, then validates
        with RGB color to reject uniform holes.
        
        Algorithm:
        1. Find minimum depth region: Filter depth [0.05m, 0.30m], find min depth,
           create mask within 0.015m of minimum (isolates tube cap from rack)
        2. Clean depth mask: Morphological opening (5×5), find contours,
           filter by area 100-3000px², take contour with minimum mean depth
        3. Validate with RGB: Check color variance < 15 (reject uniform yellow holes)
        4. Return bbox: Bounding rect + 15px padding, clamped to image bounds
        
        Args:
            rgb_frame: RGB image array (for color validation)
            depth_frame: Raw depth frame in uint16 format
            
        Returns:
            Tuple of (x, y, width, height) or None if no valid ROI found
        """
        # ─────────────────────────────────────────────────────────────
        # STEP 1: FIND MINIMUM DEPTH REGION (isolates tube cap)
        # ─────────────────────────────────────────────────────────────
        
        depth_min_m = 0.05    # 5cm (closest to camera)
        depth_max_m = 0.30    # 30cm (far from rack)
        depth_margin_m = 0.015  # 15mm tolerance around minimum depth
        
        # Convert depth frame to meters
        depth_m = depth_frame.astype(np.float32) * self._depth_scale
        
        # Create basic depth filter mask [0.05m, 0.30m]
        depth_filter = (depth_m >= depth_min_m) & (depth_m <= depth_max_m)
        
        if not depth_filter.any():
            logger.warning(
                f"[extract_top_down] No pixels in depth range [{depth_min_m:.2f}m, {depth_max_m:.2f}m]"
            )
            return None
        
        # Find minimum depth value within filtered region
        depth_filtered = depth_m.copy()
        depth_filtered[~depth_filter] = depth_max_m + 1  # Exclude out-of-range pixels
        min_depth = depth_filtered.min()
        
        # Create mask: pixels within depth_margin of minimum (isolates tube cap)
        cap_mask = np.abs(depth_m - min_depth) <= depth_margin_m
        cap_mask = cap_mask.astype(np.uint8) * 255
        
        logger.debug(
            f"[extract_top_down] Minimum depth: {min_depth:.4f}m, margin={depth_margin_m:.4f}m, "
            f"cap_mask nonzero pixels: {cv2.countNonZero(cap_mask)}"
        )
        
        if cv2.countNonZero(cap_mask) < 50:
            logger.warning("[extract_top_down] Too few pixels in cap mask (likely no tube)")
            return None
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: CLEAN DEPTH MASK (remove noise, find contours)
        # ─────────────────────────────────────────────────────────────
        
        # Morphological opening (5×5 kernel) to remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(cap_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("[extract_top_down] No contours found after morphological opening")
            return None
        
        logger.debug(f"[extract_top_down] Found {len(contours)} contours after opening")
        
        # Filter contours by area [100px², 3000px²]
        min_area_px = 100
        max_area_px = 3000
        
        candidates = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if not (min_area_px <= area <= max_area_px):
                logger.debug(
                    f"[extract_top_down] REJECT contour {i}: area={area:.0f}px² "
                    f"outside [{min_area_px}, {max_area_px}]"
                )
                continue
            
            # Compute mean depth of this contour region
            mask_contour = np.zeros_like(clean_mask, dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], 0, 255, -1)
            
            contour_depths = depth_m[mask_contour == 255]
            mean_depth = contour_depths.mean()
            
            logger.debug(
                f"[extract_top_down] Contour {i}: area={area:.0f}px², "
                f"mean_depth={mean_depth:.4f}m"
            )
            
            candidates.append((i, contour, area, mean_depth))
        
        if not candidates:
            logger.warning("[extract_top_down] No contours passed area filter")
            return None
        
        # Select contour with minimum mean depth (closest to camera = tube cap)
        best_idx, best_contour, best_area, best_depth = min(
            candidates, key=lambda c: c[3]  # sort by mean_depth
        )
        
        logger.debug(
            f"[extract_top_down] Selected contour {best_idx}: "
            f"area={best_area:.0f}px², min_depth={best_depth:.4f}m"
        )
        
        # Get bounding rect
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # ─────────────────────────────────────────────────────────────
        # STEP 3: VALIDATE WITH RGB (reject uniform yellow holes)
        # ─────────────────────────────────────────────────────────────
        
        # Crop RGB with 20px padding
        pad_validate = 20
        x_crop = max(0, x - pad_validate)
        y_crop = max(0, y - pad_validate)
        x_crop_end = min(rgb_frame.shape[1], x + w + pad_validate)
        y_crop_end = min(rgb_frame.shape[0], y + h + pad_validate)
        
        rgb_crop = rgb_frame[y_crop:y_crop_end, x_crop:x_crop_end]
        
        if rgb_crop.size == 0:
            logger.warning("[extract_top_down] RGB crop is empty, rejecting")
            return None
        
        # Compute color variance in RGB crop
        if len(rgb_crop.shape) == 3 and rgb_crop.shape[2] == 3:
            # Compute per-channel variance and take average
            r_var = rgb_crop[:, :, 0].var()
            g_var = rgb_crop[:, :, 1].var()
            b_var = rgb_crop[:, :, 2].var()
            color_variance = (r_var + g_var + b_var) / 3.0
        else:
            # Grayscale
            color_variance = rgb_crop.var()
        
        logger.debug(
            f"[extract_top_down] RGB validation: crop_shape={rgb_crop.shape}, "
            f"color_variance={color_variance:.2f}"
        )
        
        # Threshold: uniform yellow hole has variance < 15, tube cap has more variation
        variance_threshold = 15
        if color_variance < variance_threshold:
            logger.warning(
                f"[extract_top_down] REJECT: color_variance={color_variance:.2f} < {variance_threshold} "
                f"(likely uniform yellow hole)"
            )
            return None
        
        logger.debug(
            f"[extract_top_down] RGB validation PASS: variance={color_variance:.2f} >= {variance_threshold}"
        )
        
        # ─────────────────────────────────────────────────────────────
        # STEP 4: RETURN BBOX WITH PADDING
        # ─────────────────────────────────────────────────────────────
        
        # Add 15px padding on all sides
        pad_bbox = 15
        x_out = max(0, x - pad_bbox)
        y_out = max(0, y - pad_bbox)
        w_out = w + 2 * pad_bbox
        h_out = h + 2 * pad_bbox
        
        # Clamp to image bounds
        H, W = rgb_frame.shape[:2]
        x_out = max(0, min(x_out, W - 1))
        y_out = max(0, min(y_out, H - 1))
        w_out = min(w_out, W - x_out)
        h_out = min(h_out, H - y_out)
        
        logger.info(
            f"[extract_top_down] ROI extracted: ({x_out},{y_out},{w_out},{h_out}) "
            f"(original contour: ({x},{y},{w},{h}), depth={best_depth:.4f}m, "
            f"rgb_variance={color_variance:.2f})"
        )
        
        return (x_out, y_out, w_out, h_out)
    
    def extract_multi_top(self, depth_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Extract bounding boxes for ALL tubes from top-down view (multi_top mode).
        
        For rack-mounted tubes viewed from above, detecting multiple tubes
        in the same frame. Uses watershed to separate touching blobs,
        then applies HoughCircles to each region with contour fallback.
        
        Args:
            depth_frame: Raw depth frame in uint16 format
            
        Returns:
            List of (x, y, width, height) tuples sorted by x coordinate.
            Empty list if no valid ROIs found.
        """
        # Step 1 — Preprocessing
        depth_min = self.cfg.pipeline.top_depth_min_m
        depth_max = self.cfg.pipeline.top_depth_max_m
        mask = self._preprocess_depth(depth_frame, depth_min, depth_max)
        
        # Step 2 — Watershed to separate touching blobs
        
        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        
        # Sure foreground: pixels far from any edge (centers of tubes)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Sure background: dilated mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # Unknown region (border between fg and bg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # background → 1 (not 0)
        markers[unknown == 255] = 0  # unknown → 0 (let watershed decide)
        
        # Apply watershed (needs 3-channel image)
        rgb_proxy = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(rgb_proxy, markers)
        
        # Each unique label > 1 is a separated tube region
        unique_labels = np.unique(markers)
        tube_labels = [l for l in unique_labels if l > 1]  # skip bg (1) and border (-1)
        
        if not tube_labels:
            logger.warning("Watershed found no separated regions")
            return []
        
        # Step 3 — For each separated region, run HoughCircles then contour fallback
        
        all_rois = []
        
        for label in tube_labels:
            # Extract single region mask
            region_mask = np.zeros_like(mask, dtype=np.uint8)
            region_mask[markers == label] = 255
            
            # Skip tiny regions (noise)
            if cv2.countNonZero(region_mask) < self.cfg.pipeline.top_min_roi_area_px:
                continue
            
            # PRIMARY: HoughCircles on this region
            circles = cv2.HoughCircles(
                region_mask,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=20,
                param1=50,
                param2=25,
                minRadius=int(self.cfg.pipeline.top_min_tube_dim_px // 2),
                maxRadius=int(self.cfg.pipeline.top_max_tube_dim_px // 2)
            )
            
            if circles is not None:
                cx, cy, r = np.uint16(np.around(circles[0][0]))
                # Expand bbox: capture full tube body (cap + shaft)
                # Expand downward by 2.5x the radius
                x = int(cx - r)
                y = int(cy - r)  # Top of cap
                w = int(2 * r)
                h = int(2 * r + 2.5 * r)  # Original height + 2.5x expansion downward
                
                # Add 15px padding on all sides
                pad = 15
                x -= pad
                y -= pad
                w += 2 * pad
                h += 2 * pad
                
                # Clamp to image bounds
                H, W = mask.shape
                x = max(0, x)
                y = max(0, y)
                w = min(w, W - x)
                h = min(h, H - y)
            else:
                # FALLBACK: bounding rect of this region's contour
                contours, _ = cv2.findContours(
                    region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            
            # Border margin check
            m = self.cfg.pipeline.top_border_margin_px
            H, W = mask.shape
            if x < m or y < m or x + w > W - m or y + h > H - m:
                continue
            
            all_rois.append((x, y, w, h))
        
        # Sort left-to-right by x coordinate
        all_rois.sort(key=lambda r: r[0])
        
        logger.debug(
            f"[multi_top] Watershed separated {len(tube_labels)} regions → "
            f"{len(all_rois)} valid ROIs: {all_rois}"
        )
        
        return all_rois
