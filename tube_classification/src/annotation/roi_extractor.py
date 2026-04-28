import numpy as np
import cv2

from loguru import logger
from config.parser import get_config


def detect_depth_inversion(
    depth_frame: np.ndarray,
    depth_near_m: float,
    depth_far_m: float,
    depth_scale: float = 0.001,
) -> tuple[bool, float, float]:
    """Detect if holder is closer to camera than tube (depth-gate inversion).

    At steep/elevated camera angles the holder surface can appear as the
    NEAREST object — its depth-gate blob is wide and flat (low aspect ratio)
    rather than tall and narrow like a tube.  When that signature is detected
    the function returns adjusted depth bounds that reach the tube's actual depth.

    Returns:
        (is_inverted, adjusted_near_m, adjusted_far_m)
    """
    scale = 1.0 / depth_scale
    near_mm = depth_near_m * scale
    far_mm = depth_far_m * scale

    gate_mask = np.zeros(depth_frame.shape, dtype=np.uint8)
    gate_mask[(depth_frame >= near_mm) & (depth_frame <= far_mm)] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gate_mask, connectivity=8)
    if num_labels < 2:
        return False, depth_near_m, depth_far_m

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    lw = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    lh = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    ly = int(stats[largest_idx, cv2.CC_STAT_TOP])
    frame_h = depth_frame.shape[0]

    blob_aspect = lh / max(lw, 1)
    blob_bottom_frac = (ly + lh) / frame_h

    # Holder signature: wide blob that extends toward bottom of frame
    is_holder_dominant = (blob_aspect < 0.6) and (blob_bottom_frac > 0.55)
    if not is_holder_dominant:
        return False, depth_near_m, depth_far_m

    # Tube lives ABOVE and FURTHER than the holder — search that strip
    holder_top_y = ly
    search_top = max(0, holder_top_y - 80)
    search_region = depth_frame[search_top:holder_top_y, :]
    valid_depths = search_region[(search_region > near_mm) & (search_region < 65000)]

    if len(valid_depths) > 20:
        tube_depth_est = float(np.median(valid_depths))
        new_near = min(near_mm, tube_depth_est - 50) / scale
        new_far = max(far_mm, tube_depth_est + 80) / scale
        return True, new_near, new_far

    return True, depth_near_m, min(depth_far_m + 0.15, 0.70)


def create_depth_mask_dual_zone(
    depth_frame: np.ndarray,
    tube_near_mm: float,
    tube_far_mm: float,
    holder_near_mm: float,
    holder_far_mm: float,
) -> np.ndarray:
    """Build a binary depth mask that spans the tube zone but subtracts the holder zone.

    Used when depth inversion is detected: the holder occupies the near depth
    band while the tube sits further away.  By dilating the holder zone before
    subtraction we also erode the shared boundary pixels.

    Returns:
        uint8 mask (0 or 255), shape (H, W).
    """
    combined = (depth_frame >= tube_near_mm) & (depth_frame <= tube_far_mm)
    holder_zone = (depth_frame >= holder_near_mm) & (depth_frame <= holder_far_mm)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    holder_zone_dilated = cv2.dilate(holder_zone.astype(np.uint8), kernel)

    tube_only = combined.astype(np.uint8) & (~holder_zone_dilated.astype(bool)).astype(np.uint8)
    return (tube_only * 255).astype(np.uint8)


def find_holder_top_y(
    depth_frame: np.ndarray,
    depth_near_mm: float,
    depth_far_mm: float,
    frame_width: int,
) -> int | None:
    """Find the top-edge Y coordinate of the holder blob in the near depth zone.

    When the holder is closer than the tube the holder's top edge in image space
    marks where the tube body ends.  Returns None when no holder-like blob is
    found.
    """
    holder_mask = (
        (depth_frame >= depth_near_mm) & (depth_frame <= depth_far_mm)
    ).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holder_mask, connectivity=8)
    if num_labels < 2:
        return None

    best_label = -1
    best_score = -1.0
    for i in range(1, num_labels):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if w > frame_width * 0.15 and area > 5000 and (w / max(h, 1)) > 1.2:
            score = float(w * area)
            if score > best_score:
                best_score = score
                best_label = i

    if best_label == -1:
        return None

    holder_top = int(stats[best_label, cv2.CC_STAT_TOP])
    return holder_top + 5


def detect_camera_orientation(depth_frame: np.ndarray) -> str:
    """Infer camera tilt from the vertical depth gradient of the center column strip.

    In side-view the scene depth is nearly uniform across rows (tube sits upright,
    background is flat).  In angled/top-down view the camera tilt means depth
    increases monotonically from top to bottom — producing a large row-wise depth
    range across the frame center.

    Thresholds in raw mm units (depth_scale=0.001, so 1 unit ≈ 1 mm).

    Returns:
        'side' | 'angled' | 'top'
    """
    h, w = depth_frame.shape
    col_start, col_end = w // 4, 3 * w // 4
    center_strip = depth_frame[:, col_start:col_end].astype(float)
    center_strip[center_strip < 100] = np.nan
    center_strip[center_strip > 65000] = np.nan

    row_medians = np.nanmedian(center_strip, axis=1)
    valid_rows = row_medians[~np.isnan(row_medians)]

    if len(valid_rows) < 10:
        return 'side'

    depth_range = float(np.nanmax(row_medians) - np.nanmin(row_medians))

    if depth_range > 350:
        return 'top'
    if depth_range > 180:
        return 'angled'
    return 'side'


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
        self._debug_nearest_saved = False
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._orientation: str = 'side'
        self._is_inverted: bool = False

    @property
    def orientation(self) -> str:
        """Camera orientation detected on the most recent frame."""
        return self._orientation

    @property
    def is_inverted(self) -> bool:
        """True when depth-gate inversion was detected on the most recent frame."""
        return self._is_inverted

    def _nearest_object_mask(
        self,
        depth_frame: np.ndarray,
        band_m: float = 0.04
    ) -> np.ndarray:
        """Build a binary mask around the nearest valid depth band."""
        mm_min = int(0.05 / self._depth_scale)
        mm_max = int(1.0 / self._depth_scale)
        valid_mask = (depth_frame > mm_min) & (depth_frame < mm_max)

        if int(np.count_nonzero(valid_mask)) < 500:
            logger.warning("_nearest_object_mask: insufficient valid depth pixels")
            return np.zeros(depth_frame.shape, dtype=np.uint8)

        nearest_depth_mm = float(np.percentile(depth_frame[valid_mask], 2))
        nearest_depth_m = nearest_depth_mm * self._depth_scale
        logger.debug(
            f"Nearest object depth: {nearest_depth_m:.4f}m, "
            f"band=[{nearest_depth_m:.4f}, {nearest_depth_m + band_m:.4f}]m"
        )
        if nearest_depth_m > 0.28:
            logger.warning(f"Nearest object depth appears far: {nearest_depth_m:.4f}m")

        band_mm = int(band_m / self._depth_scale)
        nearest_mask = (
            (depth_frame >= int(nearest_depth_mm))
            & (depth_frame <= int(nearest_depth_mm + band_mm))
            & valid_mask
        ).astype(np.uint8) * 255

        if not self._debug_nearest_saved:
            try:
                from pathlib import Path
                depth_norm = cv2.normalize(
                    depth_frame, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
                mask_vis = cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR)
                debug_combo = cv2.hconcat([depth_vis, mask_vis])
                cv2.imwrite(str(Path("debug_nearest_mask.png")), debug_combo)
                self._debug_nearest_saved = True
            except Exception as e:
                logger.debug(f"[DEBUG] Failed to save nearest debug image: {e}")

        return nearest_mask
    
    def _preprocess_depth(
        self,
        depth_frame: np.ndarray,
        depth_min: float,
        depth_max: float,
        initial_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Clean depth frame and return binary mask for valid depth range.

        Applies depth-to-meter conversion, in-range masking, median blur
        (salt-pepper noise removal), and morphological close/open operations.

        Args:
            depth_frame: Raw depth frame (uint16)
            depth_min: Minimum valid depth (meters)
            depth_max: Maximum valid depth (meters)
            initial_mask: Optional pre-computed binary mask (0/255).  When
                provided, skips the in-range computation and uses this mask
                directly (used for dual-zone depth-inversion mode).

        Returns:
            Binary uint8 mask (0 or 255), shape (H, W)
        """
        if initial_mask is not None:
            # Dual-zone path: caller already computed the correct binary mask
            mask = initial_mask.copy()
        else:
            depth_min_mm = int(depth_min / self._depth_scale)
            depth_max_mm = int(depth_max / self._depth_scale)
            # Build in-range mask in uint16 space to avoid large float allocations
            in_range = ((depth_frame >= depth_min_mm) & (depth_frame <= depth_max_mm))
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

        # Remove holder/turntable-like components connected to the bottom edge.
        # These are typically wide blobs that dominate ROI selection.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        H, W = mask.shape
        cleaned = mask.copy()
        for lbl in range(1, num_labels):
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            touches_bottom = (y + h) >= (H - 2)
            width_ratio = w / float(W)
            area_ratio = area / float(H * W)

            remove_bottom_holder = (
                touches_bottom
                and (
                    width_ratio >= 0.06
                    or area_ratio >= 0.003
                    or (h >= int(0.06 * H) and width_ratio >= 0.045)
                )
            )
            if remove_bottom_holder:
                cleaned[labels == lbl] = 0

        mask = cleaned
        
        if not self._debug_mask_saved:
            try:
                from pathlib import Path
                debug_path = Path("debug_mask.png")
                cv2.imwrite(str(debug_path), mask)
                
                # Also save a depth visualization
                depth_norm = cv2.normalize(
                    depth_frame, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                if bool(getattr(self.cfg.pipeline, "depth_preview_invert", True)):
                    cv2.bitwise_not(depth_norm, dst=depth_norm)
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

    def _single_side_param(self, key: str, default: float) -> float:
        return float(getattr(self.cfg.pipeline, key, default))

    def _is_tray_like(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        area: float,
        solidity: float,
        frame_shape: tuple[int, int]
    ) -> bool:
        frame_h, frame_w = frame_shape
        area_ratio = area / float(frame_h * frame_w)
        width_ratio = w / float(frame_w)
        tray_max_area_ratio = self._single_side_param("single_side_tray_max_area_ratio", 0.22)
        max_roi_area_ratio = self._single_side_param("single_side_max_roi_area_ratio", 0.012)
        tray_max_width_ratio = self._single_side_param("single_side_tray_max_width_ratio", 0.55)
        tray_min_solidity = self._single_side_param("single_side_tray_min_solidity", 0.92)
        bottom_tray_y_ratio = self._single_side_param("single_side_bottom_tray_y_ratio", 0.90)
        bottom_tray_min_area_ratio = self._single_side_param("single_side_bottom_tray_min_area_ratio", 0.015)
        is_bottom_large = ((y + h) >= (frame_h * bottom_tray_y_ratio)) and (area_ratio >= bottom_tray_min_area_ratio)
        return (
            (area_ratio >= max_roi_area_ratio)
            or
            (area_ratio >= tray_max_area_ratio and solidity >= tray_min_solidity)
            or (width_ratio >= tray_max_width_ratio and solidity >= tray_min_solidity)
            or (is_bottom_large and solidity >= (tray_min_solidity * 0.9))
        )

    def _is_holder_like_candidate(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_shape: tuple[int, int],
    ) -> bool:
        """Hard-reject low wide blobs that match holder/turntable geometry."""
        frame_h, frame_w = frame_shape
        width_ratio = w / float(max(frame_w, 1))
        y_ratio = y / float(max(frame_h, 1))
        bottom_ratio = (y + h) / float(max(frame_h, 1))
        aspect_wh = w / float(max(h, 1))

        # Require width > 2.5× height to avoid false-positives on large-tube holders,
        # which are proportionally taller than turntable bases.
        if y_ratio >= 0.62 and width_ratio >= 0.10 and aspect_wh >= 2.5:
            return True
        if bottom_ratio >= 0.93 and width_ratio >= 0.07 and aspect_wh >= 2.5:
            return True
        if y_ratio >= 0.70 and width_ratio >= 0.06 and aspect_wh >= 2.5:
            return True
        return False

    def _candidate_score(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        short_dim: float,
        long_dim: float,
        solidity: float,
        frame_shape: tuple[int, int]
    ) -> float:
        frame_h, frame_w = frame_shape
        cx = x + (w * 0.5)
        cy = y + (h * 0.5)
        frame_cx = frame_w * 0.5
        frame_cy = frame_h * 0.5
        center_dist = np.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
        center_norm = center_dist / (np.sqrt(frame_w ** 2 + frame_h ** 2) + 1e-6)
        center_score = max(0.0, 1.0 - center_norm)

        aspect = long_dim / (short_dim + 1e-6)
        aspect_score = max(0.0, 1.0 - min(abs(aspect - 2.0) / 2.0, 1.0))
        solidity_score = max(0.0, min(1.0, solidity))
        area = float(max(w * h, 1))
        min_area = self._single_side_param("single_side_min_roi_area_px", 900.0)
        area_scale = max(min_area * 1.5, 1.0)
        area_score = max(0.0, min(1.0, (area - min_area) / area_scale))
        min_short = self._single_side_param(
            "single_side_min_tube_dim_px",
            float(self.cfg.pipeline.min_tube_dim_px),
        )
        min_long = self._single_side_param(
            "single_side_min_tube_length_px",
            float(self.cfg.pipeline.min_tube_length_px),
        )
        short_scale = max(min_short * 1.0, 1.0)
        long_scale = max(min_long * 1.0, 1.0)
        short_size_score = max(0.0, min(1.0, (short_dim - min_short) / short_scale))
        long_size_score = max(0.0, min(1.0, (long_dim - min_long) / long_scale))
        size_score = 0.5 * short_size_score + 0.5 * long_size_score
        shape_score = (0.32 * aspect_score) + (0.28 * solidity_score) + (0.20 * size_score) + (0.20 * area_score)

        temporal_score = 0.5
        if self._last_bbox is not None:
            lx, ly, lw, lh = self._last_bbox
            lcx = lx + (lw * 0.5)
            lcy = ly + (lh * 0.5)
            tdist = np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)
            tnorm = tdist / (np.sqrt(frame_w ** 2 + frame_h ** 2) + 1e-6)
            last_area = float(max(lw * lh, 1))
            area = float(max(w * h, 1))
            area_consistency = min(last_area, area) / max(last_area, area)
            temporal_score = max(0.0, 1.0 - tnorm) * 0.7 + area_consistency * 0.3

        w_shape = self._single_side_param("single_side_score_shape_weight", 0.55)
        w_center = self._single_side_param("single_side_score_center_weight", 0.25)
        w_temporal = self._single_side_param("single_side_score_temporal_weight", 0.20)
        total_w = max(w_shape + w_center + w_temporal, 1e-6)
        score = (
            (w_shape * shape_score)
            + (w_center * center_score)
            + (w_temporal * temporal_score)
        ) / total_w
        # Suppress persistent false positives from lower tray/turntable region.
        # At angled view the tube appears lower in frame — use more lenient thresholds.
        if self._orientation == 'angled':
            if cy > (frame_h * 0.93):
                score *= 0.55
            elif cy > (frame_h * 0.88):
                score *= 0.75
        else:
            if cy > (frame_h * 0.90):
                score *= 0.45
            elif cy > (frame_h * 0.85):
                score *= 0.65
            elif cy > (frame_h * 0.80):
                score *= 0.82

        return float(score)

    @staticmethod
    def _bbox_overlap_ratio(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        ix0 = max(ax, bx)
        iy0 = max(ay, by)
        ix1 = min(ax + aw, bx + bw)
        iy1 = min(ay + ah, by + bh)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = float(iw * ih)
        area_a = float(max(aw * ah, 1))
        return inter / area_a

    def _refine_with_candidate_depth(
        self,
        depth_frame: np.ndarray,
        contour: np.ndarray,
        fallback_bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        dist_min = self._single_side_param("single_side_depth_min_m", 0.18)
        dist_max = self._single_side_param("single_side_depth_max_m", 0.40)
        dist_min_mm = int(dist_min / self._depth_scale)
        dist_max_mm = int(dist_max / self._depth_scale)

        contour_mask = np.zeros(depth_frame.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        valid = (depth_frame >= dist_min_mm) & (depth_frame <= dist_max_mm) & (depth_frame > 0)
        contour_valid = valid & (contour_mask > 0)
        if int(np.count_nonzero(contour_valid)) < 100:
            return fallback_bbox

        p_low = self._single_side_param("single_side_depth_band_low_percentile", 20.0)
        p_high = self._single_side_param("single_side_depth_band_high_percentile", 80.0)
        band_margin = self._single_side_param("single_side_depth_band_margin_m", 0.03)
        values_mm = depth_frame[contour_valid]
        near_mm = float(np.percentile(values_mm, p_low))
        far_mm = float(np.percentile(values_mm, p_high))
        band_margin_mm = float(max(1, int(band_margin / self._depth_scale)))
        band_min_mm = max(float(dist_min_mm), near_mm - band_margin_mm)
        band_max_mm = min(float(dist_max_mm), far_mm + band_margin_mm)
        logger.debug(
            f"[single_side] candidate depth band: median={(np.median(values_mm) * self._depth_scale):.4f}m "
            f"range=[{(band_min_mm * self._depth_scale):.4f}, {(band_max_mm * self._depth_scale):.4f}]m"
        )

        band_mask = (
            (depth_frame >= int(band_min_mm))
            & (depth_frame <= int(band_max_mm))
            & valid
        ).astype(np.uint8) * 255
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        local_support = cv2.dilate(contour_mask, dilate_kernel, iterations=1)
        band_mask = cv2.bitwise_and(band_mask, local_support)
        band_mask = cv2.medianBlur(band_mask, 5)
        band_mask = cv2.morphologyEx(
            band_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        band_mask = cv2.morphologyEx(
            band_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        refined_contours, _ = cv2.findContours(
            band_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not refined_contours:
            return fallback_bbox

        fx, fy, fw, fh = fallback_bbox
        best_refined = None
        best_overlap = -1.0
        for rc in refined_contours:
            rx, ry, rw, rh = cv2.boundingRect(rc)
            ix0 = max(fx, rx)
            iy0 = max(fy, ry)
            ix1 = min(fx + fw, rx + rw)
            iy1 = min(fy + fh, ry + rh)
            iw = max(0, ix1 - ix0)
            ih = max(0, iy1 - iy0)
            overlap = float(iw * ih)
            if overlap > best_overlap:
                best_overlap = overlap
                best_refined = (rx, ry, rw, rh)
        if best_refined is None:
            return fallback_bbox

        rx, ry, rw, rh = best_refined
        fallback_area = float(max(fw * fh, 1))
        refined_area = float(max(rw * rh, 1))
        if refined_area < (0.60 * fallback_area):
            return fallback_bbox
        return best_refined

    def _expand_single_side_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Expand final single-side ROI to avoid tube-tip cropping."""
        x, y, w, h = bbox
        H, W = frame_shape

        short_dim = float(min(w, h))
        long_dim = float(max(w, h))
        target_short = self._single_side_param("single_side_final_min_bbox_short_px", 38.0)
        target_long = self._single_side_param("single_side_final_min_bbox_long_px", 100.0)
        pad_px = int(self._single_side_param("single_side_final_bbox_pad_px", 8.0))

        scale = max(
            target_short / max(short_dim, 1.0),
            target_long / max(long_dim, 1.0),
            1.0
        )

        new_w = int(round(w * scale)) + (2 * pad_px)
        new_h = int(round(h * scale)) + (2 * pad_px)
        cx = x + (w // 2)
        cy = y + (h // 2)

        nx = max(0, cx - (new_w // 2))
        ny = max(0, cy - (new_h // 2))
        nx2 = min(W, nx + new_w)
        ny2 = min(H, ny + new_h)

        final_w = max(1, nx2 - nx)
        final_h = max(1, ny2 - ny)
        return (int(nx), int(ny), int(final_w), int(final_h))

    def _find_tube_bottom_from_depth(
        self,
        depth_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        orientation: str = 'side',
        is_inverted: bool = False,
    ) -> int:
        """Detect tube-to-holder boundary using vertical depth gradient.

        For side view: raw gradient of row-mean depth detects the sharp depth
        discontinuity at the tube-holder junction.

        For angled view: the whole scene tilts so raw depth changes continuously
        along the column. We detrend the linear tilt component first and then
        look for residual spikes (genuine structural discontinuities).

        For inverted+angled: the holder's top edge in the depth image is the
        direct tube-bottom boundary — skip gradient analysis and use it directly.

        Args:
            depth_frame: Raw depth frame (uint16, mm units).
            bbox: Bounding box as (x, y, width, height).
            orientation: 'side' | 'angled' | 'top'
            is_inverted: True when holder is closer to camera than tube.

        Returns:
            y-coordinate of the trimmed bottom boundary (pixels).
            Returns bbox bottom (y + h) when no discontinuity is detected.
        """
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Inverted angled: holder top edge is the most reliable tube-bottom signal
        if orientation == 'angled' and is_inverted:
            depth_near_mm = (
                self._single_side_param("single_side_depth_min_m", self.cfg.camera.depth_min_m)
                / self._depth_scale
            )
            depth_far_mm = (
                self._single_side_param("single_side_depth_max_m", self.cfg.camera.depth_max_m)
                / self._depth_scale
            )
            holder_y = find_holder_top_y(
                depth_frame, depth_near_mm, depth_far_mm, depth_frame.shape[1]
            )
            if holder_y is not None:
                result = min(int(holder_y), y2)
                if result > y1:
                    logger.debug(
                        f"[extract] Inverted-angled holder-top trim: {y2} → {result} "
                        f"(holder top detected at y={holder_y})"
                    )
                    return result
            return y2

        col_center = (x1 + x2) // 2
        col_half = max(1, (x2 - x1) // 8)
        col_left = max(0, col_center - col_half)
        col_right = min(depth_frame.shape[1], col_center + col_half)
        row_start = max(0, y1)
        row_end = min(depth_frame.shape[0], y2)
        if row_end <= row_start or col_right <= col_left:
            return y2
        strip = depth_frame[row_start:row_end, col_left:col_right].astype(np.float64)
        strip[strip == 0] = np.nan
        row_means = np.nanmean(strip, axis=1)

        if orientation == 'side':
            threshold_mm = self._single_side_param("depth_gradient_threshold_mm", 15.0)
            valid = ~np.isnan(row_means)
            if int(valid.sum()) < 4:
                return y2
            grad = np.abs(np.gradient(np.where(valid, row_means, 0.0)))
            hit = np.where((grad > threshold_mm) & valid)[0]
            if len(hit) > 0:
                return min(y1 + int(hit[0]) + 10, y2)
            return y2

        elif orientation == 'angled':
            # Detrend the linear scene-tilt component, then spike-detect residuals.
            threshold_mm = self._single_side_param("angled_depth_gradient_threshold_mm", 22.0)
            n = len(row_means)
            x_idx = np.arange(n, dtype=float)
            valid_mask = ~np.isnan(row_means)
            if valid_mask.sum() < 5:
                return y2
            coeffs = np.polyfit(x_idx[valid_mask], row_means[valid_mask], 1)
            trend = np.polyval(coeffs, x_idx)
            residual = np.abs(np.nan_to_num(row_means) - trend)
            spike_thr = max(threshold_mm, float(np.nanstd(residual)) * 2.5)
            indices = np.where(residual > spike_thr)[0]
            # Ignore top 40% of bbox — the holder is never there
            indices = indices[indices > int(n * 0.4)]
            if len(indices) > 0:
                return min(y1 + int(indices[0]) + 8, y2)
            # No clear discontinuity at angle — trust SAM + neg prompts, don't trim
            return y2

        else:  # top-down
            return y2

    def extract(self, depth_frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Extract bounding box of tube from depth frame (single_side mode).
        
        Converts depth to meters, builds binary mask for valid depth range,
        finds contours, applies orientation-invariant filters using minAreaRect,
        and returns bounding box of best candidate by true solidity.
        
        Args:
            depth_frame: Raw depth frame in uint16 format
            
        Returns:
            Tuple of (x, y, width, height) or None if no valid ROI found
        """
        # Step 0 — Detect (or honour forced) camera orientation once per frame
        forced = getattr(self.cfg.pipeline, 'force_camera_orientation', None)
        self._orientation = forced if forced else detect_camera_orientation(depth_frame)
        orientation = self._orientation
        logger.debug(f"[extract] camera orientation={orientation}")

        # Step 1 — Preprocessing (tube-range first, not global nearest object)
        depth_min = self._single_side_param("single_side_depth_min_m", self.cfg.camera.depth_min_m)
        depth_max = self._single_side_param("single_side_depth_max_m", self.cfg.camera.depth_max_m)

        # Detect depth-gate inversion (angled views only: holder can be nearer than tube)
        self._is_inverted = False
        initial_mask: np.ndarray | None = None
        if orientation == 'angled':
            is_inverted, adj_near, adj_far = detect_depth_inversion(
                depth_frame, depth_min, depth_max, self._depth_scale
            )
            self._is_inverted = is_inverted
            if is_inverted:
                logger.info(
                    f"[extract] Depth inversion detected — holder is closer than tube. "
                    f"Adjusted depth gate: {adj_near:.3f}m–{adj_far:.3f}m "
                    f"(original: {depth_min:.3f}m–{depth_max:.3f}m). "
                    f"Applying dual-zone subtraction."
                )
                holder_near_mm = depth_min / self._depth_scale
                holder_far_mm = depth_max / self._depth_scale
                tube_near_mm = adj_near / self._depth_scale
                tube_far_mm = adj_far / self._depth_scale
                initial_mask = create_depth_mask_dual_zone(
                    depth_frame, tube_near_mm, tube_far_mm, holder_near_mm, holder_far_mm
                )
        is_inverted = self._is_inverted

        # Select filter profile based on orientation
        if orientation == 'angled':
            min_area_px = float(getattr(self.cfg.pipeline, 'angled_min_roi_area_px', 600))
            min_short_px = float(getattr(self.cfg.pipeline, 'angled_min_tube_dim_px', 20))
            min_long_px = float(getattr(self.cfg.pipeline, 'angled_min_tube_length_px', 40))
        else:
            min_area_px = self._single_side_param(
                "single_side_min_roi_area_px",
                max(float(self.cfg.pipeline.min_roi_area_px), 900.0),
            )
            min_short_px = self._single_side_param(
                "single_side_min_tube_dim_px",
                max(float(self.cfg.pipeline.min_tube_dim_px), 24.0),
            )
            min_long_px = self._single_side_param(
                "single_side_min_tube_length_px",
                max(float(self.cfg.pipeline.min_tube_length_px), 55.0),
            )
        mask = self._preprocess_depth(depth_frame, depth_min, depth_max, initial_mask=initial_mask)
        
        # Step 2 — Find contours (RETR_EXTERNAL only)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        used_nearest_fallback = False

        if not contours:
            # Fallback: use nearest-depth band and suppress the bottom strip where
            # tube holder / turntable artifacts are most common.
            nearest_mask = self._nearest_object_mask(depth_frame, band_m=0.06)
            H, W = nearest_mask.shape
            suppress_rows = int(round(H * 0.16))
            if suppress_rows > 0:
                nearest_mask[H - suppress_rows:, :] = 0
            contours, _ = cv2.findContours(
                nearest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                used_nearest_fallback = True
                logger.debug(
                    f"[extract] nearest-depth fallback produced {len(contours)} contours"
                )
            else:
                logger.warning("No contours found in depth frame")
                return None
        
        logger.debug(f"[extract] findContours found {len(contours)} raw contours")
        
        # Step 3 — Apply filters to each contour
        candidates = []
        for contour in contours:
            # Filter A — Area gate
            area = cv2.contourArea(contour)
            contour_min_area = min_area_px
            if used_nearest_fallback:
                contour_min_area = max(220.0, min_area_px * 0.30)
            if not (contour_min_area <= area <= self.cfg.pipeline.max_roi_area_px):
                logger.debug(
                    f"[extract] REJECT area={area:.0f}px² "
                    f"(allowed [{int(contour_min_area)}, "
                    f"{self.cfg.pipeline.max_roi_area_px}])"
                )
                continue
            
            # Filter B — Use minAreaRect for orientation-invariant dimensions
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            
            short_dim = min(rw, rh)
            long_dim = max(rw, rh)
            
            # Filter C — Dimension validation
            contour_min_short = min_short_px
            contour_min_long = min_long_px
            if used_nearest_fallback:
                contour_min_short = max(12.0, min_short_px * 0.70)
                contour_min_long = max(30.0, min_long_px * 0.60)

            if not (contour_min_short <= short_dim <= self.cfg.pipeline.max_tube_dim_px):
                logger.debug(
                    f"[extract] REJECT short_dim={short_dim:.1f}px "
                    f"(allowed [{contour_min_short:.1f}, "
                    f"{self.cfg.pipeline.max_tube_dim_px}])"
                )
                continue
            if not (contour_min_long <= long_dim <= self.cfg.pipeline.max_tube_length_px):
                logger.debug(
                    f"[extract] REJECT long_dim={long_dim:.1f}px "
                    f"(allowed [{contour_min_long:.1f}, "
                    f"{self.cfg.pipeline.max_tube_length_px}])"
                )
                continue
            if orientation == 'angled':
                min_aspect_ratio = float(getattr(self.cfg.pipeline, 'angled_min_aspect_ratio', 1.05))
                max_aspect_ratio = float(getattr(self.cfg.pipeline, 'angled_max_aspect_ratio', 5.0))
            else:
                min_aspect_ratio = self._single_side_param("single_side_min_aspect_ratio", 1.35)
                max_aspect_ratio = float(self.cfg.pipeline.max_tube_length_px)  # effectively no upper cap
            aspect_ratio = long_dim / (short_dim + 1e-6)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                logger.debug(
                    f"[extract] REJECT aspect={aspect_ratio:.2f} "
                    f"(allowed [{min_aspect_ratio:.2f}, {max_aspect_ratio:.2f}])"
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
            bbox_short = float(min(w, h))
            bbox_long = float(max(w, h))
            min_bbox_short = self._single_side_param("single_side_min_bbox_short_px", 26.0)
            min_bbox_long = self._single_side_param("single_side_min_bbox_long_px", 60.0)
            if bbox_short < min_bbox_short or bbox_long < min_bbox_long:
                logger.debug(
                    f"[extract] REJECT bbox_dims=({w}x{h}) "
                    f"(min short={min_bbox_short:.1f}, long={min_bbox_long:.1f})"
                )
                continue
            m = self.cfg.pipeline.border_margin_px
            H, W = mask.shape
            if x < m or y < m or x + w > W - m or y + h > H - m:
                logger.debug(
                    f"[extract] REJECT border_margin: bbox=({x},{y},{w},{h}) "
                    f"frame=({W}x{H}) "
                    f"margin={m}"
                )
                continue

            if self._is_holder_like_candidate(x, y, w, h, (H, W)):
                logger.debug(
                    f"[extract] REJECT holder_like: bbox=({x},{y},{w},{h})"
                )
                continue

            if self._is_tray_like(x, y, w, h, area, solidity, (H, W)):
                logger.debug(
                    f"[extract] REJECT tray_like: bbox=({x},{y},{w},{h}) "
                    f"solidity={solidity:.3f} area={area:.0f}"
                )
                continue

            score = self._candidate_score(
                x=x,
                y=y,
                w=w,
                h=h,
                short_dim=short_dim,
                long_dim=long_dim,
                solidity=solidity,
                frame_shape=(H, W),
            )
            
            logger.debug(
                f"[extract] PASS contour: bbox=({x},{y},{w},{h}) "
                f"short={short_dim:.1f} long={long_dim:.1f} "
                f"solidity={solidity:.3f} area={area:.0f} score={score:.3f}"
            )
            
            candidates.append((contour, x, y, w, h, solidity, short_dim, long_dim, score))
        
        # Step 4 — Select best candidate (tube-likeness + center + temporal)
        if not candidates:
            logger.debug(
                f"[extract] 0 candidates after all filters "
                f"from {len(contours)} raw contours"
            )

            # Recovery path: if we had a valid ROI recently, relax gates and
            # only accept contours that are spatially consistent with it.
            recovered = None
            if self._last_bbox is not None:
                lx, ly, lw, lh = self._last_bbox
                H, W = mask.shape
                wx0 = max(0, lx - int(max(24, lw * 0.9)))
                wy0 = max(0, ly - int(max(24, lh * 0.9)))
                wx1 = min(W, lx + lw + int(max(24, lw * 0.9)))
                wy1 = min(H, ly + lh + int(max(24, lh * 0.9)))
                recovery_window = (wx0, wy0, max(1, wx1 - wx0), max(1, wy1 - wy0))

                relaxed_min_area = max(120.0, min_area_px * 0.20)
                relaxed_min_short = max(14.0, min_short_px * 0.70)
                relaxed_min_long = max(35.0, min_long_px * 0.70)
                relaxed_min_aspect = max(1.05, self._single_side_param("single_side_min_aspect_ratio", 1.35) * 0.75)
                relaxed_min_solidity = max(0.05, float(self.cfg.pipeline.min_solidity) * 0.70)
                relaxed_margin = max(4, int(self.cfg.pipeline.border_margin_px) // 2)

                best_recovery = None
                best_recovery_score = -1.0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if not (relaxed_min_area <= area <= self.cfg.pipeline.max_roi_area_px):
                        continue

                    rect = cv2.minAreaRect(contour)
                    (_, _), (rw, rh), _ = rect
                    short_dim = min(rw, rh)
                    long_dim = max(rw, rh)
                    if short_dim < relaxed_min_short or long_dim < relaxed_min_long:
                        continue

                    aspect_ratio = long_dim / (short_dim + 1e-6)
                    if aspect_ratio < relaxed_min_aspect:
                        continue

                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area <= 0:
                        continue
                    solidity = area / hull_area
                    if solidity < relaxed_min_solidity:
                        continue

                    box_pts = cv2.boxPoints(rect).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(box_pts)
                    if self._bbox_overlap_ratio((x, y, w, h), recovery_window) <= 0.0:
                        continue

                    if x < relaxed_margin or y < relaxed_margin or x + w > W - relaxed_margin or y + h > H - relaxed_margin:
                        continue
                    if self._is_holder_like_candidate(x, y, w, h, (H, W)):
                        continue
                    if self._is_tray_like(x, y, w, h, area, solidity, (H, W)):
                        continue

                    base_score = self._candidate_score(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        short_dim=short_dim,
                        long_dim=long_dim,
                        solidity=solidity,
                        frame_shape=(H, W),
                    )
                    overlap_score = self._bbox_overlap_ratio((x, y, w, h), recovery_window)
                    recovery_score = (0.65 * base_score) + (0.35 * overlap_score)

                    if recovery_score > best_recovery_score:
                        best_recovery_score = recovery_score
                        best_recovery = (contour, x, y, w, h, solidity, short_dim, long_dim, recovery_score)

                if best_recovery is not None:
                    logger.debug(
                        f"[extract] Recovery accepted near last ROI: "
                        f"score={best_recovery_score:.3f}, "
                        f"window=({wx0},{wy0},{wx1 - wx0},{wy1 - wy0})"
                    )
                    recovered = best_recovery

            if recovered is None:
                logger.warning("No contours passed all filters")
                return None
            best = recovered
        else:
            best = max(candidates, key=lambda c: c[-1])
        contour, x, y, w, h, solidity, short_dim, long_dim, score = best

        x, y, w, h = self._refine_with_candidate_depth(
            depth_frame=depth_frame,
            contour=contour,
            fallback_bbox=(x, y, w, h),
        )
        x, y, w, h = self._expand_single_side_bbox((x, y, w, h), mask.shape)

        # Trim bbox bottom to tube-holder depth boundary so SAM never sees
        # the holder region inside the prompt box.
        new_bottom = self._find_tube_bottom_from_depth(
            depth_frame, (x, y, w, h), orientation, is_inverted=is_inverted
        )
        if new_bottom < y + h:
            trimmed_h = new_bottom - y
            if trimmed_h >= int(h * 0.50):
                logger.debug(
                    f"[extract] Depth-gradient bottom trim: {y + h} → {new_bottom} "
                    f"(removed {y + h - new_bottom}px)"
                )
                h = trimmed_h

        self._last_bbox = (x, y, w, h)
        
        # Step 5 — Return with debug log
        logger.debug(
            f"[single_side] ROI: ({x},{y},{w},{h})  solidity={solidity:.3f}  "
            f"long={long_dim:.0f}px  short={short_dim:.0f}px  score={score:.3f}"
        )
        
        return (x, y, w, h)
    
    def extract_top(self, depth_frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Extract bounding box of tube from top-down depth view (single_top mode).
        
        For rack-mounted tubes viewed from above. Uses HoughCircles as primary
        path with contour-based fallback. Returns single best ROI.
        
        Args:
            depth_frame: Raw depth frame in uint16 format
            
        Returns:
            Tuple of (x, y, width, height) or None if no valid ROI found
        """
        # Step 1 — Preprocessing
        depth_min = self.cfg.pipeline.top_depth_min_m
        depth_max = self.cfg.pipeline.top_depth_max_m
        mask = self._preprocess_depth(depth_frame, depth_min, depth_max)
        
        # ─────────────────────────────────
        # PRIMARY PATH — HoughCircles
        # ─────────────────────────────────
        
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=int(self.cfg.pipeline.top_min_tube_dim_px // 2),
            maxRadius=int(self.cfg.pipeline.top_max_tube_dim_px // 2)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            candidates = []
            
            for (cx, cy, r) in circles:
                # Convert circle to bbox
                x = int(cx - r)
                y = int(cy - r)
                w = int(2 * r)
                h = int(2 * r)
                
                # Border margin check
                m = self.cfg.pipeline.top_border_margin_px
                H, W = mask.shape
                if x < m or y < m or x + w > W - m or y + h > H - m:
                    continue
                
                candidates.append((x, y, w, h, r))
            
            if candidates:
                # Select circle closest to frame center
                frame_cx, frame_cy = mask.shape[1] // 2, mask.shape[0] // 2
                best = min(
                    candidates,
                    key=lambda c: (c[0] + c[2] // 2 - frame_cx) ** 2
                                  + (c[1] + c[3] // 2 - frame_cy) ** 2
                )
                x, y, w, h, _ = best
                logger.debug(f"[single_top] HoughCircles ROI: ({x},{y},{w},{h})")
                return (x, y, w, h)
        
        # ─────────────────────────────────
        # FALLBACK PATH — Contour-based
        # ─────────────────────────────────
        
        logger.debug("[single_top] HoughCircles found nothing — falling back to contour")
        
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
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
            H, W = mask.shape
            if x < m or y < m or x + w > W - m or y + h > H - m:
                continue
            
            candidates.append((x, y, w, h, solidity))
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda c: c[-1])
        x, y, w, h, _ = best
        logger.debug(f"[single_top] Contour fallback ROI: ({x},{y},{w},{h})")
        return (x, y, w, h)
    
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
                x, y, w, h = int(cx - r), int(cy - r), int(2 * r), int(2 * r)
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
