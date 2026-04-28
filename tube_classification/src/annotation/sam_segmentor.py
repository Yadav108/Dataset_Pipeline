import numpy as np
import torch
import cv2
from pathlib import Path

from loguru import logger
from mobile_sam import sam_model_registry, SamPredictor
from config.parser import get_config


class SAMSegmentor:
    """Load and run MobileSAM for semantic segmentation.
    
    Loads MobileSAM weights at startup and runs inference using
    bounding box prompts to generate binary masks.
    """
    
    def __init__(self):
        """Initialize segmentor with config."""
        self.cfg = get_config()
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self) -> None:
        """Load MobileSAM weights and initialize predictor.
        
        Raises:
            FileNotFoundError: If weights file not found
        """
        weights_path = Path(self.cfg.storage.sam_weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"MobileSAM weights not found at {weights_path}"
            )
        
        # Load model
        sam = sam_model_registry["vit_t"](checkpoint=str(weights_path))
        sam.to(self.device)
        sam.eval()
        
        # Create predictor
        self.predictor = SamPredictor(sam)
        
        logger.info(f"MobileSAM loaded on {self.device}")
    
    def _find_bright_column(
        self,
        rgb_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> int | None:
        """Find the brightest vertical column inside bbox (indicates tube center).
        
        Scans all vertical columns within the bbox and returns the x-coordinate
        of the brightest column. This is more stable than using fixed bbox ratios
        when the tube shifts between frames.
        
        Args:
            rgb_frame: RGB image array
            bbox: Bounding box as (x, y, width, height)
            
        Returns:
            x-coordinate of brightest column, or None if std < 10 (fallback to bbox_cx)
        """
        x, y, w, h = bbox
        
        # Extract region inside bbox
        region = rgb_frame[y:y+h, x:x+w]
        if region.size == 0:
            return None
        
        # Convert to grayscale if RGB
        if len(region.shape) == 3 and region.shape[2] == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region if len(region.shape) == 2 else region[:, :, 0]
        
        # Compute mean brightness of each column
        col_brightness = gray.mean(axis=0)
        
        # Check if variance is sufficient (std >= 10 indicates clear bright/dark contrast)
        if col_brightness.std() < 10:
            logger.debug(
                f"Bright column detection: std={col_brightness.std():.2f} < 10, "
                f"falling back to bbox_cx"
            )
            return None
        
        # Find brightest column
        brightest_col_idx = np.argmax(col_brightness)
        brightest_col_x = x + brightest_col_idx
        
        logger.debug(
            f"Bright column detected at x={brightest_col_x} "
            f"(relative={brightest_col_idx}, brightness={col_brightness[brightest_col_idx]:.1f}, "
            f"std={col_brightness.std():.2f})"
        )
        
        return brightest_col_x
    
    def _get_point_coords(
        self,
        anchor_x: int,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate point prompts anchored to anchor_x.
        
        Args:
            anchor_x: x-coordinate to anchor positive points to (from bright column or bbox_cx)
            bbox: Bounding box as (x, y, width, height)
            frame_shape: RGB frame shape for in-bounds point clamping
            
        Returns:
            Tuple of (point_coords, point_labels) for SAM predictor
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        bbox_cy = y + h // 2
        use_bottom_negative = h >= int(1.35 * max(w, 1))
        bg_bottom_y = y + h + max(18, int(0.25 * h))
        safe_anchor_x = int(np.clip(anchor_x, 0, frame_w - 1))
        safe_bbox_cy = int(np.clip(bbox_cy, 0, frame_h - 1))
        safe_bg_bottom_y = int(np.clip(bg_bottom_y, 0, frame_h - 1))

        # POSITIVE POINTS (foreground) - mark tube at multiple locations
        # All anchored to the same detected/fallback x-coordinate
        point_cap = np.array([[safe_anchor_x, int(np.clip(y + 0.1 * h, 0, frame_h - 1))]])
        point_upper = np.array([[safe_anchor_x, int(np.clip(y + 0.4 * h, 0, frame_h - 1))]])
        point_lower = np.array([[safe_anchor_x, int(np.clip(y + 0.75 * h, 0, frame_h - 1))]])
        
        # NEGATIVE POINTS (background) - mark regions outside tube
        point_bg_left = np.array([[int(np.clip(x - 20, 0, frame_w - 1)), safe_bbox_cy]])
        point_bg_right = np.array([[int(np.clip(x + w + 20, 0, frame_w - 1)), safe_bbox_cy]])
        prompt_points = [
            point_cap, point_upper, point_lower,
            point_bg_left, point_bg_right
        ]
        prompt_labels = [1, 1, 1, 0, 0]
        if use_bottom_negative:
            point_bg_bottom = np.array([[safe_anchor_x, safe_bg_bottom_y]])
            prompt_points.append(point_bg_bottom)
            prompt_labels.append(0)

        point_coords = np.vstack(prompt_points)
        point_labels = np.array(prompt_labels)
        
        return point_coords, point_labels

    @staticmethod
    def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
        """Convert SAM output mask to uint8 [0, 255] with minimal extra allocations."""
        binary = mask.astype(np.uint8, copy=False)
        np.multiply(binary, 255, out=binary, casting="unsafe")
        return binary

    @staticmethod
    def _expand_prompt_box(
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        """Expand SAM prompt box to avoid clipping tube ends near ROI edges."""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        if h >= w:
            pad_x = max(10, int(round(w * 0.18)))
            pad_y = max(14, int(round(h * 0.22)))
        else:
            pad_x = max(14, int(round(w * 0.22)))
            pad_y = max(10, int(round(h * 0.18)))

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    @staticmethod
    def _select_best_candidate_mask(
        masks: np.ndarray,
        scores: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, float]:
        """Pick SAM candidate favoring full tube extent, not only IoU confidence."""
        x, y, w, h = bbox
        bbox_area = float(max(1, w * h))
        best_idx = 0
        best_score = -1e9

        for idx in range(len(scores)):
            mask = masks[idx].astype(np.uint8, copy=False) * 255
            area = float(cv2.countNonZero(mask))
            rows = np.where(mask.any(axis=1))[0]
            mask_height = float(rows[-1] - rows[0] + 1) if len(rows) > 0 else 0.0

            vertical_coverage = mask_height / max(float(h), 1.0)
            area_coverage = area / bbox_area

            composite = (
                float(scores[idx])
                + 0.35 * min(vertical_coverage, 1.25)
                + 0.15 * min(area_coverage, 1.75)
            )
            if composite > best_score:
                best_score = composite
                best_idx = idx

        return masks[best_idx], float(scores[best_idx])

    @staticmethod
    def _trim_bottom_holder_contact(
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, int]:
        """Trim bottom-attached holder region when it blooms wider than tube body."""
        _, _, w, h = bbox
        if h < int(1.35 * w):
            return mask, 0

        rows = np.where(mask.any(axis=1))[0]
        if len(rows) < 24:
            return mask, 0

        y0 = int(rows[0])
        y1 = int(rows[-1])
        span = y1 - y0 + 1
        if span < 26:
            return mask, 0

        row_widths = np.count_nonzero(mask > 0, axis=1).astype(np.int32)
        lower_band_start = y0 + int(0.62 * span)
        upper_rows = np.arange(y0, max(y0 + 1, lower_band_start), dtype=np.int32)
        upper_rows = upper_rows[row_widths[upper_rows] > 0]
        if len(upper_rows) < 8:
            return mask, 0

        core_width = float(np.median(row_widths[upper_rows]))
        if core_width < 6.0:
            return mask, 0

        bloom_threshold = max(core_width * 1.48, core_width + 12.0)
        cut_start = None
        run = 0

        for yy in range(y1, lower_band_start - 1, -1):
            if row_widths[yy] >= bloom_threshold:
                run += 1
            else:
                if run >= 3:
                    cut_start = yy + 1
                    break
                run = 0

        if cut_start is None and run >= 3:
            cut_start = lower_band_start
        if cut_start is None:
            return mask, 0

        trimmed = mask.copy()
        trimmed[cut_start : y1 + 1, :] = 0

        # Secondary light trim for small residual bottom bleed-through.
        # If bottom strip is still noticeably wider than tube core, shave only
        # the last few rows so tube tip is preserved.
        tr_rows = np.where(trimmed.any(axis=1))[0]
        if len(tr_rows) > 12:
            ty0 = int(tr_rows[0])
            ty1 = int(tr_rows[-1])
            tspan = ty1 - ty0 + 1
            tr_widths = np.count_nonzero(trimmed > 0, axis=1).astype(np.int32)
            core_rows = np.arange(ty0, ty0 + max(6, int(0.45 * tspan)), dtype=np.int32)
            core_rows = core_rows[tr_widths[core_rows] > 0]
            if len(core_rows) >= 6:
                tcore = float(np.median(tr_widths[core_rows]))
                residual_threshold = max(tcore * 1.28, tcore + 8.0)
                bottom_run = 0
                for yy in range(ty1, max(ty0, ty1 - int(0.10 * tspan)), -1):
                    if tr_widths[yy] >= residual_threshold:
                        bottom_run += 1
                    else:
                        break
                if bottom_run >= 4:
                    shave_from = max(ty0, ty1 - bottom_run + 1)
                    trimmed[shave_from : ty1 + 1, :] = 0

        contours, _ = cv2.findContours(trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, 0
        main = np.zeros_like(trimmed)
        cv2.drawContours(main, [max(contours, key=cv2.contourArea)], 0, 255, -1)

        after_rows = np.where(main.any(axis=1))[0]
        if len(after_rows) == 0:
            return mask, 0
        after_span = int(after_rows[-1] - after_rows[0] + 1)
        if after_span < int(0.72 * span):
            return mask, 0

        removed = int(cv2.countNonZero(mask) - cv2.countNonZero(main))
        return main, max(0, removed)

    @staticmethod
    def _select_tube_component(
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, int]:
        """Keep the most tube-like connected component and drop lower blob artifacts."""
        x, y, w, h = bbox
        binary = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 2:
            return mask, 0

        roi_cx = x + (w * 0.5)
        original_area = float(cv2.countNonZero(mask))
        bbox_area = float(max(1, w * h))
        best_label = -1
        best_score = -1e9
        best_area = 0.0
        best_height = 0.0
        max_component_area = 0.0
        max_component_height = 0.0

        # Pre-scan component stats so we can reject tiny fragments relative to
        # the dominant component (prevents selecting cap-like blobs for large tubes).
        for lbl in range(1, num_labels):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            ch = float(max(1, stats[lbl, cv2.CC_STAT_HEIGHT]))
            if area > max_component_area:
                max_component_area = area
            if ch > max_component_height:
                max_component_height = ch

        for lbl in range(1, num_labels):
            cx = float(stats[lbl, cv2.CC_STAT_LEFT] + (stats[lbl, cv2.CC_STAT_WIDTH] * 0.5))
            cy = float(stats[lbl, cv2.CC_STAT_TOP] + (stats[lbl, cv2.CC_STAT_HEIGHT] * 0.5))
            cw = float(max(1, stats[lbl, cv2.CC_STAT_WIDTH]))
            ch = float(max(1, stats[lbl, cv2.CC_STAT_HEIGHT]))
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            if area < 40.0:
                continue

            # Reject tiny fragments when a much larger/taller component exists.
            if max_component_area > 0 and area < max(0.32 * max_component_area, 0.08 * bbox_area):
                continue
            if max_component_height > 0 and ch < max(0.50 * max_component_height, 0.32 * float(h)):
                continue

            height_cov = min(ch / max(float(h), 1.0), 1.3)
            area_cov = min(area / max(bbox_area, 1.0), 1.6)
            aspect = min((ch / cw), 3.0)
            x_center_score = max(0.0, 1.0 - (abs(cx - roi_cx) / max(float(w) * 0.65, 1.0)))
            top_align_score = max(0.0, 1.0 - ((max(0.0, cy - (y + 0.58 * h))) / max(float(h), 1.0)))
            lower_penalty = max(0.0, (cy - (y + 0.78 * h)) / max(float(h), 1.0))

            score = (
                0.32 * height_cov
                + 0.26 * area_cov
                + 0.16 * aspect
                + 0.16 * x_center_score
                + 0.10 * top_align_score
                - 0.30 * lower_penalty
            )

            if score > best_score:
                best_score = score
                best_label = lbl
                best_area = area
                best_height = ch

        if best_label < 0:
            return mask, 0

        # Fail-safe: never reduce to tiny fragment for small tubes.
        min_area_after = max(120.0, 0.14 * bbox_area, 0.35 * original_area)
        min_height_after = max(16.0, 0.40 * float(h))
        if best_area < min_area_after or best_height < min_height_after:
            return mask, 0

        selected = np.zeros_like(mask, dtype=np.uint8)
        selected[labels == best_label] = 255
        removed = int(cv2.countNonZero(mask) - cv2.countNonZero(selected))
        if removed <= 0:
            return mask, 0

        # Additional safety: if pruning is too destructive, keep original.
        selected_area = float(cv2.countNonZero(selected))
        if selected_area < (0.45 * original_area):
            return mask, 0
        return selected, removed

    @staticmethod
    def _enforce_tube_profile(
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, int]:
        """Clamp lower over-wide mask regions to a tube-like vertical profile."""
        _, _, w, h = bbox
        rows = np.where(mask.any(axis=1))[0]
        if len(rows) < 24:
            return mask, 0

        y0 = int(rows[0])
        y1 = int(rows[-1])
        span = y1 - y0 + 1
        if span < 24:
            return mask, 0

        original_area = float(cv2.countNonZero(mask))
        if original_area <= 0:
            return mask, 0

        left = np.full(mask.shape[0], -1, dtype=np.int32)
        right = np.full(mask.shape[0], -1, dtype=np.int32)
        widths = np.zeros(mask.shape[0], dtype=np.int32)
        centers = np.zeros(mask.shape[0], dtype=np.float32)

        for yy in rows:
            cols = np.where(mask[yy] > 0)[0]
            if len(cols) == 0:
                continue
            left[yy] = int(cols[0])
            right[yy] = int(cols[-1])
            widths[yy] = int(cols[-1] - cols[0] + 1)
            centers[yy] = float((cols[0] + cols[-1]) * 0.5)

        core_end = y0 + int(0.52 * span)
        core_rows = np.arange(y0, core_end + 1, dtype=np.int32)
        core_rows = core_rows[widths[core_rows] > 0]
        if len(core_rows) < 10:
            return mask, 0

        core_width = float(np.median(widths[core_rows]))
        if core_width < 6.0:
            return mask, 0
        allowed_width = int(round(max(core_width * 1.50, core_width + 10.0)))
        core_center = float(np.median(centers[core_rows]))

        clamped = mask.copy()
        lower_start = y0 + int(0.50 * span)
        changed_rows = 0

        for yy in range(lower_start, y1 + 1):
            row_w = widths[yy]
            if row_w <= allowed_width:
                continue

            row_center = centers[yy] if centers[yy] > 0 else core_center
            target_center = (0.70 * row_center) + (0.30 * core_center)
            new_l = int(round(target_center - (allowed_width * 0.5)))
            new_r = int(round(target_center + (allowed_width * 0.5)))
            new_l = max(0, new_l)
            new_r = min(mask.shape[1] - 1, new_r)
            if new_r <= new_l:
                continue

            clamped[yy, :new_l] = 0
            clamped[yy, new_r + 1 :] = 0
            changed_rows += 1

        if changed_rows < 3:
            return mask, 0

        contours, _ = cv2.findContours(clamped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, 0
        main = np.zeros_like(clamped)
        cv2.drawContours(main, [max(contours, key=cv2.contourArea)], 0, 255, -1)

        after_rows = np.where(main.any(axis=1))[0]
        if len(after_rows) == 0:
            return mask, 0
        after_span = int(after_rows[-1] - after_rows[0] + 1)
        after_area = float(cv2.countNonZero(main))

        if after_span < int(0.70 * span):
            return mask, 0
        if after_area < (0.50 * original_area):
            return mask, 0

        removed = int(original_area - after_area)
        if removed <= 0:
            return mask, 0
        return main, removed
    
    @staticmethod
    def build_tube_sam_prompts(
        bbox: tuple[int, int, int, int],
        orientation: str = 'side',
        is_inverted: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build SAM point prompts tuned to camera orientation.

        side:           positive=upper 60%, negative=bottom corners
        angled:         positive=upper-center + mid body, negative=lower-right zone
        angled+inverted: positive=upper 55%, negative=dense grid across bottom 35%
                         (holder occupies the wide lower portion when depth-inverted)
        top:            positive=center only

        Args:
            bbox: Bounding box as (x, y, width, height).
            orientation: 'side' | 'angled' | 'top'
            is_inverted: True when holder is closer to camera than tube.

        Returns:
            Tuple of (point_coords, point_labels) where labels are 1=fg / 0=bg.
        """
        x1, y1 = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]

        if orientation == 'angled' and is_inverted:
            # Tube occupies upper portion of bbox; holder is wide lower region.
            pos_points = np.array([
                [x1 + w // 2, y1 + int(h * 0.15)],
                [x1 + w // 2, y1 + int(h * 0.35)],
                [x1 + w // 2, y1 + int(h * 0.55)],
            ], dtype=np.float32)
            # Dense negative grid across bottom 35% — aggressively suppress holder
            neg_ys = [y1 + int(h * 0.72), y1 + int(h * 0.82), y1 + int(h * 0.92)]
            neg_xs = [x1 + int(w * f) for f in [0.10, 0.30, 0.50, 0.70, 0.90]]
            neg_points = np.array(
                [[nx, ny] for ny in neg_ys for nx in neg_xs], dtype=np.float32
            )
        elif orientation == 'angled':
            pos_points = np.array([
                [x1 + w // 2,          y1 + int(h * 0.18)],
                [x1 + w // 2,          y1 + int(h * 0.40)],
                [x1 + int(w * 0.45),   y1 + int(h * 0.60)],
            ], dtype=np.float32)
            # Assumption: camera is elevated on the LEFT side of the setup so the
            # holder/rack recedes to the lower-RIGHT in the projected image.
            # If your camera is mounted on the right, mirror the x-ratios
            # (swap 0.70/0.85 → 0.15/0.30 and 0.15 → 0.70).
            neg_points = np.array([
                [x1 + int(w * 0.70), y1 + int(h * 0.80)],
                [x1 + int(w * 0.85), y1 + int(h * 0.88)],
                [x1 + int(w * 0.15), y1 + int(h * 0.90)],
                [x1 + int(w * 0.50), y1 + int(h * 0.95)],
            ], dtype=np.float32)
        elif orientation == 'top':
            pos_points = np.array([
                [x1 + w // 2, y1 + h // 2],
            ], dtype=np.float32)
            neg_points = np.empty((0, 2), dtype=np.float32)
        else:  # side
            pos_points = np.array([
                [x1 + w // 2, y1 + int(h * 0.20)],
                [x1 + w // 2, y1 + int(h * 0.45)],
            ], dtype=np.float32)
            neg_points = np.array([
                [x1 + int(w * 0.20), y1 + int(h * 0.82)],
                [x1 + int(w * 0.80), y1 + int(h * 0.82)],
                [x1 + int(w * 0.10), y1 + int(h * 0.95)],
                [x1 + int(w * 0.90), y1 + int(h * 0.95)],
            ], dtype=np.float32)

        if len(neg_points) > 0:
            points = np.vstack([pos_points, neg_points])
            labels = np.array(
                [1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32
            )
        else:
            points = pos_points
            labels = np.array([1] * len(pos_points), dtype=np.int32)
        return points, labels

    @staticmethod
    def filter_mask_by_tube_depth(
        sam_mask: np.ndarray,
        depth_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        tolerance_mm: int = 40,
    ) -> np.ndarray:
        """Intersect SAM mask with a depth band sampled from the tube body.

        Samples the upper 40% of the bbox to estimate the tube's median depth,
        then retains only mask pixels within ±tolerance_mm of that depth.
        This removes holder/rack pixels that share the SAM mask but sit at a
        different depth plane.

        Args:
            sam_mask: Binary mask (0/255, uint8) from SAM post-processing.
            depth_frame: Raw depth frame (uint16, mm units).
            bbox: Bounding box as (x, y, width, height).
            tolerance_mm: Half-width of the accepted depth band around the tube
                median (mm). Wider tolerances survive tilted tubes; narrower
                tolerances cut holders more aggressively.

        Returns:
            Refined binary mask (0/255, uint8).  Returns sam_mask unchanged when
            fewer than 10 valid depth samples are available in the sample region.
        """
        x, y, w, h = bbox
        sample_h = int(h * 0.40)
        tube_region = depth_frame[y:y + sample_h, x:x + w].astype(np.float32)
        valid = tube_region[(tube_region > 100) & (tube_region < 60000)]
        if len(valid) < 10:
            return sam_mask
        median_depth = float(np.median(valid))
        depth_band = (
            (depth_frame >= median_depth - tolerance_mm)
            & (depth_frame <= median_depth + tolerance_mm)
        ).astype(np.uint8) * 255
        return cv2.bitwise_and(sam_mask, depth_band)

    @staticmethod
    def keep_tube_component(mask: np.ndarray, orientation: str = 'side') -> np.ndarray:
        """Keep the most tube-like connected component, discarding squat blobs.

        Scores each component by ``aspect × log(area)`` where aspect = h/w.
        The minimum aspect threshold is relaxed for angled-view captures where
        foreshortening reduces the projected tube aspect ratio to ~0.85–1.1.

        Args:
            mask: Binary mask (0/255, uint8).
            orientation: 'side' | 'angled' | 'top'

        Returns:
            Binary mask (0/255, uint8) with only the winning component.
            Returns the original mask unchanged if no component qualifies.
        """
        min_aspect = 0.85 if orientation == 'angled' else 1.2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )
        best_label, best_score = -1, -1.0
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < 400:
                continue
            aspect = ch / max(cw, 1)
            if aspect < min_aspect:
                continue
            score = aspect * float(np.log(area + 1))
            if score > best_score:
                best_score = score
                best_label = i
        if best_label == -1:
            return mask
        return (labels == best_label).astype(np.uint8) * 255

    def segment(
        self,
        rgb_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        depth_frame: np.ndarray | None = None,
        orientation: str = 'side',
        is_inverted: bool = False,
    ) -> tuple[np.ndarray, float] | None:
        """Run segmentation on RGB frame with holder-aware prompting.

        Strategy:
        1. PRIMARY: Box + negative points in the bottom 25% of the bbox to
           suppress holder bleed-through from the first SAM call.
        2. SECONDARY: If IoU is low or vertical coverage is weak, retry with
           bright-column-anchored 5-point prompts.
        3. POST-PROCESS: Morphological closing, hole fill, component selection.
        4. DEPTH FILTER: Intersect mask with tube-body depth band (±40 mm).
        5. COMPONENT FILTER: Keep only the tallest/narrowest blob (aspect ≥ 1.2).
        6. VALIDATION: Area and IoU thresholds.

        Args:
            rgb_frame: RGB image array (H, W, 3).
            bbox: Bounding box as (x, y, width, height).
            depth_frame: Optional raw depth frame (uint16, mm units).  When
                provided, enables the depth intersection filter (step 4) which
                removes holder pixels that share the SAM mask but sit at a
                different depth plane.

        Returns:
            Tuple of (binary_mask, iou_score) or None if segmentation failed.
            binary_mask: 0-255 uint8 array with holes filled.
            iou_score: MobileSAM predicted IoU (0.0-1.0).
        """
        if self.predictor is None:
            logger.error("SAM not loaded. Call load() first.")
            return None
        
        # Set image for prediction
        self.predictor.set_image(rgb_frame)
        
        # Convert bbox format: (x, y, w, h) -> (x1, y1, x2, y2)
        x, y, w, h = bbox
        bbox_cx = x + w // 2
        input_box = self._expand_prompt_box((x, y, w, h), rgb_frame.shape)
        
        logger.debug(
            f"SAM input: roi=[{x},{y},{x+w},{y+h}] "
            f"prompt_box=[{int(input_box[0])},{int(input_box[1])},{int(input_box[2])},{int(input_box[3])}] "
            f"(w={w}, h={h})"
        )
        
        # ─────────────────────────────────────────────────────────────
        # STEP 1: PRIMARY ATTEMPT - BOX + HOLDER-SUPPRESSING NEGATIVE POINTS
        # Negative points are placed in the bottom 25% of the bbox on every
        # call to prevent SAM from bleeding into the tube holder/rack.
        # ─────────────────────────────────────────────────────────────

        tube_point_coords, tube_point_labels = self.build_tube_sam_prompts((x, y, w, h), orientation, is_inverted)
        masks, scores, _ = self.predictor.predict(
            point_coords=tube_point_coords,
            point_labels=tube_point_labels,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        selected_mask, iou_score = self._select_best_candidate_mask(
            masks=masks,
            scores=scores,
            bbox=(x, y, w, h),
        )
        mask = self._to_binary_mask(selected_mask)
        
        logger.debug(f"SAM primary attempt (box+neg_points, orientation={orientation}): IoU={iou_score:.3f}, area={cv2.countNonZero(mask)}px²")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: CONFIDENCE CHECK - RETRY WITH 5-POINT IF BOX-ONLY IS WEAK
        # ─────────────────────────────────────────────────────────────
        
        # Retry not only on low IoU, but also on thin/fragmented masks.
        mask_rows = np.where(mask.any(axis=1))[0]
        mask_height = int(mask_rows[-1] - mask_rows[0] + 1) if len(mask_rows) > 0 else 0
        weak_vertical_coverage = (h >= 70) and (mask_height < int(0.50 * h))
        if iou_score < 0.50 or weak_vertical_coverage:
            logger.debug(
                f"Weak primary mask: iou={iou_score:.3f}, "
                f"mask_height={mask_height}px, bbox_h={h}px; retrying with bright-column 5-point prompts"
            )
            
            # Find brightest column for tube center
            bright_col_x = self._find_bright_column(rgb_frame, bbox)
            anchor_x = bright_col_x if bright_col_x is not None else bbox_cx
            
            # Generate 5-point prompt
            point_coords, point_labels = self._get_point_coords(anchor_x, bbox, rgb_frame.shape)
            
            # Retry with points + box
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box[None, :],
                multimask_output=True,
            )
            
            selected_mask, iou_score = self._select_best_candidate_mask(
                masks=masks,
                scores=scores,
                bbox=(x, y, w, h),
            )
            mask = self._to_binary_mask(selected_mask)
            
            logger.debug(f"SAM 5-point retry: IoU={iou_score:.3f}, area={cv2.countNonZero(mask)}px²")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 3: POST-PROCESSING: Morphological closing + hole filling
        # ─────────────────────────────────────────────────────────────
        
        # Adaptive kernel size based on bbox size (for 22cm distance)
        # Smaller bbox → smaller kernel
        kernel_size = max(5, min(11, w // 15))  # Range 5-11 based on width
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd number
        
        # Morphological closing with adaptive kernel
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Dilate to connect any fragmented regions
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Fill interior holes
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)

        # Keep only connected-component isolation here.
        # Width-based cuts were over-trimming wider tubes (e.g., 10ml) and
        # collapsing the downstream rembg crop.
        mask, removed_blob_pixels = self._select_tube_component(mask, (x, y, w, h))
        if removed_blob_pixels > 0:
            logger.debug(f"SAM blob-prune applied: removed_pixels={removed_blob_pixels}")

        # ─────────────────────────────────────────────────────────────
        # STEP 3a: DEPTH INTERSECTION — remove holder pixels at a different depth
        # ─────────────────────────────────────────────────────────────
        if depth_frame is not None:
            pre_depth_area = cv2.countNonZero(mask)
            depth_filtered = self.filter_mask_by_tube_depth(mask, depth_frame, (x, y, w, h))
            post_depth_area = cv2.countNonZero(depth_filtered)
            if pre_depth_area > 0 and post_depth_area >= int(0.30 * pre_depth_area):
                logger.debug(
                    f"Depth filter applied: {pre_depth_area} → {post_depth_area}px² "
                    f"({pre_depth_area - post_depth_area}px removed)"
                )
                mask = depth_filtered
            else:
                # Depth sample unreliable (transparent cap / no depth data) — skip filter
                logger.debug(
                    f"Depth filter skipped: would reduce {pre_depth_area} → {post_depth_area}px² (>70%)"
                )

        # ─────────────────────────────────────────────────────────────
        # STEP 3b: COMPONENT FILTER — discard squat blobs (aspect < 1.2)
        # ─────────────────────────────────────────────────────────────
        pre_comp_area = cv2.countNonZero(mask)
        comp_filtered = self.keep_tube_component(mask, orientation)
        post_comp_area = cv2.countNonZero(comp_filtered)
        if pre_comp_area > 0 and post_comp_area >= int(0.35 * pre_comp_area):
            if post_comp_area < pre_comp_area:
                logger.debug(
                    f"keep_tube_component: {pre_comp_area} → {post_comp_area}px² "
                    f"({pre_comp_area - post_comp_area}px removed)"
                )
            mask = comp_filtered
        else:
            logger.debug(
                f"keep_tube_component reverted: {pre_comp_area} → {post_comp_area}px²"
            )

        final_area = cv2.countNonZero(mask)
        logger.debug(f"SAM post-process: area={final_area}px², kernel_size={kernel_size}")
        final_rows = np.where(mask.any(axis=1))[0]
        final_height = int(final_rows[-1] - final_rows[0] + 1) if len(final_rows) > 0 else 0
        
        # ─────────────────────────────────────────────────────────────
        # STEP 4: VALIDATION - CHECK AREA AND IOU THRESHOLDS
        # ─────────────────────────────────────────────────────────────
        
        # More lenient area threshold for small tubes at 22cm
        min_area = max(100, w * h * 0.15)  # At least 15% of bbox
        if final_area < min_area:
            logger.warning(
                f"SAM mask too small: area={final_area}px² < min={min_area:.0f}px² "
                f"(bbox {w}x{h})"
            )
            return None
        if h >= 70 and final_height < int(0.45 * h):
            logger.warning(
                f"SAM mask too short vertically: height={final_height}px < "
                f"{int(0.45 * h)}px (bbox_h={h})"
            )
            return None
        
        # Check IoU threshold (more lenient for small objects)
        min_iou = self.cfg.pipeline.sam_iou_threshold - 0.05  # Allow 5% lower for small tubes
        if iou_score < min_iou:
            logger.warning(
                f"SAM mask rejected: IoU={iou_score:.3f} below threshold "
                f"{min_iou:.3f} (small tube at 22cm)"
            )
            return None
        
        logger.info(
            f"SAM PASSED: IoU={iou_score:.3f}, area={final_area}px², "
            f"bbox={w}x{h}, kernel={kernel_size}"
        )
        
        
        
        return (mask, iou_score)
        
        # Convert bbox format: (x, y, w, h) -> (x1, y1, x2, y2)
        x, y, w, h = bbox
        bbox_cx = x + w // 2
        input_box = np.array([x, y, x + w, y + h])
        
        # ─────────────────────────────────────────────────────────────
        # STEP 1: DYNAMIC BRIGHT COLUMN DETECTION
        # ─────────────────────────────────────────────────────────────
        
        bright_col_x = self._find_bright_column(rgb_frame, bbox)
        anchor_x = bright_col_x if bright_col_x is not None else bbox_cx
        
        strategy_used = "bright_column" if bright_col_x is not None else "bbox_cx"
        logger.debug(f"Point anchor strategy: {strategy_used} (x={anchor_x})")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: PRIMARY ATTEMPT - 5-POINT PROMPT WITH DYNAMIC ANCHORING
        # ─────────────────────────────────────────────────────────────
        
        point_coords, point_labels = self._get_point_coords(anchor_x, bbox)
        
        logger.debug(
            f"SAM 5-point prompt: cap=({int(point_coords[0, 0])},{int(point_coords[0, 1])}) "
            f"upper=({int(point_coords[1, 0])},{int(point_coords[1, 1])}) "
            f"lower=({int(point_coords[2, 0])},{int(point_coords[2, 1])}) "
            f"anchor_strategy={strategy_used} bbox=[{x},{y},{x+w},{y+h}]"
        )
        
        # Run prediction with 5-point prompt + bbox constraint
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        iou_score = float(scores[0])
        mask = masks[0].astype(np.uint8) * 255
        
        logger.debug(f"SAM 5-point initial: IoU={iou_score:.3f}, area={cv2.countNonZero(mask)}px²")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 3: CONFIDENCE CHECK - RETRY WITH BOX-ONLY IF WEAK MASK
        # ─────────────────────────────────────────────────────────────
        
        # Calculate mask height (vertical extent of mask)
        mask_rows = np.where(mask.any(axis=1))[0]
        mask_height = mask_rows[-1] - mask_rows[0] if len(mask_rows) > 0 else 0
        
        retry_attempted = False
        if mask_height < 0.4 * h:
            logger.debug(
                f"Weak mask detected: height={mask_height}px < 0.4*bbox_h={0.4*h:.0f}px, "
                f"retrying with box-only prompt"
            )
            
            # Retry with box prompt only (no point prompts)
            # This forces SAM to fill the entire bbox
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            iou_score = float(scores[0])
            mask = masks[0].astype(np.uint8) * 255
            retry_attempted = True
            
            logger.debug(f"SAM box-only retry: IoU={iou_score:.3f}, area={cv2.countNonZero(mask)}px²")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 4: POST-PROCESSING: Morphological closing + hole filling
        # ─────────────────────────────────────────────────────────────
        
        # Step 4a: Morphological closing with larger kernel to fill gaps in tube body
        # Use 11x11 kernel with 4 iterations for better filling of transparent regions
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
        
        # Step 4b: Dilate slightly to connect fragmented regions, then erode to restore size
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel_dilate, iterations=2)
        mask = cv2.erode(mask, kernel_dilate, iterations=1)
        
        # Step 4c: Fill any remaining holes inside the mask contour
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        final_area = cv2.countNonZero(mask)
        logger.debug(f"SAM post-process: area after closing & filling={final_area}px²")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 5: VALIDATION - CHECK AREA AND IOU THRESHOLDS
        # ─────────────────────────────────────────────────────────────
        
        # Validate mask area is reasonable for full tube
        # Reduced from 400 to 300 since improved morphological operations fill better
        if final_area < 300:
            logger.warning(
                f"SAM mask too small: area={final_area}px² < 300px² "
                f"(likely only cap, not full tube body)"
            )
            return None
        
        # Check IoU threshold
        if iou_score < self.cfg.pipeline.sam_iou_threshold:
            logger.warning(
                f"SAM mask rejected: IoU={iou_score:.3f} below threshold "
                f"{self.cfg.pipeline.sam_iou_threshold:.3f}"
            )
            return None
        
        # Log success with strategy information
        final_mask_height = np.where(mask.any(axis=1))[0]
        final_height = final_mask_height[-1] - final_mask_height[0] if len(final_mask_height) > 0 else 0
        
        logger.info(
            f"SAM PASSED: anchor_strategy={strategy_used}, "
            f"retry={retry_attempted}, IoU={iou_score:.3f}, "
            f"area={final_area}px², height={final_height}px"
        )
        
        return (mask, iou_score)

