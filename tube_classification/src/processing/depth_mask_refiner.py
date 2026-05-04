import logging

import cv2
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DepthMaskRefinerConfig(BaseModel):
    enabled: bool = True
    depth_band_tolerance_mm: float = 40.0
    depth_sample_region_ratio: float = 0.30
    zero_depth_support_kernel_size: int = 9
    zero_depth_fallback_min_keep_ratio: float = 0.55
    zero_depth_fallback_min_zero_ratio: float = 0.35
    holder_height_mm: float = 120.0
    holder_width_mm: float = 40.0
    holder_x_pad_ratio: float = 0.20
    morph_closing_kernel_size: int = 5
    morph_closing_iterations: int = 2
    min_contour_area_px: int = 500
    apply_only_orientations: list[str] = ["side", "angled"]


def refine_mask(
    mask: np.ndarray,
    depth_frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    intrinsics: object,
    config: DepthMaskRefinerConfig,
    orientation: str = "side",
) -> tuple[np.ndarray, float | None]:
    """Refine a SAM binary mask using depth geometry.

    Applies four sequential steps:
      1. Depth band filter — removes pixels outside the tube's depth plane.
      2. Morphological closing — reconnects gaps from transparent regions.
      3. Holder exclusion — zeros the projected holder footprint below the tube.
      4. Tube contour selection — keeps the most tube-like connected component.

    Zero-depth pixels (RealSense measurement failures) are treated as "no data"
    and are passed through in Step 1 so they do not carve holes in the mask.

    Args:
        mask:        uint8 H×W binary mask from SAM (0 or 255).
        depth_frame: uint16 H×W aligned depth frame in mm.
        bbox:        (x, y, w, h) ROI from the extractor.
        intrinsics:  RealSense rs.intrinsics with a .fy attribute (focal length).
        config:      Refinement configuration.
        orientation: Capture orientation string; refinement is skipped unless
                     it appears in config.apply_only_orientations.

    Returns:
        Tuple of (refined_mask, median_depth_mm).  refined_mask is a uint8
        H×W binary mask, never empty (Step 4 safety fallback).
        median_depth_mm is the median depth of the bbox upper region in mm,
        or None if Step 1 was skipped (no valid depth samples).
    """
    if not config.enabled or orientation not in config.apply_only_orientations:
        return mask, None

    original_mask = mask.copy()
    refined = mask.copy()
    x, y, w, h = bbox
    _, W = refined.shape
    median_depth: float | None = None

    def _select_tube_contour(
        contours: list[np.ndarray],
        roi_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray | None, float, float, float]:
        """Pick contour that best matches tube geometry (not just largest area)."""
        if not contours:
            return None, 0.0, 0.0, 0.0

        bx, by, bw, bh = roi_bbox
        roi_area = float(max(1, bw * bh))
        roi_cx = bx + (bw * 0.5)
        upper_limit = by + (0.62 * bh)

        best_contour: np.ndarray | None = None
        best_score = -1e9
        best_area = 0.0
        best_h = 0.0
        max_area = 0.0
        max_h = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue

            cx, cy, cw, ch = cv2.boundingRect(contour)
            ccx = float(cx + (cw * 0.5))
            ccy = float(cy + (ch * 0.5))
            max_area = max(max_area, area)
            max_h = max(max_h, float(ch))

            area_cov = min(area / roi_area, 1.8)
            height_cov = min(float(ch) / max(float(bh), 1.0), 1.3)
            x_center_score = max(
                0.0,
                1.0 - (abs(ccx - roi_cx) / max(float(bw) * 0.75, 1.0)),
            )
            top_align = max(
                0.0,
                1.0 - (max(0.0, ccy - upper_limit) / max(float(bh), 1.0)),
            )

            score = (
                0.45 * area_cov
                + 0.35 * height_cov
                + 0.20 * top_align
                + 0.10 * x_center_score
            )

            if score > best_score:
                best_score = score
                best_contour = contour
                best_area = area
                best_h = float(ch)

        return best_contour, best_area, best_h, max(max_area, 1.0), max(max_h, 1.0)

    # ── Step 1: Dynamic depth band filter ────────────────────────────────────
    sample_h = max(1, int(h * config.depth_sample_region_ratio))
    sample_region = depth_frame[y : y + sample_h, x : x + w]
    valid_samples = sample_region[sample_region > 0]

    if valid_samples.size > 0:
        median_depth = float(np.median(valid_samples))
        tol = config.depth_band_tolerance_mm
        lo, hi = median_depth - tol, median_depth + tol

        depth_band = (depth_frame >= lo) & (depth_frame <= hi)
        zero_depth = depth_frame == 0

        # Preserve invalid-depth pixels only when they are spatially connected to
        # in-band mask support; this prevents distant holder regions from leaking in.
        band_support = depth_band & (refined > 0)
        if np.count_nonzero(band_support) > 0 and np.count_nonzero(zero_depth) > 0:
            kz = max(1, int(config.zero_depth_support_kernel_size))
            if kz % 2 == 0:
                kz += 1
            support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kz, kz))
            support = cv2.dilate(
                (band_support.astype(np.uint8) * 255), support_kernel, iterations=1
            ) > 0
            zero_supported = zero_depth & support
        else:
            zero_supported = zero_depth

        valid_depth = (depth_band | zero_supported).astype(np.uint8) * 255

        before = int(np.count_nonzero(refined))
        candidate = cv2.bitwise_and(refined, valid_depth)
        candidate_count = int(np.count_nonzero(candidate))

        # Transparent/small tubes often have sparse valid depth on the body.
        # If support-gated zero-depth keeps too little mask while many mask
        # pixels have invalid depth, fall back to permissive zero-depth keep.
        zero_ratio_in_mask = float(np.count_nonzero(zero_depth & (refined > 0))) / max(
            float(before), 1.0
        )
        keep_ratio = candidate_count / max(float(before), 1.0)
        needs_zero_fallback = (
            before > 0
            and zero_ratio_in_mask >= config.zero_depth_fallback_min_zero_ratio
            and keep_ratio < config.zero_depth_fallback_min_keep_ratio
        )
        if needs_zero_fallback:
            permissive_valid_depth = (depth_band | zero_depth).astype(np.uint8) * 255
            refined = cv2.bitwise_and(refined, permissive_valid_depth)
            logger.debug(
                "Step 1 zero-depth fallback: keep_ratio=%.2f zero_ratio=%.2f",
                keep_ratio,
                zero_ratio_in_mask,
            )
        else:
            refined = candidate

        after = int(np.count_nonzero(refined))
        logger.debug(
            "Step 1 depth band: median=%.1fmm band=[%.0f, %.0f]mm pixels %d->%d",
            median_depth, lo, hi, before, after,
        )
    else:
        logger.debug("Step 1 skipped: no valid depth samples in bbox upper region")

    # ── Step 2: Morphological closing ────────────────────────────────────────
    k = config.morph_closing_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_CLOSE, kernel, iterations=config.morph_closing_iterations
    )
    logger.debug(
        "Step 2 closing: kernel=%dx%d iters=%d pixels=%d",
        k, k, config.morph_closing_iterations, int(np.count_nonzero(refined)),
    )

    # ── Step 3: Holder pixel exclusion ───────────────────────────────────────
    if median_depth is not None and median_depth > 0:
        fy = float(intrinsics.fy)
        y_bottom = y + h
        h_holder_px = int((fy * config.holder_height_mm) / median_depth)
        pad_px = max(15, int(w * config.holder_x_pad_ratio))
        x_start = max(0, x - pad_px)
        x_end = min(W, x + w + pad_px)

        before = int(np.count_nonzero(refined))
        refined[y_bottom : y_bottom + h_holder_px, x_start : x_end] = 0
        after = int(np.count_nonzero(refined))
        logger.debug(
            "Step 3 holder exclusion: y_bottom=%d h_holder_px=%d x=[%d, %d] pixels %d->%d",
            y_bottom, h_holder_px, x_start, x_end, before, after,
        )
    else:
        logger.debug("Step 3 skipped: no valid median_depth for holder projection")

    # ── Step 4: Tube-like contour selection ───────────────────────────────────
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("Step 4: no contours found, returning refined mask as-is")
        return refined, median_depth

    selected_contour, selected_area, selected_h, max_area, max_h = _select_tube_contour(
        contours, bbox
    )
    if selected_contour is None:
        logger.debug("Step 4: no valid contour selected, returning refined mask as-is")
        return refined, median_depth

    # Safety for small tubes: avoid selecting cap-only fragments when a larger
    # or taller connected component exists.
    too_small_vs_max_area = selected_area < (0.50 * max_area)
    too_short_vs_max_height = selected_h < (0.66 * max_h)
    if (too_small_vs_max_area and too_short_vs_max_height) and len(contours) > 1:
        largest = max(contours, key=cv2.contourArea)
        selected_contour = largest
        selected_area = float(cv2.contourArea(largest))

    if selected_area < config.min_contour_area_px:
        logger.debug(
            "Step 4 safety fallback: selected area=%.0fpx² < min=%dpx², returning original SAM mask",
            selected_area, config.min_contour_area_px,
        )
        return original_mask, median_depth

    canvas = np.zeros_like(refined, dtype=np.uint8)
    cv2.drawContours(canvas, [selected_contour], 0, 255, -1)
    logger.debug("Step 4: kept contour area=%.0fpx²", selected_area)

    return canvas, median_depth
