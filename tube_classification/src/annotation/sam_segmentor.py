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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate 5-point prompt coordinates anchored to anchor_x.
        
        Args:
            anchor_x: x-coordinate to anchor positive points to (from bright column or bbox_cx)
            bbox: Bounding box as (x, y, width, height)
            
        Returns:
            Tuple of (point_coords, point_labels) for SAM predictor
        """
        x, y, w, h = bbox
        bbox_cy = y + h // 2
        
        # POSITIVE POINTS (foreground) - mark tube at multiple locations
        # All anchored to the same detected/fallback x-coordinate
        point_cap = np.array([[anchor_x, int(y + 0.1 * h)]])
        point_upper = np.array([[anchor_x, int(y + 0.4 * h)]])
        point_lower = np.array([[anchor_x, int(y + 0.75 * h)]])
        
        # NEGATIVE POINTS (background) - mark regions outside tube
        point_bg_left = np.array([[max(0, x - 20), bbox_cy]])
        point_bg_right = np.array([[x + w + 20, bbox_cy]])
        
        point_coords = np.vstack([
            point_cap, point_upper, point_lower,
            point_bg_left, point_bg_right
        ])
        
        point_labels = np.array([1, 1, 1, 0, 0])
        
        return point_coords, point_labels
    
    def segment(
        self,
        rgb_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, float] | None:
        """Run segmentation on RGB frame with robust box-based prompting.
        
        At 22cm camera distance, tubes are small. Use primarily box prompting
        with selective point-based retry for better reliability.
        
        Strategy:
        1. PRIMARY: Box-only prompt (forces SAM to fill entire bbox)
        2. SECONDARY: If box-only IoU is low, retry with 5-point prompts
        3. POST-PROCESS: Morphological operations to fill gaps
        4. VALIDATION: Check area and IoU thresholds
        
        Args:
            rgb_frame: RGB image array
            bbox: Bounding box as (x, y, width, height)
            
        Returns:
            Tuple of (binary_mask, iou_score) or None if segmentation failed.
            binary_mask: 0-255 uint8 array with holes filled
            iou_score: MobileSAM predicted IoU (0.0-1.0)
        """
        if self.predictor is None:
            logger.error("SAM not loaded. Call load() first.")
            return None
        
        # Set image for prediction
        self.predictor.set_image(rgb_frame)
        
        # Convert bbox format: (x, y, w, h) -> (x1, y1, x2, y2)
        x, y, w, h = bbox
        bbox_cx = x + w // 2
        input_box = np.array([x, y, x + w, y + h])
        
        logger.debug(f"SAM input: bbox=[{x},{y},{x+w},{y+h}] (w={w}, h={h})")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 1: PRIMARY ATTEMPT - BOX-ONLY PROMPT (more reliable at small size)
        # ─────────────────────────────────────────────────────────────
        
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        iou_score = float(scores[0])
        mask = masks[0].astype(np.uint8) * 255
        
        logger.debug(f"SAM box-only attempt: IoU={iou_score:.3f}, area={cv2.countNonZero(mask)}px²")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: CONFIDENCE CHECK - RETRY WITH 5-POINT IF BOX-ONLY IS WEAK
        # ─────────────────────────────────────────────────────────────
        
        # Only retry if IoU is particularly low (< 0.5)
        if iou_score < 0.50:
            logger.debug(
                f"Low IoU detected: {iou_score:.3f} < 0.50, "
                f"retrying with 5-point prompts"
            )
            
            # Find brightest column for tube center
            bright_col_x = self._find_bright_column(rgb_frame, bbox)
            anchor_x = bright_col_x if bright_col_x is not None else bbox_cx
            
            # Generate 5-point prompt
            point_coords, point_labels = self._get_point_coords(anchor_x, bbox)
            
            # Retry with points + box
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            iou_score = float(scores[0])
            mask = masks[0].astype(np.uint8) * 255
            
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
        
        final_area = cv2.countNonZero(mask)
        logger.debug(f"SAM post-process: area={final_area}px², kernel_size={kernel_size}")
        
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
