import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from pydantic import BaseModel


class FillLevel(Enum):
    """Categorical fill-level labels for a tube ROI."""

    FULL = "full"
    HALF = "half"
    EMPTY = "empty"


@dataclass
class FillLevelResult:
    """Result payload for fill-level detection."""

    level: FillLevel
    confidence: str
    boundary_ratio: float


class FillLevelConfig(BaseModel):
    """Configuration for Sobel-based fill-level detection."""

    ignore_top_pct: float = 0.15
    ignore_bottom_pct: float = 0.15
    strip_width_pct: float = 0.30
    half_threshold: float = 0.33
    empty_threshold: float = 0.66
    edge_strength_min: int = 10
    brightness_empty_threshold: float = 200.0
    brightness_full_threshold: float = 80.0


class FillLevelDetector:
    """Detect tube fill level from a cropped RGB ROI using Sobel edges."""

    def __init__(self, config: FillLevelConfig):
        """Initialize the detector with algorithm configuration."""
        self.config = config

    def detect(self, roi: np.ndarray) -> FillLevelResult:
        """Detect fill level from ROI using edge-first logic with brightness fallback.

        Args:
            roi: Cropped RGB tube ROI image.

        Returns:
            FillLevelResult with detected level, confidence source, and boundary ratio.

        Raises:
            ValueError: If ROI is None or has invalid geometry.
        """
        if roi is None:
            raise ValueError("ROI cannot be None.")
        if roi.ndim < 2:
            raise ValueError("ROI must have at least 2 dimensions.")

        height, width = roi.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("ROI must have non-zero height and width.")

        # Step 1: Convert ROI to grayscale.
        if roi.ndim == 2:
            gray = roi
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Step 2: Extract center vertical strip.
        strip_width = max(1, int(round(width * self.config.strip_width_pct)))
        strip_width = min(strip_width, width)
        x_start = (width - strip_width) // 2
        x_end = x_start + strip_width
        gray_strip = gray[:, x_start:x_end]

        # Step 3: Apply Sobel(dy=1, dx=0, ksize=3) on strip.
        sobel = cv2.Sobel(gray_strip, cv2.CV_64F, 0, 1, ksize=3)

        # Step 4: Absolute value and convert to uint8.
        sobel_abs = np.absolute(sobel)
        sobel_u8 = np.clip(sobel_abs, 0, 255).astype(np.uint8)

        # Step 5: Ignore top/bottom rows.
        ignore_top = int(height * self.config.ignore_top_pct)
        ignore_bottom = int(height * self.config.ignore_bottom_pct)
        valid_start = ignore_top
        valid_end = height - ignore_bottom
        if valid_end <= valid_start:
            valid_start = 0
            valid_end = height

        valid_sobel = sobel_u8[valid_start:valid_end, :]

        # Step 6: Per-row mean Sobel response.
        row_means = np.mean(valid_sobel, axis=1)

        # Step 7: Max-response row as boundary candidate.
        max_idx_local = int(np.argmax(row_means))
        boundary_y = valid_start + max_idx_local
        max_response = float(row_means[max_idx_local])

        # Step 8: Edge-strength gate with brightness fallback.
        if max_response < self.config.edge_strength_min:
            # Step 9: Brightness fallback in valid zone.
            valid_gray = gray[valid_start:valid_end, :]
            mean_brightness = float(np.mean(valid_gray))

            if mean_brightness > self.config.brightness_empty_threshold:
                level = FillLevel.EMPTY
            elif mean_brightness < self.config.brightness_full_threshold:
                level = FillLevel.FULL
            else:
                level = FillLevel.HALF

            # Step 10: Boundary ratio for fallback.
            boundary_ratio = -1.0
            result = FillLevelResult(
                level=level,
                confidence="brightness_fallback",
                boundary_ratio=boundary_ratio,
            )
            # Step 12: Log result.
            logger.debug(
                "Fill level detected (brightness_fallback): "
                f"level={result.level.value}, mean_brightness={mean_brightness:.2f}, "
                f"valid_rows=({valid_start}, {valid_end})"
            )
            return result

        # Step 10: Compute boundary_ratio from edge boundary row.
        boundary_ratio = boundary_y / float(height)

        # Step 11: Map ratio to fill level.
        if boundary_ratio < self.config.half_threshold:
            level = FillLevel.FULL
        elif boundary_ratio < self.config.empty_threshold:
            level = FillLevel.HALF
        else:
            level = FillLevel.EMPTY

        result = FillLevelResult(
            level=level,
            confidence="edge_detected",
            boundary_ratio=boundary_ratio,
        )

        # Step 12: Log result.
        logger.debug(
            "Fill level detected (edge_detected): "
            f"level={result.level.value}, boundary_y={boundary_y}, "
            f"boundary_ratio={boundary_ratio:.4f}, edge_strength={max_response:.2f}, "
            f"valid_rows=({valid_start}, {valid_end})"
        )
        return result
