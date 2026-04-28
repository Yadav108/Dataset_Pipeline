"""
Preprocessing module: Bilateral filtering, normalization, temporal smoothing, quality metrics.

Implements PROMPT 1-4 from DETAILED_PROMPTS.md
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, asdict
from collections import deque
import json
import time
from loguru import logger


# ============================================================================
# PROMPT 1: BILATERAL FILTER PREPROCESSING
# ============================================================================

def preprocess_depth_bilateral(
    depth_frame: np.ndarray,
    spatial_sigma: float = 15.0,
    range_sigma: float = 50.0,
    diameter: int = 25,
    iterations: int = 2,
    min_valid_ratio: float = 0.85,
    max_processing_time_ms: float = 300.0,
    logger_instance = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply bilateral filtering to depth frame with edge preservation.
    
    Args:
        depth_frame: Input depth (uint16, mm)
        spatial_sigma: Spatial domain smoothing (pixels)
        range_sigma: Range domain smoothing (mm)
        diameter: Filter diameter (must be odd)
        iterations: Number of filter passes
        min_valid_ratio: Minimum valid pixel ratio threshold
        max_processing_time_ms: Maximum processing time allowed (ms)
        logger_instance: Optional logger instance
    
    Returns:
        (filtered_frame, validity_mask)
        - filtered_frame: Smoothed depth (uint16)
        - validity_mask: Valid pixel mask (uint8, 255 for valid)
    
    Raises:
        ValueError: If input shape invalid or min_valid_ratio violated
        RuntimeError: If processing exceeds max_processing_time_ms
    """
    log = logger_instance or logger
    start_time = time.time()
    
    # Input validation
    if depth_frame.ndim != 2:
        raise ValueError(f"Invalid shape: {depth_frame.shape}, expected 2D depth frame")
    if depth_frame.dtype != np.uint16:
        raise ValueError(f"Invalid dtype: {depth_frame.dtype}, expected uint16")
    if diameter % 2 == 0:
        raise ValueError(f"Diameter must be odd, got {diameter}")
    if not (0.0 < spatial_sigma < 100.0):
        raise ValueError(f"spatial_sigma out of range: {spatial_sigma}")
    if not (0.0 < range_sigma < 1000.0):
        raise ValueError(f"range_sigma out of range: {range_sigma}")
    
    # Create validity mask (non-zero pixels)
    validity_mask = (depth_frame > 0).astype(np.uint8) * 255
    valid_ratio_before = np.count_nonzero(depth_frame) / depth_frame.size
    
    # Convert to float for processing
    depth_float = depth_frame.astype(np.float32)
    filtered = depth_float.copy()
    
    # Apply bilateral filter iteratively
    for i in range(iterations):
        filtered = cv2.bilateralFilter(
            filtered,
            diameter,
            range_sigma,
            spatial_sigma
        )
    
    # Convert back to uint16
    filtered_frame = np.clip(filtered, 0, 65535).astype(np.uint16)
    
    # Update validity mask (filter may create new zeros)
    validity_mask = (filtered_frame > 0).astype(np.uint8) * 255
    valid_ratio_after = np.count_nonzero(filtered_frame) / filtered_frame.size
    
    # Check valid pixel ratio
    if valid_ratio_after < min_valid_ratio:
        raise ValueError(
            f"Valid pixel ratio {valid_ratio_after:.2%} below threshold {min_valid_ratio:.2%}"
        )
    
    # Check processing time (scale budget by frame size; config value is baseline for 640x480)
    elapsed_ms = (time.time() - start_time) * 1000
    baseline_pixels = 480 * 640
    scaled_limit_ms = max_processing_time_ms * (depth_frame.size / baseline_pixels)
    if elapsed_ms > scaled_limit_ms:
        raise RuntimeError(
            f"Processing time {elapsed_ms:.1f}ms exceeds {scaled_limit_ms:.1f}ms limit "
            f"(base {max_processing_time_ms:.1f}ms @ 640x480)"
        )
    
    log.info(
        f"Bilateral filter applied | valid_ratio: {valid_ratio_after:.2%} | "
        f"time: {elapsed_ms:.1f}ms"
    )
    
    return filtered_frame, validity_mask


# ============================================================================
# PROMPT 2: DEPTH NORMALIZATION
# ============================================================================

def normalize_depth(
    depth_frame: np.ndarray,
    min_mm: float,
    max_mm: float,
    invalid_pixel_value: float = np.nan,
    logger_instance = None
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize depth to [0, 1] range, invertible.
    
    Args:
        depth_frame: Input depth (uint16, mm)
        min_mm: Minimum valid depth (mm)
        max_mm: Maximum valid depth (mm)
        invalid_pixel_value: Value for zero/invalid pixels
        logger_instance: Optional logger
    
    Returns:
        (normalized_frame, metadata)
    
    Raises:
        ValueError: If min_mm >= max_mm
    """
    log = logger_instance or logger
    
    if min_mm >= max_mm:
        raise ValueError(f"min_mm ({min_mm}) must be < max_mm ({max_mm})")
    if invalid_pixel_value not in [np.nan, -1.0]:
        raise TypeError(f"invalid_pixel_value must be np.nan or -1.0, got {invalid_pixel_value}")
    
    # Normalize
    normalized = (depth_frame.astype(np.float32) - min_mm) / (max_mm - min_mm)
    normalized = np.clip(normalized, 0, 1)
    
    # Mark invalid pixels
    invalid_mask = depth_frame == 0
    if isinstance(invalid_pixel_value, float):
        if np.isnan(invalid_pixel_value):
            normalized[invalid_mask] = np.nan
        else:
            normalized[invalid_mask] = invalid_pixel_value
    
    invalid_count = np.count_nonzero(invalid_mask)
    valid_ratio = 1.0 - (invalid_count / depth_frame.size)
    
    metadata = {
        'min_mm': float(min_mm),
        'max_mm': float(max_mm),
        'invalid_count': int(invalid_count),
        'valid_ratio': float(valid_ratio),
        'min_normalized': float(np.nanmin(normalized)),
        'max_normalized': float(np.nanmax(normalized))
    }
    
    if valid_ratio < 0.8:
        log.warning(f"High invalid ratio: {valid_ratio:.2%}")
    
    log.info(f"Normalized depth | valid_ratio: {valid_ratio:.2%}")
    
    return normalized.astype(np.float32), metadata


def denormalize_depth(
    normalized_frame: np.ndarray,
    min_mm: float,
    max_mm: float,
    logger_instance = None
) -> np.ndarray:
    """
    Inverse operation: [0,1] → mm
    
    Returns:
        depth_frame (uint16, mm)
    
    Raises:
        ValueError: If normalized values outside [0, 1]
    """
    log = logger_instance or logger
    
    # Check bounds (allow NaN)
    valid_mask = ~np.isnan(normalized_frame)
    if np.any((normalized_frame[valid_mask] < 0) | (normalized_frame[valid_mask] > 1)):
        raise ValueError("Normalized values outside [0, 1] range")
    
    # Denormalize
    depth_mm = normalized_frame * (max_mm - min_mm) + min_mm
    depth_frame = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    
    log.info("Denormalized depth to mm")
    
    return depth_frame


# ============================================================================
# PROMPT 3: TEMPORAL SMOOTHING FILTER
# ============================================================================

class TemporalSmoothingFilter:
    """EMA-based temporal smoothing for depth frames."""
    
    def __init__(
        self,
        alpha: float = 0.2,
        window_size: int = 5,
        jitter_threshold_mm: float = 10.0,
        logger_instance = None
    ):
        """
        Args:
            alpha: Smoothing factor [0.1, 0.5]
            window_size: History buffer size
            jitter_threshold_mm: Outlier detection threshold
            logger_instance: Optional logger
        """
        if not (0.1 <= alpha <= 0.5):
            raise ValueError(f"alpha must be in [0.1, 0.5], got {alpha}")
        
        self.alpha = alpha
        self.window_size = window_size
        self.jitter_threshold_mm = jitter_threshold_mm
        self.log = logger_instance or logger
        
        self.history: deque = deque(maxlen=window_size)
        self.frames_processed = 0
        self.jitter_reductions: List[float] = []
    
    def smooth(
        self,
        depth_frame: np.ndarray,
        detect_outliers: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """Apply temporal smoothing to current frame."""
        start_time = time.time()
        
        if len(self.history) == 0:
            # First frame: return as-is
            smoothed = depth_frame.copy()
            jitter_reduction = 0.0
            outlier_count = 0
        else:
            # Get previous frame
            prev_frame = self.history[-1]
            
            # Calculate jitter (frame-to-frame difference)
            diff = np.abs(depth_frame.astype(np.int32) - prev_frame.astype(np.int32))
            
            # Detect outliers
            if detect_outliers:
                outlier_mask = diff > self.jitter_threshold_mm
                outlier_count = np.count_nonzero(outlier_mask)
                
                # Clamp outliers to threshold
                diff_clipped = np.clip(diff, 0, self.jitter_threshold_mm)
                smoothed_temp = prev_frame + (diff_clipped * np.sign(diff)).astype(np.int32)
            else:
                smoothed_temp = prev_frame.astype(np.int32) + diff.astype(np.int32)
                outlier_count = 0
            
            # Apply EMA
            smoothed = (
                self.alpha * depth_frame.astype(np.float32) +
                (1 - self.alpha) * prev_frame.astype(np.float32)
            ).astype(np.uint16)
            
            # Calculate jitter reduction
            variance_before = np.var(diff)
            variance_after = np.var(np.abs(smoothed.astype(np.int32) - prev_frame.astype(np.int32)))
            jitter_reduction = 100 * (1 - variance_after / (variance_before + 1e-6))
        
        # Store in history
        self.history.append(depth_frame.copy())
        self.frames_processed += 1
        self.jitter_reductions.append(jitter_reduction)
        
        # Calculate temporal variance
        if len(self.history) > 1:
            temporal_variance = float(np.var([f.astype(np.float32) for f in self.history]))
        else:
            temporal_variance = 0.0
        
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 5:
            self.log.warning(f"Smoothing exceeded 5ms target: {elapsed_ms:.1f}ms")
        
        metrics = {
            'jitter_reduction_pct': float(jitter_reduction),
            'temporal_variance_mm2': float(temporal_variance),
            'outlier_count': int(outlier_count),
            'processing_time_ms': float(elapsed_ms)
        }
        
        return smoothed, metrics
    
    def reset(self) -> None:
        """Clear history buffer."""
        self.history.clear()
        self.log.info("Temporal smoothing filter reset")
    
    def get_statistics(self) -> Dict:
        """Get filter statistics."""
        mean_jitter = float(np.mean(self.jitter_reductions)) if self.jitter_reductions else 0.0
        current_var = float(np.var([f.astype(np.float32) for f in self.history])) if len(self.history) > 1 else 0.0
        
        return {
            'mean_jitter_reduction': mean_jitter,
            'frames_processed': self.frames_processed,
            'current_variance': current_var
        }


# ============================================================================
# PROMPT 4: QUALITY METRICS VALIDATOR
# ============================================================================

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics container."""
    timestamp: float
    frame_id: int
    
    # Depth metrics
    valid_pixel_ratio: float
    depth_min_mm: float
    depth_max_mm: float
    depth_mean_mm: float
    depth_std_mm: float
    depth_snr_db: float
    depth_uniformity: float
    
    # RGB metrics
    blur_score: float
    contrast_ratio: float
    edge_density: float
    saturation_score: float
    hue_variance: float
    illumination_level: float
    
    # Mask metrics (optional)
    mask_area_px: Optional[int] = None
    mask_compactness: Optional[float] = None
    mask_coverage_ratio: Optional[float] = None
    
    # Meta
    processing_time_ms: float = 0.0
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dict, handling optional fields."""
        return {k: v for k, v in asdict(self).items()}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


def compute_quality_metrics(
    depth_frame: np.ndarray,
    rgb_frame: np.ndarray,
    mask: np.ndarray = None,
    frame_id: int = 0,
    logger_instance = None
) -> QualityMetrics:
    """
    Compute comprehensive quality metrics.
    
    Args:
        depth_frame: uint16, mm
        rgb_frame: uint8, BGR or RGB
        mask: Optional segmentation mask (uint8)
        frame_id: Frame number
        logger_instance: Optional logger
    
    Returns:
        QualityMetrics
    
    Raises:
        ValueError: If shapes invalid
    """
    log = logger_instance or logger
    start_time = time.time()
    timestamp = time.time()
    
    # Validate shapes
    if depth_frame.ndim != 2:
        raise ValueError(f"Invalid depth shape: {depth_frame.shape}")
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError(f"Invalid RGB shape: {rgb_frame.shape}")
    if rgb_frame.shape[:2] != depth_frame.shape:
        raise ValueError(
            f"Depth/RGB spatial mismatch: depth {depth_frame.shape} vs rgb {rgb_frame.shape[:2]}"
        )
    
    # ===== DEPTH METRICS =====
    valid_pixels = depth_frame > 0
    valid_pixel_ratio = np.count_nonzero(valid_pixels) / depth_frame.size
    
    if valid_pixel_ratio > 0:
        valid_depths = depth_frame[valid_pixels]
        depth_min_mm = float(np.min(valid_depths))
        depth_max_mm = float(np.max(valid_depths))
        depth_mean_mm = float(np.mean(valid_depths))
        depth_std_mm = float(np.std(valid_depths))
    else:
        depth_min_mm = depth_max_mm = depth_mean_mm = depth_std_mm = 0.0
    
    # SNR calculation
    signal_power = float(np.mean(depth_frame[valid_pixels] ** 2)) if valid_pixel_ratio > 0 else 0.0
    noise_variance = float(np.std(depth_frame[valid_pixels]) ** 2) if valid_pixel_ratio > 0 else 1.0
    depth_snr_db = float(10 * np.log10((signal_power + 1e-6) / (noise_variance + 1e-6)))
    
    # Depth uniformity (inverse of gradient magnitude)
    if valid_pixel_ratio > 0:
        grad_x = cv2.Sobel(depth_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_frame, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        depth_uniformity = float(1.0 / (1.0 + np.mean(grad_magnitude[valid_pixels])))
    else:
        depth_uniformity = 0.0
    
    # ===== RGB METRICS =====
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY) if len(rgb_frame.shape) == 3 else rgb_frame
    
    # Blur score (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    blur_score = float(laplacian.var())
    
    # Contrast
    contrast_ratio = float(np.max(gray) / (np.min(gray) + 1))
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    
    # Color metrics
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    saturation_score = float(np.mean(s) / 255.0)
    hue_variance = float(np.std(h) / 180.0)  # Normalized to [0,1]
    illumination_level = float(np.mean(v))
    
    # ===== MASK METRICS =====
    mask_area_px = None
    mask_compactness = None
    mask_coverage_ratio = None
    
    if mask is not None and mask.size > 0:
        mask_area_px = int(np.count_nonzero(mask))
        
        # Compactness (perimeter² / (4π * area))
        if mask_area_px > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest, True)
                area = cv2.contourArea(largest)
                if area > 0:
                    mask_compactness = float(min(1.0, (perimeter ** 2) / (4 * np.pi * area)))
                    
                    # Estimate coverage against full frame
                    mask_coverage_ratio = float(area / depth_frame.size)
    
    # ===== QUALITY SCORE =====
    quality_score = (
        (valid_pixel_ratio * 2) +
        (min(blur_score / 50, 2)) +
        (min(contrast_ratio / 3, 2)) +
        (edge_density * 2) +
        (saturation_score * 1.5) +
        (depth_uniformity * 1.5)
    ) / 10.0
    quality_score = float(np.clip(quality_score, 0, 10))
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    if elapsed_ms > 100:
        log.warning(f"Quality metrics computation exceeded 100ms: {elapsed_ms:.1f}ms")
    
    if quality_score < 5.0:
        log.warning(
            f"Low quality ({quality_score:.1f}): "
            f"blur={blur_score:.1f}, contrast={contrast_ratio:.1f}, edges={edge_density:.2%}"
        )
    
    metrics = QualityMetrics(
        timestamp=timestamp,
        frame_id=frame_id,
        valid_pixel_ratio=float(valid_pixel_ratio),
        depth_min_mm=depth_min_mm,
        depth_max_mm=depth_max_mm,
        depth_mean_mm=depth_mean_mm,
        depth_std_mm=depth_std_mm,
        depth_snr_db=depth_snr_db,
        depth_uniformity=float(depth_uniformity),
        blur_score=blur_score,
        contrast_ratio=contrast_ratio,
        edge_density=float(edge_density),
        saturation_score=float(saturation_score),
        hue_variance=float(hue_variance),
        illumination_level=illumination_level,
        mask_area_px=mask_area_px,
        mask_compactness=mask_compactness,
        mask_coverage_ratio=mask_coverage_ratio,
        processing_time_ms=elapsed_ms,
        quality_score=quality_score
    )
    
    log.info(f"Quality metrics computed | score={quality_score:.1f} | time={elapsed_ms:.1f}ms")
    
    return metrics
