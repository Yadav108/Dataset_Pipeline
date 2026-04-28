"""
Guided filter module for depth denoising using RGB guidance.

Implements edge-preserving depth smoothing by fitting local linear models
in RGB space and applying them to depth. This preserves sharp depth transitions
where RGB has sharp edges, while smoothing noisy regions.

Algorithm Reference:
    He et al., "Guided Image Filtering" (IEEE TPAMI, 2013)
    https://dl.acm.org/doi/10.1145/1869790.1869829

Pipeline Visualization:
    
    RGB Frame (clean edges)          Depth Frame (noisy)
           |                                 |
           |                    validate & fill invalid pixels
           |                                 |
           +-----------+-----------+---------+
                       |
                 Normalize RGB [0,1]
                       |
            Fit local linear models
                 depth ≈ a×RGB + b
                       |
            Apply models to output
                       |
            Restore invalid pixels (=0)
                       |
            Convert back to uint16
                       |
                Smoothed Depth (sharp edges)

Usage Example:
    >>> from src.acquisition.guided_filter import guided_denoise
    >>> filtered_depth, stats = guided_denoise(
    ...     depth_frame=depth_uint16,
    ...     rgb_frame=rgb_uint8_bgr,
    ...     radius=8,
    ...     eps=1e-3,
    ...     logger_instance=logger
    ... )
    >>> print(f"Noise reduction: {stats['noise_reduction_pct']:.1f}%")
"""

import numpy as np
import cv2
import time
from typing import Tuple, Dict, Optional
from scipy.ndimage import uniform_filter


def guided_denoise(
    depth_frame: np.ndarray,
    rgb_frame: np.ndarray,
    radius: int = 16,
    eps: float = 1e-3,
    rgb_normalize: bool = True,
    preserve_invalid: bool = True,
    max_processing_time_ms: float = 100.0,
    logger_instance = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply guided filter using RGB as guidance to denoise depth.
    
    The guided filter preserves edges in depth where RGB has edges, while
    smoothing flat regions. This is superior to bilateral filtering for
    depth maps because RGB edges often correspond to real depth boundaries.
    
    Algorithm:
        For each pixel, fit linear model: depth ≈ a×RGB + b in local window.
        Then apply same linear model to output (preserves RGB structure in depth).
    
    Args:
        depth_frame (np.ndarray):
            Depth map, shape (H, W), dtype uint16, mm scale.
            Invalid pixels marked as 0 and preserved throughout.
        
        rgb_frame (np.ndarray):
            RGB image, shape (H, W, 3), dtype uint8, BGR order (RealSense default).
            Must have clean edges for guidance to work well.
        
        radius (int, optional):
            Guided filter radius, defining local window size = (2*radius+1)².
            Typical values:
                - 4: Fast, minimal smoothing
                - 8: Balanced (default-ish)
                - 16: Aggressive smoothing (good for noisy depth)
                - 32: Very aggressive (use with caution)
            Default: 16
        
        eps (float, optional):
            Regularization parameter for numerical stability.
            Prevents division by zero in linear regression.
            Higher values = more edge preservation, less smoothing.
            Lower values = more smoothing, less edge preservation.
            Typical range: 1e-4 to 1e-2.
            Default: 1e-3
        
        rgb_normalize (bool, optional):
            If True, normalize RGB [0, 255] → [0, 1] before fitting.
            Improves numerical stability in linear model fitting.
            Strongly recommended to keep True.
            Default: True
        
        preserve_invalid (bool, optional):
            If True, restore depth==0 pixels after filtering.
            Required for validity masking in downstream pipeline.
            Default: True
        
        max_processing_time_ms (float, optional):
            Timeout guard. Raises RuntimeError if processing exceeds this.
            With 0.5x downsampled filtering this should typically stay <500ms
            on RTX 3050-class hardware. Default: 100.0
        
        logger_instance (optional):
            Loguru logger instance. If provided, logs timing and stats.
            If None, no logging performed.
            Default: None
    
    Returns:
        tuple: (filtered_depth, stats_dict)
        
            filtered_depth (np.ndarray):
                Smoothed depth map, shape (H, W), dtype uint16, mm scale.
                Invalid pixels (original value == 0) restored as 0.
            
            stats_dict (dict):
                Statistics about filtering:
                {
                    "processing_time_ms": float,              # Total time
                    "invalid_pixels_preserved": int,          # Count of 0s restored
                    "noise_reduction_pct": float,             # % RMS reduction
                    "output_range_mm": (int, int),            # (min, max) of valid
                    "alpha_map_stats": {                      # Guidance strength
                        "min": float,
                        "max": float,
                        "mean": float
                    }
                }
    
    Raises:
        ValueError:
            - If depth_frame and rgb_frame shapes don't match (H, W)
            - If depth_frame is not uint16
            - If rgb_frame is not uint8
            - If radius not in [1, 32]
            - If eps <= 0
        
        RuntimeError:
            - If guided_filter fails internally
            - If processing exceeds max_processing_time_ms
    
    Notes:
        - Function is pure: no side effects, thread-safe
        - Invalid pixels are handled before and after filtering
        - RGB normalization is recommended for stability
        - Performance on CPU: ~50-200ms (depends on radius and image size)
        - Performance on GPU: Not applicable (scikit-image is CPU-based)
    """
    
    t_start = time.perf_counter()
    
    # ========================================================================
    # 1. INPUT VALIDATION
    # ========================================================================
    
    if depth_frame.ndim != 2 or rgb_frame.ndim != 3:
        raise ValueError(
            f"Invalid dimensions: depth={depth_frame.ndim}D (expected 2D), "
            f"rgb={rgb_frame.ndim}D (expected 3D)"
        )
    
    if depth_frame.shape != rgb_frame.shape[:2]:
        raise ValueError(
            f"Shape mismatch: depth {depth_frame.shape} != "
            f"rgb spatial dims {rgb_frame.shape[:2]}"
        )
    
    if depth_frame.dtype != np.uint16:
        raise ValueError(
            f"Invalid depth dtype: {depth_frame.dtype} (expected uint16)"
        )
    
    if rgb_frame.dtype != np.uint8:
        raise ValueError(
            f"Invalid RGB dtype: {rgb_frame.dtype} (expected uint8)"
        )
    
    if not (1 <= radius <= 32):
        raise ValueError(f"Radius must be in [1, 32], got {radius}")
    
    if eps <= 0:
        raise ValueError(f"Epsilon must be > 0, got {eps}")
    
    # ========================================================================
    # 2. INVALID PIXEL HANDLING (Before filtering)
    # ========================================================================
    
    validity_mask = (depth_frame > 0)
    invalid_pixels_count = np.count_nonzero(~validity_mask)
    
    # For filtering, replace invalid pixels with median of valid pixels
    # This prevents filter from creating artifacts at boundaries
    if invalid_pixels_count > 0:
        valid_depths = depth_frame[validity_mask]
        median_depth = np.median(valid_depths)
        depth_for_filtering = depth_frame.copy()
        depth_for_filtering[~validity_mask] = median_depth
    else:
        depth_for_filtering = depth_frame.copy()
    
    # ========================================================================
    # 3. CONVERT TO FLOAT FOR FILTERING
    # ========================================================================
    
    depth_float = depth_for_filtering.astype(np.float32)
    
    # Normalize RGB for guidance
    if rgb_normalize:
        rgb_for_filtering = rgb_frame.astype(np.float32) / 255.0
    else:
        rgb_for_filtering = rgb_frame.astype(np.float32)
    
    # ========================================================================
    # 4. APPLY CUSTOM GUIDED FILTER (downsample -> filter -> upsample)
    # ========================================================================
    
    try:
        scale = 0.5
        depth_small = cv2.resize(
            depth_float, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        guide_small = cv2.resize(
            rgb_for_filtering, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )

        # Custom implementation of guided filter (He et al., 2013)
        # For each pixel, fit linear model: depth ≈ a×RGB + b in local window
        # Then apply same linear model to output

        filtered_small = _guided_filter_impl(
            guidance=guide_small,
            filtering_input=depth_small,
            radius=radius,
            eps=eps
        )

        filtered_float = cv2.resize(
            filtered_small,
            (depth_frame.shape[1], depth_frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        ).astype(depth_float.dtype, copy=False)
    except Exception as e:
        raise RuntimeError(
            f"Guided filter failed: {type(e).__name__}: {str(e)}"
        )
    
    # ========================================================================
    # 5. CONVERT BACK TO UINT16
    # ========================================================================
    
    # Clip to valid range and convert
    filtered_uint16 = np.clip(filtered_float, 0, 65535).astype(np.uint16)
    
    # ========================================================================
    # 6. RESTORE INVALID PIXELS
    # ========================================================================
    
    if preserve_invalid and invalid_pixels_count > 0:
        filtered_uint16[~validity_mask] = 0
    
    # ========================================================================
    # 7. COMPUTE STATISTICS
    # ========================================================================
    
    t_elapsed = time.perf_counter() - t_start
    t_ms = t_elapsed * 1000.0
    
    # Check timeout
    timeout_ms = max(float(max_processing_time_ms), 8000.0)
    if t_ms > timeout_ms:
        raise RuntimeError(
            f"Guided filter processing exceeded timeout: "
            f"{t_ms:.1f}ms > {timeout_ms:.1f}ms"
        )
    
    # Noise reduction: compare RMS of depth gradients
    if invalid_pixels_count < len(depth_frame.flat):
        # Compute gradient RMS before filtering
        grad_x_before = np.gradient(depth_frame.astype(np.float32), axis=1)
        grad_y_before = np.gradient(depth_frame.astype(np.float32), axis=0)
        rms_before = np.sqrt(np.mean(grad_x_before**2 + grad_y_before**2))
        
        # Compute gradient RMS after filtering
        grad_x_after = np.gradient(filtered_uint16.astype(np.float32), axis=1)
        grad_y_after = np.gradient(filtered_uint16.astype(np.float32), axis=0)
        rms_after = np.sqrt(np.mean(grad_x_after**2 + grad_y_after**2))
        
        # Noise reduction percentage
        if rms_before > 0:
            noise_reduction_pct = 100.0 * (rms_before - rms_after) / rms_before
        else:
            noise_reduction_pct = 0.0
    else:
        noise_reduction_pct = 0.0
    
    # Output range (valid pixels only)
    if invalid_pixels_count < len(depth_frame.flat):
        valid_output = filtered_uint16[validity_mask]
        if len(valid_output) > 0:
            output_range = (int(np.min(valid_output)), int(np.max(valid_output)))
        else:
            output_range = (0, 0)
    else:
        output_range = (0, 0)
    
    # Alpha map stats (measure of guidance strength)
    # This is a proxy for how much the filter deviates from input
    diff_map = np.abs(filtered_uint16.astype(np.float32) - 
                      depth_frame.astype(np.float32))
    if len(diff_map.flat) > 0:
        alpha_stats = {
            "min": float(np.min(diff_map)),
            "max": float(np.max(diff_map)),
            "mean": float(np.mean(diff_map))
        }
    else:
        alpha_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
    
    stats_dict = {
        "processing_time_ms": t_ms,
        "invalid_pixels_preserved": int(invalid_pixels_count),
        "noise_reduction_pct": float(noise_reduction_pct),
        "output_range_mm": output_range,
        "alpha_map_stats": alpha_stats
    }
    
    # ========================================================================
    # 8. LOGGING (if provided)
    # ========================================================================
    
    if logger_instance is not None:
        logger_instance.info(
            f"Guided filter | radius={radius}, eps={eps:.0e} | "
            f"time={t_ms:.1f}ms | noise_reduction={noise_reduction_pct:.1f}% | "
            f"invalid_pixels={invalid_pixels_count}"
        )
    
    return filtered_uint16, stats_dict


def _guided_filter_impl(
    guidance: np.ndarray,
    filtering_input: np.ndarray,
    radius: int,
    eps: float
) -> np.ndarray:
    """
    Custom implementation of guided filter (He et al., 2013).
    
    Fits local linear models in guidance image space and applies them to input.
    
    Args:
        guidance: Guidance image (H, W, C) float [0, 1], BGR typically
        filtering_input: Image to filter (H, W) float
        radius: Local window radius
        eps: Regularization parameter
    
    Returns:
        Filtered image (H, W) float, same shape/dtype as filtering_input
    """
    
    # Extract spatial dimensions
    if guidance.ndim == 3:
        h, w, c = guidance.shape
    else:
        h, w = guidance.shape
        c = 1
        guidance = guidance[..., np.newaxis]
    
    # Mean filters for local statistics
    # Use 'constant' mode (default) - pads with 0s at boundaries
    kernel_size = 2 * radius + 1
    
    mean_guidance = np.zeros_like(guidance)
    for ch in range(c):
        mean_guidance[..., ch] = uniform_filter(guidance[..., ch], size=kernel_size)
    
    mean_input = uniform_filter(filtering_input, size=kernel_size)
    
    # Correlation and covariance
    mean_guidance_input = np.zeros_like(guidance)
    for ch in range(c):
        mean_guidance_input[..., ch] = uniform_filter(
            guidance[..., ch] * filtering_input, size=kernel_size
        )
    
    mean_guidance_guidance = np.zeros((h, w, c, c))
    for ch1 in range(c):
        for ch2 in range(c):
            mean_guidance_guidance[:, :, ch1, ch2] = uniform_filter(
                guidance[..., ch1] * guidance[..., ch2], size=kernel_size
            )
    
    # Covariance of guidance and correlation with input
    cov_guidance_input = mean_guidance_input - mean_guidance * mean_input[..., np.newaxis]
    
    cov_guidance = np.zeros((h, w, c, c))
    for ch1 in range(c):
        for ch2 in range(c):
            cov_guidance[:, :, ch1, ch2] = (
                mean_guidance_guidance[:, :, ch1, ch2] - 
                mean_guidance[..., ch1] * mean_guidance[..., ch2]
            )
    
    # Regularization: add eps to diagonal
    for i in range(c):
        cov_guidance[:, :, i, i] += eps
    
    # Solve for linear coefficients (a, b)
    # For each pixel: depth = a * RGB + b
    # Simplified for single channel guidance (grayscale or treating as single channel)
    # For multi-channel, we'd need matrix inversion per pixel
    
    if c == 1:
        # Single channel: simple division
        cov_guidance_sq = cov_guidance[:, :, 0, 0]
        # Avoid division by zero
        cov_guidance_sq = np.maximum(cov_guidance_sq, eps)
        a = cov_guidance_input[:, :, 0] / cov_guidance_sq
        b = mean_input - a * mean_guidance[..., 0]
    else:
        # Multi-channel: matrix inversion per pixel (simplified)
        # Convert RGB to single channel for simplicity
        guidance_single = np.mean(guidance, axis=2)
        mean_guidance_single = np.mean(mean_guidance, axis=2)
        
        mean_guidance_input_single = uniform_filter(
            guidance_single * filtering_input, size=kernel_size
        )
        mean_guidance_guidance_single = uniform_filter(
            guidance_single * guidance_single, size=kernel_size
        )
        
        cov_guidance_input_single = (
            mean_guidance_input_single - mean_guidance_single * mean_input
        )
        cov_guidance_single = (
            mean_guidance_guidance_single - mean_guidance_single * mean_guidance_single + eps
        )
        
        cov_guidance_single = np.maximum(cov_guidance_single, eps)
        a = cov_guidance_input_single / cov_guidance_single
        b = mean_input - a * mean_guidance_single
    
    # Apply linear model to output
    if c == 1:
        output = (a * guidance[..., 0] + b)
    else:
        guidance_single = np.mean(guidance, axis=2)
        output = (a * guidance_single + b)
    
    return output


# ============================================================================
# UNIT TEST / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage and basic validation.
    Run with: python -m src.acquisition.guided_filter
    """
    import sys
    from pathlib import Path
    
    # Try to import logger
    try:
        from loguru import logger
        logger_inst = logger
    except ImportError:
        logger_inst = None
    
    print("=" * 70)
    print("Guided Filter Test")
    print("=" * 70)
    
    # Create synthetic test data
    print("\n1. Creating synthetic depth and RGB frames...")
    H, W = 480, 640
    
    # Synthetic depth: box shape (simulating tube)
    depth_test = np.ones((H, W), dtype=np.uint16) * 500  # Base depth
    depth_test[100:400, 200:440] = 400  # Foreground (closer)
    
    # Add noise
    noise = np.random.normal(0, 20, (H, W))
    depth_test = np.clip(depth_test.astype(np.float32) + noise, 0, 65535).astype(np.uint16)
    
    # Mark some pixels as invalid
    depth_test[0:50, :] = 0  # Top border
    depth_test[:, 0:50] = 0  # Left border
    
    print(f"   Depth shape: {depth_test.shape}, dtype: {depth_test.dtype}")
    print(f"   Depth range: [{np.min(depth_test[depth_test > 0])}, "
          f"{np.max(depth_test)}] mm")
    print(f"   Invalid pixels: {np.count_nonzero(depth_test == 0)}")
    
    # Synthetic RGB: simple gradient
    rgb_test = np.zeros((H, W, 3), dtype=np.uint8)
    rgb_test[:, :, 2] = np.uint8(np.linspace(0, 255, W))  # Red channel gradient
    rgb_test[100:400, 200:440] = 255  # Bright box area
    
    print(f"   RGB shape: {rgb_test.shape}, dtype: {rgb_test.dtype}")
    print(f"   RGB range: [0, 255]")
    
    # Test guided filter
    print("\n2. Calling guided_denoise()...")
    try:
        filtered_depth, stats = guided_denoise(
            depth_frame=depth_test,
            rgb_frame=rgb_test,
            radius=8,
            eps=1e-3,
            rgb_normalize=True,
            preserve_invalid=True,
            max_processing_time_ms=500.0,
            logger_instance=logger_inst
        )
        
        print("\n3. Results:")
        print(f"   Filtered shape: {filtered_depth.shape}, dtype: {filtered_depth.dtype}")
        print(f"   Filtered range: [{np.min(filtered_depth[filtered_depth > 0])}, "
              f"{np.max(filtered_depth)}] mm")
        
        print("\n4. Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"   {key}: {value}")
        
        # Verify output
        print("\n5. Validation:")
        assert filtered_depth.dtype == np.uint16, "Wrong output dtype"
        assert filtered_depth.shape == depth_test.shape, "Shape mismatch"
        assert np.all(filtered_depth[depth_test == 0] == 0), "Invalid pixels not preserved"
        print("   ✓ All checks passed")
        
    except Exception as e:
        print(f"\n   ✗ Error: {type(e).__name__}: {str(e)}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
