"""
Unit tests for preprocessing module (PROMPT 1-4)
"""

import pytest
import numpy as np
import cv2
from src.acquisition.preprocessing import (
    preprocess_depth_bilateral,
    normalize_depth,
    denormalize_depth,
    TemporalSmoothingFilter,
    compute_quality_metrics,
    QualityMetrics
)


# ============================================================================
# PROMPT 1: BILATERAL FILTER TESTS
# ============================================================================

class TestBilateralFilter:
    """Test bilateral filtering (5 tests)"""
    
    def test_valid_depth_frame(self):
        """Test with valid depth frame"""
        depth = np.random.randint(100, 1000, (480, 848), dtype=np.uint16)
        filtered, mask = preprocess_depth_bilateral(depth)
        
        assert filtered.shape == (480, 848)
        assert filtered.dtype == np.uint16
        assert mask.shape == (480, 848)
        assert mask.dtype == np.uint8
    
    def test_edge_preservation(self):
        """Test edge preservation"""
        # Create depth with sharp edge
        depth = np.zeros((480, 848), dtype=np.uint16)
        depth[:, :400] = 500
        depth[:, 400:] = 800
        
        filtered, _ = preprocess_depth_bilateral(depth)
        
        # Check edge region (around column 400)
        edge_diff = np.abs(filtered[:, 400] - filtered[:, 399])
        assert np.mean(edge_diff) > 50  # Sharp transition preserved
    
    def test_noise_reduction(self):
        """Test noise reduction"""
        # Create noisy depth
        depth = np.full((480, 848), 500, dtype=np.uint16)
        depth += np.random.randint(-50, 50, (480, 848)).astype(np.uint16)
        
        filtered, _ = preprocess_depth_bilateral(depth)
        
        # Variance should decrease
        noise_before = np.var(depth)
        noise_after = np.var(filtered)
        assert noise_after < noise_before
    
    def test_invalid_shape(self):
        """Test invalid shape raises error"""
        depth = np.zeros((640, 480), dtype=np.uint16)
        with pytest.raises(ValueError, match="Invalid shape"):
            preprocess_depth_bilateral(depth)
    
    def test_invalid_diameter(self):
        """Test invalid diameter raises error"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        with pytest.raises(ValueError, match="Diameter must be odd"):
            preprocess_depth_bilateral(depth, diameter=24)


# ============================================================================
# PROMPT 2: NORMALIZATION TESTS
# ============================================================================

class TestNormalization:
    """Test depth normalization (5 tests)"""
    
    def test_normalize_basic(self):
        """Test basic normalization"""
        depth = np.array([[170, 220]], dtype=np.uint16).reshape(1, 1)
        depth = np.tile(depth, (480, 848))
        
        normalized, meta = normalize_depth(depth, 170, 270)
        
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        assert meta['valid_ratio'] > 0.9
    
    def test_invertibility(self):
        """Test round-trip invertibility"""
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        
        normalized, meta = normalize_depth(depth, 170, 270)
        recovered = denormalize_depth(normalized, 170, 270)
        
        # Check round-trip error < 1mm
        error = np.abs(depth.astype(np.float32) - recovered.astype(np.float32))
        assert np.mean(error) < 1.0
    
    def test_invalid_range(self):
        """Test invalid min/max raises error"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        with pytest.raises(ValueError, match="min_mm.*max_mm"):
            normalize_depth(depth, 300, 200)
    
    def test_zero_pixels_handling(self):
        """Test zero pixel handling"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        depth[100:200, 100:200] = 200  # Some valid pixels
        
        normalized, meta = normalize_depth(depth, 170, 270, invalid_pixel_value=-1.0)
        
        assert np.all(normalized[0:100, 0:100] == -1.0)
        assert meta['invalid_count'] > 0
    
    def test_denormalize_bounds_check(self):
        """Test denormalization bounds check"""
        normalized = np.array([[1.5]], dtype=np.float32)  # Out of bounds
        with pytest.raises(ValueError, match="outside"):
            denormalize_depth(normalized, 170, 270)


# ============================================================================
# PROMPT 3: TEMPORAL SMOOTHING TESTS
# ============================================================================

class TestTemporalSmoothing:
    """Test temporal smoothing (5 tests)"""
    
    def test_filter_initialization(self):
        """Test filter initialization"""
        smoother = TemporalSmoothingFilter(alpha=0.2)
        assert smoother.alpha == 0.2
        assert smoother.frames_processed == 0
    
    def test_first_frame_passthrough(self):
        """Test first frame is passed through"""
        smoother = TemporalSmoothingFilter()
        depth = np.full((480, 848), 200, dtype=np.uint16)
        
        smoothed, metrics = smoother.smooth(depth)
        
        assert np.allclose(smoothed, depth)
        assert metrics['jitter_reduction_pct'] == 0.0
    
    def test_jitter_reduction(self):
        """Test jitter reduction"""
        smoother = TemporalSmoothingFilter(alpha=0.3)
        
        # Frame 1: baseline
        frame1 = np.full((480, 848), 200, dtype=np.uint16)
        smoother.smooth(frame1)
        
        # Frame 2: with jitter
        frame2 = np.full((480, 848), 220, dtype=np.uint16)
        frame2[100:110, 100:110] = 300  # Add spike
        
        smoothed, metrics = smoother.smooth(frame2)
        
        # Smoothed should be between frame1 and frame2
        assert np.mean(smoothed) < np.mean(frame2)
        assert np.mean(smoothed) > np.mean(frame1)
    
    def test_invalid_alpha(self):
        """Test invalid alpha raises error"""
        with pytest.raises(ValueError, match="alpha must be"):
            TemporalSmoothingFilter(alpha=0.8)
    
    def test_reset(self):
        """Test filter reset"""
        smoother = TemporalSmoothingFilter()
        depth = np.full((480, 848), 200, dtype=np.uint16)
        smoother.smooth(depth)
        
        assert len(smoother.history) > 0
        smoother.reset()
        assert len(smoother.history) == 0


# ============================================================================
# PROMPT 4: QUALITY METRICS TESTS
# ============================================================================

class TestQualityMetrics:
    """Test quality metrics (10 tests)"""
    
    def test_metrics_computation(self):
        """Test basic metrics computation"""
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        metrics = compute_quality_metrics(depth, rgb, frame_id=1)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.frame_id == 1
        assert 0 <= metrics.quality_score <= 10
    
    def test_depth_metrics(self):
        """Test depth-specific metrics"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        rgb = np.full((480, 848, 3), 128, dtype=np.uint8)
        
        metrics = compute_quality_metrics(depth, rgb)
        
        assert metrics.depth_mean_mm == 200.0
        assert metrics.depth_std_mm == 0.0
        assert metrics.valid_pixel_ratio == 1.0
    
    def test_blur_detection(self):
        """Test blur score"""
        rgb_sharp = np.zeros((480, 848), dtype=np.uint8)
        rgb_sharp[100:200, 100:200] = 255  # Sharp edge
        rgb_sharp = cv2.cvtColor(rgb_sharp, cv2.COLOR_GRAY2BGR)
        
        rgb_blurry = cv2.GaussianBlur(rgb_sharp, (31, 31), 0)
        
        depth = np.full((480, 848), 200, dtype=np.uint16)
        
        metrics_sharp = compute_quality_metrics(depth, rgb_sharp)
        metrics_blurry = compute_quality_metrics(depth, rgb_blurry)
        
        # Sharp should have higher blur score
        assert metrics_sharp.blur_score > metrics_blurry.blur_score
    
    def test_quality_score_formula(self):
        """Test quality score is in valid range"""
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(50, 200, (480, 848, 3), dtype=np.uint8)
        
        metrics = compute_quality_metrics(depth, rgb)
        
        assert 0 <= metrics.quality_score <= 10
    
    def test_mask_metrics(self):
        """Test mask-specific metrics"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        rgb = np.full((480, 848, 3), 128, dtype=np.uint8)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255  # 100x100 square
        
        metrics = compute_quality_metrics(depth, rgb, mask=mask)
        
        assert metrics.mask_area_px == 10000
        assert 0 <= metrics.mask_compactness <= 1
        assert metrics.mask_coverage_ratio > 0
    
    def test_serialization(self):
        """Test JSON serialization"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        rgb = np.full((480, 848, 3), 128, dtype=np.uint8)
        
        metrics = compute_quality_metrics(depth, rgb)
        
        json_str = metrics.to_json()
        assert isinstance(json_str, str)
        assert 'quality_score' in json_str
    
    def test_invalid_depth_shape(self):
        """Test invalid depth shape raises error"""
        depth = np.zeros((640, 480), dtype=np.uint16)
        rgb = np.zeros((480, 848, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Invalid depth shape"):
            compute_quality_metrics(depth, rgb)
    
    def test_invalid_rgb_shape(self):
        """Test invalid RGB shape raises error"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        rgb = np.zeros((480, 848), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Invalid RGB shape"):
            compute_quality_metrics(depth, rgb)
    
    def test_all_metrics_present(self):
        """Test all 14+ metrics are computed"""
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        metrics = compute_quality_metrics(depth, rgb)
        
        # Check all required fields
        required_fields = [
            'valid_pixel_ratio', 'depth_min_mm', 'depth_max_mm', 'depth_mean_mm',
            'depth_std_mm', 'depth_snr_db', 'depth_uniformity',
            'blur_score', 'contrast_ratio', 'edge_density', 'saturation_score',
            'hue_variance', 'illumination_level', 'quality_score'
        ]
        
        for field in required_fields:
            assert hasattr(metrics, field)
            assert getattr(metrics, field) is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPreprocessingIntegration:
    """Integration tests across modules"""
    
    def test_pipeline_flow(self):
        """Test full preprocessing pipeline"""
        # Generate test data
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        # Step 1: Bilateral filter
        filtered, mask = preprocess_depth_bilateral(depth)
        assert filtered.shape == depth.shape
        
        # Step 2: Normalization
        normalized, meta = normalize_depth(filtered, 170, 270)
        assert 0 <= np.nanmin(normalized) and np.nanmax(normalized) <= 1
        
        # Step 3: Temporal smoothing
        smoother = TemporalSmoothingFilter()
        smoothed1, _ = smoother.smooth(filtered)
        smoothed2, metrics = smoother.smooth(filtered)
        assert metrics['jitter_reduction_pct'] >= 0
        
        # Step 4: Quality metrics
        quality = compute_quality_metrics(smoothed2, rgb)
        assert 0 <= quality.quality_score <= 10
    
    def test_performance_targets(self):
        """Test all processing meets performance targets"""
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        import time
        
        # Bilateral: <50ms
        start = time.time()
        preprocess_depth_bilateral(depth)
        assert (time.time() - start) < 0.050
        
        # Normalization: <10ms
        start = time.time()
        normalize_depth(depth, 170, 270)
        assert (time.time() - start) < 0.010
        
        # Temporal smoothing: <5ms
        smoother = TemporalSmoothingFilter()
        smoother.smooth(depth)
        start = time.time()
        smoother.smooth(depth)
        assert (time.time() - start) < 0.005
        
        # Quality metrics: <100ms
        start = time.time()
        compute_quality_metrics(depth, rgb)
        assert (time.time() - start) < 0.100


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
