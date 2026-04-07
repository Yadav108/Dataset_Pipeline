"""Tests for confirmation preview."""

import numpy as np
import pytest
from tube_classification.src.preview.confirmation_preview import (
    ConfirmationPreviewRenderer, QualityMetrics, ConfirmationAction
)


@pytest.fixture
def grid_map():
    return {
        'rows': 5,
        'cols': 10,
    }


@pytest.fixture
def confirmation(grid_map):
    config = {
        'zoom_factor': 4,
        'depth_stable_variance_mm': 5.0,
        'blur_threshold': 100.0,
        'mask_confidence_threshold': 0.85,
        'panel_width_px': 300,
    }
    return ConfirmationPreviewRenderer(grid_map, config)


def test_get_slot_roi(confirmation):
    """Test ROI extraction for slot."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Slot (0, 0) should be top-left
    x1, y1, x2, y2 = confirmation._get_slot_roi(frame, (0, 0))
    assert x1 == 0
    assert y1 == 0
    assert x2 == 128  # 1280 / 10
    assert y2 == 144  # 720 / 5


def test_extract_roi(confirmation):
    """Test ROI extraction."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 255
    
    roi = confirmation._extract_roi(frame, (0, 0))
    assert roi.shape == (144, 128, 3)


def test_calculate_blur_score(confirmation):
    """Test blur score calculation."""
    # Sharp image should have high blur score
    sharp_roi = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    blur_score = confirmation._calculate_blur_score(sharp_roi)
    assert blur_score >= 0


def test_calculate_depth_metrics(confirmation):
    """Test depth metrics calculation."""
    depth_roi = np.full((100, 100), 330000, dtype=np.float32)
    
    mean, variance, stable = confirmation._calculate_depth_metrics(depth_roi)
    
    assert mean > 0
    assert variance >= 0
    assert isinstance(stable, bool)


def test_calculate_mask_confidence(confirmation):
    """Test mask confidence calculation."""
    # All foreground
    mask = np.full((100, 100), 255, dtype=np.uint8)
    confidence = confirmation._calculate_mask_confidence(mask)
    assert confidence == 1.0
    
    # Half foreground
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, :] = 255
    confidence = confirmation._calculate_mask_confidence(mask)
    assert confidence == pytest.approx(0.5, abs=0.01)


def test_calculate_quality_metrics(confirmation):
    """Test full metrics calculation."""
    rgb_roi = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    depth_roi = np.full((100, 100), 330000, dtype=np.float32)
    mask_roi = np.full((100, 100), 255, dtype=np.uint8)
    
    metrics = confirmation._calculate_quality_metrics(rgb_roi, depth_roi, mask_roi)
    
    assert isinstance(metrics, QualityMetrics)
    assert metrics.depth_mean > 0
    assert metrics.overall_quality in ['GOOD', 'FAIR', 'POOR']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
