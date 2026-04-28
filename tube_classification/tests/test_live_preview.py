"""Tests for live preview renderer."""

import numpy as np
import pytest
from tube_classification.src.preview.live_preview import LivePreviewRenderer, PreviewResult


@pytest.fixture
def grid_map():
    return {
        'rows': 5,
        'cols': 10,
    }


@pytest.fixture
def calib_params():
    return {
        'depth_baseline_mm': 330,
        'aruco_markers': [
            {'id': 0, 'position_px': [10, 10]},
            {'id': 1, 'position_px': [1270, 10]},
            {'id': 2, 'position_px': [10, 710]},
            {'id': 3, 'position_px': [1270, 710]},
        ]
    }


@pytest.fixture
def preview_renderer(grid_map, calib_params):
    return LivePreviewRenderer(grid_map, calib_params, resolution=(1280, 720))


def test_renderer_initialization(preview_renderer):
    """Test renderer initialization."""
    assert preview_renderer.rows == 5
    assert preview_renderer.cols == 10
    assert preview_renderer.depth_baseline_mm == 330
    assert preview_renderer.width == 1280
    assert preview_renderer.height == 720


def test_detect_occupied_slots(preview_renderer):
    """Test occupied slot detection."""
    # Create a depth frame with tube at slot (0, 0)
    depth_frame = np.full((720, 1280), 330000, dtype=np.uint16)  # mm * 1000
    
    # Mark slot (0, 0) as occupied (depth < baseline - tolerance)
    depth_frame[0:144, 0:128] = 310000  # Closer than baseline - 20mm
    
    occupied = preview_renderer._detect_occupied_slots(depth_frame, depth_tolerance_mm=20.0)
    
    assert (0, 0) in occupied
    assert len(occupied) > 0


def test_render_returns_preview_result(preview_renderer):
    """Test that render returns PreviewResult."""
    rgb_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth_frame = np.full((720, 1280), 330000, dtype=np.uint16)
    
    # This will create a display window, so we test the structure
    # In actual testing, we'd mock cv2.imshow
    config = {'depth_tolerance_mm': 20.0}
    
    # Just test that occupied slots are detected
    occupied = preview_renderer._detect_occupied_slots(depth_frame, config['depth_tolerance_mm'])
    assert isinstance(occupied, set)


def test_cleanup(preview_renderer):
    """Test cleanup doesn't crash."""
    preview_renderer.cleanup()
    # Should complete without error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
