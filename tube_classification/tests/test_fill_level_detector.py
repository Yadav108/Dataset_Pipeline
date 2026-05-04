import pytest
import numpy as np
from src.acquisition.fill_level_detector import (
    FillLevelDetector,
    FillLevelConfig,
    FillLevelResult,
    FillLevel,
)


def test_half_tube_edge_detected():
    """Validates Sobel edge detection classifies a clear mid-tube boundary as HALF."""
    config = FillLevelConfig()
    detector = FillLevelDetector(config)

    height, width = 100, 50
    boundary_row = int(height * ((config.half_threshold + config.empty_threshold) / 2.0))

    top_air = np.full((boundary_row, width), 230, dtype=np.uint8)
    bottom_liquid = np.full((height - boundary_row, width), 20, dtype=np.uint8)
    gray = np.vstack([top_air, bottom_liquid])
    roi = np.stack([gray] * 3, axis=-1)

    result = detector.detect(roi)

    assert isinstance(result, FillLevelResult)
    assert result.level == FillLevel.HALF
    assert result.confidence == "edge_detected"
    assert config.half_threshold <= result.boundary_ratio <= config.empty_threshold


def test_empty_tube():
    """Validates a bright tube ROI is classified as EMPTY."""
    config = FillLevelConfig()
    detector = FillLevelDetector(config)

    height, width = 100, 50
    boundary_row = int(height * ((config.empty_threshold + 1.0) / 2.0))

    top_air = np.full((boundary_row, width), 230, dtype=np.uint8)
    bottom_air = np.full((height - boundary_row, width), 230, dtype=np.uint8)
    gray = np.vstack([top_air, bottom_air])
    roi = np.stack([gray] * 3, axis=-1)

    result = detector.detect(roi)

    assert isinstance(result, FillLevelResult)
    assert result.level == FillLevel.EMPTY


def test_full_tube():
    """Validates a dark tube ROI is classified as FULL."""
    config = FillLevelConfig()
    detector = FillLevelDetector(config)

    height, width = 100, 50
    boundary_row = int(height * (config.half_threshold / 2.0))

    top_liquid = np.full((boundary_row, width), 20, dtype=np.uint8)
    bottom_liquid = np.full((height - boundary_row, width), 20, dtype=np.uint8)
    gray = np.vstack([top_liquid, bottom_liquid])
    roi = np.stack([gray] * 3, axis=-1)

    result = detector.detect(roi)

    assert isinstance(result, FillLevelResult)
    assert result.level == FillLevel.FULL


def test_brightness_fallback_triggered():
    """Validates uniform mid-brightness ROI triggers brightness fallback path."""
    config = FillLevelConfig()
    detector = FillLevelDetector(config)

    height, width = 100, 50
    split_row = height // 2

    upper = np.full((split_row, width), 128, dtype=np.uint8)
    lower = np.full((height - split_row, width), 128, dtype=np.uint8)
    gray = np.vstack([upper, lower])
    roi = np.stack([gray] * 3, axis=-1)

    result = detector.detect(roi)

    assert isinstance(result, FillLevelResult)
    assert result.confidence == "brightness_fallback"
    assert result.boundary_ratio == -1.0


def test_invalid_none_raises():
    """Validates passing None ROI raises ValueError."""
    config = FillLevelConfig()
    detector = FillLevelDetector(config)

    with pytest.raises(ValueError):
        detector.detect(None)


def test_invalid_zero_height_raises():
    """Validates zero-height ROI raises ValueError."""
    config = FillLevelConfig()
    detector = FillLevelDetector(config)
    roi = np.zeros((0, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        detector.detect(roi)


def test_invalid_zero_width_raises():
    """Validates zero-width ROI raises ValueError."""
    config = FillLevelConfig()
    detector = FillLevelDetector(config)
    roi = np.zeros((100, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        detector.detect(roi)
