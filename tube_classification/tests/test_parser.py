import pytest
from pathlib import Path
from pydantic import ValidationError

from config.parser import AppConfig, CameraConfig, load_config, get_config

@pytest.fixture
def config_path():
    return Path(__file__).parent.parent / "config" / "config.yaml"


def test_valid_config_loads(config_path):
    """Test that valid config loads successfully."""
    config = load_config(config_path)
    assert isinstance(config, AppConfig)
    assert config.camera.fps == 30


def test_invalid_fps_raises():
    """Test that invalid fps value raises ValidationError."""
    invalid_config = {
        "camera": {
            "width": 848,
            "height": 480,
            "fps": 45,
            "depth_min_m": 0.32,
            "depth_max_m": 0.56,
        },
        "storage": {
            "root_dir": "data/captures",
            "registry_path": "config/registry.yaml",
            "sam_weights_path": "models/mobile_sam.pt",
            "session_prefix": "session",
        },
        "pipeline": {
            "stability_frames": 10,
            "blur_threshold": 100.0,
            "duplicate_hash_threshold": 8,
            "min_coverage_ratio": 0.6,
            "min_roi_area_px": 500,
            "sam_iou_threshold": 0.88,
            "depth_stability_threshold": 0.002,
        },
    }
    with pytest.raises(ValidationError):
        AppConfig(**invalid_config)


def test_depth_range_invalid():
    """Test that depth_min_m >= depth_max_m raises ValidationError."""
    invalid_config = {
        "camera": {
            "width": 848,
            "height": 480,
            "fps": 30,
            "depth_min_m": 0.6,
            "depth_max_m": 0.3,
        },
        "storage": {
            "root_dir": "data/captures",
            "registry_path": "config/registry.yaml",
            "sam_weights_path": "models/mobile_sam.pt",
            "session_prefix": "session",
        },
        "pipeline": {
            "stability_frames": 10,
            "blur_threshold": 100.0,
            "duplicate_hash_threshold": 8,
            "min_coverage_ratio": 0.6,
            "min_roi_area_px": 500,
            "sam_iou_threshold": 0.88,
            "depth_stability_threshold": 0.002,
        },
    }
    with pytest.raises(ValidationError):
        AppConfig(**invalid_config)


def test_depth_negative_raises():
    """Test that negative depth value raises ValidationError."""
    invalid_config = {
        "camera": {
            "width": 848,
            "height": 480,
            "fps": 30,
            "depth_min_m": -0.1,
            "depth_max_m": 0.56,
        },
        "storage": {
            "root_dir": "data/captures",
            "registry_path": "config/registry.yaml",
            "sam_weights_path": "models/mobile_sam.pt",
            "session_prefix": "session",
        },
        "pipeline": {
            "stability_frames": 10,
            "blur_threshold": 100.0,
            "duplicate_hash_threshold": 8,
            "min_coverage_ratio": 0.6,
            "min_roi_area_px": 500,
            "sam_iou_threshold": 0.88,
            "depth_stability_threshold": 0.002,
        },
    }
    with pytest.raises(ValidationError):
        AppConfig(**invalid_config)


def test_get_config_singleton():
    """Test that get_config returns the same object on multiple calls."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2
