from pathlib import Path
from typing import Literal, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, model_validator


class CameraConfig(BaseModel):
    width: Literal[640, 848, 1280]
    height: Literal[360, 480, 720]
    fps: Literal[6, 15, 30, 60]
    depth_min_m: float
    depth_max_m: float

    @model_validator(mode="after")
    def validate_depth_range(self):
        if self.depth_min_m <= 0 or self.depth_max_m <= 0:
            raise ValueError("depth_min_m and depth_max_m must be positive")
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError("depth_min_m must be less than depth_max_m")
        return self


class StorageConfig(BaseModel):
    root_dir: Path
    registry_path: Path
    sam_weights_path: Path
    # Path to MobileSAM weights file (mobile_sam.pt)
    session_prefix: str


class PipelineConfig(BaseModel):
    stability_frames: int
    blur_threshold: float
    duplicate_hash_threshold: int
    min_coverage_ratio: float
    # Minimum mask-to-bbox coverage ratio to accept an annotation.
    # Below this → bbox is considered inaccurate → image rejected.
    # Typical value: 0.6
    min_roi_area_px: int
    max_roi_area_px: int = 18000
    min_tube_dim_px: int = 8
    max_tube_dim_px: int = 70
    min_tube_length_px: int = 40
    max_tube_length_px: int = 300
    min_solidity: float = 0.35
    border_margin_px: int = 20
    
    # Top-down mode shape filter
    top_min_roi_area_px: int = 200
    top_max_roi_area_px: int = 6000
    top_min_tube_dim_px: int = 8
    top_max_tube_dim_px: int = 80
    top_max_circularity_ratio: float = 2.5
    top_min_solidity: float = 0.35
    top_border_margin_px: int = 30
    
    # Top-down depth zone
    top_depth_min_m: float = 0.10
    top_depth_max_m: float = 0.35
    
    # Checkpoint configuration for crash recovery
    checkpoint_interval: int = 25
    
    # Live preview window
    show_preview: bool = True
    
    # Per-class capture target
    target_images_per_class: int = 500
    
    # Background removal (final cleaning step)
    background_removal: bool = True
    
    sam_iou_threshold: float
    depth_stability_threshold: float
    # Maximum allowed mean absolute difference (in meters) between
    # consecutive depth frames to consider the scene stable.
    # Typical value: 0.002


class AppConfig(BaseModel):
    camera: CameraConfig
    storage: StorageConfig
    pipeline: PipelineConfig


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

_config: Optional[AppConfig] = None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    global _config

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    _config = AppConfig(**data)
    logger.info(f"Config loaded from {path}")
    return _config


def get_config() -> AppConfig:
    global _config

    if _config is None:
        load_config()

    return _config
