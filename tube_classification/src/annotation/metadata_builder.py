import numpy as np
import datetime

from loguru import logger
from pydantic import BaseModel, Field, field_validator


def build_metadata(
    image_id: str,
    class_id: str,
    volume_ml: float,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    rgb_shape: tuple[int, int, int],
    sam_iou_score: float | None = None,
    calibration_metadata: dict | None = None,
) -> dict:
    """Build structured annotation metadata from capture data.
    
    Computes mask statistics and packages all capture information
    into a standardized dictionary structure.
    
    Args:
        image_id: Unique image identifier
        class_id: Tube class identifier
        volume_ml: Volume in milliliters
        bbox: Bounding box as (x, y, width, height)
        mask: Binary segmentation mask
        rgb_shape: RGB image shape as (height, width, channels)
        sam_iou_score: MobileSAM predicted IoU score (0.0-1.0), optional
        calibration_metadata: Distance calibration info, optional
        
    Returns:
        Dictionary with structured annotation metadata
    """
    # Compute mask statistics
    mask_area = int(np.sum(mask > 0))
    x, y, w, h = bbox
    bbox_area = w * h
    coverage_ratio = round(mask_area / bbox_area, 4) if bbox_area > 0 else 0.0
    
    metadata = {
        "image_id": image_id,
        "class_id": class_id,
        "volume_ml": volume_ml,
        "timestamp": datetime.datetime.now().isoformat(),
        "image_shape": {
            "height": rgb_shape[0],
            "width": rgb_shape[1],
            "channels": rgb_shape[2],
        },
        "bbox": {
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
        },
        "mask_area_px": mask_area,
        "bbox_area_px": bbox_area,
        "coverage_ratio": coverage_ratio,
    }
    
    # Add SAM IoU score if provided
    if sam_iou_score is not None:
        metadata["sam_iou_score"] = round(sam_iou_score, 4)
    
    # Add distance calibration info if provided
    if calibration_metadata:
        metadata.update(calibration_metadata)

    return metadata


# ---------------------------------------------------------------------------
# Supporting value-object models — derived from the JSON shapes written by
# pipeline.py so these models stay round-trippable with stored annotations.
# ---------------------------------------------------------------------------

class FillLevelMetadata(BaseModel):
    """Structured representation of a single fill-level detection result."""

    level: str
    confidence: str
    boundary_ratio: float


class CalibrationMetadata(BaseModel):
    """Distance-calibration snapshot captured at session time."""

    measured_distance_m: float | None = None
    calibrated_depth_min_m: float | None = None
    calibrated_depth_max_m: float | None = None


class QualityMetrics(BaseModel):
    """Per-frame quality scores written into annotation metadata by the pipeline."""

    image_id: str
    class_id: str
    blur_score: float = 0.0
    coverage_ratio: float = 0.0
    sam_iou_score: float = 0.0
    depth_variance: float = 0.0
    quality_score: float = 0.0


# ---------------------------------------------------------------------------
# Primary structured annotation models.
# ---------------------------------------------------------------------------

class InstanceMetadata(BaseModel):
    """Structured annotation for a single tube instance within an image.

    Each captured ROI produces one InstanceMetadata record.  Optional fields
    are populated only when the corresponding pipeline stage ran successfully
    (e.g. fill-level detection, distance calibration).
    """

    instance_id: str
    class_id: str
    volume_ml: float
    mask_file: str
    bbox: dict[str, int]          # keys: x, y, w, h
    mask_area_px: int
    bbox_area_px: int
    coverage_ratio: float = Field(default=0.0, description="mask_area / bbox_area, rounded to 4 dp")
    rotation_angle_deg: float = 0.0
    occlusion_percent: float = 0.0
    distance_m: float | None = None
    sam_iou_score: float | None = None
    fill_level: FillLevelMetadata | None = None
    distance_calibration: CalibrationMetadata | None = None

    @field_validator("coverage_ratio", mode="before")
    @classmethod
    def _round_coverage(cls, v: float) -> float:
        return round(float(v), 4)


class ImageMetadata(BaseModel):
    """Top-level annotation record for one captured image.

    Groups all per-instance annotations together with image-level context
    (timestamp, resolution, capture mode) and optional quality metrics.
    Intended as the canonical in-memory form of each ``*_metadata.json`` file.
    """

    image_id: str
    timestamp: str                          # ISO-8601
    image_shape: dict[str, int]             # keys: height, width, channels
    capture_mode: str = "unknown"
    lighting_condition: str = "unknown"
    quality_metrics: QualityMetrics | None = None
    instances: list[InstanceMetadata] = []


def build_image_metadata(
    image_id: str,
    rgb_image: np.ndarray,
    instances: list[dict],
    capture_mode: str = "unknown",
    lighting_condition: str = "unknown",
    quality_metrics: dict | None = None,
    timestamp: str | None = None,
) -> ImageMetadata:
    """Build a validated ImageMetadata record from raw capture data.

    Computes per-instance mask statistics, constructs InstanceMetadata objects
    for each entry in the instances list, and assembles the top-level
    ImageMetadata.  Pure function — no file I/O.

    Args:
        image_id: Unique identifier for the image.
        rgb_image: RGB frame as a uint8 HxWx3 numpy array; used only to read shape.
        instances: List of dicts, one per segmented tube ROI.  Required keys:
            instance_id, class_id, volume_ml, mask_file, mask (np.ndarray),
            bbox (x, y, w, h).  Optional keys: rotation_angle_deg,
            occlusion_percent, distance_m, sam_iou_score, fill_level (dict),
            distance_calibration (dict).
        capture_mode: Pipeline mode string (e.g. "single_side").
        lighting_condition: Lighting descriptor (e.g. "controlled").
        quality_metrics: Dict matching QualityMetrics fields, or None.
        timestamp: ISO-8601 string; defaults to current UTC time if omitted.

    Returns:
        Fully validated ImageMetadata instance.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    ndim = rgb_image.ndim
    image_shape = {
        "height": int(rgb_image.shape[0]),
        "width": int(rgb_image.shape[1]),
        "channels": int(rgb_image.shape[2]) if ndim == 3 else 1,
    }

    if not instances:
        logger.warning(
            "build_image_metadata: empty instances list for image_id=%r", image_id
        )

    instance_models: list[InstanceMetadata] = []
    for inst in instances:
        mask: np.ndarray = inst["mask"]
        x, y, w, h = inst["bbox"]
        mask_area_px = int((mask > 0).sum())
        bbox_area_px = int(w * h)
        coverage_ratio = round(mask_area_px / bbox_area_px, 4) if bbox_area_px > 0 else 0.0

        fill_level_raw = inst.get("fill_level")
        fill_level = FillLevelMetadata(**fill_level_raw) if fill_level_raw is not None else None

        cal_raw = inst.get("distance_calibration")
        distance_calibration = CalibrationMetadata(**cal_raw) if cal_raw is not None else None

        instance_models.append(
            InstanceMetadata(
                instance_id=inst["instance_id"],
                class_id=inst["class_id"],
                volume_ml=float(inst["volume_ml"]),
                mask_file=inst["mask_file"],
                bbox={"x": x, "y": y, "w": w, "h": h},
                mask_area_px=mask_area_px,
                bbox_area_px=bbox_area_px,
                coverage_ratio=coverage_ratio,
                rotation_angle_deg=float(inst.get("rotation_angle_deg", 0.0)),
                occlusion_percent=float(inst.get("occlusion_percent", 0.0)),
                distance_m=inst.get("distance_m"),
                sam_iou_score=inst.get("sam_iou_score"),
                fill_level=fill_level,
                distance_calibration=distance_calibration,
            )
        )

    qm = QualityMetrics(**quality_metrics) if quality_metrics is not None else None

    return ImageMetadata(
        image_id=image_id,
        timestamp=timestamp,
        image_shape=image_shape,
        capture_mode=capture_mode,
        lighting_condition=lighting_condition,
        quality_metrics=qm,
        instances=instance_models,
    )
