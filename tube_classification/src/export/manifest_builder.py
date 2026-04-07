import json
import csv
from pathlib import Path

from loguru import logger
from config.parser import get_config


class ManifestBuilder:
    """Scan cleaned dataset directory and build master CSV index.
    
    Recursively finds all annotations and builds a structured CSV
    manifest linking all images, depth data, masks, and metadata.
    """
    
    def __init__(self):
        """Initialize builder with config."""
        self.cfg = get_config()
    
    def build(self) -> Path:
        """Build dataset manifest CSV from cleaned annotations.
        
        Scans all metadata files and creates a comprehensive index
        with paths and metadata for all images.
        
        Returns:
            Path to the generated manifest CSV file
        """
        root = Path(self.cfg.storage.root_dir)
        cleaned_ann_root = root / "cleaned" / "annotations"
        cleaned_raw_root = root / "cleaned" / "raw"
        manifest_dir = root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = manifest_dir / "dataset_index.csv"
        
        # Collect all rows
        rows = []
        
        # Find all metadata files recursively
        for metadata_file in cleaned_ann_root.rglob("*_metadata.json"):
            try:
                # Parse metadata
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Extract fields
                image_id = metadata["image_id"]
                class_id = metadata["class_id"]
                volume_ml = metadata["volume_ml"]
                coverage_ratio = metadata["coverage_ratio"]
                bbox = metadata["bbox"]
                # Optional: sam_iou_score (may not be present in older metadata)
                sam_iou_score = metadata.get("sam_iou_score", None)
                
                # Derive session_id from parent folder path
                session_id = metadata_file.parent.name
                
                # Build relative paths
                rgb_path = (
                    cleaned_raw_root
                    / class_id
                    / session_id
                    / f"{image_id}_rgb.png"
                )
                depth_path = (
                    cleaned_raw_root
                    / class_id
                    / session_id
                    / f"{image_id}_depth.npy"
                )
                mask_path = (
                    cleaned_ann_root
                    / class_id
                    / session_id
                    / f"{image_id}_mask.png"
                )
                
                # Build row
                row = {
                    "image_id": image_id,
                    "class_id": class_id,
                    "volume_ml": volume_ml,
                    "session_id": session_id,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "mask_path": str(mask_path),
                    "bbox_x": bbox["x"],
                    "bbox_y": bbox["y"],
                    "bbox_w": bbox["w"],
                    "bbox_h": bbox["h"],
                    "coverage_ratio": coverage_ratio,
                    "sam_iou_score": sam_iou_score,
                }
                rows.append(row)
            
            except Exception as e:
                logger.warning(f"Failed to process {metadata_file.name}: {e}")
                continue
        
        # Write CSV
        fieldnames = [
            "image_id",
            "class_id",
            "volume_ml",
            "session_id",
            "rgb_path",
            "depth_path",
            "mask_path",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "coverage_ratio",
            "sam_iou_score",
        ]
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Manifest built: {len(rows)} images → {output_path}")
        
        return output_path
