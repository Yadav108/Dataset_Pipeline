import json
from pathlib import Path

from loguru import logger
import shutil
from config.parser import get_config


class YOLOExporter:
    """Export cleaned dataset to YOLO format.
    
    Converts bounding boxes to YOLO normalized format and organizes
    images and labels for YOLO training.
    """
    
    def __init__(self):
        """Initialize exporter with config."""
        self.cfg = get_config()
    
    def export(self) -> Path:
        """Export cleaned dataset to YOLO format.
        
        Creates normalized bounding box labels and organizes images
        for YOLO training framework.
        
        Returns:
            Path to the export directory
        """
        root = Path(self.cfg.storage.root_dir)
        cleaned_ann_root = root / "cleaned" / "annotations"
        cleaned_raw_root = root / "cleaned" / "raw"
        export_dir = root / "exports" / "yolo"
        labels_dir = export_dir / "labels"
        images_dir = export_dir / "images"
        
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all metadata files
        metadata_files = sorted(cleaned_ann_root.rglob("*_metadata.json"))
        
        # Build sorted class list
        unique_classes = set()
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                class_id = metadata["class_id"]
                unique_classes.add(class_id)
            except Exception:
                continue
        
        sorted_classes = sorted(unique_classes)
        class_to_index = {class_id: i for i, class_id in enumerate(sorted_classes)}
        
        # Write classes.txt
        classes_file = export_dir / "classes.txt"
        with open(classes_file, "w") as f:
            for class_id in sorted_classes:
                f.write(f"{class_id}\n")
        
        # Process each metadata file
        image_count = 0
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Extract fields
                image_id = metadata["image_id"]
                class_id = metadata["class_id"]
                bbox_data = metadata["bbox"]
                image_shape = metadata["image_shape"]
                
                # Get image dimensions
                img_w = image_shape["width"]
                img_h = image_shape["height"]
                
                # Extract bbox
                x = bbox_data["x"]
                y = bbox_data["y"]
                w = bbox_data["w"]
                h = bbox_data["h"]
                
                # Normalize to YOLO format
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                # Get class index
                class_index = class_to_index[class_id]
                
                # Write label file
                label_file = labels_dir / f"{image_id}.txt"
                with open(label_file, "w") as f:
                    f.write(
                        f"{class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
                    )
                
                # Find and copy RGB image
                session_id = metadata_file.parent.name
                rgb_src = (
                    cleaned_raw_root
                    / class_id
                    / session_id
                    / f"{image_id}_rgb.png"
                )
                
                if rgb_src.exists():
                    rgb_dst = images_dir / f"{image_id}_rgb.png"
                    shutil.copy2(str(rgb_src), str(rgb_dst))
                
                image_count += 1
            
            except Exception as e:
                logger.warning(f"Failed to process {metadata_file.name}: {e}")
                continue
        
        logger.info(
            f"YOLO export complete: {image_count} images → {export_dir}"
        )
        
        return export_dir
