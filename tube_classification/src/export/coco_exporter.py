import json
from pathlib import Path
import datetime

from loguru import logger
from config.parser import get_config


class COCOExporter:
    """Export cleaned dataset to COCO JSON format.
    
    Converts cleaned annotations into standard COCO format for
    compatibility with detection and segmentation frameworks.
    """
    
    def __init__(self):
        """Initialize exporter with config."""
        self.cfg = get_config()
    
    def export(self) -> Path:
        """Export cleaned dataset to COCO format.
        
        Scans all metadata files and generates a complete COCO JSON
        dataset with images, annotations, and categories.
        
        Returns:
            Path to the generated COCO annotations JSON file
        """
        root = Path(self.cfg.storage.root_dir)
        cleaned_ann_root = root / "cleaned" / "annotations"
        export_dir = root / "exports" / "coco"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = export_dir / "annotations.json"
        
        # Collect all metadata files
        metadata_files = sorted(cleaned_ann_root.rglob("*_metadata.json"))
        
        # Build category mapping
        unique_classes = set()
        class_to_id = {}
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                class_id = metadata["class_id"]
                unique_classes.add(class_id)
            except Exception:
                continue
        
        # Assign integer category IDs (1-indexed)
        categories = []
        for i, class_id in enumerate(sorted(unique_classes), start=1):
            class_to_id[class_id] = i
            categories.append({
                "id": i,
                "name": class_id,
            })
        
        # Build images and annotations
        images = []
        annotations = []
        annotation_id = 1
        
        for image_idx, metadata_file in enumerate(metadata_files, start=1):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                # Extract fields
                image_id = metadata["image_id"]
                class_id = metadata["class_id"]
                image_shape = metadata["image_shape"]
                bbox_data = metadata["bbox"]
                
                # Build image entry
                image_entry = {
                    "id": image_idx,
                    "file_name": f"{image_id}_rgb.png",
                    "width": image_shape["width"],
                    "height": image_shape["height"],
                }
                images.append(image_entry)
                
                # Build annotation entry
                x = bbox_data["x"]
                y = bbox_data["y"]
                w = bbox_data["w"]
                h = bbox_data["h"]
                area = w * h
                
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_idx,
                    "category_id": class_to_id[class_id],
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
                annotations.append(annotation_entry)
                annotation_id += 1
            
            except Exception as e:
                logger.warning(f"Failed to process {metadata_file.name}: {e}")
                continue
        
        # Assemble COCO dataset
        coco_dict = {
            "info": {
                "description": "Tube Classification Dataset",
                "version": "1.0",
                "date_created": datetime.datetime.now().isoformat(),
            },
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        
        # Write to file
        with open(output_path, "w") as f:
            json.dump(coco_dict, f, indent=2)
        
        logger.info(
            f"COCO export complete: {len(images)} images → {output_path}"
        )
        
        return output_path
