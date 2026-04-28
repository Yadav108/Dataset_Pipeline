#!/usr/bin/env python3
"""
Image Quality Analysis Tool - Comprehensive evaluation of capture quality metrics
Analyzes blur, coverage, depth stability, and segmentation confidence across dataset.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import statistics

from config.parser import get_config
from loguru import logger


@dataclass
class QualityMetrics:
    """Container for comprehensive quality metrics."""
    image_id: str
    blur_score: float
    coverage_ratio: float
    sam_iou_score: Optional[float] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    depth_mean: Optional[float] = None
    depth_variance: Optional[float] = None
    mask_area_px: Optional[int] = None
    bbox_area_px: Optional[int] = None
    roi_size_px: Optional[int] = None
    
    def is_sharp(self, threshold: float = None) -> bool:
        threshold = threshold or 40.0
        return self.blur_score >= threshold
    
    def is_good_coverage(self, threshold: float = None) -> bool:
        threshold = threshold or 0.35
        return self.coverage_ratio >= threshold
    
    def is_good_sam(self, threshold: float = None) -> bool:
        if self.sam_iou_score is None:
            return None
        threshold = threshold or 0.60
        return self.sam_iou_score >= threshold


class ImageQualityAnalyzer:
    """Analyze and report on image quality across entire dataset."""
    
    def __init__(self):
        self.cfg = get_config()
        self.metrics_list: list[QualityMetrics] = []
        self.rejection_reasons = defaultdict(int)
    
    def compute_blur_score(self, image_path: Path) -> float:
        """Compute Laplacian variance (blur score) for an image.
        
        Higher values indicate sharper images.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return 0.0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(variance)
        except Exception as e:
            logger.warning(f"Error computing blur score for {image_path}: {e}")
            return 0.0
    
    def load_metadata(self, metadata_path: Path) -> Optional[dict]:
        """Load metadata JSON file."""
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading metadata {metadata_path}: {e}")
            return None
    
    def load_depth_data(self, depth_path: Path) -> Optional[np.ndarray]:
        """Load depth data from NPZ or PNG."""
        try:
            if str(depth_path).endswith('.npy'):
                return np.load(str(depth_path))
            elif str(depth_path).endswith('.png'):
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                return depth.astype(np.float32) * 0.001  # Convert to meters
            elif str(depth_path).endswith('.npz'):
                data = np.load(str(depth_path))
                if 'depth' in data:
                    return data['depth'].astype(np.float32) * 0.001
        except Exception as e:
            logger.warning(f"Error loading depth data {depth_path}: {e}")
        return None
    
    def analyze_image_set(self, dataset_dir: Path) -> list[QualityMetrics]:
        """Recursively analyze all images in dataset directory."""
        self.metrics_list = []
        image_count = 0
        
        # Find all metadata files
        metadata_files = list(dataset_dir.rglob("*_metadata.json"))
        logger.info(f"Found {len(metadata_files)} metadata files to analyze")
        
        for metadata_path in metadata_files:
            image_id = metadata_path.stem.replace("_metadata", "")
            metadata = self.load_metadata(metadata_path)
            
            if metadata is None:
                continue
            
            # Find corresponding image file
            image_dir = metadata_path.parent
            rgb_path = image_dir / f"{image_id}_rgb.png"
            depth_path = image_dir / f"{image_id}_depth.npy"
            
            if not rgb_path.exists():
                logger.warning(f"RGB file not found: {rgb_path}")
                continue
            
            # Compute blur score
            blur_score = self.compute_blur_score(rgb_path)
            
            # Load depth stats
            coverage_ratio = metadata.get("coverage_ratio", 0.0)
            sam_iou_score = metadata.get("sam_iou_score", None)
            mask_area = metadata.get("mask_area_px", 0)
            bbox_area = metadata.get("bbox_area_px", 0)
            
            # Extract ROI size
            bbox = metadata.get("bbox", {})
            roi_size = bbox.get("w", 0) * bbox.get("h", 0) if bbox else 0
            
            # Load depth data if available
            depth_data = None
            depth_min, depth_max, depth_mean, depth_variance = None, None, None, None
            
            if depth_path.exists():
                depth_data = self.load_depth_data(depth_path)
                if depth_data is not None:
                    valid_mask = (depth_data > 0.05) & (depth_data < 1.0)
                    if np.any(valid_mask):
                        valid_depths = depth_data[valid_mask]
                        depth_min = float(np.min(valid_depths))
                        depth_max = float(np.max(valid_depths))
                        depth_mean = float(np.mean(valid_depths))
                        depth_variance = float(np.var(valid_depths))
            
            # Create metrics
            metrics = QualityMetrics(
                image_id=image_id,
                blur_score=blur_score,
                coverage_ratio=coverage_ratio,
                sam_iou_score=sam_iou_score,
                depth_min=depth_min,
                depth_max=depth_max,
                depth_mean=depth_mean,
                depth_variance=depth_variance,
                mask_area_px=mask_area,
                bbox_area_px=bbox_area,
                roi_size_px=roi_size,
            )
            
            self.metrics_list.append(metrics)
            image_count += 1
            
            if image_count % 100 == 0:
                logger.info(f"Analyzed {image_count} images...")
        
        logger.info(f"Total images analyzed: {image_count}")
        return self.metrics_list
    
    def generate_report(self) -> dict:
        """Generate comprehensive quality report."""
        if not self.metrics_list:
            return {"error": "No metrics to analyze"}
        
        # Blur statistics
        blur_scores = [m.blur_score for m in self.metrics_list]
        sharp_count = sum(1 for m in self.metrics_list if m.is_sharp())
        
        # Coverage statistics
        coverage_ratios = [m.coverage_ratio for m in self.metrics_list]
        good_coverage_count = sum(1 for m in self.metrics_list if m.is_good_coverage())
        
        # SAM IoU statistics
        sam_scores = [m.sam_iou_score for m in self.metrics_list if m.sam_iou_score is not None]
        good_sam_count = sum(1 for m in self.metrics_list if m.is_good_sam())
        
        # ROI size statistics
        roi_sizes = [m.roi_size_px for m in self.metrics_list if m.roi_size_px is not None]
        
        # Depth statistics (if available)
        depth_means = [m.depth_mean for m in self.metrics_list if m.depth_mean is not None]
        depth_variances = [m.depth_variance for m in self.metrics_list if m.depth_variance is not None]
        
        report = {
            "total_images": len(self.metrics_list),
            "blur_analysis": {
                "threshold": self.cfg.pipeline.blur_threshold,
                "sharp_count": sharp_count,
                "sharp_percentage": round(100 * sharp_count / len(self.metrics_list), 2),
                "mean": round(statistics.mean(blur_scores), 2),
                "median": round(statistics.median(blur_scores), 2),
                "min": round(min(blur_scores), 2),
                "max": round(max(blur_scores), 2),
                "stdev": round(statistics.stdev(blur_scores), 2) if len(blur_scores) > 1 else 0,
                "distribution_percentiles": {
                    "p10": round(np.percentile(blur_scores, 10), 2),
                    "p25": round(np.percentile(blur_scores, 25), 2),
                    "p50": round(np.percentile(blur_scores, 50), 2),
                    "p75": round(np.percentile(blur_scores, 75), 2),
                    "p90": round(np.percentile(blur_scores, 90), 2),
                }
            },
            "coverage_analysis": {
                "threshold": self.cfg.pipeline.min_coverage_ratio,
                "good_coverage_count": good_coverage_count,
                "good_coverage_percentage": round(100 * good_coverage_count / len(self.metrics_list), 2),
                "mean": round(statistics.mean(coverage_ratios), 4),
                "median": round(statistics.median(coverage_ratios), 4),
                "min": round(min(coverage_ratios), 4),
                "max": round(max(coverage_ratios), 4),
                "stdev": round(statistics.stdev(coverage_ratios), 4) if len(coverage_ratios) > 1 else 0,
            },
            "sam_analysis": {
                "threshold": self.cfg.pipeline.sam_iou_threshold,
                "images_with_sam_scores": len(sam_scores),
                "good_sam_count": good_sam_count,
                "good_sam_percentage": round(100 * good_sam_count / len(sam_scores), 2) if sam_scores else 0,
                "mean": round(statistics.mean(sam_scores), 4) if sam_scores else None,
                "median": round(statistics.median(sam_scores), 4) if sam_scores else None,
                "min": round(min(sam_scores), 4) if sam_scores else None,
                "max": round(max(sam_scores), 4) if sam_scores else None,
            },
            "roi_analysis": {
                "total_with_roi": len(roi_sizes),
                "mean_size_px": round(statistics.mean(roi_sizes), 0) if roi_sizes else 0,
                "median_size_px": round(statistics.median(roi_sizes), 0) if roi_sizes else 0,
                "min_size_px": min(roi_sizes) if roi_sizes else 0,
                "max_size_px": max(roi_sizes) if roi_sizes else 0,
            },
            "depth_analysis": {
                "images_with_depth": len(depth_means),
                "mean_depth_m": round(statistics.mean(depth_means), 4) if depth_means else None,
                "depth_variance_mean": round(statistics.mean(depth_variances), 6) if depth_variances else None,
            }
        }
        
        return report
    
    def export_detailed_metrics(self, output_path: Path) -> None:
        """Export detailed metrics to CSV for further analysis."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Header
            f.write("image_id,blur_score,is_sharp,coverage_ratio,is_good_coverage,"
                   "sam_iou_score,is_good_sam,roi_size_px,depth_mean_m,depth_variance\n")
            
            # Data rows
            for m in self.metrics_list:
                f.write(
                    f"{m.image_id},"
                    f"{m.blur_score:.2f},"
                    f"{m.is_sharp()},"
                    f"{m.coverage_ratio:.4f},"
                    f"{m.is_good_coverage()},"
                    f"{m.sam_iou_score or ''},"
                    f"{m.is_good_sam()},"
                    f"{m.roi_size_px or ''},"
                    f"{m.depth_mean or ''},"
                    f"{m.depth_variance or ''}\n"
                )
        
        logger.info(f"Exported detailed metrics to {output_path}")


def main():
    """Main entry point."""
    cfg = get_config()
    dataset_dir = Path(cfg.storage.root_dir)
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return
    
    analyzer = ImageQualityAnalyzer()
    logger.info(f"Analyzing dataset in {dataset_dir}...")
    
    # Analyze all images
    metrics = analyzer.analyze_image_set(dataset_dir)
    
    # Generate report
    report = analyzer.generate_report()
    
    # Print report
    print("\n" + "="*80)
    print("IMAGE QUALITY ANALYSIS REPORT")
    print("="*80)
    print(json.dumps(report, indent=2))
    
    # Export detailed metrics
    output_csv = Path("quality_metrics.csv")
    analyzer.export_detailed_metrics(output_csv)
    
    print(f"\n✓ Detailed metrics exported to {output_csv}")
    print("="*80)


if __name__ == "__main__":
    main()
