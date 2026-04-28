"""
PROMPT 8: Pipeline Integration Module

Hooks all preprocessing modules (PROMPTS 1-7) into the main pipeline.
Provides configuration-driven preprocessing pipeline.
"""

import logging
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from config.parser import get_config

from src.acquisition.preprocessing import (
    preprocess_depth_bilateral,
    normalize_depth,
    denormalize_depth,
    TemporalSmoothingFilter,
    compute_quality_metrics,
    QualityMetrics
)
from src.acquisition.advanced_preprocessing import (
    inpaint_depth_telea,
    DepthGuidedMaskRefinement,
    PNG16Compressor
)
from src.acquisition.guided_filter import guided_denoise


logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Preprocessing pipeline modes."""
    NONE = "none"              # No preprocessing
    BASIC = "basic"            # PROMPTS 1-4 (core)
    ADVANCED = "advanced"      # PROMPTS 1-7 (all)
    EXPORT = "export"          # PNG16 compression for export


@dataclass
class PreprocessingPipeline:
    """Complete preprocessing pipeline with configuration-driven behavior."""
    
    cfg = None  # Will be set during __init__
    mode: ProcessingMode = ProcessingMode.BASIC
    
    # Core modules
    bilateral_enabled: bool = True
    normalization_enabled: bool = True
    temporal_smoothing_enabled: bool = True
    quality_metrics_enabled: bool = True
    guided_filter_enabled: bool = True
    
    # Advanced modules
    inpainting_enabled: bool = False
    mask_refinement_enabled: bool = False
    png16_export_enabled: bool = False
    
    # State
    temporal_smoother: Optional[TemporalSmoothingFilter] = None
    frame_count: int = 0
    
    # Metrics tracking
    total_frames_processed: int = 0
    total_processing_time_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize preprocessing pipeline from config."""
        self.cfg = get_config()
        
        # Load settings from config
        if hasattr(self.cfg, 'preprocessing'):
            prep = self.cfg.preprocessing
            
            # Core modules
            self.bilateral_enabled = prep.bilateral.enabled
            self.guided_filter_enabled = prep.guided_filter.enabled
            self.normalization_enabled = prep.normalization.enabled
            self.temporal_smoothing_enabled = prep.temporal_smoothing.enabled
            self.quality_metrics_enabled = prep.quality_metrics.enabled
            
            # Advanced modules
            self.inpainting_enabled = prep.inpainting.enabled
            self.mask_refinement_enabled = prep.mask_refinement.enabled
            self.png16_export_enabled = prep.png16_export.enabled
            
            # Determine mode
            if self.inpainting_enabled or self.mask_refinement_enabled:
                self.mode = ProcessingMode.ADVANCED
            elif self.bilateral_enabled or self.guided_filter_enabled or self.temporal_smoothing_enabled:
                self.mode = ProcessingMode.BASIC
            else:
                self.mode = ProcessingMode.NONE
        
        # Initialize temporal smoother if needed
        if self.temporal_smoothing_enabled:
            alpha = self.cfg.preprocessing.temporal_smoothing.alpha
            window_size = self.cfg.preprocessing.temporal_smoothing.window_size
            threshold = self.cfg.preprocessing.temporal_smoothing.jitter_threshold_mm
            
            self.temporal_smoother = TemporalSmoothingFilter(
                alpha=alpha,
                window_size=window_size,
                jitter_threshold_mm=threshold,
                logger_instance=logger
            )
        
        logger.info(f"Preprocessing pipeline initialized: mode={self.mode.value}")
    
    def process_depth_frame(
        self,
        depth_frame: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        frame_id: int = 0,
        compute_metrics: bool = True
    ) -> Tuple[np.ndarray, Optional[QualityMetrics], Dict[str, Any]]:
        """
        Process depth frame through complete preprocessing pipeline.
        
        Pipeline stages (configured via config.yaml):
        1. Bilateral filtering (noise reduction + edge preservation)
        2. Guided filtering (RGB-guided denoising, requires rgb_frame)
        3. Inpainting (fill holes)
        4. Temporal smoothing (jitter reduction)
        5. Quality metrics (if compute_metrics=True and rgb_frame provided)
        
        Args:
            depth_frame: Input depth map (uint16, mm scale, shape H×W)
            rgb_frame: Optional RGB frame for guided filter and quality metrics
                       (uint8, BGR, same spatial shape as depth)
            mask: Optional segmentation mask for quality metrics
            frame_id: Frame identifier for logging
            compute_metrics: Whether to compute quality metrics
        
        Returns:
            (processed_depth, quality_metrics, stats)
            - processed_depth: Preprocessed depth frame (uint16, mm)
            - quality_metrics: QualityMetrics object or None
            - stats: Dictionary with processing statistics including:
                * processing_steps: List of applied stages
                * timing_ms: Timing for each stage
                * guided_filter_noise_reduction_pct: % noise reduction (if guided filter applied)
                * quality_score: Quality score (if metrics computed)
        
        Raises:
            ValueError: If input shape/dtype invalid or shapes don't match
            RuntimeError: If processing fails (e.g., timeout, guided filter failure)
        """
        import time
        start_time = time.time()
        
        self.frame_count += 1
        stats = {
            'frame_id': frame_id,
            'processing_steps': [],
            'timing_ms': {}
        }
        
        try:
            # Validate input
            if depth_frame.ndim != 2:
                raise ValueError(f"Invalid depth shape: {depth_frame.shape}")
            if depth_frame.dtype != np.uint16:
                raise ValueError(f"Invalid depth dtype: {depth_frame.dtype}")
            
            # Validate RGB frame if provided
            if rgb_frame is not None:
                if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
                    raise ValueError(f"Invalid RGB shape: {rgb_frame.shape}")
                if rgb_frame.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB dtype: {rgb_frame.dtype}")
                # RGB is resized to match depth where operations require spatial match.
            
            current_depth = depth_frame.copy()
            
            # STEP 1: Bilateral filtering (PROMPT 1)
            if self.bilateral_enabled:
                step_start = time.time()
                
                current_depth, validity_mask = preprocess_depth_bilateral(
                    current_depth,
                    spatial_sigma=self.cfg.preprocessing.bilateral.spatial_sigma,
                    range_sigma=self.cfg.preprocessing.bilateral.range_sigma,
                    diameter=self.cfg.preprocessing.bilateral.diameter,
                    iterations=self.cfg.preprocessing.bilateral.iterations,
                    min_valid_ratio=self.cfg.preprocessing.bilateral.min_valid_ratio,
                    max_processing_time_ms=self.cfg.preprocessing.bilateral.max_processing_time_ms,
                    logger_instance=logger
                )
                
                step_time = (time.time() - step_start) * 1000
                stats['processing_steps'].append('bilateral_filter')
                stats['timing_ms']['bilateral_filter'] = step_time
                logger.debug(f"Bilateral filtering: {step_time:.1f}ms")
            
            # STEP 1.5: Guided Filter (PROMPT 2 + integrated here)
            if self.guided_filter_enabled:
                if rgb_frame is None:
                    logger.warning(
                        "Guided filter enabled but rgb_frame not provided. Skipping."
                    )
                if rgb_frame is not None:
                    try:
                        step_start = time.time()
                        
                        # Resize RGB to match current depth spatial dimensions.
                        # Guided filter requires matching spatial dims for guidance
                        rgb_for_filtering = rgb_frame
                        target_h, target_w = current_depth.shape
                        if rgb_frame.shape[:2] != (target_h, target_w):
                            import cv2
                            rgb_for_filtering = cv2.resize(
                                rgb_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                            )
                        
                        try:
                            current_depth, gf_stats = guided_denoise(
                                depth_frame=current_depth,
                                rgb_frame=rgb_for_filtering,
                                radius=self.cfg.preprocessing.guided_filter.radius,
                                eps=self.cfg.preprocessing.guided_filter.eps,
                                rgb_normalize=self.cfg.preprocessing.guided_filter.rgb_normalize,
                                preserve_invalid=self.cfg.preprocessing.guided_filter.preserve_invalid,
                                max_processing_time_ms=self.cfg.preprocessing.guided_filter.max_processing_time_ms,
                                logger_instance=logger
                            )
                        except RuntimeError as e:
                            logger.warning(f"Guided filter skipped: {e}")
                            gf_stats = {"skipped": True}
                         
                        step_time = (time.time() - step_start) * 1000
                        stats['processing_steps'].append('guided_filter')
                        stats['timing_ms']['guided_filter'] = step_time
                        if gf_stats.get('skipped'):
                            logger.debug(f"Guided filter: {step_time:.1f}ms, skipped=True")
                        else:
                            stats['guided_filter_noise_reduction_pct'] = gf_stats['noise_reduction_pct']
                            logger.debug(
                                f"Guided filter: {step_time:.1f}ms, "
                                f"noise_reduction={gf_stats['noise_reduction_pct']:.1f}%"
                            )
                    except Exception as e:
                        logger.error(f"Guided filter failed: {e}")
                        raise
            
            # STEP 2: Depth inpainting (PROMPT 5)
            if self.inpainting_enabled:
                step_start = time.time()
                
                current_depth = inpaint_depth_telea(
                    current_depth,
                    inpainting_radius=self.cfg.preprocessing.inpainting.radius,
                    min_valid_ratio=self.cfg.preprocessing.inpainting.min_valid_ratio,
                    logger_instance=logger
                )
                
                step_time = (time.time() - step_start) * 1000
                stats['processing_steps'].append('inpainting')
                stats['timing_ms']['inpainting'] = step_time
                logger.debug(f"Depth inpainting: {step_time:.1f}ms")
            
            # STEP 3: Temporal smoothing (PROMPT 3)
            if self.temporal_smoothing_enabled and self.temporal_smoother:
                step_start = time.time()
                
                current_depth, temporal_metrics = self.temporal_smoother.smooth(
                    current_depth,
                    detect_outliers=True
                )
                
                step_time = (time.time() - step_start) * 1000
                stats['processing_steps'].append('temporal_smoothing')
                stats['timing_ms']['temporal_smoothing'] = step_time
                stats['jitter_reduction_pct'] = temporal_metrics.get('jitter_reduction_pct', 0)
                logger.debug(f"Temporal smoothing: {step_time:.1f}ms")
            
            # STEP 4: Compute quality metrics (PROMPT 4)
            quality_metrics = None
            if self.quality_metrics_enabled and compute_metrics and rgb_frame is not None:
                step_start = time.time()
                
                # Resize RGB to match current depth spatial dimensions for quality metrics.
                rgb_for_metrics = rgb_frame
                target_h, target_w = current_depth.shape
                if rgb_frame.shape[:2] != (target_h, target_w):
                    import cv2
                    rgb_for_metrics = cv2.resize(
                        rgb_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )
                
                quality_metrics = compute_quality_metrics(
                    current_depth,
                    rgb_for_metrics,
                    mask=mask,
                    frame_id=frame_id,
                    logger_instance=logger
                )
                
                step_time = (time.time() - step_start) * 1000
                stats['processing_steps'].append('quality_metrics')
                stats['timing_ms']['quality_metrics'] = step_time
                stats['quality_score'] = quality_metrics.quality_score
                logger.debug(f"Quality metrics: {step_time:.1f}ms (score={quality_metrics.quality_score:.2f})")
            
            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            stats['total_time_ms'] = total_time
            
            self.total_frames_processed += 1
            self.total_processing_time_ms += total_time
            
            logger.debug(
                f"Frame {self.frame_count} preprocessing complete: "
                f"steps={len(stats['processing_steps'])}, "
                f"total_time={total_time:.1f}ms"
            )
            
            return current_depth, quality_metrics, stats
        
        except Exception as e:
            logger.error(f"Preprocessing failed for frame {frame_id}: {e}")
            raise
    
    def refine_mask(
        self,
        depth_frame: np.ndarray,
        mask: np.ndarray,
        frame_id: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Refine SAM segmentation mask using depth geometry (PROMPT 6).
        
        Args:
            depth_frame: Depth frame (uint16, mm)
            mask: SAM segmentation mask (uint8)
            frame_id: Frame identifier
        
        Returns:
            (refined_mask, refinement_stats)
        
        Raises:
            ValueError: If input invalid
            RuntimeError: If refinement fails
        """
        if not self.mask_refinement_enabled:
            logger.debug("Mask refinement disabled, returning original mask")
            return mask, {'refined': False, 'reason': 'disabled'}
        
        import time
        start_time = time.time()
        
        try:
            refiner = DepthGuidedMaskRefinement(
                depth_frame,
                mask,
                depth_sigma=self.cfg.preprocessing.mask_refinement.depth_sigma,
                morpho_kernel_size=self.cfg.preprocessing.mask_refinement.morpho_kernel_size,
                min_mask_area=self.cfg.preprocessing.mask_refinement.min_mask_area,
                logger_instance=logger
            )
            
            refined_mask, refinement_metrics = refiner.refine()
            
            total_time = (time.time() - start_time) * 1000
            refinement_metrics['total_time_ms'] = total_time
            refinement_metrics['refined'] = True
            
            logger.debug(
                f"Frame {frame_id} mask refinement: "
                f"IoU+{refinement_metrics['iou_improvement_pct']:.1f}%, "
                f"correlation={refinement_metrics['depth_correlation']:.2f}, "
                f"time={total_time:.1f}ms"
            )
            
            return refined_mask, refinement_metrics
        
        except Exception as e:
            logger.error(f"Mask refinement failed for frame {frame_id}: {e}")
            return mask, {'refined': False, 'error': str(e)}
    
    def export_depth_frame(
        self,
        depth_frame: np.ndarray,
        output_path,
        frame_id: int = 0
    ) -> Dict[str, Any]:
        """
        Export depth frame as PNG16 for storage (PROMPT 7).
        
        Args:
            depth_frame: Depth frame (uint16, mm)
            output_path: Output file path (Path or str)
            frame_id: Frame identifier
        
        Returns:
            Export metrics dictionary
        
        Raises:
            IOError: If export fails
        """
        if not self.png16_export_enabled:
            logger.debug("PNG16 export disabled, skipping export")
            return {'exported': False, 'reason': 'disabled'}
        
        try:
            from pathlib import Path
            output_path = Path(output_path)
            
            metrics = PNG16Compressor.save_depth_png16(
                depth_frame,
                output_path,
                logger_instance=logger
            )
            
            metrics['exported'] = True
            metrics['frame_id'] = frame_id
            
            logger.debug(
                f"Frame {frame_id} PNG16 export: "
                f"file_size={metrics['file_size_bytes']} bytes, "
                f"compression={metrics['compression_ratio']:.1f}%"
            )
            
            return metrics
        
        except Exception as e:
            logger.error(f"PNG16 export failed for frame {frame_id}: {e}")
            return {'exported': False, 'error': str(e)}
    
    def reset(self) -> None:
        """Reset preprocessing state (e.g., between sessions)."""
        self.frame_count = 0
        
        if self.temporal_smoother:
            self.temporal_smoother.reset()
        
        logger.info("Preprocessing pipeline reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing pipeline statistics."""
        avg_time = (
            self.total_processing_time_ms / self.total_frames_processed
            if self.total_frames_processed > 0
            else 0.0
        )
        
        stats = {
            'mode': self.mode.value,
            'total_frames': self.total_frames_processed,
            'total_time_ms': self.total_processing_time_ms,
            'avg_time_per_frame_ms': avg_time,
            'enabled_steps': {
                'bilateral_filter': self.bilateral_enabled,
                'normalization': self.normalization_enabled,
                'temporal_smoothing': self.temporal_smoothing_enabled,
                'quality_metrics': self.quality_metrics_enabled,
                'inpainting': self.inpainting_enabled,
                'mask_refinement': self.mask_refinement_enabled,
                'png16_export': self.png16_export_enabled,
            }
        }
        
        if self.temporal_smoother:
            stats['temporal_smoothing_stats'] = self.temporal_smoother.get_statistics()
        
        return stats


# Singleton instance
_pipeline_instance: Optional[PreprocessingPipeline] = None


def get_preprocessing_pipeline() -> PreprocessingPipeline:
    """Get or create preprocessing pipeline singleton."""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = PreprocessingPipeline()
    
    return _pipeline_instance


def process_frame(
    depth_frame: np.ndarray,
    rgb_frame: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    frame_id: int = 0,
    compute_metrics: bool = True
) -> Tuple[np.ndarray, Optional[QualityMetrics], Dict[str, Any]]:
    """
    Convenience function to process frame through global pipeline.
    
    Args:
        depth_frame: Input depth (uint16, mm)
        rgb_frame: Optional RGB frame for metrics
        mask: Optional segmentation mask for metrics
        frame_id: Frame identifier
        compute_metrics: Whether to compute quality metrics
    
    Returns:
        (processed_depth, quality_metrics, stats)
    """
    pipeline = get_preprocessing_pipeline()
    return pipeline.process_depth_frame(
        depth_frame, rgb_frame, mask, frame_id, compute_metrics
    )


def refine_mask(
    depth_frame: np.ndarray,
    mask: np.ndarray,
    frame_id: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to refine mask using global pipeline.
    
    Args:
        depth_frame: Depth frame (uint16, mm)
        mask: SAM segmentation mask (uint8)
        frame_id: Frame identifier
    
    Returns:
        (refined_mask, refinement_stats)
    """
    pipeline = get_preprocessing_pipeline()
    return pipeline.refine_mask(depth_frame, mask, frame_id)


def export_depth_frame(
    depth_frame: np.ndarray,
    output_path,
    frame_id: int = 0
) -> Dict[str, Any]:
    """
    Convenience function to export depth frame using global pipeline.
    
    Args:
        depth_frame: Depth frame (uint16, mm)
        output_path: Output file path
        frame_id: Frame identifier
    
    Returns:
        Export metrics dictionary
    """
    pipeline = get_preprocessing_pipeline()
    return pipeline.export_depth_frame(depth_frame, output_path, frame_id)
