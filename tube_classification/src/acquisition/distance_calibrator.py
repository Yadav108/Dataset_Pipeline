"""Auto-calibrate camera-to-tube distance at session start.

Measures median depth, adjusts depth range thresholds, and stores calibration
metadata for reproducibility.
"""

import numpy as np
import time
from pathlib import Path
from loguru import logger
from config.parser import get_config


class DistanceCalibrator:
    """Automatically calibrate camera-to-tube distance at session start.
    
    Measures median depth over multiple frames, adjusts depth range thresholds,
    and stores calibration info for this session.
    """
    
    def __init__(self, depth_scale: float = 0.001):
        """Initialize calibrator.
        
        Args:
            depth_scale: Depth scale factor from RealSense camera (mm to m)
        """
        self.cfg = get_config()
        self.depth_scale = depth_scale
        self.measured_distance_m = None
        self.original_depth_min = None
        self.original_depth_max = None
        self.calibrated_depth_min = None
        self.calibrated_depth_max = None
    
    def measure_distance(
        self,
        streamer,
        duration_seconds: float = 2.0,
        percentile: float = 50.0
    ) -> dict:
        """Measure camera-to-tube distance using median depth.
        
        Collects depth frames for specified duration, computes percentile
        (default: median), and stores calibration info.
        
        Args:
            streamer: RealSenseStreamer instance (must be started)
            duration_seconds: How long to collect frames (default 2s)
            percentile: Which percentile to use (50=median, 75=upper quartile)
            
        Returns:
            Dict with calibration results:
            {
                'measured_distance_m': float,
                'valid_pixel_ratio': float,
                'sample_count': int,
                'duration_seconds': float
            }
        """
        logger.info("DISTANCE CALIBRATION: Starting camera-to-tube distance measurement...")
        logger.info(f"Collecting frames for {duration_seconds}s (collecting valid depth pixels)...")
        
        collected_depths = []
        start_time = time.time()
        frame_count = 0
        
        # Collect depth frames
        while time.time() - start_time < duration_seconds:
            frames = streamer.get_aligned_frames()
            if frames is None:
                continue
            
            rgb_frame, depth_frame = frames
            frame_count += 1
            
            # Extract valid depth pixels (non-zero)
            valid_depths = depth_frame[depth_frame > 0]
            
            if len(valid_depths) > 0:
                # Convert from uint16 (mm) to float (m)
                depths_m = valid_depths.astype(np.float32) * self.depth_scale
                collected_depths.extend(depths_m.tolist())
        
        elapsed = time.time() - start_time
        
        if not collected_depths:
            logger.error("DISTANCE CALIBRATION: No valid depth pixels collected!")
            return None
        
        # Compute percentile
        collected_depths_array = np.array(collected_depths)
        measured_distance = np.percentile(collected_depths_array, percentile)
        valid_pixel_ratio = len(collected_depths) / (frame_count * 480 * 640) if frame_count > 0 else 0
        
        # Store measurement
        self.measured_distance_m = measured_distance
        
        logger.info(
            f"DISTANCE CALIBRATION: Measurement complete"
            f" | distance={measured_distance:.3f}m"
            f" | frames={frame_count}"
            f" | valid_pixels={len(collected_depths):,}"
            f" | valid_ratio={valid_pixel_ratio:.2%}"
        )
        
        return {
            'measured_distance_m': float(measured_distance),
            'valid_pixel_ratio': float(valid_pixel_ratio),
            'sample_count': len(collected_depths),
            'duration_seconds': elapsed,
            'frame_count': frame_count,
        }
    
    def auto_adjust_depth_range(self, measured_distance_m: float) -> dict:
        """Automatically adjust depth range thresholds based on measured distance.
        
        Computes optimal depth_min and depth_max around the measured distance
        to maximize valid depth data while excluding background/foreground noise.
        
        Args:
            measured_distance_m: Camera-to-tube distance in meters
            
        Returns:
            Dict with calibration results:
            {
                'calibrated_depth_min_m': float,
                'calibrated_depth_max_m': float,
                'original_depth_min_m': float,
                'original_depth_max_m': float,
                'margin_m': float,
            }
        """
        # Store originals
        self.original_depth_min = self.cfg.camera.depth_min_m
        self.original_depth_max = self.cfg.camera.depth_max_m
        
        # Compute calibrated range centered on measured distance
        # Use wider margin around the measured distance for better coverage
        margin_m = 0.25  # ±250mm around detected distance (increased from 150mm)
        
        calibrated_min = max(0.08, measured_distance_m - margin_m)  # Never go below 8cm
        calibrated_max = min(2.5, measured_distance_m + margin_m)   # Never exceed 2.5m
        
        self.calibrated_depth_min = calibrated_min
        self.calibrated_depth_max = calibrated_max
        
        logger.info(
            f"DISTANCE CALIBRATION: Auto-adjusted depth range"
            f" | original=[{self.original_depth_min:.2f}m, {self.original_depth_max:.2f}m]"
            f" | calibrated=[{calibrated_min:.2f}m, {calibrated_max:.2f}m]"
            f" | margin=±{margin_m:.2f}m"
        )
        
        return {
            'calibrated_depth_min_m': calibrated_min,
            'calibrated_depth_max_m': calibrated_max,
            'original_depth_min_m': self.original_depth_min,
            'original_depth_max_m': self.original_depth_max,
            'margin_m': margin_m,
        }
    
    def run_calibration(self, streamer) -> dict:
        """Run full auto-calibration sequence.
        
        Measures distance, auto-adjusts depth range, and returns complete
        calibration metadata.
        
        Args:
            streamer: RealSenseStreamer instance
            
        Returns:
            Complete calibration results dict
        """
        print("\n" + "="*70)
        print("AUTO-CALIBRATION: Camera-to-Tube Distance")
        print("="*70)
        print("Instructions: Place tube in capture zone and keep it steady.\n")
        
        # Step 1: Measure distance
        measurement = self.measure_distance(streamer, duration_seconds=2.0)
        if measurement is None:
            logger.error("DISTANCE CALIBRATION: Measurement failed!")
            return None
        
        # Step 2: Auto-adjust depth range
        adjustment = self.auto_adjust_depth_range(measurement['measured_distance_m'])
        
        # Step 3: Combine results
        calibration_result = {
            **measurement,
            **adjustment,
        }
        
        print("\n" + "-"*70)
        print(f"✓ Measured distance: {measurement['measured_distance_m']:.3f}m")
        print(f"✓ Adjusted depth range: [{adjustment['calibrated_depth_min_m']:.2f}m, {adjustment['calibrated_depth_max_m']:.2f}m]")
        print(f"✓ Original range: [{adjustment['original_depth_min_m']:.2f}m, {adjustment['original_depth_max_m']:.2f}m]")
        print(f"✓ Valid samples: {measurement['sample_count']:,} pixels from {measurement['frame_count']} frames")
        print("="*70 + "\n")
        
        return calibration_result
    
    def get_calibration_metadata(self) -> dict:
        """Get metadata dict for storing with session.
        
        Returns:
            Dict with calibration info for session metadata
        """
        if self.measured_distance_m is None:
            return {}
        
        return {
            'distance_calibration': {
                'measured_distance_m': self.measured_distance_m,
                'calibrated_depth_min_m': self.calibrated_depth_min,
                'calibrated_depth_max_m': self.calibrated_depth_max,
                'original_depth_min_m': self.original_depth_min,
                'original_depth_max_m': self.original_depth_max,
            }
        }
