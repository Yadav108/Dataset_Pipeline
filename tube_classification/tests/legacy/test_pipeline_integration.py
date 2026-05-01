"""
Unit tests for PROMPT 8: Pipeline Integration
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import time

from src.acquisition.pipeline_integration import (
    PreprocessingPipeline,
    ProcessingMode,
    get_preprocessing_pipeline,
    process_frame,
    refine_mask,
    export_depth_frame
)


class TestPreprocessingPipeline:
    """Test preprocessing pipeline integration."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = PreprocessingPipeline()
        
        assert pipeline is not None
        assert pipeline.mode in [ProcessingMode.BASIC, ProcessingMode.ADVANCED, ProcessingMode.NONE]
        assert pipeline.total_frames_processed == 0
    
    def test_process_depth_frame_basic(self):
        """Test basic frame processing."""
        pipeline = PreprocessingPipeline()
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        
        processed, metrics, stats = pipeline.process_depth_frame(depth)
        
        assert processed.shape == (480, 848)
        assert processed.dtype == np.uint16
        assert 'total_time_ms' in stats
        assert stats['total_time_ms'] > 0
    
    def test_process_depth_frame_with_rgb(self):
        """Test frame processing with RGB for quality metrics."""
        pipeline = PreprocessingPipeline()
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        processed, metrics, stats = pipeline.process_depth_frame(depth, rgb_frame=rgb)
        
        if pipeline.quality_metrics_enabled:
            assert metrics is not None
            assert 0 <= metrics.quality_score <= 10
    
    def test_process_depth_frame_with_mask(self):
        """Test frame processing with mask."""
        pipeline = PreprocessingPipeline()
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        processed, metrics, stats = pipeline.process_depth_frame(
            depth, rgb_frame=rgb, mask=mask
        )
        
        if pipeline.quality_metrics_enabled:
            assert metrics is not None
    
    def test_process_multiple_frames(self):
        """Test processing multiple frames sequentially."""
        pipeline = PreprocessingPipeline()
        
        for i in range(10):
            depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
            processed, _, _ = pipeline.process_depth_frame(depth, frame_id=i)
            
            assert processed.shape == (480, 848)
        
        assert pipeline.total_frames_processed == 10
    
    def test_temporal_smoothing_effect(self):
        """Test temporal smoothing reduces jitter."""
        pipeline = PreprocessingPipeline()
        
        # Frame 1: baseline
        depth1 = np.full((480, 848), 200, dtype=np.uint16)
        processed1, _, _ = pipeline.process_depth_frame(depth1)
        
        # Frame 2: with jitter
        depth2 = np.full((480, 848), 210, dtype=np.uint16)
        depth2[100:110, 100:110] = 300  # Spike
        processed2, _, stats2 = pipeline.process_depth_frame(depth2)
        
        # Check stats if temporal smoothing is enabled
        if pipeline.temporal_smoothing_enabled:
            assert 'jitter_reduction_pct' in stats2
    
    def test_invalid_depth_shape(self):
        """Test invalid depth shape raises error."""
        pipeline = PreprocessingPipeline()
        depth = np.zeros((640, 480), dtype=np.uint16)
        
        with pytest.raises(ValueError, match="Invalid depth shape"):
            pipeline.process_depth_frame(depth)
    
    def test_invalid_depth_dtype(self):
        """Test invalid depth dtype raises error."""
        pipeline = PreprocessingPipeline()
        depth = np.zeros((480, 848), dtype=np.float32)
        
        with pytest.raises(ValueError, match="Invalid depth dtype"):
            pipeline.process_depth_frame(depth)
    
    def test_refine_mask_basic(self):
        """Test mask refinement."""
        pipeline = PreprocessingPipeline()
        depth = np.full((480, 848), 200, dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        refined, stats = pipeline.refine_mask(depth, mask)
        
        assert refined.shape == (480, 848)
        assert refined.dtype == np.uint8
        if pipeline.mask_refinement_enabled:
            assert stats['refined'] == True
    
    def test_refine_mask_disabled(self):
        """Test mask refinement returns original when disabled."""
        pipeline = PreprocessingPipeline()
        pipeline.mask_refinement_enabled = False
        
        depth = np.full((480, 848), 200, dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        refined, stats = pipeline.refine_mask(depth, mask)
        
        assert np.array_equal(refined, mask)
        assert stats['refined'] == False
    
    def test_export_depth_frame(self):
        """Test PNG16 export."""
        pipeline = PreprocessingPipeline()
        if not pipeline.png16_export_enabled:
            pytest.skip("PNG16 export disabled")
        
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frame.png"
            metrics = pipeline.export_depth_frame(depth, output_path)
            
            assert output_path.exists()
            assert metrics['exported'] == True
            assert metrics['file_size_bytes'] > 0
    
    def test_pipeline_reset(self):
        """Test pipeline reset."""
        pipeline = PreprocessingPipeline()
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        
        # Process a frame
        pipeline.process_depth_frame(depth)
        assert pipeline.frame_count == 1
        
        # Reset
        pipeline.reset()
        assert pipeline.frame_count == 0
    
    def test_get_statistics(self):
        """Test statistics gathering."""
        pipeline = PreprocessingPipeline()
        
        for i in range(5):
            depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
            pipeline.process_depth_frame(depth)
        
        stats = pipeline.get_statistics()
        
        assert stats['total_frames'] == 5
        assert stats['total_time_ms'] > 0
        assert stats['avg_time_per_frame_ms'] > 0
        assert 'enabled_steps' in stats


class TestGlobalPipelineInterface:
    """Test global pipeline functions."""
    
    def test_get_preprocessing_pipeline_singleton(self):
        """Test singleton pattern."""
        pipeline1 = get_preprocessing_pipeline()
        pipeline2 = get_preprocessing_pipeline()
        
        assert pipeline1 is pipeline2
    
    def test_process_frame_function(self):
        """Test global process_frame function."""
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        processed, metrics, stats = process_frame(depth, rgb_frame=rgb)
        
        assert processed.shape == (480, 848)
        assert processed.dtype == np.uint16
    
    def test_refine_mask_function(self):
        """Test global refine_mask function."""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        refined, stats = refine_mask(depth, mask)
        
        assert refined.shape == (480, 848)
        assert refined.dtype == np.uint8
    
    def test_export_depth_frame_function(self):
        """Test global export_depth_frame function."""
        depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frame.png"
            metrics = export_depth_frame(depth, output_path)
            
            assert 'exported' in metrics


class TestPipelineIntegration:
    """Integration tests for complete preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self):
        """Test full preprocessing pipeline end-to-end."""
        pipeline = PreprocessingPipeline()
        
        # Create test data
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        # Process depth
        processed_depth, metrics, stats = pipeline.process_depth_frame(
            depth, rgb_frame=rgb
        )
        
        # Verify output
        assert processed_depth.shape == (480, 848)
        assert processed_depth.dtype == np.uint16
        assert 'total_time_ms' in stats
        assert stats['total_time_ms'] < 500  # Should be reasonably fast
    
    def test_mask_refinement_integration(self):
        """Test mask refinement with processed depth."""
        pipeline = PreprocessingPipeline()
        
        # Create test data
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        # Process depth
        processed_depth, _, _ = pipeline.process_depth_frame(depth, rgb_frame=rgb)
        
        # Refine mask
        refined_mask, refinement_stats = pipeline.refine_mask(processed_depth, mask)
        
        assert refined_mask.shape == (480, 848)
        assert refined_mask.dtype == np.uint8
    
    def test_export_after_processing(self):
        """Test exporting processed depth."""
        pipeline = PreprocessingPipeline()
        if not pipeline.png16_export_enabled:
            pytest.skip("PNG16 export disabled")
        
        # Create test data
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        # Process depth
        processed_depth, _, _ = pipeline.process_depth_frame(depth, rgb_frame=rgb)
        
        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "processed.png"
            export_metrics = pipeline.export_depth_frame(processed_depth, output_path)
            
            assert output_path.exists()
            assert export_metrics['exported'] == True
    
    def test_continuous_frame_processing(self):
        """Test continuous frame processing (like live capture)."""
        pipeline = PreprocessingPipeline()
        
        frame_times = []
        
        for i in range(20):
            depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
            
            start = time.time()
            processed, _, stats = pipeline.process_depth_frame(depth, frame_id=i)
            frame_times.append(stats['total_time_ms'])
        
        # Verify performance
        avg_time = np.mean(frame_times)
        max_time = np.max(frame_times)
        
        assert avg_time < 300  # Average <300ms per frame
        assert max_time < 500  # Max <500ms per frame
        assert pipeline.total_frames_processed == 20
    
    def test_performance_targets(self):
        """Test all performance targets are met."""
        pipeline = PreprocessingPipeline()
        
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        rgb = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)
        
        # Process 10 frames and measure
        times = []
        for i in range(10):
            start = time.time()
            _, _, stats = pipeline.process_depth_frame(depth, rgb_frame=rgb, frame_id=i)
            times.append(stats['total_time_ms'])
        
        avg_time = np.mean(times)
        
        # Verify target (should be <250ms for all steps combined)
        assert avg_time < 250, f"Average time {avg_time:.1f}ms exceeds target of 250ms"


class TestPipelineConfiguration:
    """Test pipeline configuration loading."""
    
    def test_configuration_loading(self):
        """Test pipeline loads configuration correctly."""
        pipeline = PreprocessingPipeline()
        
        # Verify configuration is loaded
        assert pipeline.cfg is not None
        if hasattr(pipeline.cfg, 'preprocessing'):
            assert hasattr(pipeline.cfg.preprocessing, 'bilateral')
            assert hasattr(pipeline.cfg.preprocessing, 'temporal_smoothing')
    
    def test_mode_selection(self):
        """Test preprocessing mode is selected correctly."""
        pipeline = PreprocessingPipeline()
        
        # Mode should be set based on enabled components
        assert pipeline.mode in [
            ProcessingMode.NONE,
            ProcessingMode.BASIC,
            ProcessingMode.ADVANCED,
            ProcessingMode.EXPORT
        ]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
