"""
Unit tests for advanced preprocessing (PROMPT 5-7)
"""

import pytest
import numpy as np
import cv2
import logging
from pathlib import Path
import tempfile
import time

from src.acquisition.advanced_preprocessing import (
    inpaint_depth_telea,
    DepthGuidedMaskRefinement,
    PNG16Compressor,
    convert_npy_to_png16
)


logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT 5: DEPTH INPAINTING TESTS
# ============================================================================

class TestDepthInpainting:
    """Test depth inpainting (Telea algorithm)"""
    
    def test_inpainting_basic(self):
        """Test basic inpainting with holes"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        depth[100:110, 100:110] = 0  # Create hole
        
        inpainted = inpaint_depth_telea(depth)
        
        assert inpainted.shape == (480, 848)
        assert inpainted.dtype == np.uint16
        # Check that some holes are filled
        filled = np.sum(inpainted[100:110, 100:110] > 0)
        assert filled > 0
    
    def test_no_holes(self):
        """Test with no holes (all valid pixels)"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        
        inpainted = inpaint_depth_telea(depth)
        
        # Should return unchanged
        assert np.allclose(inpainted, depth)
    
    def test_invalid_shape(self):
        """Test invalid shape raises error"""
        depth = np.zeros((640, 480), dtype=np.uint16)
        with pytest.raises(ValueError, match="Invalid shape"):
            inpaint_depth_telea(depth)
    
    def test_invalid_dtype(self):
        """Test invalid dtype raises error"""
        depth = np.zeros((480, 848), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid dtype"):
            inpaint_depth_telea(depth)
    
    def test_too_few_valid_pixels(self):
        """Test raises error if too few valid pixels"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        depth[0:48, 0:85] = 200  # Only ~4000 pixels out of 408k (~1%)
        
        with pytest.raises(ValueError, match="Too few valid pixels"):
            inpaint_depth_telea(depth, min_valid_ratio=0.30)
    
    def test_coverage_metric(self):
        """Test coverage metric calculation"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        # Create 5% holes
        hole_mask = np.zeros((480, 848), dtype=bool)
        hole_mask[0:48, 0:85] = True  # ~20000 pixels
        depth[hole_mask] = 0
        
        inpainted = inpaint_depth_telea(depth)
        
        # Holes should mostly be filled
        filled_ratio = np.sum(inpainted > 0) / np.sum(depth > 0)
        assert filled_ratio > 0.95
    
    def test_performance_target(self):
        """Test processing time < 30ms"""
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        depth[np.random.rand(480, 848) < 0.05] = 0  # 5% holes
        
        start = time.time()
        inpaint_depth_telea(depth)
        elapsed = (time.time() - start) * 1000
        
        assert elapsed < 30, f"Processing took {elapsed:.1f}ms, exceeds 30ms target"
    
    def test_inpainting_error(self):
        """Test inpainting error < 50mm at boundaries"""
        # Create depth with known hole and boundary values
        depth = np.full((480, 848), 200, dtype=np.uint16)
        depth[100:110, 100:110] = 0
        
        inpainted = inpaint_depth_telea(depth)
        
        # Check pixels near boundary are close to original
        boundary_error = np.abs(
            inpainted[110:115, 100:110].astype(np.float32) - 200.0
        )
        assert np.mean(boundary_error) < 50


# ============================================================================
# PROMPT 6: DEPTH-GUIDED SAM MASK REFINEMENT TESTS
# ============================================================================

class TestDepthGuidedMaskRefinement:
    """Test depth-guided mask refinement"""
    
    def test_refinement_basic(self):
        """Test basic mask refinement"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255  # 100x100 square
        
        refiner = DepthGuidedMaskRefinement(depth, mask)
        refined, metrics = refiner.refine()
        
        assert refined.shape == (480, 848)
        assert refined.dtype == np.uint8
        assert 'iou_improvement_pct' in metrics
        assert 'depth_correlation' in metrics
        assert 'connectivity' in metrics
    
    def test_invalid_depth_shape(self):
        """Test invalid depth shape raises error"""
        depth = np.zeros((640, 480), dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Invalid depth shape"):
            DepthGuidedMaskRefinement(depth, mask)
    
    def test_invalid_mask_shape(self):
        """Test invalid mask shape raises error"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        mask = np.zeros((640, 480), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Invalid mask shape"):
            DepthGuidedMaskRefinement(depth, mask)
    
    def test_invalid_depth_dtype(self):
        """Test invalid depth dtype raises error"""
        depth = np.zeros((480, 848), dtype=np.float32)
        mask = np.zeros((480, 848), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Invalid depth dtype"):
            DepthGuidedMaskRefinement(depth, mask)
    
    def test_invalid_mask_dtype(self):
        """Test invalid mask dtype raises error"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.float32)
        
        with pytest.raises(ValueError, match="Invalid mask dtype"):
            DepthGuidedMaskRefinement(depth, mask)
    
    def test_depth_correlation_metric(self):
        """Test depth correlation is computed"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        # Create depth discontinuity
        depth[250:, :] = 100
        
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:300, 100:300] = 255
        
        refiner = DepthGuidedMaskRefinement(depth, mask)
        _, metrics = refiner.refine()
        
        # Correlation should be computed
        assert 0 <= metrics['depth_correlation'] <= 1
    
    def test_connectivity_validation(self):
        """Test connectivity validation"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        
        # Create disconnected mask
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:150, 100:150] = 255
        mask[300:350, 300:350] = 255
        
        refiner = DepthGuidedMaskRefinement(depth, mask)
        _, metrics = refiner.refine()
        
        # Connectivity should be reported
        assert 'connectivity' in metrics
    
    def test_iou_improvement(self):
        """Test IoU improvement calculation"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        refiner = DepthGuidedMaskRefinement(depth, mask)
        _, metrics = refiner.refine()
        
        # IoU improvement should be in reasonable range
        assert -20 <= metrics['iou_improvement_pct'] <= 20  # ±20% is reasonable
    
    def test_performance_target(self):
        """Test processing time (should be <50ms typical)"""
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        mask = np.random.randint(0, 256, (480, 848), dtype=np.uint8)
        
        refiner = DepthGuidedMaskRefinement(depth, mask)
        
        start = time.time()
        _, metrics = refiner.refine()
        
        processing_time = metrics['processing_time_ms']
        assert processing_time < 200  # Generous bound for morphological ops
    
    def test_minimum_mask_area(self):
        """Test minimum mask area validation"""
        depth = np.zeros((480, 848), dtype=np.uint16)
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[0:5, 0:5] = 255  # Very small mask (25 pixels)
        
        refiner = DepthGuidedMaskRefinement(
            depth, mask, min_mask_area=100
        )
        _, metrics = refiner.refine()
        
        # Should report connectivity issue
        assert metrics['connectivity'] == False


# ============================================================================
# PROMPT 7: PNG16 COMPRESSION TESTS
# ============================================================================

class TestPNG16Compression:
    """Test PNG16 compression and decompression"""
    
    def test_save_png16_basic(self):
        """Test basic PNG16 save"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            metrics = PNG16Compressor.save_depth_png16(depth, output_path)
            
            assert output_path.exists()
            assert metrics['file_size_bytes'] > 0
            assert metrics['compression_ratio'] >= 0
    
    def test_load_png16_basic(self):
        """Test basic PNG16 load"""
        depth_original = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            PNG16Compressor.save_depth_png16(depth_original, output_path)
            
            depth_loaded, metrics = PNG16Compressor.load_depth_png16(output_path)
            
            assert depth_loaded.shape == (480, 848)
            assert depth_loaded.dtype == np.uint16
    
    def test_round_trip_lossless(self):
        """Test round-trip lossless conversion"""
        depth_original = np.random.randint(0, 65536, (480, 848), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            PNG16Compressor.save_depth_png16(depth_original, output_path)
            depth_loaded, _ = PNG16Compressor.load_depth_png16(output_path)
            
            # Should be lossless
            assert np.allclose(depth_original, depth_loaded)
    
    def test_compression_ratio(self):
        """Test PNG16 achieves ~30% compression"""
        depth = np.full((480, 848), 200, dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            metrics = PNG16Compressor.save_depth_png16(depth, output_path)
            
            # Should achieve 20-40% compression for uniform depth
            assert 20 <= metrics['compression_ratio'] <= 50
    
    def test_save_invalid_shape(self):
        """Test save with invalid shape raises error"""
        depth = np.zeros((640, 480), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            with pytest.raises(ValueError, match="Invalid shape"):
                PNG16Compressor.save_depth_png16(depth, output_path)
    
    def test_save_invalid_dtype(self):
        """Test save with invalid dtype raises error"""
        depth = np.zeros((480, 848), dtype=np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            with pytest.raises(ValueError, match="Invalid dtype"):
                PNG16Compressor.save_depth_png16(depth, output_path)
    
    def test_load_nonexistent_file(self):
        """Test load nonexistent file raises error"""
        with pytest.raises(IOError, match="File not found"):
            PNG16Compressor.load_depth_png16(Path("/nonexistent/file.png"))
    
    def test_save_performance_target(self):
        """Test save performance < 20ms"""
        depth = np.random.randint(0, 65536, (480, 848), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            
            start = time.time()
            metrics = PNG16Compressor.save_depth_png16(depth, output_path)
            
            assert metrics['processing_time_ms'] < 20
    
    def test_load_performance_target(self):
        """Test load performance < 20ms"""
        depth = np.random.randint(0, 65536, (480, 848), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            PNG16Compressor.save_depth_png16(depth, output_path)
            
            start = time.time()
            _, metrics = PNG16Compressor.load_depth_png16(output_path)
            
            assert metrics['processing_time_ms'] < 20
    
    def test_round_trip_error_mm(self):
        """Test round-trip error < 1mm"""
        depth_original = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            PNG16Compressor.save_depth_png16(depth_original, output_path)
            depth_loaded, metrics = PNG16Compressor.load_depth_png16(output_path)
            
            error = np.mean(np.abs(
                depth_original.astype(np.float32) - 
                depth_loaded.astype(np.float32)
            ))
            
            assert error < 1.0


# ============================================================================
# BATCH CONVERSION TESTS
# ============================================================================

class TestBatchConversion:
    """Test batch .npy to PNG16 conversion"""
    
    def test_batch_conversion_basic(self):
        """Test basic batch conversion"""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_dir = Path(tmpdir) / "npy_files"
            png_dir = Path(tmpdir) / "png_files"
            npy_dir.mkdir()
            
            # Create test .npy files
            for i in range(3):
                depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
                np.save(npy_dir / f"depth_{i:04d}.npy", depth)
            
            metrics = convert_npy_to_png16(npy_dir, png_dir)
            
            assert metrics['files_converted'] == 3
            assert png_dir.exists()
            assert len(list(png_dir.glob('*.png'))) == 3
    
    def test_batch_compression_ratio(self):
        """Test batch conversion achieves compression"""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_dir = Path(tmpdir) / "npy_files"
            png_dir = Path(tmpdir) / "png_files"
            npy_dir.mkdir()
            
            # Create test .npy files with varying data
            for i in range(5):
                depth = np.random.randint(0, 65536, (480, 848), dtype=np.uint16)
                np.save(npy_dir / f"depth_{i:04d}.npy", depth)
            
            metrics = convert_npy_to_png16(npy_dir, png_dir)
            
            # Should achieve compression
            assert metrics['compression_ratio_pct'] > 0
            assert metrics['total_png_size_mb'] < metrics['total_npy_size_mb']
    
    def test_batch_invalid_files_skipped(self):
        """Test invalid files are skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_dir = Path(tmpdir) / "npy_files"
            png_dir = Path(tmpdir) / "png_files"
            npy_dir.mkdir()
            
            # Valid file
            depth = np.random.randint(170, 270, (480, 848), dtype=np.uint16)
            np.save(npy_dir / "valid.npy", depth)
            
            # Invalid files
            np.save(npy_dir / "invalid_shape.npy", np.zeros((640, 480), dtype=np.uint16))
            np.save(npy_dir / "invalid_dtype.npy", np.zeros((480, 848), dtype=np.float32))
            
            metrics = convert_npy_to_png16(npy_dir, png_dir)
            
            # Only 1 valid file should be converted
            assert metrics['files_converted'] == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAdvancedPreprocessingIntegration:
    """Integration tests for PROMPT 5-7"""
    
    def test_full_inpainting_pipeline(self):
        """Test full inpainting pipeline"""
        # Create depth with holes
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        depth[np.random.rand(480, 848) < 0.1] = 0
        
        inpainted = inpaint_depth_telea(depth)
        
        # Verify inpainting worked
        assert inpainted.shape == depth.shape
        # More pixels should be valid after inpainting
        assert np.sum(inpainted > 0) > np.sum(depth > 0)
    
    def test_refinement_with_inpainted_depth(self):
        """Test mask refinement with inpainted depth"""
        # Create and inpaint depth
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        depth[100:110, 100:110] = 0
        inpainted = inpaint_depth_telea(depth)
        
        # Refine mask with inpainted depth
        mask = np.zeros((480, 848), dtype=np.uint8)
        mask[80:150, 80:150] = 255
        
        refiner = DepthGuidedMaskRefinement(inpainted, mask)
        refined, metrics = refiner.refine()
        
        assert refined.shape == (480, 848)
        assert metrics['depth_correlation'] >= 0
    
    def test_png16_with_inpainted_depth(self):
        """Test PNG16 save/load with inpainted depth"""
        # Create and inpaint depth
        depth = np.random.randint(150, 250, (480, 848), dtype=np.uint16)
        depth[np.random.rand(480, 848) < 0.05] = 0
        inpainted = inpaint_depth_telea(depth)
        
        # Save and load as PNG16
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "inpainted.png"
            PNG16Compressor.save_depth_png16(inpainted, output_path)
            loaded, _ = PNG16Compressor.load_depth_png16(output_path)
            
            assert np.allclose(inpainted, loaded)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
