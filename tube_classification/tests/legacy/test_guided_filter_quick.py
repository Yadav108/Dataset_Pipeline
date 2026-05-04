#!/usr/bin/env python
"""Quick test of guided filter custom implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("TESTING GUIDED FILTER CUSTOM IMPLEMENTATION")
print("="*70)

# Test 1: Import guided_filter module
print("\n1. Testing guided_denoise import...")
try:
    from src.acquisition.guided_filter import guided_denoise
    print("   ✓ Successfully imported guided_denoise")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test with synthetic data
print("\n2. Creating synthetic test data...")
try:
    import numpy as np
    
    depth_test = np.ones((480, 848), dtype=np.uint16) * 1000
    depth_test[100:400, 200:700] = 800  # Box
    depth_test[:50, :] = 0  # Invalid top
    
    rgb_test = np.ones((480, 848, 3), dtype=np.uint8) * 128
    rgb_test[100:400, 200:700] = 200  # Bright box
    
    print(f"   ✓ Depth: {depth_test.shape}, dtype {depth_test.dtype}")
    print(f"   ✓ RGB: {rgb_test.shape}, dtype {rgb_test.dtype}")
    print(f"   ✓ Invalid pixels: {np.count_nonzero(depth_test == 0)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Run guided_denoise
print("\n3. Running guided_denoise...")
try:
    filtered, stats = guided_denoise(
        depth_frame=depth_test,
        rgb_frame=rgb_test,
        radius=8,
        eps=1e-3,
        logger_instance=None
    )
    print(f"   ✓ Execution successful")
    print(f"   ✓ Output shape: {filtered.shape}, dtype: {filtered.dtype}")
    print(f"   ✓ Processing time: {stats['processing_time_ms']:.1f}ms")
    print(f"   ✓ Noise reduction: {stats['noise_reduction_pct']:.1f}%")
    print(f"   ✓ Invalid pixels preserved: {stats['invalid_pixels_preserved']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Validate output
print("\n4. Validating output...")
try:
    assert filtered.dtype == np.uint16, f"Wrong dtype: {filtered.dtype}"
    assert filtered.shape == depth_test.shape, f"Wrong shape: {filtered.shape}"
    # Check invalid pixels were restored to 0
    invalid_mask = depth_test == 0
    if np.any(invalid_mask):
        assert np.all(filtered[invalid_mask] == 0), "Invalid pixels not preserved"
    print("   ✓ All validation checks passed")
except AssertionError as e:
    print(f"   ✗ Validation failed: {e}")
    sys.exit(1)

# Test 5: Import config
print("\n5. Testing config integration...")
try:
    from config.parser import load_config
    cfg = load_config()
    gf_cfg = cfg.preprocessing.guided_filter
    print(f"   ✓ Config loaded")
    print(f"   ✓ guided_filter.enabled: {gf_cfg.enabled}")
    print(f"   ✓ guided_filter.radius: {gf_cfg.radius}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test pipeline integration
print("\n6. Testing pipeline integration...")
try:
    from src.acquisition.pipeline_integration import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    print(f"   ✓ Pipeline initialized")
    print(f"   ✓ guided_filter_enabled: {pipeline.guided_filter_enabled}")
    
    # Test process_depth_frame
    result = pipeline.process_depth_frame(
        depth_frame=depth_test,
        rgb_frame=rgb_test,
        frame_id="test"
    )
    
    processed_depth, quality_metrics, proc_stats = result
    print(f"   ✓ process_depth_frame executed")
    print(f"   ✓ Processing steps: {proc_stats.get('processing_steps', [])}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL TESTS PASSED - GUIDED FILTER INTEGRATION WORKING")
print("="*70 + "\n")
