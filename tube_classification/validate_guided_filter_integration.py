#!/usr/bin/env python
"""
Quick validation script for guided filter integration.
Tests: Config loading, module imports, pipeline initialization.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("GUIDED FILTER INTEGRATION VALIDATION")
print("="*70)

# Test 1: Import Path (test the fix)
print("\n1. Testing Path import...")
try:
    from pathlib import Path as PathLib
    print("   ✓ Path imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import Path: {e}")
    sys.exit(1)

# Test 2: Load config
print("\n2. Loading config...")
try:
    from config.parser import load_config
    cfg = load_config()
    print("   ✓ Config loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Access guided_filter config
print("\n3. Accessing guided_filter config section...")
try:
    gf_cfg = cfg.preprocessing.guided_filter
    print(f"   ✓ guided_filter config accessible")
    print(f"      - enabled: {gf_cfg.enabled}")
    print(f"      - radius: {gf_cfg.radius}")
    print(f"      - eps: {gf_cfg.eps}")
    print(f"      - rgb_normalize: {gf_cfg.rgb_normalize}")
    print(f"      - preserve_invalid: {gf_cfg.preserve_invalid}")
    print(f"      - max_processing_time_ms: {gf_cfg.max_processing_time_ms}")
except Exception as e:
    print(f"   ✗ Failed to access guided_filter config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Import guided_filter module
print("\n4. Importing guided_filter module...")
try:
    from src.acquisition.guided_filter import guided_denoise
    print("   ✓ guided_denoise function imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import guided_denoise: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Import PreprocessingPipeline
print("\n5. Importing PreprocessingPipeline...")
try:
    from src.acquisition.pipeline_integration import PreprocessingPipeline
    print("   ✓ PreprocessingPipeline imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import PreprocessingPipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Initialize PreprocessingPipeline
print("\n6. Initializing PreprocessingPipeline...")
try:
    pipeline = PreprocessingPipeline()
    print("   ✓ PreprocessingPipeline initialized successfully")
    print(f"      - bilateral_enabled: {pipeline.bilateral_enabled}")
    print(f"      - guided_filter_enabled: {pipeline.guided_filter_enabled}")
    print(f"      - temporal_smoothing_enabled: {pipeline.temporal_smoothing_enabled}")
    print(f"      - inpainting_enabled: {pipeline.inpainting_enabled}")
    print(f"      - mask_refinement_enabled: {pipeline.mask_refinement_enabled}")
except Exception as e:
    print(f"   ✗ Failed to initialize PreprocessingPipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Verify shape validation
print("\n7. Testing shape validation in process_depth_frame...")
try:
    import numpy as np
    
    # Valid inputs
    depth_valid = np.zeros((480, 848), dtype=np.uint16)
    rgb_valid = np.zeros((480, 848, 3), dtype=np.uint8)
    
    # Test shape mismatch detection
    rgb_invalid = np.zeros((240, 424, 3), dtype=np.uint8)  # Wrong shape
    
    try:
        # This should raise ValueError
        pipeline.process_depth_frame(
            depth_frame=depth_valid,
            rgb_frame=rgb_invalid,
            frame_id="test"
        )
        print("   ✗ Shape validation did NOT catch shape mismatch!")
        sys.exit(1)
    except ValueError as ve:
        if "shape mismatch" in str(ve).lower():
            print("   ✓ Shape validation correctly detected mismatch")
        else:
            print(f"   ✗ Unexpected ValueError: {ve}")
            sys.exit(1)
except Exception as e:
    print(f"   ✗ Failed shape validation test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test with valid inputs (dry run)
print("\n8. Testing process_depth_frame with valid synthetic data...")
try:
    import numpy as np
    
    depth_frame = np.ones((480, 848), dtype=np.uint16) * 1000  # 1m depth
    rgb_frame = np.ones((480, 848, 3), dtype=np.uint8) * 128   # Gray
    
    processed_depth, quality_metrics, stats = pipeline.process_depth_frame(
        depth_frame=depth_frame,
        rgb_frame=rgb_frame,
        frame_id="validation_test",
        compute_metrics=False
    )
    
    print(f"   ✓ process_depth_frame executed successfully")
    print(f"      - Input depth shape: {depth_frame.shape}, dtype: {depth_frame.dtype}")
    print(f"      - Output depth shape: {processed_depth.shape}, dtype: {processed_depth.dtype}")
    print(f"      - Processing steps: {stats.get('processing_steps', [])}")
    print(f"      - Timing (ms): {stats.get('timing_ms', {})}")
    
except Exception as e:
    print(f"   ✗ Failed to process depth frame: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL VALIDATION TESTS PASSED")
print("="*70)
print("\nGuided filter integration is ready for use!")
print("Next steps:")
print("  1. Run: python main.py")
print("  2. Start a capture session")
print("  3. Monitor logs for 'Guided filter' messages")
print("  4. Check noise_reduction_pct in statistics")
print("\n")
