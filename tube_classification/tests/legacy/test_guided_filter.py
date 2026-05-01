#!/usr/bin/env python
"""
Quick test script for guided filter module.
Run: python test_guided_filter.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.acquisition.guided_filter import guided_denoise
import numpy as np

print("=" * 70)
print("Guided Filter Module Test")
print("=" * 70)

# Create synthetic test data
print("\n✓ Creating synthetic depth and RGB frames...")
H, W = 480, 640

# Synthetic depth: box shape (simulating tube)
depth_test = np.ones((H, W), dtype=np.uint16) * 500  # Base depth
depth_test[100:400, 200:440] = 400  # Foreground (closer)

# Add noise
noise = np.random.normal(0, 20, (H, W))
depth_test = np.clip(depth_test.astype(np.float32) + noise, 0, 65535).astype(np.uint16)

# Mark some pixels as invalid
depth_test[0:50, :] = 0  # Top border
depth_test[:, 0:50] = 0  # Left border

print(f"  - Depth shape: {depth_test.shape}, dtype: {depth_test.dtype}")
print(f"  - Depth range: [{np.min(depth_test[depth_test > 0])}, {np.max(depth_test)}] mm")
print(f"  - Invalid pixels: {np.count_nonzero(depth_test == 0)}")

# Synthetic RGB: simple gradient
rgb_test = np.zeros((H, W, 3), dtype=np.uint8)
rgb_test[:, :, 2] = np.uint8(np.linspace(0, 255, W))  # Red channel gradient
rgb_test[100:400, 200:440] = 255  # Bright box area

print(f"  - RGB shape: {rgb_test.shape}, dtype: {rgb_test.dtype}")
print(f"  - RGB range: [0, 255]")

# Test guided filter
print("\n✓ Calling guided_denoise()...")
try:
    filtered_depth, stats = guided_denoise(
        depth_frame=depth_test,
        rgb_frame=rgb_test,
        radius=8,
        eps=1e-3,
        rgb_normalize=True,
        preserve_invalid=True,
        max_processing_time_ms=500.0,
        logger_instance=None
    )
    
    print("\n✓ Results:")
    print(f"  - Filtered shape: {filtered_depth.shape}, dtype: {filtered_depth.dtype}")
    print(f"  - Filtered range: [{np.min(filtered_depth[filtered_depth > 0])}, {np.max(filtered_depth)}] mm")
    
    print("\n✓ Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  - {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.4f}")
                else:
                    print(f"      {k}: {v}")
        else:
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")
    
    # Verify output
    print("\n✓ Validation:")
    assert filtered_depth.dtype == np.uint16, "Wrong output dtype"
    assert filtered_depth.shape == depth_test.shape, "Shape mismatch"
    assert np.all(filtered_depth[depth_test == 0] == 0), "Invalid pixels not preserved"
    print("  - All checks passed ✓")
    
    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
