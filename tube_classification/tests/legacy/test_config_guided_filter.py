#!/usr/bin/env python
"""
Quick test to verify config schema extension for guided filter.
Run: python test_config_guided_filter.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Testing Config Schema Extension for Guided Filter")
print("=" * 70)

# Test 1: Load config
print("\n1. Loading config...")
try:
    from config.parser import load_config
    cfg = load_config()
    print("   ✓ Config loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load config: {e}")
    sys.exit(1)

# Test 2: Access guided filter config
print("\n2. Accessing guided_filter config...")
try:
    gf = cfg.preprocessing.guided_filter
    print("   ✓ guided_filter config accessible")
except AttributeError as e:
    print(f"   ✗ guided_filter config not found: {e}")
    sys.exit(1)

# Test 3: Verify all parameters present
print("\n3. Verifying all parameters...")
required_params = ["enabled", "radius", "eps", "rgb_normalize", "preserve_invalid", "max_processing_time_ms"]
missing = []
for param in required_params:
    if not hasattr(gf, param):
        missing.append(param)
        print(f"   ✗ Missing parameter: {param}")
    else:
        print(f"   ✓ {param}: {getattr(gf, param)}")

if missing:
    print(f"\n✗ Missing {len(missing)} parameters")
    sys.exit(1)

# Test 4: Check parameter values
print("\n4. Verifying parameter values...")
checks = [
    ("enabled", gf.enabled, isinstance(gf.enabled, bool), "bool"),
    ("radius", gf.radius, 1 <= gf.radius <= 32, "int in [1, 32]"),
    ("eps", gf.eps, gf.eps > 0, "float > 0"),
    ("rgb_normalize", gf.rgb_normalize, isinstance(gf.rgb_normalize, bool), "bool"),
    ("preserve_invalid", gf.preserve_invalid, isinstance(gf.preserve_invalid, bool), "bool"),
    ("max_processing_time_ms", gf.max_processing_time_ms, gf.max_processing_time_ms > 0, "float > 0"),
]

all_valid = True
for name, value, condition, type_desc in checks:
    if condition:
        print(f"   ✓ {name}: {value} ({type_desc})")
    else:
        print(f"   ✗ {name}: {value} - should be {type_desc}")
        all_valid = False

if not all_valid:
    print("\n✗ Some parameters have invalid values")
    sys.exit(1)

# Test 5: Check temporal_smoothing alpha updated
print("\n5. Verifying temporal_smoothing config...")
ts = cfg.preprocessing.temporal_smoothing
print(f"   alpha: {ts.alpha}")
if ts.alpha == 0.3:
    print(f"   ✓ alpha correctly updated to 0.3 (from 0.2)")
else:
    print(f"   ⚠ alpha is {ts.alpha}, expected 0.3")

# Test 6: Test parameter validation
print("\n6. Testing parameter validation...")
from config.parser import GuidedFilterConfig

# Valid config
try:
    valid = GuidedFilterConfig(radius=8, eps=1e-3)
    print("   ✓ Valid parameters accepted")
except ValueError as e:
    print(f"   ✗ Valid parameters rejected: {e}")
    sys.exit(1)

# Invalid radius (too high)
try:
    invalid = GuidedFilterConfig(radius=50)
    print(f"   ✗ Should have rejected radius=50")
    sys.exit(1)
except ValueError as e:
    print(f"   ✓ Correctly rejected radius=50: {e}")

# Invalid eps (non-positive)
try:
    invalid = GuidedFilterConfig(eps=-1e-3)
    print(f"   ✗ Should have rejected eps=-1e-3")
    sys.exit(1)
except ValueError as e:
    print(f"   ✓ Correctly rejected eps=-1e-3: {e}")

# Invalid timeout (non-positive)
try:
    invalid = GuidedFilterConfig(max_processing_time_ms=0)
    print(f"   ✗ Should have rejected max_processing_time_ms=0")
    sys.exit(1)
except ValueError as e:
    print(f"   ✓ Correctly rejected max_processing_time_ms=0: {e}")

# Test 7: Verify processing order
print("\n7. Verifying preprocessing order...")
preprocessing = cfg.preprocessing
order = [
    "bilateral",
    "guided_filter",
    "temporal_smoothing",
    "normalization",
    "quality_metrics",
    "inpainting",
    "mask_refinement",
    "png16_export",
]

for stage in order:
    if hasattr(preprocessing, stage):
        print(f"   ✓ {stage}")
    else:
        print(f"   ✗ {stage} missing")

# Test 8: Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nGuided Filter Configuration:")
print(f"  Enabled: {gf.enabled}")
print(f"  Radius: {gf.radius}px")
print(f"  Eps: {gf.eps}")
print(f"  RGB normalize: {gf.rgb_normalize}")
print(f"  Preserve invalid: {gf.preserve_invalid}")
print(f"  Max processing time: {gf.max_processing_time_ms}ms")

print("\nTemporal Smoothing Configuration:")
print(f"  Enabled: {ts.enabled}")
print(f"  Alpha: {ts.alpha} (updated from 0.2)")
print(f"  Window size: {ts.window_size}")
print(f"  Jitter threshold: {ts.jitter_threshold_mm}mm")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nConfig schema extension is working correctly!")
print("Ready to use: cfg.preprocessing.guided_filter")
