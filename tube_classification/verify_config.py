#!/usr/bin/env python
"""Verify that the config loads correctly with preprocessing section."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.parser import load_config, get_config

print("=" * 70)
print("Config Verification")
print("=" * 70)

try:
    # Load config
    cfg = load_config()
    print("✓ Config loaded successfully")
    
    # Check preprocessing section
    print(f"✓ Preprocessing section exists: {hasattr(cfg, 'preprocessing')}")
    
    if hasattr(cfg, 'preprocessing'):
        prep = cfg.preprocessing
        print(f"  ├─ Bilateral: {prep.bilateral.enabled}")
        print(f"  ├─ Temporal smoothing: {prep.temporal_smoothing.enabled}")
        print(f"  │  └─ alpha: {prep.temporal_smoothing.alpha}")
        print(f"  ├─ Normalization: {prep.normalization.enabled}")
        print(f"  ├─ Quality metrics: {prep.quality_metrics.enabled}")
        print(f"  ├─ Inpainting: {prep.inpainting.enabled}")
        print(f"  ├─ Mask refinement: {prep.mask_refinement.enabled}")
        print(f"  └─ PNG16 export: {prep.png16_export.enabled}")
    
    print("")
    print("=" * 70)
    print("✓ Config OK! Ready to run pipeline.")
    print("=" * 70)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
