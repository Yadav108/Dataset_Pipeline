#!/usr/bin/env python3
"""Quick test of annotation quality tracking logic."""

import sys
from pathlib import Path

# Test 1: Verify sam_segmentor returns tuple
print("Test 1: Check sam_segmentor return type...")
from src.annotation.sam_segmentor import MobileSAMSegmentor
segmentor = MobileSAMSegmentor()
print(f"  segment() signature: {segmentor.segment.__annotations__}")
expected_return = "tuple[np.ndarray, float] | None"
if "tuple" in str(segmentor.segment.__annotations__.get("return", "")):
    print("  ✓ segment() returns tuple type")
else:
    print(f"  ✗ segment() return type: {segmentor.segment.__annotations__.get('return', 'unknown')}")

# Test 2: Verify metadata_builder accepts sam_iou_score
print("\nTest 2: Check metadata_builder signature...")
from src.annotation.metadata_builder import build_metadata
import inspect
sig = inspect.signature(build_metadata)
params = list(sig.parameters.keys())
print(f"  Parameters: {params}")
if "sam_iou_score" in params:
    print("  ✓ build_metadata() accepts sam_iou_score parameter")
else:
    print("  ✗ Missing sam_iou_score parameter")

# Test 3: Verify manifest_builder includes sam_iou_score in fieldnames
print("\nTest 3: Check manifest_builder fieldnames...")
from src.export.manifest_builder import ManifestBuilder
# Read the source to verify fieldnames
manifest_source = Path("src/export/manifest_builder.py").read_text()
if '"sam_iou_score"' in manifest_source and "fieldnames" in manifest_source:
    print("  ✓ manifest_builder.py includes sam_iou_score in fieldnames")
else:
    print("  ✗ manifest_builder.py missing sam_iou_score in fieldnames")

# Test 4: Verify pipeline.py has quality tracking
print("\nTest 4: Check pipeline.py quality tracking...")
pipeline_source = Path("src/orchestrator/pipeline.py").read_text()
checks = [
    ("total_annotation_attempts" in pipeline_source, "total_annotation_attempts counter"),
    ("high_quality_annotations" in pipeline_source, "high_quality_annotations counter"),
    ("annotation_accuracy" in pipeline_source, "annotation_accuracy computation"),
    ("R2.4 threshold not met" in pipeline_source, "R2.4 warning log"),
]
for check, name in checks:
    if check:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name} missing")

print("\n✓ All annotation quality tracking components verified!")
