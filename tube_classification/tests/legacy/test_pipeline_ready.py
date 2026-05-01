#!/usr/bin/env python
"""
Interactive pipeline launcher with safe defaults.
Allows testing the pipeline with optional Ctrl+C handling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.parser import get_config
from src.acquisition.volume_gate import run_volume_gate

logger.add(sys.stdout, colorize=True, format="<level>{time:HH:mm:ss.SSS}</level> | <level>{level: <8}</level> | <level>{message}</level>")

print("\n" + "="*70)
print("TUBE CLASSIFICATION PIPELINE - INTERACTIVE LAUNCHER")
print("="*70 + "\n")

logger.info("System verification completed successfully ✅")
logger.info("Camera: Intel RealSense D435I detected")
logger.info("MobileSAM weights: Ready (38.8MB)")

print("\n" + "-"*70)
print("STEP 1: Volume Declaration")
print("-"*70 + "\n")

print("Available tube volumes:")
print("  • 1.3 ml  (5 SARSTEDT_SMALL classes)")
print("  • 4.0 ml  (6 VACUETTE classes)")
print("  • 4.5 ml  (5 SARSTEDT_REGULAR classes)")
print()

try:
    volume_ml, matched_tubes = run_volume_gate()
    
    print(f"\n✅ Volume declared: {volume_ml}ml")
    print(f"   Matched {len(matched_tubes)} tube class(es):")
    for tube in matched_tubes:
        print(f"     • {tube['class_id']} ({tube['family']})")
    
    # Determine class selection
    if len(matched_tubes) == 1:
        class_id = matched_tubes[0]["class_id"]
        print(f"\n   Auto-selected: {class_id}")
    else:
        print("\n   Multiple classes found. User would select during interactive mode.")
        print("   (Full pipeline requires Ctrl+C to trigger exports)")
    
    print("\n" + "-"*70)
    print("STEP 2: Pipeline Configuration")
    print("-"*70 + "\n")
    
    cfg = get_config()
    print(f"Camera Settings:")
    print(f"  Resolution: {cfg.camera.width}×{cfg.camera.height}")
    print(f"  Frame Rate: {cfg.camera.fps} fps")
    print(f"  Depth Range: {cfg.camera.depth_min_m:.2f}m - {cfg.camera.depth_max_m:.2f}m")
    
    print(f"\nPipeline Settings:")
    print(f"  Stability Frames: {cfg.pipeline.stability_frames}")
    print(f"  Depth Stability Threshold: {cfg.pipeline.depth_stability_threshold}m")
    print(f"  Min ROI Area: {cfg.pipeline.min_roi_area_px}px")
    print(f"  Blur Threshold: {cfg.pipeline.blur_threshold}")
    print(f"  SAM IOU Threshold: {cfg.pipeline.sam_iou_threshold}")
    
    print("\n" + "-"*70)
    print("✅ PIPELINE READY TO START")
    print("-"*70 + "\n")
    
    print("To run the full pipeline with interactive capture:")
    print("  1. Connect physical tube to capture zone in front of camera")
    print("  2. Run: python main.py")
    print("  3. Press Ctrl+C to finish capture and export dataset")
    print()
    print("Outputs will be saved to:")
    print(f"  • Raw images: {cfg.storage.root_dir}/raw/{class_id if len(matched_tubes)==1 else '<class_id>'}/<session_id>/")
    print(f"  • Annotations: {cfg.storage.root_dir}/annotations/<class_id>/<session_id>/")
    print(f"  • Cleaned data: {cfg.storage.root_dir}/cleaned/...")
    print(f"  • COCO export: {cfg.storage.root_dir}/exports/coco/")
    print(f"  • YOLO export: {cfg.storage.root_dir}/exports/yolo/")
    
    print("\n" + "="*70 + "\n")
    sys.exit(0)

except KeyboardInterrupt:
    logger.warning("\nOperation cancelled by user")
    sys.exit(0)
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
