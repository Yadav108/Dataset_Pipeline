#!/usr/bin/env python3
"""Quick test to verify all imports work after resolution changes."""

try:
    print("Testing imports...")
    from src.acquisition.streamer import RealSenseStreamer
    print("✓ streamer.py")
    
    from src.acquisition.preprocessing import preprocess_depth_bilateral, compute_quality_metrics
    print("✓ preprocessing.py")
    
    from src.acquisition.advanced_preprocessing import inpaint_depth_telea
    print("✓ advanced_preprocessing.py")
    
    from src.acquisition.pipeline_integration import PreprocessingPipeline
    print("✓ pipeline_integration.py")
    
    from config.parser import get_config
    print("✓ config.parser")
    
    config = get_config()
    print(f"✓ Config loaded: {config.camera.width}x{config.camera.height} @ {config.camera.fps}fps")
    
    print("\n✅ All imports successful! Ready to run pipeline.")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
