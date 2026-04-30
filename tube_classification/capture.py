"""Main entry point for interactive tube capture."""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import yaml


def load_yaml(path: str) -> dict:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    """Save YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def run_startup_checks(config: dict, calib: dict) -> bool:
    """Run 8-point startup verification."""
    print("\n[STARTUP CHECKS]")
    checks = []
    
    # 1. Config file
    try:
        assert config is not None and 'camera' in config
        print("✓ Config loaded: config/config.yaml")
        checks.append(True)
    except:
        print("✗ Config file missing or invalid")
        checks.append(False)
    
    # 2. Calibration file
    try:
        assert calib is not None and 'grid_map' in calib
        print("✓ Calibration loaded: calibration.yaml")
        checks.append(True)
    except:
        print("✗ Calibration file missing or invalid")
        checks.append(False)
    
    # 3. Grid map
    try:
        grid_map = calib.get('grid_map', {})
        rows = grid_map.get('rows', 5)
        cols = grid_map.get('cols', 10)
        print(f"✓ Grid map loaded: {rows}×{cols} grid")
        checks.append(True)
    except:
        print("✗ Grid map invalid")
        checks.append(False)
    
    # 4. Depth baseline
    try:
        baseline = calib.get('depth_baseline_mm', 330)
        tolerance = config.get('pipeline', {}).get('preview', {}).get('depth_tolerance_mm', 20)
        print(f"✓ Depth baseline: {baseline}mm ± {tolerance}mm")
        checks.append(True)
    except:
        print("✗ Depth baseline invalid")
        checks.append(False)
    
    # 5. Dataset directory
    try:
        session_dir = Path(config.get('pipeline', {}).get('session', {}).get('base_directory', 'dataset/sessions'))
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Dataset directory ready: {session_dir}")
        checks.append(True)
    except:
        print("✗ Cannot create dataset directory")
        checks.append(False)
    
    # 6. Camera (try import)
    try:
        import pyrealsense2 as rs
        print("✓ RealSense SDK available")
        checks.append(True)
    except ImportError:
        print("⚠ RealSense SDK not installed (required for actual capture)")
        checks.append(True)  # Don't fail on this
    
    # 7. OpenCV
    try:
        import cv2
        print("✓ OpenCV available")
        checks.append(True)
    except ImportError:
        print("✗ OpenCV not installed")
        checks.append(False)
    
    # 8. PyYAML
    try:
        import yaml
        print("✓ PyYAML available")
        checks.append(True)
    except ImportError:
        print("✗ PyYAML not installed")
        checks.append(False)
    
    all_pass = all(checks)
    print(f"\n[RESULT] {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")
    return all_pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive tube capture system with grid calibration"
    )
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--calibration', default='config/calibration.yaml', help='Calibration file path')
    parser.add_argument('--session-id', default=None, help='Resume interrupted session')
    parser.add_argument('--skip-preview', action='store_true', help='Skip live preview (testing)')
    parser.add_argument('--dry-run', action='store_true', help='Run without camera')
    parser.add_argument('--skip-pre-capture', action='store_true', help='Skip pre-capture data entry')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  TUBE CLASSIFICATION PIPELINE - INTERACTIVE CAPTURE")
    print("="*60)
    
    # Load configuration
    try:
        config = load_yaml(args.config)
        print(f"\n[CONFIG] Loaded: {args.config}")
    except Exception as e:
        print(f"\n✗ Failed to load config: {e}")
        return 1
    
    # Load calibration
    try:
        calib = load_yaml(args.calibration)
        print(f"[CALIB] Loaded: {args.calibration}")
    except Exception as e:
        print(f"✗ Failed to load calibration: {e}")
        return 1
    
    # Run startup checks
    if not run_startup_checks(config, calib):
        if not args.dry_run:
            return 1
        print("\n[DRY-RUN] Continuing despite failures...")
    
    active_session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pre-capture workflow (persistent SQLite store)
    if not args.skip_pre_capture:
        try:
            from pre_capture import run_pre_capture_workflow
            base_dir = Path(config.get('pipeline', {}).get('session', {}).get('base_directory', 'dataset/sessions'))
            run_pre_capture_workflow(session_id=active_session_id, base_dir=base_dir)
        except Exception as e:
            print(f"✗ Pre-capture workflow failed: {e}")
            return 1

    # Initialize camera
    if not args.dry_run:
        try:
            from src.acquisition.camera_manager import CameraManager
            print("\n[CAMERA] Initializing...")
            camera = CameraManager(config)
            if not camera.is_connected():
                print("✗ Camera not detected. Check USB connection.")
                return 1
            print(f"✓ Camera connected: Intel RealSense")
        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            return 1
    else:
        camera = None
        print("\n[DRY-RUN] Camera skipped")
    
    # Run orchestrator
    try:
        from src.preview.session_orchestrator import CaptureSessionOrchestrator
        
        print("\n[SESSION] Starting capture session...")
        print("[INFO] Press [S] for SINGLE, [B] for BATCH, [C] for calibration, [Q] to quit")
        
        orchestrator = CaptureSessionOrchestrator(
            camera_manager=camera,
            grid_map=calib.get('grid_map', {}),
            calib_params=calib,
            config=config,
            session_id=active_session_id,
        )
        
        summary = orchestrator.run_session()
        
        print(f"\n[OK] Session complete!")
        print(f"  Accepted: {summary['accepted']}")
        print(f"  Rejected: {summary['rejected']}")
        print(f"  Location: {summary['session_path']}")
        
        orchestrator.cleanup()
        return 0
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Saving session...")
        if camera:
            camera.stop()
        return 0
    except Exception as e:
        print(f"\n✗ Session error: {e}")
        import traceback
        traceback.print_exc()
        if camera:
            camera.stop()
        return 1


if __name__ == '__main__':
    exit(main())
