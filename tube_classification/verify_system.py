#!/usr/bin/env python
"""
Direct verification test - checks each component without running full pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
import shutil

logger.remove()  # Remove default handler
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<level>{message}</level>")

print("\n" + "="*60)
print("TUBE CLASSIFICATION PIPELINE - SYSTEM CHECK")
print("="*60 + "\n")

# Import config parser
from config.parser import load_config, get_config, DEFAULT_CONFIG_PATH

# CHECK 1: Config file exists
print("CHECK 1: Config file exists... ", end="", flush=True)
if not DEFAULT_CONFIG_PATH.exists():
    print(f"❌ FAIL\n  Expected at: {DEFAULT_CONFIG_PATH}")
    sys.exit(1)
print("✅ PASS")

# CHECK 2: Config schema validation
print("CHECK 2: Config schema valid... ", end="", flush=True)
try:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    print("✅ PASS")
except Exception as e:
    print(f"❌ FAIL\n  {e}")
    sys.exit(1)

# CHECK 3: Output directory writable + disk space
print("CHECK 3: Output directory writable... ", end="", flush=True)
cfg = get_config()
root_dir = Path(cfg.storage.root_dir)
root_dir.mkdir(parents=True, exist_ok=True)

try:
    test_file = root_dir / ".write_test"
    test_file.write_text("test")
    test_file.unlink()
    print("✅ PASS")
except Exception as e:
    print(f"❌ FAIL\n  {e}")
    sys.exit(1)

print("CHECK 4: Disk space (>1GB)... ", end="", flush=True)
disk_usage = shutil.disk_usage(root_dir)
min_space = 1_073_741_824  # 1GB
if disk_usage.free < min_space:
    free_gb = disk_usage.free // 1_073_741_824
    print(f"❌ FAIL\n  Only {free_gb}GB free, need 1GB")
    sys.exit(1)
else:
    free_gb = disk_usage.free // 1_073_741_824
    print(f"✅ PASS ({free_gb}GB free)")

# CHECK 4: RealSense camera connected
print("CHECK 5: RealSense camera connected... ", end="", flush=True)
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("❌ FAIL\n  No RealSense camera detected")
        sys.exit(1)
    else:
        print(f"✅ PASS ({len(devices)} device(s) detected)")
        for i, device in enumerate(devices):
            dev_name = device.get_info(rs.camera_info.name)
            dev_serial = device.get_info(rs.camera_info.serial_number)
            print(f"     Device {i}: {dev_name} (Serial: {dev_serial})")
except Exception as e:
    print(f"❌ FAIL\n  {e}")
    sys.exit(1)

# CHECK 5: MobileSAM weights exist
print("CHECK 6: MobileSAM weights exist... ", end="", flush=True)
weights_path = Path(cfg.storage.sam_weights_path)
if not weights_path.exists():
    print(f"❌ FAIL\n  Expected at: {weights_path}")
    sys.exit(1)
else:
    size_mb = weights_path.stat().st_size / (1024*1024)
    print(f"✅ PASS ({size_mb:.1f}MB)")

print("\n" + "="*60)
print("✅ ALL CHECKS PASSED - SYSTEM READY")
print("="*60 + "\n")
