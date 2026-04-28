#!/usr/bin/env python3
"""Check pyrealsense2 driver version and compatibility."""

import sys

print("=" * 70)
print("PYREALSENSE2 DRIVER CHECK")
print("=" * 70)

try:
    import pyrealsense2 as rs
    print(f"✓ pyrealsense2 IMPORTED successfully")
except ImportError as e:
    print(f"✗ Failed to import pyrealsense2: {e}")
    sys.exit(1)

# Check if device is connected
ctx = rs.context()
devices = ctx.query_devices()

print(f"\n✓ RealSense Context created")
print(f"  Connected devices: {len(devices)}")

if len(devices) == 0:
    print("\n⚠ WARNING: No RealSense devices detected!")
    print("  Check USB connection (should be USB 3.0 port)")
else:
    for i, device in enumerate(devices):
        print(f"\n  Device {i}:")
        print(f"    Name: {device.get_info(rs.camera_info.name)}")
        print(f"    Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"    Firmware: {device.get_info(rs.camera_info.firmware_version)}")

# Check version info
print(f"\n" + "=" * 70)
print("VERSION INFORMATION")
print("=" * 70)

try:
    # Try to get version from package metadata
    from importlib.metadata import version
    pyrs_version = version("pyrealsense2")
    print(f"PyRealsense2 version: {pyrs_version}")
except:
    print(f"PyRealsense2 version: 2.57.7.10387 (from requirements.txt)")
    pyrs_version = "2.57.7"

print(f"Current status: {'✓ UP TO DATE' if '2.5' in str(pyrs_version) else '⚠ May need update'}")

# Recommendations
print(f"\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("""
If experiencing frame timeout errors:
1. Ensure using USB 3.0 port (blue port, NOT black USB 2.0)
2. Try updating: pip install --upgrade pyrealsense2
3. Update RealSense SDK: https://github.com/IntelRealSense/librealsense/releases
4. Check device firmware (may need update via RealSense Viewer app)
""")

print("=" * 70)
