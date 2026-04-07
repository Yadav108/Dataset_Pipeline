#!/usr/bin/env python3
"""
DEVICE FIRMWARE EXPLANATION & CHECK
======================================

What is Device Firmware?
------------------------
Firmware is the LOW-LEVEL software that runs ON THE CAMERA HARDWARE itself.
It's separate from the Python driver (pyrealsense2).

Think of it like:
- Python driver = instructions your computer gives to the camera
- Device firmware = instructions that run INSIDE the camera processor

Why Update Firmware?
--------------------
1. BUG FIXES - Fix frame timeout, USB issues, stability problems
2. PERFORMANCE - Better frame capture, faster processing
3. COMPATIBILITY - Support new features, better Python driver compatibility
4. STABILITY - Reduces crashes and errors like "Frame didn't arrive"

Your D435i Camera Firmware
---------------------------
The firmware version is typically shown as: 5.X.X.X (for D435i)

Current vs Latest:
- OLD firmware (2022-2023): Often has frame timeout issues
- NEW firmware (2024+): Much more stable, recommended

Frame Timeout Root Causes (Firmware Related):
- Outdated firmware doesn't properly signal frame ready
- USB bandwidth negotiation fails with new drivers
- Depth stream buffer overflow in low-memory situations
"""

import pyrealsense2 as rs

print(__doc__)

print("\n" + "=" * 70)
print("CHECKING YOUR CAMERA FIRMWARE")
print("=" * 70)

try:
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("⚠ No RealSense devices detected!")
        print("Make sure camera is connected to USB 3.0 port (BLUE port)")
    else:
        for i, device in enumerate(devices):
            print(f"\nDevice {i}:")
            print(f"  Name: {device.get_info(rs.camera_info.name)}")
            print(f"  Serial: {device.get_info(rs.camera_info.serial_number)}")
            
            firmware_version = device.get_info(rs.camera_info.firmware_version)
            print(f"  Firmware Version: {firmware_version}")
            
            # Parse version
            try:
                major = int(firmware_version.split('.')[0])
                if major >= 5:
                    print(f"  Status: ✓ RECENT (5.x.x.x is current)")
                else:
                    print(f"  Status: ⚠ OUTDATED (Consider updating)")
            except:
                print(f"  Status: ? Cannot determine from version string")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)
print("HOW TO UPDATE FIRMWARE")
print("=" * 70)
print("""
STEP 1: Download RealSense Viewer
   Go to: https://github.com/IntelRealSense/librealsense/releases
   Download: Intel.RealSense.Viewer-WIN.exe (latest version)
   
STEP 2: Install RealSense Viewer
   Run the installer
   Choose "Custom Installation" and select "Viewer" and "Tools"
   
STEP 3: Launch RealSense Viewer
   Connect camera to USB 3.0 port (BLUE port, not black)
   Open Intel RealSense Viewer application
   
STEP 4: Check for Firmware Updates
   Go to: Tools → Firmware Update
   Or: Menu → Devices → Firmware Update
   
STEP 5: Follow Update Wizard
   The app will:
   - Detect your camera
   - Check latest firmware version
   - Download firmware if newer available
   - Update device (takes 1-2 minutes)
   - Camera will disconnect/reconnect during update
   
STEP 6: Verify Update
   Run this script again to confirm new firmware version
""")

print("\n" + "=" * 70)
print("IMPORTANT NOTES")
print("=" * 70)
print("""
1. DO NOT DISCONNECT CAMERA during firmware update!
   - This can BRICK the device
   - Keep USB connected, let it complete
   
2. USE USB 3.0 PORT ONLY
   - Firmware update fails on USB 2.0
   - Look for BLUE USB port on your computer
   - Not all USB 3.0 ports work - try different ports if fails
   
3. CLOSE OTHER APPLICATIONS
   - Close any Python scripts using the camera
   - Close RealSense SDK applications
   - Let Viewer have exclusive access
   
4. Expected Firmware Versions for D435i
   - Version 5.10+: Latest (2024)
   - Version 5.8-5.9: Recent (2023-2024)
   - Version 5.0-5.7: Older (may have timeout issues)
   - Version 4.x or older: Outdated (NOT recommended)
""")

print("=" * 70)
