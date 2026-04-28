# 🔧 CAMERA FRAME TIMEOUT FIX
## Resolving "Frame didn't arrive within 5000" Error

---

## ❌ ERROR DESCRIPTION

```
RuntimeError: Frame didn't arrive within 5000
  File ".../src/acquisition/streamer.py", line 117, in get_aligned_frames
    frames = self.pipeline.wait_for_frames()
```

**Meaning:** Camera initialized successfully but not delivering frames

---

## 🔍 DIAGNOSIS STEPS

### Step 1: Run Diagnostics
```bash
python diagnose_camera.py
```

This will check:
- ✓ Hardware (camera connected?)
- ✓ Configuration (settings valid?)
- ✓ Supported modes (what resolutions work?)
- ✓ Frame capture (can it get frames?)
- ✓ USB connection (proper port?)

### Step 2: Analyze Results
```
✅ PASS: Hardware      → Camera detected
✅ PASS: Configuration → Settings correct
❌ FAIL: Frame Capture → Camera not sending frames
```

---

## 🛠️ SOLUTIONS (In Order of Likelihood)

### Solution 1: USB Connection Issue (Most Common)

**Check:**
1. Is camera in **USB 3.0 port** (blue, not black USB 2.0)?
2. Is the cable **directly connected** (not through hub)?
3. Is the cable **high quality** (not old/flimsy)?

**Fix:**
```
1. Unplug camera
2. Plug into different USB 3.0 port (try all blue ports)
3. Use direct connection (no USB hub)
4. If available, use powered USB hub
5. Restart camera
```

**Verify:**
```bash
python diagnose_camera.py
# Should show: ✅ PASS: Frame Capture
```

### Solution 2: Camera Driver Issue

**Update Drivers:**
```bash
# Windows: Update pyrealsense2
pip install --upgrade pyrealsense2

# Alternatively, install latest version
pip install pyrealsense2==2.54.2.5678
```

**Or Update RealSense SDK:**
```
Download from: https://github.com/IntelRealSense/librealsense/releases
Install latest SDK
Restart computer
```

**Verify:**
```bash
python diagnose_camera.py
```

### Solution 3: Resolution Not Supported

**Check supported modes:**
```bash
python diagnose_camera.py
# Look for "SUPPORTED" resolutions
```

**If 1280×720 not supported, use fallback:**
```yaml
# Edit config/config.yaml:
camera:
  width: 1024    # instead of 1280
  height: 768    # instead of 720
  fps: 30        # instead of 15
```

**Or use most conservative:**
```yaml
camera:
  width: 640
  height: 480
  fps: 30
```

### Solution 4: RealSense Service Issue

**Windows:**
```
1. Open Device Manager
2. Right-click on Intel RealSense D435
3. Select "Uninstall device"
4. Disconnect camera
5. Wait 10 seconds
6. Reconnect camera
7. Windows will reinstall drivers
```

### Solution 5: Increase Timeout

If camera is very slow, increase timeout:

**Temporary (for testing):**
```python
# In Python code:
frames = streamer.get_aligned_frames(timeout_ms=10000)  # 10 seconds instead of 5
```

**Permanent (in streamer.py):**
```python
def get_aligned_frames(self, timeout_ms: int = 10000):  # Increase from 5000
    ...
```

---

## 📋 TROUBLESHOOTING CHECKLIST

### Hardware
- [ ] Camera connected to USB port
- [ ] Using USB 3.0 port (blue, not black)
- [ ] Cable is high quality
- [ ] No USB hub in between
- [ ] Camera powered (LED indicator on?)

### Software
- [ ] RealSense drivers installed
- [ ] pyrealsense2 package installed
- [ ] config.yaml has valid resolution
- [ ] Camera not in use by another program

### Configuration
- [ ] Width: Valid value
- [ ] Height: Valid value
- [ ] FPS: Valid value
- [ ] Resolution combination supported by camera

### Camera Firmware
- [ ] Firmware is up to date
- [ ] No pending firmware updates
- [ ] Camera not in recovery mode

---

## ✅ QUICK FIX PROCESS

If you see frame timeout error:

```bash
# 1. Run diagnostics
python diagnose_camera.py

# 2. Based on results:
#    - If "❌ FAIL: Hardware" → Check USB cable
#    - If "❌ FAIL: Resolution Support" → Use lower resolution
#    - If "❌ FAIL: Frame Capture" → Try different USB port

# 3. If USB issue:
#    - Try different USB 3.0 port
#    - Use direct connection (no hub)
#    - Update drivers

# 4. Test again:
python diagnose_camera.py

# 5. When diagnostics pass:
python capture.py
```

---

## 🎯 EXPECTED SUCCESS

When fixed, you should see:

```
✅ PASS: Hardware
✅ PASS: Configuration  
✅ PASS: Resolution Support
✅ PASS: Frame Capture

Camera: 1280×720 @ 15fps
First frame: 1280×720
✅ Ready to capture
```

---

## 📞 STILL STUCK?

### If "Hardware Not Detected"
```
- Check camera connection
- Try different USB port
- Use Device Manager to verify camera is listed
- Reinstall RealSense SDK
```

### If "Frame Capture Fails"
```
- Try USB 3.0 port (must be blue)
- Update drivers
- Try lower resolution (640×480@30)
- Restart computer and camera
```

### If "Resolution Not Supported"
```
- Use 1024×768@30 (good alternative)
- Or 848×480@30 (original config)
- Or 640×480@30 (safe fallback)
```

---

## 🚀 WHEN FIXED

Once diagnostics pass:

```bash
# Run pipeline
python capture.py

# Monitor for HD capture:
# - Console: "Camera: 1280×720@15fps"
# - File sizes: ~2.5 MB per image
# - Blur scores: >50
```

---

## 📊 RESOLUTION FALLBACK SEQUENCE

Camera will try in this order:
```
1. 1280×720 @ 15fps  (HD - preferred)
2. 1024×768 @ 30fps  (Good balance)
3. 848×480 @ 30fps   (Original)
4. 640×480 @ 30fps   (Emergency)
```

If 1280×720 fails, it automatically tries next resolution.

---

## ✨ SUMMARY

**Frame timeout error usually means:**
- ✗ USB connection issue (most common)
- ✗ Camera not in USB 3.0 port
- ✗ Old USB cable or hub
- ✗ Outdated drivers
- ✗ Unsupported resolution

**Fix:**
1. Run `python diagnose_camera.py`
2. Follow the guidance
3. Test with different USB port
4. Update drivers if needed
5. Try lower resolution if needed

**Result:** Camera will capture frames, pipeline will run, HD images will be captured!

---

**Status:** Frame timeout identified and fixable
**Next Action:** Run `python diagnose_camera.py`
**Expected:** Fixed within 30 minutes

