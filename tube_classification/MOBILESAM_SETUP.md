# MobileSAM Weights Setup - Quick Solutions

## Problem
```
ERROR | MobileSAM weights not found at models\mobile_sam.pt
```

The pipeline needs MobileSAM model weights (~40MB) to run segmentation.

---

## Solution 1: Automatic Download (Easiest) ✅

Run the setup script:

```bash
cd tube_classification
python setup_mobilesam.py
```

This will:
1. Create `models/` directory
2. Download `mobile_sam.pt` from GitHub (~40MB)
3. Verify the download
4. Test that it loads correctly
5. Show success message

**Time needed**: 2-5 minutes (depends on internet speed)

---

## Solution 2: Manual Download

If automatic download fails:

1. **Visit**: https://github.com/ChaoningZhang/MobileSAM/releases
2. **Download**: `mobile_sam.pt` (v1 release)
3. **Create folder**: `tube_classification/models/`
4. **Place file**: `tube_classification/models/mobile_sam.pt`
5. **Verify**: File should be ~38-40MB

**Folder structure**:
```
tube_classification/
├── models/
│   └── mobile_sam.pt          ← Place file here
├── config/
├── src/
└── ...
```

---

## Solution 3: Quick Command (curl/wget)

**Windows (PowerShell)**:
```powershell
mkdir models
cd models
Invoke-WebRequest -Uri "https://github.com/ChaoningZhang/MobileSAM/releases/download/v1/mobile_sam.pt" -OutFile "mobile_sam.pt"
cd ..
```

**Linux/Mac (bash)**:
```bash
mkdir -p models
cd models
wget https://github.com/ChaoningZhang/MobileSAM/releases/download/v1/mobile_sam.pt
cd ..
```

---

## Verify It Works

After setup, run:

```bash
python main.py
```

You should see:
```
INFO | Verification gate: Checking MobileSAM weights...
INFO | Verification gate passed. All systems ready.
```

If still failing, check:

```python
# Verify in Python
from pathlib import Path
import torch

weights_path = Path("models/mobile_sam.pt")
print(f"File exists: {weights_path.exists()}")
print(f"File size: {weights_path.stat().st_size / 1e6:.1f}MB")

# Try loading
state_dict = torch.load(str(weights_path), map_location="cpu")
print(f"Load successful! Keys: {len(state_dict)}")
```

---

## Troubleshooting

**Q: Download is stuck**
A: The file is ~40MB. On slow internet it can take 5-10 min. If it times out after 30 min, try manual download.

**Q: "File corrupted" error**
A: Delete `models/mobile_sam.pt` and re-run setup script.

**Q: "Permission denied" error**
A: Make sure you have write permissions to `tube_classification/` folder.

**Q: Still not working?**
A: Download manually from GitHub and place at exact path: `tube_classification/models/mobile_sam.pt`

---

## Alternative: Use Pre-Downloaded Weights

If you already have `mobile_sam.pt` from another location:

1. Copy it to: `tube_classification/models/mobile_sam.pt`
2. Verify: File should be ~38-40MB
3. Run: `python main.py`

---

## File Size Reference

Correct file should be:
- **Size**: 38-40MB (exact: 37,935,640 bytes)
- **Format**: PyTorch state dict (.pt)
- **Model**: MobileSAM v1

If your file is much smaller (<30MB) or larger (>50MB), it's likely wrong and should be re-downloaded.

---

## Next Steps After Setup

Once weights are in place:

```bash
# Verify all systems
python verify_system.py

# Run pipeline
python main.py

# If issues, check logs
cat logs/*.log
```

You're all set! 
