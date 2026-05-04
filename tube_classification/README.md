# Tube Classification Dataset Pipeline

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Automated dataset collection and annotation pipeline for tube classification using RealSense D435i depth camera and MobileSAM segmentation.**

## 📋 Overview

This project provides a complete pipeline for:
- 🎥 **Capture**: High-quality image acquisition at 848×480@30fps from RealSense D435i
- 📍 **ROI Extraction**: Depth-based region of interest detection
- 🎯 **Segmentation**: MobileSAM-based semantic segmentation for tube boundaries
- 🧹 **Cleaning**: Blur detection, duplicate removal, quality filtering
- 📊 **Export**: COCO and YOLO format dataset export

**Key Features:**
- ✅ Optimized for 22cm camera-to-tube distance
- ✅ Adaptive quality thresholds (blur, coverage, IoU)
- ✅ Memory-efficient processing (848×480 resolution)
- ✅ Real-time depth stability detection
- ✅ Automatic checkpoint/recovery system
- ✅ Comprehensive quality metrics and reporting

---

## 🏗️ Project Organization

This project follows industrial-grade structure with clear separation of concerns:

- **src/** — Production code (capture, annotation, cleaning, export)
- **tests/** — Test suite with active tests + preserved legacy scripts
- **docs/** — Organized documentation (guides, troubleshooting, archive)
- **config/** — Configuration files
- **dataset/**, **models/**, **logs/** — Data and artifacts

**Recent cleanup (May 2026):**
- Documentation reorganized into [docs/guides/](docs/guides/), [docs/troubleshooting/](docs/troubleshooting/), [docs/archive/](docs/archive/)
- Legacy test scripts moved to [tests/legacy/](tests/legacy/) (preserved, not deleted)
- Root directory cleaned: removed debug artifacts, standardized for production workflows
- Python requirements cleaned and pinned to compatible versions (>=3.10,<3.13)

---

### Prerequisites
- **Camera**: Intel RealSense D435i
- **Python**: 3.10+ (tested on 3.12)
- **OS**: Windows 10/11 (Linux support: use WSL2)
- **Storage**: ~10GB for full dataset

### Installation

1. **Clone repository**
```bash
git clone https://github.com/Yadav108/Dataset_Pipeline.git
cd tube_classification
```

2. **Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download MobileSAM weights**
```bash
mkdir models
# Download from: https://github.com/ChaoningZhang/MobileSAM/releases
# Place: models/mobile_sam.pt
```

### First Run

```bash
python main.py
```

This will:
1. ✅ Run verification gate (checks camera, config, dependencies)
2. ✅ Initialize RealSense camera
3. ✅ Start interactive capture mode
4. ✅ Process and annotate images
5. ✅ Export dataset in COCO/YOLO formats

---

## 🎮 Usage

### Basic Capture Session
```bash
python main.py
```

The pipeline will:
- Run pre-capture data entry (volume/class records) with SQLite persistence
- Wait for stable depth frames (4 consecutive stable frames)
- Extract ROI containing tube
- Run MobileSAM segmentation
- Apply quality filters (blur, coverage, IoU)
- Export annotated image + mask + metadata

To skip pre-capture data entry:
```bash
python capture.py --skip-pre-capture
```

### Capture with Custom Config
Edit `config/config.yaml` before running:
```yaml
camera:
  width: 848
  height: 480
  fps: 30
  depth_min_m: 0.17      # 22cm - 5cm margin
  depth_max_m: 0.27      # 22cm + 5cm margin

pipeline:
  blur_threshold: 52.0           # Higher = stricter
  min_coverage_ratio: 0.48       # Mask coverage requirement
  sam_iou_threshold: 0.57        # Segmentation confidence
  show_preview: false            # Display live preview
```

### Export Dataset
```bash
python -m src.export.export_coco    # COCO format
python -m src.export.export_yolo    # YOLO format
```

### Tube Data Manager (Volume/Class Capture)
Open `tube_data_manager.html` in a browser to manage tube classes and capture per-class quantity/batch/expiry/location notes with local persistent storage.

---

## 📁 Project Structure

```
tube_classification/
├── main.py                          # Entry point
├── config/
│   └── config.yaml                  # Configuration (camera, thresholds, etc)
├── src/
│   ├── acquisition/                 # Camera & frame capture
│   │   ├── streamer.py             # RealSense D435i interface
│   │   ├── stability_detector.py    # Depth stability analysis
│   │   └── volume_gate.py           # Volume validation
│   ├── annotation/                  # Image annotation
│   │   ├── sam_segmentor.py        # MobileSAM segmentation
│   │   ├── roi_extractor.py        # ROI detection from depth
│   │   ├── annotation_writer.py    # Metadata serialization
│   │   └── metadata_builder.py     # Annotation metadata
│   ├── cleaning/                    # Quality filtering
│   │   ├── blur_detector.py        # Blur detection (Laplacian variance)
│   │   ├── duplicate_remover.py    # Perceptual duplicate filtering
│   │   ├── bbox_quality_filter.py  # Mask quality validation
│   │   └── background_remover.py   # Background cleanup
│   ├── export/                      # Dataset export
│   │   ├── coco_exporter.py        # COCO JSON format
│   │   ├── yolo_exporter.py        # YOLO TXT format
│   │   └── dataset_stats.py        # Dataset balance reporting
│   └── orchestrator/                # Pipeline orchestration
│       └── pipeline.py              # Main pipeline loop
├── dataset/
│   ├── raw/                         # Captured images
│   ├── annotations/                 # Segmentation masks
│   ├── cleaned/                     # Post-processed images
│   └── exports/                     # COCO/YOLO datasets
├── models/
│   └── mobile_sam.pt               # (Download required)
├── docs/                            # Complete documentation
│   ├── guides/                      # Setup & how-to guides
│   ├── troubleshooting/             # Known issues & solutions
│   └── archive/                     # Historical docs
├── tests/                           # Test suite
│   ├── unit/legacy/                 # Legacy test scripts (preserved)
│   └── *.py                        # Active pytest suite
├── requirements.txt                 # Python dependencies (>=3.10,<3.13)
└── README.md                        # This file
```

### Documentation Organization
- **Start here:** [README.md](README.md) + [QUICK_START.md](QUICK_START.md)
- **Setup guides:** [docs/guides/](docs/guides/)
- **Configuration:** [docs/PREPROCESSING_CONFIG_GUIDE.md](docs/PREPROCESSING_CONFIG_GUIDE.md)
- **Troubleshooting:** [docs/troubleshooting/](docs/troubleshooting/)
- **Historical/milestones:** [docs/archive/](docs/archive/)

### Test Suite
- **Run tests:** `pytest` (active suite in `tests/`)
- **Documentation:** [tests/README.md](tests/README.md)
- **Legacy scripts:** [tests/legacy/](tests/legacy/) (reference/migration)

---

## 🎯 Configuration Guide

### Camera Settings

**Distance Calibration (22cm):**
```yaml
camera:
  depth_min_m: 0.17      # 22cm - 5cm margin
  depth_max_m: 0.27      # 22cm + 5cm margin
```

Replace with your actual camera-to-tube distance. For example:
- 20cm: `0.15` - `0.25`
- 25cm: `0.20` - `0.30`
- 30cm: `0.25` - `0.35`

### Quality Thresholds

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `blur_threshold` | 52.0 | 30-80 | Laplacian variance (higher = less strict) |
| `min_coverage_ratio` | 0.48 | 0.3-0.7 | Min mask coverage of bbox (higher = stricter) |
| `sam_iou_threshold` | 0.57 | 0.4-0.8 | SAM confidence (higher = stricter) |
| `duplicate_hash_threshold` | 12 | 5-30 | Image similarity (lower = stricter) |

**Tuning Tips:**
- **Low yield?** → Decrease thresholds (more lenient)
- **Poor quality?** → Increase thresholds (stricter)
- **Too much blur?** → Increase `blur_threshold`
- **Fragmented masks?** → Decrease `sam_iou_threshold`

---

## 🔧 Hardware Setup

### Camera Connection
- **Required**: USB 3.0 port (BLUE port)
- **NOT supported**: USB 2.0 (BLACK port)
- **Distance**: Position camera 22cm from tube
- **Angle**: Perpendicular to tube surface for best results

### Firmware Update (IMPORTANT!)
If experiencing frame timeout errors:

1. Download **RealSense Viewer**:
   ```
   https://github.com/IntelRealSense/librealsense/releases
   ```

2. Install and launch RealSense Viewer

3. Connect camera to USB 3.0 port

4. Go to: `Tools → Firmware Update`

5. Follow the update wizard

**Expected firmware**: Version 5.8+ (check in RealSense Viewer)

### Performance Notes
- Resolution: 848×480 (optimized for small objects)
- Frame rate: 30fps (stability important)
- Depth scale: 0.001 (D435i default)
- Processing: ~2 seconds/image (capture + annotation + cleaning)

---

## 📊 Quality Metrics

The pipeline tracks:
- **Capture Rate**: % of frames meeting stability requirements
- **Blur Rate**: % of images rejected as blurry
- **Mask Quality**: IoU scores from MobileSAM (target: >0.57)
- **Coverage**: Mask coverage ratio vs bbox area
- **Duplicate Rate**: % of images filtered as duplicates

Check metrics:
```bash
python analyze_image_quality.py    # Detailed quality analysis
python verify_hd_capture.py        # HD capture verification
```

---

## 🐛 Troubleshooting

### Frame Timeout Error: "Frame didn't arrive within 5000"
**Cause**: Camera not delivering frames
**Solutions**:
1. ✅ Check USB 3.0 connection (use BLUE port, not black)
2. ✅ Update device firmware (see Hardware Setup)
3. ✅ Try different USB 3.0 port on computer
4. ✅ Update pyrealsense2: `pip install --upgrade pyrealsense2`

### Low Capture Yield
**Solutions**:
1. Check blur detection threshold (increase if rejecting too many)
2. Verify camera distance is exactly 22cm
3. Ensure good lighting on tube
4. Check depth stability (may need >4 frames)

### Memory Errors
**Solutions**:
1. Disable preview: `show_preview: false` in config.yaml
2. Reduce batch size if running multiple processes
3. Close other applications
4. Increase virtual memory (Windows Settings)

### Poor Segmentation Masks
**Solutions**:
1. Lower SAM IoU threshold (more lenient)
2. Ensure camera focus at 22cm distance
3. Check lighting (darker background helps)
4. Update MobileSAM weights

---

## 📈 Performance Benchmarks

Typical session performance (on i7, 16GB RAM):

| Metric | Value |
|--------|-------|
| Frames captured/min | 40-50 |
| Processing time/image | 1.5-2.5s |
| Blur rejection rate | 15-25% |
| Segmentation success | 85-95% |
| Quality score | 8.2-8.8/10 |

---

## 🔄 Pipeline Workflow

```
1. CAPTURE PHASE
   └─ Wait for depth stability (4 frames)
   └─ Extract ROI from depth
   └─ Run MobileSAM segmentation
   
2. QUALITY FILTERING
   └─ Blur detection (Laplacian variance)
   └─ Coverage ratio check (mask vs bbox)
   └─ IoU threshold validation
   
3. CLEANING PHASE
   └─ Duplicate detection (ImageHash)
   └─ Background removal
   └─ BBox quality filtering
   
4. EXPORT PHASE
   └─ Save annotated images
   └─ Export COCO/YOLO format
   └─ Generate dataset statistics
```

---

## 📝 File Formats

### Dataset Structure
```
dataset/raw/
├── raw/class_id/
│   ├── frame_001.png
│   ├── frame_001.npz    (depth)
│   └── frame_001_mask.png
├── annotations/class_id/
│   ├── frame_001.json   (COCO format)
│   └── metadata/
└── manifests/
    └── manifest.yaml    (session metadata)
```

### COCO Format
```json
{
  "image_id": 1,
  "annotations": [
    {
      "id": 1,
      "bbox": [x, y, w, h],
      "segmentation": {...},
      "area": 1234,
      "iscrowd": 0
    }
  ]
}
```

### YOLO Format
```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Multi-camera support
- [ ] Real-time preview optimization
- [ ] Extended model support (SAM2, etc)
- [ ] Cloud export integration
- [ ] Web dashboard

---

## 📜 License

MIT License - see LICENSE file for details

---

## 📧 Support

For issues or questions:
1. Check Troubleshooting section above
2. Review configuration examples in `config/`
3. Check camera firmware version
4. Review logs in `logs/` directory

---

## 🎓 References

- **RealSense SDK**: https://github.com/IntelRealSense/librealsense
- **MobileSAM**: https://github.com/ChaoningZhang/MobileSAM
- **COCO Format**: https://cocodataset.org/
- **YOLO Format**: https://docs.ultralytics.com/

---

**Last Updated**: April 2025  
**Tested On**: Python 3.12, OpenCV 4.13, PyTorch 2.5, pyrealsense2 2.57.7
