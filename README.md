# Tube Classification Dataset Pipeline
 
 ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
 ![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
 ![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)
 ![License](https://img.shields.io/badge/License-MIT-yellow)
 
 **Automated dataset collection, annotation, cleaning, and export pipeline for tube classification using an Intel RealSense D435i and MobileSAM.**
 
 ## Overview
 
 This project provides an end-to-end pipeline for:
 
 - **Capture**: RealSense RGB + depth acquisition with depth stability checks
 - **Pre-capture data entry**: SQLite-backed class/volume/session metadata
 - **ROI extraction**: Depth-based tube localization
 - **Segmentation**: MobileSAM mask generation
 - **Cleaning**: Blur filtering, duplicate detection, bbox quality checks, background removal
 - **Export**: COCO and YOLO dataset export plus session manifests and stats
 
 ## Key Features
 
 - RealSense D435i support
 - Side and top-down capture workflows
 - Verification gate for config, disk, camera, and MobileSAM weights
 - Session-based capture with recovery support
 - Quality thresholds for blur, mask coverage, IoU, and duplicates
 - Guided filtering and depth-mask refinement
 - Optional background removal
 - Dataset exports in COCO and YOLO formats
 
 ---
 
 ## Quick Start
 
 ### Prerequisites
 
 - Intel RealSense D435i
 - Python 3.10 to 3.12
 - Windows 10/11
 - USB 3.0 connection
 - Enough disk space for your dataset
 
 ### Installation
 
 ```bash
 git clone https://github.com/Yadav108/Dataset_Pipeline.git
 cd tube_classification
 
 python -m venv venv
 .\venv\Scripts\Activate.ps1
 
 pip install -r requirements.txt

MobileSAM Weights

Download MobileSAM weights and place them here:

 models/MobileSAM/weights/mobile_sam.pt

Run the Pipeline

 python capture.py

This will:

 1. Run the verification gate
 2. Prompt for pre-capture class/volume data
 3. Start the capture pipeline
 4. Segment and clean captures
 5. Save session outputs and exports

To skip the pre-capture metadata step:

 python capture.py --skip-pre-capture

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Usage

Main Capture Flow

 python capture.py

Legacy / Interactive Entry Point

 python main.py

Quality Analysis

 python analyze_image_quality.py

System Checks

 python verify_system.py
 python verify_hd_capture.py
 python verify_mobilesam.py

Dataset Export

 python -m src.export.coco_exporter
 python -m src.export.yolo_exporter

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Configuration

Edit config/config.yaml before running.

Current Defaults

 camera:
   width: 848
   height: 480
   fps: 30
   color_width: 1920
   color_height: 1080
   color_fps: 30
   depth_min_m: 0.28
   depth_max_m: 0.42
 
 pipeline:
   stability_frames: 4
   blur_threshold: 52.0
   duplicate_hash_threshold: 12
   min_coverage_ratio: 0.48
   sam_iou_threshold: 0.57
   show_preview: false

Tuning Notes

 - Increase blur_threshold to reject more blurry images
 - Increase min_coverage_ratio for stricter mask quality
 - Increase sam_iou_threshold for stricter segmentation acceptance
 - Adjust depth_min_m / depth_max_m for your working distance

Guided Filter Defaults

 preprocessing:
   guided_filter:
     enabled: true
     radius: 8
     max_processing_time_ms: 500.0

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Project Structure

 tube_classification/
 ├── main.py
 ├── capture.py
 ├── pre_capture.py
 ├── config/
 │   ├── config.yaml
 │   └── calibration.yaml
 ├── src/
 │   ├── acquisition/
 │   ├── annotation/
 │   ├── cleaning/
 │   ├── export/
 │   ├── orchestrator/
 │   ├── preview/
 │   ├── processing/
 │   └── reconstruction/
 ├── dataset/
 ├── data/
 ├── docs/
 │   ├── guides/
 │   ├── troubleshooting/
 │   └── archive/
 ├── models/
 ├── tests/
 └── README.md

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Output Files

Typical session outputs include:

 - RGB images
 - Depth frames (.npz)
 - Segmentation masks
 - Annotation JSON
 - Session logs
 - COCO / YOLO exports
 - Quality reports

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Troubleshooting

Camera Not Detected

 - Use a USB
  3.0 port
 - Reconnect the RealSense D435i
 - Update firmware with RealSense Viewer
 - Verify pyrealsense2 is installed correctly

Low Capture Yield

 - Increase lighting consistency
 - Tune blur and IoU thresholds
 - Check camera distance
 - Confirm depth range in config.yaml

Poor Segmentation

 - Verify MobileSAM weights path
 - Adjust ROI and depth range
 - Tune sam_iou_threshold
 - Check focus and scene background

Memory Issues

 - Keep show_preview: false
 - Close other camera applications
 - Reduce capture load
 - Confirm enough virtual memory

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Performance Targets

Typical session behavior:

 - Stable depth gating before capture
 - Fast ROI extraction and segmentation
 - Quality-based rejection of bad samples
 - Session-level logs and recovery metadata

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Documentation

 - README.md — project overview
 - QUICK_START.md — quick reference
 - docs/README.md — documentation map
 - docs/guides/ — setup and optimization guides
 - docs/troubleshooting/ — issue fixes

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Contributing

Contributions are welcome.

Areas that are still open for improvement:

 - Multi-camera support
 - Better preview performance
 - Additional model support
 - Cloud export integration
 - Web dashboard

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

License

MIT License

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

References

 - RealSense SDK: https://github.com/IntelRealSense/librealsense (https://github.com/IntelRealSense/librealsense)
 - MobileSAM: https://github.com/ChaoningZhang/MobileSAM (https://github.com/ChaoningZhang/MobileSAM)
 - COCO: https://cocodataset.org/ (https://cocodataset.org/)
 - YOLO: https://docs.ultralytics.com/ (https://docs.ultralytics.com/)
