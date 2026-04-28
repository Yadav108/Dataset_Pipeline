#!/usr/bin/env python3
"""TIER 2 Cleanup - Archive obsolete files (standalone execution)."""
import shutil
from pathlib import Path
import os
from datetime import datetime

# Change to project directory
project_dir = r'C:\Users\Aryan\OneDrive\Desktop\Projects\Dataset_Pipeline\tube_classification'
os.chdir(project_dir)

# Create archive directories
print("Creating archive directory structure...")
Path("archive/test_files").mkdir(parents=True, exist_ok=True)
Path("archive/configs").mkdir(parents=True, exist_ok=True)
Path("archive/docs").mkdir(parents=True, exist_ok=True)
Path("archive/notebooks").mkdir(parents=True, exist_ok=True)
Path("archive/experimental").mkdir(parents=True, exist_ok=True)

# List of test files to archive (from root level)
test_files = [
    "test_verification.py",
    "test_startup.py",
    "test_roi_import.py",
    "test_preprocessing.py",
    "test_pipeline_ready.py",
    "test_pipeline_integration.py",
    "test_guided_filter_quick.py",
    "test_guided_filter.py",
    "test_diagnostics.py",
    "test_config_guided_filter.py",
    "test_complete_system.py",
    "test_camera_startup.py",
    "test_annotation_quality.py",
    "test_advanced_preprocessing.py",
]

print("\n" + "=" * 70)
print("TIER 2 CLEANUP: Archiving Obsolete Files")
print("=" * 70)

archived = []
skipped = []

# Archive test files
print("\nArchiving test files from root:")
for test_file in test_files:
    source = Path(test_file)
    if source.exists():
        dest = Path("archive/test_files") / test_file
        shutil.copy2(source, dest)
        archived.append(test_file)
        print(f"  [OK] {test_file}")
    else:
        skipped.append(test_file)
        print(f"  [SKIP] {test_file} (not found)")

# Archive config files if they exist
print("\nArchiving config files:")
config_path = Path("config")
if config_path.exists():
    for config_file in config_path.glob("config_backup*.yaml"):
        dest = Path("archive/configs") / config_file.name
        shutil.copy2(config_file, dest)
        archived.append(str(config_file.relative_to(".")))
        print(f"  [OK] {config_file.name}")
    
    for config_file in config_path.glob("config_old.yaml"):
        dest = Path("archive/configs") / config_file.name
        shutil.copy2(config_file, dest)
        archived.append(str(config_file.relative_to(".")))
        print(f"  [OK] {config_file.name}")
else:
    print("  [SKIP] config directory not found")

# Archive old docs if they exist
print("\nArchiving documentation:")
doc_files = [
    ("README.old", "archive/docs/README.old"),
    ("CHANGELOG.old", "archive/docs/CHANGELOG.old"),
]

for src, dest in doc_files:
    src_path = Path(src)
    if src_path.exists():
        shutil.copy2(src_path, dest)
        archived.append(src)
        print(f"  [OK] {src}")
    else:
        print(f"  [SKIP] {src} (not found)")

# Create manifest
manifest = f"""===== ARCHIVE CONTENTS =====
Date archived: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reason: TIER 2 Codebase cleanup - obsolete and experimental files

ARCHIVED FILES:

test_files/ (14 files):
  - test_verification.py - Pipeline verification script
  - test_startup.py - Pipeline startup verification
  - test_roi_import.py - ROI import testing
  - test_preprocessing.py - Preprocessing testing  
  - test_pipeline_ready.py - Pipeline readiness check
  - test_pipeline_integration.py - Pipeline integration test
  - test_guided_filter_quick.py - Guided filter quick test
  - test_guided_filter.py - Guided filter testing
  - test_diagnostics.py - Diagnostics testing
  - test_config_guided_filter.py - Config guided filter test
  - test_complete_system.py - Complete system test
  - test_camera_startup.py - Camera startup test
  - test_annotation_quality.py - Annotation quality test
  - test_advanced_preprocessing.py - Advanced preprocessing test

NOTE: These test files have been consolidated into the tests/ directory.
The main test suite is now in: tests/

ACTIVE TEST DIRECTORY: ./tests/
  - test_capture_workflow.py
  - test_confirmation_preview.py
  - test_live_preview.py
  - test_parser.py

TO RESTORE ANY FILE:
  On Windows (PowerShell):
    Copy-Item -Path archive/[subdir]/[filename] -Destination [original location]
  
  On Unix/Linux/Mac:
    cp archive/[subdir]/[filename] [original location]

EXAMPLES:
  Copy-Item archive/test_files/test_verification.py -Destination .
  Copy-Item archive/configs/config_old.yaml -Destination config/
"""

with open("archive/CONTENTS.txt", "w") as f:
    f.write(manifest)

print(f"\n[OK] Created manifest at archive/CONTENTS.txt")

# Summary
print("\n" + "=" * 70)
print("TIER 2 CLEANUP SUMMARY")
print("=" * 70)
print(f"\nFiles archived: {len(archived)}")
print(f"Files skipped: {len(skipped)}")
print(f"Archive location: ./archive/")

# List all files in archive
archive_files = list(Path("archive").rglob("*"))
archive_files = [f for f in archive_files if f.is_file()]
total_size = sum(f.stat().st_size for f in archive_files)

print(f"Total archive size: {total_size / 1024:.1f} KB ({total_size / 1024 / 1024:.2f} MB)")

print(f"\nArchive contents by category:")
for subdir in ["test_files", "configs", "docs", "notebooks", "experimental"]:
    subdir_path = Path("archive") / subdir
    files = list(subdir_path.glob("*"))
    files = [f for f in files if f.is_file()]
    if files:
        print(f"\n  {subdir}/ ({len(files)} files)")
        for f in files:
            size = f.stat().st_size / 1024
            print(f"    - {f.name} ({size:.1f} KB)")
    else:
        print(f"\n  {subdir}/ (empty)")

print("\nArchived files summary:")
test_count = len([f for f in archived if 'test_' in f and f.endswith('.py')])
config_count = len([f for f in archived if 'config' in f])
doc_count = len([f for f in archived if f.endswith('.old')])

print(f"  - Test files: {test_count}")
print(f"  - Config files: {config_count}")
print(f"  - Documentation: {doc_count}")

print("\n" + "=" * 70)
print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
print("=" * 70)

# Save summary to file
summary_file = "archive/CLEANUP_SUMMARY.txt"
with open(summary_file, "w") as f:
    f.write(f"TIER 2 CLEANUP SUMMARY\n")
    f.write(f"======================\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Files archived: {len(archived)}\n")
    f.write(f"Files skipped: {len(skipped)}\n")
    f.write(f"Total archive size: {total_size / 1024:.1f} KB ({total_size / 1024 / 1024:.2f} MB)\n\n")
    f.write(f"Archived by category:\n")
    f.write(f"  - Test files: {test_count}\n")
    f.write(f"  - Config files: {config_count}\n")
    f.write(f"  - Documentation: {doc_count}\n\n")
    f.write(f"ARCHIVED FILES:\n")
    for item in sorted(archived):
        f.write(f"  - {item}\n")

print(f"\n✓ Summary saved to {summary_file}")
