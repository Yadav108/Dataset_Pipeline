#!/usr/bin/env python3
"""
Archive setup utility - creates the TIER 2 archive structure and copies files.
This script creates the archive directories and manifest without external tool dependencies.
"""

import shutil
import os
from pathlib import Path
from datetime import datetime

def setup_archive():
    """Create archive structure and copy files."""
    
    project_dir = Path(r'C:\Users\Aryan\OneDrive\Desktop\Projects\Dataset_Pipeline\tube_classification')
    os.chdir(project_dir)
    
    print("=" * 70)
    print("TIER 2 CLEANUP: Archiving Obsolete Files")
    print("=" * 70)
    
    # Create archive directories
    print("\n1. Creating archive directory structure...")
    archive_dirs = [
        "archive/test_files",
        "archive/configs",
        "archive/docs",
        "archive/notebooks",
        "archive/experimental"
    ]
    
    for dir_path in archive_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path}/")
    
    # List of test files to archive
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
    
    # Archive test files
    print("\n2. Archiving test files from root:")
    archived = []
    skipped = []
    
    for test_file in test_files:
        source = Path(test_file)
        if source.exists():
            dest = Path("archive/test_files") / test_file
            shutil.copy2(source, dest)
            archived.append(test_file)
            print(f"   ✓ {test_file}")
        else:
            skipped.append(test_file)
            print(f"   ✗ {test_file} (not found)")
    
    # Archive config files
    print("\n3. Archiving config files:")
    config_path = Path("config")
    if config_path.exists():
        for config_file in config_path.glob("config_backup*.yaml"):
            dest = Path("archive/configs") / config_file.name
            shutil.copy2(config_file, dest)
            archived.append(str(config_file.relative_to(".")))
            print(f"   ✓ {config_file.name}")
        
        for config_file in config_path.glob("config_old.yaml"):
            dest = Path("archive/configs") / config_file.name
            shutil.copy2(config_file, dest)
            archived.append(str(config_file.relative_to(".")))
            print(f"   ✓ {config_file.name}")
    else:
        print("   ✗ config directory not found")
    
    # Archive old docs
    print("\n4. Archiving documentation:")
    doc_files = [
        ("README.old", "archive/docs/README.old"),
        ("CHANGELOG.old", "archive/docs/CHANGELOG.old"),
    ]
    
    for src, dest in doc_files:
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, dest)
            archived.append(src)
            print(f"   ✓ {src}")
        else:
            print(f"   ✗ {src} (not found)")
    
    # Create manifest
    print("\n5. Creating manifest...")
    manifest_content = f"""===== ARCHIVE CONTENTS =====
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

WHY ARCHIVED:
- These root-level test files were consolidated into the tests/ directory
- The main test infrastructure is now in tests/
- These older individual test scripts have been superseded by the integrated test suite
- Archiving them reduces clutter in the project root while preserving them for reference

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
        f.write(manifest_content)
    print("   ✓ archive/CONTENTS.txt")
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("TIER 2 CLEANUP SUMMARY")
    print("=" * 70)
    
    print(f"\nFiles archived: {len(archived)}")
    print(f"Files skipped: {len(skipped)}")
    print(f"Archive location: ./archive/")
    
    # Calculate archive size
    archive_files = list(Path("archive").rglob("*"))
    archive_files = [f for f in archive_files if f.is_file()]
    total_size = sum(f.stat().st_size for f in archive_files)
    
    size_kb = total_size / 1024
    size_mb = size_kb / 1024
    
    print(f"Total archive size: {size_kb:.1f} KB ({size_mb:.2f} MB)")
    
    # List by category
    print(f"\nArchive contents by category:")
    for subdir in ["test_files", "configs", "docs", "notebooks", "experimental"]:
        subdir_path = Path("archive") / subdir
        files = list(subdir_path.glob("*"))
        files = [f for f in files if f.is_file()]
        if files:
            print(f"\n  {subdir}/ ({len(files)} files)")
            for f in sorted(files):
                size = f.stat().st_size / 1024
                print(f"    - {f.name} ({size:.1f} KB)")
        else:
            print(f"\n  {subdir}/ (empty)")
    
    # Summary by type
    print("\n\nArchived files summary:")
    test_count = len([f for f in archived if 'test_' in f and f.endswith('.py')])
    config_count = len([f for f in archived if 'config' in f])
    doc_count = len([f for f in archived if f.endswith('.old')])
    
    print(f"  - Test files: {test_count}")
    print(f"  - Config files: {config_count}")
    print(f"  - Documentation: {doc_count}")
    
    # Create summary file
    print("\n6. Creating cleanup summary report...")
    summary_content = f"""TIER 2 CLEANUP SUMMARY
======================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Files archived: {len(archived)}
Files skipped: {len(skipped)}
Total archive size: {size_kb:.1f} KB ({size_mb:.2f} MB)

Archived by category:
  - Test files: {test_count}
  - Config files: {config_count}
  - Documentation: {doc_count}

ARCHIVED FILES:
"""
    
    for item in sorted(archived):
        summary_content += f"  - {item}\n"
    
    with open("archive/CLEANUP_SUMMARY.txt", "w") as f:
        f.write(summary_content)
    print("   ✓ archive/CLEANUP_SUMMARY.txt")
    
    print("\n" + "=" * 70)
    print("✓ ALL OPERATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return {
        "archived_count": len(archived),
        "skipped_count": len(skipped),
        "total_size_kb": size_kb,
        "total_size_mb": size_mb,
        "test_count": test_count,
        "config_count": config_count,
        "doc_count": doc_count,
    }

if __name__ == "__main__":
    try:
        result = setup_archive()
        print(f"\n✓ Archive setup completed successfully!")
        print(f"  - {result['archived_count']} files archived")
        print(f"  - Total size: {result['total_size_mb']:.2f} MB")
    except Exception as e:
        print(f"\n✗ Error during archive setup: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
