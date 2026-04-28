#!/usr/bin/env python3
"""Delete temporary audit files"""

import os
from pathlib import Path

root = Path(__file__).parent

files_to_delete = [
    "src/orchestrator_proxy.py",
    "src/export_proxy.py",
    "REORGANIZATION_NEEDED.txt",
    "README_REORGANIZATION.md",
    "STRUCTURE_AUDIT_REPORT.md",
    "AUDIT_SUMMARY.md",
    "AUDIT_CHECKLIST.md",
    "reorganize.py",
    "reorganize.bat",
]

print("Deleting temporary files...")
for file in files_to_delete:
    path = root / file
    if path.exists():
        path.unlink()
        print(f"✓ Deleted: {file}")
    else:
        print(f"✗ Not found: {file}")

print("\n✓ Cleanup complete!")
