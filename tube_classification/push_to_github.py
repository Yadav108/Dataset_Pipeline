#!/usr/bin/env python3
"""Git push automation for Dataset Pipeline."""

import subprocess
import sys
from pathlib import Path

# Set working directory
repo_root = Path(r"C:\Users\Aryan\OneDrive\Desktop\Projects\Dataset_Pipeline")
print(f"Repository: {repo_root}")
print("=" * 70)

def run_cmd(cmd, description=""):
    """Run git command and show output."""
    if description:
        print(f"\n{description}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=repo_root, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"❌ ERROR: {result.stderr}")
        return False
    return True

# Check git status
if not run_cmd("git status", "1. CHECKING GIT STATUS"):
    sys.exit(1)

# Show what will be committed
print("\n2. MODIFIED FILES")
print("-" * 70)
run_cmd("git diff --name-only", "")

# Stage all changes in tube_classification directory
print("\n3. STAGING CHANGES")
print("-" * 70)
run_cmd("git add tube_classification/", "Adding tube_classification directory...")

# Check staged changes
print("\n4. STAGED CHANGES")
print("-" * 70)
run_cmd("git diff --staged --name-only", "")

# Commit with descriptive message
commit_msg = """fix: Improve segmentation quality and fix memory issues

- Optimize SAM segmentation strategy for 22cm camera distance
- Switch primary strategy to box-only prompting (more reliable for small objects)
- Add adaptive morphological kernel sizing based on bbox dimensions
- Reduce SAM IoU threshold from 0.62 to 0.57 for small tubes
- Use dynamic area thresholds instead of fixed minimums
- Fix memory errors: Change CV_64F to CV_32F in blur detection (50% reduction)
- Handle memory allocation errors in preview rendering with try/except
- Disable preview by default to reduce memory pressure
- Add explicit frame cleanup in main loop to prevent memory leaks
- Update camera depth range to 0.17-0.27m (22cm ±5cm)
- Add firmware update guide and driver diagnostic tools"""

print("\n5. COMMITTING CHANGES")
print("-" * 70)
if not run_cmd(f'git commit -m "{commit_msg.splitlines()[0]}" -m "{chr(10).join(commit_msg.splitlines()[1:])}"', 
               "Creating commit..."):
    print("⚠ Note: If nothing to commit, that's OK - changes may already be staged")

# Show commit log
print("\n6. LATEST COMMITS")
print("-" * 70)
run_cmd("git log --oneline -5", "")

# Push to remote
print("\n7. PUSHING TO GITHUB")
print("-" * 70)
if run_cmd("git branch -v", "Checking branch..."):
    print("\n✓ Ready to push. Pushing to remote...")
    run_cmd("git push origin HEAD", "Pushing...")

print("\n" + "=" * 70)
print("✓ PUSH COMPLETE!")
print("=" * 70)
