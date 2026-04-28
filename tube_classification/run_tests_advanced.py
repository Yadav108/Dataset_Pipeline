#!/usr/bin/env python
"""Test runner for PROMPT 5-7 advanced preprocessing"""

import subprocess
import sys

# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", 
     "test_advanced_preprocessing.py", 
     "-v", "--tb=short"],
    cwd="C:\\Users\\Aryan\\OneDrive\\Desktop\\Projects\\Dataset_Pipeline\\tube_classification"
)

sys.exit(result.returncode)
