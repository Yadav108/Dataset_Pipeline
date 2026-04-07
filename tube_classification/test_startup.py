#!/usr/bin/env python
"""
Test script to verify pipeline startup without interactive mode.
Runs all verification gate checks to confirm system readiness.
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.verification_gate import run_verification_gate

if __name__ == "__main__":
    logger.info("=== Starting Verification Gate Test ===")
    try:
        run_verification_gate()
        logger.info("✅ All verification checks passed!")
        logger.info("Pipeline is ready to capture data.")
        sys.exit(0)
    except SystemExit as e:
        if e.code == 1:
            logger.error("❌ Verification gate failed. See errors above.")
            sys.exit(1)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
