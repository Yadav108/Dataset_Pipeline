import sys
import shutil
from pathlib import Path

from loguru import logger
import pyrealsense2 as rs
from pydantic import ValidationError
from config.parser import load_config, get_config, DEFAULT_CONFIG_PATH


def run_verification_gate(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Run 5 sequential startup checks before pipeline begins.
    
    Performs critical system checks and exits with status 1 if any fail.
    All systems must be ready before pipeline can proceed.
    
    Args:
        config_path: Path to configuration YAML file
    """
    
    # CHECK 1 — Config file exists
    logger.info("Verification gate: Checking config file existence...")
    if not config_path.exists():
        logger.error(f"Config file not found. Expected at: {config_path}")
        sys.exit(1)
    
    # CHECK 2 — Config schema valid
    logger.info("Verification gate: Validating config schema...")
    try:
        load_config(config_path)
    except ValidationError as e:
        logger.error(f"Config validation failed: {e}")
        sys.exit(1)
    
    # CHECK 3 — Output directory writable + disk space
    logger.info("Verification gate: Checking output directory writability and disk space...")
    cfg = get_config()
    root_dir = Path(cfg.storage.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # Test write permission
    try:
        test_file = root_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
    except Exception:
        logger.error(f"Output directory not writable: {root_dir}")
        sys.exit(1)
    
    # Check disk space
    disk_usage = shutil.disk_usage(root_dir)
    min_space = 1_073_741_824  # 1GB
    if disk_usage.free < min_space:
        free_gb = disk_usage.free // 1_073_741_824
        logger.error(
            f"Insufficient disk space: {free_gb}GB free at {root_dir}"
        )
        sys.exit(1)
    
    # CHECK 4 — RealSense camera connected
    logger.info("Verification gate: Checking RealSense camera...")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            logger.error(
                "No RealSense camera detected. Connect the D435if and retry."
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"RealSense error: {e}")
        sys.exit(1)
    
    # CHECK 5 — MobileSAM weights exist
    logger.info("Verification gate: Checking MobileSAM weights...")
    weights_path = Path(cfg.storage.sam_weights_path)
    if not weights_path.exists():
        logger.error(
            f"MobileSAM weights not found at {weights_path}. "
            f"Download from: https://github.com/ChaoningZhang/MobileSAM"
        )
        sys.exit(1)
    
    # All checks passed
    logger.info("Verification gate passed. All systems ready.")
