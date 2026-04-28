import sys
import yaml
from pathlib import Path

from loguru import logger
from config.parser import get_config, AppConfig


def load_registry(registry_path: Path) -> list[dict]:
    """Load tube registry from YAML file.
    
    Args:
        registry_path: Path to registry YAML file
        
    Returns:
        List of tube dictionaries
        
    Raises:
        FileNotFoundError: If registry file does not exist
    """
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found at {registry_path}")
    
    with open(registry_path, "r") as f:
        data = yaml.safe_load(f)
    
    return data["tubes"]


def get_tubes_by_volume(tubes: list[dict], volume_ml: float) -> list[dict]:
    """Filter tubes by volume.
    
    Args:
        tubes: List of tube dictionaries
        volume_ml: Volume in milliliters to filter by
        
    Returns:
        List of tubes matching the specified volume
    """
    return [tube for tube in tubes if tube["volume_ml"] == volume_ml]


def run_volume_gate() -> tuple[float, list[dict]]:
    """Interactive CLI for volume declaration.
    
    Prompts user to declare tube volume, queries registry, and applies
    pipeline config overrides for the session. Supports custom volumes not in registry.
    
    Returns:
        Tuple of (volume_ml, matched_tubes)
    """
    config = get_config()
    registry_path = Path(config.storage.registry_path)
    tubes = load_registry(registry_path)
    
    # Get unique volumes from registry for validation
    unique_volumes = sorted(set(tube["volume_ml"] for tube in tubes))
    
    while True:
        try:
            volume_input = input("Enter tube volume in ml (e.g. 1.3, 3.5, 4.0, 4.5): ")
            volume_ml = float(volume_input)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        matched_tubes = get_tubes_by_volume(tubes, volume_ml)
        
        if not matched_tubes:
            # If no tubes found in registry, create a dynamic custom entry
            print(f"\nNo predefined tubes for {volume_ml}ml in registry.")
            print(f"Available volumes: {unique_volumes}")
            
            confirm = input(f"Create custom tube class for {volume_ml}ml? (y/n): ").lower()
            if confirm == 'y':
                # Create a dynamic tube entry for this volume
                custom_tube = {
                    "class_id": f"CUSTOM_{volume_ml}ml",
                    "family": "CUSTOM",
                    "volume_ml": volume_ml,
                    "diameter_mm": 13.0,
                    "cap_color": "unknown",
                    "cap_texture": "unknown",
                    "top_down_pattern": "unknown",
                    "ambiguous_with": []
                }
                matched_tubes = [custom_tube]
                logger.info(f"Volume declared: {volume_ml}ml — Created custom tube class")
                print(f"  {custom_tube['class_id']} ({custom_tube['family']})")
            else:
                print("Please try another volume.\n")
                continue
        else:
            # Volume found and matched
            logger.info(f"Volume declared: {volume_ml}ml — {len(matched_tubes)} tube class(es) matched")
            
            for tube in matched_tubes:
                print(f"  {tube['class_id']} ({tube['family']})")
        
        # Apply config overrides for smaller tubes
        if volume_ml <= 1.5:
            config.pipeline.min_roi_area_px = 200
            config.pipeline.stability_frames = 15
        
        return (volume_ml, matched_tubes)
