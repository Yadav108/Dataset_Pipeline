"""Dataset balance and statistics reporting."""

import yaml
from pathlib import Path
from loguru import logger
from config.parser import get_config


def generate_balance_report() -> dict:
    """Generate dataset balance report for all 16 classes.
    
    Scans dataset/raw/{class_id}/ for RGB images and counts captures
    per class. Loads class names from registry.yaml.
    
    Returns:
        Dictionary mapping class_id to stats:
        {
            class_id: {
                "name": str,           # Display name (e.g., "Vacuette Purple")
                "count": int,          # Number of images captured
                "target": int,         # Target images for this class
                "pct": float,          # Percentage of target (0-100)
            },
            ...
        }
    """
    cfg = get_config()
    root = Path(cfg.storage.root_dir)
    raw_root = root / "raw"
    
    # Load registry to get class names
    registry_path = Path("config") / "registry.yaml"
    registry = {}
    try:
        with open(registry_path, "r") as f:
            data = yaml.safe_load(f)
            if data and "tubes" in data:
                for tube in data["tubes"]:
                    class_id = tube["class_id"]
                    # Build human-readable name from family and cap color
                    family = tube.get("family", "").replace("_", " ").title()
                    color = tube.get("cap_color", "").replace("_", " ").title()
                    volume = tube.get("volume_ml", "")
                    name = f"{family} {color} ({volume}ml)"
                    registry[class_id] = name.strip()
    except Exception as e:
        logger.warning(f"Failed to load registry: {e}")
        registry = {}
    
    target = cfg.pipeline.target_images_per_class  # Default 500
    
    # Scan all class folders
    report = {}
    if raw_root.exists():
        for class_folder in raw_root.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_id = class_folder.name
            
            # Count RGB images in all session subfolders
            count = 0
            for session_folder in class_folder.iterdir():
                if session_folder.is_dir():
                    rgb_files = list(session_folder.glob("*_rgb.png"))
                    count += len(rgb_files)
            
            pct = round((count / target * 100) if target > 0 else 0, 1)
            
            report[class_id] = {
                "name": registry.get(class_id, class_id),
                "count": count,
                "target": target,
                "pct": pct,
            }
    
    return report


def print_balance_report(report: dict, title: str = "DATASET BALANCE REPORT") -> None:
    """Print formatted balance report table to console.
    
    Args:
        report: Dictionary from generate_balance_report()
        title: Report title to display
    """
    # Sort by count descending
    sorted_classes = sorted(
        report.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    # Calculate totals
    total_count = sum(r["count"] for r in report.values())
    total_target = sum(r["target"] for r in report.values())
    overall_pct = round((total_count / total_target * 100) if total_target > 0 else 0, 1)
    
    # Print header
    logger.info(f"\n{'='*100}")
    logger.info(f"{title:^100}")
    logger.info(f"{'='*100}")
    
    # Print column headers
    header = f"{'CLASS ID':<20} | {'NAME':<30} | {'COUNT':>8} | {'TARGET':>8} | {'%':>6} | STATUS"
    logger.info(header)
    logger.info("-" * 100)
    
    # Print rows
    for class_id, stats in sorted_classes:
        count = stats["count"]
        target = stats["target"]
        pct = stats["pct"]
        name = stats["name"][:30]  # Truncate to fit
        
        # Determine status indicator
        if pct >= 100:
            status = "✅ COMPLETE"
        elif pct >= 75:
            status = "🟩 75-99%"
        elif pct >= 50:
            status = "🟨 50-74%"
        else:
            status = "🟥 <50% ⚠️"
        
        row = f"{class_id:<20} | {name:<30} | {count:>8} | {target:>8} | {pct:>5.1f}% | {status}"
        logger.info(row)
    
    # Print footer with totals
    logger.info("-" * 100)
    footer = f"{'TOTAL':<20} | {'':<30} | {total_count:>8} | {total_target:>8} | {overall_pct:>5.1f}%"
    logger.info(footer)
    logger.info(f"{'='*100}\n")


def print_balance_report_with_delta(
    current_report: dict,
    previous_report: dict | None = None,
    title: str = "SESSION SUMMARY — DATASET BALANCE"
) -> None:
    """Print balance report with delta showing images added this session.
    
    Args:
        current_report: Dictionary from generate_balance_report()
        previous_report: Previous report dict (if None, shows no delta)
        title: Report title to display
    """
    # Sort by count descending
    sorted_classes = sorted(
        current_report.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    # Calculate totals
    total_count = sum(r["count"] for r in current_report.values())
    total_target = sum(r["target"] for r in current_report.values())
    overall_pct = round(
        (total_count / total_target * 100) if total_target > 0 else 0, 1
    )
    
    # Print header
    logger.info(f"\n{'='*115}")
    logger.info(f"{title:^115}")
    logger.info(f"{'='*115}")
    
    # Print column headers
    if previous_report:
        header = f"{'CLASS ID':<18} | {'NAME':<28} | {'COUNT':>8} | {'+':>4} | {'TARGET':>8} | {'%':>6} | STATUS"
    else:
        header = f"{'CLASS ID':<18} | {'NAME':<28} | {'COUNT':>8} | {'TARGET':>8} | {'%':>6} | STATUS"
    logger.info(header)
    logger.info("-" * 115)
    
    # Print rows
    for class_id, stats in sorted_classes:
        count = stats["count"]
        target = stats["target"]
        pct = stats["pct"]
        name = stats["name"][:28]  # Truncate to fit
        
        # Calculate delta if previous report available
        delta_str = ""
        if previous_report and class_id in previous_report:
            previous_count = previous_report[class_id]["count"]
            delta = count - previous_count
            delta_str = f" | {delta:>4}"
        
        # Determine status indicator
        if pct >= 100:
            status = "✅ COMPLETE"
        elif pct >= 75:
            status = "🟩 75-99%"
        elif pct >= 50:
            status = "🟨 50-74%"
        else:
            status = "🟥 <50% ⚠️"
        
        if delta_str:
            row = f"{class_id:<18} | {name:<28} | {count:>8}{delta_str} | {target:>8} | {pct:>5.1f}% | {status}"
        else:
            row = f"{class_id:<18} | {name:<28} | {count:>8} | {target:>8} | {pct:>5.1f}% | {status}"
        logger.info(row)
    
    # Print footer with totals
    logger.info("-" * 115)
    if previous_report:
        previous_total = sum(r["count"] for r in previous_report.values())
        delta_total = total_count - previous_total
        footer = f"{'TOTAL':<18} | {'':<28} | {total_count:>8} | {delta_total:>4} | {total_target:>8} | {overall_pct:>5.1f}%"
    else:
        footer = f"{'TOTAL':<18} | {'':<28} | {total_count:>8} | {total_target:>8} | {overall_pct:>5.1f}%"
    logger.info(footer)
    logger.info(f"{'='*115}\n")
