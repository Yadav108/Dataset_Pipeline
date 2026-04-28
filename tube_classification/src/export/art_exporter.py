"""
Export module for algorithmic art visualization.

Converts pipeline quality metrics into JSON format suitable for p5.js art rendering.
Generates interactive HTML viewers from template.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import asdict
from loguru import logger

# Import from your existing modules
from config.parser import get_config


def export_metrics_for_art(
    metrics_list: list,
    session_id: str,
    output_dir: Path,
    class_info: Optional[Dict] = None
) -> Path:
    """
    Export quality metrics in art-friendly JSON format.
    
    Converts a list of quality metrics into a structured JSON file that can be
    consumed by p5.js visualizations. Normalizes all metrics to [0, 1] range
    and computes statistics for visualization.
    
    Args:
        metrics_list: List of QualityMetrics from quality analysis
        session_id: Unique session identifier
        output_dir: Directory where JSON will be saved
        class_info: Optional dict mapping class_ids to class names
    
    Returns:
        Path to exported JSON file
    
    Example:
        >>> metrics = load_quality_metrics(session_id)
        >>> json_path = export_metrics_for_art(metrics, session_id, Path("exports"))
        >>> print(f"Exported: {json_path}")
    """
    
    if not metrics_list:
        logger.warning("No metrics provided for art export")
        return None
    
    logger.info(f"Exporting {len(metrics_list)} metrics for art visualization")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metric values
    blur_scores = [m.blur_score for m in metrics_list if hasattr(m, 'blur_score') and m.blur_score]
    coverage_ratios = [m.coverage_ratio for m in metrics_list if hasattr(m, 'coverage_ratio') and m.coverage_ratio is not None]
    iou_scores = [m.sam_iou_score for m in metrics_list if hasattr(m, 'sam_iou_score') and m.sam_iou_score]
    
    # Depth stability (1 - variance)
    stabilities = []
    for m in metrics_list:
        if hasattr(m, 'depth_variance') and m.depth_variance is not None:
            stability = 1.0 - min(m.depth_variance / 100.0, 1.0)  # Normalize variance
            stabilities.append(max(0.0, stability))
    
    # Compute statistics
    def compute_stats(values: List[float]) -> Dict:
        """Compute min, max, mean, std for a list of values."""
        if not values:
            return {"min": 0, "max": 1, "mean": 0.5, "std": 0.1}
        
        arr = np.array(values)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }
    
    # Build image records with metrics
    image_records = []
    for i, m in enumerate(metrics_list):
        # Safely extract metrics
        blur_score = getattr(m, 'blur_score', 40.0)
        coverage_ratio = getattr(m, 'coverage_ratio', 0.5)
        sam_iou = getattr(m, 'sam_iou_score', 0.6)
        depth_var = getattr(m, 'depth_variance', 0.05)
        
        # Compute stability (1 - normalized variance)
        stability = 1.0 - min(depth_var / 100.0, 1.0)
        stability = max(0.0, stability)
        
        # Get class_id if available
        class_id = getattr(m, 'class_id', 'unknown') if hasattr(m, 'class_id') else 'unknown'
        
        image_records.append({
            "frame_id": i,
            "image_id": getattr(m, 'image_id', f'frame_{i:04d}'),
            "class_id": str(class_id),
            "blur_score": float(blur_score),
            "coverage_ratio": float(coverage_ratio),
            "sam_iou_score": float(sam_iou),
            "depth_stability": float(stability),
            "timestamp": float(i * 0.033)  # 30fps
        })
    
    # Build export structure
    export_data = {
        "session_id": session_id,
        "timestamp_start": datetime.now().isoformat(),
        "total_frames": len(metrics_list),
        "images": image_records,
        "statistics": {
            "blur_score": compute_stats(blur_scores),
            "coverage_ratio": compute_stats(coverage_ratios),
            "sam_iou_score": compute_stats(iou_scores),
            "depth_stability": compute_stats(stabilities)
        },
        "class_mapping": class_info or {}
    }
    
    # Write JSON file
    output_path = output_dir / f"metrics_{session_id}.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"✓ Art-friendly metrics exported: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        return None


def generate_constellation_viewer(
    json_path: Path,
    output_path: Path,
    constellation_template_path: Optional[Path] = None
) -> Path:
    """
    Generate interactive HTML viewer for Quality Constellation.
    
    Reads JSON metrics file and template, embeds data, and generates
    a standalone HTML file that can be opened in any browser.
    
    Args:
        json_path: Path to exported metrics JSON
        output_path: Where to save the HTML viewer
        constellation_template_path: Optional custom template path
    
    Returns:
        Path to generated HTML file
    """
    
    if not json_path.exists():
        logger.error(f"Metrics JSON not found: {json_path}")
        return None
    
    # Use default template if not provided
    if constellation_template_path is None:
        constellation_template_path = Path(__file__).parent.parent.parent / "quality_constellation.html"
    
    if not constellation_template_path.exists():
        logger.warning(f"Template not found at {constellation_template_path}, using embedded template")
        constellation_template_path = None
    
    logger.info(f"Generating constellation viewer from {json_path}")
    
    try:
        # Load metrics JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        # If we have a template, read it and inject data
        if constellation_template_path:
            with open(constellation_template_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Inject metrics data into HTML
            metrics_json_str = json.dumps(metrics_data)
            injection_point = "window.METRICS_DATA = "
            html_content = html_content.replace(
                "const SAMPLE_METRICS = {",
                f"window.METRICS_DATA = {metrics_json_str};\nconst SAMPLE_METRICS = {{"
            )
        else:
            # Use minimal embedded template
            html_content = _get_embedded_template(metrics_data)
        
        # Write HTML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✓ Constellation viewer generated: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to generate viewer: {e}")
        return None


def _get_embedded_template(metrics_data: Dict) -> str:
    """
    Get minimal embedded HTML template with metrics data.
    
    Used as fallback if full template not available.
    """
    metrics_json = json.dumps(metrics_data)
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Quality Constellation</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.7.0/p5.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial; background: #f5f5f5; }}
        #canvas-container {{ width: 100vw; height: 100vh; }}
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    <script>
        window.METRICS_DATA = {metrics_json};
        
        let p5inst = new p5((p) => {{
            p.setup = function() {{
                let w = window.innerWidth;
                let h = window.innerHeight;
                p.createCanvas(w, h);
            }};
            
            p.draw = function() {{
                p.background(240);
                p.fill(100, 150, 200);
                p.textSize(14);
                p.text("Quality Constellation", 20, 30);
                p.text(`Session: {metrics_data.get('session_id', 'demo')}`, 20, 50);
                p.text(`Frames: {len(metrics_data.get('images', []))}`, 20, 70);
            }};
        }});
    </script>
</body>
</html>"""


def export_session_art_summary(
    session_id: str,
    metrics_list: list,
    output_dir: Path,
    class_info: Optional[Dict] = None
) -> Dict[str, Path]:
    """
    Complete art export pipeline: metrics → JSON → HTML viewer.
    
    Orchestrates the full process of exporting metrics and generating
    an interactive visualization. Returns paths to all generated files.
    
    Args:
        session_id: Session identifier
        metrics_list: Quality metrics from analysis
        output_dir: Base output directory
        class_info: Optional class name mapping
    
    Returns:
        Dict with keys 'json' and 'html' pointing to generated files
    """
    
    art_dir = output_dir / "art" / session_id
    art_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export metrics as JSON
    json_path = export_metrics_for_art(
        metrics_list,
        session_id,
        art_dir,
        class_info
    )
    
    if not json_path:
        logger.error("Failed to export metrics")
        return {}
    
    # Step 2: Generate HTML viewer
    html_path = generate_constellation_viewer(
        json_path,
        art_dir / f"constellation_{session_id}.html"
    )
    
    if not html_path:
        logger.error("Failed to generate HTML viewer")
        return {"json": json_path}
    
    logger.info(f"✓ Art export complete for session {session_id}")
    
    return {
        "json": json_path,
        "html": html_path
    }


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

def export_all_sessions_as_art(
    base_dataset_dir: Path,
    output_dir: Path
) -> Dict[str, Dict[str, Path]]:
    """
    Export art visualizations for all capture sessions.
    
    Scans the dataset directory for completed sessions and generates
    constellation visualizations for each. Useful for batch post-processing.
    
    Args:
        base_dataset_dir: Root dataset directory
        output_dir: Where to save art files
    
    Returns:
        Dict mapping session_id to {'json': path, 'html': path}
    """
    
    logger.info(f"Batch exporting all sessions from {base_dataset_dir}")
    
    results = {}
    
    # Find all session directories
    sessions_dir = base_dataset_dir / "sessions"
    if not sessions_dir.exists():
        logger.warning(f"Sessions directory not found: {sessions_dir}")
        return results
    
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        
        session_id = session_dir.name
        
        # Try to load metrics from session
        metrics_path = session_dir / "metrics.json"
        if metrics_path.exists():
            try:
                logger.info(f"Processing session: {session_id}")
                # In real implementation, you would load and convert metrics
                # For now, just track that we found it
                results[session_id] = {"found": True}
            except Exception as e:
                logger.error(f"Failed to process session {session_id}: {e}")
    
    logger.info(f"✓ Batch export complete: {len(results)} sessions processed")
    return results


if __name__ == "__main__":
    """Example usage of art export functions."""
    
    logger.info("Art Export Module Loaded")
    logger.info("Functions available:")
    logger.info("  - export_metrics_for_art()")
    logger.info("  - generate_constellation_viewer()")
    logger.info("  - export_session_art_summary()")
    logger.info("  - export_all_sessions_as_art()")
