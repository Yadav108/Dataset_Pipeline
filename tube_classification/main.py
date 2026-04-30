import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger
from src.orchestrator.verification_gate import run_verification_gate
from src.orchestrator.pipeline import run_pipeline
from pre_capture import run_pre_capture_workflow


def _load_base_session_dir(config_path: str = "config/config.yaml") -> Path:
    """Load session base directory from config, with safe fallback."""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return Path(
            config.get("pipeline", {})
            .get("session", {})
            .get("base_directory", "dataset/sessions")
        )
    except Exception:
        return Path("dataset/sessions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tube classification pipeline entrypoint")
    parser.add_argument(
        "--skip-pre-capture",
        action="store_true",
        help="Skip pre-capture data entry and go directly to pipeline",
    )
    args = parser.parse_args()

    logger.info("=== Tube Classification Pipeline Starting ===")
    pre_capture_context = None
    if not args.skip_pre_capture:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        pre_capture_context = run_pre_capture_workflow(
            session_id=session_id,
            base_dir=_load_base_session_dir(),
        )
    run_verification_gate()
    if pre_capture_context:
        run_pipeline(
            preselected_volume_ml=float(pre_capture_context["volume_ml"]),
            preselected_class_id=str(pre_capture_context["class_id"]),
            preselected_capture_mode=str(pre_capture_context["capture_mode"]),
        )
    else:
        run_pipeline()
