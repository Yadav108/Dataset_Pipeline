from loguru import logger
from src.orchestrator.verification_gate import run_verification_gate
from src.orchestrator.pipeline import run_pipeline

if __name__ == "__main__":
    logger.info("=== Tube Classification Pipeline Starting ===")
    run_verification_gate()
    run_pipeline()
