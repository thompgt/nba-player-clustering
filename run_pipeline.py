import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def run_command(command: list[str]) -> None:
    logger.info("Executing: %s", " ".join(command))
    result = subprocess.run(command, shell=False)
    if result.returncode != 0:
        logger.error("Command failed with return code %s", result.returncode)
        sys.exit(1)

def main() -> None:
    logger.info("Starting NBA Player Clustering Pipeline...")

    # 1. Preprocess
    run_command([sys.executable, "preprocess.py"])

    # 2. Validate
    run_command([sys.executable, "validate_model.py"])

    # 3. Test
    run_command([sys.executable, "-m", "pytest", "test_preprocess.py"])

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
