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

    # 2. Re-derive k and write model_selection.csv. Advisory by default: it
    #    warns when the shipped N_CLUSTERS is no longer defensible rather than
    #    breaking the build, since choosing k is an editorial decision.
    run_command([sys.executable, "select_k.py"])

    # 3. Validate
    run_command([sys.executable, "validate_model.py"])

    # 4. Test
    run_command([sys.executable, "-m", "pytest"])

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
