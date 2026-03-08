from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


LOGGER = logging.getLogger(__name__)

SCRIPT_FETCH = Path("scripts/fetch_chirps_history.py")
SCRIPT_FEATURES = Path("scripts/build_chirps_features.py")
SCRIPT_ENRICH = Path("scripts/enrich_training_data_with_chirps.py")

RAW_CHIRPS_PATH = Path("data/raw/chirps_daily.csv")
FEATURES_PATH = Path("data/processed/chirps_features_daily.csv")
ENRICHED_PATH = Path("data/processed/weather_hourly_clean_enriched.csv")


def setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_step(command: List[str], step_name: str) -> None:
    """Run one subprocess step and raise on failure."""
    LOGGER.info("Starting step: %s", step_name)
    LOGGER.info("Command: %s", " ".join(command))
    subprocess.run(command, check=True)
    LOGGER.info("Completed step: %s", step_name)


def summarize_file(path: Path) -> str:
    """Return a concise output summary for a CSV file."""
    if not path.exists():
        return f"{path} (missing)"

    try:
        rows = len(pd.read_csv(path))
    except Exception:
        rows = -1

    if rows >= 0:
        return f"{path} (rows={rows})"
    return f"{path} (exists, row_count_unavailable)"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the full CHIRPS data-feature-enrichment pipeline.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for subprocess steps (default: current interpreter).",
    )
    parser.add_argument(
        "--config",
        default="config/chirps_config.yaml",
        help="Config path passed to CHIRPS fetch/feature scripts.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for CHIRPS pipeline runner."""
    setup_logging()
    args = parse_args()

    py = args.python
    config_path = args.config

    try:
        run_step([py, str(SCRIPT_FETCH), "--config", config_path], "Fetch CHIRPS rainfall")
        run_step(
            [py, str(SCRIPT_FEATURES), "--config", config_path],
            "Build CHIRPS rainfall features",
        )
        run_step([py, str(SCRIPT_ENRICH)], "Enrich hourly training data with CHIRPS")
    except subprocess.CalledProcessError as exc:
        LOGGER.error("CHIRPS pipeline failed at command: %s", exc.cmd)
        LOGGER.error("Exit code: %s", exc.returncode)
        return int(exc.returncode) if exc.returncode else 1

    LOGGER.info("CHIRPS pipeline complete. Output summary:")
    LOGGER.info("- %s", summarize_file(RAW_CHIRPS_PATH))
    LOGGER.info("- %s", summarize_file(FEATURES_PATH))
    LOGGER.info("- %s", summarize_file(ENRICHED_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
