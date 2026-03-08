from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)

SCRIPT_CHIRPS_PIPELINE = Path("scripts/run_chirps_pipeline.py")
SCRIPT_ENRICH = Path("scripts/enrich_training_data_with_chirps.py")
SCRIPT_RETRAIN = Path("scripts/update_data_and_retrain.py")

BASE_DATASET = Path("data/processed/weather_hourly_clean.csv")
ENRICHED_DATASET = Path("data/processed/weather_hourly_clean_enriched.csv")
BASE_FEATURE_META = Path("models/hgb_D_next_6h_meta.json")

CHIRPS_FEATURE_COLUMNS = [
    "chirps_lag1d_mm",
    "chirps_lag2d_mm",
    "chirps_lag3d_mm",
    "chirps_roll3d_mm",
    "chirps_roll7d_mm",
    "chirps_roll14d_mm",
    "chirps_roll30d_mm",
    "chirps_rainy_days_7d",
    "chirps_rainy_days_30d",
    "chirps_max3d_mm",
    "chirps_max7d_mm",
    "chirps_api",
    "chirps_month_mean_mm",
    "chirps_month_std_mm",
    "chirps_anomaly",
    "chirps_wet_spell_len",
    "chirps_dry_spell_len",
]


def setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_step(command: list[str], step_name: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command with consistent logging."""
    LOGGER.info("Starting: %s", step_name)
    LOGGER.info("Command: %s", " ".join(command))
    result = subprocess.run(command, check=check)
    LOGGER.info("Finished: %s (exit=%s)", step_name, result.returncode)
    return result


def dataset_is_valid(path: Path) -> bool:
    """Validate that an enriched dataset exists and has required CHIRPS columns."""
    if not path.exists():
        return False

    try:
        sample = pd.read_csv(path, nrows=5)
    except Exception:
        return False

    required = {"datetime", *CHIRPS_FEATURE_COLUMNS}
    return required.issubset(sample.columns)


def resolve_training_dataset() -> Path:
    """
    Resolve training dataset path for CHIRPS-enabled retraining.

    - Prefer enriched dataset when valid.
    - Fallback to base dataset otherwise.
    """
    if dataset_is_valid(ENRICHED_DATASET):
        LOGGER.info("Resolved training dataset: %s", ENRICHED_DATASET)
        return ENRICHED_DATASET

    LOGGER.warning(
        "Enriched dataset not available/valid. Falling back to baseline dataset: %s",
        BASE_DATASET,
    )
    return BASE_DATASET


def stage_dataset_for_retrain(source_dataset: Path, target_dataset: Path) -> Optional[Path]:
    """Stage a source dataset at the target path expected by the legacy retrain script."""
    if source_dataset.resolve() == target_dataset.resolve():
        return None

    if not target_dataset.exists():
        raise FileNotFoundError(f"Target dataset missing: {target_dataset}")

    backup_path = target_dataset.with_suffix(".pre_chirps_backup.csv")
    shutil.copy2(target_dataset, backup_path)
    shutil.copy2(source_dataset, target_dataset)

    LOGGER.info("Staged CHIRPS-enriched dataset for retraining: %s -> %s", source_dataset, target_dataset)
    return backup_path


def apply_temporary_feature_meta_override() -> Optional[Path]:
    """
    Temporarily append CHIRPS features to retraining feature metadata.

    The original meta file is restored after retraining so existing scripts that
    depend on baseline metadata remain unchanged.
    """
    if not BASE_FEATURE_META.exists():
        LOGGER.warning("Feature meta file not found; skipping CHIRPS feature override: %s", BASE_FEATURE_META)
        return None

    backup_path = BASE_FEATURE_META.with_suffix(".pre_chirps_backup.json")
    shutil.copy2(BASE_FEATURE_META, backup_path)

    meta = json.loads(BASE_FEATURE_META.read_text())
    features = list(meta.get("features", []))

    for col in CHIRPS_FEATURE_COLUMNS:
        if col not in features:
            features.append(col)

    meta["features"] = features
    BASE_FEATURE_META.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Applied temporary CHIRPS feature override in %s", BASE_FEATURE_META)
    return backup_path


def restore_file(backup_path: Optional[Path], target_path: Path) -> None:
    """Restore a backup file to target path if backup exists."""
    if backup_path is None:
        return

    if backup_path.exists():
        shutil.move(str(backup_path), str(target_path))
        LOGGER.info("Restored backup: %s -> %s", backup_path, target_path)


def remove_file_if_exists(path: Optional[Path]) -> None:
    """Delete a file if it exists."""
    if path is None:
        return
    if path.exists():
        path.unlink()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run full training workflow with CHIRPS intelligence layer and safe fallback "
            "to the legacy retraining process."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for subprocess steps (default: current interpreter).",
    )
    parser.add_argument(
        "--skip-chirps",
        action="store_true",
        help="Skip CHIRPS pipeline and run only legacy retraining.",
    )
    parser.add_argument(
        "--strict-chirps",
        action="store_true",
        help="Fail fast if CHIRPS pipeline fails (do not run legacy fallback).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for full training pipeline."""
    setup_logging()
    args = parse_args()
    py = args.python

    LOGGER.info("Starting full training pipeline")

    chirps_ok = False
    staged_dataset_backup: Optional[Path] = None
    staged_meta_backup: Optional[Path] = None
    retrain_ok = False

    try:
        if args.skip_chirps:
            LOGGER.info("CHIRPS pipeline skipped by CLI flag.")
        else:
            chirps_result = run_step(
                [py, str(SCRIPT_CHIRPS_PIPELINE), "--python", py],
                "CHIRPS pipeline",
                check=False,
            )
            chirps_ok = chirps_result.returncode == 0

            if not chirps_ok:
                message = "CHIRPS pipeline failed, falling back to original retraining workflow."
                if args.strict_chirps:
                    LOGGER.error(message)
                    return chirps_result.returncode or 1
                LOGGER.warning(message)

        if chirps_ok:
            resolved_dataset = resolve_training_dataset()
            staged_dataset_backup = stage_dataset_for_retrain(
                source_dataset=resolved_dataset,
                target_dataset=BASE_DATASET,
            )

            if resolved_dataset.resolve() == ENRICHED_DATASET.resolve():
                staged_meta_backup = apply_temporary_feature_meta_override()

        run_step([py, str(SCRIPT_RETRAIN)], "Legacy 48-hour retraining workflow", check=True)
        retrain_ok = True

        if chirps_ok:
            # Keep enriched artifact synchronized with the latest hourly dataset.
            run_step([py, str(SCRIPT_ENRICH)], "Refresh enriched dataset after retraining", check=False)

        LOGGER.info("Full training pipeline completed successfully.")
        return 0

    except subprocess.CalledProcessError as exc:
        LOGGER.error("Training pipeline failed at command: %s", exc.cmd)
        LOGGER.error("Exit code: %s", exc.returncode)
        return exc.returncode or 1

    finally:
        # Always restore metadata so baseline inference scripts remain unchanged.
        restore_file(staged_meta_backup, BASE_FEATURE_META)

        # If retraining failed after staging dataset, roll back to avoid corrupt state.
        if not retrain_ok and staged_dataset_backup is not None:
            restore_file(staged_dataset_backup, BASE_DATASET)
        else:
            remove_file_if_exists(staged_dataset_backup)


if __name__ == "__main__":
    raise SystemExit(main())
