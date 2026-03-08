from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_WEATHER_PATH = Path("data/processed/weather_hourly_clean.csv")
DEFAULT_CHIRPS_FEATURES_PATH = Path("data/processed/chirps_features_daily.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/weather_hourly_clean_enriched.csv")

TIMESTAMP_CANDIDATES = [
    "datetime",
    "timestamp",
    "time",
    "date_time",
    "DateTime",
]

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
    """Configure script logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def detect_timestamp_column(df: pd.DataFrame) -> str:
    """Detect the timestamp column used in the hourly weather dataset."""
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate not in df.columns:
            continue
        parsed = pd.to_datetime(df[candidate], errors="coerce")
        valid_ratio = float(parsed.notna().mean())
        if valid_ratio >= 0.95:
            return candidate

    raise ValueError(
        "Unable to detect timestamp column. Tried: " + ", ".join(TIMESTAMP_CANDIDATES)
    )


def load_weather_hourly(path: Path) -> tuple[pd.DataFrame, str]:
    """Load existing hourly weather data and normalize timestamp parsing."""
    if not path.exists():
        raise FileNotFoundError(f"Weather dataset not found: {path}")

    df = pd.read_csv(path)
    ts_col = detect_timestamp_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return df, ts_col


def load_chirps_features(path: Path) -> pd.DataFrame:
    """Load daily CHIRPS feature file with strict schema checks."""
    if not path.exists():
        raise FileNotFoundError(f"CHIRPS feature file not found: {path}")

    df = pd.read_csv(path)
    expected = {"date", *CHIRPS_FEATURE_COLUMNS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CHIRPS feature file missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return df[["date", *CHIRPS_FEATURE_COLUMNS]]


def merge_chirps_features(
    weather_df: pd.DataFrame,
    timestamp_column: str,
    chirps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily CHIRPS features onto every hourly weather row by date key."""
    weather = weather_df.copy()
    chirps = chirps_df.copy()

    join_col = "__join_date"
    weather[join_col] = weather[timestamp_column].dt.normalize()
    chirps[join_col] = chirps["date"].dt.normalize()

    # Prevent duplicate suffix columns when rerunning enrichment.
    existing_chirps_cols = [c for c in CHIRPS_FEATURE_COLUMNS if c in weather.columns]
    if existing_chirps_cols:
        LOGGER.info("Dropping pre-existing CHIRPS columns before merge: %s", existing_chirps_cols)
        weather = weather.drop(columns=existing_chirps_cols)

    pre_rows = len(weather)
    merged = weather.merge(
        chirps[[join_col, *CHIRPS_FEATURE_COLUMNS]],
        on=join_col,
        how="left",
        validate="many_to_one",
    )

    if len(merged) != pre_rows:
        raise RuntimeError(
            f"Row count changed after merge ({pre_rows} -> {len(merged)}). "
            "This indicates a non-deterministic join."
        )

    for col in CHIRPS_FEATURE_COLUMNS:
        null_count = int(merged[col].isna().sum())
        if null_count:
            LOGGER.warning(
                "Column %s has %d missing values after join; filling with 0.0",
                col,
                null_count,
            )
            merged[col] = merged[col].fillna(0.0)

    merged = merged.drop(columns=[join_col])
    merged = merged.sort_values(timestamp_column).reset_index(drop=True)
    return merged


def save_enriched_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Write enriched hourly dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved %d enriched hourly rows to %s", len(df), output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Merge daily CHIRPS rainfall-memory features into hourly training data."
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=DEFAULT_WEATHER_PATH,
        help=f"Input hourly weather dataset (default: {DEFAULT_WEATHER_PATH}).",
    )
    parser.add_argument(
        "--chirps",
        type=Path,
        default=DEFAULT_CHIRPS_FEATURES_PATH,
        help=f"Input CHIRPS feature dataset (default: {DEFAULT_CHIRPS_FEATURES_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output enriched dataset path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for hourly dataset enrichment."""
    setup_logging()
    args = parse_args()

    weather_df, timestamp_column = load_weather_hourly(args.weather)
    LOGGER.info("Detected timestamp column: %s", timestamp_column)

    chirps_df = load_chirps_features(args.chirps)
    merged_df = merge_chirps_features(weather_df, timestamp_column, chirps_df)
    save_enriched_dataset(merged_df, args.output)


if __name__ == "__main__":
    main()
