from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/raw/chirps_daily.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/chirps_features_daily.csv")
DEFAULT_CONFIG_PATH = Path("config/chirps_config.yaml")
DEFAULT_RAIN_THRESHOLD_MM = 0.1
DEFAULT_API_DECAY_K = 0.85


def setup_logging() -> None:
    """Configure script logging output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_feature_config(config_path: Path) -> Tuple[float, float]:
    """Load configurable feature constants from YAML."""
    if not config_path.exists():
        LOGGER.warning("Config file %s not found. Using default feature constants.", config_path)
        return DEFAULT_RAIN_THRESHOLD_MM, DEFAULT_API_DECAY_K

    raw = yaml.safe_load(config_path.read_text()) or {}
    rain_threshold = float(raw.get("rain_threshold_mm", DEFAULT_RAIN_THRESHOLD_MM))
    api_decay_k = float(raw.get("api_decay_k", DEFAULT_API_DECAY_K))
    return rain_threshold, api_decay_k


def load_chirps_daily(input_path: Path) -> pd.DataFrame:
    """Load raw CHIRPS daily precipitation cache."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    expected = {"date", "precip_mm"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("CHIRPS input is empty after parsing.")

    return df[["date", "precip_mm"]]


def ensure_daily_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to complete daily frequency for stable rolling features."""
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    out = df.set_index("date").reindex(full_dates)
    out.index.name = "date"

    missing_days = int(out["precip_mm"].isna().sum())
    if missing_days:
        LOGGER.warning(
            "Found %d missing CHIRPS days. Filling with 0.0 mm for continuity.",
            missing_days,
        )

    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce").fillna(0.0)
    return out.reset_index()


def compute_api(prior_precip: pd.Series, api_decay_k: float) -> pd.Series:
    """
    Compute antecedent precipitation index without leakage.

    API(t) = P(t-1) + k * API(t-1)
    """
    values = prior_precip.to_numpy(dtype=float)
    api = np.full(shape=len(values), fill_value=np.nan, dtype=float)

    running_api = 0.0
    for idx, p_prev in enumerate(values):
        if np.isnan(p_prev):
            api[idx] = np.nan
            continue
        running_api = p_prev + (api_decay_k * running_api)
        api[idx] = running_api

    return pd.Series(api, index=prior_precip.index, name="chirps_api")


def spell_lengths(condition: pd.Series) -> pd.Series:
    """Return consecutive streak length for each day given a boolean condition."""
    lengths: List[int] = []
    current = 0
    for is_true in condition.fillna(False).astype(bool):
        if is_true:
            current += 1
        else:
            current = 0
        lengths.append(current)
    return pd.Series(lengths, index=condition.index, dtype=float)


def build_feature_frame(
    df: pd.DataFrame,
    rain_threshold_mm: float,
    api_decay_k: float,
) -> pd.DataFrame:
    """Build leakage-safe CHIRPS rainfall-memory features."""
    precip = df["precip_mm"].astype(float)
    prior = precip.shift(1)

    features = pd.DataFrame({"date": df["date"]})

    # A) Basic lag features
    features["chirps_lag1d_mm"] = precip.shift(1)
    features["chirps_lag2d_mm"] = precip.shift(2)
    features["chirps_lag3d_mm"] = precip.shift(3)

    # B) Rolling totals over prior days only
    features["chirps_roll3d_mm"] = prior.rolling(window=3, min_periods=3).sum()
    features["chirps_roll7d_mm"] = prior.rolling(window=7, min_periods=7).sum()
    features["chirps_roll14d_mm"] = prior.rolling(window=14, min_periods=14).sum()
    features["chirps_roll30d_mm"] = prior.rolling(window=30, min_periods=30).sum()

    # C) Rain persistence
    rainy_prior = (prior > rain_threshold_mm).astype(float)
    features["chirps_rainy_days_7d"] = rainy_prior.rolling(window=7, min_periods=7).sum()
    features["chirps_rainy_days_30d"] = rainy_prior.rolling(window=30, min_periods=30).sum()

    # D) Rainfall intensity in prior windows
    features["chirps_max3d_mm"] = prior.rolling(window=3, min_periods=3).max()
    features["chirps_max7d_mm"] = prior.rolling(window=7, min_periods=7).max()

    # E) Antecedent precipitation index (prior-day driven)
    features["chirps_api"] = compute_api(prior_precip=prior, api_decay_k=api_decay_k)

    # F) Monthly climatology from historical CHIRPS for this location
    month_index = df["date"].dt.month
    month_stats = (
        pd.DataFrame({"month": month_index, "precip_mm": precip})
        .groupby("month", as_index=True)["precip_mm"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "month_mean", "std": "month_std"})
    )
    month_mean_map: Dict[int, float] = month_stats["month_mean"].to_dict()
    month_std_map: Dict[int, float] = month_stats["month_std"].fillna(0.0).to_dict()

    features["chirps_month_mean_mm"] = month_index.map(month_mean_map).astype(float)
    features["chirps_month_std_mm"] = month_index.map(month_std_map).astype(float)

    # G) Anomaly uses prior-day rainfall only to avoid same-day leakage
    features["chirps_anomaly"] = features["chirps_lag1d_mm"] - features["chirps_month_mean_mm"]

    # H) Wet/dry regime streaks (prior-day streak lengths)
    wet_streak = spell_lengths(precip > rain_threshold_mm)
    dry_streak = spell_lengths(precip <= rain_threshold_mm)
    features["chirps_wet_spell_len"] = wet_streak.shift(1)
    features["chirps_dry_spell_len"] = dry_streak.shift(1)

    return features


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Write daily CHIRPS feature dataframe."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    LOGGER.info("Saved %d feature rows to %s", len(out), output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build advanced leakage-safe CHIRPS rainfall features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CHIRPS cache CSV path (default: {DEFAULT_INPUT_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CHIRPS features CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Optional CHIRPS config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for CHIRPS feature generation."""
    setup_logging()
    args = parse_args()

    rain_threshold_mm, api_decay_k = load_feature_config(args.config)
    LOGGER.info(
        "Building rainfall features (rain_threshold_mm=%.3f, api_decay_k=%.3f)",
        rain_threshold_mm,
        api_decay_k,
    )

    raw_df = load_chirps_daily(args.input)
    daily_df = ensure_daily_continuity(raw_df)
    feature_df = build_feature_frame(
        df=daily_df,
        rain_threshold_mm=rain_threshold_mm,
        api_decay_k=api_decay_k,
    )
    save_features(feature_df, args.output)


if __name__ == "__main__":
    main()
