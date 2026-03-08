from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/chirps_config.yaml")
DEFAULT_OUTPUT_PATH = Path("data/raw/chirps_daily.csv")
CHIRPS_COLLECTION_ID = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_BAND = "precipitation"
DEFAULT_START_DATE = date(1981, 1, 1)
DEFAULT_SCALE_METERS = 5566
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class FetchConfig:
    """Runtime configuration for CHIRPS download."""

    latitude: float
    longitude: float
    start_date: date
    end_date: date
    scale_meters: int
    cache_enabled: bool
    max_retries: int
    retry_delay_seconds: float


def setup_logging() -> None:
    """Configure structured script logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_iso_date(value: str, field_name: str) -> date:
    """Parse an ISO date in YYYY-MM-DD format."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format, got {value!r}") from exc


def parse_float(value: str, field_name: str) -> float:
    """Parse float values with a consistent validation message."""
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def parse_lat_lon_pair(raw: str) -> Optional[Tuple[float, float]]:
    """Parse a comma-separated latitude,longitude pair."""
    if not raw:
        return None

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        return None

    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError:
        return None

    return lat, lon


def resolve_location() -> Tuple[float, float, str]:
    """
    Resolve location in priority order:
    1) runtime env vars
    2) .env values loaded via dotenv
    3) optional non-secret YAML defaults
    """
    env_pairs = [
        ("LOCATION_LAT", "LOCATION_LON", "LOCATION_LAT/LOCATION_LON"),
        ("CHIRPS_LAT", "CHIRPS_LON", "CHIRPS_LAT/CHIRPS_LON"),
        ("LATITUDE", "LONGITUDE", "LATITUDE/LONGITUDE"),
    ]

    for lat_key, lon_key, source_label in env_pairs:
        lat_val = os.getenv(lat_key)
        lon_val = os.getenv(lon_key)
        if lat_val and lon_val:
            lat = parse_float(lat_val, lat_key)
            lon = parse_float(lon_val, lon_key)
            return lat, lon, source_label

    vc_location = os.getenv("VISUAL_CROSSING_LOCATION", "")
    parsed = parse_lat_lon_pair(vc_location)
    if parsed is not None:
        lat, lon = parsed
        return lat, lon, "VISUAL_CROSSING_LOCATION"

    raise ValueError(
        "Location not found in environment. Set LOCATION_LAT/LOCATION_LON (preferred) "
        "or provide coordinates in VISUAL_CROSSING_LOCATION via .env or CI secrets."
    )


def load_config(config_path: Path) -> FetchConfig:
    """Load fetch configuration from YAML and environment variables."""
    load_dotenv()

    if config_path.exists():
        raw_config = yaml.safe_load(config_path.read_text()) or {}
    else:
        raw_config = {}
        LOGGER.warning("Config file %s not found. Using defaults + environment only.", config_path)

    latitude, longitude, source = resolve_location()
    LOGGER.info("Resolved CHIRPS location from %s", source)

    start_raw = str(raw_config.get("start_date", DEFAULT_START_DATE.isoformat()))
    end_raw = raw_config.get("end_date")
    if end_raw in (None, "", "null"):
        end_value = date.today()
    else:
        end_value = parse_iso_date(str(end_raw), "end_date")

    start_value = parse_iso_date(start_raw, "start_date")
    if end_value < start_value:
        raise ValueError("end_date must be greater than or equal to start_date")

    return FetchConfig(
        latitude=latitude,
        longitude=longitude,
        start_date=start_value,
        end_date=end_value,
        scale_meters=int(raw_config.get("scale_meters", DEFAULT_SCALE_METERS)),
        cache_enabled=bool(raw_config.get("cache_enabled", True)),
        max_retries=max(1, int(raw_config.get("max_retries", DEFAULT_MAX_RETRIES))),
        retry_delay_seconds=float(
            raw_config.get("retry_delay_seconds", DEFAULT_RETRY_DELAY_SECONDS)
        ),
    )


def upsert_env_var(env_path: Path, key: str, value: str) -> None:
    """Create or update one environment variable in a local .env file."""
    lines: List[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    updated = False
    out_lines: List[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            out_lines.append(f"{key}={value}")
            updated = True
        else:
            out_lines.append(line)

    if not updated:
        out_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(out_lines).rstrip() + "\n")


def resolve_ee_project() -> Optional[str]:
    """
    Resolve EE project from environment.

    For local interactive sessions, prompt once if missing and persist to .env.
    """
    project = os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if project:
        return project

    is_ci = os.getenv("CI", "").lower() == "true"
    if is_ci or not sys.stdin.isatty():
        return None

    print(
        "Earth Engine requires a Google Cloud project ID (EE_PROJECT). "
        "Enter it now to save in .env for future runs."
    )
    entered = input("EE_PROJECT: ").strip()
    if not entered:
        return None

    os.environ["EE_PROJECT"] = entered
    try:
        upsert_env_var(Path(".env"), "EE_PROJECT", entered)
        LOGGER.info("Saved EE_PROJECT to local .env")
    except Exception as exc:
        LOGGER.warning("Could not persist EE_PROJECT to .env: %s", exc)

    return entered


def get_ee_module() -> Any:
    """Import the Earth Engine module and return it."""
    try:
        import ee  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "earthengine-api is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return ee


def initialize_earth_engine(ee_module: Any) -> None:
    """Initialize Earth Engine for CI/service-account or local interactive use."""
    project = resolve_ee_project()

    try:
        if project:
            ee_module.Initialize(project=project)
        else:
            ee_module.Initialize()
        LOGGER.info("Earth Engine initialized.")
        return
    except Exception as init_exc:
        is_ci = os.getenv("CI", "").lower() == "true"
        if is_ci:
            raise RuntimeError(
                "Earth Engine initialization failed in CI. Configure GEE credentials/secrets."
            ) from init_exc

    LOGGER.info("Earth Engine not initialized yet. Starting local authentication flow.")
    ee_module.Authenticate()
    project = project or resolve_ee_project()
    if project:
        ee_module.Initialize(project=project)
    else:
        raise RuntimeError(
            "Earth Engine authenticated, but no EE project is configured. "
            "Set EE_PROJECT in .env (or environment) and retry."
        )
    LOGGER.info("Earth Engine initialized after authentication.")


def iter_year_windows(start_date: date, end_date: date) -> Iterator[Tuple[date, date]]:
    """Yield yearly windows to keep API calls bounded and resumable."""
    cursor = start_date
    while cursor <= end_date:
        window_end = min(date(cursor.year, 12, 31), end_date)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def run_with_retries(func, *, description: str, retries: int, delay_seconds: float):
    """Execute a callable with retry handling for transient failures."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - runtime safety path
            last_exc = exc
            if attempt >= retries:
                break
            LOGGER.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                description,
                attempt,
                retries,
                exc,
                delay_seconds,
            )
            time.sleep(delay_seconds)

    raise RuntimeError(f"{description} failed after {retries} attempts") from last_exc


def fetch_window_records(
    ee_module: Any,
    point: Any,
    start_date: date,
    end_date: date,
    scale_meters: int,
    max_retries: int,
    retry_delay_seconds: float,
) -> List[dict]:
    """Fetch CHIRPS daily records for one date window."""
    end_exclusive = end_date + timedelta(days=1)

    def _fetch() -> List[dict]:
        collection = (
            ee_module.ImageCollection(CHIRPS_COLLECTION_ID)
            .filterBounds(point)
            .filterDate(start_date.isoformat(), end_exclusive.isoformat())
            .select(CHIRPS_BAND)
        )

        def image_to_feature(image: Any) -> Any:
            precipitation = image.reduceRegion(
                reducer=ee_module.Reducer.mean(),
                geometry=point,
                scale=scale_meters,
                bestEffort=True,
                maxPixels=1_000_000,
            ).get(CHIRPS_BAND)

            return ee_module.Feature(
                None,
                {
                    "date": ee_module.Date(image.get("system:time_start")).format("YYYY-MM-dd"),
                    "precip_mm": precipitation,
                },
            )

        feature_collection = ee_module.FeatureCollection(collection.map(image_to_feature))
        payload = feature_collection.getInfo() or {}
        features = payload.get("features", [])

        out: List[dict] = []
        for feature in features:
            props = feature.get("properties", {})
            row_date = props.get("date")
            if row_date is None:
                continue
            precip_value = props.get("precip_mm")
            out.append(
                {
                    "date": row_date,
                    "precip_mm": None if precip_value is None else float(precip_value),
                }
            )
        return out

    return run_with_retries(
        _fetch,
        description=f"CHIRPS fetch {start_date} -> {end_date}",
        retries=max_retries,
        delay_seconds=retry_delay_seconds,
    )


def load_existing_cache(path: Path) -> pd.DataFrame:
    """Load cached CHIRPS CSV if it exists and is valid."""
    if not path.exists():
        return pd.DataFrame(columns=["date", "precip_mm"])

    cached = pd.read_csv(path)
    expected = {"date", "precip_mm"}
    missing = expected - set(cached.columns)
    if missing:
        raise ValueError(f"Cache file {path} missing required columns: {sorted(missing)}")

    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
    cached["precip_mm"] = pd.to_numeric(cached["precip_mm"], errors="coerce")
    cached = cached.dropna(subset=["date"]).sort_values("date")
    cached = cached.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return cached


def fetch_chirps_dataframe(config: FetchConfig, output_path: Path, ee_module: Any) -> pd.DataFrame:
    """Fetch CHIRPS daily series with cache-aware incremental updates."""
    existing = load_existing_cache(output_path)
    point = ee_module.Geometry.Point([config.longitude, config.latitude])

    fetch_start = config.start_date
    if config.cache_enabled and not existing.empty:
        cached_last_date = existing["date"].max().date()
        fetch_start = max(config.start_date, cached_last_date + timedelta(days=1))

    if fetch_start > config.end_date:
        LOGGER.info("CHIRPS cache is already up to date through %s", config.end_date)
        return existing

    LOGGER.info("Fetching CHIRPS rainfall from %s to %s", fetch_start, config.end_date)
    records: List[dict] = []

    for window_start, window_end in iter_year_windows(fetch_start, config.end_date):
        LOGGER.info("Request window: %s -> %s", window_start, window_end)
        window_records = fetch_window_records(
            ee_module=ee_module,
            point=point,
            start_date=window_start,
            end_date=window_end,
            scale_meters=config.scale_meters,
            max_retries=config.max_retries,
            retry_delay_seconds=config.retry_delay_seconds,
        )
        LOGGER.info("Received %d rows for %s -> %s", len(window_records), window_start, window_end)
        records.extend(window_records)

    new_df = pd.DataFrame(records, columns=["date", "precip_mm"])
    if not new_df.empty:
        new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
        new_df["precip_mm"] = pd.to_numeric(new_df["precip_mm"], errors="coerce")
        new_df = new_df.dropna(subset=["date"]).sort_values("date")

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    combined = combined[["date", "precip_mm"]].reset_index(drop=True)

    if combined.empty:
        raise RuntimeError("No CHIRPS records available after fetch.")

    return combined


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Persist CHIRPS daily cache CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    LOGGER.info("Saved %d CHIRPS rows to %s", len(out), output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fetch historical CHIRPS daily precipitation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to CHIRPS YAML config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for CHIRPS history download."""
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    ee_module = get_ee_module()
    initialize_earth_engine(ee_module)

    df = fetch_chirps_dataframe(config=config, output_path=args.output, ee_module=ee_module)
    save_dataframe(df=df, output_path=args.output)


if __name__ == "__main__":
    main()
