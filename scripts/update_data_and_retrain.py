from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
UNIT_GROUP = "metric"
CONTENT_TYPE = "json"

# Fetch policy (keep small for free-tier)
PULL_DAYS = 1
LIVE_HOURS_KEEP = 48
RECLEAN_DAYS = 14

HORIZON_NAME = "D_next_6h"

DATA_PROCESSED = Path("data/processed/weather_hourly_clean.csv")
CHIRPS_FEATURES_DAILY = Path("data/processed/chirps_features_daily.csv")
CHIRPS_ENRICHED_OUTPUT = Path("data/processed/weather_hourly_clean_enriched.csv")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_MODEL = MODELS_DIR / f"hgb_{HORIZON_NAME}_current.pkl"
PREVIOUS_MODEL = MODELS_DIR / f"hgb_{HORIZON_NAME}_previous.pkl"
CURRENT_META = MODELS_DIR / f"hgb_{HORIZON_NAME}_current_meta.json"
PREVIOUS_META = MODELS_DIR / f"hgb_{HORIZON_NAME}_previous_meta.json"

SNAPSHOT_DIR = MODELS_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# API QUOTA GUARD (records/day)
# -------------------------
MAX_DAILY_RECORDS = 900  # safety margin under 1000 free/day
RECORDS_PER_DAY_HOURLY = 24


# =========================================================
# ENV (LOCAL + GITHUB)
# =========================================================

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("VISUAL_CROSSING_KEY")
LOCATION = os.getenv("VISUAL_CROSSING_LOCATION")
ENABLE_CHIRPS_TRAINING = os.getenv("ENABLE_CHIRPS_TRAINING", "1") == "1"

if not API_KEY:
    raise RuntimeError("VISUAL_CROSSING_KEY not found in environment")

if not LOCATION:
    raise RuntimeError("VISUAL_CROSSING_LOCATION not found in environment")

print("Using location from secure environment configuration.")


# =========================================================
# QUOTA GUARD
# =========================================================

def estimate_vc_records(days: int, include: str = "hours") -> int:
    """
    Conservative estimate of records returned by Visual Crossing.
    - include=hours  -> ~24 records/day
    - include=minutes -> ~1440 records/day (guarded)
    """
    include_lower = include.lower()

    if "minutes" in include_lower:
        return days * 1440

    if "hours" in include_lower:
        return days * RECORDS_PER_DAY_HOURLY

    # conservative fallback
    return days * 100


# =========================================================
# VISUAL CROSSING FETCH
# =========================================================

def fetch_vc_json(start_date: str, end_date: str, include: str = "hours") -> dict:
    url = (
        f"{BASE_URL}"
        f"{urllib.parse.quote_plus(LOCATION)}/"
        f"{start_date}/{end_date}"
        f"?unitGroup={UNIT_GROUP}"
        f"&include={urllib.parse.quote_plus(include)}"
        f"&contentType={CONTENT_TYPE}"
        f"&key={API_KEY}"
    )

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read())

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: Visual Crossing request failed")

    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e.reason}")

    except json.JSONDecodeError:
        raise RuntimeError("JSON decode error")


# =========================================================
# PARSE JSON → HOURLY DF
# =========================================================

def vc_json_to_hours_df(weather_json: dict, hours_keep: int = 72) -> pd.DataFrame:
    records = []

    for day in weather_json.get("days", []):
        for h in day.get("hours", []):
            dt = pd.to_datetime(h.get("datetime"))

            records.append(
                {
                    "datetime": dt,
                    "temp": h.get("temp"),
                    "humidity": h.get("humidity"),
                    "pressure": h.get("pressure"),
                    "cloudcover": h.get("cloudcover"),
                    "windspeed": h.get("windspeed"),
                    "visibility": h.get("visibility"),
                    "precip": h.get("precip", 0.0),
                    "feelslike": h.get("feelslike"),
                    "dew": h.get("dew"),
                    "precipprob": h.get("precipprob"),
                    "snow": h.get("snow"),
                    "snowdepth": h.get("snowdepth"),
                    "windgust": h.get("windgust"),
                    "winddir": h.get("winddir"),
                    "sealevelpressure": h.get("sealevelpressure"),
                    "solarradiation": h.get("solarradiation"),
                    "solarenergy": h.get("solarenergy"),
                    "uvindex": h.get("uvindex"),
                    "severerisk": h.get("severerisk"),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.sort_values("datetime")

    for c in df.columns:
        if c != "datetime":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.tail(hours_keep).reset_index(drop=True)


# =========================================================
# CLEAN RECENT WINDOW
# =========================================================

def reclean_recent_window(df_all: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    df = df_all.copy()

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    df = df.sort_values("datetime").drop_duplicates("datetime")

    cutoff = df["datetime"].max() - pd.Timedelta(days=days)

    old = df[df["datetime"] < cutoff].copy()
    recent = df[df["datetime"] >= cutoff].copy()

    if recent.empty:
        # nothing to reclean, return original
        return df.ffill().bfill()

    idx = pd.date_range(recent["datetime"].min(), recent["datetime"].max(), freq="h")

    recent = recent.set_index("datetime").reindex(idx)
    recent.index.name = "datetime"

    num_cols = recent.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        recent[num_cols] = recent[num_cols].interpolate(method="time", limit_direction="both")

    recent = recent.ffill().bfill().reset_index()

    # Rain features (assumes precip exists)
    if "precip" in recent.columns:
        recent["rain_1h"] = (recent["precip"] >= 0.1).astype(int)
        recent["rain_6h"] = recent["precip"].rolling(6, min_periods=1).sum()
        recent["rain_24h"] = recent["precip"].rolling(24, min_periods=1).sum()
    else:
        # Ensure columns exist
        recent["precip"] = 0.0
        recent["rain_1h"] = 0
        recent["rain_6h"] = 0.0
        recent["rain_24h"] = 0.0

    out = pd.concat([old, recent], ignore_index=True)
    out = out.sort_values("datetime").ffill().bfill()
    return out


def strip_chirps_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pre-existing CHIRPS columns before recleaning baseline weather."""
    chirps_cols = [c for c in df.columns if c.startswith("chirps_")]
    if chirps_cols:
        print(f"Dropping {len(chirps_cols)} existing CHIRPS columns before reclean.")
        return df.drop(columns=chirps_cols)
    return df


def load_chirps_daily_features(path: Path) -> tuple[pd.DataFrame, List[str]]:
    """Load daily CHIRPS feature table."""
    if not path.exists():
        raise FileNotFoundError(f"CHIRPS feature file not found: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise RuntimeError(f"CHIRPS feature file {path} missing 'date' column")

    chirps_cols = [c for c in df.columns if c.startswith("chirps_")]
    if not chirps_cols:
        raise RuntimeError(f"CHIRPS feature file {path} has no 'chirps_' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df[["date", *chirps_cols]], chirps_cols


def merge_chirps_features(df_hourly: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Merge daily CHIRPS features onto hourly records."""
    daily_chirps, chirps_cols = load_chirps_daily_features(CHIRPS_FEATURES_DAILY)

    work = df_hourly.copy()
    work["date"] = pd.to_datetime(work["datetime"]).dt.normalize()

    merged = work.merge(daily_chirps, on="date", how="left", validate="many_to_one")
    merged = merged.drop(columns=["date"])

    for col in chirps_cols:
        missing = int(merged[col].isna().sum())
        if missing:
            print(f"CHIRPS column {col} missing on {missing} rows; filling with 0.0")
            merged[col] = merged[col].fillna(0.0)

    CHIRPS_ENRICHED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CHIRPS_ENRICHED_OUTPUT, index=False)
    print(f"Saved CHIRPS-enriched dataset: {CHIRPS_ENRICHED_OUTPUT}")
    return merged, chirps_cols


def extend_features_with_chirps(base_features: List[str], df: pd.DataFrame) -> List[str]:
    """Append available CHIRPS columns to training feature list."""
    features = list(base_features)
    chirps_cols = [c for c in df.columns if c.startswith("chirps_")]

    for col in chirps_cols:
        if col not in features:
            features.append(col)

    return features


# =========================================================
# FEATURES / TARGET
# =========================================================

def load_feature_list(horizon: str) -> List[str]:
    paths = [
        MODELS_DIR / f"hgb_{horizon}_meta.json",
        CURRENT_META,
    ]

    for p in paths:
        if p.exists():
            meta = json.loads(p.read_text())
            if "features" not in meta:
                raise RuntimeError(f"Meta file {p} missing 'features'")
            return meta["features"]

    raise RuntimeError("No feature meta file found")


def build_target(rain: pd.Series, h: int) -> pd.Series:
    shifts = [rain.shift(-i) for i in range(1, h + 1)]
    return pd.concat(shifts, axis=1).max(axis=1)


def time_split(df: pd.DataFrame, frac: float = 0.8):
    n = len(df)
    n_tr = int(n * frac)
    return df.iloc[:n_tr].copy(), df.iloc[n_tr:].copy()


# =========================================================
# TRAIN
# =========================================================

@dataclass
class TrainResult:
    roc_auc: float
    pr_auc: float
    rows: int
    positive_rate: float


def train_model(df: pd.DataFrame, h: int, features: List[str]):
    work = df.copy().sort_values("datetime")

    if "rain_1h" not in work.columns:
        raise RuntimeError("Expected column 'rain_1h' not found. Did cleaning run?")

    work["target"] = build_target(work["rain_1h"], h)
    work = work.dropna(subset=["target"])
    work["target"] = work["target"].astype(int)

    for f in features:
        if f not in work.columns:
            work[f] = np.nan

    tr, va = time_split(work)

    X_tr, y_tr = tr[features], tr["target"]
    X_va, y_va = va[features], va["target"]

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=1200,
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.1,
        random_state=42,
    )

    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_va)[:, 1]

    roc = roc_auc_score(y_va, prob) if y_va.nunique() > 1 else np.nan
    pr = average_precision_score(y_va, prob) if y_va.nunique() > 1 else np.nan

    res = TrainResult(
        roc_auc=float(roc),
        pr_auc=float(pr),
        rows=int(len(work)),
        positive_rate=float(work["target"].mean()),
    )

    return model, res


# =========================================================
# SAVE / ROTATE MODELS
# =========================================================

def rotate_models(model, meta: dict):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # move current -> previous
    if CURRENT_MODEL.exists():
        CURRENT_MODEL.replace(PREVIOUS_MODEL)
    if CURRENT_META.exists():
        CURRENT_META.replace(PREVIOUS_META)

    # save new current
    joblib.dump(model, CURRENT_MODEL)
    CURRENT_META.write_text(json.dumps(meta, indent=2))

    # snapshot
    joblib.dump(model, SNAPSHOT_DIR / f"hgb_{HORIZON_NAME}_{ts}.pkl")
    (SNAPSHOT_DIR / f"hgb_{HORIZON_NAME}_{ts}_meta.json").write_text(json.dumps(meta, indent=2))


# =========================================================
# MAIN
# =========================================================

def main():
    if not DATA_PROCESSED.exists():
        raise RuntimeError("Clean dataset missing")

    df_old = pd.read_csv(DATA_PROCESSED, parse_dates=["datetime"])
    df_old = strip_chirps_columns(df_old)

    end = datetime.utcnow().date()
    start = end - timedelta(days=PULL_DAYS)

    include_mode = "hours"

    # -------------------------
    # API QUOTA GUARD
    # -------------------------
    est_records = estimate_vc_records(PULL_DAYS, include=include_mode)
    print(f"Estimated VC records this run: {est_records} (include={include_mode}, days={PULL_DAYS})")

    if est_records > MAX_DAILY_RECORDS:
        print("🚨 API quota guard triggered.")
        print(f"Estimate {est_records} > limit {MAX_DAILY_RECORDS}")
        print("Skipping VC fetch and retraining to protect free tier.")
        return

    print("Fetching:", start, "→", end)

    wj = fetch_vc_json(start.isoformat(), end.isoformat(), include=include_mode)
    df_live = vc_json_to_hours_df(wj, hours_keep=LIVE_HOURS_KEEP)

    if df_live.empty:
        print("No live VC hours returned. Skipping retraining.")
        return

    df_all = pd.concat([df_old, df_live], ignore_index=True)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"]).dt.floor("h")
    df_all = df_all.sort_values("datetime").drop_duplicates("datetime")

    df_clean = reclean_recent_window(df_all, days=RECLEAN_DAYS)
    df_train = df_clean.copy()
    chirps_cols_in_use: List[str] = []

    if ENABLE_CHIRPS_TRAINING:
        try:
            df_train, chirps_cols_in_use = merge_chirps_features(df_clean)
            print(f"CHIRPS enrichment enabled with {len(chirps_cols_in_use)} features.")
        except Exception as e:
            print(f"WARNING: CHIRPS enrichment unavailable, using baseline features. ({e})")
            df_train = df_clean.copy()
    else:
        print("CHIRPS enrichment disabled by ENABLE_CHIRPS_TRAINING=0")

    features = load_feature_list(HORIZON_NAME)
    features = extend_features_with_chirps(features, df_train)
    h = 6 if "6h" in HORIZON_NAME else 3 if "3h" in HORIZON_NAME else 1
    chirps_feature_count = len([f for f in features if f.startswith("chirps_")])
    if chirps_feature_count > 0:
        print(f"Training with {chirps_feature_count} CHIRPS feature columns.")

    # Keep runtime dataset aligned with the model's expected features.
    for f in features:
        if f not in df_train.columns:
            if f.startswith("chirps_"):
                df_train[f] = 0.0
            else:
                df_train[f] = np.nan

    # Save updated processed dataset (baseline + optional CHIRPS columns)
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(DATA_PROCESSED, index=False)

    model, res = train_model(df_train, h, features)

    meta = {
        "horizon": HORIZON_NAME,
        "hours": h,
        "trained_at": datetime.utcnow().isoformat(),
        "rows": res.rows,
        "positive_rate": res.positive_rate,
        "roc_auc": res.roc_auc,
        "pr_auc": res.pr_auc,
        "features": features,
        "chirps_enabled": bool(chirps_feature_count > 0),
        "chirps_feature_count": int(chirps_feature_count),
    }

    rotate_models(model, meta)

    print("✅ Retraining complete")
    print("ROC:", res.roc_auc)
    print("PR :", res.pr_auc)

    # -------------------------
    # Append metrics history
    # -------------------------
    HISTORY_PATH = MODELS_DIR / "history" / "metrics_history.csv"
    HISTORY_PATH.parent.mkdir(exist_ok=True)

    row = {
        "timestamp": meta["trained_at"],
        "horizon": meta["horizon"],
        "roc_auc": meta["roc_auc"],
        "pr_auc": meta["pr_auc"],
        "positive_rate": meta["positive_rate"],
    }

    df_row = pd.DataFrame([row])

    if HISTORY_PATH.exists():
        df_row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORY_PATH, index=False)


if __name__ == "__main__":
    main()
