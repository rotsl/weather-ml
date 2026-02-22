from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple, List

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

PULL_DAYS = 3
LIVE_HOURS_KEEP = 72
RECLEAN_DAYS = 14

HORIZON_NAME = "D_next_6h"

DATA_PROCESSED = Path("data/processed/weather_hourly_clean.csv")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_MODEL = MODELS_DIR / f"hgb_{HORIZON_NAME}_current.pkl"
PREVIOUS_MODEL = MODELS_DIR / f"hgb_{HORIZON_NAME}_previous.pkl"
CURRENT_META = MODELS_DIR / f"hgb_{HORIZON_NAME}_current_meta.json"
PREVIOUS_META = MODELS_DIR / f"hgb_{HORIZON_NAME}_previous_meta.json"

SNAPSHOT_DIR = MODELS_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


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


if not API_KEY:
    raise RuntimeError("VISUAL_CROSSING_KEY not found in environment")

if not LOCATION:
    raise RuntimeError("VISUAL_CROSSING_LOCATION not found in environment")


print("Using location:", LOCATION)


# =========================================================
# VISUAL CROSSING FETCH
# =========================================================

def fetch_vc_json(start_date: str, end_date: str) -> dict:

    include = "hours,current"

    url = (
        f"{BASE_URL}"
        f"{urllib.parse.quote_plus(LOCATION)}/"
        f"{start_date}/{end_date}"
        f"?unitGroup={UNIT_GROUP}"
        f"&include={include}"
        f"&contentType={CONTENT_TYPE}"
        f"&key={API_KEY}"
    )

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read())

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}")

    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e.reason}")

    except json.JSONDecodeError:
        raise RuntimeError("JSON decode error")


# =========================================================
# PARSE JSON → HOURLY DF
# =========================================================

def vc_json_to_hours_df(weather_json: dict, hours_keep=72) -> pd.DataFrame:

    records = []

    for day in weather_json.get("days", []):
        for h in day.get("hours", []):

            dt = pd.to_datetime(h.get("datetime"))

            records.append({

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
            })

    df = pd.DataFrame(records).sort_values("datetime")

    for c in df.columns:
        if c != "datetime":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.tail(hours_keep).reset_index(drop=True)


# =========================================================
# CLEAN RECENT WINDOW
# =========================================================

def reclean_recent_window(df_all: pd.DataFrame, days=14) -> pd.DataFrame:

    df = df_all.copy()

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")

    df = df.sort_values("datetime").drop_duplicates("datetime")

    cutoff = df["datetime"].max() - pd.Timedelta(days=days)

    old = df[df["datetime"] < cutoff]
    recent = df[df["datetime"] >= cutoff]

    idx = pd.date_range(
        recent["datetime"].min(),
        recent["datetime"].max(),
        freq="h"
    )

    recent = recent.set_index("datetime").reindex(idx)
    recent.index.name = "datetime"

    num_cols = recent.select_dtypes(include="number").columns

    recent[num_cols] = recent[num_cols].interpolate(
        method="time",
        limit_direction="both"
    )

    recent = recent.ffill().bfill().reset_index()

    # Rain features
    recent["rain_1h"] = (recent["precip"] >= 0.1).astype(int)
    recent["rain_6h"] = recent["precip"].rolling(6, min_periods=1).sum()
    recent["rain_24h"] = recent["precip"].rolling(24, min_periods=1).sum()

    out = pd.concat([old, recent], ignore_index=True)

    return out.sort_values("datetime").ffill().bfill()


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
            return meta["features"]

    raise RuntimeError("No feature meta file found")


def build_target(rain: pd.Series, h: int):

    shifts = [rain.shift(-i) for i in range(1, h + 1)]

    return pd.concat(shifts, axis=1).max(axis=1)


def time_split(df, frac=0.8):

    n = len(df)
    n_tr = int(n * frac)

    return df.iloc[:n_tr], df.iloc[n_tr:]


# =========================================================
# TRAIN
# =========================================================

@dataclass
class TrainResult:
    roc_auc: float
    pr_auc: float
    rows: int
    positive_rate: float


def train_model(df, h, features):

    work = df.copy().sort_values("datetime")

    work["target"] = build_target(work["rain_1h"], h)

    work = work.dropna(subset=["target"])
    work["target"] = work["target"].astype(int)

    for f in features:
        if f not in work.columns:
            work[f] = np.nan

    X = work[features]
    y = work["target"]

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
        rows=len(work),
        positive_rate=float(y.mean())
    )

    return model, res


# =========================================================
# SAVE / ROTATE MODELS
# =========================================================

def rotate_models(model, meta):

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if CURRENT_MODEL.exists():
        CURRENT_MODEL.replace(PREVIOUS_MODEL)

    if CURRENT_META.exists():
        CURRENT_META.replace(PREVIOUS_META)

    joblib.dump(model, CURRENT_MODEL)
    CURRENT_META.write_text(json.dumps(meta, indent=2))

    joblib.dump(
        model,
        SNAPSHOT_DIR / f"hgb_{HORIZON_NAME}_{ts}.pkl"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not DATA_PROCESSED.exists():
        raise RuntimeError("Clean dataset missing")

    df_old = pd.read_csv(
        DATA_PROCESSED,
        parse_dates=["datetime"]
    )

    end = datetime.utcnow().date()
    start = end - timedelta(days=PULL_DAYS)

    print("Fetching:", start, "→", end)

    wj = fetch_vc_json(start.isoformat(), end.isoformat())

    df_live = vc_json_to_hours_df(wj, LIVE_HOURS_KEEP)

    df_all = pd.concat([df_old, df_live])

    df_all["datetime"] = pd.to_datetime(df_all["datetime"]).dt.floor("h")

    df_all = df_all.drop_duplicates("datetime")

    df_clean = reclean_recent_window(df_all, RECLEAN_DAYS)

    df_clean.to_csv(DATA_PROCESSED, index=False)

    features = load_feature_list(HORIZON_NAME)

    h = 6 if "6h" in HORIZON_NAME else 3 if "3h" in HORIZON_NAME else 1

    model, res = train_model(df_clean, h, features)

    meta = {

        "horizon": HORIZON_NAME,
        "hours": h,
        "trained_at": datetime.utcnow().isoformat(),

        "rows": res.rows,
        "positive_rate": res.positive_rate,

        "roc_auc": res.roc_auc,
        "pr_auc": res.pr_auc,

        "features": features,
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