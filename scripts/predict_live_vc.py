import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO
import urllib.parse

import pandas as pd
import joblib
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
HOURS_LOOKBACK = 24   # for rolling features
HORIZON_NAME = "D_next_6h"

MODEL_PATH = Path("models/hgb_D_next_6h.pkl")
META_PATH = Path("models/hgb_D_next_6h_meta.json")

# Decision threshold (from training)
DEFAULT_THRESHOLD = 0.05

# -----------------------------
# Load secrets
# -----------------------------
load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_KEY")
LOCATION = os.getenv("VISUAL_CROSSING_LOCATION")

if not API_KEY:
    raise ValueError("VISUAL_CROSSING_KEY not found in .env")
if not LOCATION:
    raise ValueError("VISUAL_CROSSING_LOCATION not found in .env")

# -----------------------------
# Load model + metadata
# -----------------------------
print("Loading model...")

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())

FEATURES = meta["features"]
THRESHOLD = meta.get("threshold", DEFAULT_THRESHOLD)

print(f"Using threshold: {THRESHOLD}")

# -----------------------------
# Fetch recent weather
# -----------------------------
def fetch_recent_weather(hours: int = 24) -> pd.DataFrame:

    end = datetime.utcnow().date()
    start = end - timedelta(days=2)

    url = (
        "https://weather.visualcrossing.com/"
        "VisualCrossingWebServices/rest/services/timeline/"
        f"{urllib.parse.quote_plus(LOCATION)}/{start.isoformat()}/{end.isoformat()}"
        "?include=hours"
        "&unitGroup=metric"
        f"&key={API_KEY}"
        "&contentType=csv"
    )

    print("Fetching recent weather...")

    r = requests.get(url, timeout=60)

    if not r.ok:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")

    df = pd.read_csv(StringIO(r.text))

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    return df.tail(hours).copy()


# -----------------------------
# Feature engineering
# -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Ensure numeric
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Rolling precip
    df["rain_6h"] = df["precip"].rolling(6, min_periods=1).sum()
    df["rain_24h"] = df["precip"].rolling(24, min_periods=1).sum()

    # Last row = now
    latest = df.iloc[-1]

    X = latest[FEATURES].to_frame().T

    return X


# -----------------------------
# Predict
# -----------------------------
def main():

    recent = fetch_recent_weather(HOURS_LOOKBACK)

    X = build_features(recent)

    prob = model.predict_proba(X)[0, 1]

    will_rain = prob >= THRESHOLD

    print("\n=== Rain Forecast (Next 6 Hours) ===")
    print(f"Probability : {prob:.3f}")
    print(f"Threshold   : {THRESHOLD:.3f}")
    print(f"Prediction  : {'RAIN ⚠️' if will_rain else 'NO RAIN ☀️'}")

    return prob, will_rain


if __name__ == "__main__":
    main()
