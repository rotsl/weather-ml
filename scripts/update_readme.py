import json
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import os
from pathlib import Path
from datetime import datetime


README = Path("README.md")
DATA = Path("data/processed/weather_hourly_clean.csv")
MODELS = Path("models")
HISTORY = MODELS / "history" / "metrics_history.csv"


# =====================================================
# Utils
# =====================================================

def sparkline(series, width=12):

    ticks = "▁▂▃▄▅▆▇█"

    if len(series) < 2:
        return "n/a"

    s = np.array(series[-width:], dtype=float)

    s = (s - s.min()) / (s.max() - s.min() + 1e-9)

    return "".join(ticks[int(x * 7)] for x in s)


# =====================================================
# Load metadata
# =====================================================

meta_file = list(MODELS.glob("*_current_meta.json"))[0]
meta = json.loads(meta_file.read_text())


# =====================================================
# Dataset info
# =====================================================

df = pd.read_csv(DATA, parse_dates=["datetime"])

rows = len(df)
start = df["datetime"].min().strftime("%Y-%m-%d")
end = df["datetime"].max().strftime("%Y-%m-%d")


# =====================================================
# Metrics history
# =====================================================

hist = pd.read_csv(HISTORY, parse_dates=["timestamp"])

roc_trend = sparkline(hist["roc_auc"])
pr_trend = sparkline(hist["pr_auc"])

latest = hist.iloc[-1]
prev = hist.iloc[-2] if len(hist) > 1 else None


# =====================================================
# Degradation check
# =====================================================

warning = ""

if prev is not None:

    if latest["roc_auc"] < prev["roc_auc"] - 0.05:
        warning = "🚨 ROC-AUC dropped > 0.05"

    if latest["pr_auc"] < prev["pr_auc"] - 0.05:
        warning += " | 🚨 PR-AUC dropped > 0.05"


if not warning:
    warning = "✅ No degradation detected"


# =====================================================
# Live forecast
# =====================================================

API_KEY = os.getenv("VISUAL_CROSSING_KEY")
LOCATION = os.getenv("VISUAL_CROSSING_LOCATION")

forecast = "Unavailable"

if API_KEY and LOCATION:

    url = (
        "https://weather.visualcrossing.com/"
        "VisualCrossingWebServices/rest/services/timeline/"
        f"{urllib.parse.quote_plus(LOCATION)}"
        "?unitGroup=metric&include=current&contentType=json"
        f"&key={API_KEY}"
    )

    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            js = json.loads(r.read())

        cur = js["currentConditions"]

        rain_p = cur.get("precipprob", 0)
        temp = cur.get("temp")
        hum = cur.get("humidity")

        forecast = f"{rain_p}% rain | {temp}°C | {hum}% RH"

    except Exception:
        forecast = "API error"


# =====================================================
# Build dashboard
# =====================================================

block = f"""
---

## 📊 Live ML Dashboard (Auto-Updated)

### 🧠 Model

| Field | Value |
|-------|-------|
| Horizon | {meta["horizon"]} ({meta["hours"]}h) |
| Last trained | {meta["trained_at"]} |
| Features | {len(meta["features"])} |
| Positive rate | {meta["positive_rate"]:.4f} |

---

### 📉 Performance

| Metric | Latest | Trend |
|--------|--------|-------|
| ROC-AUC | {latest["roc_auc"]:.4f} | {roc_trend} |
| PR-AUC | {latest["pr_auc"]:.4f} | {pr_trend} |

---

### 🚨 Health

> {warning}

---

### 🌧️ Current Weather

> {forecast}

---

### 📁 Dataset

| Field | Value |
|-------|-------|
| Rows | {rows:,} |
| Range | {start} → {end} |

_Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}_
"""


# =====================================================
# Inject into README
# =====================================================

text = README.read_text() if README.exists() else "# weather-ml\n"

marker = "## 📊 Live ML Dashboard (Auto-Updated)"

if marker in text:
    base = text.split(marker)[0]
    out = base + block
else:
    out = text.rstrip() + "\n" + block


README.write_text(out)

print("✅ README dashboard updated")