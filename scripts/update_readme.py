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

    if np.allclose(s.max(), s.min()):
        return ticks[0] * len(s)

    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return "".join(ticks[int(x * 7)] for x in s)


# =====================================================
# Load metadata
# =====================================================

meta_files = list(MODELS.glob("*_current_meta.json"))

if not meta_files:
    raise RuntimeError("No current model metadata found.")

meta_file = meta_files[0]
meta = json.loads(meta_file.read_text())


# =====================================================
# Dataset info
# =====================================================

if DATA.exists():
    df = pd.read_csv(DATA, parse_dates=["datetime"])
    rows = len(df)
    start = df["datetime"].min().strftime("%Y-%m-%d")
    end = df["datetime"].max().strftime("%Y-%m-%d")
else:
    rows = 0
    start = "N/A"
    end = "N/A"


# =====================================================
# Metrics history
# =====================================================

if HISTORY.exists():
    hist = pd.read_csv(HISTORY, parse_dates=["timestamp"])

    if len(hist) > 0:
        roc_trend = sparkline(hist["roc_auc"])
        pr_trend = sparkline(hist["pr_auc"])

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else None
    else:
        roc_trend = "n/a"
        pr_trend = "n/a"
        latest = None
        prev = None
else:
    roc_trend = "n/a"
    pr_trend = "n/a"
    latest = None
    prev = None


# =====================================================
# Degradation check
# =====================================================

warning = "No history yet."

if latest is not None:

    warning = ""

    if prev is not None:
        if latest["roc_auc"] < prev["roc_auc"] - 0.05:
            warning = "🚨 ROC-AUC dropped > 0.05"

        if latest["pr_auc"] < prev["pr_auc"] - 0.05:
            if warning:
                warning += " | "
            warning += "🚨 PR-AUC dropped > 0.05"

    if not warning:
        warning = "✅ No degradation detected"


# =====================================================
# Live forecast (Free-tier safe)
# =====================================================

API_KEY = os.getenv("VISUAL_CROSSING_KEY")
LOCATION = os.getenv("VISUAL_CROSSING_LOCATION")
ENABLE_LIVE_FORECAST = os.getenv("ENABLE_LIVE_FORECAST", "0") == "1"

if not ENABLE_LIVE_FORECAST:
    forecast = "Disabled (free-tier safe mode)"
elif not API_KEY or not LOCATION:
    forecast = "Unavailable (missing VISUAL_CROSSING_KEY or VISUAL_CROSSING_LOCATION)"
else:
    forecast = "Unavailable"
    try:
        url = (
            "https://weather.visualcrossing.com/"
            "VisualCrossingWebServices/rest/services/timeline/"
            f"{urllib.parse.quote_plus(LOCATION)}"
            "?unitGroup=metric&include=current&contentType=json"
            f"&key={API_KEY}"
        )

        with urllib.request.urlopen(url, timeout=30) as r:
            js = json.loads(r.read())

        cur = js.get("currentConditions", {})

        rain_p = cur.get("precipprob")
        temp = cur.get("temp")
        hum = cur.get("humidity")

        parts = []
        if rain_p is not None:
            parts.append(f"{rain_p}% rain")
        if temp is not None:
            parts.append(f"{temp}°C")
        if hum is not None:
            parts.append(f"{hum}% RH")

        forecast = " | ".join(parts) if parts else "Unavailable"

    except Exception:
        forecast = "API error"


# =====================================================
# Build dashboard
# =====================================================

latest_roc = f"{latest['roc_auc']:.4f}" if latest is not None else "N/A"
latest_pr = f"{latest['pr_auc']:.4f}" if latest is not None else "N/A"
pos_rate = f"{meta.get('positive_rate', 0):.4f}" if "positive_rate" in meta else "N/A"

block = f"""
---

## 📊 Live ML Dashboard (Auto-Updated)

### 🧠 Model

| Field | Value |
|-------|-------|
| Horizon | {meta.get("horizon")} ({meta.get("hours")}h) |
| Last trained | {meta.get("trained_at")} |
| Features | {len(meta.get("features", []))} |
| Positive rate | {pos_rate} |

---

### 📉 Performance

| Metric | Latest | Trend |
|--------|--------|-------|
| ROC-AUC | {latest_roc} | {roc_trend} |
| PR-AUC | {latest_pr} | {pr_trend} |

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