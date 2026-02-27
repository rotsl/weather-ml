# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import os
import re
from pathlib import Path
from datetime import datetime, timezone


# =====================================================
# Paths
# =====================================================

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


def replace_section(text, start_marker, end_marker, new_block):
    """
    Replace content between two markers (inclusive start).
    """
    pattern = re.compile(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        re.DOTALL,
    )

    match = pattern.search(text)

    if not match:
        print(f"⚠️ Section not found: {start_marker}")
        return text

    return text[: match.start()] + new_block + text[match.end():]


# =====================================================
# Load metadata
# =====================================================

meta_files = list(MODELS.glob("*_current_meta.json"))

if not meta_files:
    raise RuntimeError("No current model metadata found")

meta = json.loads(meta_files[0].read_text())


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

latest = None
prev = None
roc_trend = "n/a"
pr_trend = "n/a"

if HISTORY.exists():
    hist = pd.read_csv(HISTORY, parse_dates=["timestamp"])

    if len(hist) > 0:
        roc_trend = sparkline(hist["roc_auc"])
        pr_trend = sparkline(hist["pr_auc"])

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else None


# =====================================================
# Degradation check
# =====================================================

warning = "No history yet"

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
# Live forecast (Quota-safe)
# =====================================================

API_KEY = os.getenv("VISUAL_CROSSING_KEY")
LOCATION = os.getenv("VISUAL_CROSSING_LOCATION")
ENABLE_LIVE = os.getenv("ENABLE_LIVE_FORECAST", "0") == "1"

if not ENABLE_LIVE:
    forecast = "Disabled (free-tier safe mode)"

elif not API_KEY or not LOCATION:
    forecast = "Unavailable (missing credentials)"

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

        rain = cur.get("precipprob")
        temp = cur.get("temp")
        hum = cur.get("humidity")

        parts = []

        if rain is not None:
            parts.append(f"{rain}% rain")
        if temp is not None:
            parts.append(f"{temp}°C")
        if hum is not None:
            parts.append(f"{hum}% RH")

        forecast = " | ".join(parts) if parts else "Unavailable"

    except Exception:
        forecast = "API error"


# =====================================================
# Build Live Model Status block
# =====================================================
roc_auc_val = f"{latest['roc_auc']:.4f}" if latest is not None else "N/A"
pr_auc_val = f"{latest['pr_auc']:.4f}" if latest is not None else "N/A"
status_block = f"{latest:.4f}" if latest is not None else "N/A"
positive_rate_val = f"{float(meta.get('positive_rate', 0)):.4f}"
## 📊 Live Model Status (Auto-Updated)

| Field | Value |
|-------|-------|
| Last retrain (UTC) | {meta.get("trained_at")} |
| Active horizon | {meta.get("horizon")} ({meta.get("hours")}h) |
| Dataset rows | {rows:,} |
| Data range | {start} --> {end} |
| ROC-AUC | {roc_auc_val} |
| PR-AUC | {pr_auc_val} |
| Positive rate | {positive_rate_val} |
| Features used | {len(meta.get("features", []))} |

_Last updated automatically by GitHub Actions._

---
"""


# =====================================================
# Build Dashboard block
# =====================================================

dashboard_block = f"""
## 📊 Live ML Dashboard (Auto-Updated)

### 🧠 Model

| Field | Value |
|-------|-------|
| Horizon | {meta.get("horizon")} ({meta.get("hours")}h) |
| Last trained | {meta.get("trained_at")} |
| Features | {len(meta.get("features", []))} |
| Positive rate | {positive_rate_val} |

---

### 📉 Performance

| Metric | Latest | Trend |
|--------|--------|-------|
| ROC-AUC | {roc_auc_val} | {roc_trend} |
| PR-AUC | {pr_auc_val} | {pr_trend} |

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
| Range | {start} --> {end} |

_Last updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}_

---
"""


# =====================================================
# Update README safely
# =====================================================

text = README.read_text(encoding="utf-8")


# Replace Live Model Status
text = replace_section(
    text,
    "## 📊 Live Model Status (Auto-Updated)",
    "_Last updated automatically by GitHub Actions._\n\n---",
    status_block,
)


# Replace Dashboard
text = replace_section(
    text,
    "## 📊 Live ML Dashboard (Auto-Updated)",
    "_Last updated:",
    dashboard_block,
)


README.write_text(text, encoding="utf-8")

print("✅ README updated safely")
