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
from dotenv import load_dotenv


# =====================================================
# Paths
# =====================================================

README = Path("README.md")
DATA = Path("data/processed/weather_hourly_clean.csv")
CHIRPS_RAW = Path("data/raw/chirps_daily.csv")
CHIRPS_FEATURES = Path("data/processed/chirps_features_daily.csv")
CHIRPS_ENRICHED = Path("data/processed/weather_hourly_clean_enriched.csv")
MODELS = Path("models")
HISTORY = MODELS / "history" / "metrics_history.csv"
STATUS_START = "<!-- AUTO_STATUS_START -->"
STATUS_END = "<!-- AUTO_STATUS_END -->"
DASHBOARD_START = "<!-- AUTO_DASHBOARD_START -->"
DASHBOARD_END = "<!-- AUTO_DASHBOARD_END -->"


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


def replace_marked_or_legacy_section(text, marker_start, marker_end, heading, body):
    marked_block = f"{marker_start}\n{body.strip()}\n{marker_end}"

    marker_pattern = re.compile(
        rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
        re.DOTALL,
    )
    if marker_pattern.search(text):
        return marker_pattern.sub(marked_block, text, count=1)

    # Legacy fallback: replace the whole section from heading to next level-2 heading.
    legacy_pattern = re.compile(
        rf"{re.escape(heading)}.*?(?=\n## [^\n]+|\Z)",
        re.DOTALL,
    )
    if legacy_pattern.search(text):
        return legacy_pattern.sub(marked_block + "\n\n", text, count=1)

    # If neither exists, append at end to keep script non-destructive.
    return text.rstrip() + "\n\n" + marked_block + "\n"


# =====================================================
# Load metadata
# =====================================================

meta_files = list(MODELS.glob("*_current_meta.json"))

if not meta_files:
    raise RuntimeError("No current model metadata found")

meta = json.loads(meta_files[0].read_text())
chirps_feature_count = len([f for f in meta.get("features", []) if str(f).startswith("chirps_")])
chirps_status = "Enabled" if chirps_feature_count > 0 else "Disabled"


# =====================================================
# Dataset info
# =====================================================

rows = 0
start = "N/A"
end = "N/A"

if DATA.exists():
    df = pd.read_csv(DATA, parse_dates=["datetime"])
    if not df.empty:
        rows = len(df)
        start = df["datetime"].min().strftime("%Y-%m-%d")
        end = df["datetime"].max().strftime("%Y-%m-%d")

chirps_raw_rows = 0
chirps_feature_rows = 0
chirps_enriched_rows = 0

if CHIRPS_RAW.exists():
    chirps_raw_rows = len(pd.read_csv(CHIRPS_RAW))
if CHIRPS_FEATURES.exists():
    chirps_feature_rows = len(pd.read_csv(CHIRPS_FEATURES))
if CHIRPS_ENRICHED.exists():
    chirps_enriched_rows = len(pd.read_csv(CHIRPS_ENRICHED))


# =====================================================
# Metrics history
# =====================================================

latest = None
prev = None
roc_trend = "n/a"
pr_trend = "n/a"

if HISTORY.exists():
    hist = pd.read_csv(HISTORY, parse_dates=["timestamp"])

    if not hist.empty:
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

load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_KEY")
LOCATION = os.getenv("VISUAL_CROSSING_LOCATION")
ENABLE_LIVE = os.getenv("ENABLE_LIVE_FORECAST", "0") == "1"

forecast = "Disabled (free-tier safe mode)"

if ENABLE_LIVE and API_KEY and LOCATION:
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
# Safe metric formatting
# =====================================================

roc_auc_val = f"{latest['roc_auc']:.4f}" if latest is not None else "N/A"
pr_auc_val = f"{latest['pr_auc']:.4f}" if latest is not None else "N/A"
positive_rate_val = f"{float(meta.get('positive_rate', 0)):.4f}"


# =====================================================
# Build Live Model Status block
# =====================================================

status_block = f"""## 📊 Live Model Status (Auto-Updated)

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
| CHIRPS training | {chirps_status} |
| CHIRPS feature count | {chirps_feature_count} |
| CHIRPS raw rows | {chirps_raw_rows:,} |
| CHIRPS feature rows | {chirps_feature_rows:,} |
| CHIRPS enriched rows | {chirps_enriched_rows:,} |

_Last updated automatically by GitHub Actions._
"""


# =====================================================
# Build Dashboard block
# =====================================================

dashboard_block = f"""## 📊 Live ML Dashboard (Auto-Updated)

### 🧠 Model

| Field | Value |
|-------|-------|
| Horizon | {meta.get("horizon")} ({meta.get("hours")}h) |
| Last trained | {meta.get("trained_at")} |
| Features | {len(meta.get("features", []))} |
| CHIRPS features | {chirps_feature_count} |
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
"""


# =====================================================
# Update README safely
# =====================================================

if not README.exists():
    raise FileNotFoundError("README.md not found")

text = README.read_text(encoding="utf-8")

text = replace_marked_or_legacy_section(
    text,
    STATUS_START,
    STATUS_END,
    "## 📊 Live Model Status (Auto-Updated)",
    status_block,
)

text = replace_marked_or_legacy_section(
    text,
    DASHBOARD_START,
    DASHBOARD_END,
    "## 📊 Live ML Dashboard (Auto-Updated)",
    dashboard_block,
)

README.write_text(text, encoding="utf-8")

print("✅ README updated safely")
