import json
from pathlib import Path
import pandas as pd
from datetime import datetime


README_PATH = Path("README.md")
DATA_PATH = Path("data/processed/weather_hourly_clean.csv")
MODELS_DIR = Path("models")


# -------------------------------------------------
# Load latest model metadata
# -------------------------------------------------

meta_files = sorted(
    MODELS_DIR.glob("*_current_meta.json"),
    reverse=True
)

if not meta_files:
    raise RuntimeError("No model metadata found")

latest_meta_path = meta_files[0]
meta = json.loads(latest_meta_path.read_text())


# -------------------------------------------------
# Load dataset info
# -------------------------------------------------

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

rows = len(df)
start_date = df["datetime"].min().strftime("%Y-%m-%d")
end_date = df["datetime"].max().strftime("%Y-%m-%d")


# -------------------------------------------------
# Extract metrics
# -------------------------------------------------

trained_at = meta["trained_at"]
roc = meta.get("roc_auc", meta.get("roc_auc_val_timeforward", "N/A"))
pr = meta.get("pr_auc", meta.get("pr_auc_val_timeforward", "N/A"))
pos_rate = meta.get("positive_rate", "N/A")
horizon = meta["horizon"]
hours = meta["hours"]
features = len(meta["features"])


# -------------------------------------------------
# Build status section
# -------------------------------------------------

status_block = f"""
---

## 📊 Live Model Status (Auto-Updated)

| Field | Value |
|-------|-------|
| Last retrain (UTC) | {trained_at} |
| Active horizon | {horizon} ({hours}h) |
| Dataset rows | {rows:,} |
| Data range | {start_date} → {end_date} |
| ROC-AUC | {roc:.4f} |
| PR-AUC | {pr:.4f} |
| Positive rate | {pos_rate:.4f} |
| Features used | {features} |

_Last updated automatically by GitHub Actions._
"""


# -------------------------------------------------
# Replace or append section
# -------------------------------------------------

if README_PATH.exists():
    text = README_PATH.read_text()
else:
    text = "# weather-ml\n\n"


marker = "## 📊 Live Model Status (Auto-Updated)"

if marker in text:
    before = text.split(marker)[0]
    new_text = before + status_block
else:
    new_text = text.rstrip() + "\n" + status_block


README_PATH.write_text(new_text)

print("✅ README updated")
