import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import os


# =====================================================
# Paths
# =====================================================

CHANGELOG = Path("CHANGELOG.md")
MODELS = Path("models")
DATA = Path("data/processed/weather_hourly_clean.csv")
HISTORY = MODELS / "history" / "metrics_history.csv"


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

rows = "N/A"

if DATA.exists():
    df = pd.read_csv(DATA)
    rows = f"{len(df):,}"


# =====================================================
# Metrics
# =====================================================

roc = "N/A"
pr = "N/A"
health = "Unknown"

if HISTORY.exists():

    hist = pd.read_csv(HISTORY)

    if len(hist) > 0:

        latest = hist.iloc[-1]
        roc = f"{latest['roc_auc']:.4f}"
        pr = f"{latest['pr_auc']:.4f}"

        # Degradation check
        if len(hist) > 1:
            prev = hist.iloc[-2]

            degraded = (
                latest["roc_auc"] < prev["roc_auc"] - 0.05 or
                latest["pr_auc"] < prev["pr_auc"] - 0.05
            )

            health = "⚠️ Degraded" if degraded else "✅ Stable"

        else:
            health = "✅ Stable"


# =====================================================
# Git info
# =====================================================

commit = os.getenv("GITHUB_SHA", "local")[:8]


# =====================================================
# Build entry
# =====================================================

ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

entry = f"""
## {ts}

- **Horizon:** {meta.get("horizon")} ({meta.get("hours")}h)
- **Dataset rows:** {rows}
- **ROC-AUC:** {roc}
- **PR-AUC:** {pr}
- **Health:** {health}
- **Commit:** `{commit}`

---
"""


# =====================================================
# Append to changelog
# =====================================================

if not CHANGELOG.exists():
    CHANGELOG.write_text("# Changelog\n\n---\n", encoding="utf-8")

old = CHANGELOG.read_text(encoding="utf-8")

new = old.rstrip() + "\n" + entry

CHANGELOG.write_text(new, encoding="utf-8")

print("✅ Changelog updated")