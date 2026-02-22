import pandas as pd
from pathlib import Path
from datetime import datetime


HISTORY = Path("models/history/metrics_history.csv")
OUT = Path("reports")

OUT.mkdir(exist_ok=True)


def main():

    if not HISTORY.exists():
        print("No history yet")
        return

    df = pd.read_csv(HISTORY, parse_dates=["timestamp"])

    df["month"] = df["timestamp"].dt.to_period("M")

    monthly = df.groupby("month").agg({
        "roc_auc": "mean",
        "pr_auc": "mean",
        "positive_rate": "mean"
    }).reset_index()

    monthly["month"] = monthly["month"].astype(str)

    fname = OUT / f"report_{datetime.utcnow().strftime('%Y_%m')}.csv"

    monthly.to_csv(fname, index=False)

    print("Saved:", fname)


if __name__ == "__main__":
    main()