from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/processed/weather_hourly_clean.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = {
    "A_same_hour": 0,
    "B_next_1h": 1,
    "C_next_3h": 3,
    "D_next_6h": 6,
}

BASE_EXCLUDE = {
    "datetime",
    "rain_1h",
    "precip",
    "rain_6h",
    "rain_24h",
}

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


# -----------------------------
# Helpers
# -----------------------------
def make_future_any_rain_target(rain_now: pd.Series, h: int) -> pd.Series:
    if h == 0:
        return rain_now.astype(int)

    shifts = [rain_now.shift(-i) for i in range(1, h + 1)]
    return pd.concat(shifts, axis=1).max(axis=1)


def time_split(df: pd.DataFrame):
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    return train, val, test


def best_f1_threshold(y, p):
    best_f1 = -1
    best_t = 0.5

    for t in np.linspace(0.05, 0.95, 19):
        y_hat = (p >= t).astype(int)

        _, _, f1, _ = precision_recall_fscore_support(
            y, y_hat, average="binary", zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t


def eval_at(y, p, t):
    y_hat = (p >= t).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_hat, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()

    return prec, rec, f1, tn, fp, fn, tp


# -----------------------------
# Main
# -----------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

results = []

for name, h in HORIZONS.items():

    print(f"\n=== {name} (h={h}) ===")

    df[f"target_h{h}"] = make_future_any_rain_target(df["rain_1h"], h)

    work = df.dropna(subset=[f"target_h{h}"]).copy()
    work[f"target_h{h}"] = work[f"target_h{h}"].astype(int)

    # Exclude target + datetime explicitly
    exclude = {"datetime", f"target_h{h}"}

    # Start with numeric columns only
    numeric_cols = work.select_dtypes(include=["number"]).columns.tolist()

    # Remove target and rain leakage columns if present
    leak_cols = {"rain_1h", "precip", "rain_6h", "rain_24h"}
    numeric_cols = [c for c in numeric_cols if c not in leak_cols]
    numeric_cols = [c for c in numeric_cols if not c.startswith("target_h")]

    features = numeric_cols

    train, val, test = time_split(work)

    X_train = train[features]
    y_train = train[f"target_h{h}"]

    X_val = val[features]
    y_val = val[f"target_h{h}"]

    X_test = test[features]
    y_test = test[f"target_h{h}"]

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=600,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_p = model.predict_proba(X_val)[:, 1]
    test_p = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, test_p)
    pr = average_precision_score(y_test, test_p)

    thr = best_f1_threshold(y_val, val_p)

    prec, rec, f1, tn, fp, fn, tp = eval_at(y_test, test_p, thr)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr:.4f}")
    print(f"Thr    : {thr:.2f}")
    print(f"P/R/F1 : {prec:.3f} {rec:.3f} {f1:.3f}")
    print(f"TN FP FN TP: {tn} {fp} {fn} {tp}")

    model_path = MODELS_DIR / f"hgb_{name}.pkl"
    meta_path = MODELS_DIR / f"hgb_{name}_meta.json"

    import joblib

    joblib.dump(model, model_path)

    meta = {
        "horizon": name,
        "hours": h,
        "rows": len(work),
        "features": features,
        "threshold": float(thr),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2))

    results.append(meta)

pd.DataFrame(results).to_csv(
    MODELS_DIR / "metrics_multihorizon.csv", index=False
)

print("\n✅ Training complete.")
