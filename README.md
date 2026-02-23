# weather-ml

---
[![Retrain Workflow](https://github.com/rotsl/weather-ml/actions/workflows/retrain.yml/badge.svg)](https://github.com/rotsl/weather-ml/actions/workflows/retrain.yml)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://rotsl.github.io/weather-ml/)
[![Last Commit](https://img.shields.io/github/last-commit/rotsl/weather-ml)](https://github.com/rotsl/weather-ml)
[![Repo Size](https://img.shields.io/github/repo-size/rotsl/weather-ml)](https://github.com/rotsl/weather-ml)
![Workflow Uptime](https://img.shields.io/github/actions/workflow/status/rotsl/weather-ml/retrain.yml?label=Automation&logo=github)
![Dataset Updated](https://img.shields.io/github/last-commit/rotsl/weather-ml/main?label=Dataset&logo=databricks)
![Model Health](https://img.shields.io/badge/Model-Healthy-green)

---

## 🌐 Public Dashboard

➡️ Live dashboard (auto-updated, no API calls):

https://rotsl.github.io/weather-ml/

---

Hourly rain forecasting project using Visual Crossing weather data and scikit-learn gradient boosting models for multiple prediction horizons.

## Project Structure

## 📁 Project Structure

```text
weather-ml/
├── .env                         # Local secrets (not committed)
├── README.md
├── requirements.txt
├── docs/                        # GitHub Pages dashboard
│   └── index.html
├── data/
│   ├── raw/
│   └── processed/
│       └── weather_hourly_clean.csv
├── models/
│   ├── hgb_*_current.pkl
│   ├── hgb_*_previous.pkl
│   ├── *_meta.json
│   ├── history/
│   │   └── metrics_history.csv
│   └── snapshots/
├── notebooks/
├── scripts/
│   └── *.py
└── weather/                     # Local virtual environment
```

## Environment Setup

1. Activate your project environment (for example `./Scikit` if you are using that venv).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add API key to `.env`:

```env
VISUAL_CROSSING_KEY=your_key_here
VISUAL_CROSSING_LOCATION=your_location_query
```

## Pipeline (Run Order)

1. Download raw hourly weather data:

```bash
python3 scripts/download_visual_crossing.py
```

2. Clean and engineer base features (`rain_1h`, `rain_6h`, `rain_24h`):

```bash
python3 scripts/clean_visual_crossing.py
```

3. Train multi-horizon HistGradientBoosting models:

```bash
python3 scripts/train_boosting_multihorizon.py
```

4. Run live inference for the 6-hour horizon:

```bash
python3 scripts/predict_live_vc.py
```

5. Open notebooks for analysis and interactive inspection:

```bash
jupyter notebook notebooks/model_analysis.ipynb
jupyter notebook notebooks/interactive_dashboard.ipynb
```

## Key Outputs

- Clean dataset: `data/processed/weather_hourly_clean.csv`
- Trained model artifacts and metadata: `models/*.pkl`, `models/*_meta.json`
- Horizon metrics table: `models/metrics_multihorizon.csv`

## Notes

- Scripts read location from `VISUAL_CROSSING_LOCATION` in `.env`.
- `interactive_dashboard.ipynb` includes horizon selection, threshold/date controls, live overlay toggle, refresh button, and schema inspector for live API vs model features.

---

## 📊 Live Model Status (Auto-Updated)

| Field | Value |
|-------|-------|
| Last retrain (UTC) | 2026-02-22T18:26:59.129808 |
| Active horizon | D_next_6h (6h) |
| Dataset rows | 17,568 |
| Data range | 2024-02-22 → 2026-02-22 |
| ROC-AUC | 0.9178 |
| PR-AUC | 0.8178 |
| Positive rate | 0.3182 |
| Features used | 17 |

_Last updated automatically by GitHub Actions._

---


---


---

## 📊 Live ML Dashboard (Auto-Updated)

### 🧠 Model

| Field | Value |
|-------|-------|
| Horizon | D_next_6h (6h) |
| Last trained | 2026-02-23T04:25:35.778027 |
| Features | 17 |
| Positive rate | 0.3186 |

---

### 📉 Performance

| Metric | Latest | Trend |
|--------|--------|-------|
| ROC-AUC | 0.9124 | ▇▇▇▁ |
| PR-AUC | 0.8038 | ▇▇▇▁ |

---

### 🚨 Health

> ✅ No degradation detected

---

### 🌧️ Current Weather

> Disabled (free-tier safe mode)

---

### 📁 Dataset

| Field | Value |
|-------|-------|
| Rows | 17,592 |
| Range | 2024-02-22 → 2026-02-23 |

_Last updated: 2026-02-23 04:25 UTC_
