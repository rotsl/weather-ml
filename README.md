# weather-ml

Hourly rain forecasting project using Visual Crossing weather data and scikit-learn gradient boosting models for multiple prediction horizons.

## Project Structure

```text
weather-ml/
├── .env
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── copyofdata.csv
│   │   ├── vc_progress.txt
│   │   └── visual_crossing_2024_2026_hourly.csv
│   └── processed/
│       └── weather_hourly_clean.csv
├── models/
│   ├── hgb_A_same_hour.pkl
│   ├── hgb_A_same_hour_meta.json
│   ├── hgb_B_next_1h.pkl
│   ├── hgb_B_next_1h_meta.json
│   ├── hgb_C_next_3h.pkl
│   ├── hgb_C_next_3h_meta.json
│   ├── hgb_D_next_6h.pkl
│   ├── hgb_D_next_6h_meta.json
│   └── metrics_multihorizon.csv
├── notebooks/
│   ├── interactive_dashboard.ipynb
│   └── model_analysis.ipynb
├── scripts/
│   ├── download_visual_crossing.py
│   ├── clean_visual_crossing.py
│   ├── train_boosting_multihorizon.py
│   └── predict_live_vc.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
└── weather/  # local virtual environment directory in this repo
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

## 📊 Live ML Dashboard (Auto-Updated)

### 🧠 Model

| Field | Value |
|-------|-------|
| Horizon | D_next_6h (6h) |
| Last trained | 2026-02-22T18:44:17.461952 |
| Features | 17 |
| Positive rate | 0.3182 |

---

### 📉 Performance

| Metric | Latest | Trend |
|--------|--------|-------|
| ROC-AUC | 0.9178 | n/a |
| PR-AUC | 0.8178 | n/a |

---

### 🚨 Health

> ✅ No degradation detected

---

### 🌧️ Current Weather

> 0.0% rain | 28.0°C | 83.8% RH

---

### 📁 Dataset

| Field | Value |
|-------|-------|
| Rows | 17,568 |
| Range | 2024-02-22 → 2026-02-22 |

_Last updated: 2026-02-22 18:44 UTC_
