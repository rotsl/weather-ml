# weather-ml

---
[![Retrain Workflow](https://github.com/rotsl/weather-ml/actions/workflows/retrain.yml/badge.svg)](https://github.com/rotsl/weather-ml/actions/workflows/retrain.yml)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen?logo=github)](https://rotsl.github.io/weather-ml/)
[![Last Commit](https://img.shields.io/github/last-commit/rotsl/weather-ml)](https://github.com/rotsl/weather-ml)
[![Repo Size](https://img.shields.io/github/repo-size/rotsl/weather-ml)](https://github.com/rotsl/weather-ml)
![Workflow Uptime](https://img.shields.io/github/actions/workflow/status/rotsl/weather-ml/retrain.yml?label=Automation&logo=github)
![Dataset Updated](https://img.shields.io/github/last-commit/rotsl/weather-ml/main?label=Dataset&logo=databricks)
![Model Health](https://img.shields.io/badge/Model-Healthy-green)
[![CHIRPS Dataset](https://img.shields.io/badge/CHIRPS-UCSB--CHG%2FCHIRPS%2FDAILY-1f7a8c)](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-34a853)
![CHIRPS Pipeline](https://img.shields.io/badge/CHIRPS%20Pipeline-Additive%20%26%20Fallback%20Safe-2a9d8f)

[![npm version](https://img.shields.io/npm/v/weather-ml-edge.svg)](https://www.npmjs.com/package/weather-ml-edge)
[![npm downloads](https://img.shields.io/npm/dm/weather-ml-edge.svg)](https://www.npmjs.com/package/weather-ml-edge)

---
[![DOI](https://zenodo.org/badge/1163996139.svg)](https://doi.org/10.5281/zenodo.18738567)
---
## 🪟 Automated Rain-Responsive Shutter System (Raspberry Pi)

This project n includes a physical automation layer that controls  shutters for massive areacnut drying areas using a Raspberry Pi and the trained ML model.

### Key Capabilities

- 🤖 ML-based rain prediction (6h horizon)
- ⏱️ Automatic shutter closing ~10 minutes before rain
- ☀️ Automatic reopening when safe
- 🔄 Uses latest model from GitHub
- 🖐️ Manual override via hardware button
- 💾 Persistent state recovery after power loss

### Control Logic

| Condition | Action |
|-----------|---------|
| Rain probability ≥ threshold | Close shutters |
| Rain probability < safe margin | Open shutters |
| Manual button press | Override to open |

No external API calls are made from the hardware device.

---

Hourly rain forecasting project using Visual Crossing weather data and scikit-learn gradient boosting models for multiple prediction horizons.

## 📁 Project Structure

```text
weather-ml/
├── .env                         # Local secrets (not committed)
├── .env.example                 # Safe example variables (no secrets)
├── README.md
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── config/
│   └── chirps_config.yaml       # Non-secret CHIRPS defaults
├── docs/                        # GitHub Pages dashboard
│   └── index.html
├── data/
│   ├── raw/
│   │   └── chirps_daily.csv
│   └── processed/
│       └── weather_hourly_clean.csv
│       ├── chirps_features_daily.csv
│       └── weather_hourly_clean_enriched.csv
├── models/
│   ├── hgb_*_current.pkl
│   ├── hgb_*_previous.pkl
│   ├── *_meta.json
│   ├── history/
│   │   └── metrics_history.csv
│   └── snapshots/
├── hardware/
│   ├── shutter_controller.py
│   ├── gpio_config.py
│   └── state.txt
├── notebooks/
├── scripts/
│   ├── update_data_and_retrain.py
│   ├── run_full_training_pipeline.py
│   ├── run_chirps_pipeline.py
│   ├── fetch_chirps_history.py
│   ├── build_chirps_features.py
│   └── enrich_training_data_with_chirps.py
└── weather/                     # Local virtual environment              
```

## Environment Setup

1. Activate your project environment (for example `./Scikit` if you are using that venv).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add runtime secrets to `.env` (local only, never commit):

```env
VISUAL_CROSSING_KEY=your_key_here
VISUAL_CROSSING_LOCATION=latitude,longitude
LOCATION_LAT=your_latitude
LOCATION_LON=your_longitude
EE_PROJECT=your_gcp_project_id
```

4. (Optional but recommended for CHIRPS) authenticate Earth Engine locally once:

```bash
earthengine authenticate
```

## Pipeline (Run Order)

### Baseline (unchanged)

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

### Full retraining pipeline (recommended, CHIRPS + fallback-safe)

```bash
python3 scripts/run_full_training_pipeline.py
```

### GitHub Actions Secrets (for CHIRPS in CI)

Configure these repository secrets:

- `VISUAL_CROSSING_KEY`
- `VISUAL_CROSSING_LOCATION`
- `LOCATION_LAT`
- `LOCATION_LON`
- `EE_PROJECT`
- `GOOGLE_CREDENTIALS` (full Google service-account JSON)

### CHIRPS-only pipeline (standalone)

```bash
python3 scripts/run_chirps_pipeline.py
```

This executes:

1. `scripts/fetch_chirps_history.py`
2. `scripts/build_chirps_features.py`
3. `scripts/enrich_training_data_with_chirps.py`

### Edge npm package (`weather-ml-edge`)

- Version `1.0.7` adds CHIRPS-aware model metadata support.
- CLI output now includes CHIRPS training status and CHIRPS feature count from model metadata.
- The package still reads runtime secrets from environment (`.env` locally or CI secrets) and does not embed credentials.

```bash
npm install -g weather-ml-edge
weather-ml
```


## 🛠️ Hardware Setup (Raspberry Pi Shutter Control)

> 🔩 For full details for hardware refer [Hardware_Setup](https://github.com/rotsl/weather-ml/blob/main/hardware/hardware_setup.md)

### Requirements

- Raspberry Pi (3/4/5)
- Servo motor or relay module
- Manual override button
- 5V power supply
- GPIO wiring

### Installation (On Raspberry Pi)

```bash
sudo apt update
sudo apt install python3-pigpio pigpio
pip install gpiozero pandas joblib
```

- Enable GPIO daemon:
```
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

- Run controller:

```
python hardware/shutter_controller.py
```

- Auto-Start on Boot

The controller runs as a systemd service:

```
sudo systemctl status weather-shutter
```

Shutters will operate automatically using the latest trained model.

---
## Key Outputs

- Clean dataset: `data/processed/weather_hourly_clean.csv`
- CHIRPS daily cache: `data/raw/chirps_daily.csv`
- CHIRPS daily features: `data/processed/chirps_features_daily.csv`
- Enriched training dataset: `data/processed/weather_hourly_clean_enriched.csv`
- Trained model artifacts and metadata: `models/*.pkl`, `models/*_meta.json`
- Horizon metrics table: `models/metrics_multihorizon.csv`
---

## Notes

- Training and inference use cloud + edge hybrid architecture.
- Secrets are read from `.env` locally and GitHub Secrets in CI.
- API keys and location coordinates are not hardcoded in tracked source.
- Hardware devices never expose API keys.
- Raspberry Pi fetches models from GitHub.
- Public dashboard performs no external requests.
- System is quota-safe for free-tier usage.

---

<!-- AUTO_STATUS_START -->
## 📊 Live Model Status (Auto-Updated)

| Field | Value |
|-------|-------|
| Last retrain (UTC) | 2026-03-27T04:34:37.248923 |
| Active horizon | D_next_6h (6h) |
| Dataset rows | 18,360 |
| Data range | 2024-02-22 --> 2026-03-27 |
| ROC-AUC | 0.8078 |
| PR-AUC | 0.5860 |
| Positive rate | 0.3105 |
| Features used | 34 |
| CHIRPS training | Enabled |
| CHIRPS feature count | 17 |
| CHIRPS raw rows | 0 |
| CHIRPS feature rows | 16,467 |
| CHIRPS enriched rows | 18,360 |

_Last updated automatically by GitHub Actions._
<!-- AUTO_STATUS_END -->


## 🛡️ Safety & Fail-Safes

The shutter system includes multiple safety mechanisms:

| Feature | Purpose |
|---------|----------|
| State file | Restores last position after reboot |
| Manual override | Immediate user control |
| Hysteresis | Prevents rapid toggling |
| Model fallback | Uses previous model if current fails |
| Watchdog restart | Auto-restart on crash |

Future enhancements include rain sensors and limit switches.


---

<!-- AUTO_DASHBOARD_START -->
## 📊 Live ML Dashboard (Auto-Updated)

### 🧠 Model

| Field | Value |
|-------|-------|
| Horizon | D_next_6h (6h) |
| Last trained | 2026-03-27T04:34:37.248923 |
| Features | 34 |
| CHIRPS features | 17 |
| Positive rate | 0.3105 |

---

### 📉 Performance

| Metric | Latest | Trend |
|--------|--------|-------|
| ROC-AUC | 0.8078 | ▇▇▇▇▆▆▅▃▃▁▁▁ |
| PR-AUC | 0.5860 | ▇▇▇▅▇▇▆▅▅▃▂▁ |

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
| Rows | 18,360 |
| Range | 2024-02-22 --> 2026-03-27 |

_Last updated: 2026-03-27 04:34 UTC_
<!-- AUTO_DASHBOARD_END -->


## 🔁 End-to-End System Flow

```text
Visual Crossing API
        ↓
CHIRPS (Google Earth Engine)
        ↓
GitHub Actions (48h)
        ↓
Data Cleaning
        ↓
CHIRPS Feature Enrichment
        ↓
Model Training
        ↓
Model Commit
        ↓
Raspberry Pi (git pull)
        ↓
Live Prediction
        ↓
Shutter Control

```

---
## 📜 License 
This project is licensed under the MIT License. 
See LICENSE for details. 

--- 

## 🤝 Contributing 
See CONTRIBUTING.md for guidelines.

--- 
## 🔐 Security 

See SECURITY.md for reporting vulnerabilities.

---

<div align="center">

### Weather ML

*Don't let those  crops get wet*

**Minimal • Automation • ML**

[GitHub](https://github.com/rotsl/weather-ml) • [GitLab](https://gitlab.com/rotsl/weather-ml) • [Dashboard ](https://rotsl.github.io/weather-ml/)

Built by **@rotsl** 💙

</div>

---
