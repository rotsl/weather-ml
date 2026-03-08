# 🌧️ weather-ml-edge

Edge client for ML-based rain prediction and automated shutter control.

This package connects local devices (Raspberry Pi, servers, PCs) to the
**weather-ml** machine learning system and enables real-time weather monitoring
and automation — without exposing private API keys or location data.

---

## ✨ Features

- 🤖 ML-based rain probability monitoring
- 🪟 Smart shutter / relay control support
- ⚡ Lightweight CLI tool
- 🔐 Secure environment-based configuration
- 🌍 Uses your own Visual Crossing account
- 🌧️ CHIRPS-aware model metadata status (from upstream trained model)
- 📦 No bundled datasets or credentials
- 🖥️ Designed for edge / IoT deployment

---

## 🔒 Security Notice

> ⚠️ This package **does NOT provide** API access.

You must supply:

- Your own Visual Crossing API key
- Your own location

The author does **not** proxy, store, or distribute credentials.

All API usage is billed to your own account.

---

## 📦 Installation

### Global Install (Recommended)

```bash
npm install -g weather-ml-edge
````

### Local Install

```bash
npm install weather-ml-edge
```

---

## ⚙️ Configuration

Create a `.env` file in your project directory:

```env
VISUAL_CROSSING_KEY=your_api_key_here
VISUAL_CROSSING_LOCATION=lat,lon
WEATHER_ML_MODEL_META_URL=https://raw.githubusercontent.com/rotsl/weather-ml/main/models/hgb_D_next_6h_current_meta.json
```

Example:

```env
VISUAL_CROSSING_KEY=abc123xyz
VISUAL_CROSSING_LOCATION=lat,lon
```

`WEATHER_ML_MODEL_META_URL` is optional; the default points to the main `weather-ml` model metadata artifact.

Never commit this file.

---

## 🚀 Usage

### CLI Mode

After installing globally:

```bash
weather-ml
```

Output:

```text
🌧️ Weather Status
Temp: 26.4°C
Rain %: 12
Humidity: 81
CHIRPS training: Enabled
CHIRPS features: 17
Model horizon: D_next_6h
Model trained at: 2026-03-08T11:15:46.848136
```

---

### Programmatic Usage

```js
import { getChirpsStatus, getWeather, loadConfig, loadModelMeta } from "weather-ml-edge";

const cfg = loadConfig();

const weather = await getWeather(cfg.apiKey, cfg.location);
const meta = await loadModelMeta(cfg.modelMetaUrl);
const chirps = getChirpsStatus(meta);

console.log(weather.precipprob);
console.log(chirps.enabled, chirps.count);
```

---

## 🪟 Raspberry Pi Integration

Designed for use with:

* Servo motors
* Relay boards
* GPIO buttons
* Smart window systems

Typical setup:

```text
Prediction → Threshold → GPIO → Motor
```

Example:

```js
if (rainProb > 0.5) {
  closeShutters();
}
```

See main project:

[https://github.com/rotsl/weather-ml](https://github.com/rotsl/weather-ml)

---

## 📊 Data Sources

Weather data provided by:

* Visual Crossing Weather API
  [https://www.visualcrossing.com/](https://www.visualcrossing.com/)

Model metadata may include CHIRPS-derived features from:

* CHIRPS Daily (Google Earth Engine)
  [https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)

Subject to their terms and pricing.

---

## 📁 Project Structure

```text
weather-ml-edge/
├── src/
│   ├── index.js
│   ├── predictor.js
│   ├── modelLoader.js
│   └── config.js
├── package.json
└── README.md
```

---

## 🛠️ Requirements

* Node.js ≥ 18
* Internet connection
* Visual Crossing API account

---

## 🚧 Limitations

* No offline forecasting
* Requires external API
* CHIRPS data is not downloaded directly by this package
* No bundled ML models
* No centralized service

This package is a client, not a server.

---

## 🗺️ Roadmap

Planned features:

* MQTT output
* Home Assistant integration
* Camera verification
* Local caching
* Edge retraining hooks
* Mobile notifications

---

## 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork repository
2. Create feature branch
3. Submit PR

See main project for guidelines.

---

## 📜 License

MIT License

See `LICENSE` for details.

---

## 👤 Author

**Rohan (rotsl)**

GitHub: [https://github.com/rotsl](https://github.com/rotsl)
Project: [https://github.com/rotsl/weather-ml](https://github.com/rotsl/weather-ml)

---

## 📄 License

MIT © [Rohan R.](https://github.com/rotsl)

--- 

## 🌱 Philosophy

> "Build systems that predict, adapt, and protect — automatically."

Minimal • Secure • Edge-first • ML-powered


---
