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
```

Example:

```env
VISUAL_CROSSING_KEY=abc123xyz
VISUAL_CROSSING_LOCATION=lat,lon
```

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
```

---

### Programmatic Usage

```js
import { loadConfig } from "weather-ml-edge";
import { getWeather } from "weather-ml-edge";

const cfg = loadConfig();

const data = await getWeather(cfg.apiKey, cfg.location);

console.log(data.precipprob);
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
