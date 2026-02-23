
# Raspberry Pi Hardware Setup Guide
## Automated Rain-Responsive Shutter System

This document explains how to deploy the weather-ml shutter automation system on a Raspberry Pi.

It enables automatic window shutter control based on machine learning rain forecasts generated in this repository.

---

## 1. Supported Hardware

### Recommended

| Component | Specification |
|-----------|---------------|
| Raspberry Pi | 3B+, 4B, or 5 |
| OS | Raspberry Pi OS (64-bit) |
| Storage | ≥16 GB microSD |
| Servo / Relay | 5V compatible |
| Button | Momentary push button |
| Power | 5V / 3A PSU |

### Optional

- Rain sensor (for redundancy)
- Limit switches (position safety)
- UPS HAT (power backup)

---

## 2. Operating System Installation

### Install Raspberry Pi OS

1. Download Raspberry Pi Imager  
   https://www.raspberrypi.com/software/

2. Flash **Raspberry Pi OS (64-bit)**

3. Enable before flashing:
   - ✅ SSH
   - ✅ Wi-Fi
   - ✅ Username/password

4. Boot Pi and login

---

## 3. System Preparation

Update system:

```bash
sudo apt update && sudo apt upgrade -y
````

Install system packages:

```bash
sudo apt install -y \
  git \
  python3-pip \
  python3-venv \
  pigpio \
  i2c-tools
```

Enable GPIO daemon:

```bash
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

Verify:

```bash
sudo systemctl status pigpiod
```

---

## 4. Clone Repository

```bash
cd ~
git clone https://github.com/rotsl/weather-ml.git
cd weather-ml
```

---

## 5. Python Environment

Create virtual environment:

```bash
python3 -m venv weather
source weather/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install gpiozero pigpio joblib pandas numpy
```

---

## 6. Environment Variables

Create `.env` file:

```bash
nano .env
```

Add:

```env
VISUAL_CROSSING_KEY=your_api_key_here
VISUAL_CROSSING_LOCATION=lat,lon
```

⚠️ This file must never be committed.

---

## 7. GPIO Wiring

### Servo / Relay Connection

| Function | GPIO   | Pin    |
| -------- | ------ | ------ |
| Control  | GPIO18 | Pin 12 |
| Power    | 5V     | Pin 2  |
| Ground   | GND    | Pin 6  |

### Manual Override Button

| Function | GPIO   | Pin    |
| -------- | ------ | ------ |
| Button   | GPIO23 | Pin 16 |
| GND      | GND    | Pin 14 |

Enable pull-up in software.

---

## 8. Hardware Directory

Ensure directory exists:

```bash
mkdir -p hardware
```

Required files:

```
hardware/
├── shutter_controller.py
├── gpio_config.py
└── state.txt
```

---

## 9. Download Latest Models

Initial sync:

```bash
git pull origin main
```

Verify:

```bash
ls models/*_current.pkl
```

---

## 10. Test Hardware Manually

Run controller in debug mode:

```bash
python hardware/shutter_controller.py --debug
```

Expected:

* Servo moves
* Button toggles
* Logs appear

Stop with `Ctrl+C`.

---

## 11. Auto-Start Service (Systemd)

### Create Service

```bash
sudo nano /etc/systemd/system/weather-shutter.service
```

Paste:

```ini
[Unit]
Description=Weather ML Shutter Controller
After=network.target pigpiod.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/weather-ml
ExecStart=/home/pi/weather-ml/weather/bin/python hardware/shutter_controller.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Adjust `User` if needed.

---

### Enable Service

```bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

sudo systemctl enable weather-shutter
sudo systemctl start weather-shutter
```

Check status:

```bash
sudo systemctl status weather-shutter
```

View logs:

```bash
journalctl -u weather-shutter -f
```

---

## 12. Automatic Updates

Enable auto-pull via cron:

```bash
crontab -e
```

Add:

```cron
0 */6 * * * cd /home/pi/weather-ml && git pull && systemctl restart weather-shutter
```

Updates every 6 hours.

---

## 13. Safety & Recovery

### State File

Controller stores last position:

```
hardware/state.txt
```

Used after reboot.

### Watchdog

Systemd auto-restarts on crash.

### Manual Override

Button always overrides ML.

---

## 14. Power Protection (Recommended)

Use:

* UPS HAT
* Surge protector
* SD card backup

Backup:

```bash
rsync -av ~/weather-ml ~/backup/weather-ml
```

---

## 15. Troubleshooting

### Servo Not Moving

* Check GPIO pin
* Check power
* Check pigpio

```bash
pigs t 18 1500
```

### Service Won't Start

```bash
journalctl -xe
```

### No Model Found

```bash
git pull
ls models/
```

### Permission Errors

```bash
sudo chown -R pi:pi ~/weather-ml
```

---

## 16. Maintenance Checklist

Monthly:

* Check logs
* Inspect wiring
* Test manual override
* Verify models updated
* Clean dust

---

## 17. Future Extensions

Planned upgrades:

* Rain sensor validation
* MQTT integration
* Mobile alerts
* Camera verification
* Battery monitoring

---

## 18. Support

For issues:

Open GitHub Issue:

[https://github.com/rotsl/weather-ml/issues](https://github.com/rotsl/weather-ml/issues)

Include:

* Logs
* Hardware model
* OS version
* Photos (if wiring-related)

---

© weather-ml | rotsl | 2026

---


