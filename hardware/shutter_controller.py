import time
import os
import joblib
import pandas as pd
from datetime import datetime

from hardware.gpio_config import *


# ================================
# CONFIG
# ================================

MODEL_PATH = "models/hgb_D_next_6h_current.pkl"
DATA_PATH = "data/processed/weather_hourly_clean.csv"

RAIN_THRESHOLD = 0.40
CHECK_INTERVAL = 300   # seconds (5 min)

STATE_FILE = "hardware/state.txt"


# ================================
# GPIO SETUP
# ================================

if MODE == "servo":

    from gpiozero import Servo
    from gpiozero.pins.pigpio import PiGPIOFactory

    factory = PiGPIOFactory()
    servo = Servo(SERVO_PIN, pin_factory=factory)


elif MODE == "relay":

    from gpiozero import OutputDevice

    relay_open = OutputDevice(RELAY_OPEN_PIN)
    relay_close = OutputDevice(RELAY_CLOSE_PIN)


from gpiozero import Button
button = Button(BUTTON_PIN)


# ================================
# SHUTTER ACTIONS
# ================================

def open_shutter():

    print("☀️ Opening shutter")

    if MODE == "servo":
        servo.max()
        time.sleep(SERVO_MOVE_SEC)
        servo.detach()

    else:
        relay_open.on()
        time.sleep(MOTOR_RUN_SEC)
        relay_open.off()


def close_shutter():

    print("🌧️ Closing shutter")

    if MODE == "servo":
        servo.min()
        time.sleep(SERVO_MOVE_SEC)
        servo.detach()

    else:
        relay_close.on()
        time.sleep(MOTOR_RUN_SEC)
        relay_close.off()


# ================================
# STATE HANDLING
# ================================

def load_state():

    if os.path.exists(STATE_FILE):
        return open(STATE_FILE).read().strip()

    return "open"


def save_state(s):

    with open(STATE_FILE, "w") as f:
        f.write(s)


state = load_state()


# ================================
# MODEL
# ================================

print("Loading ML model...")
model = joblib.load(MODEL_PATH)

print("Model loaded")


# ================================
# FEATURES
# ================================

def get_latest_features():

    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

    latest = df.sort_values("datetime").iloc[-1]

    features = model.feature_names_in_

    X = latest[features].to_frame().T

    return X


# ================================
# MANUAL OVERRIDE
# ================================

def manual_open():

    global state

    print("⚠️ Manual override")

    open_shutter()
    state = "open"
    save_state(state)


button.when_pressed = manual_open


# ================================
# MAIN LOOP
# ================================

print("🚀 Shutter controller running")

while True:

    try:

        X = get_latest_features()

        prob = model.predict_proba(X)[0, 1]

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        print(f"[{now}] Rain prob: {prob:.3f}")


        # ===== Decision Logic =====

        if prob >= RAIN_THRESHOLD and state != "closed":

            close_shutter()
            state = "closed"
            save_state(state)


        elif prob < RAIN_THRESHOLD * 0.5 and state != "open":

            open_shutter()
            state = "open"
            save_state(state)


    except Exception as e:

        print("ERROR:", e)


    time.sleep(CHECK_INTERVAL)