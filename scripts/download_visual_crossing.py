import os
from datetime import date
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from io import StringIO
import urllib.parse

# ---------------------------------
# CONFIG
# ---------------------------------
START_DATE = date(2024, 2, 22)
END_DATE   = date(2026, 2, 22)

OUTPUT_PATH = Path("data/raw/visual_crossing_2024_2026_hourly.csv")

# ---------------------------------
# Load API key
# ---------------------------------
load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_KEY")
LOCATION = os.getenv("VISUAL_CROSSING_LOCATION")

if not API_KEY:
    raise ValueError("VISUAL_CROSSING_KEY not found in .env")
if not LOCATION:
    raise ValueError("VISUAL_CROSSING_LOCATION not found in .env")

# ---------------------------------
# Build URL
# ---------------------------------
url = (
    "https://weather.visualcrossing.com/"
    "VisualCrossingWebServices/rest/services/timeline/"
    f"{urllib.parse.quote_plus(LOCATION)}/{START_DATE.isoformat()}/{END_DATE.isoformat()}"
    "?include=hours"
    "&unitGroup=metric"
    f"&key={API_KEY}"
    "&contentType=csv"
)

print("Downloading full 2-year hourly dataset...")
print(f"Range: {START_DATE} → {END_DATE}")

# ---------------------------------
# Request
# ---------------------------------
response = requests.get(url, timeout=300)

if response.status_code != 200:
    print(f"HTTP Error {response.status_code}: Visual Crossing request failed")
    raise SystemExit(1)

# ---------------------------------
# Save CSV
# ---------------------------------
df = pd.read_csv(StringIO(response.text))

df.to_csv(OUTPUT_PATH, index=False)

print("\nDownload complete.")
print(f"Rows saved: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")
