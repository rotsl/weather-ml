import pandas as pd
from pathlib import Path

# ---------------------------------
# Paths
# ---------------------------------
RAW_PATH = Path("data/raw/visual_crossing_2024_2026_hourly.csv")
OUT_PATH = Path("data/processed/weather_hourly_clean.csv")

# ---------------------------------
# Load
# ---------------------------------
print("Loading raw data...")
df = pd.read_csv(RAW_PATH)

print(f"Raw shape: {df.shape}")

# ---------------------------------
# Parse + normalize datetime
# ---------------------------------
print("Parsing datetime...")

df["datetime"] = pd.to_datetime(df["datetime"])

# Force exact hourly alignment
df["datetime"] = df["datetime"].dt.floor("h")

# ---------------------------------
# Sort + remove duplicates
# ---------------------------------
print("Sorting and removing duplicates...")

df = df.sort_values("datetime")
df = df.drop_duplicates(subset="datetime")

print(f"After dedup: {df.shape}")

# ---------------------------------
# Build full hourly index
# ---------------------------------
print("Building full hourly timeline...")

full_index = pd.date_range(
    start=df["datetime"].min(),
    end=df["datetime"].max(),
    freq="h"   # new pandas standard
)

df = df.set_index("datetime")
df = df.reindex(full_index)

df.index.name = "datetime"

# ---------------------------------
# Check missing hours
# ---------------------------------
missing_rows = df.isna().any(axis=1).sum()

print(f"Missing hours detected: {missing_rows}")

# ---------------------------------
# Fill numeric columns (time interp)
# ---------------------------------
print("Interpolating numeric values...")

numeric_cols = df.select_dtypes(include="number").columns

df[numeric_cols] = df[numeric_cols].interpolate(
    method="time",
    limit_direction="both"
)

# ---------------------------------
# Fill categorical columns (ffill/bfill)
# ---------------------------------
print("Filling categorical values...")

cat_cols = df.select_dtypes(exclude="number").columns

df[cat_cols] = df[cat_cols].ffill().bfill()

# ---------------------------------
# Rain variable
# ---------------------------------
print("Creating rain labels...")

if "precip" not in df.columns:
    raise ValueError("Column 'precip' not found!")

df["rain_1h"] = (df["precip"] >= 0.1).astype(int)

# ---------------------------------
# Rolling features
# ---------------------------------
print("Creating rolling rain features...")

df["rain_6h"] = df["precip"].rolling(6, min_periods=1).sum()
df["rain_24h"] = df["precip"].rolling(24, min_periods=1).sum()

# ---------------------------------
# Reset index
# ---------------------------------
df = df.reset_index()

# ---------------------------------
# Final validation
# ---------------------------------
print("Final validation...")

assert df["datetime"].is_monotonic_increasing

null_count = df.isna().sum().sum()
print(f"Total NaNs: {null_count}")

assert null_count == 0

# ---------------------------------
# Save
# ---------------------------------
print("Saving clean dataset...")

df.to_csv(OUT_PATH, index=False)

print("\nCleaning complete.")
print(f"Saved to: {OUT_PATH}")
print(f"Final shape: {df.shape}")