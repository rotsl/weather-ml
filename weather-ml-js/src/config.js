import dotenv from "dotenv";

dotenv.config();

export function loadConfig() {
  const key = process.env.VISUAL_CROSSING_KEY;
  const location = process.env.VISUAL_CROSSING_LOCATION;
  const modelMetaUrl =
    process.env.WEATHER_ML_MODEL_META_URL ||
    "https://raw.githubusercontent.com/rotsl/weather-ml/main/models/hgb_D_next_6h_current_meta.json";

  if (!key) throw new Error("Missing VISUAL_CROSSING_KEY");
  if (!location) throw new Error("Missing VISUAL_CROSSING_LOCATION");

  return {
    apiKey: key,
    location,
    modelMetaUrl
  };
}
