import axios from "axios";
import fs from "fs";

const DEFAULT_MODEL_URL =
  "https://raw.githubusercontent.com/rotsl/weather-ml/main/models/hgb_D_next_6h_current.pkl";

const DEFAULT_MODEL_META_URL =
  "https://raw.githubusercontent.com/rotsl/weather-ml/main/models/hgb_D_next_6h_current_meta.json";

export async function loadModel(modelUrl = DEFAULT_MODEL_URL) {
  const res = await axios.get(modelUrl, { responseType: "arraybuffer" });

  fs.writeFileSync("model.pkl", res.data);

  return "model.pkl";
}

export async function loadModelMeta(metaUrl = DEFAULT_MODEL_META_URL) {
  const res = await axios.get(metaUrl, { responseType: "json" });
  return res.data;
}

export function getChirpsStatus(meta) {
  const features = Array.isArray(meta?.features) ? meta.features : [];
  const chirpsFeatures = features.filter(
    feature => typeof feature === "string" && feature.startsWith("chirps_")
  );

  return {
    enabled: chirpsFeatures.length > 0,
    count: chirpsFeatures.length,
    featureNames: chirpsFeatures,
    horizon: meta?.horizon || "unknown",
    trainedAt: meta?.trained_at || "unknown"
  };
}
