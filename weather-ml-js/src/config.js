import dotenv from "dotenv";

dotenv.config();

export function loadConfig() {
  const key = process.env.VISUAL_CROSSING_KEY;
  const location = process.env.VISUAL_CROSSING_LOCATION;

  if (!key) throw new Error("Missing VISUAL_CROSSING_KEY");
  if (!location) throw new Error("Missing VISUAL_CROSSING_LOCATION");

  return {
    apiKey: key,
    location
  };
}