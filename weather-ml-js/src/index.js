#!/usr/bin/env node

import { loadConfig } from "./config.js";
import { getWeather } from "./predictor.js";

async function main() {
  const cfg = loadConfig();

  const cur = await getWeather(cfg.apiKey, cfg.location);

  console.log("🌧️ Weather Status");
  console.log("Temp:", cur.temp);
  console.log("Rain %:", cur.precipprob);
  console.log("Humidity:", cur.humidity);
}

main().catch(err => {
  console.error("Error:", err.message);
  process.exit(1);
});