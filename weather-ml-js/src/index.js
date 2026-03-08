#!/usr/bin/env node

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { loadConfig } from "./config.js";
import { getWeather } from "./predictor.js";
import { getChirpsStatus, loadModelMeta } from "./modelLoader.js";

export async function runCli() {
  const cfg = loadConfig();

  const cur = await getWeather(cfg.apiKey, cfg.location);
  let chirps = null;
  let chirpsError = null;

  try {
    const meta = await loadModelMeta(cfg.modelMetaUrl);
    chirps = getChirpsStatus(meta);
  } catch (err) {
    chirpsError = err;
  }

  console.log("🌧️ Weather Status");
  console.log("Temp:", cur.temp);
  console.log("Rain %:", cur.precipprob);
  console.log("Humidity:", cur.humidity);

  if (chirpsError) {
    console.log("CHIRPS model context:", "Unavailable");
  } else if (chirps) {
    console.log("CHIRPS training:", chirps.enabled ? "Enabled" : "Disabled");
    console.log("CHIRPS features:", chirps.count);
    console.log("Model horizon:", chirps.horizon);
    console.log("Model trained at:", chirps.trainedAt);
  }
}

const currentFilePath = fileURLToPath(import.meta.url);
const launchedScriptPath = process.argv[1] ? path.resolve(process.argv[1]) : "";
const launchedScriptRealPath = launchedScriptPath
  ? fs.realpathSync(launchedScriptPath)
  : "";
const currentFileRealPath = fs.realpathSync(currentFilePath);

if (launchedScriptRealPath === currentFileRealPath) {
  runCli().catch(err => {
    console.error("Error:", err.message);
    process.exit(1);
  });
}
