import axios from "axios";
import fs from "fs";

export async function loadModel() {
  const url =
    "https://raw.githubusercontent.com/rotsl/weather-ml/main/models/hgb_D_next_6h_current.pkl";

  const res = await axios.get(url, { responseType: "arraybuffer" });

  fs.writeFileSync("model.pkl", res.data);

  return "model.pkl";
}