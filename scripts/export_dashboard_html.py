import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from string import Template


# =====================================================
# Paths
# =====================================================

MODELS = Path("models")
HISTORY = MODELS / "history" / "metrics_history.csv"
DATA = Path("data/processed/weather_hourly_clean.csv")
OUT = Path("docs/index.html")


# =====================================================
# Load metadata
# =====================================================

meta_file = list(MODELS.glob("*_current_meta.json"))[0]
meta = json.loads(meta_file.read_text())


# =====================================================
# Load history
# =====================================================

hist = pd.read_csv(HISTORY, parse_dates=["timestamp"])
hist["t"] = hist["timestamp"].dt.strftime("%Y-%m-%d")

roc_vals = hist["roc_auc"].round(4).tolist()
pr_vals = hist["pr_auc"].round(4).tolist()
dates = hist["t"].tolist()

latest = hist.iloc[-1]
prev = hist.iloc[-2] if len(hist) > 1 else None


# =====================================================
# Dataset info
# =====================================================

df = pd.read_csv(DATA, parse_dates=["datetime"])
rows = len(df)
start = df["datetime"].min().date()
end = df["datetime"].max().date()


# =====================================================
# Degradation
# =====================================================

warning = "Healthy"

if prev is not None:
    if latest.roc_auc < prev.roc_auc - 0.05:
        warning = "ROC degradation"

    if latest.pr_auc < prev.pr_auc - 0.05:
        warning += " + PR degradation"

status_class = "ok" if warning == "Healthy" else "warn"


# =====================================================
# Metrics Table HTML
# =====================================================

rows_html = ""
for d, r, p in zip(dates, roc_vals, pr_vals):
    rows_html += f"""
<tr>
<td>{d}</td>
<td>{r:.4f}</td>
<td>{p:.4f}</td>
</tr>
"""


# =====================================================
# HTML Template (SAFE)
# =====================================================

template = Template("""
<!DOCTYPE html>
<html>

<head>
<meta charset="utf-8">
<title>Weather ML Dashboard</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body {
  font-family: system-ui, Arial;
  background: #fafafa;
  max-width: 1100px;
  margin: auto;
  padding: 20px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit,minmax(300px,1fr));
  gap: 15px;
}

.card {
  background: white;
  border-radius: 10px;
  padding: 15px;
  border: 1px solid #ddd;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.ok { color: green; }
.warn { color: red; }

.hidden { display: none; }

</style>

</head>

<body>

<h1>🌧️ Weather ML Dashboard</h1>

<small>
Last updated: $updated | Static / No API calls
</small>

<div class="grid">

<div class="card">
<h2>🧠 Model</h2>
<ul>
<li>Horizon: $horizon ($hours h)</li>
<li>Features: $n_features</li>
<li>Positive rate: $pos_rate</li>
<li>Last trained: $trained</li>
</ul>
</div>

<div class="card">
<h2>🚨 Health</h2>
<p class="$status_class">$warning</p>
</div>

<div class="card">
<h2>📁 Dataset</h2>
<ul>
<li>Rows: $rows</li>
<li>Range: $start → $end</li>
</ul>
</div>

</div>

<div class="card">
<h2>📈 Performance History</h2>
<canvas id="perfChart"></canvas>
</div>

<div class="card hidden" id="rawdata">
<h2>📄 Metrics Table</h2>
<table border="1" cellpadding="6">
<tr><th>Date</th><th>ROC-AUC</th><th>PR-AUC</th></tr>
$table_rows
</table>
</div>

<script>

const dates = $dates;
const roc = $roc;
const pr = $pr;

const ctx = document.getElementById("perfChart");

new Chart(ctx, {
  type: "line",
  data: {
    labels: dates,
    datasets: [
      { label: "ROC-AUC", data: roc, borderColor: "#1976d2", tension: 0.3 },
      { label: "PR-AUC", data: pr, borderColor: "#2e7d32", tension: 0.3 }
    ]
  },
  options: {
    responsive: true,
    scales: { y: { min: 0, max: 1 } }
  }
});

</script>

</body>
</html>
""")


# =====================================================
# Render HTML
# =====================================================

html = template.substitute(
    updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    horizon=meta["horizon"],
    hours=meta["hours"],
    n_features=len(meta["features"]),
    pos_rate=f"{meta['positive_rate']:.4f}",
    trained=meta["trained_at"],
    status_class=status_class,
    warning=warning,
    rows=f"{rows:,}",
    start=start,
    end=end,
    table_rows=rows_html,
    dates=json.dumps(dates),
    roc=json.dumps(roc_vals),
    pr=json.dumps(pr_vals),
)


OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html)

print("✅ Exported dashboard to docs/index.html")