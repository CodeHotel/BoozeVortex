#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

from controller import Controller

BASE = Path(__file__).resolve().parent
app = FastAPI()
ctl = Controller(BASE)

# Simple single-page UI (no build tools)
INDEX_HTML = Template(r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Soju Vortex BO</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system; margin: 20px; }
    .row { display:flex; gap:12px; align-items:center; margin-bottom: 12px; flex-wrap: wrap; }
    button { padding: 8px 12px; border-radius: 8px; border: 1px solid #ccc; cursor:pointer; }
    button.start { background: #e7fff0; }
    button.pause { background: #fff7e6; }
    button.stop { background: #ffecec; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; margin: 10px 0; }
    .badge { padding: 3px 8px; border-radius: 999px; font-size: 12px; border:1px solid #ddd; }
    .ok { color: #0a7; }
    .warn { color: #b80; }
    .bad { color: #c33; }
    pre { background:#0b1020; color:#e6e6e6; padding: 10px; border-radius: 12px; overflow:auto; max-height: 240px; }
  </style>
</head>
<body>
  <h2>Soju Vortex — Bayesian Optimization</h2>

  <div class="row">
    <button class="start" onclick="cmd('/start')">Start / Resume</button>
    <button class="pause" onclick="cmd('/pause')">Pause</button>
    <button class="stop" onclick="cmd('/force_stop')">Force Stop</button>
    <span id="statusLine" class="badge">loading...</span>
  </div>

  <div class="card">
    <div><b>Machines</b></div>
    <div id="machines"></div>
  </div>

  <div class="card">
    <div><b>Learning</b></div>
    <div id="learning"></div>
    <canvas id="chart" height="120"></canvas>
  </div>

  <div class="card">
    <div><b>Logs</b></div>
    <pre id="logs"></pre>
  </div>

<script>
let chart = null;

async function cmd(path) {
  await fetch(path, {method:'POST'});
  await refresh();
}

async function refresh() {
  const st = await (await fetch('/status')).json();
  const plot = await (await fetch('/plot')).json();
  const logs = await (await fetch('/logs')).json();

  const paused = st.paused;
  const stopping = st.stopping;
  const badge = document.getElementById('statusLine');
  badge.textContent = stopping ? "STOPPING" : (paused ? "PAUSED" : "RUNNING");
  badge.className = "badge " + (stopping ? "bad" : (paused ? "warn" : "ok"));

  // machines
  const mdiv = document.getElementById('machines');
  let html = "";
  for (const m of st.machines) {
    const running = (st.running[m] || []).join(", ");
    html += `<div>• <b>${m}</b> — running: ${running || "(idle)"}</div>`;
  }
  mdiv.innerHTML = html;

  // learning
  const ldiv = document.getElementById('learning');
  ldiv.innerHTML = `evals: <b>${st.n_evals}</b> &nbsp; best: <b>${(st.best ?? "—")}</b>`;

  // chart
  const ctx = document.getElementById('chart').getContext('2d');
  const labels = plot.iters;
  const data = plot.scores;

  if (!chart) {
    chart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets: [{ label: 'E_water', data }]},
      options: { responsive:true, animation:false }
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
  }

  // logs
  document.getElementById('logs').textContent = logs.lines.join("\n");
}

// plot endpoint expects results.csv exists; if not, handle empty
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>
""")

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML.render()

@app.post("/start")
def start():
    ctl.start()
    return JSONResponse({"ok": True})

@app.post("/pause")
def pause():
    ctl.pause()
    return JSONResponse({"ok": True})

@app.post("/resume")
def resume():
    ctl.resume()
    return JSONResponse({"ok": True})

@app.post("/force_stop")
def force_stop():
    ctl.force_stop()
    return JSONResponse({"ok": True})

@app.get("/status")
def status():
    return JSONResponse(ctl.status())

@app.get("/logs")
def logs():
    return JSONResponse({"lines": ctl.get_logs()})

@app.get("/plot")
def plot():
    # read results.csv if exists
    import pandas as pd
    path = BASE / "results.csv"
    if not path.exists():
        return JSONResponse({"iters": [], "scores": []})
    df = pd.read_csv(path)
    scores = df["score"].astype(float).tolist()
    iters = list(range(1, len(scores) + 1))
    return JSONResponse({"iters": iters, "scores": scores})

