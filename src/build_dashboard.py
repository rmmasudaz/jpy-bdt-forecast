#!/usr/bin/env python3
"""
build_dashboard.py — Generate a self-contained interactive HTML dashboard.

Reads model_forecast.json (produced by train_lstm.py) and writes dashboard.html
with the data embedded inline, so the file works standalone (double-click to
open; only Chart.js is pulled from a CDN).

Usage:
    python3 build_dashboard.py [--json model_forecast.json] [--out dashboard.html]
"""

from __future__ import annotations

import argparse
import json
import os

# --------------------------------------------------------------------------- #
# HTML/CSS/JS template. The token __MODEL_DATA__ is replaced with the JSON.
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>JPY → BDT · Exchange-Rate Forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0c1420; --bg2:#0f1927; --card:#121e30; --edge:#1c2b40;
    --ink:#e9f0f7; --muted:#8ba2b8; --faint:#5a728b;
    --teal:#2dd4bf; --amber:#f5b93f; --green:#3dd68c; --rose:#f36d8d;
    --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
    --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    --disp:"Space Grotesk",var(--sans);
    --r:14px;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0}
  body{
    background:
      radial-gradient(1200px 500px at 15% -10%, #12233a 0%, transparent 60%),
      radial-gradient(900px 420px at 90% 0%, #0f2a2b 0%, transparent 55%),
      var(--bg);
    color:var(--ink);
    font-family:var(--sans);
    font-size:15px;
    line-height:1.5;
    -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1180px;margin:0 auto;padding:28px 22px 60px}

  /* ---------- top bar ---------- */
  .topbar{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:26px}
  .brand{display:flex;align-items:center;gap:12px}
  .brand .dot{width:9px;height:9px;border-radius:50%;background:var(--teal);box-shadow:0 0 0 4px rgba(45,212,191,.14)}
  .brand h1{font-family:var(--disp);font-weight:600;font-size:19px;letter-spacing:.02em;margin:0}
  .brand h1 b{color:var(--teal)}
  .brand .tag{
    font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;color:#9fd6d0;
    border:1px solid rgba(45,212,191,.35);background:rgba(45,212,191,.08);
    padding:3px 8px;border-radius:999px;text-transform:uppercase;
  }
  .updated{font-family:var(--mono);font-size:11.5px;color:var(--faint)}

  /* ---------- hero spot rate ---------- */
  .hero{
    position:relative;overflow:hidden;
    background:linear-gradient(180deg,var(--bg2),var(--card));
    border:1px solid var(--edge);border-radius:var(--r);
    padding:26px 30px 22px;margin-bottom:16px;
    display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap;
  }
  .hero::after{
    content:"";position:absolute;right:-60px;top:-60px;width:260px;height:260px;
    background:radial-gradient(circle,rgba(45,212,191,.12),transparent 65%);
    pointer-events:none;
  }
  .hero .label{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--faint);margin-bottom:2px}
  .hero .rate{font-family:var(--mono);font-size:64px;font-weight:600;line-height:1;letter-spacing:-.03em}
  .hero .rate small{font-size:22px;color:var(--muted);font-weight:400;letter-spacing:.02em}
  .hero .pairline{font-family:var(--mono);font-size:13px;color:var(--muted);margin-top:10px}
  .hero .pairline b{color:var(--ink)}
  .hero .delta{display:flex;align-items:center;gap:10px;margin-top:12px}
  .hero .delta .pill{
    font-family:var(--mono);font-size:13px;font-weight:600;padding:5px 11px;border-radius:8px;
    background:rgba(61,214,140,.12);color:var(--green);border:1px solid rgba(61,214,140,.3);
  }
  .hero .delta .pill.down{background:rgba(243,109,141,.12);color:var(--rose);border-color:rgba(243,109,141,.3)}
  .hero .delta .note{font-size:12.5px;color:var(--faint)}
  .hero .spark{width:210px;height:56px}

  /* ---------- KPI cards ---------- */
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
  .kpi{background:var(--card);border:1px solid var(--edge);border-radius:var(--r);padding:16px 18px}
  .kpi .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--faint)}
  .kpi .v{font-family:var(--mono);font-size:24px;font-weight:600;margin-top:6px;letter-spacing:-.01em}
  .kpi .s{font-size:12px;color:var(--muted);margin-top:2px}
  .kpi .s b{color:var(--ink);font-weight:600}
  .kpi .s .up{color:var(--green)} .kpi .s .down{color:var(--rose)} .kpi .s .flat{color:var(--amber)}
  .kpi .bar{height:4px;border-radius:4px;background:var(--edge);margin-top:12px;overflow:hidden}
  .kpi .bar i{display:block;height:100%;border-radius:4px;background:var(--teal)}

  /* ---------- panels ---------- */
  .panel{background:var(--card);border:1px solid var(--edge);border-radius:var(--r);padding:18px 20px 14px;margin-bottom:16px}
  .panel h2{font-family:var(--disp);font-weight:600;font-size:14px;margin:0 0 2px;letter-spacing:.01em}
  .panel .sub{font-size:12.5px;color:var(--muted);margin-bottom:10px}
  .ranges{display:flex;gap:6px;margin-bottom:12px}
  .ranges button{
    font-family:var(--mono);font-size:11px;color:var(--muted);
    background:transparent;border:1px solid var(--edge);border-radius:7px;
    padding:4px 10px;cursor:pointer;transition:all .15s;
  }
  .ranges button:hover{color:var(--ink);border-color:var(--faint)}
  .ranges button.active{color:#0c1420;background:var(--teal);border-color:var(--teal);font-weight:600}
  .chartbox{position:relative;height:340px}
  .chartbox.tall{height:380px}
  .chartbox.short{height:300px}
  .grid2{display:grid;grid-template-columns:1.5fr 1fr;gap:16px}
  .grid2 .panel{margin-bottom:0}

  /* ---------- legend chips ---------- */
  .chips{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 10px}
  .chips span{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;color:var(--muted)}
  .chips i{width:9px;height:9px;border-radius:3px;display:inline-block}

  /* ---------- methodology ---------- */
  details.method{
    background:var(--card);border:1px solid var(--edge);border-radius:var(--r);
    padding:16px 20px;margin-top:4px;
  }
  details.method summary{
    cursor:pointer;font-family:var(--disp);font-weight:600;font-size:14px;list-style:none;
    display:flex;align-items:center;justify-content:space-between;
  }
  details.method summary::-webkit-details-marker{display:none}
  details.method summary .caret{transition:transform .2s;color:var(--faint)}
  details.method[open] summary .caret{transform:rotate(90deg)}
  details.method .body{display:grid;grid-template-columns:1fr 1fr;gap:6px 34px;margin-top:14px;font-size:13.5px;color:var(--muted)}
  details.method .body .col{min-width:0}
  details.method h4{font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--faint);margin:14px 0 6px}
  details.method h4:first-child{margin-top:0}
  details.method p{margin:0 0 8px}
  details.method code{font-family:var(--mono);font-size:12px;color:var(--teal);background:rgba(45,212,191,.07);padding:1px 5px;border-radius:5px}
  details.method .note{background:rgba(245,185,63,.07);border:1px solid rgba(245,185,63,.25);border-radius:10px;padding:10px 12px;color:#e8d9a8;margin-top:8px}
  .foot{margin-top:26px;text-align:center;font-family:var(--mono);font-size:11px;color:var(--faint)}
  .foot b{color:var(--muted);font-weight:500}

  @media (max-width:900px){
    .kpis{grid-template-columns:repeat(2,1fr)}
    .grid2{grid-template-columns:1fr}
    .hero .rate{font-size:46px}
    details.method .body{grid-template-columns:1fr}
  }
  @media (prefers-reduced-motion: reduce){
    *{animation:none!important;transition:none!important}
  }
  @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
  .hero,.kpi,.panel,details.method{animation:rise .5s ease both}
  .kpi:nth-child(2){animation-delay:.05s}.kpi:nth-child(3){animation-delay:.1s}.kpi:nth-child(4){animation-delay:.15s}
  .panel:nth-of-type(2){animation-delay:.1s}.panel:nth-of-type(3){animation-delay:.15s}
</style>
</head>
<body>
<div class="wrap">

  <!-- ===================== top bar ===================== -->
  <div class="topbar">
    <div class="brand">
      <span class="dot"></span>
      <h1>JPY → BDT <b>·</b> Exchange-Rate Forecast</h1>
      <span class="tag">structural FX model</span>
    </div>
    <div class="updated" id="updatedBar">loading…</div>
  </div>

  <!-- ===================== hero ===================== -->
  <div class="hero">
    <div>
      <div class="label">Spot rate · ৳ per ¥</div>
      <div class="rate" id="spotRate">—</div>
      <div class="pairline">Japanese Yen <b>→</b> Bangladeshi Taka</div>
      <div class="delta">
        <span class="pill" id="spotDelta">—</span>
        <span class="note" id="spotDeltaNote"></span>
      </div>
    </div>
    <canvas class="spark" id="spark"></canvas>
  </div>

  <!-- ===================== KPI cards ===================== -->
  <div class="kpis">
    <div class="kpi">
      <div class="k">7-day outlook</div>
      <div class="v" id="k7">—</div>
      <div class="s" id="k7b">—</div>
    </div>
    <div class="kpi">
      <div class="k">30-day outlook</div>
      <div class="v" id="k30">—</div>
      <div class="s" id="k30b">—</div>
    </div>
    <div class="kpi">
      <div class="k">Directional accuracy</div>
      <div class="v" id="kdir">—</div>
      <div class="s" id="kdirb">out-of-sample · coin flip = 50%</div>
      <div class="bar"><i id="bardir"></i></div>
    </div>
    <div class="kpi">
      <div class="k">1-step level error (MAE)</div>
      <div class="v" id="kmae">—</div>
      <div class="s" id="kmaeb">—</div>
      <div class="bar"><i id="barmae"></i></div>
    </div>
  </div>

  <!-- ===================== main forecast ===================== -->
  <div class="panel">
    <h2>Spot history &amp; 30-day projection</h2>
    <div class="sub">Structural model (taka-leg mean reversion) · shaded area = 95% interval, widening with horizon.</div>
    <div class="ranges" id="mainRanges">
      <button data-r="30">1M</button>
      <button data-r="90">3M</button>
      <button data-r="180">6M</button>
      <button data-r="365">1Y</button>
      <button data-r="0" class="active">ALL</button>
    </div>
    <div class="chartbox tall"><canvas id="mainChart"></canvas></div>
  </div>

  <!-- ===================== backtest + scatter ===================== -->
  <div class="grid2">
    <div class="panel">
      <h2>Out-of-sample backtest</h2>
      <div class="sub">Last <span id="btRangeLabel">182</span> days held out · 1-step-ahead structural forecasts.</div>
      <div class="ranges" id="btRanges">
        <button data-r="30">1M</button>
        <button data-r="60">2M</button>
        <button data-r="90">3M</button>
        <button data-r="0" class="active">ALL</button>
      </div>
      <div class="chartbox short"><canvas id="btChart"></canvas></div>
    </div>
    <div class="panel">
      <h2>Predicted vs. actual</h2>
      <div class="sub">1-step test predictions · points near the dashed line are accurate.</div>
      <div class="chartbox short"><canvas id="scatterChart"></canvas></div>
    </div>
  </div>

  <!-- ===================== methodology ===================== -->
  <details class="method">
    <summary>Methodology &amp; honest caveats <span class="caret">▸</span></summary>
    <div class="body">
      <div class="col">
        <h4>Data</h4>
        <p><code>data/jpy_bdt_daily.csv</code> — daily JPY→BDT rates (৳ per ¥) from the free
          <code>fawazahmed0/currency-api</code> service, <span id="mN"></span> trading days,
          <span id="mRange"></span>. Missing weekends/holidays are forward-filled.</p>

        <h4>Why returns, not the level</h4>
        <p>Exchange rates behave close to a random walk, so the best naïve forecast of
          tomorrow's rate is today's rate. A model trained on the raw level tends to
          smooth the series and looks <em>worse</em> than that trivial baseline. Instead the
          model predicts the daily <b>log-return</b> <code>r = ln(yₜ / yₜ₋₁)</code> — a stationary
          target — and levels are reconstructed cumulatively.</p>

        <h4>The model (and why it changed)</h4>
        <p>We started with an LSTM (<code>64 → Dropout → 32 → Dropout → Dense(1)</code>) on a
          <code><span id="mSeq"></span></code>-day window of returns — but its apparent
          directional edge (~60%) was <b>not stable</b>: the same config scored anywhere from
          ~40% to ~60% depending on the random seed. That is the signature of noise, not
          signal.</p>
        <p>Digging into the data showed why, through an exact identity:
          <code>JPY/BDT = (BDT/USD) ÷ (JPY/USD)</code>, so
          <code>r(JPY/BDT) = r(USD/BDT) − r(USD/JPY)</code>. The two legs are not alike.
          The <b>managed USD/BDT leg</b> (Bangladesh Bank) <b>mean-reverts</b> — lag-1
          autocorrelation ≈ <span id="mRho"></span> — while the free-floating
          <b>USD/JPY leg is a random walk</b>.</p>
        <p><b>Final model</b> = structural mean reversion of the taka leg:
          <code>r̂(JPY/BDT) = β · r(USD/BDT)ₜ₋₁</code>, with β fit on train only
          (<span id="mBeta"></span>). This lifts directional accuracy to <span id="mDir"></span>,
          consistently across many train/test splits.</p>
      </div>
      <div class="col">
        <h4>Metrics &amp; model comparison</h4>
        <p>Test period <span id="mTestRange"></span>. <b>Directional accuracy</b> is the share of
          days the model's sign matches reality. <b>MAE / RMSE / MAPE</b> are measured on
          reconstructed levels and compared against the persistence baseline.</p>
        <div style="overflow-x:auto;margin:8px 0 4px">
          <table style="border-collapse:collapse;width:100%;font-size:12.5px;color:var(--muted)">
            <thead>
              <tr style="color:var(--faint);text-align:left">
                <th style="padding:4px 8px;border-bottom:1px solid var(--edge)">Model</th>
                <th style="padding:4px 8px;border-bottom:1px solid var(--edge)">Direction</th>
                <th style="padding:4px 8px;border-bottom:1px solid var(--edge)">1-step MAE</th>
              </tr>
            </thead>
            <tbody>
              <tr><td style="padding:5px 8px;color:var(--teal)">Structural (final)</td>
                  <td style="padding:5px 8px;font-family:var(--mono)" id="cDirS">—</td>
                  <td style="padding:5px 8px;font-family:var(--mono)" id="cMaeS">—</td></tr>
              <tr><td style="padding:5px 8px">LSTM (v2, deep baseline)</td>
                  <td style="padding:5px 8px;font-family:var(--mono)" id="cDirL">—</td>
                  <td style="padding:5px 8px;font-family:var(--mono)" id="cMaeL">—</td></tr>
              <tr><td style="padding:5px 8px">"No change" baseline</td>
                  <td style="padding:5px 8px;font-family:var(--mono)">50% (coin flip)</td>
                  <td style="padding:5px 8px;font-family:var(--mono)" id="cMaeB">—</td></tr>
            </tbody>
          </table>
        </div>

        <h4>Uncertainty bands</h4>
        <p>95% bands grow like <code>± 1.96 · σ_resid · √h</code>, where σ_resid is the std of
          1-step test errors — i.e. uncertainty scales with the square root of horizon, as
          it does for a random walk.</p>

        <div class="note">
          <b>Read this honestly.</b> The accuracy gain here is <b>directional</b> (~62% vs 50%),
          not magical level prediction: daily FX levels are dominated by noise, so every model
          sits at the "no change" floor on 1-day MAE. The real lesson for a machine-learning
          project is that understanding the data-generating process (the managed taka leg
          mean-reverts; the yen leg is noise) beat a deep network. Not investment advice.
        </div>
      </div>
    </div>
  </details>

  <div class="foot">Generated <b id="genAt"></b> · seed <b id="genSeed"></b> · data source <b>fawazahmed0/currency-api</b></div>
</div>

<script>
const DATA = __MODEL_DATA__;

/* ---------- formatting helpers ---------- */
const fmt5 = n => n==null ? "—" : n.toFixed(5);
const fmt4 = n => n==null ? "—" : n.toFixed(4);
const fmtPct = n => (n>=0?"+":"") + n.toFixed(2) + "%";

/* ---------- derive values ---------- */
const S = DATA.series;
const lastRate = S[S.length-1].y;
const lastDate = S[S.length-1].d;
const H = S.length;
const F = DATA.forecast.length;
const M = DATA.metrics;

const rate7ago = S[Math.max(0,H-8)].y;
const rate30ago = S[Math.max(0,H-31)].y;
const delta7 = (lastRate/rate7ago - 1)*100;
const delta30 = (lastRate/rate30ago - 1)*100;

const fc7 = DATA.forecast[Math.min(6,F-1)];
const fc30 = DATA.forecast[F-1];
const band30 = (fc30.upper - fc30.lower)/2 / fc30.mean * 100;

const dirAcc = M.directional_accuracy_pct;         // structural (final model)
const dirL = M.lstm_directional_accuracy_pct;      // v2 LSTM deep baseline
const maeS = M.structural_1step.mae;
const maeL = M.lstm_1step.mae;
const maeB = M.baseline_1step.mae;
const maeBetter = maeS < maeB;
const maePctVsBase = M.structural_vs_baseline_mae_pct;

/* ---------- hero ---------- */
document.getElementById("spotRate").innerHTML = lastRate.toFixed(5) + `<small>&nbsp;৳</small>`;
const pill = document.getElementById("spotDelta");
pill.textContent = fmtPct(delta30);
pill.classList.toggle("down", delta30 < 0);
document.getElementById("spotDeltaNote").textContent =
  `vs. rate ${rate30ago.toFixed(4)} thirty days ago`;
document.getElementById("updatedBar").textContent =
  `data through ${lastDate} · ${H} daily points · generated ${DATA.meta.generated_at.slice(0,10)}`;

/* ---------- KPI cards ---------- */
document.getElementById("k7").textContent = fmt4(fc7.mean);
document.getElementById("k7b").innerHTML =
  `range ${fmt4(fc7.lower)} – ${fmt4(fc7.upper)}`;
document.getElementById("k30").textContent = fmt4(fc30.mean);
document.getElementById("k30b").innerHTML =
  `range ${fmt4(fc30.lower)} – ${fmt4(fc30.upper)} · band ±${band30.toFixed(1)}%`;
document.getElementById("kdir").textContent = dirAcc.toFixed(1) + "%";
document.getElementById("bardir").style.width = Math.min(100, dirAcc*1.5) + "%";
document.getElementById("kdirb").innerHTML =
  `out-of-sample · coin flip = 50% · v2 LSTM was <b>${dirL.toFixed(1)}%</b>`;
document.getElementById("kmae").textContent = maeS.toFixed(5);
document.getElementById("kmaeb").innerHTML =
  `structural ${maeS.toFixed(5)} vs no-change ${maeB.toFixed(5)}` +
  (maeBetter ? ` · <span class="up">beats baseline</span>` : ` · <span class="flat">≈ baseline</span>`);
document.getElementById("barmae").style.width =
  Math.min(100, (maeB/maeS)*100) + "%";

/* ---------- model comparison table ---------- */
document.getElementById("cDirS").textContent = dirAcc.toFixed(1) + "%";
document.getElementById("cMaeS").textContent = maeS.toFixed(5);
document.getElementById("cDirL").textContent = dirL.toFixed(1) + "%";
document.getElementById("cMaeL").textContent = maeL.toFixed(5);
document.getElementById("cMaeB").textContent = maeB.toFixed(5);

/* ---------- chart config ---------- */
Chart.defaults.color = "#8ba2b8";
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.borderColor = "rgba(255,255,255,.06)";
const TICK = {color:"#5a728b"};
const GRID = {color:"rgba(255,255,255,.045)"};

/* ---------- main forecast chart ---------- */
const histDates = S.map(p=>p.d);
const histVals = S.map(p=>p.y);
const fDates0 = DATA.forecast.map(p=>p.d);
const fMean0 = DATA.forecast.map(p=>p.mean);
const fLow0  = DATA.forecast.map(p=>p.lower);
const fUp0   = DATA.forecast.map(p=>p.upper);
const labels = histDates.concat(fDates0);
const N = labels.length;
const histSeries = labels.map((d,i)=> i<H ? histVals[i] : null);
const fSeries = labels.map((d,i)=> i===H-1 ? lastRate : (i>=H ? fMean0[i-H] : null));
const fLow    = labels.map((d,i)=> i===H-1 ? lastRate : (i>=H ? fLow0[i-H]  : null));
const fUp     = labels.map((d,i)=> i===H-1 ? lastRate : (i>=H ? fUp0[i-H]   : null));

function mainData(back){
  const start = back>0 ? Math.max(0, H-back) : 0;
  return {
    labels: labels.slice(start),
    datasets: [
      { label:"95% band (upper)", data:fUp.slice(start),   borderWidth:0, pointRadius:0, fill:false,   order:0 },
      { label:"95% band (lower)", data:fLow.slice(start),  borderWidth:0, pointRadius:0, fill:"-1",
        backgroundColor:"rgba(245,185,63,.16)", order:0 },
      { label:"Spot (JPY→BDT)", data:histSeries.slice(start), borderColor:"#2dd4bf",
        backgroundColor:"rgba(45,212,191,.06)", borderWidth:1.8, pointRadius:0, tension:.15, fill:true, order:1 },
      { label:"Structural projection", data:fSeries.slice(start), borderColor:"#f5b93f",
        borderWidth:2.2, pointRadius:0, borderDash:[6,4], tension:.25, order:1 }
    ]
  };
}
const mainChart = new Chart(document.getElementById("mainChart"), {
  type:"line",
  data: mainData(0),
  options:{
    responsive:true, maintainAspectRatio:false, interaction:{mode:"index",intersect:false},
    plugins:{
      legend:{display:false},
      tooltip:{
        backgroundColor:"#0c1420", borderColor:"#1c2b40", borderWidth:1, titleColor:"#e9f0f7",
        bodyColor:"#c3d2e0", padding:10,
        callbacks:{
          label: ctx => {
            if(ctx.datasetIndex===1) return null;
            return " "+ctx.dataset.label+":  "+Number(ctx.parsed.y).toFixed(5);
          },
          afterBody: items => {
            const i = items[0].dataIndex + (0);
            const idx = i; // index within current (sliced) labels
            const lvl = fLow[idx]; const up = fUp[idx];
            if(lvl!=null && up!=null) return " 95% band: "+lvl.toFixed(5)+" – "+up.toFixed(5);
            return null;
          }
        }
      }
    },
    scales:{
      x:{ticks:{...TICK, maxTicksLimit:10, maxRotation:0, autoSkip:true}, grid:{display:false}},
      y:{ticks:{...TICK}, grid:GRID, title:{display:true,text:"৳ per ¥",color:"#5a728b",font:{size:11}}}
    }
  }
});

/* ---------- backtest chart ---------- */
const bt = DATA.test_one_step;
const btDates = bt.map(p=>p.d);
function btData(back){
  const start = back>0 ? Math.max(0, btDates.length-back) : 0;
  const d = btDates.slice(start);
  const z = arr => d.map((_,i)=> arr[start+i]);
  return {
    labels:d,
    datasets:[
      {label:"Actual", data:z(bt.map(p=>p.actual)), borderColor:"#3dd68c", borderWidth:1.8, pointRadius:0, tension:.12},
      {label:"Structural 1-step", data:z(bt.map(p=>p.pred)), borderColor:"#f5b93f", borderWidth:1.8, pointRadius:0, tension:.12},
      {label:"No-change baseline", data:z(bt.map(p=>p.baseline)), borderColor:"#5a728b", borderWidth:1.2, pointRadius:0, borderDash:[3,4], tension:.12}
    ]
  };
}
const btChart = new Chart(document.getElementById("btChart"), {
  type:"line", data:btData(0),
  options:{
    responsive:true, maintainAspectRatio:false, interaction:{mode:"index",intersect:false},
    plugins:{legend:{display:false}, tooltip:{backgroundColor:"#0c1420",borderColor:"#1c2b40",borderWidth:1,padding:10,titleColor:"#e9f0f7",bodyColor:"#c3d2e0",
      callbacks:{label:ctx=>" "+ctx.dataset.label+":  "+Number(ctx.parsed.y).toFixed(5)}}},
    scales:{x:{ticks:{...TICK,maxTicksLimit:8,maxRotation:0,autoSkip:true},grid:{display:false}},y:{ticks:{...TICK},grid:GRID}}
  }
});
document.getElementById("btRangeLabel").textContent = btDates.length;

/* ---------- scatter ---------- */
const scatter = new Chart(document.getElementById("scatterChart"), {
  type:"scatter",
  data:{
    datasets:[
      {label:"test point", data:bt.map(p=>({x:p.actual,y:p.pred})),
       backgroundColor:"rgba(45,212,191,.55)", pointRadius:2.2, pointHoverRadius:4},
      {label:"parity", data:[{x:0.72,y:0.72},{x:0.86,y:0.86}],
       borderColor:"#5a728b", borderDash:[4,4], borderWidth:1.2, pointRadius:0, showLine:true, fill:false}
    ]
  },
  options:{
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}, tooltip:{backgroundColor:"#0c1420",borderColor:"#1c2b40",borderWidth:1,padding:10,titleColor:"#e9f0f7",bodyColor:"#c3d2e0",
      callbacks:{label:ctx=>" actual "+ctx.parsed.x.toFixed(5)+" · pred "+ctx.parsed.y.toFixed(5)}}},
    scales:{
      x:{type:"linear",title:{display:true,text:"actual",color:"#5a728b",font:{size:11}},ticks:{...TICK},grid:GRID},
      y:{title:{display:true,text:"predicted",color:"#5a728b",font:{size:11}},ticks:{...TICK},grid:GRID}
    }
  }
});

/* ---------- sparkline in hero ---------- */
new Chart(document.getElementById("spark"), {
  type:"line",
  data:{labels:S.slice(-30).map(p=>p.d),
    datasets:[{data:S.slice(-30).map(p=>p.y), borderColor:"#2dd4bf", borderWidth:1.6, pointRadius:0, tension:.2, fill:true,
      backgroundColor:"rgba(45,212,191,.08)"}]},
  options:{responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}, tooltip:{enabled:false}},
    scales:{x:{display:false},y:{display:false}},
    elements:{line:{tension:.2}}}
});

/* ---------- range buttons ---------- */
function wireRange(btnId, chart, getData, startBack){
  const setActive = el => { el.querySelectorAll("button").forEach(b=>b.classList.toggle("active", b===el)); };
  document.querySelectorAll("#"+btnId+" button").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const back = parseInt(btn.dataset.r,10);
      chart.data = getData(back);
      chart.update();
      setActive(btn);
    });
  });
}
wireRange("mainRanges", mainChart, mainData, 0);
wireRange("btRanges", btChart, btData, 0);

/* ---------- methodology dynamic text ---------- */
const meta = DATA.meta;
document.getElementById("mN").textContent = meta.n;
document.getElementById("mRange").textContent = meta.first_date + " → " + meta.last_date;
document.getElementById("mSeq").textContent = meta.seq_len;
document.getElementById("mBeta").textContent = meta.beta_taka_ar1;
document.getElementById("mRho").textContent = meta.autocorr_r_ub;
document.getElementById("mDir").textContent = dirAcc.toFixed(1) + "%";
document.getElementById("mTestRange").textContent =
  DATA.split.split_date + " → " + meta.last_date + " (" + DATA.split.test_n + " days)";
document.getElementById("genAt").textContent = meta.generated_at.slice(0,19).replace("T"," ");
document.getElementById("genSeed").textContent = meta.seed;
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="models/model_forecast.json")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    payload = json.dumps(data)
    html = TEMPLATE.replace("__MODEL_DATA__", payload)

    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
