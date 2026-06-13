// web/goals.js — Goal-diagnostics panel: repertoire size, per-goal achievement
// rate + learning-progress, win-ply fraction, and the wishful-thinking
// thermometer for a selected goal run. Vendored uPlot only, dark theme.
import { getJSON, showError } from "./common.js";

// uPlot dark theme colors (match dashboard.js).
const AXIS_STROKE = "#9aa0b5";
const GRID_STROKE = "#2b2d3a";
const SERIES = ["#7aa2f7", "#9ece6a", "#f7768e", "#e0af68", "#bb9af7",
                "#7dcfff", "#ff9e64", "#41a6b5"];

// Chart instance registry — destroyed before recreation to avoid leaks.
const charts = { repertoire: null, rate: null, lp: null, winply: null };

let selectedRunId = null;
let refreshTimer = null;

function getRefreshSeconds() {
  const params = new URLSearchParams(location.search);
  const raw = params.get("refresh");
  if (raw === null) return 30;
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) return 30;
  return n; // 0 = off
}

function updateRefreshIndicator(seconds) {
  const el = document.getElementById("refresh-indicator");
  if (!el) return;
  el.textContent = seconds > 0 ? `auto-refresh: ${seconds}s` : "auto-refresh: off";
}

// Record axis labels on window.__chartAxes for test assertions (mirror dashboard).
function recordChartAxes(chartId, xLabel, yLabel) {
  if (!window.__chartAxes) window.__chartAxes = {};
  window.__chartAxes[chartId] = { x: xLabel, y: yLabel };
}

function destroyChart(key) {
  if (charts[key]) {
    try { charts[key].destroy(); } catch (_) {}
    charts[key] = null;
  }
}

function lineChart(elId, chartKey, xs, seriesDefs, xLabel, yLabel) {
  destroyChart(chartKey);
  recordChartAxes(elId, xLabel, yLabel);

  const el = document.getElementById(elId);
  el.innerHTML = "";
  const hasData = xs.length && seriesDefs.some((s) => s.data.some((v) => v != null));
  if (!hasData) {
    el.innerHTML = `<div class="hint">(no ${yLabel} data)</div>`;
    return;
  }
  const series = [{ label: xLabel, stroke: AXIS_STROKE }];
  const data = [xs];
  seriesDefs.forEach((s, i) => {
    series.push({ label: s.label, stroke: SERIES[i % SERIES.length], width: 2, spanGaps: true });
    data.push(s.data);
  });
  const u = new uPlot({
    width: el.clientWidth || 460, height: 220,
    scales: { x: { time: false } },
    series,
    axes: [
      { label: xLabel, stroke: AXIS_STROKE, grid: { stroke: GRID_STROKE }, ticks: { stroke: GRID_STROKE } },
      { label: yLabel, stroke: AXIS_STROKE, grid: { stroke: GRID_STROKE }, ticks: { stroke: GRID_STROKE } },
    ],
    legend: { live: false },
  }, data, el);
  charts[chartKey] = u;
}

// Render the wishful-thinking thermometer: one row per goal kind with a bar
// for self-play achievement and a marker for held-out (vs-Stockfish) rate.
function renderThermometer(wishful) {
  const host = document.getElementById("thermometer");
  host.innerHTML = "";
  const kinds = Object.keys(wishful || {}).sort();
  if (!kinds.length) {
    host.innerHTML = `<div class="hint">(no thermometer data yet)</div>`;
    return;
  }
  for (const kind of kinds) {
    const row = document.createElement("div");
    row.className = "thermo-row";
    const sp = wishful[kind].self_play;
    const vs = wishful[kind].vs_stockfish;
    const gap = wishful[kind].gap;

    const label = document.createElement("span");
    label.className = "thermo-label";
    label.textContent = kind;

    const track = document.createElement("div");
    track.className = "thermo-track";
    const fill = document.createElement("div");
    fill.className = "thermo-fill";
    fill.style.width = `${Math.max(0, Math.min(1, sp ?? 0)) * 100}%`;
    track.appendChild(fill);
    if (vs != null) {
      const marker = document.createElement("div");
      marker.className = "thermo-marker";
      marker.style.left = `${Math.max(0, Math.min(1, vs)) * 100}%`;
      marker.title = `vs-Stockfish: ${(vs * 100).toFixed(0)}%`;
      track.appendChild(marker);
    }

    const val = document.createElement("span");
    val.className = "thermo-val";
    const spTxt = sp != null ? `${(sp * 100).toFixed(0)}%` : "—";
    const gapTxt = gap != null ? ` (gap ${(gap * 100).toFixed(0)}%)` : "";
    val.textContent = `${spTxt}${gapTxt}`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(val);
    host.appendChild(row);
  }
}

async function loadDiagnostics(runId) {
  let d;
  try {
    d = await getJSON(`/api/runs/${runId}/goals`);
  } catch (e) {
    showError("Failed to load goal diagnostics: " + e.message);
    return;
  }

  const panels = document.getElementById("goal-panels");
  const noGoals = document.getElementById("no-goals");
  if (!d.is_goal_run) {
    panels.style.display = "none";
    noGoals.style.display = "block";
    return;
  }
  panels.style.display = "";
  noGoals.style.display = "none";

  const steps = d.steps || [];

  lineChart("repertoire-chart", "repertoire", steps, [
    { label: "templates", data: (d.repertoire_size || []).map((v) => v ?? null) },
  ], "step", "repertoire size");

  const rateSeries = (d.goal_kinds || []).map((k) => ({
    label: k, data: (d.achievement_rate[k] || []).map((v) => v ?? null),
  }));
  lineChart("rate-chart", "rate", steps, rateSeries, "step", "achievement rate");

  const lpSeries = (d.goal_kinds || []).map((k) => ({
    label: k, data: (d.learning_progress[k] || []).map((v) => v ?? null),
  }));
  lineChart("lp-chart", "lp", steps, lpSeries, "step", "learning progress");

  lineChart("winply-chart", "winply", steps, [
    { label: "win-ply frac", data: (d.win_ply_fraction || []).map((v) => v ?? null) },
  ], "step", "win-ply fraction");

  renderThermometer(d.wishful_thinking);
}

function markSelected(ul, li) {
  for (const c of ul.children) c.classList.remove("selected");
  if (li) li.classList.add("selected");
}

async function selectRun(runId, li) {
  selectedRunId = runId;
  if (li && li.parentElement) markSelected(li.parentElement, li);
  document.getElementById("sel-run").textContent = runId;
  await loadDiagnostics(runId);
}

function renderRunList(runs) {
  const ul = document.getElementById("run-list");
  ul.innerHTML = "";
  if (!runs.length) { ul.innerHTML = "<li class='hint'>(no runs found)</li>"; return null; }
  let selLi = null;
  for (const r of runs) {
    const li = document.createElement("li");
    const games = (r.state && r.state.games) ?? "—";
    li.textContent = `${r.run_id}`;
    const sub = document.createElement("small");
    sub.className = "muted";
    sub.textContent = ` ${games} games`;
    li.appendChild(sub);
    li.onclick = () => selectRun(r.run_id, li);
    ul.appendChild(li);
    if (r.run_id === selectedRunId) selLi = li;
  }
  return selLi;
}

async function refreshAll() {
  let runs = [];
  try {
    runs = await getJSON("/api/runs");
  } catch (e) {
    return; // don't wipe display on transient refresh failures
  }
  runs = runs.slice().reverse();
  const selLi = renderRunList(runs);
  if (!runs.length) return;
  if (selLi) {
    selLi.classList.add("selected");
    await loadDiagnostics(selectedRunId);
  } else {
    await selectRun(runs[0].run_id, document.getElementById("run-list").firstChild);
  }
}

function scheduleRefresh(seconds) {
  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
  if (seconds <= 0) return;
  refreshTimer = setInterval(() => { if (!document.hidden) refreshAll(); }, seconds * 1000);
}

async function init() {
  const refreshSeconds = getRefreshSeconds();
  updateRefreshIndicator(refreshSeconds);

  let runs = [];
  try {
    runs = await getJSON("/api/runs");
  } catch (e) {
    showError("Failed to load runs: " + e.message);
    return;
  }
  runs = runs.slice().reverse();
  renderRunList(runs);
  if (!runs.length) return;
  await selectRun(runs[0].run_id, document.getElementById("run-list").firstChild);
  scheduleRefresh(refreshSeconds);
}

init();
