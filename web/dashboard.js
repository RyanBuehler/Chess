// web/dashboard.js — Training monitor: run list + charts. No game browser.
import { getJSON, showError } from "./common.js";

// uPlot dark theme colors.
const AXIS_STROKE = "#9aa0b5";
const GRID_STROKE = "#2b2d3a";
const SERIES = ["#7aa2f7", "#bb9af7", "#9ece6a", "#e0af68", "#f7768e"];

// Chart instance registry — destroyed before recreation to avoid leaks.
const charts = { loss: null, rate: null, elo: null };

// Currently selected run ID.
let selectedRunId = null;

// Auto-refresh state.
let refreshTimer = null;

// Parse ?refresh=<seconds> from query string. Default 30, min 1, 0 = off.
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

// Record axis labels on window.__chartAxes for test assertions.
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
    width: el.clientWidth || 420, height: 200,
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

function markSelected(ul, li) {
  for (const c of ul.children) c.classList.remove("selected");
  if (li) li.classList.add("selected");
}

async function loadCharts(runId) {
  let metrics = [], elo = [];
  try {
    [metrics, elo] = await Promise.all([
      getJSON(`/api/runs/${runId}/metrics`),
      getJSON(`/api/runs/${runId}/elo`),
    ]);
  } catch (e) {
    showError("Failed to load metrics/elo: " + e.message);
    return;
  }
  const steps = metrics.map((m) => m.step);
  lineChart("loss-chart", "loss", steps, [
    { label: "policy_loss", data: metrics.map((m) => m.policy_loss ?? null) },
    { label: "value_loss", data: metrics.map((m) => m.value_loss ?? null) },
  ], "step", "loss");
  lineChart("rate-chart", "rate", steps, [
    { label: "games/hr", data: metrics.map((m) => m.games_per_hour ?? null) },
  ], "step", "games / hour");
  lineChart("elo-chart", "elo", elo.map((e) => e.step), [
    { label: "Elo", data: elo.map((e) => e.elo) },
  ], "step", "Elo");
}

async function selectRun(runId, li) {
  selectedRunId = runId;
  if (li && li.parentElement) markSelected(li.parentElement, li);
  document.getElementById("sel-run").textContent = runId;
  await loadCharts(runId);
}

async function refreshAll() {
  // Re-fetch run list and redraw charts for the selected run.
  let runs = [];
  try {
    runs = await getJSON("/api/runs");
  } catch (e) {
    // Don't wipe existing display on transient refresh failures.
    return;
  }
  runs = runs.slice().reverse();

  const ul = document.getElementById("run-list");
  ul.innerHTML = "";
  if (!runs.length) { ul.innerHTML = "<li class='hint'>(no runs found)</li>"; return; }

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

  // Re-select previously selected run if still present, otherwise select newest.
  if (selLi) {
    selLi.classList.add("selected");
    await loadCharts(selectedRunId);
  } else {
    await selectRun(runs[0].run_id, ul.firstChild);
  }
}

function scheduleRefresh(seconds) {
  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
  if (seconds <= 0) return;

  refreshTimer = setInterval(() => {
    if (!document.hidden) refreshAll();
  }, seconds * 1000);
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

  const ul = document.getElementById("run-list");
  ul.innerHTML = "";
  if (!runs.length) { ul.innerHTML = "<li class='hint'>(no runs found)</li>"; return; }
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
  }
  // Auto-select the newest run (first in the reversed list).
  await selectRun(runs[0].run_id, ul.firstChild);

  scheduleRefresh(refreshSeconds);
}

init();
