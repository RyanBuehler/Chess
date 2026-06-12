// web/compare.js — Architecture comparison page.
// Fetches all runs' metrics, elo, and provenance up-front; caches in JS.
// Rebuilds charts on x-axis change or checkbox toggle. No build step.
import { getJSON, showError, clearError } from "./common.js";

// -----------------------------------------------------------------------
// Colour palette (fixed, cycles if more runs than palette entries)
// -----------------------------------------------------------------------
const PALETTE = [
  "#7aa2f7", // accent blue
  "#9ece6a", // green
  "#f7768e", // red
  "#e0af68", // yellow
  "#bb9af7", // purple
  "#7dcfff", // cyan
  "#ff9e64", // orange
  "#41a6b5", // teal
];

function runColor(idx) {
  return PALETTE[idx % PALETTE.length];
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
function fmtParams(n) {
  if (!Number.isFinite(n)) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function fmtNum(n) {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString();
}

function fmtElo(n) {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toFixed(0);
}

// Downsample an array of {x, y} points to at most maxPts points.
// Uses uniform stride so shape is preserved.
function downsample(pts, maxPts) {
  if (pts.length <= maxPts) return pts;
  const stride = Math.ceil(pts.length / maxPts);
  const out = [];
  for (let i = 0; i < pts.length; i += stride) out.push(pts[i]);
  return out;
}

// Build a sorted union of x values from multiple series arrays.
// series: array of Float64Array (or null/undefined).
function unionX(seriesArr) {
  const seen = new Set();
  for (const s of seriesArr) {
    if (!s) continue;
    for (const v of s) seen.add(v);
  }
  const arr = Array.from(seen);
  arr.sort((a, b) => a - b);
  return arr;
}

// Given a sorted union x-array and per-series {xs, ys} objects,
// produce per-series y-arrays aligned to the union (null where absent).
function alignSeries(xUnion, series) {
  return series.map((s) => {
    if (!s) return xUnion.map(() => null);
    const { xs, ys } = s;
    const map = new Map();
    for (let i = 0; i < xs.length; i++) map.set(xs[i], ys[i]);
    return xUnion.map((x) => {
      const v = map.get(x);
      return v !== undefined ? v : null;
    });
  });
}

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------
const cache = {}; // run_id -> { metrics, elo, provenance }
let runMeta = []; // [{run_id, color, checked}, ...]
let currentXAxis = "steps";

let eloChart = null;
let lossChart = null;
let gphChart = null;

// -----------------------------------------------------------------------
// Data fetching
// -----------------------------------------------------------------------
async function fetchAll() {
  let runs;
  try {
    runs = await getJSON("/api/runs");
  } catch (e) {
    showError("Failed to load runs: " + e.message);
    return [];
  }

  // Sort newest first (by run_id, which embeds timestamp)
  runs.sort((a, b) => b.run_id.localeCompare(a.run_id));

  await Promise.all(
    runs.map(async (run, idx) => {
      const id = run.run_id;
      const [metrics, elo, provenance] = await Promise.allSettled([
        getJSON(`/api/runs/${id}/metrics`),
        getJSON(`/api/runs/${id}/elo`),
        getJSON(`/api/runs/${id}/provenance`),
      ]);
      cache[id] = {
        config: run.config || {},
        state: run.state || {},
        metrics: metrics.status === "fulfilled" ? metrics.value : [],
        elo: elo.status === "fulfilled" ? elo.value : [],
        provenance: provenance.status === "fulfilled" ? provenance.value : null,
      };
    })
  );

  runMeta = runs.map((r, i) => ({
    run_id: r.run_id,
    color: runColor(i),
    checked: true,
  }));

  return runs;
}

// -----------------------------------------------------------------------
// Summary table
// -----------------------------------------------------------------------
function buildSummaryTable() {
  const tbody = document.getElementById("summary-body");
  if (!runMeta.length) {
    tbody.innerHTML =
      '<tr><td colspan="8" class="hint" style="text-align:center;padding:1rem;">No runs found.</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  runMeta.forEach((meta) => {
    const d = cache[meta.run_id];
    const prov = d.provenance;
    const net = prov && prov.network;

    const archetype = net ? net.archetype : "—";
    const params = net ? fmtParams(net.params) : "—";

    const state = d.state || {};
    const games = fmtNum(state.games);
    const steps = fmtNum(
      d.metrics.length ? d.metrics[d.metrics.length - 1].step : null
    );

    const eloVals = d.elo.map((r) => r.elo).filter(Number.isFinite);
    const peakElo = eloVals.length ? fmtElo(Math.max(...eloVals)) : "—";
    const latestElo = eloVals.length
      ? fmtElo(eloVals[eloVals.length - 1])
      : "—";

    const tr = document.createElement("tr");
    tr.dataset.runId = meta.run_id;
    tr.innerHTML = `
      <td><input type="checkbox" class="run-check" data-run="${meta.run_id}" ${meta.checked ? "checked" : ""} /></td>
      <td>
        <span class="run-dot" style="background:${meta.color};display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.4rem;"></span>
        <span class="mono">${meta.run_id}</span>
      </td>
      <td class="mono">${archetype}</td>
      <td class="mono">${params}</td>
      <td class="mono">${games}</td>
      <td class="mono">${steps}</td>
      <td class="mono">${peakElo}</td>
      <td class="mono">${latestElo}</td>
    `;
    tbody.appendChild(tr);
  });

  // Wire checkboxes
  tbody.querySelectorAll(".run-check").forEach((cb) => {
    cb.addEventListener("change", () => {
      const id = cb.dataset.run;
      const m = runMeta.find((r) => r.run_id === id);
      if (m) m.checked = cb.checked;
      rebuildCharts();
    });
  });
}

// -----------------------------------------------------------------------
// uPlot chart helpers
// -----------------------------------------------------------------------
function destroyChart(chart) {
  if (chart) {
    try {
      chart.destroy();
    } catch (_) {}
  }
  return null;
}

function makeSeries(label, color) {
  return {
    label,
    stroke: color,
    width: 2,
    spanGaps: true,
  };
}

function buildUplot(containerId, title, xLabel, yLabel, xs, seriesDefs, seriesData) {
  const el = document.getElementById(containerId);
  if (!el) return null;

  // Clear previous chart
  el.innerHTML = "";

  if (!xs || xs.length === 0) {
    el.innerHTML = '<p class="hint" style="padding:.5rem;">No data.</p>';
    return null;
  }

  const width = Math.min(el.offsetWidth || 700, 900);
  const height = 260;

  const data = [new Float64Array(xs)];
  for (const sd of seriesData) {
    data.push(new Float64Array(sd.map((v) => (v == null ? NaN : v))));
  }

  const opts = {
    title,
    width,
    height,
    // Use a plain numeric scale on x (not time-based).
    scales: {
      x: { time: false },
    },
    series: [{ label: xLabel }].concat(seriesDefs),
    axes: [
      {
        label: xLabel,
        stroke: "#c0caf5",
        ticks: { stroke: "#2b2d3a" },
        grid: { stroke: "#2b2d3a" },
      },
      {
        label: yLabel,
        stroke: "#c0caf5",
        ticks: { stroke: "#2b2d3a" },
        grid: { stroke: "#2b2d3a" },
      },
    ],
    cursor: { show: true },
    legend: { show: true },
  };

  return new uPlot(opts, data, el);
}

// -----------------------------------------------------------------------
// Elo chart data building
// -----------------------------------------------------------------------

// For each run, compute elo {xs, ys} according to the current x-axis mode.
function eloSeriesForRun(runId, xMode) {
  const d = cache[runId];
  if (!d || !d.elo.length) return null;

  const elos = d.elo;

  if (xMode === "steps") {
    const xs = elos.map((r) => r.step);
    const ys = elos.map((r) => r.elo);
    return { xs, ys };
  }

  if (xMode === "games") {
    // Map each elo point's step to games via nearest metrics row by step.
    const metrics = d.metrics;
    const metricsSorted = metrics
      .filter((m) => m.step != null)
      .sort((a, b) => a.step - b.step);

    function stepToGames(step) {
      if (!metricsSorted.length) return null;
      // Find closest metrics row by step.
      let best = metricsSorted[0];
      let bestDiff = Math.abs(best.step - step);
      for (const m of metricsSorted) {
        const diff = Math.abs(m.step - step);
        if (diff < bestDiff) {
          bestDiff = diff;
          best = m;
        }
      }
      return best.games != null ? best.games : null;
    }

    const xs = elos.map((r) => stepToGames(r.step));
    const ys = elos.map((r) => r.elo);
    // Filter out null x values
    const pairs = xs.map((x, i) => [x, ys[i]]).filter(([x]) => x != null);
    if (!pairs.length) return null;
    return { xs: pairs.map(([x]) => x), ys: pairs.map(([, y]) => y) };
  }

  if (xMode === "hours") {
    // Per-run relative time: (ts - first_ts) / 3600
    const ts0 = elos[0].ts;
    const xs = elos.map((r) => (r.ts - ts0) / 3600);
    const ys = elos.map((r) => r.elo);
    return { xs, ys };
  }

  return null;
}

// -----------------------------------------------------------------------
// Loss chart (policy_loss vs step, downsampled)
// -----------------------------------------------------------------------
function lossSeriesForRun(runId) {
  const d = cache[runId];
  if (!d || !d.metrics.length) return null;

  const pts = d.metrics
    .filter((m) => m.step != null && m.policy_loss != null)
    .map((m) => ({ x: m.step, y: m.policy_loss }));

  if (!pts.length) return null;
  const sampled = downsample(pts, 500);
  return { xs: sampled.map((p) => p.x), ys: sampled.map((p) => p.y) };
}

// -----------------------------------------------------------------------
// Games/hour chart (games_per_hour vs hours relative)
// -----------------------------------------------------------------------
function gphSeriesForRun(runId) {
  const d = cache[runId];
  if (!d || !d.metrics.length) return null;

  // Use metrics rows that have a timestamp proxy: we don't have a direct ts
  // in metrics, but we can derive approximate relative time from games_per_hour
  // and games count if available. However the spec says: use elo.ts (per-run
  // relative). For metrics we don't have ts, so we use step as x proxy with
  // an annotation. Actually, the spec says "games_per_hour vs hours-relative".
  // metrics.jsonl doesn't have ts; we'll estimate hours from games_per_hour:
  // hours_elapsed ≈ games / games_per_hour (cumulative). This is an approximation.
  // If no games_per_hour, omit.
  const pts = [];
  let cumulativeGames = 0;
  for (const m of d.metrics) {
    const gph = m.games_per_hour;
    const g = m.games;
    if (gph == null || !Number.isFinite(gph) || gph <= 0) continue;
    // hours = games / gph (gph is games/hour, so hours = games / gph)
    const hours = g != null ? g / gph : null;
    if (hours == null) continue;
    pts.push({ x: hours, y: gph });
  }

  if (!pts.length) return null;
  const sampled = downsample(pts, 500);
  return { xs: sampled.map((p) => p.x), ys: sampled.map((p) => p.y) };
}

// -----------------------------------------------------------------------
// Chart rebuild
// -----------------------------------------------------------------------
function rebuildCharts() {
  const checked = runMeta.filter((m) => m.checked);

  // --- Elo chart ---
  eloChart = destroyChart(eloChart);
  {
    const rawSeries = checked.map((m) =>
      eloSeriesForRun(m.run_id, currentXAxis)
    );
    const xUnion = unionX(rawSeries.map((s) => (s ? s.xs : null)));
    const aligned = alignSeries(xUnion, rawSeries);
    const seriesDefs = checked.map((m) => makeSeries(m.run_id, m.color));
    const xLabel =
      currentXAxis === "steps"
        ? "Step"
        : currentXAxis === "games"
        ? "Games"
        : "Hours (relative)";
    eloChart = buildUplot(
      "elo-chart",
      "",
      xLabel,
      "Elo",
      xUnion,
      seriesDefs,
      aligned
    );
  }

  // --- Policy loss chart ---
  lossChart = destroyChart(lossChart);
  {
    const rawSeries = checked.map((m) => lossSeriesForRun(m.run_id));
    const xUnion = unionX(rawSeries.map((s) => (s ? s.xs : null)));
    const aligned = alignSeries(xUnion, rawSeries);
    const seriesDefs = checked.map((m) => makeSeries(m.run_id, m.color));
    lossChart = buildUplot(
      "loss-chart",
      "",
      "Step",
      "Policy loss",
      xUnion,
      seriesDefs,
      aligned
    );
  }

  // --- Games/hour chart ---
  gphChart = destroyChart(gphChart);
  {
    const rawSeries = checked.map((m) => gphSeriesForRun(m.run_id));
    const xUnion = unionX(rawSeries.map((s) => (s ? s.xs : null)));
    const aligned = alignSeries(xUnion, rawSeries);
    const seriesDefs = checked.map((m) => makeSeries(m.run_id, m.color));
    gphChart = buildUplot(
      "gph-chart",
      "",
      "Hours (relative)",
      "Games/hr",
      xUnion,
      seriesDefs,
      aligned
    );
  }
}

// -----------------------------------------------------------------------
// Auto-refresh helpers
// -----------------------------------------------------------------------

// Parse ?refresh=<seconds> from query string. Default 30, min 1, 0 = off.
function getRefreshSeconds() {
  const params = new URLSearchParams(location.search);
  const raw = params.get("refresh");
  if (raw === null) return 30;
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) return 30;
  return n;
}

function updateRefreshIndicator(seconds) {
  const el = document.getElementById("refresh-indicator");
  if (!el) return;
  el.textContent = seconds > 0 ? `auto-refresh: ${seconds}s` : "auto-refresh: off";
}

// Save checked state keyed by run_id before re-fetching.
function snapshotChecked() {
  const map = {};
  for (const m of runMeta) map[m.run_id] = m.checked;
  return map;
}

async function refreshCompare() {
  // Snapshot current UI state so we can restore it after refetch.
  const checkedSnapshot = snapshotChecked();
  const xAxisSnapshot = currentXAxis;

  // Re-fetch all data.
  let runs;
  try {
    runs = await fetchAll();
  } catch (e) {
    // Silently skip on transient error — don't wipe existing display.
    return;
  }
  if (!runs.length) return;

  // Restore checkbox states: existing runs keep their checked state,
  // newly appeared runs default to checked (already set by fetchAll).
  for (const m of runMeta) {
    if (checkedSnapshot.hasOwnProperty(m.run_id)) {
      m.checked = checkedSnapshot[m.run_id];
    }
    // New run_ids not in snapshot remain checked (default from fetchAll).
  }

  // Restore x-axis selection.
  currentXAxis = xAxisSnapshot;
  document.querySelectorAll("input[name='xaxis']").forEach((r) => {
    r.checked = r.value === currentXAxis;
  });

  clearError();
  buildSummaryTable();
  rebuildCharts();
}

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------

// Wire all handlers before any chart init.
document.querySelectorAll("input[name='xaxis']").forEach((radio) => {
  radio.addEventListener("change", () => {
    if (radio.checked) {
      currentXAxis = radio.value;
      rebuildCharts();
    }
  });
});

// Fetch data, populate table, build charts.
(async () => {
  try {
    const refreshSeconds = getRefreshSeconds();
    updateRefreshIndicator(refreshSeconds);

    const runs = await fetchAll();
    if (runs.length === 0) {
      document.getElementById("summary-body").innerHTML =
        '<tr><td colspan="8" class="hint" style="text-align:center;padding:1rem;">No runs found.</td></tr>';
      return;
    }
    clearError();
    buildSummaryTable();
    rebuildCharts();

    // Schedule auto-refresh when tab is visible.
    if (refreshSeconds > 0) {
      setInterval(() => {
        if (!document.hidden) refreshCompare();
      }, refreshSeconds * 1000);
    }
  } catch (e) {
    showError("Compare page error: " + e.message);
  }
})();
