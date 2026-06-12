// web/dashboard.js
import { getJSON, makeBoard, lastMovePair, showError, setStatus } from "./common.js";

let board = null;
let replay = { moves: [], idx: 0, timer: null, result: "" };

// uPlot dark theme colors.
const AXIS_STROKE = "#9aa0b5";
const GRID_STROKE = "#2b2d3a";
const SERIES = ["#7aa2f7", "#bb9af7", "#9ece6a", "#e0af68", "#f7768e"];

async function init() {
  // Wire replay controls FIRST so they work even if board init fails.
  wireReplayControls();

  // Board init is fragile (chessground); never let it kill the page.
  try {
    board = makeBoard(document.getElementById("replay-board"));
  } catch (e) {
    showError("Board failed to initialize: " + e.message);
  }

  let runs = [];
  try {
    runs = await getJSON("/api/runs");
  } catch (e) {
    showError("Failed to load runs: " + e.message);
    return;
  }
  // Newest first (catalog returns ascending by run_id; reverse for newest-first).
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
  selectRun(runs[0].run_id, ul.firstChild);
}

function markSelected(ul, li) {
  for (const c of ul.children) c.classList.remove("selected");
  if (li) li.classList.add("selected");
}

function lineChart(elId, xs, seriesDefs, title) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  const hasData = xs.length && seriesDefs.some((s) => s.data.some((v) => v != null));
  if (!hasData) { el.innerHTML = `<div class="hint">(no ${title} data)</div>`; return; }
  const series = [{ label: "step", stroke: AXIS_STROKE }];
  const data = [xs];
  seriesDefs.forEach((s, i) => {
    series.push({ label: s.label, stroke: SERIES[i % SERIES.length], width: 2, spanGaps: true });
    data.push(s.data);
  });
  new uPlot({
    width: el.clientWidth || 420, height: 200, title,
    scales: { x: { time: false } },
    series,
    axes: [
      { stroke: AXIS_STROKE, grid: { stroke: GRID_STROKE }, ticks: { stroke: GRID_STROKE } },
      { stroke: AXIS_STROKE, grid: { stroke: GRID_STROKE }, ticks: { stroke: GRID_STROKE } },
    ],
    legend: { live: false },
  }, data, el);
}

async function selectRun(runId, li) {
  if (li && li.parentElement) markSelected(li.parentElement, li);
  document.getElementById("sel-run").textContent = runId;
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
  lineChart("loss-chart", steps, [
    { label: "policy_loss", data: metrics.map((m) => m.policy_loss ?? null) },
    { label: "value_loss", data: metrics.map((m) => m.value_loss ?? null) },
  ], "Training loss");
  lineChart("rate-chart", steps, [
    { label: "games/hr", data: metrics.map((m) => m.games_per_hour ?? null) },
  ], "Self-play throughput");
  lineChart("elo-chart", elo.map((e) => e.step), [
    { label: "Elo", data: elo.map((e) => e.elo) },
  ], "Evaluator Elo");
  await loadGames(runId);
}

async function loadGames(runId) {
  let games = [];
  try {
    games = await getJSON(`/api/runs/${runId}/games`);
  } catch (e) {
    showError("Failed to load games: " + e.message);
    return;
  }
  const ul = document.getElementById("game-list");
  ul.innerHTML = "";
  if (!games.length) { ul.innerHTML = "<li class='hint'>(no games)</li>"; return; }
  // Newest games first.
  for (const g of games.slice().reverse().slice(0, 200)) {
    const li = document.createElement("li");
    li.textContent = g.name;
    li.onclick = () => { markSelected(ul, li); loadReplay(runId, g.name); };
    ul.appendChild(li);
  }
}

async function loadReplay(runId, name) {
  stopPlay();
  let data;
  try {
    data = await getJSON(`/api/runs/${runId}/games/${name}/moves`);
  } catch (e) {
    showError("Failed to load game moves: " + e.message);
    return;
  }
  replay = { moves: data.moves, idx: 0, timer: null, result: data.result };
  renderReplay();
}

// Minimal UCI mover over a chessground-style piece map keyed by square.
function startMap() {
  const map = new Map();
  const back = ["r", "n", "b", "q", "k", "b", "n", "r"];
  for (let f = 0; f < 8; f++) {
    map.set(sq(f, 0), { role: roleOf(back[f]), color: "white" });
    map.set(sq(f, 1), { role: "pawn", color: "white" });
    map.set(sq(f, 6), { role: "pawn", color: "black" });
    map.set(sq(f, 7), { role: roleOf(back[f]), color: "black" });
  }
  return map;
}
function sq(file, rank) { return "abcdefgh"[file] + (rank + 1); }
function roleOf(c) {
  return { r: "rook", n: "knight", b: "bishop", q: "queen", k: "king", p: "pawn" }[c];
}
function applyUci(map, uci) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4), promo = uci[4];
  const piece = map.get(from);
  if (!piece) return;
  map.delete(from);
  if (piece.role === "pawn" && from[0] !== to[0] && !map.get(to)) {
    map.delete(to[0] + from[1]);
  }
  if (piece.role === "king" && Math.abs(from.charCodeAt(0) - to.charCodeAt(0)) === 2) {
    const rank = from[1];
    if (to[0] === "g") { map.set("f" + rank, map.get("h" + rank)); map.delete("h" + rank); }
    if (to[0] === "c") { map.set("d" + rank, map.get("a" + rank)); map.delete("a" + rank); }
  }
  map.set(to, promo ? { role: roleOf(promo), color: piece.color } : piece);
}
// Full-FEN board field from the piece map. We must replace the WHOLE board
// state each step: chessground's setPieces() is a sparse diff and never
// clears vacated squares (the "cloned pieces" bug).
function fenFromMap(map) {
  const letter = { pawn: "p", knight: "n", bishop: "b", rook: "r", queen: "q", king: "k" };
  const ranks = [];
  for (let r = 7; r >= 0; r--) {
    let row = "", empty = 0;
    for (let f = 0; f < 8; f++) {
      const p = map.get(sq(f, r));
      if (!p) { empty++; continue; }
      if (empty) { row += empty; empty = 0; }
      const ch = letter[p.role];
      row += p.color === "white" ? ch.toUpperCase() : ch;
    }
    if (empty) row += empty;
    ranks.push(row);
  }
  return ranks.join("/");
}

function renderReplay() {
  if (!board) return;
  const map = startMap();
  for (let i = 0; i < replay.idx; i++) applyUci(map, replay.moves[i]);
  const last = replay.idx > 0 ? lastMovePair(replay.moves[replay.idx - 1]) : undefined;
  board.set({ fen: fenFromMap(map), lastMove: last });
  setStatus("rp-status", `${replay.idx}/${replay.moves.length}  ${replay.result || ""}`);
}

function wireReplayControls() {
  document.getElementById("rp-prev").onclick = () => { if (replay.idx > 0) { replay.idx--; renderReplay(); } };
  document.getElementById("rp-next").onclick = () => { if (replay.idx < replay.moves.length) { replay.idx++; renderReplay(); } };
  document.getElementById("rp-play").onclick = togglePlay;
}
function togglePlay() {
  const btn = document.getElementById("rp-play");
  if (replay.timer) { stopPlay(); return; }
  if (!replay.moves.length) return;
  if (replay.idx >= replay.moves.length) { replay.idx = 0; renderReplay(); }
  btn.textContent = "⏸ pause";
  replay.timer = setInterval(() => {
    if (replay.idx >= replay.moves.length) { stopPlay(); return; }
    replay.idx++; renderReplay();
  }, 600);
}
function stopPlay() {
  if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
  const btn = document.getElementById("rp-play");
  if (btn) btn.textContent = "▶ play";
}

init();
