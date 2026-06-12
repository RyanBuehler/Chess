// web/dashboard.js
import { getJSON, makeBoard, lastMovePair } from "/common.js";

let selectedRun = null;
let board = null;
let replay = { moves: [], idx: 0, timer: null, result: "" };

async function init() {
  const runs = await getJSON("/api/runs");
  const ul = document.getElementById("run-list");
  ul.innerHTML = "";
  for (const r of runs) {
    const li = document.createElement("li");
    const step = (r.state && r.state.step) ?? "—";
    li.textContent = `${r.run_id}  (step ${step})`;
    li.onclick = () => selectRun(r.run_id);
    ul.appendChild(li);
  }
  board = makeBoard(document.getElementById("replay-board"));
  if (runs.length) selectRun(runs[0].run_id);
  wireReplayControls();
}

function lineChart(elId, xs, ys, label) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  if (!xs.length) { el.textContent = `(no ${label} data)`; return; }
  new uPlot({
    width: el.clientWidth || 360, height: 180, title: label,
    scales: { x: { time: false } },
    series: [{ label: "step" }, { label, stroke: "#3b6ea5" }],
  }, [xs, ys], el);
}

async function selectRun(runId) {
  selectedRun = runId;
  document.getElementById("sel-run").textContent = runId;
  const metrics = await getJSON(`/api/runs/${runId}/metrics`);
  const elo = await getJSON(`/api/runs/${runId}/elo`);
  const steps = metrics.map((m) => m.step);
  lineChart("loss-chart", steps, metrics.map((m) => m.loss ?? null), "loss");
  lineChart("rate-chart", steps, metrics.map((m) => m.games_per_hour ?? null), "games/hr");
  lineChart("elo-chart", elo.map((e) => e.step), elo.map((e) => e.elo), "Elo");
  await loadGames(runId);
}

async function loadGames(runId) {
  const games = await getJSON(`/api/runs/${runId}/games`);
  const ul = document.getElementById("game-list");
  ul.innerHTML = "";
  for (const g of games.slice(0, 200)) {
    const li = document.createElement("li");
    li.textContent = g.name;
    li.onclick = () => loadReplay(runId, g.name);
    ul.appendChild(li);
  }
}

async function loadReplay(runId, name) {
  const data = await getJSON(`/api/runs/${runId}/games/${name}/moves`);
  stopPlay();
  replay = { moves: data.moves, idx: 0, timer: null, result: data.result };
  renderReplay();
}

// Minimal UCI mover over a chessground-style piece map keyed by square.
function startMap() {
  const map = new Map();
  const back = ["r","n","b","q","k","b","n","r"];
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
  // en-passant: pawn moves diagonally to an empty square -> capture passed pawn
  if (piece.role === "pawn" && from[0] !== to[0] && !map.get(to)) {
    map.delete(to[0] + from[1]);
  }
  // castling: king two squares -> move the rook too
  if (piece.role === "king" && Math.abs(from.charCodeAt(0) - to.charCodeAt(0)) === 2) {
    const rank = from[1];
    if (to[0] === "g") { map.set("f" + rank, map.get("h" + rank)); map.delete("h" + rank); }
    if (to[0] === "c") { map.set("d" + rank, map.get("a" + rank)); map.delete("a" + rank); }
  }
  map.set(to, promo ? { role: roleOf(promo), color: piece.color } : piece);
}
function mapToCg(map) {
  const pieces = new Map();
  for (const [s, p] of map) pieces.set(s, { role: p.role, color: p.color });
  return pieces;
}

function renderReplay() {
  const map = startMap();
  for (let i = 0; i < replay.idx; i++) applyUci(map, replay.moves[i]);
  const last = replay.idx > 0 ? lastMovePair(replay.moves[replay.idx - 1]) : undefined;
  board.set({ fen: undefined, lastMove: last });
  board.setPieces(mapToCg(map));
  document.getElementById("rp-status").textContent =
    `${replay.idx}/${replay.moves.length}  ${replay.result || ""}`;
}

function wireReplayControls() {
  document.getElementById("rp-prev").onclick = () => { if (replay.idx > 0) { replay.idx--; renderReplay(); } };
  document.getElementById("rp-next").onclick = () => { if (replay.idx < replay.moves.length) { replay.idx++; renderReplay(); } };
  document.getElementById("rp-play").onclick = togglePlay;
}
function togglePlay() {
  if (replay.timer) { stopPlay(); return; }
  replay.timer = setInterval(() => {
    if (replay.idx >= replay.moves.length) { stopPlay(); return; }
    replay.idx++; renderReplay();
  }, 600);
}
function stopPlay() { if (replay.timer) { clearInterval(replay.timer); replay.timer = null; } }

init();
