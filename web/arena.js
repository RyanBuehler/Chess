// web/arena.js
import { getJSON, openWS, makeBoard, lastMovePair, showError, setStatus } from "./common.js";

let ws = null, board = null, runs = [];
const KINDS = ["random", "greedy", "minimax", "checkpoint", "stockfish"];

async function init() {
  // Wire controls FIRST so they work even if async init fails.
  wireControls();
  try {
    board = makeBoard(document.getElementById("arena-board"));
  } catch (e) {
    showError("Board init failed: " + e.message);
  }

  try {
    runs = await getJSON("/api/runs");
    runs = runs.slice().reverse(); // newest first
  } catch (e) {
    showError("Failed to load runs: " + e.message);
    runs = [];
  }

  for (const side of ["white", "black"]) {
    const sel = document.getElementById(`${side}-kind`);
    for (const k of KINDS) {
      const o = document.createElement("option");
      o.value = k; o.textContent = k; sel.appendChild(o);
    }
    sel.onchange = () => toggleSideControls(side);
    await fillCkpts(side);
    toggleSideControls(side);
  }
}

async function fillCkpts(side) {
  const sel = document.getElementById(`${side}-ckpt`);
  sel.innerHTML = "";
  for (const r of runs) {
    let cks = [];
    try {
      cks = await getJSON(`/api/runs/${r.run_id}/checkpoints`);
    } catch (e) {
      showError("Failed to load checkpoints: " + e.message);
      continue;
    }
    for (const c of cks.slice().sort((a, b) => a.step - b.step)) {
      const o = document.createElement("option");
      o.value = JSON.stringify({ run_id: r.run_id, checkpoint: c.name });
      o.textContent = `${r.run_id} step ${c.step}`;
      sel.appendChild(o);
    }
  }
}

function toggleSideControls(side) {
  const kind = document.getElementById(`${side}-kind`).value;
  document.getElementById(`${side}-ckpt`).hidden = kind !== "checkpoint";
  const sf = document.getElementById(`${side}-sf`);
  if (sf) sf.hidden = kind !== "stockfish";
}

function spec(side) {
  const kind = document.getElementById(`${side}-kind`).value;
  if (kind === "checkpoint") {
    const raw = document.getElementById(`${side}-ckpt`).value || "{}";
    const v = JSON.parse(raw);
    return { kind, run_id: v.run_id, checkpoint: v.checkpoint, sims: 100 };
  }
  if (kind === "stockfish") {
    const elo = Number(document.getElementById(`${side}-sf-elo`).value) || 1320;
    return { kind, elo };
  }
  return { kind };
}

function wireControls() {
  const delay = document.getElementById("delay");
  delay.oninput = () => {
    document.getElementById("delay-val").textContent = delay.value + "ms";
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "set_delay", delay_ms: Number(delay.value) }));
  };
  document.getElementById("start").onclick = start;
  for (const t of ["pause", "resume", "step", "stop"]) {
    document.getElementById(t).onclick = () => {
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: t }));
    };
  }
}

function start() {
  if (ws) { try { ws.close(); } catch {} }
  setStatus("arena-status", "connecting…");
  const maxPlies = Number(document.getElementById("max-plies").value) || 200;
  ws = openWS("/ws/arena", (msg) => {
    if (msg.type === "error") {
      setStatus("arena-status", "error: " + msg.message, true);
      return;
    }
    if (msg.type === "state") {
      if (board) board.set({ fen: msg.fen.split(" ")[0], lastMove: lastMovePair(msg.last_move) });
      setStatus("arena-status", `ply ${msg.ply} — ${msg.turn} to move`);
    }
    if (msg.type === "gameover") {
      setStatus("arena-status", `game over: ${msg.result} (z=${msg.z}) → inbox ${msg.inbox}`);
    }
  }, (e) => setStatus("arena-status", "connection error: " + e.message, true));
  ws.addEventListener("open", () => {
    let payload;
    try {
      payload = {
        type: "start", white: spec("white"), black: spec("black"),
        delay_ms: Number(document.getElementById("delay").value),
        opening_idx: 0, max_plies: maxPlies,
      };
    } catch (e) {
      setStatus("arena-status", "bad player spec: " + e.message, true);
      return;
    }
    ws.send(JSON.stringify(payload));
    setStatus("arena-status", "started…");
  });
}

init();
