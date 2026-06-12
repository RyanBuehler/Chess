// web/arena.js
import { getJSON, openWS, makeBoard, lastMovePair } from "/common.js";

let ws = null, board = null, runs = [];
const KINDS = ["random", "greedy", "minimax", "checkpoint", "stockfish"];

async function init() {
  runs = await getJSON("/api/runs");
  for (const side of ["white", "black"]) {
    const sel = document.getElementById(`${side}-kind`);
    for (const k of KINDS) { const o = document.createElement("option"); o.value = k; o.textContent = k; sel.appendChild(o); }
    sel.onchange = () => toggleCkpt(side);
    await fillCkpts(side);
  }
  board = makeBoard(document.getElementById("arena-board"));
  wireControls();
}
async function fillCkpts(side) {
  const sel = document.getElementById(`${side}-ckpt`);
  sel.innerHTML = "";
  for (const r of runs) {
    const cks = await getJSON(`/api/runs/${r.run_id}/checkpoints`);
    for (const c of cks) {
      const o = document.createElement("option");
      o.value = JSON.stringify({ run_id: r.run_id, checkpoint: c.name });
      o.textContent = `${r.run_id} step ${c.step}`;
      sel.appendChild(o);
    }
  }
}
function toggleCkpt(side) {
  const isCkpt = document.getElementById(`${side}-kind`).value === "checkpoint";
  document.getElementById(`${side}-ckpt`).hidden = !isCkpt;
}
function spec(side) {
  const kind = document.getElementById(`${side}-kind`).value;
  if (kind === "checkpoint") {
    const v = JSON.parse(document.getElementById(`${side}-ckpt`).value || "{}");
    return { kind, run_id: v.run_id, checkpoint: v.checkpoint, sims: 100 };
  }
  if (kind === "stockfish") return { kind, elo: 1320 };
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
    document.getElementById(t).onclick = () => ws && ws.send(JSON.stringify({ type: t }));
  }
}
function start() {
  if (ws) ws.close();
  ws = openWS("/ws/arena", (msg) => {
    if (msg.type === "error") { document.getElementById("arena-status").textContent = "error: " + msg.message; return; }
    if (msg.type === "state") board.set({ fen: msg.fen.split(" ")[0], lastMove: lastMovePair(msg.last_move) });
    if (msg.type === "gameover") document.getElementById("arena-status").textContent = `result ${msg.result} (z=${msg.z}) -> ${msg.inbox}`;
  });
  ws.addEventListener("open", () => ws.send(JSON.stringify({
    type: "start", white: spec("white"), black: spec("black"),
    delay_ms: Number(document.getElementById("delay").value), opening_idx: 0, max_plies: 200,
  })));
}

init();
