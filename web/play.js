// web/play.js
import { getJSON, openWS, makeBoard, lastMovePair, showError, setStatus } from "./common.js";

let ws = null, board = null, myColor = "white", lastFen = "start";

async function populateRuns() {
  let runs = [];
  try {
    runs = await getJSON("/api/runs");
  } catch (e) {
    showError("Failed to load runs: " + e.message);
    return;
  }
  runs = runs.slice().reverse(); // newest first
  const runSel = document.getElementById("run");
  runSel.innerHTML = "";
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.run_id; o.textContent = r.run_id; runSel.appendChild(o);
  }
  runSel.onchange = populateCkpts;
  if (runs.length) await populateCkpts();
}

async function populateCkpts() {
  const runId = document.getElementById("run").value;
  let cks = [];
  try {
    cks = await getJSON(`/api/runs/${runId}/checkpoints`);
  } catch (e) {
    showError("Failed to load checkpoints: " + e.message);
    return;
  }
  const sel = document.getElementById("ckpt");
  sel.innerHTML = "";
  // Smallest step first so the default pick is the lightest/fastest checkpoint.
  for (const c of cks.slice().sort((a, b) => a.step - b.step)) {
    const o = document.createElement("option");
    o.value = c.name; o.textContent = `step ${c.step}`; sel.appendChild(o);
  }
}

function setupBoard() {
  try {
    board = makeBoard(document.getElementById("play-board"), {
      viewOnly: false,
      movable: { free: false, color: myColor, showDests: true, events: { after: onUserMove } },
    });
  } catch (e) {
    showError("Board init failed: " + e.message);
  }
}

// Promotion: if a pawn reaches the last rank, auto-queen (send uci + "q").
function withPromotion(from, to) {
  const piece = board && board.state && board.state.pieces.get(to);
  const toRank = to[1];
  if (piece && piece.role === "pawn" && (toRank === "8" || toRank === "1")) {
    return from + to + "q";
  }
  return from + to;
}

function onUserMove(from, to) {
  sendMove(withPromotion(from, to));
}

function sendMove(uci) {
  if (!ws || ws.readyState !== 1) { showError("No active game — click New game first."); return; }
  ws.send(JSON.stringify({ type: "move", uci }));
}
// Test/accessibility hook: send a raw UCI move programmatically.
window.__sendMove = sendMove;

function applyState(msg) {
  lastFen = msg.fen;
  if (board) {
    board.set({
      fen: msg.fen.split(" ")[0], lastMove: lastMovePair(msg.last_move),
      turn: msg.turn,
      movable: { color: msg.turn === myColor ? myColor : undefined },
    });
  }
  if (msg.mover === "agent") {
    const agentColor = myColor === "white" ? "black" : "white";
    const q = agentColor === "white" ? msg.eval : -msg.eval;
    const pct = Math.round((q + 1) / 2 * 100);
    const fill = document.getElementById("eval-fill");
    if (fill) fill.style.height = Math.max(0, Math.min(100, pct)) + "%";
  }
  const t = document.getElementById("thoughts");
  t.innerHTML = "";
  for (const [uci, frac] of (msg.thoughts || [])) {
    const li = document.createElement("li");
    li.textContent = `${uci}  ${(frac * 100).toFixed(0)}%`;
    t.appendChild(li);
  }
  setStatus("status", `${msg.status}${msg.turn ? " — " + msg.turn + " to move" : ""}`);
}

function newGame() {
  myColor = document.getElementById("color").value;
  if (ws) { try { ws.close(); } catch {} }
  setupBoard();
  setStatus("status", "connecting…");
  ws = openWS("/ws/play", (msg) => {
    if (msg.type === "error") {
      if (board) board.set({ fen: lastFen.split(" ")[0] });
      setStatus("status", msg.message, true);
      return;
    }
    if (msg.type === "state") applyState(msg);
  }, (e) => setStatus("status", "connection error: " + e.message, true));
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({
      type: "new", run_id: document.getElementById("run").value,
      checkpoint: document.getElementById("ckpt").value,
      simulations: Number(document.getElementById("sims").value),
      color: myColor,
    }));
    setStatus("status", "game started — your move");
  });
}

// Wire handlers BEFORE any async board/network init.
document.getElementById("newgame").onclick = newGame;
const uciBtn = document.getElementById("uci-send");
if (uciBtn) {
  uciBtn.onclick = () => {
    const inp = document.getElementById("uci-input");
    const v = (inp.value || "").trim().toLowerCase();
    if (v) { sendMove(v); inp.value = ""; }
  };
}

populateRuns();
