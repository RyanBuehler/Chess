// web/play.js
import { getJSON, openWS, makeBoard, lastMovePair } from "/common.js";

let ws = null, board = null, myColor = "white", lastFen = "start";

async function populateRuns() {
  const runs = await getJSON("/api/runs");
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
  const cks = await getJSON(`/api/runs/${runId}/checkpoints`);
  const sel = document.getElementById("ckpt");
  sel.innerHTML = "";
  for (const c of cks) {
    const o = document.createElement("option");
    o.value = c.name; o.textContent = `step ${c.step}`; sel.appendChild(o);
  }
}

function setupBoard() {
  board = makeBoard(document.getElementById("play-board"), {
    viewOnly: false,
    movable: { free: false, color: myColor, events: { after: onUserMove } },
  });
}
function onUserMove(from, to) {
  const uci = from + to;   // promotion: server accepts q-promo; UI keeps it simple
  ws.send(JSON.stringify({ type: "move", uci }));
}

function applyState(msg) {
  lastFen = msg.fen;
  board.set({ fen: msg.fen.split(" ")[0], lastMove: lastMovePair(msg.last_move),
    turn: msg.turn, movable: { color: msg.turn === myColor ? myColor : undefined } });
  // eval bar: msg.eval is from the agent's search perspective (agent's color).
  // Only update the bar after an agent move (msg.mover === "agent").
  if (msg.mover === "agent") {
    const agentColor = myColor === "white" ? "black" : "white";
    // Normalize from agent's perspective to white's absolute perspective.
    const q = agentColor === "white" ? msg.eval : -msg.eval;
    const pct = Math.round((q + 1) / 2 * 100);
    document.getElementById("eval-fill").style.height = pct + "%";
  }
  const t = document.getElementById("thoughts");
  t.innerHTML = "";
  for (const [uci, frac] of (msg.thoughts || [])) {
    const li = document.createElement("li");
    li.textContent = `${uci}  ${(frac * 100).toFixed(0)}%`;
    t.appendChild(li);
  }
  document.getElementById("status").textContent = msg.status;
}

function newGame() {
  myColor = document.getElementById("color").value;
  if (ws) ws.close();
  setupBoard();
  ws = openWS("/ws/play", (msg) => {
    if (msg.type === "error") { board.set({ fen: lastFen.split(" ")[0] }); document.getElementById("status").textContent = msg.message; return; }
    if (msg.type === "state") applyState(msg);
  });
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({
      type: "new", run_id: document.getElementById("run").value,
      checkpoint: document.getElementById("ckpt").value,
      simulations: Number(document.getElementById("sims").value),
      color: myColor,
    }));
  });
}

document.getElementById("newgame").onclick = newGame;
populateRuns();
