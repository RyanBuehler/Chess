// web/live.js
import { openWS, makeBoard, lastMovePair, showError, setStatus } from "./common.js";

const MAX_BOARDS = 12;
const cells = new Map();   // game_id -> { board, el }

function ensureCell(gameId) {
  if (cells.has(gameId)) return cells.get(gameId);
  if (cells.size >= MAX_BOARDS) return null;   // sampled + capped at 12
  const wrap = document.createElement("div");
  wrap.className = "live-cell card";
  const title = document.createElement("div");
  title.className = "live-title";
  title.textContent = gameId;
  const boardEl = document.createElement("div");
  boardEl.className = "board-sm";
  // Generic ancillary key/value list (e.g. goal info). Schema-agnostic: we
  // render whatever [key, value] pairs the feed carries, knowing nothing about
  // what they mean.
  const auxEl = document.createElement("dl");
  auxEl.className = "live-aux";
  wrap.append(title, boardEl, auxEl);
  document.getElementById("live-grid").appendChild(wrap);
  let board;
  try {
    board = makeBoard(boardEl, { coordinates: false });
  } catch (e) {
    showError("Board init failed: " + e.message);
    return null;
  }
  const cell = { board, el: wrap, auxEl };
  cells.set(gameId, cell);
  return cell;
}

// Render a generic list of [key, value] string pairs into a <dl>. Empty/absent
// aux => render nothing. Never hardcodes any specific key.
export function renderAux(dl, aux) {
  dl.replaceChildren();
  if (!Array.isArray(aux) || aux.length === 0) return;
  for (const pair of aux) {
    if (!Array.isArray(pair) || pair.length < 2) continue;
    const dt = document.createElement("dt");
    dt.textContent = String(pair[0]);
    const dd = document.createElement("dd");
    dd.textContent = String(pair[1]);
    dl.append(dt, dd);
  }
}

function onMsg(msg) {
  if (msg.type === "roster") {
    if (!msg.games || !msg.games.length) {
      setStatus("live-hint", "(no live feed — start a run with selfplay.feed_port set, then serve --feed-ports …)");
    } else {
      setStatus("live-hint", `${msg.games.length} active game(s)`);
    }
    return;
  }
  if (msg.type === "update") {
    const g = msg.game;
    if (!g || !g.game_id) return;
    const cell = ensureCell(g.game_id);
    if (!cell) return;
    const fen = (g.fen || "start").split(" ")[0];
    cell.board.set({ fen, lastMove: lastMovePair(g.last_move_uci) });
    cell.el.style.opacity = g.done ? 0.5 : 1;
    renderAux(cell.auxEl, g.aux);
  }
}

setStatus("live-hint", "connecting…");
openWS("/ws/live", onMsg, (e) => showError("Live feed connection error: " + e.message));
