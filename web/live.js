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
  wrap.append(title, boardEl);
  document.getElementById("live-grid").appendChild(wrap);
  let board;
  try {
    board = makeBoard(boardEl, { coordinates: false });
  } catch (e) {
    showError("Board init failed: " + e.message);
    return null;
  }
  const cell = { board, el: wrap };
  cells.set(gameId, cell);
  return cell;
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
  }
}

setStatus("live-hint", "connecting…");
openWS("/ws/live", onMsg, (e) => showError("Live feed connection error: " + e.message));
