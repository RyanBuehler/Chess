// web/live.js
import { openWS, makeBoard, lastMovePair } from "/common.js";

const MAX_BOARDS = 12;
const cells = new Map();   // game_id -> { board, el }

function ensureCell(gameId) {
  if (cells.has(gameId)) return cells.get(gameId);
  if (cells.size >= MAX_BOARDS) return null;   // sampled + capped at 12
  const wrap = document.createElement("div");
  wrap.className = "live-cell";
  const title = document.createElement("div");
  title.textContent = gameId;
  title.style.font = "11px monospace";
  const boardEl = document.createElement("div");
  boardEl.className = "board-sm";
  wrap.append(title, boardEl);
  document.getElementById("live-grid").appendChild(wrap);
  const board = makeBoard(boardEl, { coordinates: false });
  const cell = { board, el: wrap };
  cells.set(gameId, cell);
  return cell;
}

function onMsg(msg) {
  const hint = document.getElementById("live-hint");
  if (msg.type === "roster") {
    if (!msg.games.length) hint.textContent = "(no live feed — start a run with selfplay.feed_port set)";
    else hint.textContent = "";
    return;
  }
  if (msg.type === "update") {
    const g = msg.game;
    const cell = ensureCell(g.game_id);
    if (!cell) return;
    cell.board.set({ fen: g.fen.split(" ")[0], lastMove: lastMovePair(g.last_move_uci) });
    if (g.done) cell.el.style.opacity = 0.5;
  }
}

openWS("/ws/live", onMsg);
