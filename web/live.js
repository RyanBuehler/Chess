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
  // Generic ancillary metadata (e.g. goal info). Schema-agnostic: we render
  // whatever the feed carries (a key/value list or a labelled table), knowing
  // nothing about what it means.
  const auxEl = document.createElement("div");
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

// Render a frame's `aux` into a container. Two schema-agnostic forms:
//   - an array of [key, value] string pairs -> <dt>/<dd> list (simple/legacy);
//   - an object {cols, to_move, rows:[[label, v0, v1, ...], ...]} -> a <table>
//     with the to_move column highlighted (the two-side goal view: White & Black
//     shown together so the metadata never flip-flops with the turn).
// Empty/absent aux => render nothing. Never hardcodes any specific key/meaning.
export function renderAux(container, aux) {
  container.replaceChildren();
  if (!aux) return;
  if (Array.isArray(aux)) {
    for (const pair of aux) {
      if (!Array.isArray(pair) || pair.length < 2) continue;
      const dt = document.createElement("dt");
      dt.textContent = String(pair[0]);
      const dd = document.createElement("dd");
      dd.textContent = String(pair[1]);
      container.append(dt, dd);
    }
    return;
  }
  if (Array.isArray(aux.rows) && aux.rows.length) renderAuxTable(container, aux);
}

// Lay out {cols, to_move, rows} as an aligned table; the to_move column is
// marked (▸) and highlighted. Purely positional — no key is interpreted.
function renderAuxTable(container, aux) {
  const cols = Array.isArray(aux.cols) ? aux.cols : [];
  const toMove = Number.isInteger(aux.to_move) ? aux.to_move : -1;
  const table = document.createElement("table");
  table.className = "aux-table";
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th")); // empty corner cell
  cols.forEach((c, i) => {
    const th = document.createElement("th");
    th.textContent = (i === toMove ? "▸ " : "") + String(c);
    if (i === toMove) th.classList.add("tomove");
    head.appendChild(th);
  });
  table.appendChild(head);
  const tips = (aux.tips && typeof aux.tips === "object") ? aux.tips : {};
  for (const row of aux.rows) {
    if (!Array.isArray(row) || !row.length) continue;
    const isGoalRow = String(row[0]) === "goal";
    const tr = document.createElement("tr");
    const k = document.createElement("td");
    k.className = "aux-k";
    k.textContent = String(row[0]);
    tr.appendChild(k);
    for (let i = 1; i < row.length; i++) {
      const td = document.createElement("td");
      td.textContent = String(row[i]);
      if (i - 1 === toMove) td.classList.add("tomove");
      // Hover tooltip: on the goal row, a side cell with a tip shows the cluster's
      // delta fingerprint (chess-feature label + averaged feature deltas). Clusters
      // are opaque centroids; this is the post-hoc characterizer's description.
      if (isGoalRow) {
        const tip = tips[cols[i - 1]];
        if (tip && tip.features) {
          td.classList.add("aux-tip-host");
          // wrap the cell text so it still ellipsifies while the tip escapes the cell
          const span = document.createElement("span");
          span.className = "aux-cell-text";
          span.textContent = td.textContent;
          td.textContent = "";
          td.append(span, buildTip(tip));
        }
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
  container.appendChild(table);
}

// Build the hover tooltip for a goal cell: the chess-feature label plus a row per
// non-zero feature delta. Pure DOM, no libs.
function buildTip(tip) {
  const box = document.createElement("div");
  box.className = "aux-tip";
  if (tip.label) {
    const lab = document.createElement("div");
    lab.className = "aux-tip-label";
    lab.textContent = tip.label;
    box.appendChild(lab);
  }
  const feats = tip.features || {};
  for (const key of Object.keys(feats)) {
    const v = Number(feats[key]);
    if (!v) continue; // skip zero/empty deltas — only show what moved
    const line = document.createElement("div");
    line.className = "aux-tip-row";
    const name = document.createElement("span");
    name.textContent = key;
    const val = document.createElement("span");
    val.textContent = (v > 0 ? "+" : "") + (Math.round(v * 100) / 100);
    line.append(name, val);
    box.appendChild(line);
  }
  if (Number.isFinite(tip.cluster)) {
    const c = document.createElement("div");
    c.className = "aux-tip-foot";
    c.textContent = "cluster " + tip.cluster;
    box.appendChild(c);
  }
  return box;
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
