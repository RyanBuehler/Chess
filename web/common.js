// web/common.js — tiny shared helpers (no framework, no build step).
export async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}
export async function getText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.text();
}
export function openWS(path, onMessage) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}${path}`);
  ws.addEventListener("message", (e) => onMessage(JSON.parse(e.data), ws));
  return ws;
}
// chessground factory (loaded from CDN as window.Chessground).
export function makeBoard(el, opts = {}) {
  return window.Chessground(el, Object.assign({
    coordinates: false,
    viewOnly: true,
    fen: "start",
  }, opts));
}
// Build chessground "lastMove" highlight from a uci string.
export function lastMovePair(uci) {
  if (!uci || uci.length < 4) return undefined;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}
