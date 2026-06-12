// web/common.js — tiny shared helpers (no framework, no build step).
// chessground@9 is pure ESM, so we import it here as a module.
import { Chessground } from "./vendor/chessground.min.js";

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
export function openWS(path, onMessage, onError) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}${path}`);
  ws.addEventListener("message", (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }
    onMessage(data, ws);
  });
  if (onError) {
    ws.addEventListener("error", () => onError(new Error(`websocket error: ${path}`)));
    ws.addEventListener("close", (e) => { if (!e.wasClean) onError(new Error("websocket closed")); });
  }
  return ws;
}
// chessground factory (ESM import above).
export function makeBoard(el, opts = {}) {
  return Chessground(el, Object.assign({
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
// Surface an error in a visible on-page banner (never console-only).
// Looks for/creates a #error-banner element fixed at the top of the page.
export function showError(msg) {
  let el = document.getElementById("error-banner");
  if (!el) {
    el = document.createElement("div");
    el.id = "error-banner";
    document.body.prepend(el);
  }
  el.textContent = String(msg);
  el.style.display = "block";
  // Mirror to console so devs still see it.
  console.warn("[chessrl-ui]", msg);
}
export function clearError() {
  const el = document.getElementById("error-banner");
  if (el) el.style.display = "none";
}
// Set a status element's text, and optionally mark it as an error.
export function setStatus(id, text, isError = false) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("is-error", !!isError);
}
