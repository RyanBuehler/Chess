# Vendored web dependencies

These third-party assets are committed locally so the web UI runs with **zero CDN
references** (fully offline / LAN-only). All files were downloaded with
`Invoke-WebRequest` from jsDelivr.

| File | Package | Version | Source |
|------|---------|---------|--------|
| `chessground.min.js` | chessground | 9 (latest @9) | https://cdn.jsdelivr.net/npm/chessground@9/dist/chessground.min.js |
| `chessground.base.css` | chessground | 9 | https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.base.css |
| `chessground.brown.css` | chessground | 9 | https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.brown.css |
| `chessground.cburnett.css` | chessground | 9 | https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.cburnett.css |
| `uPlot.iife.min.js` | uplot | 1.6.31 | https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js |
| `uPlot.min.css` | uplot | 1.6.31 | https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.min.css |

## Notes

- **chessground@9 ships as pure ESM** (`export { Chessground, initModule }`). It
  must be loaded via `import { Chessground } from "./vendor/chessground.min.js"`,
  **not** a classic `<script>` tag (which throws a SyntaxError on the `export`
  keyword). `web/common.js` imports it as an ESM module.
- **uPlot** ships an IIFE build that defines `window.uPlot`, loaded via a classic
  `<script>` tag before the page module.
- All board square colors (`brown.css`) and piece images (`cburnett.css`) are
  inlined as `data:` URIs — there are **no external `url(http...)` / `url(../...)`
  references**, verified by grep. The board renders fully offline.

To refresh, re-run the downloads in `web/vendor/` and re-verify no external URLs:
`Select-String -Path *.css -Pattern "url\((?!'?data:)"`.
