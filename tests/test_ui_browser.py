"""Headless-browser end-to-end tests for the web UI (the verification gate).

These drive the *real* server (server.app.create_app pointed at the real
C:/Chess/runs, read-only) in a background uvicorn thread on port 8777, and a
headless Chromium via Playwright. Every test asserts there are NO console errors
or uncaught page errors, in addition to its functional checks.

All tests are marked @pytest.mark.slow so they're excluded from the default
suite (which runs `-m 'not slow'`). Run them with:

    .venv\\Scripts\\python -m pytest tests/test_ui_browser.py -m slow -q
"""
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

# Skip cleanly if playwright isn't installed, rather than erroring collection.
playwright_sync = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

import uvicorn  # noqa: E402

from server.app import create_app  # noqa: E402

HOST = "127.0.0.1"
PORT = 8777
BASE = f"http://{HOST}:{PORT}"


def _find_runs_root() -> Path:
    """Locate a runs/ dir with real data. Prefer the checkout's own runs/, but
    fall back to C:/Chess/runs (the canonical data) when running from a worktree
    whose runs/ is gitignored/empty."""
    here = Path(__file__).resolve().parents[1] / "runs"
    candidates = [here, Path("C:/Chess/runs"), Path.cwd() / "runs"]
    for c in candidates:
        if c.is_dir() and any(
            (p / "config.json").exists() for p in c.iterdir() if p.is_dir()
        ):
            return c
    return here


RUNS_ROOT = _find_runs_root()


@dataclass
class _ServerCfg:
    stockfish_path: str = ""


class _Server:
    """A uvicorn server running in a background thread we can tear down."""

    def __init__(self, app, host: str, port: int):
        config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                                lifespan="on")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout: float = 20.0):
        self.thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("server did not start in time")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="module")
def server():
    app = create_app(RUNS_ROOT, cfg=_ServerCfg(), device="cpu")
    app.state.feed_ports = []
    srv = _Server(app, HOST, PORT)
    srv.start()
    try:
        yield BASE
    finally:
        srv.stop()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@contextmanager
def page_with_console(browser):
    """A fresh page that records console errors + uncaught page errors."""
    page = browser.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
            if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    try:
        yield page, errors
    finally:
        page.close()


def _assert_no_console_errors(errors):
    assert not errors, "console/page errors:\n" + "\n".join(errors)


# --------------------------------------------------------------------------- #
# Dashboard — charts-only page (NO game browser)
# --------------------------------------------------------------------------- #
def test_dashboard(server, browser):
    with page_with_console(browser) as (page, errors):
        page.set_default_timeout(30000)
        page.goto(server + "/index.html", wait_until="networkidle")

        # Run list is populated.
        page.wait_for_selector("#run-list li")
        runs = page.query_selector_all("#run-list li")
        assert len(runs) >= 1

        # Click the newest baseline run (auto-selected first; click to be sure).
        baseline = page.query_selector("#run-list li:has-text('baseline')")
        assert baseline is not None, "no baseline run in list"
        baseline.click()

        # Charts render: uPlot draws <canvas> inside each chart div.
        page.wait_for_selector("#loss-chart canvas")
        page.wait_for_selector("#rate-chart canvas")
        page.wait_for_selector("#elo-chart canvas")
        assert page.query_selector("#loss-chart canvas") is not None
        assert page.query_selector("#elo-chart canvas") is not None

        # Game browser must NOT be on the dashboard page.
        assert page.query_selector("#browser") is None, \
            "#browser section should not exist on the dashboard"
        assert page.query_selector("#game-list") is None, \
            "#game-list should not exist on the dashboard"
        assert page.query_selector("#replay-board") is None, \
            "#replay-board should not exist on the dashboard"

        # Axis labels recorded on window.__chartAxes.
        axes = page.evaluate("() => window.__chartAxes || {}")
        assert axes.get("loss-chart", {}).get("x") == "step", \
            f"loss-chart x-axis label wrong: {axes.get('loss-chart')}"
        assert axes.get("loss-chart", {}).get("y") == "loss", \
            f"loss-chart y-axis label wrong: {axes.get('loss-chart')}"
        assert axes.get("rate-chart", {}).get("y") == "games / hour", \
            f"rate-chart y-axis label wrong: {axes.get('rate-chart')}"
        assert axes.get("elo-chart", {}).get("y") == "Elo", \
            f"elo-chart y-axis label wrong: {axes.get('elo-chart')}"

        _assert_no_console_errors(errors)


# --------------------------------------------------------------------------- #
# Replays — game browser + replayer
# --------------------------------------------------------------------------- #
def test_replays(server, browser):
    with page_with_console(browser) as (page, errors):
        page.set_default_timeout(30000)
        page.goto(server + "/replays.html", wait_until="networkidle")

        # Run list is populated.
        page.wait_for_selector("#run-list li")
        runs = page.query_selector_all("#run-list li")
        assert len(runs) >= 1

        # Click the newest baseline run.
        baseline = page.query_selector("#run-list li:has-text('baseline')")
        assert baseline is not None, "no baseline run in list"
        baseline.click()

        # Game list appears.
        page.wait_for_selector("#game-list li")

        # Click the first game, then play.
        page.query_selector("#game-list li").click()
        page.wait_for_selector("#replay-board piece")

        page.click("#rp-play")
        # Wait for the counter to advance well into the game (>= 5 plies) so
        # vacated squares exist — that's what catches stale-piece rendering.
        page.wait_for_function(
            r"() => parseInt(document.getElementById('rp-status').textContent) >= 5",
            timeout=20000,
        )
        page.click("#rp-play")  # pause so the ply index stays fixed
        status = page.inner_text("#rp-status")
        import re

        m = re.match(r"(\d+)/(\d+)", status.strip())
        assert m and int(m.group(1)) >= 5, f"counter did not advance: {status!r}"
        ply = int(m.group(1))

        # POSITION CORRECTNESS: rendered piece count must equal ground truth
        # from python-chess at the same ply. chessground's setPieces() is a
        # sparse diff that never clears vacated squares; anything but a
        # full-state set leaves clones behind (the "cloned pieces" bug).
        import chess as pychess
        import httpx

        run_id = page.inner_text("#sel-run").strip()
        game_name = page.query_selector("#game-list li.selected")
        game_name_text = game_name.inner_text().strip() if game_name else \
            page.inner_text("#game-list li").strip()
        moves = httpx.get(
            f"{server}/api/runs/{run_id}/games/{game_name_text}/moves"
        ).json()["moves"]
        truth = pychess.Board()
        for uci in moves[:ply]:
            truth.push(pychess.Move.from_uci(uci))
        expected = len(truth.piece_map())
        # Let chessground's move/fade animation (~200ms) settle: fading-out
        # elements remain in the DOM briefly and would inflate the count.
        page.wait_for_timeout(500)
        # Exclude chessground's permanent hidden drag-ghost placeholder.
        rendered = len(
            page.query_selector_all("#replay-board piece:not(.fading):not(.ghost)")
        )
        assert rendered == expected, (
            f"board shows {rendered} pieces but the position after ply {ply} "
            f"has {expected} (stale/cloned pieces?)"
        )

        _assert_no_console_errors(errors)


# --------------------------------------------------------------------------- #
# Auto-refresh — dashboard refreshes metrics endpoint periodically
# --------------------------------------------------------------------------- #
def test_autorefresh(server, browser):
    with page_with_console(browser) as (page, errors):
        page.set_default_timeout(30000)

        # Track requests to any metrics endpoint.
        metrics_requests: list[str] = []
        page.on("request", lambda req: metrics_requests.append(req.url)
                if "/metrics" in req.url else None)

        page.goto(server + "/index.html?refresh=1", wait_until="networkidle")

        # Wait for initial load: run list + charts rendered.
        page.wait_for_selector("#run-list li")
        page.wait_for_selector("#loss-chart canvas")

        # Snapshot how many metrics requests happened during initial load.
        initial_count = len(metrics_requests)

        # Wait ~3.5 seconds — at refresh=1s, expect at least 2 more fetches.
        page.wait_for_timeout(3500)

        additional = len(metrics_requests) - initial_count
        assert additional >= 2, (
            f"Expected >= 2 additional metrics fetches after 3.5s with refresh=1, "
            f"got {additional}"
        )

        # Zero console errors throughout.
        _assert_no_console_errors(errors)


# --------------------------------------------------------------------------- #
# Live
# --------------------------------------------------------------------------- #
def test_live(server, browser):
    with page_with_console(browser) as (page, errors):
        page.set_default_timeout(30000)
        page.goto(server + "/live.html", wait_until="networkidle")
        # With no feed configured, expect the no-feed hint (or, if a feed exists,
        # boards). Either way the hint element should have resolved off "connecting".
        page.wait_for_function(
            "() => { const h = document.getElementById('live-hint');"
            "return h && h.textContent && h.textContent !== 'connecting…'; }",
            timeout=15000,
        )
        hint = page.inner_text("#live-hint")
        boards = page.query_selector_all("#live-grid .live-cell")
        assert ("no live feed" in hint) or (len(boards) > 0), \
            f"neither no-feed hint nor boards present: hint={hint!r}"
        _assert_no_console_errors(errors)


# --------------------------------------------------------------------------- #
# Play
# --------------------------------------------------------------------------- #
def test_play(server, browser):
    with page_with_console(browser) as (page, errors):
        page.set_default_timeout(30000)
        page.goto(server + "/play.html", wait_until="networkidle")

        # Run + checkpoint dropdowns populate (options aren't "visible", use attached).
        page.wait_for_selector("#run option", state="attached")
        page.wait_for_selector("#ckpt option", state="attached")
        # Pick the baseline run.
        page.select_option("#run", label="baseline-20260612-001337")
        page.wait_for_function(
            "() => document.querySelectorAll('#ckpt option').length > 0")
        # Smallest checkpoint is first (sorted ascending in play.js).
        first_ckpt = page.eval_on_selector("#ckpt option", "o => o.value")
        page.select_option("#ckpt", value=first_ckpt)
        # Light search for speed.
        page.fill("#sims", "8")

        page.click("#newgame")
        # Wait for the game to start (status leaves "connecting").
        page.wait_for_function(
            "() => { const s = document.getElementById('status');"
            "return s && s.textContent && !s.textContent.includes('connecting'); }",
            timeout=30000,
        )
        # Send a legal opening move via the test hook; agent should reply.
        page.evaluate("() => window.__sendMove('e2e4')")
        # Agent reply (black) → board pieces present + status reflects a turn /
        # move list updates. Wait for the board to have pieces and status to be
        # a known game state.
        page.wait_for_selector("#play-board piece", timeout=30000)
        page.wait_for_function(
            "() => { const s = document.getElementById('status').textContent;"
            "return /playing|checkmate|stalemate|draw/.test(s); }",
            timeout=60000,
        )
        assert len(page.query_selector_all("#play-board piece")) > 0
        _assert_no_console_errors(errors)


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #
def test_arena(server, browser):
    with page_with_console(browser) as (page, errors):
        page.set_default_timeout(30000)
        page.goto(server + "/arena.html", wait_until="networkidle")

        page.wait_for_selector("#white-kind option", state="attached")
        page.select_option("#white-kind", value="random")
        page.select_option("#black-kind", value="greedy")
        # Delay 0 for speed; small ply cap so random/greedy ends quickly.
        page.eval_on_selector("#delay", "el => { el.value = '0'; }")
        page.fill("#max-plies", "60")

        page.click("#start")
        # Wait for gameover status.
        page.wait_for_function(
            "() => /game over/.test(document.getElementById('arena-status').textContent)",
            timeout=90000,
        )
        status = page.inner_text("#arena-status")
        assert "game over" in status, f"arena did not finish: {status!r}"
        _assert_no_console_errors(errors)


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
def test_compare(server, browser):
    """Compare page: summary table, Elo chart, x-axis switching, checkbox toggle.

    The real run (baseline-20260612-001337) has provenance.json WITHOUT the
    network key — the page must handle that gracefully by showing '—' rather
    than raising an error.
    """
    with page_with_console(browser) as (page, errors):
        page.set_default_timeout(30000)
        page.goto(server + "/compare.html", wait_until="networkidle")

        # Summary table has at least 1 data row (one per run).
        page.wait_for_function(
            "() => document.querySelectorAll('#summary-body tr').length >= 1"
            " && !document.querySelector('#summary-body td[colspan]')",
            timeout=20000,
        )
        rows = page.query_selector_all("#summary-body tr")
        assert len(rows) >= 1, "summary table has no run rows"

        # Elo chart canvas must be present.
        page.wait_for_selector("#elo-chart canvas", timeout=15000)
        assert page.query_selector("#elo-chart canvas") is not None, \
            "Elo chart canvas not found"

        # Refresh indicator is present (auto-refresh: 30s by default).
        indicator = page.query_selector("#refresh-indicator")
        assert indicator is not None, "refresh-indicator element missing on compare"
        indicator_text = indicator.inner_text()
        assert "auto-refresh" in indicator_text, \
            f"refresh-indicator text unexpected: {indicator_text!r}"

        # Switch x-axis to "hours" — chart should re-render without errors.
        page.click("input[name='xaxis'][value='hours']")
        # Give the chart a moment to re-render.
        page.wait_for_timeout(500)
        # Canvas must still be present after the switch.
        assert page.query_selector("#elo-chart canvas") is not None, \
            "Elo chart canvas missing after x-axis switch to hours"

        # Switch x-axis to "games".
        page.click("input[name='xaxis'][value='games']")
        page.wait_for_timeout(300)

        # Switch back to steps.
        page.click("input[name='xaxis'][value='steps']")
        page.wait_for_timeout(300)

        # Uncheck the first run checkbox — chart should redraw without errors.
        first_cb = page.query_selector("#summary-body input[type='checkbox']")
        if first_cb:
            first_cb.uncheck()
            page.wait_for_timeout(500)
            # Chart may show "No data." or a canvas with remaining runs.
            # Either way, no errors and the elo-chart div is still there.
            assert page.query_selector("#elo-chart") is not None, \
                "Elo chart container missing after unchecking run"

            # Re-check it for a clean state.
            first_cb.check()
            page.wait_for_timeout(300)

        # No console or page errors throughout.
        _assert_no_console_errors(errors)
