"""Take a full-page screenshot of the compare page."""
import threading
import time
from pathlib import Path

import uvicorn
from playwright.sync_api import sync_playwright

from server.app import create_app

runs_root = Path("C:/Chess/runs")
app = create_app(runs_root, device="cpu")
app.state.feed_ports = []


class _Server:
    def __init__(self):
        cfg = uvicorn.Config(
            app, host="127.0.0.1", port=8779, log_level="warning", lifespan="on"
        )
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)


srv = _Server()
srv.start()

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://127.0.0.1:8779/compare.html", wait_until="networkidle")
    # Wait for table to populate with real data rows
    page.wait_for_function(
        "() => document.querySelectorAll('#summary-body tr').length >= 1"
        " && !document.querySelector('#summary-body td[colspan]')",
        timeout=20000,
    )
    page.wait_for_selector("#elo-chart canvas", timeout=15000)
    page.wait_for_timeout(800)
    Path("ui_screenshots").mkdir(exist_ok=True)
    page.screenshot(path="ui_screenshots/compare.png", full_page=True)
    b.close()
    print("Screenshot saved to ui_screenshots/compare.png")

srv.stop()
