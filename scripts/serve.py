"""Run the chessrl web server (REST catalog + websockets + static UI).

  python scripts/serve.py --runs-root runs
  python scripts/serve.py --runs-root runs --feed-ports 5550,5551,5552,5553
  python scripts/serve.py --runs-root runs --device cuda          # opt-in GPU
  python scripts/serve.py --runs-root runs --stockfish tools/stockfish/stockfish.exe

The server reads run dirs READ-ONLY and submits arena results to
runs/ladder_inbox/ (never ladder.sqlite). Bind 127.0.0.1 by default (LAN-only,
trusted network -- no auth, per the spec).
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure the project root (parent of scripts/) is on sys.path so that
# `server`, `chessrl`, etc. resolve whether this script is run as:
#   python scripts/serve.py   (scripts/ is on path; project root is NOT)
#   python -m scripts.serve   (project root is on path already)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn

from server.app import create_app


@dataclass
class ServerConfig:
    stockfish_path: str = ""


def build(argv=None):
    ap = argparse.ArgumentParser(description="chessrl web server")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--feed-ports", default="", help="comma-separated PUB ports to subscribe (live view)")
    ap.add_argument("--device", default="cpu", help="server inference device (cpu default; cuda opt-in)")
    ap.add_argument("--stockfish", default="", help="path to a Stockfish binary (enables stockfish arena players)")
    args = ap.parse_args(argv)

    feed_ports = [int(p) for p in args.feed_ports.split(",") if p.strip()]
    cfg = ServerConfig(stockfish_path=args.stockfish)
    app = create_app(Path(args.runs_root), cfg=cfg, device=args.device)
    app.state.feed_ports = feed_ports
    return app, args


def main(argv=None) -> int:
    app, args = build(argv)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
