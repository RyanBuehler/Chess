"""FastAPI app factory: read-only REST catalog + websocket routes (wired in
later tasks) + static web/ mount. create_app(runs_root, cfg, device) is the
single entry point used by scripts/serve.py and every test."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from server import catalog

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app(runs_root, cfg=None, device: str = "cpu") -> FastAPI:
    runs_root = Path(runs_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Shutdown: stop the LiveHub background thread if it was created.
        hub = getattr(app.state, "_live_hub", None)
        if hub is not None:
            hub.stop()

    app = FastAPI(title="chessrl server", lifespan=lifespan)
    app.state.runs_root = runs_root
    app.state.cfg = cfg
    app.state.device = device

    def _require_run(run_id: str) -> Path:
        rdir = catalog.run_dir(runs_root, run_id)
        if rdir is None:
            raise HTTPException(status_code=404, detail="run not found")
        return rdir

    @app.get("/api/runs")
    def get_runs():
        return catalog.list_runs(runs_root)

    @app.get("/api/runs/{run_id}/provenance")
    def get_provenance(run_id: str):
        rdir = _require_run(run_id)
        p = rdir / "provenance.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="provenance not found")
        return catalog._read_json(p, default={})

    @app.get("/api/runs/{run_id}/metrics")
    def get_metrics(run_id: str):
        return catalog.read_jsonl(_require_run(run_id) / "metrics.jsonl")

    @app.get("/api/runs/{run_id}/elo")
    def get_elo(run_id: str):
        return catalog.read_jsonl(_require_run(run_id) / "elo.jsonl")

    @app.get("/api/runs/{run_id}/goals")
    def get_goal_diagnostics(run_id: str):
        return catalog.goal_diagnostics(_require_run(run_id))

    @app.get("/api/runs/{run_id}/checkpoints")
    def get_checkpoints(run_id: str):
        return catalog.list_checkpoints(_require_run(run_id))

    @app.get("/api/runs/{run_id}/games")
    def get_games(run_id: str):
        return catalog.list_games(_require_run(run_id))

    @app.get("/api/runs/{run_id}/games/{name}/pgn", response_class=PlainTextResponse)
    def get_game_pgn(run_id: str, name: str):
        p = catalog.game_pgn_path(_require_run(run_id), name)
        if p is None:
            raise HTTPException(status_code=404, detail="game not found")
        return p.read_text()

    @app.get("/api/runs/{run_id}/games/{name}/moves")
    def get_game_moves(run_id: str, name: str):
        p = catalog.game_pgn_path(_require_run(run_id), name)
        if p is None:
            raise HTTPException(status_code=404, detail="game not found")
        return catalog.pgn_to_moves(p.read_text())

    # Websocket routes are attached by later tasks (rooms/arena/live) via
    # register_* functions to keep this factory cohesive.
    from server.rooms import register_play_ws
    from server.arena import register_arena_ws
    from server.live import register_live_ws

    register_play_ws(app)
    register_arena_ws(app)
    register_live_ws(app)

    # Static UI last so /api and /ws take precedence.
    if _WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
    return app
