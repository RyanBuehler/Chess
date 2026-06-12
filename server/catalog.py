"""Read-only reading of the run-dir layout (config.json, state.json,
metrics.jsonl, elo.jsonl, checkpoints/, games/). Pure functions over a runs_root
Path; NOTHING here writes inside a run dir. All run-id / file-name inputs are
validated so a request can never escape runs_root (path-traversal safe)."""
import json
from io import StringIO as _StringIO
from pathlib import Path

import chess
import chess.pgn

_SAFE_REJECT = ("/", "\\", "..")


def _safe_name(name: str) -> bool:
    return bool(name) and not any(tok in name for tok in _SAFE_REJECT)


def run_dir(runs_root: Path, run_id: str) -> Path | None:
    """Resolved run dir if run_id is a safe, existing run (has config.json),
    else None."""
    if not _safe_name(run_id):
        return None
    d = Path(runs_root) / run_id
    if d.is_dir() and (d / "config.json").exists():
        return d
    return None


def list_runs(runs_root: Path) -> list[dict]:
    out = []
    root = Path(runs_root)
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.json").exists()):
        out.append({
            "run_id": d.name,
            "config": _read_json(d / "config.json", default={}),
            "state": _read_json(d / "state.json", default={}),
        })
    return out


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def list_checkpoints(rdir: Path) -> list[dict]:
    out = []
    for c in sorted((Path(rdir) / "checkpoints").glob("ckpt_*.pt")):
        try:
            step = int(c.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        out.append({"name": c.name, "step": step})
    return out


def list_games(rdir: Path) -> list[dict]:
    gdir = Path(rdir) / "games"
    if not gdir.exists():
        return []
    return [{"name": p.name} for p in sorted(gdir.glob("*.pgn"))]


def game_pgn_path(rdir: Path, name: str) -> Path | None:
    if not _safe_name(name) or not name.endswith(".pgn"):
        return None
    p = Path(rdir) / "games" / name
    return p if p.exists() else None


def pgn_to_moves(pgn_text: str) -> dict:
    """Parse a PGN into a UCI move list + result (client replays known-legal
    moves, no client-side legality engine needed)."""
    game = chess.pgn.read_game(_StringIO(pgn_text))
    if game is None:
        return {"moves": [], "result": "*"}
    board = game.board()
    moves = []
    for mv in game.mainline_moves():
        moves.append(mv.uci())
        board.push(mv)
    return {"moves": moves, "result": game.headers.get("Result", "*")}


def _read_json(path: Path, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return default
