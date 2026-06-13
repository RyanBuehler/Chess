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


def goal_diagnostics(rdir: Path) -> dict:
    """Assemble the goal-run diagnostics payload (spec sec 16, Task 5.3).

    Reads ``metrics.jsonl`` (per-cycle goal series) + ``repertoire.json``
    (per-template stats) from a run dir and returns a compact, chart-ready dict:

        {
          "is_goal_run": bool,            # any goal series present at all
          "steps": [...],                 # x-axis (training step per cycle)
          "games": [...],                 # alt x-axis (cumulative games)
          "repertoire_size": [...],       # per-cycle template count (or null)
          "win_ply_fraction": [...],      # per-cycle fraction under g=win
          "goal_kinds": [...],            # union of goal kinds seen
          "achievement_rate": {kind: [...]},   # per-kind self-play rate per cycle
          "learning_progress": {kind: [...]},  # per-kind LP per cycle
          "wishful_thinking": {kind: {"self_play","vs_stockfish","gap"}},  # latest
          "repertoire": {                 # latest repertoire snapshot
            "size": int,
            "templates": [{kind,params,deadline,attempts,successes,window_rate}],
          },
        }

    Pure / read-only. Missing files degrade to empty series, never raise."""
    metrics = read_jsonl(Path(rdir) / "metrics.jsonl")

    steps: list = []
    games: list = []
    rep_size: list = []
    win_frac: list = []
    rate_by_kind: dict[str, list] = {}
    lp_by_kind: dict[str, list] = {}
    kinds: set[str] = set()
    latest_wishful: dict = {}

    # First pass: discover every goal kind across all cycles so each kind's
    # per-cycle series stays aligned (null where the kind is absent that cycle).
    for m in metrics:
        for key in ("goal_achievement_rate", "learning_progress"):
            block = m.get(key)
            if isinstance(block, dict):
                kinds.update(block.keys())
    sorted_kinds = sorted(kinds)
    for k in sorted_kinds:
        rate_by_kind[k] = []
        lp_by_kind[k] = []

    any_goal = False
    for m in metrics:
        steps.append(m.get("step"))
        games.append(m.get("games"))
        rep_size.append(m.get("repertoire_size"))
        win_frac.append(m.get("win_ply_fraction"))
        rate = m.get("goal_achievement_rate") or {}
        lp = m.get("learning_progress") or {}
        if rate or lp or m.get("win_ply_fraction") is not None:
            any_goal = True
        for k in sorted_kinds:
            rate_by_kind[k].append(rate.get(k))
            lp_by_kind[k].append(lp.get(k))
        wt = m.get("wishful_thinking")
        if isinstance(wt, dict) and wt:
            latest_wishful = wt

    rep = _read_json(Path(rdir) / "repertoire.json", default={})
    rep_templates = []
    for t in rep.get("templates", []) if isinstance(rep, dict) else []:
        attempts = int(t.get("attempts", 0) or 0)
        window = t.get("window", []) or []
        wrate = (sum(window) / len(window)) if window else 0.0
        rep_templates.append({
            "kind": t.get("kind"),
            "params": t.get("params", []),
            "deadline": t.get("deadline"),
            "attempts": attempts,
            "successes": int(t.get("successes", 0) or 0),
            "window_rate": wrate,
        })
    if rep_templates:
        any_goal = True

    return {
        "is_goal_run": any_goal,
        "steps": steps,
        "games": games,
        "repertoire_size": rep_size,
        "win_ply_fraction": win_frac,
        "goal_kinds": sorted_kinds,
        "achievement_rate": rate_by_kind,
        "learning_progress": lp_by_kind,
        "wishful_thinking": latest_wishful,
        "repertoire": {"size": len(rep_templates), "templates": rep_templates},
    }


def _read_json(path: Path, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return default
