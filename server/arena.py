"""Arena websocket: pit any two players, live-adjustable delay, pause/step/stop,
and submit the result to runs/ladder_inbox/ (the M6 evaluator ingests it). The
server NEVER writes ladder.sqlite directly (spec: arena results go through the
inbox as JSON).

Protocol over /ws/arena:
  client -> {type:"start", white: spec, black: spec, delay_ms, opening_idx, max_plies}
  client -> {type:"set_delay", delay_ms} | {type:"pause"} | {type:"resume"}
            | {type:"step"} | {type:"stop"}
  server -> {type:"state", fen, last_move, ply, turn}
  server -> {type:"gameover", z, result, inbox}
  server -> {type:"error", message}
player spec: {kind:"random"|"greedy"|"minimax"} | {kind:"checkpoint", run_id,
  checkpoint, sims} | {kind:"stockfish", elo|nodes}  (stockfish only if configured)
"""
import asyncio
import json
import time
import uuid
from pathlib import Path

import chess

from chessrl.config.config import RunConfig
from chessrl.evaluation.match import _RESULT_STR
from chessrl.evaluation.openings import opening_board
from server import catalog


def _build_player(spec: dict, runs_root: Path, cfg, device: str):
    kind = spec.get("kind")
    if kind == "random":
        from chessrl.evaluation.players import RandomPlayer
        return RandomPlayer(seed=int(spec.get("seed", 0)))
    if kind == "greedy":
        from chessrl.evaluation.players import GreedyMaterialPlayer
        return GreedyMaterialPlayer(seed=int(spec.get("seed", 0)))
    if kind == "minimax":
        from chessrl.evaluation.players import MinimaxPlayer
        return MinimaxPlayer(depth=int(spec.get("depth", 2)), seed=int(spec.get("seed", 0)))
    if kind == "checkpoint":
        from chessrl.evaluation.players import NetMCTSPlayer
        rdir = catalog.run_dir(runs_root, spec.get("run_id"))
        if rdir is None:
            raise ValueError("unknown run for checkpoint player")
        name = spec.get("checkpoint", "")
        if not catalog._safe_name(name) or not name.endswith(".pt"):
            raise ValueError("bad checkpoint")
        ckpt = rdir / "checkpoints" / name
        if not ckpt.exists():
            raise ValueError("checkpoint not found")
        run_cfg = RunConfig.from_json(rdir / "config.json")
        label = f"{rdir.name}@{name}"
        return NetMCTSPlayer(label, ckpt, run_cfg.network, int(spec.get("sims", 100)), device=device)
    if kind == "stockfish":
        sf_path = getattr(cfg, "stockfish_path", "") if cfg is not None else ""
        if not sf_path:
            raise ValueError("stockfish not configured on this server")
        from chessrl.evaluation.players import StockfishPlayer
        if spec.get("elo") is not None:
            return StockfishPlayer(sf_path, elo=int(spec["elo"]))
        if spec.get("nodes") is not None:
            return StockfishPlayer(sf_path, nodes=int(spec["nodes"]))
        return StockfishPlayer(sf_path, movetime_ms=int(spec.get("movetime_ms", 100)))
    raise ValueError(f"unknown player kind: {kind}")


def _write_inbox(runs_root: Path, white: str, black: str, z: int, opening: int,
                 conditions: dict) -> str:
    inbox = Path(runs_root) / "ladder_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    name = f"arena_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.json"
    payload = {"white": white, "black": black, "z": int(z),
               "opening": int(opening), "conditions": conditions}
    (inbox / name).write_text(json.dumps(payload))
    return name


def _result_z(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0                       # ongoing-at-cap or draw -> adjudicated draw
    return 1 if outcome.winner == chess.WHITE else -1


def register_arena_ws(app):
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/arena")
    async def arena_ws(ws: WebSocket):
        await ws.accept()
        runs_root = app.state.runs_root
        cfg = app.state.cfg
        device = app.state.device
        white = black = None
        try:
            start = await ws.receive_json()
            if start.get("type") != "start":
                await ws.send_json({"type": "error", "message": "expected start"})
                return
            try:
                white = _build_player(start["white"], runs_root, cfg, device)
                black = _build_player(start["black"], runs_root, cfg, device)
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                return

            opening_idx = int(start.get("opening_idx", 0))
            board = opening_board(opening_idx)
            max_plies = int(start.get("max_plies", 200))
            delay_ms = int(start.get("delay_ms", 500))
            paused = False
            stopped = False

            step_requested = False  # True = play one move then re-pause

            async def drain_controls(block: bool):
                """Apply any queued control messages. If block, wait for one."""
                nonlocal delay_ms, paused, stopped, step_requested
                while True:
                    try:
                        if block:
                            ctrl = await ws.receive_json()
                            block = False
                        else:
                            ctrl = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                    except (asyncio.TimeoutError,):
                        return
                    t = ctrl.get("type")
                    if t == "set_delay":
                        delay_ms = int(ctrl.get("delay_ms", delay_ms))
                    elif t == "pause":
                        paused = True
                    elif t == "resume":
                        paused = False
                    elif t == "stop":
                        stopped = True
                        return
                    elif t == "step":
                        step_requested = True
                        return                # caller plays exactly one move, then re-pauses

            await ws.send_json({
                "type": "state", "fen": board.fen(), "last_move": None,
                "ply": len(board.move_stack),
                "turn": "white" if board.turn == chess.WHITE else "black",
            })

            while board.outcome(claim_draw=True) is None and len(board.move_stack) < max_plies:
                await drain_controls(block=False)
                if stopped:
                    break
                if paused and not step_requested:
                    await drain_controls(block=True)  # wait for resume/step/stop
                    if stopped:
                        break
                    if paused and not step_requested:
                        continue
                step_requested = False  # consume the step token
                player = white if board.turn == chess.WHITE else black
                mv = await asyncio.to_thread(player.play, board)
                board.push(mv)
                await ws.send_json({
                    "type": "state", "fen": board.fen(), "last_move": mv.uci(),
                    "ply": len(board.move_stack),
                    "turn": "white" if board.turn == chess.WHITE else "black",
                })
                if delay_ms > 0 and not paused:
                    # sleep in small slices so set_delay/pause/stop stay responsive
                    slept = 0
                    while slept < delay_ms and not stopped:
                        await drain_controls(block=False)
                        if stopped or paused or step_requested:
                            break
                        chunk = min(50, delay_ms - slept)
                        await asyncio.sleep(chunk / 1000.0)
                        slept += chunk

            z = _result_z(board)
            white_name = getattr(white, "name", "white")
            black_name = getattr(black, "name", "black")
            cond_w = white.conditions() if hasattr(white, "conditions") else {}
            cond = {"source": "arena", "white": white_name,
                    "black": black_name, **cond_w}
            inbox_name = _write_inbox(runs_root, white_name, black_name, z,
                                      opening_idx, cond)
            result_str = _RESULT_STR.get(z, "*")
            await ws.send_json({"type": "gameover", "z": z,
                                "result": result_str, "inbox": inbox_name})
        except WebSocketDisconnect:
            return
        finally:
            for p in (white, black):
                if p is not None:
                    getattr(p, "close", lambda: None)()
