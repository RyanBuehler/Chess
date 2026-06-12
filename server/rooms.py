"""GameRoom + the human-vs-agent Play websocket.

A GameRoom owns one chess.Board and an optional agent (NetMCTSPlayer, CPU by
default). The Play protocol is JSON over /ws/play:
  client -> {type:"new", run_id, checkpoint, simulations, color}
  client -> {type:"move", uci}
  server -> {type:"state", fen, last_move, eval, thoughts, status, turn}
  server -> {type:"error", message}
The agent reuses M6's NetMCTSPlayer (add_noise=False, argmax visits) so UI play
is the same deterministic search the evaluator rates. Agent move computation runs
in a thread executor so it never blocks the event loop.
"""
import asyncio
from pathlib import Path

import chess

from chessrl.config.config import RunConfig
from server import catalog


def _status(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.outcome(claim_draw=True) is not None:
        return "draw"
    return "playing"


class GameRoom:
    """One human-vs-agent board. agent_color is the color the AGENT plays."""

    def __init__(self):
        self.board = chess.Board()
        self.agent = None
        self.agent_color = chess.BLACK
        self.last_move = None

    def load_agent(self, runs_root: Path, run_id: str, checkpoint: str,
                   simulations: int, human_color: str, device: str) -> None:
        from chessrl.evaluation.players import NetMCTSPlayer

        rdir = catalog.run_dir(runs_root, run_id)
        if rdir is None:
            raise ValueError("unknown run")
        if not catalog._safe_name(checkpoint) or not checkpoint.endswith(".pt"):
            raise ValueError("bad checkpoint")
        ckpt_path = rdir / "checkpoints" / checkpoint
        if not ckpt_path.exists():
            raise ValueError("checkpoint not found")
        run_cfg = RunConfig.from_json(rdir / "config.json")
        self.board = chess.Board()
        self.last_move = None
        self.agent = NetMCTSPlayer(
            "agent", ckpt_path, run_cfg.network, int(simulations), device=device,
        )
        self.agent_color = chess.BLACK if human_color == "white" else chess.WHITE

    def agent_to_move(self) -> bool:
        return (self.agent is not None
                and self.board.outcome(claim_draw=True) is None
                and self.board.turn == self.agent_color)

    def apply_human(self, uci: str) -> bool:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return False
        if mv not in self.board.legal_moves:
            return False
        self.board.push(mv)
        self.last_move = uci
        return True

    def agent_move(self) -> str:
        """Synchronous agent move (run in a thread executor by the caller).
        Returns the UCI played; sets last_move."""
        mv = self.agent.play(self.board)
        self.board.push(mv)
        self.last_move = mv.uci()
        return self.last_move

    def thoughts(self) -> list:
        """Top-5 (uci, visit_frac) from the agent's last search root, or []."""
        return list(getattr(self.agent, "_last_thoughts", [])) if self.agent else []

    def root_q(self) -> float:
        return float(getattr(self.agent, "_last_root_q", 0.0))

    def state_msg(self, mover: str | None = None) -> dict:
        """Build the state payload for the client.

        mover: "human" | "agent" | None (start-of-game / unknown).
        The JS eval-bar uses mover to decide when to normalize eval and in which
        direction, because msg.eval is always from the agent's search perspective.
        """
        return {
            "type": "state",
            "fen": self.board.fen(),
            "last_move": self.last_move,
            "eval": self.root_q(),
            "thoughts": self.thoughts(),
            "status": _status(self.board),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "mover": mover,
        }


def register_play_ws(app):
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/play")
    async def play_ws(ws: WebSocket):
        await ws.accept()
        room = GameRoom()
        runs_root = app.state.runs_root
        device = app.state.device
        try:
            while True:
                msg = await ws.receive_json()
                mtype = msg.get("type")
                if mtype == "new":
                    try:
                        await asyncio.to_thread(
                            room.load_agent, runs_root, msg.get("run_id"),
                            msg.get("checkpoint"), msg.get("simulations", 100),
                            msg.get("color", "white"), device,
                        )
                    except Exception as e:
                        await ws.send_json({"type": "error", "message": str(e)})
                        continue
                    if room.agent_to_move():
                        await asyncio.to_thread(room.agent_move)
                        await ws.send_json(room.state_msg(mover="agent"))
                    else:
                        await ws.send_json(room.state_msg(mover=None))
                elif mtype == "move":
                    if room.agent is None:
                        await ws.send_json({"type": "error", "message": "no game"})
                        continue
                    if not room.apply_human(msg.get("uci", "")):
                        await ws.send_json({"type": "error", "message": "illegal move"})
                        continue
                    await ws.send_json(room.state_msg(mover="human"))
                    if room.agent_to_move():
                        await asyncio.to_thread(room.agent_move)
                        await ws.send_json(room.state_msg(mover="agent"))
                else:
                    await ws.send_json({"type": "error", "message": "unknown type"})
        except WebSocketDisconnect:
            return
        finally:
            if room.agent is not None:
                getattr(room.agent, "close", lambda: None)()
