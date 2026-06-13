"""Ladder players. Every player exposes .name, .play(board) -> chess.Move, and
optionally .close(). Floors are pure-Python and seeded-deterministic; the agent
wraps the reference MCTS with noise/temperature OFF; Stockfish wraps a pinned,
timeout-guarded, auto-restarting UCI subprocess.
"""
from pathlib import Path

import chess
import chess.engine
import numpy as np

# Material values (king = 0; captures of the king never occur in legal chess).
PIECE_VALUE = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
MATE_SCORE = 1000.0


def default_stockfish_path() -> Path | None:
    """Provisioned Stockfish binary, or None. tools/stockfish/stockfish.exe
    (Windows) or tools/stockfish/stockfish (POSIX), resolved relative to repo root."""
    root = Path(__file__).resolve().parents[2]   # chessrl/evaluation/players.py -> repo root
    for name in ("stockfish.exe", "stockfish"):
        cand = root / "tools" / "stockfish" / name
        if cand.exists():
            return cand
    return None


def _material(board: chess.Board, color: bool) -> float:
    total = 0.0
    for sq, piece in board.piece_map().items():
        v = PIECE_VALUE[piece.piece_type]
        total += v if piece.color == color else -v
    return total


class RandomPlayer:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def play(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        return moves[int(self._rng.integers(len(moves)))]


class GreedyMaterialPlayer:
    """Maximizes immediate material after the move (mate = +MATE_SCORE), with a
    seeded random tiebreak among equal-best moves."""

    name = "greedy"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def play(self, board: chess.Board) -> chess.Move:
        me = board.turn
        best_score, best_moves = -1e18, []
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                score = MATE_SCORE
            else:
                score = _material(board, me)
            board.pop()
            if score > best_score + 1e-9:
                best_score, best_moves = score, [mv]
            elif abs(score - best_score) <= 1e-9:
                best_moves.append(mv)
        return best_moves[int(self._rng.integers(len(best_moves)))]


class MinimaxPlayer:
    """Negamax (alpha-beta) with a material leaf eval at fixed depth (1-3), plus
    optional Gaussian eval noise on leaves for diversity, and a seeded random
    tiebreak among equal-best root moves. Eval is from the side-to-move's view."""

    def __init__(self, depth: int = 2, seed: int = 0, noise: float = 0.0):
        assert 1 <= depth <= 3
        self.depth = depth
        self.noise = noise
        self.name = f"minimax{depth}"
        self._rng = np.random.default_rng(seed)

    def _leaf_eval(self, board: chess.Board) -> float:
        val = _material(board, board.turn)
        if self.noise > 0.0:
            val += float(self._rng.normal(0.0, self.noise))
        return val

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if board.is_checkmate():
            return -MATE_SCORE                      # side to move is mated
        if board.is_game_over(claim_draw=True):
            return 0.0
        if depth == 0:
            return self._leaf_eval(board)
        best = -1e18
        for mv in board.legal_moves:
            board.push(mv)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def play(self, board: chess.Board) -> chess.Move:
        best_score, best_moves = -1e18, []
        for mv in board.legal_moves:
            board.push(mv)
            score = -self._negamax(board, self.depth - 1, -1e18, 1e18)
            board.pop()
            if score > best_score + 1e-9:
                best_score, best_moves = score, [mv]
            elif abs(score - best_score) <= 1e-9:
                best_moves.append(mv)
        return best_moves[int(self._rng.integers(len(best_moves)))]


class _SingleFromBatched:
    """Adapts a BatchedNetEvaluator (.evaluate_many) to the single-board
    .evaluate(board) -> (policy, value) the reference MCTS expects."""

    def __init__(self, batched):
        self._batched = batched

    def evaluate(self, board: chess.Board):
        policies, values = self._batched.evaluate_many([board])
        return policies[0], float(values[0])


class NetMCTSPlayer:
    """The agent. Loads a checkpoint, searches with the reference MCTS at a fixed
    simulation count, noise and temperature OFF (eval mode), plays argmax visits."""

    def __init__(
        self,
        name: str,
        checkpoint_path,
        network_cfg,
        simulations: int,
        device: str = "cpu",
        seed: int = 0,
    ):
        # Imported here so chessrl.evaluation.players has no hard torch import at
        # module load (the floors and ratings code must import without torch).
        from chessrl.config.config import MCTSConfig
        from chessrl.mcts.reference import ReferenceMCTS
        from chessrl.model.network import BatchedNetEvaluator

        self.name = name
        batched = BatchedNetEvaluator.from_checkpoint(
            checkpoint_path, network_cfg, device=device
        )
        self._eval = _SingleFromBatched(batched)
        self._cfg = MCTSConfig(simulations=simulations)
        self._mcts = ReferenceMCTS(self._eval, self._cfg, rng=np.random.default_rng(seed))

    def play(self, board: chess.Board) -> chess.Move:
        from chessrl.chess_env.moves import index_to_move

        visits, root_q = self._mcts.search(board, add_noise=False)
        best_idx = max(visits, key=visits.get)
        # M7: stash top-5 (uci, visit_frac) + root_q for the Play view's eval bar
        # and agent "thoughts". Harmless to evaluation (read only by the server).
        flip = board.turn == chess.BLACK
        total = float(sum(visits.values())) or 1.0
        top = sorted(visits.items(), key=lambda kv: kv[1], reverse=True)[:5]
        self._last_thoughts = [
            [index_to_move(idx, flip, board).uci(), c / total] for idx, c in top
        ]
        self._last_root_q = float(root_q)
        return index_to_move(best_idx, flip, board)


class GoalNetMCTSPlayer:
    """The agent for a GOAL arm, evaluated by conditioning on g=win (spec sec 15).

    Loads a goal-conditioned checkpoint and searches with the protagonist-frame
    minimax (GoalMCTS) under WIN_GOAL, so the eval-relevant ``pi(.|win)`` /
    ``V(.|win)`` are exercised exactly as in training. The regression gate
    guarantees this reproduces negamax for g=win, so Elo is comparable to the
    vanilla arm. Interface matches NetMCTSPlayer (.name, .play)."""

    def __init__(
        self,
        name: str,
        checkpoint_path,
        network_cfg,
        simulations: int,
        device: str = "cpu",
        seed: int = 0,
    ):
        from chessrl.config.config import MCTSConfig
        from chessrl.goals.templates import WIN_GOAL
        from chessrl.mcts.reference import GoalReferenceMCTS
        from chessrl.model.network import GoalNetEvaluator

        self.name = name
        self._eval = GoalNetEvaluator.from_checkpoint(
            checkpoint_path, network_cfg, device=device
        )
        self._cfg = MCTSConfig(simulations=simulations)
        self._mcts = GoalReferenceMCTS(self._eval, self._cfg, rng=np.random.default_rng(seed))
        self._goal = WIN_GOAL

    def play(self, board: chess.Board) -> chess.Move:
        from chessrl.chess_env.moves import index_to_move

        # Protagonist is the side to move: we are choosing this side's move under
        # the win-goal, exactly as self-play does on a g=win ply.
        visits, root_q = self._mcts.search(
            board, goal=self._goal, protagonist=board.turn, add_noise=False
        )
        best_idx = max(visits, key=visits.get)
        flip = board.turn == chess.BLACK
        total = float(sum(visits.values())) or 1.0
        top = sorted(visits.items(), key=lambda kv: kv[1], reverse=True)[:5]
        self._last_thoughts = [
            [index_to_move(idx, flip, board).uci(), c / total] for idx, c in top
        ]
        self._last_root_q = float(root_q)
        return index_to_move(best_idx, flip, board)


class StockfishError(RuntimeError):
    pass


class StockfishPlayer:
    """Pinned, timeout-guarded UCI engine wrapper with one auto-restart on
    error/timeout. Exactly one of (elo, nodes, movetime) governs strength:
      * elo   -> UCI_LimitStrength=true, UCI_Elo=<elo> (calibrated anchor)
      * nodes -> fixed node budget (dead-zone rung)
      * else  -> fixed movetime_ms think time
    Threads=1, Ponder=false always (reproducibility). conditions() returns the
    pinned options + engine version id for recording per match.
    """

    def __init__(
        self,
        path: "str | list[str]",
        *,
        elo: int | None = None,
        nodes: int | None = None,
        movetime_ms: int = 100,
        name: str | None = None,
        timeout_s: float = 5.0,
    ):
        self._path = path
        self._elo = elo
        self._nodes = nodes
        self._movetime_ms = movetime_ms
        self._timeout_s = timeout_s
        self.name = name or (
            f"sf_elo{elo}" if elo is not None
            else f"sf_nodes{nodes}" if nodes is not None
            else f"sf_mt{movetime_ms}"
        )
        self._engine = None
        self._engine_id = ""
        self._start()

    def _start(self) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(
            self._path, timeout=self._timeout_s
        )
        self._engine_id = self._engine.id.get("name", "stockfish")
        # Ponder is managed by python-chess internally and must NOT be passed to
        # configure(); setting it raises EngineError. Include only the options we
        # can actually set: Threads and (optionally) UCI strength-limiting options.
        opts = {"Threads": 1}
        if self._elo is not None:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = self._elo
        # Only configure options the engine actually advertises (older builds vary).
        supported = {k: v for k, v in opts.items() if k in self._engine.options}
        if supported:
            self._engine.configure(supported)

    def _restart(self) -> None:
        try:
            if self._engine is not None:
                self._engine.quit()
        except Exception:
            pass
        self._start()

    def _limit(self) -> chess.engine.Limit:
        if self._nodes is not None:
            # Include time= so SimpleEngine._timeout_for() returns a finite bound
            # (it returns None—wait forever—when limit.time is None). The time
            # ceiling activates asyncio.wait_for; the node budget still governs
            # normal play and the engine will stop at whichever comes first.
            return chess.engine.Limit(nodes=self._nodes, time=self._timeout_s)
        return chess.engine.Limit(time=self._movetime_ms / 1000.0)

    def play(self, board: chess.Board) -> chess.Move:
        for attempt in range(2):                    # one auto-restart retry
            try:
                result = self._engine.play(board, self._limit())
                if result.move is None:
                    raise StockfishError("engine returned no move")
                return result.move
            except StockfishError:
                # Re-raise directly: StockfishError is already a terminal
                # failure (e.g. engine returned no move after restart). Folding
                # it into the generic handler below would trigger a second
                # _restart() on an already-failed attempt, masking the cause.
                raise
            except Exception:
                if attempt == 0:
                    self._restart()
                    continue
                raise StockfishError(f"{self.name} failed after restart")

    def conditions(self) -> dict:
        cond = {
            "engine_id": self._engine_id,
            "Threads": 1,
            "Ponder": False,
        }
        if self._elo is not None:
            cond["UCI_LimitStrength"] = True
            cond["UCI_Elo"] = self._elo
        if self._nodes is not None:
            cond["nodes"] = self._nodes
        else:
            cond["movetime_ms"] = self._movetime_ms
        return cond

    def close(self) -> None:
        try:
            if self._engine is not None:
                self._engine.quit()
        finally:
            self._engine = None
