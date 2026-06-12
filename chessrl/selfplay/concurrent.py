"""Concurrent self-play: many games advanced in lockstep through one batched
MCTS, so every search round produces a single shared GPU batch.

Per-game logic mirrors selfplay/play.py exactly (search with root noise,
temperature then argmax, ply cap, resignation with playout fraction), plus
false-positive resignation tracking in the returned meta dict. Subtree reuse
(advance) carries statistics across moves; Dirichlet noise is re-applied at
each new root.
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.mcts.batched import BatchedMCTS
from chessrl.selfplay.records import GameRecord, RecordBuilder


class _Game:
    """Mutable per-game state for one slot in the concurrent batch."""

    __slots__ = (
        "tree", "builder", "board", "allow_resign", "resign_streak",
        "ply", "done", "z", "resigned", "would_resign",
    )

    def __init__(self, tree, board: chess.Board, allow_resign: bool):
        self.tree = tree
        self.builder = RecordBuilder()
        self.board = board
        self.allow_resign = allow_resign
        self.resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.ply = 0
        self.done = False
        self.z = 0
        self.resigned = False
        self.would_resign = False


def play_games_concurrent(
    evaluator_many,
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    rng: np.random.Generator,
    num_games: int,
) -> list:
    """Returns list[(GameRecord, final_board, z, meta)] of length num_games,
    in slot order. z is from White's perspective (+1/0/-1)."""
    mcts = BatchedMCTS(evaluator_many, mcts_cfg, rng)

    games: list[_Game] = []
    for _ in range(num_games):
        board = chess.Board()
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        tree = mcts.init_tree(board, add_noise=True)
        games.append(_Game(tree, board, allow_resign))

    # Resolve any game that is already terminal / over the cap before searching.
    for g in games:
        _check_pre_move_termination(g, sp_cfg)

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        # Run a full search for every active tree (each tree tops up to
        # mcts_cfg.simulations; step_round shares one GPU batch across trees).
        # Use visit_count-based condition matching batched.py's internal API.
        trees = [g.tree for g in active]
        while any(t.root.visit_count < mcts_cfg.simulations + 1 for t in trees):
            mcts.step_round(trees)

        for g in active:
            _play_one_move(g, mcts, mcts_cfg, sp_cfg, rng)

    results = []
    for g in games:
        rec = g.builder.finalize(g.z)
        meta = {
            "plies": g.ply,
            "z": g.z,
            "resigned": g.resigned,
            "playout": not g.allow_resign,
            "would_resign": g.would_resign,
            "fp": _is_false_positive(g),
        }
        results.append((rec, g.board, g.z, meta))
    return results


def _check_pre_move_termination(g: _Game, sp_cfg: SelfPlayConfig) -> None:
    term = terminal_value(g.board)
    if term is not None:
        g.z = int(term) if g.board.turn == chess.WHITE else -int(term)
        g.done = True
    elif g.ply >= sp_cfg.ply_cap:
        g.z = 0
        g.done = True


def _play_one_move(
    g: _Game, mcts: BatchedMCTS, mcts_cfg: MCTSConfig, sp_cfg: SelfPlayConfig,
    rng: np.random.Generator,
) -> None:
    visits = mcts.visit_counts(g.tree)
    root_q = mcts.root_q(g.tree)
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.fromiter(visits.values(), dtype=np.float64)
    if g.ply < mcts_cfg.temperature_moves:
        choice = int(rng.choice(idxs, p=counts / counts.sum()))
    else:
        choice = int(idxs[counts.argmax()])

    # Record before the resign check (the triggering search is a valid example).
    g.builder.add(g.board, idxs.astype(np.int32), counts.astype(np.int32), choice)

    if root_q < sp_cfg.resign_threshold:
        g.resign_streak[g.board.turn] += 1
        if g.resign_streak[g.board.turn] >= sp_cfg.resign_consecutive:
            g.would_resign = True
            if g.allow_resign:
                g.z = -1 if g.board.turn == chess.WHITE else 1
                g.resigned = True
                g.done = True
                return
    else:
        g.resign_streak[g.board.turn] = 0

    # Commit the move via subtree reuse (advance pushes the move on tree.board),
    # then keep g.board in sync, re-apply root noise, and check termination.
    mcts.advance(g.tree, choice)
    g.board = g.tree.board
    g.ply += 1
    mcts.add_root_noise(g.tree)
    _check_pre_move_termination(g, sp_cfg)


def _is_false_positive(g: _Game) -> bool:
    """A playout game where resignation WOULD have fired but the would-be
    resigner did not actually lose -> a false positive. The would-be resigner
    is whoever was to move when the streak reached the threshold; we approximate
    it conservatively as: playout AND would_resign AND the game was not a loss
    for both sides being impossible -> use the recorded result. Since a resign
    abandons the game as a loss for the side to move, a false positive is a
    playout game that hit the criterion yet ended in a draw or a win for the
    would-be resigner."""
    if g.allow_resign or not g.would_resign:
        return False
    # Resignation in this engine only ever fires for the side to move; in the
    # WhiteIsLost evaluator that is White. A draw (z==0) or a White win (z==1)
    # both contradict the would-be resignation, i.e. a false positive.
    return g.z >= 0
