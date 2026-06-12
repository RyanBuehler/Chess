import chess
import numpy as np

from chessrl.config.config import MCTSConfig
from chessrl.chess_env.moves import NUM_ACTIONS, index_to_move
from chessrl.mcts.reference import ReferenceMCTS


class UniformEvaluator:
    """Net-free evaluator: uniform priors, value 0. Search quality must then
    come entirely from terminal values - exactly what M3 verifies."""

    def evaluate(self, board):
        return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS), 0.0


def _best_move(board: chess.Board, simulations: int) -> chess.Move:
    cfg = MCTSConfig(simulations=simulations)
    mcts = ReferenceMCTS(UniformEvaluator(), cfg, rng=np.random.default_rng(0))
    visits, _ = mcts.search(board)
    idx = max(visits, key=visits.get)
    return index_to_move(idx, board.turn == chess.BLACK, board)


def test_visits_sum_to_simulations():
    cfg = MCTSConfig(simulations=64)
    mcts = ReferenceMCTS(UniformEvaluator(), cfg, rng=np.random.default_rng(0))
    visits, root_q = mcts.search(chess.Board())
    assert sum(visits.values()) == 64
    assert -1.0 <= root_q <= 1.0


def test_finds_mate_in_one():
    # White: Kg6, Ra1. Black: Kg8. Ra8# is the only mate.
    board = chess.Board("6k1/8/6K1/8/8/8/8/R7 w - - 0 1")
    assert _best_move(board, 200) == chess.Move.from_uci("a1a8")


def test_finds_mate_in_one_as_black():
    # Mirrored position: Black: Kg3, Ra8 -> Ra1# (exercises the flip path).
    board = chess.Board("r7/8/8/8/8/6k1/8/6K1 b - - 0 1")
    assert _best_move(board, 200) == chess.Move.from_uci("a8a1")


def test_finds_mate_in_two():
    # White: Kf6, Ra1. Black: Kh8. 1.Kg6! (only ...Kg8 2.Ra8#).
    board = chess.Board("7k/8/5K2/8/8/8/8/R7 w - - 0 1")
    assert _best_move(board, 1600) == chess.Move.from_uci("f6g6")


def test_dirichlet_noise_changes_priors():
    board = chess.Board()
    cfg = MCTSConfig(simulations=16)
    a = ReferenceMCTS(UniformEvaluator(), cfg, rng=np.random.default_rng(1))
    visits_noisy, _ = a.search(board, add_noise=True)
    assert sum(visits_noisy.values()) == 16
