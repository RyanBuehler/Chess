# tests/test_batched_mcts.py
import chess
import numpy as np

from chessrl.chess_env.moves import NUM_ACTIONS, index_to_move
from chessrl.config.config import MCTSConfig
from chessrl.mcts.batched import BatchedMCTS, SearchTree
from chessrl.mcts.reference import ReferenceMCTS


class UniformBatchedEvaluator:
    """Batched analogue of tests.test_mcts.UniformEvaluator: uniform priors,
    value 0 for every board. Search quality comes entirely from terminal
    values, exactly as the reference UniformEvaluator."""

    def evaluate_many(self, boards):
        n = len(boards)
        policies = np.full((n, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        return policies, values


class UniformSingleEvaluator:
    """Single-board uniform evaluator for ReferenceMCTS (matches
    tests.test_mcts.UniformEvaluator)."""

    def evaluate(self, board):
        return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS), 0.0


EQUIV_FENS = [
    chess.STARTING_FEN,
    "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P1B2/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "6k1/8/6K1/8/8/8/8/R7 w - - 0 1",  # mate-in-1
]


def _ref_visits(fen, sims):
    cfg = MCTSConfig(simulations=sims)
    mcts = ReferenceMCTS(UniformSingleEvaluator(), cfg, rng=np.random.default_rng(0))
    visits, _ = mcts.search(chess.Board(fen), add_noise=False)
    return visits


def _batched_visits(fen, sims, k):
    cfg = MCTSConfig(simulations=sims, leaves_per_tree=k)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    tree = mcts.init_tree(chess.Board(fen), add_noise=False)
    mcts.run(tree)
    return mcts.visit_counts(tree)


def test_k1_exact_equivalence_with_reference():
    for fen in EQUIV_FENS:
        ref = _ref_visits(fen, 64)
        bat = _batched_visits(fen, 64, k=1)
        assert bat == ref, f"mismatch on {fen}"


def test_k1_visits_sum_to_simulations():
    bat = _batched_visits(chess.STARTING_FEN, 64, k=1)
    assert sum(bat.values()) == 64


def _best_move_batched(fen, sims, k):
    board = chess.Board(fen)
    visits = _batched_visits(fen, sims, k)
    idx = max(visits, key=visits.get)
    return index_to_move(idx, board.turn == chess.BLACK, board)


def test_batched_finds_mate_in_one_k4():
    mv = _best_move_batched("6k1/8/6K1/8/8/8/8/R7 w - - 0 1", sims=200, k=4)
    assert mv == chess.Move.from_uci("a1a8")


def test_batched_finds_mate_in_two_k4():
    mv = _best_move_batched("7k/8/5K2/8/8/8/8/R7 w - - 0 1", sims=1600, k=4)
    assert mv == chess.Move.from_uci("f6g6")


def test_k4_visit_sum_within_overshoot_bound():
    # Stop selecting once sims_done >= simulations; a final round may overshoot
    # by up to K-1. Visit total is in [simulations, simulations + K - 1].
    cfg = MCTSConfig(simulations=64, leaves_per_tree=4)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    tree = mcts.init_tree(chess.Board(), add_noise=False)
    mcts.run(tree)
    total = sum(mcts.visit_counts(tree).values())
    assert 64 <= total <= 64 + 4 - 1


def test_advance_reuses_subtree():
    cfg = MCTSConfig(simulations=64, leaves_per_tree=1)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    board = chess.Board()
    tree = mcts.init_tree(board, add_noise=False)
    mcts.run(tree)
    visits = mcts.visit_counts(tree)
    best = max(visits, key=visits.get)
    reused_count_before = visits[best]
    mcts.advance(tree, best)
    # After re-rooting, sims_done equals the reused child's visit count, so the
    # new root already carries accumulated statistics (reuse, not a fresh tree).
    assert tree.sims_done == reused_count_before
    assert tree.root.visit_count == reused_count_before
    mcts.run(tree)
    assert sum(mcts.visit_counts(tree).values()) >= 64  # topped up to simulations


def test_step_round_drives_multiple_trees():
    cfg = MCTSConfig(simulations=32, leaves_per_tree=2)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    trees = [mcts.init_tree(chess.Board(), add_noise=False) for _ in range(3)]
    while any(t.sims_done < cfg.simulations for t in trees):
        mcts.step_round(trees)
    for t in trees:
        total = sum(mcts.visit_counts(t).values())
        assert 32 <= total <= 32 + 2 - 1
