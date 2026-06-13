"""Task 3.1 — goal-conditioned batched MCTS equivalence gate.

The goal-conditioned analogue of ``tests/test_batched_mcts.py``'s K=1 gate:
at ``leaves_per_tree == 1``, a single tree, ``add_noise=False``, and the SAME
seed, the batched goal-conditioned search must reproduce ``GoalReferenceMCTS``
EXACTLY (tol=0) on fixed positions and sim counts, for BOTH a tight sub-goal AND
the win-goal.

Both searches are driven by the SAME deterministic evaluator. To make the two
drive paths comparable, the evaluator is keyed on the FULL goal-conditioned
input — the concatenated board+goal planes plus the deadline scalar — which both
the reference (single-board ``evaluate``) and the batched (``evaluate_planes`` /
``evaluate_one_goal``) adapters compute via the same ``encode_board`` +
``encode_goal``. Same (position, goal, remaining) -> same (policy, value) on
both sides, so any divergence is a search-algebra bug, not evaluator noise.
"""
import hashlib

import chess
import numpy as np

from chessrl.chess_env.encoding import encode_board, to_model_input
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.config.config import MCTSConfig
from chessrl.goals.encoding import encode_goal
from chessrl.goals.templates import GoalTemplate, WIN_GOAL
from chessrl.mcts.batched import BatchedMCTS
from chessrl.mcts.reference import GoalReferenceMCTS


# --------------------------------------------------------------------------
# Shared deterministic core: a fixed pseudo-random (policy, achievement-prob)
# keyed on the concatenated board+goal planes and the deadline scalar. Stable
# across processes (Python's hash() is salted, so we hash bytes ourselves).
# --------------------------------------------------------------------------
def _pv_from_planes(planes: np.ndarray, deadline: float):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(planes, dtype=np.float32).tobytes())
    h.update(np.float32(deadline).tobytes())
    seed = int.from_bytes(h.digest()[:8], "little")
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(NUM_ACTIONS)
    policy = np.exp(logits - logits.max())
    policy = policy / policy.sum()
    # Achievement probability in (0, 1): a sigmoid of a fixed standard normal.
    p = float(1.0 / (1.0 + np.exp(-rng.standard_normal())))
    return policy.astype(np.float64), p


class _RefGoalEvaluator:
    """Reference (single-board) adapter: encodes board+goal exactly as the
    goal-conditioned net evaluator would, then keys the deterministic core on
    that encoding."""

    def evaluate(self, board, goal, remaining, protagonist):
        board_planes = to_model_input(encode_board(board))
        goal_planes, _ = encode_goal(goal, remaining, protagonist)
        planes = np.concatenate([board_planes, goal_planes.astype(np.float32)], axis=0)
        return _pv_from_planes(planes, remaining)


class _BatchedGoalEvaluator:
    """Batched adapter: receives the pre-encoded (board+goal) planes and the
    per-leaf deadline vector from the leaf-parking path, and keys the SAME
    deterministic core on them. ``evaluate_one_goal`` mirrors the single-leaf
    path used by init_tree / advance."""

    def evaluate_planes(self, planes_batch, deadlines):
        policies, values = [], []
        for planes, d in zip(planes_batch, deadlines):
            policy, p = _pv_from_planes(planes, d)
            policies.append(policy)
            values.append(p)
        return np.asarray(policies), np.asarray(values, dtype=np.float64)

    def evaluate_one_goal(self, board, goal, remaining, protagonist):
        board_planes = to_model_input(encode_board(board))
        goal_planes, _ = encode_goal(goal, remaining, protagonist)
        planes = np.concatenate([board_planes, goal_planes.astype(np.float32)], axis=0)
        return _pv_from_planes(planes, remaining)


_POSITIONS = [
    chess.Board(),  # startpos
    chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"),  # exd5 wins the queen
    chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"),
]

# A tight sub-goal (capture a knight within 6 plies) and the apex win goal.
_GOALS = [GoalTemplate.capture(chess.KNIGHT, deadline=6), WIN_GOAL]


def _ref_visits(board, goal, protagonist, sims, seed):
    cfg = MCTSConfig(simulations=sims)
    mcts = GoalReferenceMCTS(_RefGoalEvaluator(), cfg, rng=np.random.default_rng(seed))
    visits, root_v = mcts.search(board.copy(), goal=goal, protagonist=protagonist, add_noise=False)
    return visits, root_v


def _batched_visits(board, goal, protagonist, sims, seed):
    cfg = MCTSConfig(simulations=sims, leaves_per_tree=1)
    mcts = BatchedMCTS(
        _BatchedGoalEvaluator(),
        cfg,
        rng=np.random.default_rng(seed),
        goal=goal,
        protagonist=protagonist,
    )
    tree = mcts.init_tree(board.copy(), add_noise=False)
    mcts.run(tree)
    return mcts.visit_counts(tree), mcts.root_q(tree)


def test_k1_goal_equivalence_subgoal_and_win():
    for board in _POSITIONS:
        for goal in _GOALS:
            for sims in (16, 64):
                ref_visits, ref_v = _ref_visits(board, goal, board.turn, sims, seed=0)
                bat_visits, bat_v = _batched_visits(board, goal, board.turn, sims, seed=0)
                assert bat_visits == ref_visits, (
                    f"visit mismatch fen={board.fen()} goal={goal.key()} sims={sims}"
                )
                assert abs(bat_v - ref_v) < 1e-9, (
                    f"root value mismatch fen={board.fen()} goal={goal.key()} sims={sims}"
                )


def test_k1_goal_equivalence_protagonist_is_black():
    # Black to move and the protagonist: opponent-minimizes flips for White nodes.
    board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
    for goal in _GOALS:
        ref_visits, ref_v = _ref_visits(board, goal, chess.BLACK, sims=64, seed=3)
        bat_visits, bat_v = _batched_visits(board, goal, chess.BLACK, sims=64, seed=3)
        assert bat_visits == ref_visits, f"visit mismatch goal={goal.key()}"
        assert abs(bat_v - ref_v) < 1e-9


def test_k1_goal_visits_sum_to_simulations():
    board = chess.Board()
    visits, _ = _batched_visits(board, WIN_GOAL, chess.WHITE, sims=64, seed=0)
    assert sum(visits.values()) == 64


def test_goal_mode_requires_protagonist():
    import pytest

    cfg = MCTSConfig(simulations=8)
    with pytest.raises(ValueError):
        BatchedMCTS(_BatchedGoalEvaluator(), cfg, goal=WIN_GOAL, protagonist=None)
