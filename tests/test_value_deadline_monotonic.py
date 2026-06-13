"""Task 2.3 — deadline monotonicity / calibration gate (spec sec 8/9).

Hold (state, goal, protagonist) fixed and sweep moves-remaining from high to
zero: the achievement probability must be monotone non-increasing as remaining
shrinks, and exactly 0 at remaining == 0 with the goal unachieved.

This asserts the WIRING (the monotone-by-construction post-processing hook),
per the plan: lower `remaining` cannot increase V. The same hook is the
calibration check run during training.
"""
import chess
import numpy as np

from chessrl.config.config import NetworkConfig
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.goals.templates import WIN_GOAL
from chessrl.model.network import GoalNetEvaluator, PolicyValueNet, deadline_value_sweep


def _goal_evaluator(seed=0):
    torch_seed(seed)
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16), goal_conditioned=True)
    return GoalNetEvaluator(net, device="cpu")


def torch_seed(seed):
    import torch

    torch.manual_seed(seed)


class _NoisyRemainingEvaluator:
    """A deliberately NON-monotone raw evaluator: its value wobbles with
    remaining. The hook must clamp it to a monotone sweep."""

    def evaluate(self, board, goal, remaining, protagonist):
        policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
        # Non-monotone, in [0,1]: a sine wobble around a slope.
        v = 0.5 + 0.3 * np.sin(remaining) - 0.001 * remaining
        v = float(min(1.0, max(0.0, v)))
        return policy, v


def test_sweep_is_non_increasing_as_remaining_shrinks():
    board = chess.Board()
    ev = _NoisyRemainingEvaluator()
    remainings = list(range(40, -1, -1))  # high -> 0
    vals = deadline_value_sweep(ev, board, WIN_GOAL, chess.WHITE, remainings)
    # As remaining shrinks (list goes high->low), value must not increase.
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12, f"non-monotone: {a} -> {b}"


def test_zero_at_deadline_expired_unachieved():
    board = chess.Board()
    ev = _NoisyRemainingEvaluator()
    vals = deadline_value_sweep(ev, board, WIN_GOAL, chess.WHITE, [10, 5, 1, 0])
    assert vals[-1] == 0.0  # remaining == 0, win not achieved at startpos


def test_achieved_goal_is_one_regardless_of_remaining():
    # An already-achieved goal (real mate, protagonist wins) pins V to 1 for
    # every remaining — the "achieved" terminal dominates the sweep.
    ev = _NoisyRemainingEvaluator()
    mate = chess.Board("6k1/8/6K1/8/8/8/8/R7 w - - 0 1")
    mate.push(chess.Move.from_uci("a1a8"))  # Ra8# — White (protagonist) wins
    assert mate.is_checkmate()
    vals = deadline_value_sweep(ev, mate, WIN_GOAL, chess.WHITE, [20, 5, 0])
    assert all(v == 1.0 for v in vals)


def test_real_goal_net_sweep_runs_and_is_monotone():
    """End-to-end with the actual goal-conditioned net: the hook still yields a
    monotone, in-range sweep (wiring through GoalNetEvaluator)."""
    ev = _goal_evaluator(seed=0)
    board = chess.Board()
    remainings = list(range(30, -1, -2))
    vals = deadline_value_sweep(ev, board, WIN_GOAL, chess.WHITE, remainings)
    assert all(0.0 <= v <= 1.0 for v in vals)
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-9
    assert vals[-1] == 0.0
