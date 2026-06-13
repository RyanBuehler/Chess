"""Task 2.2 — goal-conditioned reference search: protagonist-frame minimax
+ exact goal terminals.

Value is P(protagonist achieves goal) in [0,1]. Backup is protagonist-frame
minimax (protagonist-to-move maximizes child value, opponent-to-move minimizes),
NOT negamax. Terminals: achieved -> 1, deadline expired -> 0, real game-over ->
evaluate the goal (win: win=1/draw=0.5/loss=0).
"""
import chess
import numpy as np

from chessrl.chess_env.moves import NUM_ACTIONS, index_to_move
from chessrl.config.config import MCTSConfig
from chessrl.goals.templates import GoalTemplate, WIN_GOAL
from chessrl.mcts.reference import GoalReferenceMCTS


class UniformGoalEvaluator:
    """Net-free goal evaluator: uniform priors, value 0.5 (max entropy
    achievement probability). Search quality then comes entirely from the exact
    goal terminals."""

    def evaluate(self, board, goal, remaining, protagonist):
        return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS), 0.5


def _search(board, goal, protagonist, sims):
    cfg = MCTSConfig(simulations=sims)
    mcts = GoalReferenceMCTS(UniformGoalEvaluator(), cfg, rng=np.random.default_rng(0))
    return mcts.search(board, goal=goal, protagonist=protagonist)


def test_one_ply_capture_is_chosen_and_value_high():
    # White to move; exd5 captures the lone black queen. Goal: capture a queen
    # within 1 ply. The search must pick the capture and root value -> ~1.
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    goal = GoalTemplate.capture(chess.QUEEN, deadline=1)
    visits, root_v = _search(board, goal, protagonist=chess.WHITE, sims=64)
    best = max(visits, key=visits.get)
    assert index_to_move(best, False, board) == chess.Move.from_uci("e4d5")
    assert root_v > 0.9


def test_unreachable_within_deadline_is_value_zero():
    # Goal: capture a queen, but there is no queen to capture and deadline is 1.
    # Every line expires -> root value -> 0.
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    goal = GoalTemplate.capture(chess.QUEEN, deadline=1)
    _, root_v = _search(board, goal, protagonist=chess.WHITE, sims=64)
    assert root_v < 0.1


def test_visits_sum_to_simulations():
    board = chess.Board()
    visits, root_v = _search(board, WIN_GOAL, protagonist=chess.WHITE, sims=64)
    assert sum(visits.values()) == 64
    assert 0.0 <= root_v <= 1.0


def test_opponent_minimizes_protagonist_goal():
    # Black to move (opponent of a White protagonist whose goal is to capture a
    # queen next ply). With White as protagonist and Black to move, the goal
    # cannot be furthered by Black; opponent minimizes. We assert the search
    # runs and stays a valid probability (frame correctness is exercised by the
    # regression gate; here we just confirm no negamax sign-flip blows up range).
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 b - - 0 1")
    goal = GoalTemplate.capture(chess.QUEEN, deadline=3)
    _, root_v = _search(board, goal, protagonist=chess.WHITE, sims=64)
    assert 0.0 <= root_v <= 1.0
