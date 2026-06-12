import chess
import numpy as np

from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.selfplay.play import play_game
from tests.test_mcts import UniformEvaluator


def test_play_game_produces_record():
    mcts_cfg = MCTSConfig(simulations=8, temperature_moves=4)
    sp_cfg = SelfPlayConfig(ply_cap=20, games_per_iteration=1)
    rec, board, z = play_game(UniformEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0))
    assert 1 <= len(rec) <= 20
    assert z in (-1, 0, 1)
    assert board.move_stack  # at least one move was played
    # outcomes alternate sign correctly with respect to z
    if z != 0:
        assert rec.outcomes[0] == z  # position 0 is white to move


def test_ply_cap_adjudicates_draw():
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=2)
    sp_cfg = SelfPlayConfig(ply_cap=6)
    rec, board, z = play_game(UniformEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0))
    assert len(rec) <= 6
    if len(board.move_stack) == 6:   # hit the cap
        assert z == 0


class WhiteIsLostEvaluator(UniformEvaluator):
    """White-to-move positions evaluate as lost, Black's as won. (A constant
    -1 for ALL nodes would be self-contradictory: opponent replies would also
    look lost, making the root look winning after backup.)"""

    def evaluate(self, board):
        policy, _ = super().evaluate(board)
        return policy, (-1.0 if board.turn == chess.WHITE else 1.0)


def test_resignation_ends_game_early():
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=0)
    sp_cfg = SelfPlayConfig(ply_cap=100, resign_threshold=-0.5,
                            resign_consecutive=2, resign_playout_fraction=0.0)
    rec, board, z = play_game(WhiteIsLostEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0))
    assert len(rec) < 100            # white resigned long before the cap
    assert z == -1
