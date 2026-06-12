import chess
import numpy as np

from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.selfplay.concurrent import play_games_concurrent
from tests.test_batched_mcts import UniformBatchedEvaluator


class WhiteIsLostBatchedEvaluator(UniformBatchedEvaluator):
    """White-to-move positions evaluate as lost, Black's as won (mirrors
    tests.test_selfplay.WhiteIsLostEvaluator but batched)."""

    def evaluate_many(self, boards):
        policies, values = super().evaluate_many(boards)
        values = np.array(
            [-1.0 if b.turn == chess.WHITE else 1.0 for b in boards], dtype=np.float32
        )
        return policies, values


def test_four_concurrent_games_valid_records():
    mcts_cfg = MCTSConfig(simulations=8, temperature_moves=4, leaves_per_tree=2)
    sp_cfg = SelfPlayConfig(ply_cap=20, resign_playout_fraction=0.0)
    results = play_games_concurrent(
        UniformBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0), num_games=4
    )
    assert len(results) == 4
    for rec, board, z, meta in results:
        assert 1 <= len(rec) <= 20
        assert z in (-1, 0, 1)
        assert meta["plies"] <= 20
        assert board.move_stack
        if z != 0:
            assert rec.outcomes[0] == z          # position 0 is white to move
        for key in ("plies", "z", "resigned", "playout", "would_resign", "fp"):
            assert key in meta


def test_resignation_meta_fields_present_for_playout_games():
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=0)
    sp_cfg = SelfPlayConfig(
        ply_cap=100, resign_threshold=-0.5, resign_consecutive=2,
        resign_playout_fraction=1.0,   # every game is a playout (resignation disabled)
    )
    results = play_games_concurrent(
        WhiteIsLostBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0), num_games=2
    )
    for rec, board, z, meta in results:
        assert meta["playout"] is True
        assert meta["resigned"] is False         # resignation disabled -> game ran to a real end
        # White looks lost throughout, so the resign criterion should have fired.
        assert meta["would_resign"] is True


def test_resignation_ends_game_when_enabled():
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=0)
    sp_cfg = SelfPlayConfig(
        ply_cap=100, resign_threshold=-0.5, resign_consecutive=2,
        resign_playout_fraction=0.0,   # resignation enabled
    )
    results = play_games_concurrent(
        WhiteIsLostBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0), num_games=2
    )
    for rec, board, z, meta in results:
        assert meta["resigned"] is True
        assert z == -1
        assert meta["plies"] < 100


def test_deterministic_with_same_seed():
    mcts_cfg = MCTSConfig(simulations=8, temperature_moves=4)
    sp_cfg = SelfPlayConfig(ply_cap=20, resign_playout_fraction=0.0)
    r1 = play_games_concurrent(
        UniformBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(7), num_games=3
    )
    r2 = play_games_concurrent(
        UniformBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(7), num_games=3
    )
    assert [z for *_, z, _meta in [(*x,) for x in r1]] == [z for *_, z, _meta in [(*x,) for x in r2]]
    for (rec1, _b1, z1, _m1), (rec2, _b2, z2, _m2) in zip(r1, r2):
        assert z1 == z2
        np.testing.assert_array_equal(rec1.played, rec2.played)
        np.testing.assert_array_equal(rec1.outcomes, rec2.outcomes)
