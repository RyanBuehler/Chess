import chess
import numpy as np

from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.selfplay.concurrent import _Game, _is_false_positive, play_games_concurrent
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


class BlackIsLostBatchedEvaluator(UniformBatchedEvaluator):
    """Black-to-move positions evaluate as lost (from Black's perspective).
    When it is Black's turn the evaluator returns -1.0 (Black thinks it is
    losing), so Black's resign streak fires. When it is White's turn the
    evaluator returns +1.0 (White thinks it is winning).  This mirrors
    WhiteIsLostBatchedEvaluator but with the colours swapped."""

    def evaluate_many(self, boards):
        policies, values = super().evaluate_many(boards)
        values = np.array(
            [-1.0 if b.turn == chess.BLACK else 1.0 for b in boards], dtype=np.float32
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


def _make_game(allow_resign: bool, would_resign_side, z: int) -> _Game:
    """Construct a minimal _Game with the given terminal state for unit-testing
    _is_false_positive without running a full MCTS search."""
    g = object.__new__(_Game)
    g.allow_resign = allow_resign
    g.would_resign_side = would_resign_side   # chess.WHITE, chess.BLACK, or None
    g.z = z
    return g


def test_false_positive_detection_is_side_relative():
    """Regression: _is_false_positive must be side-relative.

    z is always from White's perspective (+1 = White won, -1 = Black won).
    A false positive is when the would-be resigner did NOT lose:
      - White-would-resign: fp when z >= 0 (White didn't lose: draw or White win)
      - Black-would-resign: fp when z <= 0 (Black didn't lose: draw or Black win)

    The bug: old code used `g.z >= 0` for BOTH sides, so:
      - Black-would-resign + z=+1 (Black genuinely lost) -> old: fp=True (WRONG)
      - Black-would-resign + z=-1 (Black actually won)   -> old: fp=False (WRONG)
    These two assertions exercise the two sides of the bug and will fail against
    the old code (which had a flat `return g.z >= 0`).
    """
    # --- Cases that must NOT be false positives (fp=False) ---

    # White-would-resign + z=-1: White lost -> genuine resignation signal, not fp
    assert _is_false_positive(_make_game(False, chess.WHITE, -1)) is False
    # Black-would-resign + z=+1: Black lost (z from White's POV) -> genuine, not fp
    assert _is_false_positive(_make_game(False, chess.BLACK, 1)) is False, (
        "Black-would-resign + z=+1 (Black genuinely lost) must be fp=False. "
        "Old code returned True (z >= 0 regardless of side) — this is the bug."
    )

    # --- Cases that MUST be false positives (fp=True) ---

    # White-would-resign + z=0: draw -> White didn't lose -> fp
    assert _is_false_positive(_make_game(False, chess.WHITE, 0)) is True
    # White-would-resign + z=+1: White won -> White didn't lose -> fp
    assert _is_false_positive(_make_game(False, chess.WHITE, 1)) is True
    # Black-would-resign + z=0: draw -> Black didn't lose -> fp
    assert _is_false_positive(_make_game(False, chess.BLACK, 0)) is True
    # Black-would-resign + z=-1: Black won -> Black didn't lose -> fp
    assert _is_false_positive(_make_game(False, chess.BLACK, -1)) is True, (
        "Black-would-resign + z=-1 (Black won) must be fp=True. "
        "Old code returned False (z >= 0 was False for z=-1) — this is the bug."
    )

    # --- Baseline: no resignation should always give fp=False ---
    assert _is_false_positive(_make_game(False, None, 0)) is False
    assert _is_false_positive(_make_game(True, chess.WHITE, 0)) is False
    assert _is_false_positive(_make_game(True, chess.BLACK, 1)) is False


