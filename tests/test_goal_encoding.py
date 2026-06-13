import chess
import numpy as np

from chessrl.goals.encoding import encode_goal, GOAL_PLANES
from chessrl.goals.templates import GoalTemplate, WIN_GOAL


def test_goal_planes_shape_and_compositional():
    p, deadline = encode_goal(
        GoalTemplate.capture(chess.KNIGHT, 15), remaining=10, protagonist=chess.WHITE
    )
    assert p.shape == (GOAL_PLANES, 8, 8)
    assert deadline == 10
    # a different piece type changes only the type channel, not the plane count (compositional)
    p2, _ = encode_goal(
        GoalTemplate.capture(chess.QUEEN, 15), remaining=10, protagonist=chess.WHITE
    )
    assert p2.shape == p.shape
    assert not np.array_equal(p, p2)              # the type channel differs


def test_win_goal_planes_are_neutral():
    p, _ = encode_goal(WIN_GOAL, remaining=200, protagonist=chess.WHITE)
    assert p.shape == (GOAL_PLANES, 8, 8)         # win-goal: a reserved channel, no spatial mask


def test_deadline_scalar_is_remaining():
    _, d = encode_goal(GoalTemplate.check(20), remaining=7, protagonist=chess.WHITE)
    assert d == 7
    _, d0 = encode_goal(GoalTemplate.check(20), remaining=0, protagonist=chess.WHITE)
    assert d0 == 0


def test_reach_square_spatial_mask_mirrors_for_black():
    sq = chess.E2
    pw, _ = encode_goal(
        GoalTemplate.reach_square(chess.KNIGHT, square=sq, deadline=10),
        remaining=10, protagonist=chess.WHITE,
    )
    pb, _ = encode_goal(
        GoalTemplate.reach_square(chess.KNIGHT, square=sq, deadline=10),
        remaining=10, protagonist=chess.BLACK,
    )
    # The spatial mask is plane 0. White: e2 -> (rank1, file4). Black mirrors it.
    mask_w = pw[0]
    mask_b = pb[0]
    assert mask_w[chess.square_rank(sq), chess.square_file(sq)] == 1
    msq = chess.square_mirror(sq)
    assert mask_b[chess.square_rank(msq), chess.square_file(msq)] == 1
    assert not np.array_equal(mask_w, mask_b)


def test_win_goal_has_empty_spatial_mask():
    p, _ = encode_goal(WIN_GOAL, remaining=50, protagonist=chess.WHITE)
    assert p[0].sum() == 0                         # no spatial mask for win


def test_kind_channels_distinguish_goals():
    p_cap, _ = encode_goal(GoalTemplate.capture(chess.PAWN, 10), 10, chess.WHITE)
    p_chk, _ = encode_goal(GoalTemplate.check(10), 10, chess.WHITE)
    assert not np.array_equal(p_cap, p_chk)        # different kind channel set


def test_planes_are_float32():
    p, _ = encode_goal(GoalTemplate.capture(chess.PAWN, 10), 10, chess.WHITE)
    assert p.dtype == np.float32
