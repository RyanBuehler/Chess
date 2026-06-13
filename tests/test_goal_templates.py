import chess

from chessrl.goals.templates import GoalTemplate, WIN_GOAL


def test_template_key_individuates_by_piece_type():
    g1 = GoalTemplate.capture(chess.KNIGHT, deadline=15)
    g2 = GoalTemplate.capture(chess.QUEEN, deadline=15)
    assert g1.key() != g2.key()                   # type individuated
    assert GoalTemplate.capture(chess.KNIGHT, 20).key() == g1.key()  # deadline not in identity
    assert WIN_GOAL.is_win()


def test_win_goal_is_win_others_are_not():
    assert WIN_GOAL.is_win()
    assert GoalTemplate.win(deadline=40).is_win()
    assert not GoalTemplate.capture(chess.PAWN, deadline=5).is_win()
    assert not GoalTemplate.check(deadline=5).is_win()


def test_distinct_kinds_have_distinct_keys():
    keys = {
        GoalTemplate.capture(chess.KNIGHT, 10).key(),
        GoalTemplate.reach_rank(chess.PAWN, rank=7, deadline=10).key(),
        GoalTemplate.reach_square(chess.KNIGHT, square=chess.E5, deadline=10).key(),
        GoalTemplate.check(deadline=10).key(),
        GoalTemplate.castle(deadline=10).key(),
        GoalTemplate.promote(deadline=10).key(),
        GoalTemplate.win(deadline=10).key(),
    }
    assert len(keys) == 7                          # all kinds individuated


def test_reach_rank_individuated_by_rank_and_piece():
    a = GoalTemplate.reach_rank(chess.PAWN, rank=7, deadline=10)
    b = GoalTemplate.reach_rank(chess.PAWN, rank=6, deadline=10)
    c = GoalTemplate.reach_rank(chess.ROOK, rank=7, deadline=10)
    assert a.key() != b.key()                      # rank individuates
    assert a.key() != c.key()                      # piece individuates
    assert a.key() == GoalTemplate.reach_rank(chess.PAWN, rank=7, deadline=99).key()


def test_reach_square_individuated_by_square():
    a = GoalTemplate.reach_square(chess.KNIGHT, square=chess.E5, deadline=10)
    b = GoalTemplate.reach_square(chess.KNIGHT, square=chess.D5, deadline=10)
    assert a.key() != b.key()


def test_key_is_hashable_and_stable():
    g = GoalTemplate.capture(chess.QUEEN, deadline=15)
    d = {g.key(): 1}                               # hashable
    assert g.key() == g.key()                      # stable across calls


def test_deadline_preserved_on_instance():
    g = GoalTemplate.capture(chess.KNIGHT, deadline=15)
    assert g.deadline == 15
