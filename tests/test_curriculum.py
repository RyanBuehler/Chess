"""LP curriculum: Beta-Bernoulli posterior, windowed absolute learning-progress,
attempt-count gating, w(g) ∝ LP + β·novelty with the win-floor on top
(plan Task 4.2; spec sec 12)."""
import chess
import numpy as np

from chessrl.goals.curriculum import Curriculum
from chessrl.goals.repertoire import Repertoire
from chessrl.goals.templates import WIN_GOAL, GoalTemplate


def _rep(window=200):
    return Repertoire(lp_window=window, deadline_max=60)


def _drive(rep, g, outcomes):
    rep.ensure(g)
    for o in outcomes:
        rep.record_attempt(g, bool(o))


def test_untried_template_is_novelty_driven_not_lp():
    rep = _rep()
    g = GoalTemplate.capture(chess.KNIGHT, deadline=10)
    rep.ensure(g)  # zero attempts
    cur = Curriculum(rep, novelty_beta=1.0, min_attempts_for_lp=20, win_floor=0.2)
    # Below the attempt gate -> LP is suppressed, novelty carries the weight.
    assert cur.learning_progress(g) == 0.0
    assert cur.novelty(g) > 0.0


def test_improving_template_has_high_lp():
    rep = _rep(window=40)
    g = GoalTemplate.capture(chess.QUEEN, deadline=15)
    # Early half mostly fails, later half mostly succeeds -> strong LP.
    outcomes = [0] * 20 + [1] * 20
    _drive(rep, g, outcomes)
    cur = Curriculum(rep, novelty_beta=1.0, min_attempts_for_lp=20, win_floor=0.2)
    lp = cur.learning_progress(g)
    assert lp > 0.4, lp


def test_flat_at_zero_has_near_zero_lp():
    rep = _rep(window=40)
    g = GoalTemplate.capture(chess.ROOK, deadline=3)  # impossible-ish, always fails
    _drive(rep, g, [0] * 40)
    cur = Curriculum(rep, novelty_beta=1.0, min_attempts_for_lp=20, win_floor=0.2)
    assert cur.learning_progress(g) < 0.1


def test_flat_at_high_has_near_zero_lp():
    rep = _rep(window=40)
    g = GoalTemplate.capture(chess.PAWN, deadline=30)  # mastered, always succeeds
    _drive(rep, g, [1] * 40)
    cur = Curriculum(rep, novelty_beta=1.0, min_attempts_for_lp=20, win_floor=0.2)
    assert cur.learning_progress(g) < 0.1


def test_improving_outweighs_flat():
    rep = _rep(window=40)
    improving = GoalTemplate.capture(chess.QUEEN, deadline=15)
    mastered = GoalTemplate.capture(chess.PAWN, deadline=30)
    _drive(rep, improving, [0] * 20 + [1] * 20)
    _drive(rep, mastered, [1] * 40)
    cur = Curriculum(rep, novelty_beta=0.0, min_attempts_for_lp=20, win_floor=0.0)
    assert cur.learning_progress(improving) > cur.learning_progress(mastered)


def test_win_sampled_at_least_floor():
    rep = _rep(window=40)
    # A juicy improving sub-goal that would dominate LP-only sampling.
    g = GoalTemplate.capture(chess.QUEEN, deadline=15)
    _drive(rep, g, [0] * 20 + [1] * 20)
    cur = Curriculum(rep, novelty_beta=1.0, min_attempts_for_lp=20, win_floor=0.3)
    rng = np.random.default_rng(0)
    n = 4000
    wins = sum(1 for _ in range(n) if cur.sample(rng).is_win())
    frac = wins / n
    assert frac >= 0.3 - 0.03, frac  # >= floor (allow small sampling slack below)


def test_sample_returns_repertoire_template():
    rep = _rep(window=40)
    g = GoalTemplate.capture(chess.BISHOP, deadline=12)
    _drive(rep, g, [0, 1] * 20)
    cur = Curriculum(rep, novelty_beta=1.0, min_attempts_for_lp=20, win_floor=0.2)
    rng = np.random.default_rng(1)
    keys = {t.key() for t in rep.templates()}
    for _ in range(50):
        assert cur.sample(rng).key() in keys


def test_novelty_decays_with_attempts():
    rep = _rep(window=200)
    fresh = GoalTemplate.capture(chess.KNIGHT, deadline=10)
    tried = GoalTemplate.capture(chess.BISHOP, deadline=10)
    rep.ensure(fresh)
    _drive(rep, tried, [0, 1] * 30)  # 60 attempts
    cur = Curriculum(rep, novelty_beta=1.0, min_attempts_for_lp=20, win_floor=0.2)
    assert cur.novelty(fresh) > cur.novelty(tried)
