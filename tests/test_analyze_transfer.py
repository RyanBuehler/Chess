"""Pre-registered transfer-analysis tests (plan Task 5.5).

Synthetic Elo-vs-games curves exercise the three pre-registered outcomes
(spec sec 2): a CONFIRM case (arm reaches theta in >= 15% fewer games, CI
excludes 0), a REFUTE case (CIs overlap / vanilla first), and an INCONCLUSIVE
case (arm never reaches theta, or a single seed cannot separate signal).
"""
from scripts.analyze_transfer import (
    CONFIRM,
    INCONCLUSIVE,
    REFUTE,
    analyze,
    games_to_theta,
    vanilla_theta,
)


def _curve(reach_games, theta=600.0, n=None):
    """A monotone curve whose trailing-window median first clears theta near
    ``reach_games``. Before reach_games elo sits below theta; at/after it sits
    above. Points are spaced 1000 games apart; the curve always extends a few
    points past reach_games so the crossing is observable."""
    if n is None:
        n = reach_games // 1000 + 4
    pts = []
    for i in range(n):
        g = (i + 1) * 1000
        elo = (theta + 50.0) if g >= reach_games else (theta - 100.0)
        pts.append((g, elo))
    return pts


def test_games_to_theta_sustained_crossing():
    # Below theta until 5000, then above. window=3 median clears theta once 2 of
    # the 3 trailing points are above -> crossing at 6000 (matches time_to_elo).
    curve = _curve(reach_games=5000, theta=600.0)
    assert games_to_theta(curve, 600.0, window=3) == 6000.0


def test_games_to_theta_never_reached_is_none():
    flat = [(g, 400.0) for g in (1000, 2000, 3000)]
    assert games_to_theta(flat, 600.0, window=3) is None


def test_vanilla_theta_is_min_seed_peak():
    c1 = [(1000, 700.0), (2000, 720.0)]
    c2 = [(1000, 650.0), (2000, 680.0)]
    # min of the seed peaks (720, 680) -> 680, the Elo every vanilla seed reaches.
    assert vanilla_theta([c1, c2]) == 680.0


def _seeds(reach_list, theta=600.0):
    return [_curve(r, theta=theta) for r in reach_list]


def test_confirm_case():
    # Vanilla reaches theta around 16-18k; arm reaches it around 8-9k -> ~50%
    # fewer games across several seeds, CI well clear of 0.
    theta = 600.0
    arm_curves = {
        "gp-vanilla": _seeds([16000, 17000, 18000, 16000], theta=theta),
        "gp-lp-goal": _seeds([8000, 9000, 8000, 9000], theta=theta),
    }
    res = analyze(arm_curves, "gp-vanilla", theta=theta, n_boot=500, seed=1)
    arm = res["arms"]["gp-lp-goal"]
    assert arm["decision"] == CONFIRM
    assert arm["reduction"] >= 0.15
    assert arm["ci_low"] > 0.0


def test_refute_case_vanilla_first():
    # Arm is SLOWER than vanilla -> negative reduction -> REFUTE.
    theta = 600.0
    arm_curves = {
        "gp-vanilla": _seeds([9000, 10000, 9000, 10000], theta=theta),
        "gp-random-goal": _seeds([16000, 17000, 16000, 17000], theta=theta),
    }
    res = analyze(arm_curves, "gp-vanilla", theta=theta, n_boot=500, seed=1)
    arm = res["arms"]["gp-random-goal"]
    assert arm["decision"] == REFUTE
    assert arm["reduction"] < 0.0


def test_refute_case_ci_overlaps_zero():
    # Arm and vanilla reach theta at indistinguishable, overlapping game counts
    # -> CI on the reduction straddles 0 -> REFUTE (not a confirmation).
    theta = 600.0
    arm_curves = {
        "gp-vanilla": _seeds([12000, 16000, 11000, 17000], theta=theta),
        "gp-always-win": _seeds([11000, 17000, 12000, 16000], theta=theta),
    }
    res = analyze(arm_curves, "gp-vanilla", theta=theta, n_boot=800, seed=3)
    arm = res["arms"]["gp-always-win"]
    assert arm["decision"] == REFUTE
    assert arm["ci_low"] <= 0.0 <= arm["ci_high"]


def test_inconclusive_arm_never_reaches_theta():
    # The arm never reaches theta within budget -> failed measurement.
    theta = 600.0
    never = [[(g, 400.0) for g in (1000, 2000, 3000)] for _ in range(3)]
    arm_curves = {
        "gp-vanilla": _seeds([8000, 9000, 8000], theta=theta),
        "gp-lp-goal": never,
    }
    res = analyze(arm_curves, "gp-vanilla", theta=theta, n_boot=500, seed=1)
    assert res["arms"]["gp-lp-goal"]["decision"] == INCONCLUSIVE


def test_inconclusive_single_seed_cannot_separate():
    # Phase 1: one seed per arm. Even a big apparent gap is INCONCLUSIVE because
    # the bootstrap CI is degenerate (width 0) and cannot exclude noise.
    theta = 600.0
    arm_curves = {
        "gp-vanilla": _seeds([18000], theta=theta),
        "gp-lp-goal": _seeds([8000], theta=theta),
    }
    res = analyze(arm_curves, "gp-vanilla", theta=theta, n_boot=500, seed=1)
    assert res["arms"]["gp-lp-goal"]["decision"] == INCONCLUSIVE


def test_theta_read_off_vanilla_when_not_overridden():
    arm_curves = {
        "gp-vanilla": [[(1000, 500.0), (2000, 650.0), (3000, 660.0)]],
        "gp-lp-goal": [[(1000, 500.0), (2000, 650.0), (3000, 660.0)]],
    }
    res = analyze(arm_curves, "gp-vanilla", theta=None, n_boot=100, seed=1)
    assert res["theta"] == 660.0  # min seed-peak of vanilla
