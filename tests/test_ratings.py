import numpy as np

from chessrl.evaluation.ratings import fit_ratings


def _results_from_scores(white, black, wins_w, draws, wins_b):
    """Helper: build a result list (white, black, z) with given counts."""
    out = []
    out += [(white, black, 1)] * wins_w
    out += [(white, black, 0)] * draws
    out += [(white, black, -1)] * wins_b
    return out


def test_two_players_5050_no_draws_equal_ratings():
    # P vs anchor 1000, 50/50, no draws -> P ~ anchor within a couple Elo.
    res = _results_from_scores("P", "A", wins_w=50, draws=0, wins_b=50)
    res += _results_from_scores("A", "P", wins_w=50, draws=0, wins_b=50)
    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    assert abs(ratings["A"] - 1000.0) < 1e-6        # anchor stays pinned
    assert abs(ratings["P"] - 1000.0) < 2.0


def test_75pct_vs_anchor_is_about_1191():
    # P scores 75% vs anchor 1000 (no draws). Unregularized Elo = 1000 + 400*log10(3)
    # = 1190.85. The 1000-mean prior pulls it down slightly; with many games and
    # sigma=350 the pull is small. Tolerance ~15 Elo (and we assert direction).
    res = _results_from_scores("P", "A", wins_w=150, draws=0, wins_b=50)
    res += _results_from_scores("A", "P", wins_w=50, draws=0, wins_b=150)
    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    expected = 1000.0 + 400.0 * np.log10(3.0)       # ~1190.85
    assert ratings["P"] < expected                  # prior pulls toward 1000
    assert abs(ratings["P"] - expected) < 15.0


def test_all_wins_is_finite_and_regularized():
    # P beats anchor 1000 every game -> unregularized MLE diverges to +inf.
    # The prior must keep it finite, between +200 and +800 above the anchor.
    res = _results_from_scores("P", "A", wins_w=40, draws=0, wins_b=0)
    res += _results_from_scores("A", "P", wins_w=0, draws=0, wins_b=40)
    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    assert np.isfinite(ratings["P"])
    # Draw-free saturation + the weak 1000-mean prior settle this near +750 Elo;
    # the point is finiteness and regularization, not an exact value.
    assert 1200.0 < ratings["P"] < 1800.0


def test_draws_raise_nu():
    # Draw-heavy data -> larger nu than near-draw-free data.
    drawy = _results_from_scores("P", "A", wins_w=10, draws=80, wins_b=10)
    drawy += _results_from_scores("A", "P", wins_w=10, draws=80, wins_b=10)
    _, nu_high = fit_ratings(drawy, anchors={"A": 1000.0})

    sharp = _results_from_scores("P", "A", wins_w=50, draws=2, wins_b=48)
    sharp += _results_from_scores("A", "P", wins_w=48, draws=2, wins_b=50)
    _, nu_low = fit_ratings(sharp, anchors={"A": 1000.0})

    assert nu_high > nu_low


def test_recovers_known_ratings_from_synthetic_data():
    # Generate games FROM the model with known ratings, recover within ~30 Elo.
    rng = np.random.default_rng(0)
    true = {"A": 1000.0, "M": 1200.0, "S": 1400.0}
    nu_true = 0.5

    def pi(name):
        return 10.0 ** (true[name] / 400.0)

    def sample(w, b, n):
        out = []
        pw, pb = pi(w), pi(b)
        d = pw + pb + nu_true * np.sqrt(pw * pb)
        p_w, p_d = pw / d, nu_true * np.sqrt(pw * pb) / d
        for _ in range(n):
            u = rng.random()
            out.append((w, b, 1 if u < p_w else (0 if u < p_w + p_d else -1)))
        return out

    res = []
    for x, y in [("A", "M"), ("A", "S"), ("M", "S")]:
        res += sample(x, y, 300)
        res += sample(y, x, 300)

    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    # Order preserved and magnitudes recovered within ~30 Elo (A pinned).
    assert ratings["A"] == 1000.0
    assert ratings["M"] < ratings["S"]
    assert abs(ratings["M"] - 1200.0) < 30.0
    assert abs(ratings["S"] - 1400.0) < 30.0
