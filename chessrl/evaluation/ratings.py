"""Regularized Davidson draw-model rating fit.

For a game between i and j (color-symmetric; the equal-colors match protocol
removes color advantage upstream so the fit ignores who was White), with
strengths pi = 10**(r/400):
    D          = pi_i + pi_j + nu*sqrt(pi_i*pi_j)
    P(i wins)  = pi_i / D
    P(j wins)  = pi_j / D
    P(draw)    = nu*sqrt(pi_i*pi_j) / D

We maximize the log-likelihood over the UNPINNED ratings and log(nu) by plain
numpy gradient ascent. Anchors are FIXED at their known Elo. Each unpinned rating
carries a Gaussian prior N(PRIOR_MEAN, PRIOR_SIGMA): without it an undefeated or
winless player's MLE diverges to +/-inf. PRIOR_MEAN=1000 is a deliberate FLOOR
CALIBRATION choice -- it anchors the un-anchored cloud near the random/greedy
floor region so early checkpoints land on a sensible absolute scale; PRIOR_SIGMA
=350 keeps the prior weak enough that ~100+ games dominate it (the 75%-vs-anchor
case lands within ~15 Elo of the unregularized 1190.85).

We work in the natural-log strength variable theta = r * ln(10)/400 so that
pi = exp(theta); gradients are clean and the 400-scale is reintroduced only when
converting back to Elo.
"""
import numpy as np

PRIOR_MEAN = 1000.0
PRIOR_SIGMA = 350.0
_SCALE = np.log(10.0) / 400.0          # r (Elo) -> theta (ln strength): theta = r*_SCALE


def fit_ratings(results, anchors: dict, iters: int = 4000, seed: int = 0):
    """results: iterable of (white_name, black_name, z) with z in {+1,0,-1}
    (White's perspective). anchors: {name: elo} pinned exactly. Returns
    (ratings: {name: elo_float}, nu: float)."""
    results = list(results)
    names = sorted({n for w, b, _ in results for n in (w, b)} | set(anchors))
    idx = {n: k for k, n in enumerate(names)}
    n = len(names)

    anchored = np.zeros(n, dtype=bool)
    theta = np.zeros(n, dtype=np.float64)
    for name, elo in anchors.items():
        anchored[idx[name]] = True
        theta[idx[name]] = elo * _SCALE
    # init unpinned at the prior mean
    for k in range(n):
        if not anchored[k]:
            theta[k] = PRIOR_MEAN * _SCALE

    if not results:
        ratings = {nm: theta[idx[nm]] / _SCALE for nm in names}
        return ratings, 1.0

    wi = np.array([idx[w] for w, _, _ in results])
    bj = np.array([idx[b] for _, b, _ in results])
    z = np.array([zz for _, _, zz in results], dtype=np.int64)
    is_w_win = (z == 1).astype(np.float64)     # White (i) wins
    is_b_win = (z == -1).astype(np.float64)    # Black (j) wins
    is_draw = (z == 0).astype(np.float64)

    log_nu = 0.0
    prior_var = (PRIOR_SIGMA * _SCALE) ** 2
    prior_mean_theta = PRIOR_MEAN * _SCALE

    lr = 0.05
    for it in range(iters):
        nu = np.exp(log_nu)
        ti, tj = theta[wi], theta[bj]
        # work in shifted log-space for numerical stability: terms exp(ti), exp(tj),
        # exp(0.5(ti+tj)+log_nu); subtract row max.
        a = ti
        b = tj
        c = 0.5 * (ti + tj) + log_nu
        m = np.maximum(np.maximum(a, b), c)
        ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
        denom = ea + eb + ec                    # = D / exp(m)
        p_i = ea / denom
        p_j = eb / denom
        p_d = ec / denom

        # Gradient of log-likelihood wrt theta_i for one game:
        #   d/dti log P(outcome) = [1{i wins} or 0.5*1{draw}] - (p_i + 0.5 p_d)
        # and symmetrically for theta_j. (c depends on 0.5*(ti+tj).)
        gi = is_w_win + 0.5 * is_draw - (p_i + 0.5 * p_d)
        gj = is_b_win + 0.5 * is_draw - (p_j + 0.5 * p_d)

        grad = np.zeros(n, dtype=np.float64)
        np.add.at(grad, wi, gi)
        np.add.at(grad, bj, gj)
        # Gaussian prior on unpinned thetas
        grad -= np.where(anchored, 0.0, (theta - prior_mean_theta) / prior_var)
        grad[anchored] = 0.0

        # Gradient wrt log_nu: sum over games of [1{draw} - p_d]
        g_lognu = float(np.sum(is_draw - p_d))

        step = lr / (1.0 + it / 500.0)          # simple decay schedule
        theta += step * grad
        log_nu += step * g_lognu

        if np.max(np.abs(step * grad)) < 1e-9 and abs(step * g_lognu) < 1e-9:
            break

    ratings = {nm: float(theta[idx[nm]] / _SCALE) for nm in names}
    for name, elo in anchors.items():
        ratings[name] = float(elo)              # exact pin (defends against drift)
    return ratings, float(np.exp(log_nu))
