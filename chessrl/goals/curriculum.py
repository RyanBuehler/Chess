"""Learning-progress curriculum over a Repertoire (plan Task 4.2; spec sec 12).

Selects which goal to assign next, biasing toward templates at the frontier of
competence (those whose success rate is *changing* — high absolute learning
progress) and toward novel, barely-tried templates.

LP estimator (Beta-Bernoulli, windowed absolute-LP)
---------------------------------------------------
Each template has a Beta-Bernoulli posterior over its success probability. The
learning-progress signal is the **absolute change in the posterior mean across
the window**, estimated *robustly* — NOT as a naive finite-difference of a noisy
Bernoulli rate (spec sec 12 explicitly warns against that). We split the
template's sliding window of recent outcomes into an older half and a newer half
and take the absolute difference of their Beta-posterior means:

    LP(g) = | mean(Beta(a + s_new, b + f_new)) - mean(Beta(a + s_old, b + f_old)) |

with a Beta(a,b) prior (a=b=1, uniform). The posterior shrinks each half-rate
toward 0.5 in proportion to how little data backs it, so noise in a short window
does not masquerade as progress. Both flat-at-zero (impossible) and flat-at-high
(mastered) yield ~0 LP because the two halves agree.

Attempt-count gating (spec sec 12)
----------------------------------
Templates with fewer than ``min_attempts_for_lp`` attempts have no trustworthy
LP estimate, so their LP is forced to 0 and the **novelty** term drives their
selection instead.

Novelty
-------
``novelty(g) = 1 / sqrt(1 + attempts)`` — high for untried templates, decaying as
the template is practiced. This is the exploration bonus that surfaces freshly
minted templates before their LP is meaningful.

Sampling distribution (spec sec 12)
-----------------------------------
``w(g) ∝ LP(g) + β·novelty(g)`` over all non-win templates, with the **win-floor**
applied on top: with probability ``win_floor`` the draw is forced to the apex
win-goal (regardless of LP), guaranteeing >= ``win_floor`` of assignments train
``π(·|win)`` / ``V(·|win)`` (the eval-relevant quantities, spec sec 7).
"""
from __future__ import annotations

import numpy as np

from chessrl.goals.repertoire import Repertoire
from chessrl.goals.templates import WIN_GOAL, GoalTemplate

# Uniform Beta prior (a=b=1): each half-window rate is shrunk toward 0.5.
_PRIOR_A = 1.0
_PRIOR_B = 1.0


def _beta_mean(successes: float, failures: float) -> float:
    a = _PRIOR_A + successes
    b = _PRIOR_B + failures
    return a / (a + b)


class Curriculum:
    """LP + novelty sampler over a ``Repertoire`` (spec sec 12)."""

    def __init__(
        self,
        repertoire: Repertoire,
        novelty_beta: float = 1.0,
        min_attempts_for_lp: int = 20,
        win_floor: float = 0.2,
    ):
        self.rep = repertoire
        self.novelty_beta = novelty_beta
        self.min_attempts_for_lp = min_attempts_for_lp
        self.win_floor = win_floor

    # --- signals ---------------------------------------------------------
    def learning_progress(self, goal: GoalTemplate) -> float:
        """Absolute windowed LP via split-window Beta-posterior means.

        Gated on attempt count: a template below ``min_attempts_for_lp`` returns
        0.0 (its selection is driven by novelty instead)."""
        st = self.rep.stats(goal)
        if st.attempts < self.min_attempts_for_lp:
            return 0.0
        window = list(st.window)
        n = len(window)
        if n < 4:
            return 0.0
        half = n // 2
        old, new = window[:half], window[half:]
        old_mean = _beta_mean(sum(old), len(old) - sum(old))
        new_mean = _beta_mean(sum(new), len(new) - sum(new))
        return abs(new_mean - old_mean)

    def novelty(self, goal: GoalTemplate) -> float:
        """Exploration bonus, decaying with attempts."""
        st = self.rep.stats(goal)
        return 1.0 / np.sqrt(1.0 + st.attempts)

    def weight(self, goal: GoalTemplate) -> float:
        """Unnormalized sampling weight w(g) = LP(g) + β·novelty(g)."""
        return self.learning_progress(goal) + self.novelty_beta * self.novelty(goal)

    # --- sampling --------------------------------------------------------
    def _subgoals(self) -> list[GoalTemplate]:
        return [t for t in self.rep.templates() if not t.is_win()]

    def sample(self, rng: np.random.Generator) -> GoalTemplate:
        """Draw one goal. With probability ``win_floor`` returns the apex
        win-goal; otherwise samples a sub-goal ∝ w(g). If there are no sub-goals
        yet, returns the win-goal."""
        if rng.random() < self.win_floor:
            return WIN_GOAL
        subgoals = self._subgoals()
        if not subgoals:
            return WIN_GOAL
        weights = np.array([self.weight(g) for g in subgoals], dtype=np.float64)
        total = weights.sum()
        if total <= 0.0 or not np.isfinite(total):
            # Degenerate (all-zero) weights: fall back to uniform over subgoals.
            probs = np.full(len(subgoals), 1.0 / len(subgoals))
        else:
            probs = weights / total
        return subgoals[int(rng.choice(len(subgoals), p=probs))]
