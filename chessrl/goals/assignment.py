"""Per-side goal assignment keyed on ``GoalConfig.goal_mode`` (spec sec 7, 13).

This is the single seam Stage 4 swaps to plug in the minted repertoire + LP
curriculum. Today it provides four modes:

* ``none``        -> vanilla; no goals are assigned (the legacy path; callers
                     check this mode and never construct an assigner at all).
* ``always_win``  -> every side is assigned ``WIN_GOAL``.
* ``random``      -> uniform over a small fixed enumeration of sub-goal
                     templates (capture/check/castle/reach), with the
                     **win-floor** applied: at least ``win_floor`` fraction of
                     assignments are ``WIN_GOAL``.
* ``lp``          -> Stage-4 hook; falls back to ``random`` for now.

The win-floor is enforced *per assignment* by a Bernoulli draw at rate
``win_floor`` (each assignment is, with that probability, forced to win). Over
many assignments this yields >= ``win_floor`` fraction win-goals in expectation;
callers that need a hard floor over a known batch should rely on the law of
large numbers (the spec's floor is a fraction, not a hard count) — the test
asserts the empirical fraction clears the floor over many draws.
"""
from __future__ import annotations

import chess
import numpy as np

from chessrl.config.config import GoalConfig
from chessrl.goals.templates import WIN_GOAL, GoalTemplate

# A minimal fixed sub-goal repertoire. Stage 4 replaces this source with the
# minted repertoire + LP curriculum (see plan Stage 4). Deadlines are modest,
# capped by GoalConfig.deadline_max at construction.
def _default_subgoals(deadline_max: int) -> list[GoalTemplate]:
    d = max(1, min(deadline_max, 20))
    return [
        GoalTemplate.capture(chess.PAWN, deadline=d),
        GoalTemplate.capture(chess.KNIGHT, deadline=d),
        GoalTemplate.capture(chess.BISHOP, deadline=d),
        GoalTemplate.capture(chess.ROOK, deadline=d),
        GoalTemplate.capture(chess.QUEEN, deadline=d),
        GoalTemplate.check(deadline=max(1, min(deadline_max, 8))),
        GoalTemplate.castle(deadline=max(1, min(deadline_max, 12))),
        GoalTemplate.reach_rank(chess.PAWN, rank=7, deadline=d),
    ]


class GoalAssigner:
    """Draws one goal per side at game start, per ``GoalConfig.goal_mode``.

    Construct once per run (or per worker) and call ``assign()`` once per side
    per game. ``goal_mode == "none"`` is not handled here — callers branch on
    that mode to take the legacy vanilla path unchanged.
    """

    def __init__(self, cfg: GoalConfig, rng: np.random.Generator | None = None):
        if cfg.goal_mode == "none":
            raise ValueError("GoalAssigner is not used in goal_mode='none' (vanilla)")
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self._subgoals = _default_subgoals(cfg.deadline_max)

    def assign(self) -> GoalTemplate:
        """Return a goal for one side of one game."""
        mode = self.cfg.goal_mode
        if mode == "always_win":
            return WIN_GOAL
        # random or lp (lp falls back to random until Stage 4).
        # Win-floor first: with probability win_floor, force the apex goal.
        if self.rng.random() < self.cfg.win_floor:
            return WIN_GOAL
        return self._subgoals[int(self.rng.integers(len(self._subgoals)))]


def make_assigner(cfg: GoalConfig, rng: np.random.Generator | None = None) -> GoalAssigner | None:
    """Factory: returns a ``GoalAssigner`` for goal modes, or ``None`` for
    ``goal_mode == "none"`` (the caller takes the legacy vanilla path)."""
    if cfg.goal_mode == "none":
        return None
    return GoalAssigner(cfg, rng)
