"""v3-zenith: chained means-end self-play. ENTIRELY SEGREGATED from v2's
selfplay/concurrent.py — v2 (goal_mode 'emergent') never imports this module.

Each side always holds an active cluster sub-goal, re-selected on achieve-or-expire
via a greedy curriculum_weight(g)*v_goal(s,g) score; goal-influence (the means-end
leaf alpha) fades to 0 as the position becomes decisive (alpha_schedule). 'Win as
apex' emerges from alpha->0; there is no discrete terminal goal during play."""
from __future__ import annotations


def alpha_schedule(v_win: float, alpha_max: float, win_ramp: float,
                   ply: int, ply_cap: int, endgame_margin: int) -> float:
    """Goal-influence weight for the means-end leaf. Full (alpha_max) in unclear
    positions, -> 0 as |v_win| -> win_ramp (decisive either way) or near ply_cap."""
    if ply >= ply_cap - endgame_margin:
        return 0.0
    decisiveness = min(1.0, abs(float(v_win)) / max(win_ramp, 1e-6))
    return float(alpha_max) * (1.0 - decisiveness)
