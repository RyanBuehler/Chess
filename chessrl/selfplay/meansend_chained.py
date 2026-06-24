"""v3-zenith: chained means-end self-play. ENTIRELY SEGREGATED from v2's
selfplay/concurrent.py — v2 (goal_mode 'emergent') never imports this module.

Each side always holds an active cluster sub-goal, re-selected on achieve-or-expire
via a greedy curriculum_weight(g)*v_goal(s,g) score; goal-influence (the means-end
leaf alpha) fades to 0 as the position becomes decisive (alpha_schedule). 'Win as
apex' emerges from alpha->0; there is no discrete terminal goal during play."""
from __future__ import annotations

import numpy as np
import chess

from chessrl.chess_env.encoding import encode_board, to_model_input


def alpha_schedule(v_win: float, alpha_max: float, win_ramp: float,
                   ply: int, ply_cap: int, endgame_margin: int) -> float:
    """Goal-influence weight for the means-end leaf. Full (alpha_max) in unclear
    positions, -> 0 as |v_win| -> win_ramp (decisive either way) or near ply_cap."""
    if ply >= ply_cap - endgame_margin:
        return 0.0
    decisiveness = min(1.0, abs(float(v_win)) / max(win_ramp, 1e-6))
    return float(alpha_max) * (1.0 - decisiveness)


def select_next_goal(board, goalspace, curriculum, evaluator, goal_cfg, rng):
    """Greedy state-dependent next sub-goal: softmax over
    curriculum_weight(g) * v_goal(board, centroid_g). Epsilon-explore injects a
    uniform-random cluster. Requires goalspace.ready (caller guards). Returns
    (cluster_id >= 0, centroid_vec)."""
    K = goalspace.n_clusters
    cents = np.asarray(goalspace.centroids, np.float32)            # (K, d)
    if rng.random() < getattr(goal_cfg, "epsilon", 0.0):
        c = int(rng.integers(K))
        return c, cents[c].astype(np.float32)
    planes = to_model_input(encode_board(board))
    planes_batch = np.repeat(planes[None, ...], K, axis=0)
    deadlines = np.full(K, goal_cfg.goal_window, np.float32)
    _, _v_win, v_goal = evaluator.evaluate_planes(planes_batch, cents, deadlines)
    w = np.array([curriculum.weight(c) for c in range(K)], np.float64)
    score = w * np.asarray(v_goal, np.float64)
    temp = max(float(goal_cfg.goal_select_temp), 1e-6)
    logits = score / temp
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    c = int(rng.choice(K, p=p))
    return c, cents[c].astype(np.float32)
