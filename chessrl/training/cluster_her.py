# chessrl/training/cluster_her.py
"""Cluster-goal HER sample generation for the v2 dual-head vector net (Plan 3).

Relabels goals via the frozen-encoder embedding delta -> nearest cluster
(GoalSpace), instead of v1's board predicates. Each ply yields an *active*
sample (searched cluster goal: outcome target for the tanh win head + achievement
target for the sigmoid goal head + policy target) plus HER future-positives and
negatives (goal-head achievement targets only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chessrl.training.her import HERWeights


@dataclass(frozen=True)
class ClusterGoalSample:
    ply: int
    goal_vec: np.ndarray
    cluster: int
    remaining: int
    v_win: float       # game outcome z (side-to-move), used when v_win_mask==1
    v_win_mask: float  # 1.0 on the active sample, else 0.0
    v_goal: float      # achievement label in {0,1}
    v_goal_weight: float


def _delta(embedder, states, i, t):
    e = embedder.embed_boards([states[i], states[t]])
    return (e[1] - e[0]).astype(np.float32)


def _achieved_cluster(embedder, goalspace, states, i, cluster, rem) -> bool:
    T = len(states) - 1
    end = min(i + rem, T)
    for t in range(i + 1, end + 1):
        if goalspace.achieved(_delta(embedder, states, i, t), cluster):
            return True
    return False


def cluster_goal_samples(rec, states, embedder, goalspace, rng,
                         weights: HERWeights | None = None, deadline_max: int = 60):
    if not rec.has_cluster_goals():
        return []
    w = weights or HERWeights()
    out: list[ClusterGoalSample] = []
    T_ = len(rec)
    k_clusters = goalspace.centroids.shape[0]
    for i in range(T_):
        rem = min(deadline_max, T_ - i)
        active_cluster = int(rec.active_cluster[i])
        active_vec = np.asarray(rec.active_vec[i], np.float32)
        # active sample: outcome target (tanh head) + active-goal achievement (goal head)
        ach = active_cluster >= 0 and _achieved_cluster(embedder, goalspace, states, i, active_cluster, rem)
        out.append(ClusterGoalSample(
            ply=i, goal_vec=active_vec, cluster=active_cluster, remaining=rem,
            v_win=float(rec.outcomes[i]), v_win_mask=1.0,
            v_goal=1.0 if ach else 0.0, v_goal_weight=w.search_laundered))
        if rem <= 0:
            continue
        # future positives: clusters actually reached within the window
        reached = set()
        for t in range(i + 1, min(i + rem, T_) + 1):
            reached.add(goalspace.assign(_delta(embedder, states, i, t)))
        reached.discard(-1)
        pos = sorted(reached - {active_cluster})
        if pos and w.future_samples > 0:
            for j in rng.choice(len(pos), size=min(w.future_samples, len(pos)), replace=False):
                c = int(pos[int(j)])
                out.append(ClusterGoalSample(
                    ply=i, goal_vec=goalspace.centroids[c].astype(np.float32), cluster=c,
                    remaining=rem, v_win=0.0, v_win_mask=0.0,
                    v_goal=1.0, v_goal_weight=w.her_positive))
        # negatives: clusters not reached
        neg = [c for c in range(k_clusters) if c not in reached]
        if neg and w.negative_samples > 0:
            for j in rng.choice(len(neg), size=min(w.negative_samples, len(neg)), replace=False):
                c = int(neg[int(j)])
                out.append(ClusterGoalSample(
                    ply=i, goal_vec=goalspace.centroids[c].astype(np.float32), cluster=c,
                    remaining=rem, v_win=0.0, v_win_mask=0.0,
                    v_goal=0.0, v_goal_weight=w.negative))
    return out
