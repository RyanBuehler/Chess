"""Replay buffer for the v2 dual-head vector goal net (Plan 3).

Stores flat per-sample descriptors generated from cluster-goal GameRecords via
train-time cluster HER. Yields the vector net's inputs (board planes + goal
centroid + deadline) and dual targets (tanh win outcome + sigmoid goal
achievement) with per-sample masks/weights."""
from collections import deque
from pathlib import Path

import numpy as np

from chessrl.chess_env.encoding import to_model_input
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.selfplay.records import GameRecord
from chessrl.training.her import HERWeights, reconstruct_states
from chessrl.training.cluster_her import cluster_goal_samples


class VectorGoalReplayBuffer:
    def __init__(self, capacity, embedder, goalspace, weights: HERWeights | None = None,
                 deadline_max: int = 60, lookahead_cap: int | None = None):
        self._data = deque(maxlen=capacity)
        self.embedder = embedder
        self.goalspace = goalspace
        self.weights = weights or HERWeights()
        self.deadline_max = deadline_max
        self.lookahead_cap = lookahead_cap   # v3: cap HER achievement look-ahead (None = v2)

    def __len__(self):
        return len(self._data)

    def add_game(self, rec: GameRecord, rng=None) -> None:
        if not rec.has_cluster_goals():
            return
        if rng is None:
            seed = (len(rec) * 1000003 + int(rec.played[0] if len(rec.played) else 0)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
        states = reconstruct_states(rec)
        samples = cluster_goal_samples(rec, states, self.embedder, self.goalspace, rng,
                                       self.weights, self.deadline_max, self.lookahead_cap)
        for s in samples:
            board_planes = rec.planes[s.ply]
            p_idxs = p_cnts = None
            if s.v_win_mask == 1.0:   # active sample carries the policy target
                a, b = rec.policy_offsets[s.ply], rec.policy_offsets[s.ply + 1]
                p_idxs = rec.policy_indices[a:b].astype(np.int64)
                p_cnts = rec.policy_counts[a:b].astype(np.float64)
            self._data.append((board_planes, s.goal_vec, s.remaining, s.v_win,
                               s.v_win_mask, s.v_goal, s.v_goal_weight, p_idxs, p_cnts))

    def sample(self, batch_size, rng):
        if not self._data:
            raise ValueError("cannot sample from an empty vector goal buffer")
        d = self._data[0][1].shape[0]
        picks = rng.integers(0, len(self._data), size=batch_size)
        x = np.empty((batch_size, 21, 8, 8), dtype=np.float32)
        gv = np.empty((batch_size, d), dtype=np.float32)
        deadline = np.empty(batch_size, dtype=np.float32)
        p = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
        p_mask = np.zeros(batch_size, dtype=np.float32)
        v_win = np.empty(batch_size, dtype=np.float32)
        v_win_mask = np.empty(batch_size, dtype=np.float32)
        v_goal = np.empty(batch_size, dtype=np.float32)
        v_goal_w = np.empty(batch_size, dtype=np.float32)
        for row, i in enumerate(picks):
            bp, goal_vec, rem, vw, vwm, vg, vgw, p_idxs, p_cnts = self._data[i]
            x[row] = to_model_input(bp)
            gv[row] = goal_vec
            deadline[row] = float(rem)
            v_win[row] = vw
            v_win_mask[row] = vwm
            v_goal[row] = vg
            v_goal_w[row] = vgw
            if p_idxs is not None and len(p_idxs) and p_cnts.sum() > 0:
                p[row, p_idxs] = (p_cnts / p_cnts.sum()).astype(np.float32)
                p_mask[row] = 1.0
        return x, gv, deadline, p, p_mask, v_win, v_win_mask, v_goal, v_goal_w

    @classmethod
    def from_run_dir(cls, run_dir, capacity, embedder, goalspace,
                     weights: HERWeights | None = None, deadline_max: int = 60,
                     lookahead_cap: int | None = None):
        buf = cls(capacity, embedder, goalspace, weights=weights, deadline_max=deadline_max,
                  lookahead_cap=lookahead_cap)
        files = sorted((Path(run_dir) / "games").glob("*.npz"),
                       key=lambda p: (p.stat().st_mtime, p.name))
        selected, total = [], 0
        for f in reversed(files):
            rec = GameRecord.load(f)
            if not rec.has_cluster_goals():
                continue
            selected.append(rec)
            total += len(rec)
            if total >= capacity:
                break
        for rec in reversed(selected):
            buf.add_game(rec)
        return buf
