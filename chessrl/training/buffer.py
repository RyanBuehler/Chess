"""Sliding-window replay buffer over sparse positions.

Holds (planes int8, sparse indices, sparse counts, outcome) tuples; expands
to dense float tensors only at sample time. Reconstructable from the game
records in a run directory (resume support: no buffer serialization needed).

Two buffers share this module:

* ``ReplayBuffer``       — the vanilla (negamax / tanh / MSE) buffer. UNCHANGED:
                           21-plane inputs, ``(x, p, v)`` samples.
* ``GoalReplayBuffer``   — the goal-conditioned buffer. Stores whole goal
                           ``GameRecord``s (the source of truth) and generates
                           HER value samples + active-goal policy targets AT
                           SAMPLE TIME via the verifier (spec sec 11; relabeled
                           samples are never persisted). Yields the wider
                           ``21 + GOAL_PLANES`` inputs, a deadline scalar, sparse
                           active-goal policy targets, BCE value targets, and
                           per-sample value weights.
"""
from collections import deque
from pathlib import Path

import numpy as np

import chess

from chessrl.chess_env.encoding import encode_board, to_model_input
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.goals.encoding import GOAL_PLANES as _GOAL_PLANES
from chessrl.goals.encoding import encode_goal
from chessrl.selfplay.records import GameRecord, deserialize_goal
from chessrl.training.her import HERWeights, goal_value_samples, reconstruct_states


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._data = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add_game(self, rec: GameRecord) -> None:
        for planes, idxs, cnts, outcome in rec.positions():
            self._data.append((planes, idxs, cnts, outcome))

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._data:
            raise ValueError("cannot sample from an empty replay buffer")
        picks = rng.integers(0, len(self._data), size=batch_size)
        xs = np.stack([to_model_input(self._data[i][0]) for i in picks])
        ps = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
        vs = np.empty(batch_size, dtype=np.float32)
        for row, i in enumerate(picks):
            _, idxs, cnts, outcome = self._data[i]
            ps[row, idxs] = cnts / cnts.sum()
            vs[row] = outcome
        return xs, ps, vs

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, capacity: int) -> "ReplayBuffer":
        buf = cls(capacity)
        # Sort by (mtime, name) so multi-worker write chronology is respected;
        # lexical filename order is unreliable across concurrent workers.
        files = sorted(
            (Path(run_dir) / "games").glob("*.npz"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        selected, total = [], 0
        for f in reversed(files):           # newest first
            rec = GameRecord.load(f)
            selected.append(rec)
            total += len(rec)
            if total >= capacity:
                break
        for rec in reversed(selected):      # re-add in chronological order
            buf.add_game(rec)
        return buf


# A flat goal training sample: pre-encoded board planes (int8, 21x8x8) + goal +
# remaining + BCE value target + value weight + (optional) sparse active-goal
# policy target. Stored as a tuple so the deque stays compact:
#   (board_planes, goal, remaining, v_target, v_weight, p_idxs, p_cnts)
# p_idxs/p_cnts are None for HER-only (future/negative) rows that carry no
# policy supervision (spec sec 11 — policy is trained only on the active goal).


class GoalReplayBuffer:
    """Goal-conditioned replay buffer with train-time HER (spec sec 11).

    Stores flat per-sample descriptors generated from goal ``GameRecord``s. Each
    record yields: one **active-goal** sample per ply (search-laundered value
    target + the stored visit counts as the policy target), plus HER "future"
    positives and negatives (value-only, no policy target). The buffer is capped
    by ``capacity`` *samples* (like the vanilla buffer's per-position cap).
    """

    def __init__(self, capacity: int, weights: HERWeights | None = None, deadline_max: int = 60):
        self._data = deque(maxlen=capacity)
        self.weights = weights or HERWeights()
        self.deadline_max = deadline_max

    def __len__(self) -> int:
        return len(self._data)

    def add_game(self, rec: GameRecord, rng: np.random.Generator | None = None) -> None:
        """Generate this game's HER + active-goal samples and append them.

        A deterministic per-game rng (seeded from the game length + first played
        index when none is given) keeps resume reconstruction reproducible.
        """
        if not rec.has_goals():
            return
        if rng is None:
            seed = (len(rec) * 1000003 + int(rec.played[0] if len(rec.played) else 0)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)

        states = reconstruct_states(rec)
        # Per-ply active-goal policy targets (sparse), keyed by ply.
        samples = goal_value_samples(rec, rng, self.weights, self.deadline_max)
        for s in samples:
            board = states[s.ply]
            board_planes = encode_board(board)  # int8 (21,8,8)
            # Attach the policy target only to the search-laundered (active) row
            # for this ply: that is the goal that was actually searched.
            p_idxs = p_cnts = None
            active_blob = str(rec.active_blob[s.ply])
            if s.goal == deserialize_goal(active_blob):
                a, b = rec.policy_offsets[s.ply], rec.policy_offsets[s.ply + 1]
                p_idxs = rec.policy_indices[a:b].astype(np.int64)
                p_cnts = rec.policy_counts[a:b].astype(np.float64)
            self._data.append(
                (board_planes, s.goal, s.remaining, s.target, s.weight, p_idxs, p_cnts)
            )

    def sample(self, batch_size: int, rng: np.random.Generator):
        """Return a goal training batch:

        ``(x, deadline, p, p_mask, v, v_weight)`` where
          * ``x``        (B, 21+GOAL_PLANES, 8, 8) float32 — board + goal planes
          * ``deadline`` (B,) float32 — moves-remaining scalar (raw; the trainer
                          scales it to match the evaluators' DEADLINE_SCALE)
          * ``p``        (B, NUM_ACTIONS) float32 — active-goal visit-count target
          * ``p_mask``   (B,) float32 — 1.0 where a policy target exists, else 0
          * ``v``        (B,) float32 — BCE achievement target in [0,1]
          * ``v_weight`` (B,) float32 — per-sample BCE weight
        """
        if not self._data:
            raise ValueError("cannot sample from an empty goal replay buffer")
        picks = rng.integers(0, len(self._data), size=batch_size)
        xs = np.empty((batch_size, 21 + _GOAL_PLANES, 8, 8), dtype=np.float32)
        deadline = np.empty(batch_size, dtype=np.float32)
        ps = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
        p_mask = np.zeros(batch_size, dtype=np.float32)
        vs = np.empty(batch_size, dtype=np.float32)
        vw = np.empty(batch_size, dtype=np.float32)
        for row, i in enumerate(picks):
            board_planes, goal, remaining, v_target, v_weight, p_idxs, p_cnts = self._data[i]
            protagonist = _planes_protagonist(board_planes)
            board_input = to_model_input(board_planes)
            goal_planes, _ = encode_goal(goal, remaining, protagonist)
            xs[row] = np.concatenate([board_input, goal_planes.astype(np.float32)], axis=0)
            deadline[row] = float(remaining)
            vs[row] = v_target
            vw[row] = v_weight
            if p_idxs is not None and len(p_idxs) and p_cnts.sum() > 0:
                ps[row, p_idxs] = (p_cnts / p_cnts.sum()).astype(np.float32)
                p_mask[row] = 1.0
        return xs, deadline, ps, p_mask, vs, vw

    @classmethod
    def from_run_dir(
        cls, run_dir: str | Path, capacity: int,
        weights: HERWeights | None = None, deadline_max: int = 60,
    ) -> "GoalReplayBuffer":
        buf = cls(capacity, weights=weights, deadline_max=deadline_max)
        files = sorted(
            (Path(run_dir) / "games").glob("*.npz"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        selected, total = [], 0
        for f in reversed(files):           # newest first
            rec = GameRecord.load(f)
            if not rec.has_goals():
                continue
            selected.append(rec)
            total += len(rec)
            if total >= capacity:
                break
        for rec in reversed(selected):      # re-add in chronological order
            buf.add_game(rec)
        return buf


# Plane index 12 of the board encoding is "side to move is White" (broadcast).
# The protagonist of a goal sample is exactly the side to move at that ply.
def _planes_protagonist(board_planes: np.ndarray) -> bool:
    return chess.WHITE if board_planes[12].any() else chess.BLACK
