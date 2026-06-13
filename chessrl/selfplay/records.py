"""Sparse per-game training records: the on-disk source of truth.

Policy targets are stored sparse (legal move indices + visit counts) and
ragged rows are flattened with an offsets array (offsets[t]:offsets[t+1]
slices position t's entries).

Goal fields (spec sec 7, 11; plan Task 3.2)
-------------------------------------------
For goal-conditioned self-play, each position additionally stores enough to let
the verifier recompute achieved-by-deadline deltas at HER train time:

* ``protagonist``    (T,) int8  -- 1 == White is protagonist, 0 == Black; the
                                    side to move at this ply, whose goal drove
                                    the search. (Always equals the side to move,
                                    by the protagonist-frame contract.)
* ``assigned_kind``  (T,) goal-kind code of the side-to-move's *game-assigned*
                                    goal (the one drawn at game start).
* ``assigned_blob``  (T,) the assigned goal serialized to a string (kind,
                                    params, deadline) — exact, verifier-ready.
* ``active_kind``    (T,) goal-kind code of the *active* goal actually searched
                                    at this ply (== assigned until it resolves,
                                    then WIN for the rest of the game).
* ``active_blob``    (T,) the active goal serialized to a string.

The blobs are the source of truth (kind + params + deadline). The ``*_kind``
int columns are a fast filter (e.g. "is this ply under g=win?") without parsing.

These fields are present only for goal games. ``has_goals()`` reports whether a
record carries them; vanilla records omit them entirely and the legacy format is
byte-for-byte unchanged.
"""
from dataclasses import dataclass, field

import chess
import numpy as np

from chessrl.chess_env.encoding import encode_board
from chessrl.goals import templates as T
from chessrl.goals.templates import GoalTemplate

_FIELDS = ("planes", "policy_indices", "policy_counts", "policy_offsets", "outcomes", "played")
_GOAL_FIELDS = ("protagonist", "assigned_kind", "assigned_blob", "active_kind", "active_blob")

# Integer codes for goal kinds (stable, for the fast-filter columns).
_KIND_CODES = {k: i for i, k in enumerate(T.KINDS)}
WIN_KIND_CODE = _KIND_CODES[T.WIN]


def serialize_goal(goal: GoalTemplate) -> str:
    """Exact, reversible serialization: 'kind|deadline|name=val,name=val'."""
    params = ",".join(f"{n}={v}" for n, v in goal.params)
    return f"{goal.kind}|{goal.deadline}|{params}"


def deserialize_goal(blob: str) -> GoalTemplate:
    kind, deadline, params_s = blob.split("|", 2)
    params = ()
    if params_s:
        pairs = []
        for tok in params_s.split(","):
            n, v = tok.split("=", 1)
            pairs.append((n, int(v)))
        params = tuple(sorted(pairs))
    return GoalTemplate(kind=kind, params=params, deadline=int(deadline))


@dataclass
class GameRecord:
    planes: np.ndarray          # (T, 21, 8, 8) int8
    policy_indices: np.ndarray  # flat int32
    policy_counts: np.ndarray   # flat int32
    policy_offsets: np.ndarray  # (T+1,) int64
    outcomes: np.ndarray        # (T,) int8, from side-to-move perspective
    played: np.ndarray          # (T,) int32 action indices
    # --- optional goal columns (None for vanilla records) ---------------
    protagonist: np.ndarray | None = None    # (T,) int8 (1=White,0=Black)
    assigned_kind: np.ndarray | None = None  # (T,) int8 goal-kind code
    assigned_blob: np.ndarray | None = None  # (T,) <U str blobs
    active_kind: np.ndarray | None = None    # (T,) int8 goal-kind code
    active_blob: np.ndarray | None = None    # (T,) <U str blobs

    def __len__(self) -> int:
        return len(self.planes)

    def has_goals(self) -> bool:
        return self.active_blob is not None

    def positions(self):
        for t in range(len(self)):
            a, b = self.policy_offsets[t], self.policy_offsets[t + 1]
            yield self.planes[t], self.policy_indices[a:b], self.policy_counts[a:b], int(self.outcomes[t])

    def win_ply_fraction(self) -> float:
        """Fraction of plies played under the active win-goal (control variable,
        spec sec 7/16). 0.0 for vanilla records (no active goal)."""
        if not self.has_goals() or len(self) == 0:
            return 0.0
        return float(np.mean(self.active_kind == WIN_KIND_CODE))

    def save(self, path) -> None:
        out = {f: getattr(self, f) for f in _FIELDS}
        if self.has_goals():
            for f in _GOAL_FIELDS:
                out[f] = getattr(self, f)
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path) -> "GameRecord":
        with np.load(path, allow_pickle=False) as z:
            kw = {f: z[f] for f in _FIELDS}
            if "active_blob" in z.files:
                for f in _GOAL_FIELDS:
                    kw[f] = z[f]
            return cls(**kw)


class RecordBuilder:
    def __init__(self):
        self._planes: list[np.ndarray] = []
        self._idx: list[int] = []
        self._cnt: list[int] = []
        self._off: list[int] = [0]
        self._stm: list[bool] = []
        self._played: list[int] = []
        # goal columns (parallel; empty for vanilla)
        self._proto: list[int] = []
        self._assigned: list[GoalTemplate] = []
        self._active: list[GoalTemplate] = []
        self._has_goals = False

    def add(
        self,
        board: chess.Board,
        move_indices,
        visit_counts,
        played_index: int,
        *,
        protagonist: bool | None = None,
        assigned_goal: GoalTemplate | None = None,
        active_goal: GoalTemplate | None = None,
    ) -> None:
        self._planes.append(encode_board(board))
        self._idx.extend(int(i) for i in move_indices)
        self._cnt.extend(int(c) for c in visit_counts)
        self._off.append(len(self._idx))
        self._stm.append(board.turn)
        self._played.append(int(played_index))
        if active_goal is not None:
            self._has_goals = True
            self._proto.append(1 if protagonist == chess.WHITE else 0)
            self._assigned.append(assigned_goal)
            self._active.append(active_goal)

    def finalize(self, z_white: int) -> GameRecord:
        outcomes = np.array(
            [z_white if stm == chess.WHITE else -z_white for stm in self._stm], dtype=np.int8
        )
        rec = GameRecord(
            planes=np.array(self._planes, dtype=np.int8),
            policy_indices=np.array(self._idx, dtype=np.int32),
            policy_counts=np.array(self._cnt, dtype=np.int32),
            policy_offsets=np.array(self._off, dtype=np.int64),
            outcomes=outcomes,
            played=np.array(self._played, dtype=np.int32),
        )
        if self._has_goals:
            rec.protagonist = np.array(self._proto, dtype=np.int8)
            rec.assigned_kind = np.array(
                [_KIND_CODES[g.kind] for g in self._assigned], dtype=np.int8
            )
            rec.assigned_blob = np.array([serialize_goal(g) for g in self._assigned])
            rec.active_kind = np.array(
                [_KIND_CODES[g.kind] for g in self._active], dtype=np.int8
            )
            rec.active_blob = np.array([serialize_goal(g) for g in self._active])
        return rec
