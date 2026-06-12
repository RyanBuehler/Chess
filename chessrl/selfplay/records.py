"""Sparse per-game training records: the on-disk source of truth.

Policy targets are stored sparse (legal move indices + visit counts) and
ragged rows are flattened with an offsets array (offsets[t]:offsets[t+1]
slices position t's entries).
"""
from dataclasses import dataclass

import chess
import numpy as np

from chessrl.chess_env.encoding import encode_board

_FIELDS = ("planes", "policy_indices", "policy_counts", "policy_offsets", "outcomes", "played")


@dataclass
class GameRecord:
    planes: np.ndarray          # (T, 21, 8, 8) int8
    policy_indices: np.ndarray  # flat int32
    policy_counts: np.ndarray   # flat int32
    policy_offsets: np.ndarray  # (T+1,) int64
    outcomes: np.ndarray        # (T,) int8, from side-to-move perspective
    played: np.ndarray          # (T,) int32 action indices

    def __len__(self) -> int:
        return len(self.planes)

    def positions(self):
        for t in range(len(self)):
            a, b = self.policy_offsets[t], self.policy_offsets[t + 1]
            yield self.planes[t], self.policy_indices[a:b], self.policy_counts[a:b], int(self.outcomes[t])

    def save(self, path) -> None:
        np.savez_compressed(path, **{f: getattr(self, f) for f in _FIELDS})

    @classmethod
    def load(cls, path) -> "GameRecord":
        with np.load(path) as z:
            return cls(**{f: z[f] for f in _FIELDS})


class RecordBuilder:
    def __init__(self):
        self._planes: list[np.ndarray] = []
        self._idx: list[int] = []
        self._cnt: list[int] = []
        self._off: list[int] = [0]
        self._stm: list[bool] = []
        self._played: list[int] = []

    def add(self, board: chess.Board, move_indices, visit_counts, played_index: int) -> None:
        self._planes.append(encode_board(board))
        self._idx.extend(int(i) for i in move_indices)
        self._cnt.extend(int(c) for c in visit_counts)
        self._off.append(len(self._idx))
        self._stm.append(board.turn)
        self._played.append(int(played_index))

    def finalize(self, z_white: int) -> GameRecord:
        outcomes = np.array(
            [z_white if stm == chess.WHITE else -z_white for stm in self._stm], dtype=np.int8
        )
        return GameRecord(
            planes=np.array(self._planes, dtype=np.int8),
            policy_indices=np.array(self._idx, dtype=np.int32),
            policy_counts=np.array(self._cnt, dtype=np.int32),
            policy_offsets=np.array(self._off, dtype=np.int64),
            outcomes=outcomes,
            played=np.array(self._played, dtype=np.int32),
        )
