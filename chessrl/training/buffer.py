"""Sliding-window replay buffer over sparse positions.

Holds (planes int8, sparse indices, sparse counts, outcome) tuples; expands
to dense float tensors only at sample time. Reconstructable from the game
records in a run directory (resume support: no buffer serialization needed).
"""
from collections import deque
from pathlib import Path

import numpy as np

from chessrl.chess_env.encoding import to_model_input
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.selfplay.records import GameRecord


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
        files = sorted((Path(run_dir) / "games").glob("*.npz"))
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
