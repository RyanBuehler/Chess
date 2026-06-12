import numpy as np

from chessrl.supervised.pgn_import import record_from_pgn
from chessrl.training.buffer import ReplayBuffer
from chessrl.chess_env.moves import NUM_ACTIONS

FOOLS_MATE = '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'


def test_add_and_len():
    buf = ReplayBuffer(capacity=100)
    buf.add_game(record_from_pgn(FOOLS_MATE))
    assert len(buf) == 4


def test_capacity_evicts_oldest():
    buf = ReplayBuffer(capacity=3)
    buf.add_game(record_from_pgn(FOOLS_MATE))
    assert len(buf) == 3


def test_sample_shapes_and_targets():
    buf = ReplayBuffer(capacity=100)
    buf.add_game(record_from_pgn(FOOLS_MATE))
    rng = np.random.default_rng(0)
    x, p, v = buf.sample(8, rng)
    assert x.shape == (8, 21, 8, 8) and x.dtype == np.float32
    assert p.shape == (8, NUM_ACTIONS) and p.dtype == np.float32
    assert v.shape == (8,) and v.dtype == np.float32
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(v)).issubset({-1.0, 1.0})


def test_reconstruct_from_run_dir_orders_by_mtime(tmp_path):
    import os

    games = tmp_path / "games"
    games.mkdir()
    rec = record_from_pgn(FOOLS_MATE)  # 4 positions each
    # Names deliberately NOT in chronological order; mtime is the real order.
    rec.save(games / "game_w01_0000000.npz")  # oldest (set below)
    rec.save(games / "game_w00_0000000.npz")  # newest (set below)
    base = 1_000_000_000
    os.utime(games / "game_w01_0000000.npz", (base, base))           # older
    os.utime(games / "game_w00_0000000.npz", (base + 10, base + 10)) # newer
    buf = ReplayBuffer.from_run_dir(tmp_path, capacity=6)
    assert len(buf) == 6  # newest games kept, capped at capacity
