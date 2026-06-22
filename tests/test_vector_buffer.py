# tests/test_vector_buffer.py
import numpy as np
import chess
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import NUM_ACTIONS, move_to_index
from chessrl.training.vector_buffer import VectorGoalReplayBuffer
from tests.test_cluster_her import FakeEmbedder, FakeGoalSpace


def _game(n=4):
    b = RecordBuilder(); board = chess.Board()
    for _ in range(n):
        move = list(board.legal_moves)[0]
        idx = move_to_index(move, board.turn == chess.BLACK)
        b.add(board, [idx, idx + 1], [3, 1], played_index=idx, protagonist=board.turn,
              cluster_active=1, cluster_assigned=1,
              active_vec=np.array([1, 0, 0, 0], np.float32), explore=False)
        board.push(move)
    return b.finalize(z_white=1)


def test_buffer_sample_shapes():
    buf = VectorGoalReplayBuffer(1000, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(_game(), rng=np.random.default_rng(0))
    assert len(buf) > 0
    x, gv, dl, p, pm, vw, vwm, vg, vgw = buf.sample(8, np.random.default_rng(1))
    assert x.shape == (8, 21, 8, 8)
    assert gv.shape == (8, 4)
    assert dl.shape == (8,) and vw.shape == (8,) and vg.shape == (8,)
    assert p.shape == (8, NUM_ACTIONS)
    assert set(np.unique(pm)).issubset({0.0, 1.0})
    assert set(np.unique(vwm)).issubset({0.0, 1.0})


def test_policy_only_on_active_samples():
    buf = VectorGoalReplayBuffer(1000, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(_game(), rng=np.random.default_rng(0))
    x, gv, dl, p, pm, vw, vwm, vg, vgw = buf.sample(64, np.random.default_rng(2))
    # policy mask is set exactly where the win mask is (the active sample)
    assert np.array_equal(pm, vwm)


def test_skips_vanilla():
    b = RecordBuilder(); b.add(chess.Board(), [0], [1], 0)
    buf = VectorGoalReplayBuffer(10, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(b.finalize(z_white=0))
    assert len(buf) == 0
