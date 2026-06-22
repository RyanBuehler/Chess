import numpy as np
import chess
from chessrl.selfplay.records import GameRecord, RecordBuilder


def _tiny_game(d=4):
    b = RecordBuilder()
    board = chess.Board()
    for ply in range(3):
        b.add(board, [0, 1], [3, 1], played_index=0,
              protagonist=board.turn,
              cluster_active=(ply % 2), cluster_assigned=1,
              active_vec=np.full(d, float(ply), np.float32), explore=(ply == 0))
        board.push(list(board.legal_moves)[0])
    return b.finalize(z_white=1)


def test_cluster_columns_present_and_shaped():
    rec = _tiny_game(d=4)
    assert rec.has_cluster_goals()
    assert rec.active_cluster.tolist() == [0, 1, 0]
    assert rec.assigned_cluster.tolist() == [1, 1, 1]
    assert rec.active_vec.shape == (3, 4)
    assert rec.explore.tolist() == [1, 0, 0]


def test_save_load_roundtrip(tmp_path):
    rec = _tiny_game(d=4)
    p = tmp_path / "g.npz"
    rec.save(p)
    rl = GameRecord.load(p)
    assert rl.has_cluster_goals()
    assert np.array_equal(rl.active_cluster, rec.active_cluster)
    assert np.allclose(rl.active_vec, rec.active_vec)
    assert np.array_equal(rl.explore, rec.explore)


def test_vanilla_record_unaffected():
    b = RecordBuilder()
    board = chess.Board()
    b.add(board, [0], [1], played_index=0)
    rec = b.finalize(z_white=0)
    assert rec.has_cluster_goals() is False
    assert rec.has_goals() is False
    assert rec.active_vec is None
