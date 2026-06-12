import chess
import numpy as np

from chessrl.selfplay.records import GameRecord, RecordBuilder


def _tiny_record() -> GameRecord:
    b = RecordBuilder()
    board = chess.Board()
    b.add(board, [100, 200], [3, 1], 100)       # white to move
    board.push(chess.Move.from_uci("e2e4"))
    b.add(board, [300], [5], 300)               # black to move
    return b.finalize(z_white=1)                # white won


def test_builder_outcome_perspective():
    rec = _tiny_record()
    assert len(rec) == 2
    assert rec.outcomes[0] == 1     # white to move, white won -> +1
    assert rec.outcomes[1] == -1    # black to move, white won -> -1


def test_positions_iteration():
    rec = _tiny_record()
    rows = list(rec.positions())
    planes, idxs, cnts, outcome = rows[0]
    assert planes.shape == (21, 8, 8)
    assert list(idxs) == [100, 200]
    assert list(cnts) == [3, 1]
    assert outcome == 1
    _, idxs1, cnts1, outcome1 = rows[1]
    assert list(idxs1) == [300]
    assert outcome1 == -1


def test_save_load_round_trip(tmp_path):
    rec = _tiny_record()
    path = tmp_path / "g.npz"
    rec.save(path)
    rec2 = GameRecord.load(path)
    assert len(rec2) == len(rec)
    np.testing.assert_array_equal(rec2.planes, rec.planes)
    np.testing.assert_array_equal(rec2.policy_indices, rec.policy_indices)
    np.testing.assert_array_equal(rec2.policy_counts, rec.policy_counts)
    np.testing.assert_array_equal(rec2.policy_offsets, rec.policy_offsets)
    np.testing.assert_array_equal(rec2.outcomes, rec.outcomes)
    np.testing.assert_array_equal(rec2.played, rec.played)
