from chessrl.supervised.pgn_import import record_from_pgn

FOOLS_MATE = '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'
SCHOLARS_MATE = '[Result "1-0"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n'


def test_fools_mate_record():
    rec = record_from_pgn(FOOLS_MATE)
    assert len(rec) == 4
    # black won: white-to-move positions get -1, black-to-move get +1
    assert list(rec.outcomes) == [-1, 1, -1, 1]
    # one-hot policy targets
    assert all(rec.policy_offsets[t + 1] - rec.policy_offsets[t] == 1 for t in range(4))
    assert all(c == 1 for c in rec.policy_counts)


def test_scholars_mate_record():
    rec = record_from_pgn(SCHOLARS_MATE)
    assert len(rec) == 7
    assert list(rec.outcomes) == [1, -1, 1, -1, 1, -1, 1]
