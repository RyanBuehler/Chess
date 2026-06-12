# tests/test_openings.py
import chess

from chessrl.evaluation.openings import OPENINGS, opening_board


def test_there_are_fifty_openings():
    assert len(OPENINGS) == 50


def test_every_opening_is_a_legal_sequence():
    for i, line in enumerate(OPENINGS):
        # 2-4 full moves; two canonical lines (Ruy Lopez, Italian) run to a 5th
        # half-move to be distinctive, so the bound is 5 half-moves.
        assert 2 <= len(line) <= 5, f"opening {i} wrong length: {line}"
        board = chess.Board()
        for uci in line:
            mv = chess.Move.from_uci(uci)
            assert mv in board.legal_moves, f"illegal move {uci} in opening {i}: {line}"
            board.push(mv)


def test_openings_are_distinct_positions():
    fens = set()
    for line in OPENINGS:
        b = chess.Board()
        for uci in line:
            b.push(chess.Move.from_uci(uci))
        fens.add(b.board_fen() + (" w" if b.turn else " b"))
    assert len(fens) == 50, "duplicate opening positions reduce effective sample size"


def test_opening_board_wraps_modulo():
    b0 = opening_board(0)
    b_wrap = opening_board(len(OPENINGS))
    assert b0.fen() == b_wrap.fen()
    # A fresh, independent board each call (no shared mutable state).
    b0.push(chess.Move.from_uci("a2a3"))
    assert opening_board(0).fen() != b0.fen()
