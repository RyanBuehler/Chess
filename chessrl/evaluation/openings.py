"""A suite of 50 short opening book lines (2-5 UCI half-moves) covering the
major systems. Used to seed evaluation games so a deterministic pairing yields a
varied set rather than two repeated games. Every line is a legal sequence from
the starting position and every resulting position is distinct (locked by tests).
"""
import chess

OPENINGS: list[list[str]] = [
    ["e2e4", "e7e5", "g1f3"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["e2e4", "e7e5", "g1f3", "g8f6"],
    ["e2e4", "e7e5", "f1c4"],
    ["e2e4", "e7e5", "b1c3"],
    ["e2e4", "e7e5", "f2f4"],
    ["e2e4", "e7e5", "g1f3", "f8c5"],
    ["e2e4", "e7e5", "g1f3", "d7d6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],       # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],       # Italian
    ["e2e4", "c7c5"],
    ["e2e4", "c7c5", "g1f3"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["e2e4", "c7c5", "g1f3", "b8c6"],
    ["e2e4", "c7c5", "g1f3", "e7e6"],
    ["e2e4", "c7c5", "b1c3"],
    ["e2e4", "c7c5", "c2c3"],
    ["e2e4", "c7c5", "d2d4"],
    ["e2e4", "c7c5", "f2f4"],
    ["e2e4", "e7e6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["e2e4", "c7c6"],
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    ["e2e4", "d7d5"],
    ["e2e4", "d7d5", "e4d5", "g8f6"],
    ["e2e4", "d7d6"],
    ["e2e4", "g8f6"],
    ["e2e4", "g7g6"],
    ["e2e4", "b7b6"],
    ["d2d4", "d7d5"],
    ["d2d4", "d7d5", "c2c4"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    ["d2d4", "d7d5", "c2c4", "d5c4"],
    ["d2d4", "d7d5", "g1f3"],
    ["d2d4", "d7d5", "c1f4"],
    ["d2d4", "d7d5", "e2e3"],
    ["d2d4", "g8f6"],
    ["d2d4", "g8f6", "c2c4"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
    ["d2d4", "g8f6", "c2c4", "c7c5"],
    ["d2d4", "g8f6", "c2c4", "b7b6"],               # Queen's Indian root
    ["d2d4", "g8f6", "c2c4", "d7d6"],               # Old Indian root
    ["d2d4", "g8f6", "g1f3"],
    ["d2d4", "g8f6", "c1g5"],
    ["d2d4", "f7f5"],
    ["g1f3", "d7d5"],
    ["c2c4", "e7e5"],
    ["c2c4", "g8f6"],
]


def opening_board(idx: int) -> chess.Board:
    """Fresh board with opening line (idx % len(OPENINGS)) applied."""
    line = OPENINGS[idx % len(OPENINGS)]
    board = chess.Board()
    for uci in line:
        board.push(chess.Move.from_uci(uci))
    return board
