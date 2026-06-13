"""Rule-level state-feature extraction from a chess.Board.

The feature basis is deliberately value-agnostic (spec sec 4): per-(piece-type,
color) counts, side-to-move-in-check, castling rights, and a result flag.
Distinguishing a knight from a queen is *reading the rules*, not valuing them.
Spatial goals read occupancy from the board directly; this captures the
non-spatial scalar basis used for count/check/castle/result deltas.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

# All rule-level piece types (PAWN..KING) and both colors.
PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
COLORS = (chess.WHITE, chess.BLACK)


@dataclass(frozen=True)
class BoardFeatures:
    """A value-agnostic snapshot of rule-level board state.

    counts:   {(piece_type, color): int} for every (piece_type, color) pair.
    in_check: True iff the side to move is in check.
    castling: (white_kingside, white_queenside, black_kingside, black_queenside).
    result:   "1-0" | "0-1" | "1/2-1/2" if the game is over, else None.
    """

    counts: dict
    in_check: bool
    castling: tuple
    result: str | None


def board_features(board: chess.Board) -> BoardFeatures:
    counts = {}
    for color in COLORS:
        for piece_type in PIECE_TYPES:
            counts[(piece_type, color)] = len(board.pieces(piece_type, color))

    castling = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )

    result = board.result(claim_draw=False) if board.is_game_over() else None

    return BoardFeatures(
        counts=counts,
        in_check=board.is_check(),
        castling=castling,
        result=result,
    )
