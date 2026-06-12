"""Board -> plane encoding, always from the side to move's perspective.

The board is mirrored (rank-flipped) when Black is to move, so the side to
move's pieces always advance "up" the plane array. Stored as int8; use
to_model_input() to get normalized float32 for the network.
"""
import chess
import numpy as np

NUM_PLANES = 21
PLANE_FIFTY = 18


def encode_board(board: chess.Board) -> np.ndarray:
    flip = board.turn == chess.BLACK
    pl = np.zeros((NUM_PLANES, 8, 8), dtype=np.int8)
    for color_idx, color in enumerate((board.turn, not board.turn)):
        for piece_type in range(1, 7):  # PAWN..KING
            for sq in board.pieces(piece_type, color):
                if flip:
                    sq = chess.square_mirror(sq)
                pl[color_idx * 6 + piece_type - 1, chess.square_rank(sq), chess.square_file(sq)] = 1
    pl[12] = int(board.turn == chess.WHITE)
    us, them = board.turn, not board.turn
    pl[13] = int(board.has_kingside_castling_rights(us))
    pl[14] = int(board.has_queenside_castling_rights(us))
    pl[15] = int(board.has_kingside_castling_rights(them))
    pl[16] = int(board.has_queenside_castling_rights(them))
    if board.ep_square is not None:
        ep = chess.square_mirror(board.ep_square) if flip else board.ep_square
        pl[17, chess.square_rank(ep), chess.square_file(ep)] = 1
    pl[PLANE_FIFTY] = min(board.halfmove_clock, 100)
    pl[19] = int(board.is_repetition(2))
    pl[20] = int(board.is_repetition(3))
    return pl


def to_model_input(planes: np.ndarray) -> np.ndarray:
    """int8 planes -> normalized float32 (works on single positions or batches)."""
    x = planes.astype(np.float32)
    x[..., PLANE_FIFTY, :, :] /= 100.0
    return x
