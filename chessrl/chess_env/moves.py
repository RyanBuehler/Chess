"""AlphaZero 8x8x73 move encoding, always in the side-to-move frame.

index = from_square * 73 + move_type, where move_type is:
  0..55  queen-style: direction (N,NE,E,SE,S,SW,W,NW) * 7 + (distance - 1)
  56..63 knight moves
  64..72 underpromotions: (file_delta + 1) * 3 + {KNIGHT, BISHOP, ROOK}
Queen-promotions are encoded as ordinary queen moves; decode restores the
promotion flag from board context (pawn moving to the last rank).
"""
import chess
import numpy as np

NUM_MOVE_TYPES = 73
NUM_ACTIONS = 64 * NUM_MOVE_TYPES

_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _mirror(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def move_to_index(move: chess.Move, flip: bool) -> int:
    if flip:
        move = _mirror(move)
    df = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        mtype = 64 + (df + 1) * 3 + _PROMO_PIECES.index(move.promotion)
    elif (df, dr) in _KNIGHT:
        mtype = 56 + _KNIGHT.index((df, dr))
    else:
        dist = max(abs(df), abs(dr))
        mtype = _DIRECTIONS.index((_sign(df), _sign(dr))) * 7 + dist - 1
    return move.from_square * NUM_MOVE_TYPES + mtype


def index_to_move(index: int, flip: bool, board: chess.Board) -> chess.Move:
    """Decode an action index. `board` is the real (unmirrored) position and is
    only used to restore the implicit queen-promotion flag."""
    from_sq, mtype = divmod(index, NUM_MOVE_TYPES)
    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    promotion = None
    if mtype >= 64:
        u = mtype - 64
        df, dr = u // 3 - 1, 1
        promotion = _PROMO_PIECES[u % 3]
    elif mtype >= 56:
        df, dr = _KNIGHT[mtype - 56]
    else:
        d, dist_minus_1 = divmod(mtype, 7)
        df = _DIRECTIONS[d][0] * (dist_minus_1 + 1)
        dr = _DIRECTIONS[d][1] * (dist_minus_1 + 1)
    move = chess.Move(from_sq, chess.square(ff + df, fr + dr), promotion=promotion)
    if flip:
        move = _mirror(move)
    if (
        move.promotion is None
        and board.piece_type_at(move.from_square) == chess.PAWN
        and chess.square_rank(move.to_square) in (0, 7)
    ):
        move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move


def legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    flip = board.turn == chess.BLACK
    for mv in board.legal_moves:
        mask[move_to_index(mv, flip)] = True
    return mask
