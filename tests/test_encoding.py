import chess
import numpy as np

from chessrl.chess_env.encoding import NUM_PLANES, PLANE_FIFTY, encode_board, to_model_input


def test_startpos_shape_and_counts():
    pl = encode_board(chess.Board())
    assert pl.shape == (NUM_PLANES, 8, 8)
    assert pl.dtype == np.int8
    assert pl[0].sum() == 8        # our pawns
    assert pl[6].sum() == 8        # their pawns
    assert pl[5].sum() == 1        # our king
    assert pl[0, 1].sum() == 8     # our pawns on rank index 1
    assert pl[12].min() == 1       # white to move
    assert pl[13:17].sum() == 4 * 64  # all castling rights


def test_black_perspective_mirrors():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    pl = encode_board(board)        # black to move
    assert pl[12].max() == 0        # side to move is not white
    assert pl[0, 1].sum() == 8      # OUR (black) pawns appear on rank index 1 after mirroring
    # their e-pawn stands on e4; mirrored to e5 = rank index 4
    assert pl[6, 4, 4] == 1


def test_en_passant_plane():
    board = chess.Board()
    for uci in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        board.push(chess.Move.from_uci(uci))
    pl = encode_board(board)        # white to move, ep square d6
    assert pl[17, 5, 3] == 1        # d6 = rank idx 5, file idx 3
    assert pl[17].sum() == 1


def test_fifty_and_normalization():
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 37 90")
    pl = encode_board(board)
    assert pl[PLANE_FIFTY, 0, 0] == 37
    x = to_model_input(pl)
    assert x.dtype == np.float32
    assert abs(x[PLANE_FIFTY, 0, 0] - 0.37) < 1e-6
    assert x[0].max() == 1.0


def test_repetition_planes():
    board = chess.Board()
    assert encode_board(board)[19].max() == 0
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push(chess.Move.from_uci(uci))
    pl = encode_board(board)        # startpos repeated (2nd occurrence)
    assert pl[19].min() == 1
    assert pl[20].max() == 0
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push(chess.Move.from_uci(uci))
    pl = encode_board(board)        # 3rd occurrence
    assert pl[20].min() == 1
