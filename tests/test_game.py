import chess

from chessrl.chess_env.game import terminal_value


def test_ongoing_is_none():
    assert terminal_value(chess.Board()) is None


def test_checkmate_is_minus_one_for_side_to_move():
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate
        board.push(chess.Move.from_uci(uci))
    assert terminal_value(board) == -1.0          # white to move, mated


def test_stalemate_is_zero():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert terminal_value(board) == 0.0


def test_insufficient_material_is_zero():
    assert terminal_value(chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")) == 0.0


def test_threefold_claimable_is_zero():
    board = chess.Board()
    for _ in range(2):
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            board.push(chess.Move.from_uci(uci))
    assert terminal_value(board) == 0.0
