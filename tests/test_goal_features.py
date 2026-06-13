import chess

from chessrl.goals.features import board_features


def test_features_startpos():
    f = board_features(chess.Board())
    assert f.counts[(chess.PAWN, chess.WHITE)] == 8
    assert f.counts[(chess.QUEEN, chess.BLACK)] == 1
    assert f.in_check is False
    assert f.result is None


def test_features_after_capture():
    b = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    b.push(chess.Move.from_uci("e4d5"))           # exd5 captures the queen
    f = board_features(b)
    assert f.counts[(chess.QUEEN, chess.BLACK)] == 0


def test_features_in_check_and_castling():
    # White king on e1 in check from a black rook on e8; no castling rights.
    b = chess.Board("4r3/8/8/8/8/8/8/4K3 w - - 0 1")
    f = board_features(b)
    assert f.in_check is True
    # castling tuple: (white_kingside, white_queenside, black_kingside, black_queenside)
    assert f.castling == (False, False, False, False)

    start = board_features(chess.Board())
    assert start.castling == (True, True, True, True)


def test_features_result_on_checkmate():
    # Fool's mate position (Black is checkmated... actually White checkmated): use a known mate.
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_checkmate()
    f = board_features(b)
    assert f.result == "0-1"


def test_counts_cover_all_piece_types_and_colors():
    f = board_features(chess.Board())
    for color in (chess.WHITE, chess.BLACK):
        for pt in range(1, 7):
            assert (pt, color) in f.counts
    assert f.counts[(chess.KING, chess.WHITE)] == 1
    assert f.counts[(chess.KNIGHT, chess.BLACK)] == 2
