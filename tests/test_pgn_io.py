import chess

from chessrl.selfplay.pgn_io import save_pgn


def test_save_pgn_writes_result_and_moves(tmp_path):
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate, black wins
        board.push(chess.Move.from_uci(uci))
    path = tmp_path / "g.pgn"
    save_pgn(board, z=-1, path=path)
    text = path.read_text()
    assert '[Result "0-1"]' in text
    assert "f3" in text and "Qh4" in text


def test_save_pgn_result_mapping(tmp_path):
    for z, expected in [(1, "1-0"), (-1, "0-1"), (0, "1/2-1/2")]:
        path = tmp_path / f"g{z}.pgn"
        save_pgn(chess.Board(), z=z, path=path)
        assert f'[Result "{expected}"]' in path.read_text()
