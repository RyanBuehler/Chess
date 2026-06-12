"""PGN -> GameRecord with one-hot policy targets (imitation learning)."""
import io

import chess
import chess.pgn

from chessrl.chess_env.moves import move_to_index
from chessrl.selfplay.records import GameRecord, RecordBuilder

_RESULT_Z = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def record_from_pgn(pgn: str) -> GameRecord:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("not a valid PGN")
    z_white = _RESULT_Z[game.headers["Result"]]
    board = game.board()
    builder = RecordBuilder()
    for move in game.mainline_moves():
        idx = move_to_index(move, board.turn == chess.BLACK)
        builder.add(board, [idx], [1], idx)
        board.push(move)
    return builder.finalize(z_white)
