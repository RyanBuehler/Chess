"""PGN writing shared by the single-process loop and the self-play workers.
Behavior is identical to the original loop._save_pgn."""
from pathlib import Path

import chess
import chess.pgn

_RESULT_STR = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}


def save_pgn(board: chess.Board, z: int, path) -> None:
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = _RESULT_STR[z]
    Path(path).write_text(str(game))
