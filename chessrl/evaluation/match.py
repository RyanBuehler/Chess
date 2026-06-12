"""Deterministic match runner over the opening suite.

z is always from White's perspective (+1/0/-1). A ply-cap hit is adjudicated a
draw. Pairings alternate colors EXACTLY evenly (games must be even) and reuse the
same opening for each color-swapped pair, so color advantage never leaks into the
ratings (which see only (white, black, z) triples).
"""
from dataclasses import dataclass

import chess
import chess.pgn

from chessrl.chess_env.game import terminal_value
from chessrl.evaluation.openings import OPENINGS, opening_board

_RESULT_STR = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}


@dataclass
class MatchResult:
    white_name: str
    black_name: str
    z: int                  # White's perspective: +1 white win, 0 draw, -1 black win
    opening_idx: int
    pgn: str


def _board_to_pgn(board: chess.Board, z: int, white_name: str, black_name: str, opening_idx: int) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = _RESULT_STR[z]
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Opening"] = str(opening_idx)
    return str(game)


def play_single(white, black, opening_idx: int, max_plies: int) -> tuple[int, str]:
    """Play one game from opening_board(opening_idx). Returns (z, pgn_str)."""
    board = opening_board(opening_idx)
    while True:
        if board.outcome(claim_draw=True) is not None:
            term = terminal_value(board)
            if term == 0.0:
                z = 0
            else:                                    # loser is the side to move
                z = 1 if board.turn == chess.BLACK else -1
            break
        if len(board.move_stack) >= max_plies:      # ply cap counts total moves in game
            z = 0
            break
        player = white if board.turn == chess.WHITE else black
        board.push(player.play(board))
    pgn = _board_to_pgn(board, z, getattr(white, "name", "white"), getattr(black, "name", "black"), opening_idx)
    return z, pgn


def play_pairing(a, b, games: int, openings_start: int, max_plies: int) -> list:
    """Play `games` (must be even) games between a and b. Colors alternate
    exactly evenly: for each opening, a-as-White then b-as-White. Returns a list
    of MatchResult."""
    if games % 2 != 0:
        raise ValueError(f"games must be even (equal colors); got {games}")
    results = []
    pairs = games // 2
    for p in range(pairs):
        opening_idx = (openings_start + p) % len(OPENINGS)
        # a as White
        z, pgn = play_single(a, b, opening_idx, max_plies)
        results.append(MatchResult(a.name, b.name, z, opening_idx, pgn))
        # b as White (same opening)
        z, pgn = play_single(b, a, opening_idx, max_plies)
        results.append(MatchResult(b.name, a.name, z, opening_idx, pgn))
    return results
