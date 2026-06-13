"""Goal -> conditioning planes + deadline scalar (spec sec 5, sec 9).

A goal is encoded as a *fixed* number of conditioning planes appended to the 21
board planes, plus a deadline scalar fed at the network head (NOT as a plane --
a spatially-constant plane is poorly legible to a conv tower; a scalar at the FC
is, per the spec's review finding).

The encoding is **compositional, never one-hot-per-goal** (spec sec 5): a goal is
a sparse target over fixed feature dimensions, so a newly minted goal is a new
*combination* of these fixed planes, needing no architecture change. Plane
layout (all protagonist-relative, mirrored for Black exactly like the board
encoding via ``chess.square_mirror``):

    plane 0          spatial mask (target square / rank), empty when non-spatial
    planes 1..6      piece-type one-hot broadcast (PAWN..KING), empty when N/A
    planes 7..13     kind one-hot broadcast (capture, reach_rank, reach_square,
                     check, castle, promote, win)

The win-goal is a reserved kind channel with no spatial mask and no piece-type
channel -- a neutral conditioning that, with the value head, collapses the goal
machinery back to the apex win objective.
"""
from __future__ import annotations

import chess
import numpy as np

from chessrl.goals import templates as T

# Fixed plane layout (compositional, never grows with the repertoire).
_SPATIAL = 0
_PIECE_BASE = 1            # planes 1..6 -> piece types 1..6 (PAWN..KING)
_N_PIECE = 6
_KIND_BASE = _PIECE_BASE + _N_PIECE   # 7
_KINDS = (T.CAPTURE, T.REACH_RANK, T.REACH_SQUARE, T.CHECK, T.CASTLE, T.PROMOTE, T.WIN)
_N_KIND = len(_KINDS)

GOAL_PLANES = 1 + _N_PIECE + _N_KIND   # 1 + 6 + 7 = 14


def _mirror_square(square: int, protagonist: bool) -> int:
    return square if protagonist == chess.WHITE else chess.square_mirror(square)


def _mirror_rank(rank: int, protagonist: bool) -> int:
    return rank if protagonist == chess.WHITE else 7 - rank


def encode_goal(goal: T.GoalTemplate, remaining: int, protagonist: bool):
    """Return (planes[GOAL_PLANES, 8, 8] float32, deadline_scalar).

    ``remaining`` is the moves-remaining-to-deadline, returned as the deadline
    scalar (fed at the network head). Planes are protagonist-relative; for a
    Black protagonist spatial targets are mirrored exactly like ``encode_board``.
    """
    planes = np.zeros((GOAL_PLANES, 8, 8), dtype=np.float32)

    # kind one-hot (broadcast).
    planes[_KIND_BASE + _KINDS.index(goal.kind)] = 1.0

    # piece-type one-hot (broadcast), when the goal names a piece type.
    pt = goal.param("piece_type")
    if pt is not None:
        planes[_PIECE_BASE + (pt - 1)] = 1.0

    # spatial mask, when the goal names a square or a rank.
    if goal.kind == T.REACH_SQUARE:
        sq = _mirror_square(goal.param("square"), protagonist)
        planes[_SPATIAL, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    elif goal.kind == T.REACH_RANK:
        rank = _mirror_rank(goal.param("rank"), protagonist)
        planes[_SPATIAL, rank, :] = 1.0

    return planes, remaining
