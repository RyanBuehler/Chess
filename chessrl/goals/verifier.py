"""Exact achieved-by-deadline verifier over a game record (spec sec 8, sec 19).

Given a sequence of board states (index 0 == the record's start position, each
subsequent index == the board after one half-move) and a ``GoalTemplate`` with a
deadline measured from ``start_ply``, decide whether the *protagonist* achieved
the goal within the deadline window, and at which ply.

This is the one component that is cheap and exact. It is value-agnostic: it only
reads rule-level features and occupancy (spec sec 4).

Conventions:
- ``achieved_ply`` is the 0-based index into ``states`` at which the goal first
  holds (the board *after* the achieving half-move).
- The deadline window is ``state_index - start_ply <= deadline`` (half-moves
  elapsed since start). The baseline for delta goals is ``states[start_ply]``.
- All goals are protagonist-relative. "Win" is the apex goal in the same
  machinery: real game-over -> evaluate the result for the protagonist.
"""
from __future__ import annotations

import chess

from chessrl.goals import templates as T
from chessrl.goals.features import board_features


def _protagonist_rank(rank: int, protagonist: bool) -> int:
    """Map a protagonist-relative rank index to an absolute board rank."""
    return rank if protagonist == chess.WHITE else 7 - rank


def _protagonist_square(square: int, protagonist: bool) -> int:
    return square if protagonist == chess.WHITE else chess.square_mirror(square)


def _holds(states, idx, goal: T.GoalTemplate, protagonist: bool, baseline) -> bool:
    """Does the goal hold at states[idx], protagonist-relative, vs baseline?"""
    board = states[idx]
    feats = board_features(board)
    kind = goal.kind
    opponent = not protagonist

    if kind == T.CAPTURE:
        pt = goal.param("piece_type")
        return feats.counts[(pt, opponent)] < baseline.counts[(pt, opponent)]

    if kind == T.CHECK:
        # Protagonist gives check: opponent is to move and in check.
        return board.is_check() and board.turn == opponent

    if kind == T.CASTLE:
        # Detect a castling move played by the protagonist into this state.
        return _move_into(states, idx, lambda b, m: b.is_castling(m), protagonist)

    if kind == T.PROMOTE:
        return _move_into(states, idx, lambda b, m: m.promotion is not None, protagonist)

    if kind == T.REACH_RANK:
        pt = goal.param("piece_type")
        target = _protagonist_rank(goal.param("rank"), protagonist)

        def pred(b, m):
            mover = b.piece_at(m.from_square)
            return (
                mover is not None
                and mover.piece_type == pt
                and chess.square_rank(m.to_square) == target
            )

        return _move_into(states, idx, pred, protagonist)

    if kind == T.REACH_SQUARE:
        pt = goal.param("piece_type")
        target = _protagonist_square(goal.param("square"), protagonist)

        def pred(b, m):
            mover = b.piece_at(m.from_square)
            return mover is not None and mover.piece_type == pt and m.to_square == target

        return _move_into(states, idx, pred, protagonist)

    if kind == T.WIN:
        if not board.is_game_over():
            return False
        res = feats.result
        if res == "1-0":
            return protagonist == chess.WHITE
        if res == "0-1":
            return protagonist == chess.BLACK
        return False  # draw -> not "achieved" as a binary win goal

    raise ValueError(f"unknown goal kind {kind}")


def _move_into(states, idx, pred, protagonist: bool) -> bool:
    """Apply ``pred(board_before, move)`` to the half-move that produced
    states[idx], requiring the mover to be the protagonist. ``board_before`` is
    states[idx-1] (so piece_at(from_square) is the moving piece pre-move)."""
    if idx == 0:
        return False
    board = states[idx]
    if not board.move_stack:
        return False
    move = board.peek()
    before = states[idx - 1]
    # The side that just moved is the one NOT to move in `before`'s successor;
    # equivalently, before.turn is the mover.
    if before.turn != protagonist:
        return False
    return pred(before, move)


def achieved_by_deadline(states, goal: T.GoalTemplate, protagonist: bool, start_ply: int):
    """Return (achieved: bool, achieved_ply: int|None).

    The goal is achieved if it first holds at some state index ``i`` with
    ``start_ply <= i`` and ``i - start_ply <= deadline``.
    """
    if start_ply >= len(states):
        return (False, None)

    baseline = board_features(states[start_ply])
    last = min(len(states) - 1, start_ply + goal.deadline)

    for idx in range(start_ply, last + 1):
        if _holds(states, idx, goal, protagonist, baseline):
            return (True, idx)
    return (False, None)
