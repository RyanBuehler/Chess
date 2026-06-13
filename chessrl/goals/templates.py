"""Goal templates: the value-agnostic delta vocabulary (spec sec 5, sec 6).

A goal is a deadline-bounded target delta expressed in the board's own feature
vocabulary. A ``GoalTemplate`` is ``(kind, params, deadline)`` at the piece-type
abstraction. ``kind`` is one of::

    capture       -- a count_delta: protagonist removes an opponent piece of a type
    reach_rank    -- protagonist gets a piece of a type onto a given rank
    reach_square  -- protagonist gets a piece of a type onto a given square
    check         -- protagonist gives check
    castle        -- protagonist castles (either side)
    promote       -- protagonist promotes a pawn
    win           -- the apex goal: protagonist wins the game

Identity (``key()``) is the delta kind + its params at the piece-type
abstraction. **The deadline is deliberately NOT part of identity** (spec sec 6):
a template's identity is the delta; the deadline is a difficulty knob refined by
child-spawning. This lets the repertoire individuate "take the queen" from "take
a pawn" while treating "take a knight in 15" and "...in 20" as the same template.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

# kinds
CAPTURE = "capture"
REACH_RANK = "reach_rank"
REACH_SQUARE = "reach_square"
CHECK = "check"
CASTLE = "castle"
PROMOTE = "promote"
WIN = "win"

KINDS = (CAPTURE, REACH_RANK, REACH_SQUARE, CHECK, CASTLE, PROMOTE, WIN)

# Human-readable piece-type names for goal descriptions (chess.PAWN..KING).
_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


@dataclass(frozen=True)
class GoalTemplate:
    """A deadline-bounded target delta.

    params is a tuple of (name, value) pairs (sorted) so the template is
    hashable and its identity is canonical and order-independent.
    """

    kind: str
    params: tuple = field(default_factory=tuple)
    deadline: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"bad goal kind {self.kind}")

    # --- canonical identity (deadline excluded) ---------------------------
    def key(self) -> tuple:
        """Canonical repertoire identity: (kind, params). Deadline excluded."""
        return (self.kind, self.params)

    def is_win(self) -> bool:
        return self.kind == WIN

    def describe(self) -> str:
        """A short human-readable phrase for this goal (UI/live-feed display).

        Schema-agnostic on the consumer side: callers treat the result as an
        opaque string. Examples: "win", "capture knight", "reach rank 8",
        "give check", "castle", "promote"."""
        if self.kind == WIN:
            return "win"
        if self.kind == CAPTURE:
            return f"capture {_PIECE_NAMES.get(self.param('piece_type'), 'piece')}"
        if self.kind == REACH_RANK:
            return f"reach rank {self.param('rank')}"
        if self.kind == REACH_SQUARE:
            sq = self.param("square")
            name = chess.square_name(sq) if isinstance(sq, int) else sq
            return f"reach {name}"
        if self.kind == CHECK:
            return "give check"
        if self.kind == CASTLE:
            return "castle"
        if self.kind == PROMOTE:
            return "promote"
        return self.kind

    def param(self, name, default=None):
        for k, v in self.params:
            if k == name:
                return v
        return default

    # --- constructors -----------------------------------------------------
    @staticmethod
    def _params(**kw) -> tuple:
        return tuple(sorted(kw.items()))

    @classmethod
    def capture(cls, piece_type: int, deadline: int) -> "GoalTemplate":
        return cls(CAPTURE, cls._params(piece_type=piece_type), deadline)

    @classmethod
    def reach_rank(cls, piece_type: int, rank: int, deadline: int) -> "GoalTemplate":
        return cls(REACH_RANK, cls._params(piece_type=piece_type, rank=rank), deadline)

    @classmethod
    def reach_square(cls, piece_type: int, square: int, deadline: int) -> "GoalTemplate":
        return cls(REACH_SQUARE, cls._params(piece_type=piece_type, square=square), deadline)

    @classmethod
    def check(cls, deadline: int) -> "GoalTemplate":
        return cls(CHECK, (), deadline)

    @classmethod
    def castle(cls, deadline: int) -> "GoalTemplate":
        return cls(CASTLE, (), deadline)

    @classmethod
    def promote(cls, deadline: int) -> "GoalTemplate":
        return cls(PROMOTE, (), deadline)

    @classmethod
    def win(cls, deadline: int) -> "GoalTemplate":
        return cls(WIN, (), deadline)


# The apex goal. Deadline is the (non-identity) horizon; a large default lets
# "win" be pursued over a whole game when assigned directly.
WIN_GOAL = GoalTemplate.win(deadline=512)
