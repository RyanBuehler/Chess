"""The goal repertoire: minting, child-spawning, per-template statistics
(plan Task 4.1; spec sec 6).

The repertoire is the append-only set of goal *templates* the agent has ever
seen achieved, keyed by ``GoalTemplate.key()`` (kind + params at the piece-type
abstraction; deadline excluded from identity). Each template carries running
statistics: attempts, successes, and a sliding window of the most recent
outcomes (window size == ``lp_window``). The curriculum (Task 4.2) reads these
to compute learning-progress and novelty.

Minting rule (spec sec 6)
-------------------------
The first time a game's verifier records a delta whose *template key* is not in
the repertoire, add that template. The deadline is initialized from the move it
occurred (the ply offset at which the protagonist first achieved it), capped by
``deadline_max``. Identity is append-only: a re-seen delta mints nothing new.

Child-spawning / refinement (spec sec 6)
---------------------------------------
When a template's windowed success rate plateaus HIGH (>= ``plateau_threshold``
over a full window of ``lp_window`` attempts), spawn a tighter-deadline child by
tightening the deadline (T -> T - ``child_delta``, floored at 1). The child
shares the parent's delta identity (same ``key()``) but a smaller deadline, so
it is a harder variant. The same first-seen trigger applies: a child already in
the repertoire is not respawned.

Serialization
-------------
``save``/``load`` round-trip the repertoire to JSON in the run dir so workers
can reload it (alongside the net checkpoint) and resume reconstructs it.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import chess

from chessrl.goals import templates as T
from chessrl.goals.templates import WIN_GOAL, GoalTemplate
from chessrl.goals.verifier import achieved_by_deadline

# The delta vocabulary the repertoire mines over (piece-type-level; spec sec 5).
# Mirrors the HER candidate vocabulary so minting and HER agree on individuation.
_CANDIDATE_KINDS = (
    [("capture", {"piece_type": pt}) for pt in
     (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)]
    + [("check", {}), ("castle", {}), ("promote", {}),
       ("reach_rank", {"piece_type": chess.PAWN, "rank": 7})]
)


def _candidate_templates(deadline: int) -> list[GoalTemplate]:
    d = max(1, deadline)
    return [
        GoalTemplate.capture(chess.PAWN, d),
        GoalTemplate.capture(chess.KNIGHT, d),
        GoalTemplate.capture(chess.BISHOP, d),
        GoalTemplate.capture(chess.ROOK, d),
        GoalTemplate.capture(chess.QUEEN, d),
        GoalTemplate.check(d),
        GoalTemplate.castle(d),
        GoalTemplate.promote(d),
        GoalTemplate.reach_rank(chess.PAWN, rank=7, deadline=d),
    ]


@dataclass
class TemplateStats:
    """Per-template running statistics.

    ``attempts``/``successes`` are lifetime counters; ``window`` is the most
    recent ``maxlen`` outcomes (1 == achieved, 0 == not) for the LP estimator.
    """

    attempts: int = 0
    successes: int = 0
    window: deque = field(default_factory=lambda: deque(maxlen=200))

    def record(self, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        self.window.append(1 if success else 0)

    def window_rate(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)


class Repertoire:
    """Append-only template set + per-template statistics (spec sec 6)."""

    def __init__(
        self,
        lp_window: int = 200,
        deadline_max: int = 60,
        plateau_threshold: float = 0.8,
        child_delta: int = 5,
    ):
        self.lp_window = lp_window
        self.deadline_max = deadline_max
        self.plateau_threshold = plateau_threshold
        self.child_delta = child_delta
        # key -> GoalTemplate (canonical, the minted deadline) ; key -> stats.
        # NOTE identity is key()==(kind,params); a tighter child shares the key
        # of its parent but a different deadline, so we additionally individuate
        # stored templates by (key, deadline) to keep parent + child distinct.
        self._templates: dict[tuple, GoalTemplate] = {}
        self._stats: dict[tuple, TemplateStats] = {}
        # The apex win-goal is always present so the curriculum can weight it.
        self.ensure(WIN_GOAL)

    # --- identity --------------------------------------------------------
    @staticmethod
    def _id(goal: GoalTemplate) -> tuple:
        """Stored identity: (kind, params, deadline). Distinguishes a parent
        from its tighter-deadline children while ``key()`` (kind, params) groups
        the delta family for novelty/LP-family queries."""
        return (goal.kind, goal.params, goal.deadline)

    # --- membership / stats ----------------------------------------------
    def templates(self) -> list[GoalTemplate]:
        return list(self._templates.values())

    def contains_key(self, goal: GoalTemplate) -> bool:
        """True if any template with this delta identity (key()) is present,
        regardless of deadline (the minting trigger is first-seen by key)."""
        return any(t.key() == goal.key() for t in self._templates.values())

    def ensure(self, goal: GoalTemplate) -> bool:
        """Add ``goal`` (identified by kind+params+deadline) if absent. Returns
        True if it was newly added."""
        gid = self._id(goal)
        if gid in self._templates:
            return False
        self._templates[gid] = goal
        self._stats[gid] = TemplateStats(window=deque(maxlen=self.lp_window))
        return True

    def stats(self, goal: GoalTemplate) -> TemplateStats:
        return self._stats[self._id(goal)]

    def record_attempt(self, goal: GoalTemplate, success: bool) -> None:
        gid = self._id(goal)
        if gid not in self._stats:
            self.ensure(goal)
            gid = self._id(goal)
        self._stats[gid].record(success)

    # --- minting from a game record --------------------------------------
    def update_from_record(self, rec, rng=None) -> list[GoalTemplate]:
        """Mint templates for every first-seen achieved delta in this game.

        Recomputes achieved deltas via the verifier (the on-disk record is the
        source of truth, spec sec 11). For each candidate delta achieved by a
        protagonist, if the delta's *key* is not yet in the repertoire, mint a
        template whose deadline is the ply offset at which it first occurred
        (capped by ``deadline_max``). Returns the list of newly minted
        templates.
        """
        from chessrl.training.her import reconstruct_states

        if not rec.has_goals():
            return []
        states = reconstruct_states(rec)
        minted: list[GoalTemplate] = []
        # Mine deltas over the whole game from ply 0 for each protagonist color.
        horizon = min(len(states) - 1, self.deadline_max)
        for protagonist in (chess.WHITE, chess.BLACK):
            for cand in _candidate_templates(horizon):
                ok, ply = achieved_by_deadline(states, cand, protagonist, start_ply=0)
                if not ok:
                    continue
                if self.contains_key(cand):
                    continue
                # Deadline initialized from the move it occurred, capped.
                deadline = max(1, min(self.deadline_max, ply))
                tmpl = GoalTemplate(kind=cand.kind, params=cand.params, deadline=deadline)
                if self.ensure(tmpl):
                    minted.append(tmpl)
        return minted

    def update_stats_from_record(self, rec) -> None:
        """Record one attempt/outcome per side for this game's *assigned* goals.

        Each side was assigned a goal at game start; whether the protagonist
        achieved that assigned goal by its deadline is one attempt (success iff
        achieved). Win-goal assignments are recorded against the apex template so
        its LP can climb (spec sec 12 — win's LP rises as competence accrues).
        The assigned goal is read from the record's stored blob (the source of
        truth) and the outcome recomputed exactly via the verifier."""
        from chessrl.selfplay.records import deserialize_goal
        from chessrl.training.her import reconstruct_states

        if not rec.has_goals():
            return
        states = reconstruct_states(rec)
        seen_proto = {}
        # The assigned goal is constant per side across the game; grab the first
        # ply each protagonist appears.
        for i in range(len(rec)):
            proto_white = rec.protagonist[i] == 1
            color = chess.WHITE if proto_white else chess.BLACK
            if color in seen_proto:
                continue
            seen_proto[color] = (i, deserialize_goal(str(rec.assigned_blob[i])))
        for color, (start_ply, goal) in seen_proto.items():
            self.ensure(goal)
            ok, _ = achieved_by_deadline(states, goal, color, start_ply=start_ply)
            self.record_attempt(goal, ok)

    def update_and_refine_from_record(self, rec) -> list[GoalTemplate]:
        """The full per-game feedback step (plan Task 4.3): mint first-seen
        deltas, update assigned-goal stats, then spawn any plateaued children.
        Returns all newly minted templates (mints + spawned children)."""
        minted = self.update_from_record(rec)
        self.update_stats_from_record(rec)
        minted += self.maybe_spawn_children()
        return minted

    # --- child-spawning --------------------------------------------------
    def maybe_spawn_children(self) -> list[GoalTemplate]:
        """Spawn tighter-deadline children for any template whose windowed
        success rate has plateaued high (full window, rate >= threshold).
        Returns the newly minted children."""
        spawned: list[GoalTemplate] = []
        # Snapshot to avoid mutating while iterating.
        for gid, tmpl in list(self._templates.items()):
            if tmpl.is_win():
                continue  # the apex goal has no tighter-deadline refinement here
            st = self._stats[gid]
            if len(st.window) < self.lp_window:
                continue  # not enough data to call a plateau
            if st.window_rate() < self.plateau_threshold:
                continue
            child_deadline = tmpl.deadline - self.child_delta
            if child_deadline < 1:
                continue
            child = GoalTemplate(kind=tmpl.kind, params=tmpl.params, deadline=child_deadline)
            if self.ensure(child):
                spawned.append(child)
        return spawned

    # --- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "lp_window": self.lp_window,
            "deadline_max": self.deadline_max,
            "plateau_threshold": self.plateau_threshold,
            "child_delta": self.child_delta,
            "templates": [
                {
                    "kind": t.kind,
                    "params": [[n, v] for n, v in t.params],
                    "deadline": t.deadline,
                    "attempts": self._stats[gid].attempts,
                    "successes": self._stats[gid].successes,
                    "window": list(self._stats[gid].window),
                }
                for gid, t in self._templates.items()
            ],
        }

    def save(self, path) -> None:
        # Atomic write so a worker reloading mid-write never reads a partial file.
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def from_dict(cls, data: dict) -> "Repertoire":
        rep = cls(
            lp_window=data.get("lp_window", 200),
            deadline_max=data.get("deadline_max", 60),
            plateau_threshold=data.get("plateau_threshold", 0.8),
            child_delta=data.get("child_delta", 5),
        )
        for row in data.get("templates", []):
            params = tuple(sorted((n, int(v)) for n, v in row.get("params", [])))
            tmpl = GoalTemplate(kind=row["kind"], params=params, deadline=int(row["deadline"]))
            rep.ensure(tmpl)
            st = rep.stats(tmpl)
            st.attempts = int(row.get("attempts", 0))
            st.successes = int(row.get("successes", 0))
            st.window = deque(
                (int(x) for x in row.get("window", [])), maxlen=rep.lp_window
            )
        return rep

    @classmethod
    def load(cls, path) -> "Repertoire":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def load_or_new(cls, path, lp_window: int = 200, deadline_max: int = 60) -> "Repertoire":
        """Load the persisted repertoire if present, else a fresh one. Used by
        workers (reload alongside the net checkpoint) and resume."""
        path = Path(path)
        if path.exists():
            try:
                return cls.load(path)
            except Exception:
                pass  # half-written; fall back to a fresh repertoire
        return cls(lp_window=lp_window, deadline_max=deadline_max)
