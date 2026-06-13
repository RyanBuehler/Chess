"""Hindsight Experience Replay sample generation (spec sec 11, plan Task 3.3).

Goal value targets are generated AT TRAIN TIME from the stored game records (the
on-disk source of truth); relabeled samples are never persisted. Policy targets
are produced ONLY for the **active (searched) goal** at each ply (the stored
visit counts) — there is no HER for policy, because valid search visit counts
exist only for the goal that was actually searched (spec sec 11).

Value targets are protagonist-frame achievement probabilities
``P(protagonist achieves g by deadline | s)`` produced exactly by the verifier
(``achieved_by_deadline``), with three flavours weighted differently:

* **search-laundered** — the *active* goal at the ply was the one searched, so
  its achieved-by-deadline label is adversarially laundered (co-authored by the
  search, not only by a weak opponent). Preferred: weighted UP.
* **raw HER positive** — a "future" delta actually achieved later in the game.
  Optimistic (co-authored by a weak opponent), so weighted DOWN.
* **negative** — a delta that did NOT happen within its deadline window (incl.
  opponent-prevented deltas) → target 0. Weighted UP so V does not collapse to
  "everything achievable" (spec sec 11).

The samples carry the goal-conditioning planes + deadline scalar + sigmoid/BCE
value target + a per-sample weight; the trainer consumes them with BCE.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from chessrl.chess_env.moves import index_to_move
from chessrl.goals import templates as T
from chessrl.goals.templates import WIN_GOAL, GoalTemplate
from chessrl.goals.verifier import achieved_by_deadline, board_features
from chessrl.selfplay.records import GameRecord, deserialize_goal


@dataclass(frozen=True)
class HERWeights:
    """Relabel weights (spec sec 11). Search-laundered/negatives weighted UP, raw
    HER positives weighted DOWN."""

    search_laundered: float = 1.0
    her_positive: float = 0.3
    negative: float = 1.0
    future_samples: int = 2     # raw HER "future" positives sampled per ply
    negative_samples: int = 2   # negatives sampled per ply


@dataclass(frozen=True)
class GoalValueSample:
    """One HER value training example for the goal-conditioned net.

    ``ply`` indexes the source position in the record (the state the net sees);
    ``goal`` is the relabeled goal; ``remaining`` is the deadline scalar (plies
    to the goal's deadline at this state); ``target`` is the BCE label in [0,1];
    ``weight`` scales the per-sample BCE loss."""

    ply: int
    goal: GoalTemplate
    remaining: int
    target: float
    weight: float


def reconstruct_states(rec: GameRecord) -> list[chess.Board]:
    """Replay the recorded moves into a list of board snapshots.

    ``states[i]`` is the board at ply ``i`` (the position the net saw / searched
    at ply ``i``); ``states[len(rec)]`` is the final board after the last move.
    The record stores ``played`` (the chosen action index per ply in the
    side-to-move frame), which is exactly what the verifier needs to recompute
    deltas (spec sec 11 — "save raw games ... generate HER samples at train
    time via the verifier")."""
    board = chess.Board()
    states = [board.copy()]
    for idx in rec.played:
        move = index_to_move(int(idx), board.turn == chess.BLACK, board)
        board.push(move)
        states.append(board.copy())
    return states


# The delta vocabulary HER relabels over (piece-type-level; spec sec 5). Kept
# small and value-agnostic; mirrors the self-play assigner's enumeration.
def _candidate_goals(deadline: int) -> list[GoalTemplate]:
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


def _achieved_deltas_after(
    states: list[chess.Board], start_ply: int, protagonist: bool, remaining: int
) -> list[GoalTemplate]:
    """"Future" relabeling (spec sec 11): the candidate deltas the protagonist
    actually achieved within ``remaining`` plies after ``start_ply``."""
    out = []
    for g in _candidate_goals(remaining):
        ok, _ = achieved_by_deadline(states, g, protagonist, start_ply)
        if ok:
            out.append(g)
    return out


def goal_value_samples(
    rec: GameRecord,
    rng: np.random.Generator,
    weights: HERWeights | None = None,
    deadline_max: int = 60,
) -> list[GoalValueSample]:
    """Generate HER value samples for one goal game (spec sec 11, Task 3.3).

    For each ply S_i (protagonist = side to move there), produce:
      * the **search-laundered** target for the *active* goal searched at S_i;
      * a few **raw HER "future" positives** sampled from deltas achieved later;
      * a few **negatives** sampled from deltas that did NOT happen by deadline.

    Every target is exact via ``achieved_by_deadline`` with the goal's deadline
    set to the chosen ``remaining``. Returns an empty list for vanilla records.
    """
    if not rec.has_goals():
        return []
    w = weights or HERWeights()
    states = reconstruct_states(rec)
    samples: list[GoalValueSample] = []
    T_ = len(rec)

    for i in range(T_):
        protagonist = rec.protagonist[i] == 1  # 1 == White
        proto_color = chess.WHITE if protagonist else chess.BLACK

        # --- search-laundered: the active goal that was actually searched ----
        active = deserialize_goal(str(rec.active_blob[i]))
        # `remaining` is the active goal's own deadline horizon from this ply.
        remaining = min(active.deadline, deadline_max)
        ok, _ = achieved_by_deadline(states, active, proto_color, start_ply=i)
        samples.append(
            GoalValueSample(
                ply=i,
                goal=active,
                remaining=remaining,
                target=1.0 if ok else 0.0,
                weight=w.search_laundered,
            )
        )

        # Horizon for relabeled sub-goals at this ply: the rest of the game,
        # capped by deadline_max.
        remaining_window = min(T_ - i, deadline_max)
        if remaining_window <= 0:
            continue

        # --- raw HER "future" positives -------------------------------------
        achieved = _achieved_deltas_after(states, i, proto_color, remaining_window)
        if achieved and w.future_samples > 0:
            k = min(w.future_samples, len(achieved))
            picks = rng.choice(len(achieved), size=k, replace=False)
            for j in picks:
                g = achieved[int(j)]
                samples.append(
                    GoalValueSample(
                        ply=i,
                        goal=g,
                        remaining=g.deadline,
                        target=1.0,
                        weight=w.her_positive,
                    )
                )

        # --- negatives: deltas that did NOT happen within deadline ----------
        achieved_keys = {g.key() for g in achieved}
        candidates = _candidate_goals(remaining_window)
        negs = [g for g in candidates if g.key() not in achieved_keys]
        # Exclude a win-goal negative cheaply (win is the apex, handled by the
        # active/search-laundered path); negatives are sub-goal deltas.
        if negs and w.negative_samples > 0:
            k = min(w.negative_samples, len(negs))
            picks = rng.choice(len(negs), size=k, replace=False)
            for j in picks:
                g = negs[int(j)]
                # Exact label (should be 0, but verify — a candidate could hold
                # at a different deadline boundary than the window heuristic).
                ok_n, _ = achieved_by_deadline(states, g, proto_color, start_ply=i)
                samples.append(
                    GoalValueSample(
                        ply=i,
                        goal=g,
                        remaining=g.deadline,
                        target=1.0 if ok_n else 0.0,
                        weight=w.negative,
                    )
                )

    return samples
