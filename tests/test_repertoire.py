"""Repertoire: minting, child-spawning, per-template stats, persistence
(plan Task 4.1; spec sec 6)."""
import chess
import numpy as np

from chessrl.goals.repertoire import Repertoire, TemplateStats
from chessrl.goals.templates import WIN_GOAL, GoalTemplate
from chessrl.selfplay.records import GameRecord, RecordBuilder


# --------------------------------------------------------------------------
# helpers: build a tiny goal GameRecord whose replayed moves achieve a known
# delta, so the repertoire's mint-from-record path can be exercised exactly.
# --------------------------------------------------------------------------
def _capture_game_record() -> GameRecord:
    """1.e4 d5 2.exd5 — White captures a pawn at ply 4 (index 4)."""
    b = chess.Board()
    builder = RecordBuilder()
    moves = ["e2e4", "d7d5", "e4d5"]
    for uci in moves:
        proto = b.turn
        # one legal-move policy target so the record is well-formed
        from chessrl.chess_env.moves import move_to_index

        idx = move_to_index(chess.Move.from_uci(uci), b.turn == chess.BLACK)
        builder.add(
            b, [idx], [1], idx,
            protagonist=proto,
            assigned_goal=GoalTemplate.capture(chess.PAWN, deadline=10),
            active_goal=GoalTemplate.capture(chess.PAWN, deadline=10),
        )
        b.push(chess.Move.from_uci(uci))
    return builder.finalize(0)


def test_first_seen_delta_mints_a_template():
    rep = Repertoire(lp_window=200, deadline_max=60)
    # only the apex win-goal is present at construction; no sub-goals yet
    assert not rep.contains_key(GoalTemplate.capture(chess.PAWN, 1))
    rec = _capture_game_record()
    minted = rep.update_from_record(rec)
    # White captured a pawn -> a capture-pawn template is minted.
    keys = {t.key() for t in rep.templates()}
    assert GoalTemplate.capture(chess.PAWN, 1).key() in keys
    assert any(m.kind == "capture" for m in minted)


def test_reseen_delta_does_not_remint():
    rep = Repertoire(lp_window=200, deadline_max=60)
    rec = _capture_game_record()
    rep.update_from_record(rec)
    n1 = len(rep.templates())
    rep.update_from_record(rec)
    n2 = len(rep.templates())
    assert n2 == n1  # identity is append-only; re-seen mints nothing new


def test_stats_track_attempts_successes_and_window():
    rep = Repertoire(lp_window=4, deadline_max=60)
    g = GoalTemplate.capture(chess.KNIGHT, deadline=10)
    rep.ensure(g)
    for outcome in (1, 0, 1, 1, 0):
        rep.record_attempt(g, bool(outcome))
    st = rep.stats(g)
    assert st.attempts == 5
    assert st.successes == 3
    # window capped at lp_window=4 -> only the last 4 outcomes retained
    assert len(st.window) == 4
    assert list(st.window) == [0, 1, 1, 0]


def test_plateau_high_spawns_tighter_deadline_child():
    rep = Repertoire(lp_window=8, deadline_max=60, plateau_threshold=0.8, child_delta=5)
    g = GoalTemplate.capture(chess.QUEEN, deadline=20)
    rep.ensure(g)
    # Drive the windowed success rate high and stable.
    for _ in range(8):
        rep.record_attempt(g, True)
    spawned = rep.maybe_spawn_children()
    # A tighter-deadline child of the same delta identity, deadline 20-5=15.
    assert any(
        c.key() == g.key() and c.deadline == 15 for c in spawned
    ), [(c.kind, c.deadline) for c in spawned]
    # Children are themselves in the repertoire (append-only).
    assert any(t.deadline == 15 and t.key() == g.key() for t in rep.templates())


def test_no_spawn_below_plateau():
    rep = Repertoire(lp_window=8, deadline_max=60, plateau_threshold=0.8, child_delta=5)
    g = GoalTemplate.capture(chess.ROOK, deadline=20)
    rep.ensure(g)
    for i in range(8):
        rep.record_attempt(g, i % 2 == 0)  # ~50% success
    assert rep.maybe_spawn_children() == []


def test_child_not_respawned():
    rep = Repertoire(lp_window=8, deadline_max=60, plateau_threshold=0.8, child_delta=5)
    g = GoalTemplate.capture(chess.QUEEN, deadline=20)
    rep.ensure(g)
    for _ in range(8):
        rep.record_attempt(g, True)
    rep.maybe_spawn_children()
    again = rep.maybe_spawn_children()
    assert again == []  # same child not minted twice


def test_save_load_roundtrip(tmp_path):
    rep = Repertoire(lp_window=4, deadline_max=60)
    g = GoalTemplate.capture(chess.BISHOP, deadline=12)
    rep.ensure(g)
    for outcome in (1, 0, 1):
        rep.record_attempt(g, bool(outcome))
    path = tmp_path / "repertoire.json"
    rep.save(path)
    rep2 = Repertoire.load(path)
    assert {t.key() for t in rep2.templates()} == {t.key() for t in rep.templates()}
    st = rep2.stats(g)
    assert st.attempts == 3
    assert st.successes == 2
    assert list(st.window) == [1, 0, 1]
    assert rep2.lp_window == 4


def test_win_template_present_by_default():
    rep = Repertoire(lp_window=200, deadline_max=60)
    # The apex win-goal is always part of the repertoire so the curriculum can
    # weight it (and the win-floor applies on top).
    assert any(t.is_win() for t in rep.templates())
