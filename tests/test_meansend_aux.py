"""Test that means-end concurrent self-play publishes live-feed frames with
cluster-goal aux (both-sides table: cols, to_move, rows with 'goal' label)."""
import chess
import numpy as np

from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.selfplay.concurrent import play_meansend_games_concurrent, _meansend_aux
from tests.test_meansend_selfplay import FakeVectorEval, ReadyGoalSpace


class _Side:
    def __init__(self, active_cluster):
        self.active_cluster = active_cluster
    def is_terminal(self):
        return self.active_cluster < 0


class _Game:
    def __init__(self, w_cluster, b_cluster):
        self.sides = {chess.WHITE: _Side(w_cluster), chess.BLACK: _Side(b_cluster)}


def test_aux_shows_label_inline_and_tips_for_hover():
    labels = {
        3: {"label": "win material (+1.1)",
            "features": {"material": 1.1, "captured": 1.0}, "n": 20},
    }
    g = _Game(w_cluster=3, b_cluster=7)   # cluster 7 has no label
    aux = _meansend_aux(g, chess.WHITE, estimator=None, cluster_labels=labels)
    goal_row = aux["rows"][0]
    assert goal_row[1] == "cluster 3 — win material (+1.1)"   # White: inline label
    assert goal_row[2] == "cluster 7"                          # Black: no label -> bare id
    # hover detail for the goal cell
    assert aux["tips"]["White"]["cluster"] == 3
    assert aux["tips"]["White"]["features"]["material"] == 1.1
    assert aux["tips"]["Black"] is None                        # no label -> no tip


def test_aux_terminal_side_has_no_tip():
    g = _Game(w_cluster=-1, b_cluster=3)   # White terminal (pursuing win)
    labels = {3: {"label": "castle", "features": {"castled": 1.0}, "n": 9}}
    aux = _meansend_aux(g, chess.BLACK, estimator=None, cluster_labels=labels)
    assert aux["rows"][0][1] == "win"
    assert aux["tips"]["White"] is None
    assert aux["tips"]["Black"]["label"] == "castle"


class _CapturingPublisher:
    def __init__(self):
        self.frames = []

    def publish(self, game_id, payload):
        self.frames.append((game_id, payload))

    def close(self):
        pass


def test_meansend_aux_publishes_goal_table():
    """play_meansend_games_concurrent publishes frames with a cluster-goal aux table."""
    gs = ReadyGoalSpace()
    pub = _CapturingPublisher()

    results = play_meansend_games_concurrent(
        FakeVectorEval(),
        MCTSConfig(simulations=4, leaves_per_tree=1),
        SelfPlayConfig(ply_cap=8, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="emergent", win_floor=0.0, goal_window=3, deadline_max=60),
        gs,
        np.full(4, -1.0, np.float32),
        np.random.default_rng(42),
        num_games=2,
        publisher=pub,
        game_id_prefix="me_",
    )

    assert len(results) == 2
    assert pub.frames, "means-end driver emitted no live-feed frames"

    for topic, payload in pub.frames:
        aux = payload.get("aux")
        assert isinstance(aux, dict), f"aux must be a dict, got {type(aux)!r} for topic {topic!r}"
        assert aux.get("cols") == ["White", "Black"], f"bad cols: {aux.get('cols')}"
        assert aux.get("to_move") in (0, 1), f"bad to_move: {aux.get('to_move')}"
        rows = aux.get("rows")
        assert isinstance(rows, list) and rows, "aux rows must be a non-empty list"
        row_labels = [r[0] for r in rows]
        assert row_labels[0] == "goal", f"first row label must be 'goal', got {row_labels}"

    for slot in ("me_0", "me_1"):
        game_frames = [p for t, p in pub.frames if t == slot]
        assert game_frames, f"no frames for {slot}"
        assert game_frames[-1]["done"] is True, f"no terminal frame for {slot}"
        assert game_frames[-1]["z"] in (-1, 0, 1)
