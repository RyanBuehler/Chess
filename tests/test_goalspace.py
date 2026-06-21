# tests/test_goalspace.py
import numpy as np
import chess
from pathlib import Path
from chessrl.config.config import GoalConfig
from chessrl.goals.goalspace import GoalSpace


class FakeEmbedder:
    """Deterministic embedder: e(board) = [material_count, n_pieces, fullmove, 0].
    d=4. Distinct boards map to distinct, meaningful vectors."""
    def embed_boards(self, boards):
        out = []
        for b in boards:
            mat = sum(len(b.pieces(pt, c)) * v for pt, v in
                      [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                       (chess.ROOK, 5), (chess.QUEEN, 9)] for c in (chess.WHITE, chess.BLACK))
            n = sum(1 for _ in b.piece_map())
            out.append([float(mat), float(n), float(b.fullmove_number), 0.0])
        return np.asarray(out, dtype=np.float32)


def _cfg(**kw):
    return GoalConfig(goal_mode="emergent", cluster_k=3, min_reservoir=20,
                      reservoir_size=200, refresh_every=50, **kw)


def test_delta_dim_and_value():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    b0 = chess.Board()
    b1 = chess.Board()
    b1.push_san("e4")
    d = gs.delta(b0, b1)
    assert d.shape == (4,)


def test_not_ready_before_min_reservoir():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    assert gs.ready is False
    b = chess.Board()
    for _ in range(5):
        gs.observe(b, b)
    assert gs.ready is False  # below min_reservoir and not fit


def test_fit_makes_ready_and_assigns():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    rng = np.random.default_rng(7)
    # synthesize 60 deltas in 3 separable groups by adding directly
    for grp in ([10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]):
        for _ in range(20):
            gs.observe_delta(np.array(grp, np.float32) + rng.normal(0, 0.1, 4).astype(np.float32))
    gs.fit()
    assert gs.ready is True
    assert gs.n_clusters == 3
    gid = gs.assign(np.array([10, 0, 0, 0], np.float32))
    assert 0 <= gid < 3
    # a delta near that centroid is achieved for that goal
    assert gs.achieved(gs.centroid(gid), gid) is True
    # a far delta is not achieved for that goal
    assert gs.achieved(np.array([0, 0, 10, 0], np.float32), gid) is False


def test_maybe_refresh_refits_on_boundary():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    for _ in range(30):
        gs.observe_delta(np.random.default_rng(0).normal(0, 1, 4).astype(np.float32))
    gs.fit()
    first = gs.centroids.copy()
    # add more, cross a refresh_every boundary
    for _ in range(30):
        gs.observe_delta(np.array([100, 100, 100, 0], np.float32))
    did = gs.maybe_refresh(games_seen=50)
    assert did is True
    assert not np.allclose(first, gs.centroids)  # refit moved centroids


def test_observe_uses_embedder_delta():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    b0 = chess.Board()
    b1 = chess.Board(); b1.push_san("e4")
    gs.observe(b0, b1)
    assert len(gs.reservoir) == 1


def test_save_load_roundtrip(tmp_path):
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    rng = np.random.default_rng(7)
    for grp in ([10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]):
        for _ in range(20):
            gs.observe_delta(np.array(grp, np.float32) + rng.normal(0, 0.1, 4).astype(np.float32))
    gs.fit()
    probe = np.array([10, 0, 0, 0], np.float32)
    gid_before = gs.assign(probe)
    tau_before = gs.tau

    gs.save(tmp_path / "goalspace")
    gs2 = GoalSpace.load(tmp_path / "goalspace", _cfg(), FakeEmbedder(), np.random.default_rng(99))

    assert gs2.ready is True
    assert gs2.n_clusters == gs.n_clusters
    assert np.allclose(gs2.centroids, gs.centroids)
    assert gs2.tau == tau_before
    assert gs2.assign(probe) == gid_before
    assert len(gs2.reservoir) == len(gs.reservoir)
