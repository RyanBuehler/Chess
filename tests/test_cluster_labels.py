import chess
import numpy as np

from chessrl.goals.cluster_labels import (
    transition_features, cluster_label, characterize_clusters,
    save_cluster_labels, load_cluster_labels, FEATURE_KEYS,
)


def test_capture_gives_material_and_captured():
    a = chess.Board()                       # white to move, equal
    b = a.copy()
    b.remove_piece_at(chess.D7)             # black loses a pawn
    f = transition_features(a, b)           # us = white
    assert f["material"] == 1.0
    assert f["captured"] == 1.0
    assert f["lost"] == 0.0


def test_material_is_stm_relative():
    a = chess.Board()
    a.turn = chess.BLACK                    # now us = black
    b = a.copy()
    b.remove_piece_at(chess.D7)             # black (us) loses a pawn -> material down
    f = transition_features(a, b)
    assert f["material"] == -1.0
    assert f["lost"] == 1.0
    assert f["captured"] == 0.0


def test_castling_detected():
    a = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1")
    b = a.copy(); b.push_san("O-O")
    f = transition_features(a, b)
    assert f["castled"] == 1.0


def test_promotion_detected():
    a = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")   # white pawn about to promote
    b = a.copy(); b.push_san("a8=Q")
    f = transition_features(a, b)
    assert f["promoted"] == 1.0


def test_label_priority():
    base = {k: 0.0 for k in FEATURE_KEYS}
    assert cluster_label({**base, "material": 1.2}).startswith("win material")
    assert cluster_label({**base, "material": -1.2}).startswith("sacrifice material")
    assert cluster_label({**base, "castled": 1.0}) == "castle"
    assert cluster_label({**base, "promoted": 1.0}) == "promote pawn"
    assert cluster_label({**base, "developed": 2.0}) == "develop pieces"
    assert cluster_label({**base, "captured": 1.0, "lost": 1.0}) == "trade pieces"
    assert cluster_label(base) == "mixed"          # nothing salient


class _AxisEmbedder:
    """e(board) = [material_white, 0, 0, 0] -> delta along axis 0 tracks material."""
    def embed_boards(self, boards):
        out = [[float(sum({chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5,
                           chess.QUEEN:9, chess.KING:0}[p.piece_type]
                          for p in b.piece_map().values() if p.color == chess.WHITE)),
                0.0, 0.0, 0.0] for b in boards]
        return np.asarray(out, np.float32)


class _TwoClusterGoalSpace:
    """cluster 0 = 'no material change', cluster 1 = 'material drop' (a capture)."""
    centroids = np.array([[0, 0, 0, 0], [-1, 0, 0, 0]], np.float32)
    def assign(self, delta):
        return 1 if float(delta[0]) <= -0.5 else 0


def test_characterize_clusters_labels_and_counts():
    # Game where white captures a black pawn at ply 0->2 (window=2).
    a = chess.Board()
    mid = a.copy()
    end = a.copy(); end.remove_piece_at(chess.D7)   # white embedding unchanged; this
    # is a synthetic state list; window=2 means transition states[0]->states[2].
    states = [a, mid, end]
    labels = characterize_clusters([states], _AxisEmbedder(), _TwoClusterGoalSpace(),
                                   goal_window=2, min_members=1)
    # white material is unchanged across a->end (only black lost a pawn), so the
    # embedding delta (white material) is 0 -> cluster 0; but transition_features
    # is stm-relative and records captured=1, material=+1 for white.
    assert 0 in labels
    assert labels[0]["n"] == 1
    assert labels[0]["features"]["captured"] == 1.0
    assert labels[0]["label"].startswith("win material")


def test_save_load_round_trip(tmp_path):
    labels = {0: {"label": "castle", "features": {"material": 0.0}, "n": 7},
              3: {"label": "win material (+1.1)", "features": {"material": 1.1}, "n": 12}}
    save_cluster_labels(labels, tmp_path)
    loaded = load_cluster_labels(tmp_path)
    assert loaded[0]["label"] == "castle"
    assert loaded[3]["n"] == 12          # int keys restored
    assert load_cluster_labels(tmp_path / "nonexistent") == {}
