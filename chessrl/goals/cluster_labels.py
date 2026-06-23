"""Post-hoc interpretation of discovered (k-means) goal clusters.

A cluster is an opaque centroid in the frozen encoder's embedding-delta space:
the learning discovers a *direction* in latent space, but it arrives unlabeled.
This module characterizes each cluster by the interpretable chess content of the
state transitions (s_i -> s_{i+w}) whose embedding-deltas fall in it: averaged
feature deltas plus a short human label (or "mixed" when too diffuse to name).

It is a *describer*, not part of the objective — the goals stay emergent; we only
read them after the fact. Regenerated each refit because cluster ids reshuffle.
Output -> goalspace/cluster_labels.json, surfaced on the LIVE view (inline label
+ hover fingerprint).

Features are computed from the perspective of the side to move at s_i, matching
the stm-relative frame the frozen encoder embeds in. Window w is even, so s_i and
s_{i+w} share a side to move (same frame). All features are endpoint-computable.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np

# Material values in pawns (king excluded — it never leaves the board).
_PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_MINORS = (chess.KNIGHT, chess.BISHOP)

# The interpretable feature keys, in display order.
FEATURE_KEYS = ("material", "captured", "lost", "developed", "pawn_advance",
                "promoted", "queen_trade", "castled")


def _material(board: chess.Board, color: bool) -> int:
    return sum(_PIECE_VAL[p.piece_type] for p in board.piece_map().values()
               if p.color == color)


def _piece_count(board: chess.Board, color: bool) -> int:
    return sum(1 for p in board.piece_map().values() if p.color == color)


def _minors_off_back_rank(board: chess.Board, color: bool) -> int:
    back = 0 if color == chess.WHITE else 7
    n = 0
    for pt in _MINORS:
        for sq in board.pieces(pt, color):
            if chess.square_rank(sq) != back:
                n += 1
    return n


def _pawn_advancement(board: chess.Board, color: bool) -> int:
    """Total ranks our pawns have advanced from their starting rank, summed."""
    total = 0
    for sq in board.pieces(chess.PAWN, color):
        r = chess.square_rank(sq)
        total += (r - 1) if color == chess.WHITE else (6 - r)
    return total


def _king_castled(a: chess.Board, b: chess.Board, color: bool) -> bool:
    """Heuristic: our king moved from the e-file to the g/c-file (a castle)."""
    ka = a.king(color)
    kb = b.king(color)
    if ka is None or kb is None:
        return False
    return chess.square_file(ka) == 4 and chess.square_file(kb) in (2, 6)


def transition_features(a: chess.Board, b: chess.Board) -> dict:
    """Interpretable chess-feature deltas of a -> b, from the perspective of the
    side to move in ``a``."""
    us = a.turn
    them = not us
    cap = max(0, _piece_count(a, them) - _piece_count(b, them))
    lost = max(0, _piece_count(a, us) - _piece_count(b, us))
    mat = ((_material(b, us) - _material(b, them))
           - (_material(a, us) - _material(a, them)))
    dev = _minors_off_back_rank(b, us) - _minors_off_back_rank(a, us)
    pawn_adv = _pawn_advancement(b, us) - _pawn_advancement(a, us)
    promoted = (len(b.pieces(chess.QUEEN, us)) > len(a.pieces(chess.QUEEN, us))
                and len(b.pieces(chess.PAWN, us)) < len(a.pieces(chess.PAWN, us)))
    queens_a = len(a.pieces(chess.QUEEN, us)) + len(a.pieces(chess.QUEEN, them))
    queens_b = len(b.pieces(chess.QUEEN, us)) + len(b.pieces(chess.QUEEN, them))
    return {
        "material": float(mat),
        "captured": float(cap),
        "lost": float(lost),
        "developed": float(dev),
        "pawn_advance": float(pawn_adv),
        "promoted": 1.0 if promoted else 0.0,
        "queen_trade": 1.0 if queens_a - queens_b >= 2 else 0.0,
        "castled": 1.0 if _king_castled(a, b, us) else 0.0,
    }


def cluster_label(feats: dict) -> str:
    """Derive a short human label from a cluster's averaged feature deltas, or
    "mixed" when no feature is salient enough to name. Priority is ordered so the
    most chess-meaningful signal wins."""
    mat = feats.get("material", 0.0)
    if mat >= 0.8:
        return f"win material ({mat:+.1f})"
    if mat <= -0.8:
        return f"sacrifice material ({mat:+.1f})"
    if feats.get("promoted", 0.0) >= 0.4:
        return "promote pawn"
    if feats.get("castled", 0.0) >= 0.4:
        return "castle"
    if feats.get("queen_trade", 0.0) >= 0.4:
        return "trade queens"
    if feats.get("developed", 0.0) >= 1.0:
        return "develop pieces"
    if feats.get("pawn_advance", 0.0) >= 2.0:
        return "advance pawns"
    if feats.get("captured", 0.0) >= 0.8 and feats.get("lost", 0.0) >= 0.8:
        return "trade pieces"
    return "mixed"


def characterize_clusters(states_per_game, embedder, goalspace, goal_window: int,
                          min_members: int = 5) -> dict:
    """Assign every window transition of every game to its cluster and aggregate
    the interpretable feature deltas per cluster.

    ``states_per_game`` is an iterable of per-game state lists (chess.Board, as
    from reconstruct_states). Returns ``{cluster_id: {"label", "features", "n"}}``
    for clusters with at least ``min_members`` transitions (others get "mixed"
    with whatever was seen)."""
    if getattr(goalspace, "centroids", None) is None:
        return {}
    k = goalspace.centroids.shape[0]
    sums = {c: {key: 0.0 for key in FEATURE_KEYS} for c in range(k)}
    counts = {c: 0 for c in range(k)}
    w = goal_window
    for states in states_per_game:
        T = len(states) - 1
        if T < w:
            continue
        emb = np.asarray(embedder.embed_boards(states), np.float32)
        for i in range(T - w + 1):
            c = goalspace.assign(emb[i + w] - emb[i])
            f = transition_features(states[i], states[i + w])
            for key in FEATURE_KEYS:
                sums[c][key] += f[key]
            counts[c] += 1
    out = {}
    for c in range(k):
        if counts[c] == 0:
            continue
        avg = {key: sums[c][key] / counts[c] for key in FEATURE_KEYS}
        label = cluster_label(avg) if counts[c] >= min_members else "mixed"
        out[c] = {"label": label,
                  "features": {key: round(avg[key], 3) for key in FEATURE_KEYS},
                  "n": counts[c]}
    return out


def save_cluster_labels(labels: dict, goalspace_dir) -> None:
    p = Path(goalspace_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "cluster_labels.json").write_text(json.dumps(labels))


def load_cluster_labels(goalspace_dir) -> dict:
    """Load the {cluster_id(int): {...}} map; {} if absent. JSON keys are strings
    on disk, so coerce back to int cluster ids."""
    p = Path(goalspace_dir) / "cluster_labels.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    return {int(k): v for k, v in raw.items()}
