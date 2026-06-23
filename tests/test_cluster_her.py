import numpy as np
import chess
from chessrl.selfplay.records import RecordBuilder
from chessrl.training.her import reconstruct_states
from chessrl.training.cluster_her import cluster_goal_samples
from chessrl.chess_env.moves import move_to_index


class FakeEmbedder:
    """e(board) = [fullmove, n_pieces, 0, 0]; deltas grow monotonically."""
    def embed_boards(self, boards):
        out = [[float(b.fullmove_number), float(sum(1 for _ in b.piece_map())), 0.0, 0.0] for b in boards]
        return np.asarray(out, dtype=np.float32)


class FakeGoalSpace:
    """3 clusters along axis 0; assign by rounding delta[0] to {0,1,2}; tau large."""
    tau = 100.0
    def assign(self, delta):
        return int(min(2, max(0, round(float(delta[0])))))
    centroids = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], np.float32)
    def achieved(self, delta, cluster):
        return self.assign(delta) == cluster and float(np.linalg.norm(delta - self.centroids[cluster])) <= self.tau


def _game(d=4, n=4):
    b = RecordBuilder(); board = chess.Board()
    for ply in range(n):
        move = list(board.legal_moves)[0]
        idx = move_to_index(move, board.turn == chess.BLACK)
        b.add(board, [idx, idx + 1], [3, 1], played_index=idx, protagonist=board.turn,
              cluster_active=1, cluster_assigned=1,
              active_vec=np.array([1, 0, 0, 0], np.float32), explore=False)
        board.push(move)
    return b.finalize(z_white=1)


def _terminal_game(d=4, n=4):
    """A terminal-pursuit game: every ply active_cluster=-1, active_vec=win_vector."""
    b = RecordBuilder(); board = chess.Board()
    win_vec = np.full(d, -1.0, np.float32)
    for ply in range(n):
        move = list(board.legal_moves)[0]
        idx = move_to_index(move, board.turn == chess.BLACK)
        b.add(board, [idx, idx + 1], [3, 1], played_index=idx, protagonist=board.turn,
              cluster_active=-1, cluster_assigned=-1, active_vec=win_vec, explore=False)
        board.push(move)
    return b.finalize(z_white=1)


def test_terminal_ply_suppresses_goal_head_weight():
    # Adversarial review Bug B/F: terminal-pursuit active samples (cluster=-1) must
    # carry the win-head signal (v_win_mask=1) but ZERO goal-head weight, so the goal
    # head is never trained on the win-vector conditioning.
    rec = _terminal_game()
    states = reconstruct_states(rec)
    samples = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(), np.random.default_rng(0))
    actives = [s for s in samples if s.v_win_mask == 1.0]
    assert actives and all(s.cluster == -1 for s in actives)
    assert all(s.v_goal_weight == 0.0 for s in actives)   # goal-head loss suppressed
    # a normal sub-goal game still carries non-zero goal-head weight on its active samples
    rec2 = _game()
    s2 = [s for s in cluster_goal_samples(rec2, reconstruct_states(rec2), FakeEmbedder(),
                                          FakeGoalSpace(), np.random.default_rng(0)) if s.v_win_mask == 1.0]
    assert any(s.v_goal_weight > 0.0 for s in s2)


def test_active_sample_per_ply_with_win_target():
    rec = _game()
    states = reconstruct_states(rec)
    samples = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(), np.random.default_rng(0))
    actives = [s for s in samples if s.v_win_mask == 1.0]
    assert len(actives) == len(rec)             # one active per ply
    for s in actives:
        assert s.v_win in (-1.0, 0.0, 1.0)
        assert s.goal_vec.shape == (4,)


def test_her_samples_have_goal_targets_no_win_mask():
    rec = _game()
    states = reconstruct_states(rec)
    samples = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(), np.random.default_rng(0))
    her = [s for s in samples if s.v_win_mask == 0.0]
    assert her, "expected HER future/negative samples"
    for s in her:
        assert s.v_goal in (0.0, 1.0)


def test_vanilla_record_yields_nothing():
    b = RecordBuilder(); b.add(chess.Board(), [0], [1], 0)
    rec = b.finalize(z_white=0)
    out = cluster_goal_samples(rec, reconstruct_states(rec), FakeEmbedder(), FakeGoalSpace(), np.random.default_rng(0))
    assert out == []


def test_returns_empty_when_goalspace_unfit():
    """Fix 2: guard against GoalSpace not yet fit (centroids is None)."""
    gs = FakeGoalSpace()
    gs.centroids = None  # simulate unfit state
    rec = _game()
    states = reconstruct_states(rec)
    out = cluster_goal_samples(rec, states, FakeEmbedder(), gs, np.random.default_rng(0))
    assert out == []


def test_positive_requires_tau():
    """Fix 3: future-positives must pass goalspace.achieved() (tau-gated), not
    merely be nearest-cluster.  A FakeGoalSpace with tau=0.01 ensures that a
    delta near-but-not-within-tau of a centroid is ASSIGNED to that cluster but
    NOT achieved → must not appear as a v_goal==1.0 positive."""

    class TightGoalSpace:
        """Like FakeGoalSpace but tau is tiny so only exact-centroid hits achieve."""
        tau = 0.01
        centroids = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], np.float32)

        def assign(self, delta):
            return int(min(2, max(0, round(float(delta[0])))))

        def achieved(self, delta, cluster):
            return (self.assign(delta) == cluster and
                    float(np.linalg.norm(delta - self.centroids[cluster])) <= self.tau)

    class OffCentroidEmbedder:
        """e(board) produces embeddings whose inter-board delta is close to
        centroid 1 ([1,0,0,0]) but NOT within tau=0.01 of it."""
        def embed_boards(self, boards):
            # First board → [0, 0, 0, 0]; subsequent boards → [1.05, 0, 0, 0].
            # delta = [1.05, 0, 0, 0]: assign→1, norm=0.05 > tau=0.01 → not achieved.
            out = []
            for i, b in enumerate(boards):
                if b.fullmove_number == 1 and b.turn == chess.WHITE:
                    out.append([0.0, 0.0, 0.0, 0.0])
                else:
                    out.append([1.05, 0.0, 0.0, 0.0])
            return np.asarray(out, dtype=np.float32)

    gs = TightGoalSpace()
    rec = _game()
    states = reconstruct_states(rec)
    samples = cluster_goal_samples(rec, states, OffCentroidEmbedder(), gs, np.random.default_rng(0))
    # Cluster 1 should be reached (assign returns 1) but not achieved (norm > tau).
    # Therefore it must NOT appear as a v_goal==1.0 future-positive.
    positives_cluster1 = [s for s in samples if s.cluster == 1 and s.v_goal == 1.0 and s.v_win_mask == 0.0]
    assert positives_cluster1 == [], (
        f"Cluster 1 is reached but not tau-achieved; should not be a positive. "
        f"Got: {positives_cluster1}"
    )


def test_embeds_states_once_per_game():
    """Perf regression guard. The hot path must embed each game's states ONCE
    (one batched forward pass), NOT once per (i, t) delta pair. The per-pair
    version was O(T * deadline) un-batched forward passes (~48k/game, ~32 s/game)
    and wedged the first real v2 run for hours when the first cluster fit
    triggered a full-buffer rebuild. Equivalence of the batched deltas is
    covered by the content tests above (FakeEmbedder is batch-independent)."""

    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            self.calls = 0
            self.boards_seen = 0

        def embed_boards(self, boards):
            self.calls += 1
            self.boards_seen += len(boards)
            return super().embed_boards(boards)

    rec = _game(n=6)
    states = reconstruct_states(rec)
    emb = CountingEmbedder()
    cluster_goal_samples(rec, states, emb, FakeGoalSpace(), np.random.default_rng(0))
    assert emb.calls == 1, f"expected ONE batched embed_boards call, got {emb.calls}"
    assert emb.boards_seen == len(states)   # embedded every state exactly once
