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
