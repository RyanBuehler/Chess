import numpy as np
import chess
from chessrl.config.config import NetworkConfig, TrainingConfig
from chessrl.model.network import PolicyValueNet
from chessrl.training.trainer import Trainer
from chessrl.training.vector_buffer import VectorGoalReplayBuffer
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import move_to_index
from tests.test_cluster_her import FakeEmbedder, FakeGoalSpace


def _cluster_game(n=4):
    b = RecordBuilder(); board = chess.Board()
    for _ in range(n):
        move = list(board.legal_moves)[0]
        idx = move_to_index(move, board.turn == chess.BLACK)
        b.add(board, [idx, idx + 1], [3, 1], played_index=idx, protagonist=board.turn,
              cluster_active=1, cluster_assigned=1,
              active_vec=np.array([1, 0, 0, 0], np.float32), explore=False)
        board.push(move)
    return b.finalize(z_white=1)


def test_train_steps_vector_runs_and_steps():
    cfg = TrainingConfig(batch_size=8, device="cpu")
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=4, goal_cond="vector"), goal_conditioned=True)
    tr = Trainer(net, cfg, run_dir=".")
    buf = VectorGoalReplayBuffer(1000, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(_cluster_game(), rng=np.random.default_rng(0))
    s0 = tr.step
    m = tr.train_steps_vector(buf, 3, np.random.default_rng(1))
    assert tr.step == s0 + 3
    assert m["policy_loss"] is not None and m["value_loss"] is not None
    assert np.isfinite(m["value_loss"])
