"""Smoke tests for the two new helpers in parallel_loop:
 - snapshot_frozen_encoder
 - observe_game_deltas

The full emergent loop is integration-tested by a short real run (8 games).
"""
import numpy as np
import chess
from chessrl.config.config import NetworkConfig
from chessrl.model.network import PolicyValueNet, VectorGoalNetEvaluator
from chessrl.training.parallel_loop import snapshot_frozen_encoder, observe_game_deltas
from chessrl.goals.goalspace import GoalSpace
from chessrl.config.config import GoalConfig
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import move_to_index


def test_snapshot_frozen_encoder(tmp_path):
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=8, goal_cond="vector"), goal_conditioned=True)
    ev = snapshot_frozen_encoder(net, tmp_path, NetworkConfig(blocks=2, filters=8, goal_cond="vector"), "cpu")
    assert isinstance(ev, VectorGoalNetEvaluator)
    assert (tmp_path / "frozen_encoder.pt").exists()
    e = ev.embed_boards([chess.Board()])
    assert e.shape == (1, 8)


def test_observe_game_deltas_fills_reservoir(tmp_path):
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=8, goal_cond="vector"), goal_conditioned=True)
    ev = snapshot_frozen_encoder(net, tmp_path, NetworkConfig(blocks=2, filters=8, goal_cond="vector"), "cpu")
    gs = GoalSpace(GoalConfig(goal_mode="emergent", goal_window=2, min_reservoir=3, cluster_k=2), ev, np.random.default_rng(0))
    b = RecordBuilder(); board = chess.Board()
    for _ in range(6):
        mv = list(board.legal_moves)[0]; idx = move_to_index(mv, board.turn == chess.BLACK)
        b.add(board, [idx], [1], idx); board.push(mv)
    rec = b.finalize(0)
    observe_game_deltas(gs, rec, ev, max_samples=4, rng=np.random.default_rng(0))
    assert len(gs.reservoir) > 0
