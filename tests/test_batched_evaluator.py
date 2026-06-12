import chess
import numpy as np
import torch

from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.config.config import NetworkConfig, TrainingConfig
from chessrl.model.network import BatchedNetEvaluator, NetEvaluator, PolicyValueNet
from chessrl.training.trainer import Trainer


def test_evaluate_many_shapes_and_softmax():
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    ev = BatchedNetEvaluator(net, device="cpu")
    boards = [chess.Board(), chess.Board()]
    boards[1].push(chess.Move.from_uci("e2e4"))
    policies, values = ev.evaluate_many(boards)
    assert policies.shape == (2, NUM_ACTIONS)
    assert policies.dtype == np.float32
    assert values.shape == (2,)
    assert values.dtype == np.float32
    np.testing.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(values >= -1.0) and np.all(values <= 1.0)


def test_evaluate_many_matches_single_evaluator():
    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    single = NetEvaluator(net, device="cpu")
    batched = BatchedNetEvaluator(net, device="cpu")
    board = chess.Board()
    p1, v1 = single.evaluate(board)
    p_many, v_many = batched.evaluate_many([board])
    np.testing.assert_allclose(p_many[0], p1, atol=1e-5)
    assert abs(v_many[0] - v1) < 1e-5


def test_empty_batch_returns_empty_arrays():
    net = PolicyValueNet(NetworkConfig(blocks=1, filters=16))
    ev = BatchedNetEvaluator(net, device="cpu")
    policies, values = ev.evaluate_many([])
    assert policies.shape == (0, NUM_ACTIONS)
    assert values.shape == (0,)


def test_from_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    trainer = Trainer(net, TrainingConfig(batch_size=4, device="cpu"), tmp_path)
    ckpt_path = trainer.save_checkpoint()    # saves ckpt_00000000.pt with {"model": ...}

    ref = BatchedNetEvaluator(net, device="cpu")
    loaded = BatchedNetEvaluator.from_checkpoint(
        ckpt_path, NetworkConfig(blocks=2, filters=16), device="cpu"
    )
    boards = [chess.Board()]
    p_ref, v_ref = ref.evaluate_many(boards)
    p_load, v_load = loaded.evaluate_many(boards)
    np.testing.assert_allclose(p_load, p_ref, atol=1e-5)
    np.testing.assert_allclose(v_load, v_ref, atol=1e-5)
