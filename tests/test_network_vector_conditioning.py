import pytest
import torch
from chessrl.config.config import NetworkConfig
from chessrl.chess_env.encoding import NUM_PLANES
from chessrl.model.network import PolicyValueNet


def test_goal_cond_defaults_to_planes():
    assert NetworkConfig().goal_cond == "planes"


def test_goal_cond_accepts_vector():
    assert NetworkConfig(goal_cond="vector").goal_cond == "vector"


def test_goal_cond_rejects_unknown():
    with pytest.raises(ValueError):
        NetworkConfig(goal_cond="bogus")


_CFG = NetworkConfig(blocks=2, filters=16, goal_cond="vector")


def _board_batch(n=4):
    return torch.zeros(n, NUM_PLANES, 8, 8)


def test_vector_forward_shapes_and_range():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters
    x = _board_batch(4)
    gv = torch.randn(4, d)
    dl = torch.zeros(4, 1)
    logits, value = net(x, deadline=dl, goal_vec=gv)
    assert logits.shape == (4, 4672)
    assert value.shape == (4, 1)
    assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0  # sigmoid


def test_vector_forward_requires_goal_vec():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    with pytest.raises(ValueError):
        net(_board_batch(2), deadline=torch.zeros(2, 1), goal_vec=None)


def test_film_actually_conditions():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters
    x = _board_batch(1)
    dl = torch.zeros(1, 1)
    # distinct goal vectors should produce distinct values (after a forward that
    # exercises the FiLM MLP with non-zero params).
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.1)
        va = net(x, deadline=dl, goal_vec=torch.full((1, d), -2.0))[1]
        vb = net(x, deadline=dl, goal_vec=torch.full((1, d), 2.0))[1]
    assert abs(float(va) - float(vb)) > 1e-5


def test_embed_shape_and_determinism():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    x = _board_batch(3)
    e1 = net.embed(x)
    e2 = net.embed(x)
    assert e1.shape == (3, _CFG.filters)
    assert torch.allclose(e1, e2)


def test_win_vector_present():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    assert net.win_vector.shape == (_CFG.filters,)


def test_planes_mode_unchanged():
    # Default goal_cond="planes" still builds and runs the legacy interface.
    cfg = NetworkConfig(blocks=2, filters=16)  # goal_cond defaults to "planes"
    from chessrl.goals.encoding import GOAL_PLANES
    net = PolicyValueNet(cfg, goal_conditioned=True).eval()
    x = torch.zeros(2, NUM_PLANES + GOAL_PLANES, 8, 8)
    logits, value = net(x, deadline=torch.zeros(2, 1))
    assert logits.shape == (2, 4672) and value.shape == (2, 1)


import numpy as np
import chess
from chessrl.chess_env.encoding import encode_board, to_model_input
from chessrl.model.network import VectorGoalNetEvaluator


def _planes(n=3):
    b = chess.Board()
    return np.stack([to_model_input(encode_board(b)) for _ in range(n)]).astype(np.float32)


def test_evaluator_requires_vector_net():
    bad = PolicyValueNet(NetworkConfig(blocks=2, filters=16))  # planes default
    with pytest.raises(AssertionError):
        VectorGoalNetEvaluator(bad)


def test_evaluate_planes_shapes():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n, d = 3, _CFG.filters
    pol, val = ev.evaluate_planes(_planes(n), np.zeros((n, d), np.float32), np.zeros(n, np.float32))
    assert pol.shape == (n, 4672) and val.shape == (n,)
    assert val.min() >= 0.0 and val.max() <= 1.0


def test_embed_boards_shape():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    e = ev.embed_boards([chess.Board(), chess.Board()])
    assert e.shape == (2, _CFG.filters)


def test_win_value_uses_win_vector():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n = 2
    wv = ev.win_value(_planes(n), np.zeros(n, np.float32))
    # equals evaluate_planes under the net's win_vector broadcast
    win_vecs = np.broadcast_to(net.win_vector.detach().numpy(), (n, _CFG.filters)).copy()
    _, val = ev.evaluate_planes(_planes(n), win_vecs, np.zeros(n, np.float32))
    assert np.allclose(wv, val, atol=1e-5)


def test_empty_batch():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    pol, val = ev.evaluate_planes(
        np.zeros((0, 21, 8, 8), np.float32), np.zeros((0, _CFG.filters), np.float32), np.zeros(0, np.float32)
    )
    assert pol.shape[0] == 0 and val.shape[0] == 0
