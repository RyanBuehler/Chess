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


def test_vector_forward_dual_head_shapes_and_ranges():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters
    x = _board_batch(4); gv = torch.randn(4, d); dl = torch.zeros(4, 1)
    logits, v_win, v_goal = net(x, deadline=dl, goal_vec=gv)
    assert logits.shape == (4, 4672)
    assert v_win.shape == (4, 1) and v_goal.shape == (4, 1)
    assert float(v_win.min()) >= -1.0 and float(v_win.max()) <= 1.0   # tanh
    assert float(v_goal.min()) >= 0.0 and float(v_goal.max()) <= 1.0  # sigmoid


def test_vector_requires_goal_vec():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    with pytest.raises(ValueError):
        net(_board_batch(2), deadline=torch.zeros(2, 1), goal_vec=None)


def test_win_head_is_goal_agnostic():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters; x = _board_batch(1); dl = torch.zeros(1, 1)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.1)
        _, vw_a, vg_a = net(x, deadline=dl, goal_vec=torch.full((1, d), -2.0))
        _, vw_b, vg_b = net(x, deadline=dl, goal_vec=torch.full((1, d), 2.0))
    assert abs(float(vw_a) - float(vw_b)) < 1e-6      # win value invariant to goal
    assert abs(float(vg_a) - float(vg_b)) > 1e-5      # goal value varies with goal


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


def test_evaluate_planes_dual_head_shapes():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n, d = 3, _CFG.filters
    pol, vw, vg = ev.evaluate_planes(_planes(n), np.zeros((n, d), np.float32), np.zeros(n, np.float32))
    assert pol.shape == (n, 4672) and vw.shape == (n,) and vg.shape == (n,)
    assert vw.min() >= -1.0 and vw.max() <= 1.0
    assert vg.min() >= 0.0 and vg.max() <= 1.0


def test_embed_boards_shape():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    e = ev.embed_boards([chess.Board(), chess.Board()])
    assert e.shape == (2, _CFG.filters)


def test_win_value_is_goal_agnostic():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n, d = 2, _CFG.filters
    wv = ev.win_value(_planes(n), np.zeros(n, np.float32))
    # equals v_win from evaluate_planes under ANY goal vectors (goal-agnostic)
    _, vw, _ = ev.evaluate_planes(_planes(n), np.full((n, d), 3.0, np.float32), np.zeros(n, np.float32))
    assert np.allclose(wv, vw, atol=1e-5)


def test_empty_batch():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    pol, vw, vg = ev.evaluate_planes(
        np.zeros((0, 21, 8, 8), np.float32), np.zeros((0, _CFG.filters), np.float32), np.zeros(0, np.float32)
    )
    assert pol.shape[0] == 0 and vw.shape[0] == 0 and vg.shape[0] == 0


def test_vector_deadline_scaled_internally():
    """Forward scales raw deadlines internally; evaluate_planes must pass raw values.

    deadline=60 and deadline=600 both clamp to 1.0 after /DEADLINE_SCALE=60,
    so they must produce the SAME v_goal.  deadline=0 yields a different scaled
    value (0.0 vs 1.0) so, with non-trivial FiLM weights, must produce a
    DIFFERENT v_goal.
    """
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    # Randomise params so FiLM is non-trivial (same pattern as test_win_head_is_goal_agnostic)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.1)
    ev = VectorGoalNetEvaluator(net)
    n, d = 1, _CFG.filters
    pl = _planes(n)
    gv = np.zeros((n, d), np.float32)

    _, _, vg_60 = ev.evaluate_planes(pl, gv, np.array([60.0], np.float32))
    _, _, vg_600 = ev.evaluate_planes(pl, gv, np.array([600.0], np.float32))
    _, _, vg_0 = ev.evaluate_planes(pl, gv, np.array([0.0], np.float32))

    # Both 60 and 600 clamp to 1.0 → same output
    assert abs(float(vg_60[0]) - float(vg_600[0])) < 1e-6, (
        f"deadline=60 and deadline=600 should give same v_goal, got {vg_60[0]} vs {vg_600[0]}"
    )
    # 0 vs 1.0 (scaled) → different output with non-trivial FiLM
    assert abs(float(vg_60[0]) - float(vg_0[0])) > 1e-5, (
        f"deadline=60 and deadline=0 should give different v_goal, got {vg_60[0]} vs {vg_0[0]}"
    )
