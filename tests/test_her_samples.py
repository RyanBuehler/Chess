"""HER value-sample generation + BCE value loss / masked CE policy loss
(plan Task 3.3) and the wishful-thinking thermometer metrics (plan Task 3.4)."""
import chess
import numpy as np
import pytest
import torch

from chessrl.chess_env.moves import NUM_ACTIONS, move_to_index
from chessrl.config.config import NetworkConfig, TrainingConfig
from chessrl.goals.templates import WIN_GOAL, GoalTemplate
from chessrl.model.network import PolicyValueNet
from chessrl.selfplay.records import RecordBuilder
from chessrl.training.buffer import GoalReplayBuffer
from chessrl.training.her import HERWeights, goal_value_samples, reconstruct_states
from chessrl.training.trainer import Trainer


def _capture_queen_game():
    """A short game where White captures Black's queen: a goal achieved at a
    known ply. Build a goal record with the protagonist pursuing capture-queen.

    1. e4 d5 2. exd5 Qxd5 3. Nc3 (attacks queen) Qe5+ ... then White plays
    a capture of the queen. We construct a forced capture sequence:
        White pawn e4, Black ...d5, White exd5 (captures a pawn — not queen),
    so instead set up: Black queen lands where White can take it next move.
    """
    board = chess.Board()
    builder = RecordBuilder()
    # Goal: White captures a QUEEN within 8 plies.
    qgoal = GoalTemplate.capture(chess.QUEEN, deadline=8)
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5e5", "f1e2", "e5e4", "e2f3", "e4f3"]
    # The last move (e4f3) is Black's queen capturing — wrong direction. Adjust:
    # We want WHITE to capture the black queen. Build a line ending in White
    # taking the queen on f-file.
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5e5", "d2d4", "e5e4", "c3e4"]
    # After c3e4 White's knight captures the black queen on e4.
    b = chess.Board()
    for uci in moves:
        mv = chess.Move.from_uci(uci)
        idx = move_to_index(mv, b.turn == chess.BLACK)
        # Build a trivial 1-move visit distribution (the played move).
        builder.add(
            b, [idx], [4], idx,
            protagonist=b.turn,
            assigned_goal=qgoal if b.turn == chess.WHITE else WIN_GOAL,
            active_goal=qgoal if b.turn == chess.WHITE else WIN_GOAL,
        )
        b.push(mv)
    z = 0
    return builder.finalize(z), len(moves)


def test_reconstruct_states_matches_played():
    rec, n = _capture_queen_game()
    states = reconstruct_states(rec)
    assert len(states) == n + 1
    # The black queen is gone by the final state (White captured it on e4).
    final = states[-1]
    assert len(list(final.pieces(chess.QUEEN, chess.BLACK))) == 0


def test_positive_value_samples_before_capture_and_negative_for_never_achieved():
    rec, n = _capture_queen_game()
    rng = np.random.default_rng(0)
    samples = goal_value_samples(rec, rng, HERWeights(), deadline_max=60)

    # The capture-queen goal is the active goal on White plies (0,2,4,6,8). For
    # those plies, within the 8-ply deadline window, the queen IS captured -> the
    # search-laundered (active-goal) target must be positive (1.0).
    # The active (searched) goal on White plies is capture-queen with deadline 8.
    # Its sample is the search-laundered one: full goal equality (deadline 8) +
    # the search-laundered weight pins it exactly (negatives use window deadlines).
    active_goal = GoalTemplate.capture(chess.QUEEN, 8)
    active_q = {
        s.ply: s for s in samples
        if s.goal == active_goal and abs(s.weight - HERWeights().search_laundered) < 1e-9
    }
    assert active_q, "expected active-goal capture-queen samples on White plies"
    # The queen is captured at state index 9. For a White ply i with deadline 8,
    # the goal is achieved iff 9 - i <= 8, i.e. i >= 1. So plies 2,4,6,8 are
    # positive (within the deadline window) and ply 0 is just out of window.
    for ply, s in active_q.items():
        expected = 1.0 if (9 - ply) <= 8 else 0.0
        assert s.target == expected, f"ply {ply}: expected {expected}, got {s.target}"
    assert any(s.target == 1.0 for s in active_q.values()), "some plies must be positive"

    # A never-achieved delta (capture a KING is impossible) sampled as a negative
    # must be labeled 0.0. The candidate set has no king-capture; instead assert
    # that every negative-weighted sample whose goal was not achieved is 0.0.
    negs = [s for s in samples if abs(s.weight - HERWeights().negative) < 1e-9]
    # At least one negative exists, and any negative castle goal early is 0
    # (neither side castles in this line).
    castle_negs = [s for s in negs if s.goal.kind == "castle"]
    assert castle_negs, "expected castle negatives (never castled in this game)"
    assert all(s.target == 0.0 for s in castle_negs)


def test_her_positive_weighted_below_negative_and_laundered():
    w = HERWeights()
    assert w.her_positive < w.negative
    assert w.her_positive < w.search_laundered


def test_goal_buffer_sample_shapes_and_bce_training_uses_sigmoid():
    from chessrl.goals.encoding import GOAL_PLANES

    rec, _ = _capture_queen_game()
    buf = GoalReplayBuffer(capacity=10_000)
    buf.add_game(rec, np.random.default_rng(0))
    assert len(buf) > 0

    x, deadline, p, p_mask, v, vw = buf.sample(16, np.random.default_rng(1))
    assert x.shape == (16, 21 + GOAL_PLANES, 8, 8) and x.dtype == np.float32
    assert deadline.shape == (16,)
    assert p.shape == (16, NUM_ACTIONS)
    assert p_mask.shape == (16,)
    assert v.shape == (16,) and set(np.unique(v)).issubset({0.0, 1.0})
    assert vw.shape == (16,)
    # Value targets are achievement probabilities in [0,1] (sigmoid/BCE domain).
    assert v.min() >= 0.0 and v.max() <= 1.0
    # At least some rows carry a policy target (the active-goal rows).
    assert p_mask.sum() > 0
    # Rows with a policy target have a normalized distribution.
    for row in range(16):
        if p_mask[row] > 0:
            np.testing.assert_allclose(p[row].sum(), 1.0, atol=1e-5)


def test_wishful_thinking_thermometer_self_play_rate_and_gap():
    from chessrl.training.loop import (
        goal_achievement_rates,
        wishful_thinking_thermometer,
    )

    # Build a game where White's assigned capture-queen goal IS achieved from
    # ply 0: deadline 10 covers the capture at state index 9.
    board = chess.Board()
    builder = RecordBuilder()
    qgoal = GoalTemplate.capture(chess.QUEEN, deadline=10)
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5e5", "d2d4", "e5e4", "c3e4"]
    b = chess.Board()
    for uci in moves:
        mv = chess.Move.from_uci(uci)
        idx = move_to_index(mv, b.turn == chess.BLACK)
        builder.add(
            b, [idx], [4], idx,
            protagonist=b.turn,
            assigned_goal=qgoal if b.turn == chess.WHITE else WIN_GOAL,
            active_goal=qgoal if b.turn == chess.WHITE else WIN_GOAL,
        )
        b.push(mv)
    rec = builder.finalize(0)

    # White pursues capture-queen (achieved by deadline 10); Black pursues WIN
    # (not achieved, z=0). Self-play achievement rates per kind:
    rates = goal_achievement_rates([rec])
    assert "capture" in rates
    assert rates["capture"] == 1.0          # White achieved its assigned capture
    assert rates["win"] == 0.0              # Black's win goal not achieved (draw)

    # Thermometer with no eval data -> gap is None.
    therm = wishful_thinking_thermometer(rates)
    assert therm["capture"]["self_play"] == 1.0
    assert therm["capture"]["vs_stockfish"] is None
    assert therm["capture"]["gap"] is None

    # With held-out vs-Stockfish data, the gap (optimism) is populated.
    therm2 = wishful_thinking_thermometer(rates, {"capture": 0.4})
    assert abs(therm2["capture"]["gap"] - 0.6) < 1e-9


def test_trainer_goal_step_uses_bce_and_masked_ce(tmp_path):
    rec, _ = _capture_queen_game()
    buf = GoalReplayBuffer(capacity=10_000)
    buf.add_game(rec, np.random.default_rng(0))

    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=1, filters=16), goal_conditioned=True)
    trainer = Trainer(net, TrainingConfig(batch_size=8, device="cpu"), tmp_path)
    m = trainer.train_steps_goal(buf, 3, np.random.default_rng(0))
    assert trainer.step == 3
    assert np.isfinite(m["policy_loss"]) and np.isfinite(m["value_loss"])
    # The goal net's value head is a sigmoid in [0,1]; confirm a forward stays in
    # range (BCE is well-defined).
    x, deadline, _p, _pm, _v, _vw = buf.sample(4, np.random.default_rng(2))
    from chessrl.model.network import DEADLINE_SCALE
    with torch.no_grad():
        _, val = net(torch.from_numpy(x), torch.from_numpy(deadline / DEADLINE_SCALE))
    assert float(val.min()) >= 0.0 and float(val.max()) <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for autocast path")
def test_trainer_goal_step_runs_under_cuda_autocast(tmp_path):
    """Regression: the goal trainer runs under CUDA autocast in production, and
    F.binary_cross_entropy is autocast-UNSAFE -- it must be computed in fp32 with
    autocast disabled or the very first training step raises RuntimeError. CPU
    tests cannot catch this (autocast is CUDA-only)."""
    rec, _ = _capture_queen_game()
    buf = GoalReplayBuffer(capacity=10_000)
    buf.add_game(rec, np.random.default_rng(0))

    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=1, filters=16), goal_conditioned=True)
    trainer = Trainer(net, TrainingConfig(batch_size=8, device="cuda"), tmp_path)
    m = trainer.train_steps_goal(buf, 2, np.random.default_rng(0))  # must not raise
    assert trainer.step == 2
    assert np.isfinite(m["policy_loss"]) and np.isfinite(m["value_loss"])
