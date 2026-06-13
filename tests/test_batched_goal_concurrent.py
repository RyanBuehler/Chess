"""Batched concurrent goal self-play driver (Fix #3).

Two kinds of assertions:

* WELL-FORMED goal records: the concurrent driver emits records with goal
  columns, the switch-to-win actually happens (some ply runs under WIN after a
  short-deadline sub-goal resolves), and the win-floor is honored (always_win
  -> every ply under WIN). Mirrors the contracts in tests/test_selfplay_goals.py
  but exercises the BATCHED concurrent path.

* PER-GAME EQUIVALENCE with the sequential play_goal_game: for a single game at
  leaves_per_tree=1 under a MATCHED seed, the concurrent driver makes the SAME
  per-ply move decisions and writes the SAME records as the sequential
  reference. Both paths are driven by ONE deterministic evaluator keyed on the
  full goal-conditioned input (board+goal planes + deadline scalar) — the same
  approach as tests/test_batched_goal_equivalence.py — so any divergence is a
  driver bug, not evaluator noise. Because the driver replicates play_goal_game's
  RNG draw order (allow_resign, then per-side assign, then per-ply Dirichlet /
  temperature draws), a matched-seed single game is bit-comparable.
"""
import hashlib

import chess
import numpy as np

from chessrl.chess_env.encoding import encode_board, to_model_input
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.goals.assignment import make_assigner
from chessrl.goals.encoding import encode_goal
from chessrl.selfplay.concurrent import play_goal_games_concurrent
from chessrl.selfplay.play import play_goal_game
from chessrl.selfplay.records import WIN_KIND_CODE


# --------------------------------------------------------------------------
# One deterministic (policy, achievement-prob) keyed on the concatenated
# board+goal planes and the deadline scalar — identical core to the Task 3.1
# equivalence gate, exposed via BOTH the reference single-board ``evaluate`` and
# the batched ``evaluate_planes`` / ``evaluate_one_goal`` interfaces.
# --------------------------------------------------------------------------
def _pv_from_planes(planes: np.ndarray, deadline: float):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(planes, dtype=np.float32).tobytes())
    h.update(np.float32(deadline).tobytes())
    seed = int.from_bytes(h.digest()[:8], "little")
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(NUM_ACTIONS)
    policy = np.exp(logits - logits.max())
    policy = policy / policy.sum()
    p = float(1.0 / (1.0 + np.exp(-rng.standard_normal())))
    return policy.astype(np.float64), p


def _encode(board, goal, remaining, protagonist):
    board_planes = to_model_input(encode_board(board))
    goal_planes, _ = encode_goal(goal, remaining, protagonist)
    return np.concatenate([board_planes, goal_planes.astype(np.float32)], axis=0)


class _RefGoalEvaluator:
    def evaluate(self, board, goal, remaining, protagonist):
        return _pv_from_planes(_encode(board, goal, remaining, protagonist), remaining)


class _BatchedGoalEvaluator:
    def evaluate_planes(self, planes_batch, deadlines):
        policies, values = [], []
        for planes, d in zip(planes_batch, deadlines):
            policy, p = _pv_from_planes(planes, d)
            policies.append(policy)
            values.append(p)
        return np.asarray(policies), np.asarray(values, dtype=np.float64)

    def evaluate_one_goal(self, board, goal, remaining, protagonist):
        return _pv_from_planes(_encode(board, goal, remaining, protagonist), remaining)


def _records_equal(a, b) -> bool:
    """Structural equality of two GameRecords incl. the goal columns."""
    if len(a) != len(b):
        return False
    for f in ("policy_indices", "policy_counts", "policy_offsets", "outcomes", "played"):
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            return False
    if not np.array_equal(a.planes, b.planes):
        return False
    if a.has_goals() != b.has_goals():
        return False
    if a.has_goals():
        for f in ("protagonist", "assigned_kind", "active_kind", "assigned_blob", "active_blob"):
            if not np.array_equal(getattr(a, f), getattr(b, f)):
                return False
    return True


# --------------------------------------------------------------------------
# Well-formed records.
# --------------------------------------------------------------------------
def test_concurrent_goal_records_well_formed_and_switch_to_win():
    # Short-deadline sub-goals so the switch-to-win fires well within the game.
    cfg_goal = GoalConfig(goal_mode="random", win_floor=0.0, deadline_max=4)
    mcts_cfg = MCTSConfig(simulations=8, temperature_moves=4, leaves_per_tree=2)
    sp_cfg = SelfPlayConfig(ply_cap=40, resign_playout_fraction=0.0)
    rng = np.random.default_rng(7)
    assigner = make_assigner(cfg_goal, rng)

    results = play_goal_games_concurrent(
        _BatchedGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, rng,
        num_games=4, assigner=assigner,
    )
    assert len(results) == 4
    saw_switch = False
    for rec, board, z, meta in results:
        assert rec.has_goals(), "concurrent goal records must carry goal columns"
        assert len(rec) > 0
        assert z in (-1, 0, 1)
        assert meta["win_ply_fraction"] == rec.win_ply_fraction()
        # A short sub-goal that resolves switches active->WIN for the rest of the
        # game: at least one ply under a non-win assigned goal that later flips.
        non_win_assigned = (rec.assigned_kind != WIN_KIND_CODE)
        win_active = (rec.active_kind == WIN_KIND_CODE)
        if np.any(non_win_assigned) and np.any(win_active & non_win_assigned):
            saw_switch = True
    assert saw_switch, "no game exhibited the switch-to-win on sub-goal resolution"


def test_concurrent_always_win_win_floor_honored():
    cfg_goal = GoalConfig(goal_mode="always_win", win_floor=1.0)
    mcts_cfg = MCTSConfig(simulations=6, temperature_moves=2, leaves_per_tree=1)
    sp_cfg = SelfPlayConfig(ply_cap=16, resign_playout_fraction=0.0)
    rng = np.random.default_rng(1)
    assigner = make_assigner(cfg_goal, rng)
    results = play_goal_games_concurrent(
        _BatchedGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, rng,
        num_games=3, assigner=assigner,
    )
    for rec, _board, _z, _meta in results:
        assert rec.has_goals()
        # always-win: every ply is under the win-goal (win-floor honored).
        assert rec.win_ply_fraction() == 1.0


# --------------------------------------------------------------------------
# Per-game equivalence with the sequential reference (single game, matched seed).
# --------------------------------------------------------------------------
def test_concurrent_single_game_matches_sequential():
    cfg_goal = GoalConfig(goal_mode="random", win_floor=0.3, deadline_max=6)
    mcts_cfg = MCTSConfig(simulations=24, temperature_moves=6, leaves_per_tree=1)
    sp_cfg = SelfPlayConfig(ply_cap=30, resign_playout_fraction=0.0)

    # Sequential reference: one game, seed S.
    seq_rng = np.random.default_rng(2024)
    seq_assigner = make_assigner(cfg_goal, seq_rng)
    seq_rec, seq_board, seq_z = play_goal_game(
        _RefGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, seq_rng, seq_assigner
    )

    # Concurrent driver: a single game, SAME seed.
    con_rng = np.random.default_rng(2024)
    con_assigner = make_assigner(cfg_goal, con_rng)
    con_results = play_goal_games_concurrent(
        _BatchedGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, con_rng,
        num_games=1, assigner=con_assigner,
    )
    assert len(con_results) == 1
    con_rec, con_board, con_z, _meta = con_results[0]

    assert con_z == seq_z, f"result mismatch {con_z} != {seq_z}"
    assert con_board.fen() == seq_board.fen(), "final board mismatch"
    assert _records_equal(con_rec, seq_rec), "record (incl. goal columns) mismatch"


def test_concurrent_single_game_matches_sequential_always_win():
    cfg_goal = GoalConfig(goal_mode="always_win", win_floor=1.0)
    mcts_cfg = MCTSConfig(simulations=16, temperature_moves=4, leaves_per_tree=1)
    sp_cfg = SelfPlayConfig(ply_cap=24, resign_playout_fraction=0.0)

    seq_rng = np.random.default_rng(99)
    seq_assigner = make_assigner(cfg_goal, seq_rng)
    seq_rec, seq_board, seq_z = play_goal_game(
        _RefGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, seq_rng, seq_assigner
    )

    con_rng = np.random.default_rng(99)
    con_assigner = make_assigner(cfg_goal, con_rng)
    con_rec, con_board, con_z, _ = play_goal_games_concurrent(
        _BatchedGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, con_rng,
        num_games=1, assigner=con_assigner,
    )[0]

    assert con_z == seq_z
    assert con_board.fen() == seq_board.fen()
    assert _records_equal(con_rec, seq_rec)


# --------------------------------------------------------------------------
# Live feed: the goal concurrent driver publishes frames (the gap this fixes).
# --------------------------------------------------------------------------
class _CapturingPublisher:
    """Stub publisher that records every published (topic, payload) frame."""

    def __init__(self):
        self.frames = []

    def publish(self, game_id, payload):
        self.frames.append((game_id, payload))

    def close(self):
        return


def test_concurrent_goal_emits_live_feed_frames():
    cfg_goal = GoalConfig(goal_mode="random", win_floor=0.0, deadline_max=4)
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=2, leaves_per_tree=1)
    sp_cfg = SelfPlayConfig(ply_cap=12, resign_playout_fraction=0.0)
    rng = np.random.default_rng(123)
    assigner = make_assigner(cfg_goal, rng)

    pub = _CapturingPublisher()
    results = play_goal_games_concurrent(
        _BatchedGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, rng,
        num_games=2, assigner=assigner,
        publisher=pub, game_id_prefix="wtest_b0_",
    )
    assert len(results) == 2

    # The driver published at least one frame per game.
    assert pub.frames, "goal concurrent driver emitted no live-feed frames"
    topics = {topic for topic, _ in pub.frames}
    assert topics == {"wtest_b0_0", "wtest_b0_1"}, topics

    # Every frame carries the live-feed schema fields.
    for topic, payload in pub.frames:
        assert payload["game_id"] == topic
        assert isinstance(payload["fen"], str) and payload["fen"]
        assert isinstance(payload["last_move_uci"], str) and payload["last_move_uci"]
        assert "ply" in payload and "root_q" in payload and "top_moves" in payload

    # Each game produces a terminal done=True frame as its last frame.
    for slot in ("wtest_b0_0", "wtest_b0_1"):
        game_frames = [p for t, p in pub.frames if t == slot]
        assert game_frames, f"no frames for {slot}"
        assert game_frames[-1]["done"] is True, f"no terminal frame for {slot}"
        assert game_frames[-1]["z"] in (-1, 0, 1)


def test_publisher_does_not_perturb_results():
    """Publishing is a pure side effect: results are identical with/without a
    publisher under a matched seed (RNG and decisions untouched)."""
    cfg_goal = GoalConfig(goal_mode="random", win_floor=0.3, deadline_max=6)
    mcts_cfg = MCTSConfig(simulations=16, temperature_moves=4, leaves_per_tree=1)
    sp_cfg = SelfPlayConfig(ply_cap=24, resign_playout_fraction=0.0)

    rng_a = np.random.default_rng(555)
    res_a = play_goal_games_concurrent(
        _BatchedGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, rng_a,
        num_games=2, assigner=make_assigner(cfg_goal, rng_a),
    )
    rng_b = np.random.default_rng(555)
    res_b = play_goal_games_concurrent(
        _BatchedGoalEvaluator(), mcts_cfg, sp_cfg, cfg_goal, rng_b,
        num_games=2, assigner=make_assigner(cfg_goal, rng_b),
        publisher=_CapturingPublisher(), game_id_prefix="x_",
    )
    for (ra, ba, za, _), (rb, bb, zb, _) in zip(res_a, res_b):
        assert za == zb
        assert ba.fen() == bb.fen()
        assert _records_equal(ra, rb)
