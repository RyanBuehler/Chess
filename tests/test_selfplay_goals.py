"""Goal-conditioned self-play: assignment, pure pursuit, switch-to-win,
win-floor, and the goal record fields (plan Task 3.2)."""
import chess
import numpy as np

from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.goals.assignment import GoalAssigner, make_assigner
from chessrl.goals.templates import WIN_GOAL, GoalTemplate
from chessrl.selfplay.play import play_goal_game
from chessrl.selfplay.records import WIN_KIND_CODE, deserialize_goal, serialize_goal


class UniformGoalEvaluator:
    """Stubbed goal-conditioned evaluator: uniform priors, constant achievement
    probability. Matches the GoalNetEvaluator.evaluate signature."""

    def __init__(self, value: float = 0.5):
        self.value = value

    def evaluate(self, board, goal, remaining, protagonist):
        return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float64), self.value


class _OneShotAssigner(GoalAssigner):
    """Hands out a fixed sub-goal to every side (a 1-ply capture-queen goal that
    cannot be satisfied at the start, so it resolves by deadline expiry)."""

    def __init__(self, subgoal: GoalTemplate):
        self._subgoal = subgoal

    def assign(self) -> GoalTemplate:
        return self._subgoal


def test_active_goal_switches_to_win_after_resolution():
    # A 1-ply capture-queen goal cannot be achieved from the opening, so each
    # side's active goal must switch to WIN after its deadline (1 ply) elapses.
    subgoal = GoalTemplate.capture(chess.QUEEN, deadline=1)
    assigner = _OneShotAssigner(subgoal)
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=0)
    sp_cfg = SelfPlayConfig(ply_cap=12, resign_playout_fraction=1.0)
    goal_cfg = GoalConfig(goal_mode="random")

    rec, board, z = play_goal_game(
        UniformGoalEvaluator(), mcts_cfg, sp_cfg, goal_cfg,
        np.random.default_rng(0), assigner,
    )

    assert rec.has_goals()
    # Early plies pursue the assigned sub-goal; later plies must be under WIN.
    assigned = [deserialize_goal(b) for b in rec.assigned_blob]
    active = [deserialize_goal(b) for b in rec.active_blob]
    assert all(g.key() == subgoal.key() for g in assigned)   # assigned never changes
    # The first ply for each side is still the sub-goal (not yet resolved).
    assert active[0].key() == subgoal.key()
    # By the end of the game every side has switched to WIN.
    assert active[-1].is_win()
    # The switch actually happened at least once.
    assert any(g.is_win() for g in active)
    # And there was a real game played to a result/cap.
    assert z in (-1, 0, 1)
    assert board.move_stack


def test_pure_pursuit_records_active_and_assigned_and_visits():
    subgoal = GoalTemplate.capture(chess.QUEEN, deadline=2)
    assigner = _OneShotAssigner(subgoal)
    mcts_cfg = MCTSConfig(simulations=6, temperature_moves=0)
    sp_cfg = SelfPlayConfig(ply_cap=8, resign_playout_fraction=1.0)
    goal_cfg = GoalConfig(goal_mode="random")

    rec, _board, _z = play_goal_game(
        UniformGoalEvaluator(), mcts_cfg, sp_cfg, goal_cfg,
        np.random.default_rng(1), assigner,
    )

    assert rec.has_goals()
    T = len(rec)
    # Per-move columns are present and aligned.
    assert rec.assigned_blob.shape == (T,)
    assert rec.active_blob.shape == (T,)
    assert rec.protagonist.shape == (T,)
    # Visit counts are stored (sparse policy targets) and non-empty per ply.
    assert rec.policy_offsets[-1] == len(rec.policy_indices)
    assert len(rec.policy_counts) == len(rec.policy_indices)
    for t in range(T):
        a, b = rec.policy_offsets[t], rec.policy_offsets[t + 1]
        assert b > a                              # at least one move's visits
        assert rec.policy_counts[a:b].sum() > 0
    # Protagonist matches the side to move at each ply (White at even, Black odd).
    for t in range(T):
        assert rec.protagonist[t] == (1 if t % 2 == 0 else 0)


def test_win_floor_fraction_over_many_assignments():
    cfg = GoalConfig(goal_mode="random", win_floor=0.3)
    assigner = make_assigner(cfg, np.random.default_rng(123))
    n = 5000
    wins = sum(1 for _ in range(n) if assigner.assign().is_win())
    frac = wins / n
    # At least win_floor are win-goals (with margin for sampling noise).
    assert frac >= cfg.win_floor - 0.02


def test_always_win_assigns_only_win():
    cfg = GoalConfig(goal_mode="always_win")
    assigner = make_assigner(cfg, np.random.default_rng(0))
    assert all(assigner.assign().is_win() for _ in range(50))


def test_lp_mode_falls_back_to_random():
    cfg = GoalConfig(goal_mode="lp", win_floor=0.0)
    assigner = make_assigner(cfg, np.random.default_rng(0))
    goals = [assigner.assign() for _ in range(200)]
    # With win_floor=0, lp should still produce diverse sub-goals (random source).
    kinds = {g.kind for g in goals}
    assert len(kinds) > 1


def test_none_mode_has_no_assigner():
    assert make_assigner(GoalConfig(goal_mode="none")) is None


def test_serialize_roundtrip():
    for g in (WIN_GOAL, GoalTemplate.capture(chess.KNIGHT, 15),
              GoalTemplate.reach_rank(chess.PAWN, 7, 30), GoalTemplate.check(5)):
        assert deserialize_goal(serialize_goal(g)) == g


def test_win_ply_fraction_metric():
    # always_win: every ply is under WIN, so the fraction is 1.0.
    assigner = make_assigner(GoalConfig(goal_mode="always_win"), np.random.default_rng(0))
    rec, _b, _z = play_goal_game(
        UniformGoalEvaluator(), MCTSConfig(simulations=4, temperature_moves=0),
        SelfPlayConfig(ply_cap=6, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="always_win"), np.random.default_rng(2), assigner,
    )
    assert rec.win_ply_fraction() == 1.0
    assert all(k == WIN_KIND_CODE for k in rec.active_kind)
