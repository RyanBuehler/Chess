"""Deadline-scalar consistency between training (HER) and inference (search).

The deadline scalar fed to the goal net's value-head FC must be BOUNDED to
[0,1] and computed CONSISTENTLY at train and inference time for a given
(goal, state). The win-goal (deadline = 512 = ply_cap) previously fed ~8.5 at
the search/inference path but ~1.0 at the HER train path -- a train/inference
distribution mismatch on V(.|win), the only value eval depends on.

Sub-goals (deadline <= deadline_max) are unaffected and must stay in range.
"""
import numpy as np

from chessrl.model.network import DEADLINE_SCALE, _deadline_tensor


# Mirror of the HER deadline-scalar computation (chessrl/training/her.py): the
# train-time scalar is `min(active.deadline, deadline_max) / DEADLINE_SCALE`
# where deadline_max == DEADLINE_SCALE (60).
DEADLINE_MAX = 60


def _her_scaled_scalar(deadline: int, deadline_max: int = DEADLINE_MAX) -> float:
    return min(deadline, deadline_max) / DEADLINE_SCALE


def _search_scaled_scalar(remaining: int) -> float:
    """The scalar the inference/search path actually feeds the FC, via the
    single canonical helper that forms the network input."""
    return float(_deadline_tensor(remaining, device="cpu").item())


def test_win_goal_scalar_consistent_and_bounded():
    # WIN_GOAL deadline at the search root: remaining == 512.
    remaining = 512
    search = _search_scaled_scalar(remaining)
    her = _her_scaled_scalar(remaining)
    assert 0.0 <= search <= 1.0, f"search scalar out of range: {search}"
    assert abs(search - her) < 1e-6, (
        f"train/inference mismatch for win-goal: search={search} her={her}"
    )


def test_subgoal_scalars_unchanged_and_in_range():
    # Sub-goals have deadline <= deadline_max, so clamping is a no-op for them:
    # the scaled scalar equals remaining / DEADLINE_SCALE and stays in [0,1].
    for remaining in (1, 5, 15, 30, 60):
        search = _search_scaled_scalar(remaining)
        her = _her_scaled_scalar(remaining)
        assert 0.0 <= search <= 1.0, f"out of range for remaining={remaining}: {search}"
        assert abs(search - her) < 1e-6, (
            f"mismatch for sub-goal remaining={remaining}: search={search} her={her}"
        )
        assert abs(search - remaining / DEADLINE_SCALE) < 1e-6


def test_batched_goal_path_matches_canonical_clamp():
    """The batched goal evaluator's deadline scaling must produce the SAME
    bounded scalar as the canonical _deadline_tensor helper."""
    from chessrl.model.network import BatchedGoalNetEvaluator, PolicyValueNet
    from chessrl.config.config import NetworkConfig
    import torch

    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=1, filters=8), goal_conditioned=True)
    ev = BatchedGoalNetEvaluator(net, device="cpu")

    # Reach into the batched path's deadline-scaling by reproducing it through a
    # tiny helper if exposed, else assert via the canonical helper equivalence.
    for remaining in (512, 60, 30, 1):
        canonical = float(_deadline_tensor(remaining, device="cpu").item())
        assert 0.0 <= canonical <= 1.0
        # The batched evaluator must clamp identically.
        scaled = ev._scale_deadlines(np.asarray([remaining], dtype=np.float32))
        assert abs(float(scaled[0]) - canonical) < 1e-6, (
            f"batched scalar {float(scaled[0])} != canonical {canonical} "
            f"for remaining={remaining}"
        )
