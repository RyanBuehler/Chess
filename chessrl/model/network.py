"""Policy-value ResNet (AlphaZero-style, configurable size).

Two variants share one class, selected by the ``goal_conditioned`` flag:

* **vanilla** (default): 21 input planes, ``tanh`` value head in [-1, 1] (the
  side-to-move zero-sum value), single-argument forward. Used by the vanilla arm
  and all pre-existing code. UNCHANGED.
* **goal-conditioned**: ``21 + GOAL_PLANES`` input planes (board + goal
  conditioning), and a value head that ends in ``sigmoid`` (achievement
  probability in [0, 1]) and takes the **deadline scalar** concatenated at its
  first Linear layer. ``forward(x, deadline)`` takes a second argument. Used by
  the goal arms (always-win / random-goal / lp-goal) per spec sec 8, sec 9.

A spatially-constant deadline plane is poorly legible to a conv tower; the
scalar fed at the FC is (spec sec 5 review finding) — hence the side-input.
"""
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

from chessrl.chess_env.encoding import NUM_PLANES, encode_board, to_model_input
from chessrl.config.config import NetworkConfig
from chessrl.goals.encoding import GOAL_PLANES, encode_goal


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetworkConfig, goal_conditioned: bool = False):
        super().__init__()
        self.goal_conditioned = goal_conditioned
        ch = cfg.filters
        in_planes = NUM_PLANES + GOAL_PLANES if goal_conditioned else NUM_PLANES
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(ch) for _ in range(cfg.blocks)])
        self.policy_conv = nn.Conv2d(ch, 73, 1)  # AZ-style conv head, NOT flatten->FC
        if goal_conditioned:
            # Value head reduces the conv features, then concatenates the
            # deadline scalar at the first Linear, and ends in sigmoid (an
            # achievement probability in [0,1], spec sec 8). Split into a conv
            # body + an FC head so the scalar can be cat'd between them.
            self.value_body = nn.Sequential(
                nn.Conv2d(ch, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.value_fc = nn.Sequential(
                nn.Linear(8 * 64 + 1, 64),   # +1 for the deadline side-input
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Conv2d(ch, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * 64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

    def forward(self, x, deadline=None):
        h = self.tower(self.stem(x))
        # (B,73,rank,file) -> (B,rank,file,73) -> flat, so index = square*73 + type
        logits = self.policy_conv(h).permute(0, 2, 3, 1).flatten(1)
        if self.goal_conditioned:
            if deadline is None:
                raise ValueError("goal-conditioned net requires a deadline scalar")
            feat = self.value_body(h)
            if deadline.dim() == 1:
                deadline = deadline.unsqueeze(1)
            value = self.value_fc(torch.cat([feat, deadline.to(feat.dtype)], dim=1))
            return logits, value
        return logits, self.value_head(h)


class NetEvaluator:
    """Single-position evaluator used by the reference MCTS."""

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        self.net = net.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        self.net.eval()
        x = torch.from_numpy(to_model_input(encode_board(board))).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        policy = torch.softmax(logits[0], dim=0).cpu().numpy()
        return policy, float(value.item())


# Deadline scalar normalization. The encoder returns `remaining` (plies to the
# deadline) raw; we feed it to the FC scaled into a small range so it is on the
# same order as the conv features. Calibration/monotonicity (Task 2.3) operates
# on the post-net probability, not this raw scale.
DEADLINE_SCALE = 60.0


def _deadline_tensor(remaining, device, scale: float = DEADLINE_SCALE):
    return torch.tensor([[float(remaining) / scale]], dtype=torch.float32, device=device)


class GoalNetEvaluator:
    """Single-position evaluator for the goal-conditioned net (spec sec 8/9).

    Returns (policy, P(protagonist achieves goal by deadline)) where the value is
    a sigmoid achievement probability in [0,1]. Mirrors NetEvaluator's interface
    but takes the goal/remaining/protagonist conditioning."""

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        assert net.goal_conditioned, "GoalNetEvaluator requires a goal-conditioned net"
        self.net = net.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, board: chess.Board, goal, remaining: int, protagonist: bool):
        self.net.eval()
        board_planes = to_model_input(encode_board(board))
        goal_planes, _ = encode_goal(goal, remaining, protagonist)
        x = np.concatenate([board_planes, goal_planes.astype(np.float32)], axis=0)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        deadline = _deadline_tensor(remaining, self.device)
        logits, value = self.net(x, deadline)
        policy = torch.softmax(logits[0], dim=0).cpu().numpy()
        return policy, float(value.item())


def deadline_value_sweep(evaluator, board, goal, protagonist, remainings):
    """Calibration/monotonicity hook (spec sec 8/9, Task 2.3).

    Hold (state, goal, protagonist) fixed and sweep ``remainings`` (a sequence of
    moves-remaining-to-deadline). Returns a list of achievement probabilities,
    one per ``remainings`` entry, that is **monotone non-decreasing in
    remaining** by construction and is exactly 0.0 whenever ``remaining <= 0``
    and the goal is not already achieved.

    Monotone-by-construction: more time to a deadline cannot DECREASE the
    probability of achieving it (a longer horizon is a superset of the shorter
    one). We therefore take a running maximum over ``remainings`` sorted
    ascending, then map back. This is the post-processing the spec calls for so
    that lower ``remaining`` can never increase V; it also gates training (any
    raw-net non-monotonicity is clamped here and surfaced by the gap).

    ``remaining <= 0`` with the goal unachieved is pinned to 0 (deadline
    expired). If the goal already holds at ``board`` (an "achieved" terminal),
    every entry is 1.0.
    """
    from chessrl.mcts.reference import _goal_achieved
    from chessrl.goals.features import board_features

    baseline = board_features(board)
    if _goal_achieved(board, goal, protagonist, baseline):
        return [1.0 for _ in remainings]

    # Raw net values per requested remaining.
    raw = {}
    for r in remainings:
        if r <= 0:
            raw[r] = 0.0
        else:
            _, v = evaluator.evaluate(board, goal, r, protagonist)
            raw[r] = float(v)

    # Enforce monotonicity in `remaining`: sort ascending, running max.
    order = sorted(set(remainings))
    running = 0.0
    mono = {}
    for r in order:
        running = max(running, raw[r])
        mono[r] = 0.0 if r <= 0 else running
    return [mono[r] for r in remainings]


class BatchedNetEvaluator:
    """Batched evaluator: one net, one batched forward per call. Owns its net
    and calls .eval() once at construction so it never shares a live training
    module with a Trainer (the documented train/eval seam). Used by batched MCTS
    and the self-play workers."""

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        self.device = device
        self.net = net.to(device)
        self.net.eval()

    @classmethod
    def from_checkpoint(
        cls, path, network_cfg: NetworkConfig, device: str = "cpu"
    ) -> "BatchedNetEvaluator":
        net = PolicyValueNet(network_cfg)
        ckpt = torch.load(Path(path), map_location=device)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    @torch.no_grad()
    def evaluate_planes(self, planes_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """planes_batch: pre-encoded float32 array of shape (N, NUM_PLANES, 8, 8).
        Returns (policies (N, NUM_ACTIONS) softmaxed float32, values (N,) float32).
        Empty input (N==0) -> empty arrays."""
        n = planes_batch.shape[0]
        if n == 0:
            return (
                np.zeros((0, self.net.policy_conv.out_channels * 64), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        x = torch.from_numpy(planes_batch).to(self.device)
        logits, value = self.net(x)
        policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        values = value.squeeze(1).cpu().numpy().astype(np.float32)
        return policies, values

    @torch.no_grad()
    def evaluate_many(self, boards: list) -> tuple[np.ndarray, np.ndarray]:
        """boards: list[chess.Board]. Returns (policies (N,4672) softmaxed
        float32, values (N,) float32). Empty input -> empty arrays."""
        if not boards:
            return (
                np.zeros((0, self.net.policy_conv.out_channels * 64), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        planes_batch = np.stack([to_model_input(encode_board(b)) for b in boards])
        return self.evaluate_planes(planes_batch)


class BatchedGoalNetEvaluator:
    """Batched evaluator for the goal-conditioned net (spec sec 8/9, Task 3.1).

    Used by the goal-conditioned batched MCTS. ``evaluate_planes`` takes the
    pre-encoded ``(N, NUM_PLANES + GOAL_PLANES, 8, 8)`` planes AND a per-leaf
    ``deadlines`` vector (the moves-remaining-to-deadline scalars), and returns
    softmaxed policies plus sigmoid achievement probabilities P(protagonist
    achieves goal) in [0,1]. ``evaluate_one_goal`` is the single-leaf path used
    by init_tree / advance (not part of the batch)."""

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        assert net.goal_conditioned, "BatchedGoalNetEvaluator requires a goal-conditioned net"
        self.device = device
        self.net = net.to(device)
        self.net.eval()

    @classmethod
    def from_checkpoint(
        cls, path, network_cfg: NetworkConfig, device: str = "cpu"
    ) -> "BatchedGoalNetEvaluator":
        net = PolicyValueNet(network_cfg, goal_conditioned=True)
        ckpt = torch.load(Path(path), map_location=device)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    @torch.no_grad()
    def evaluate_planes(
        self, planes_batch: np.ndarray, deadlines: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """planes_batch: float32 (N, NUM_PLANES + GOAL_PLANES, 8, 8). deadlines:
        the per-leaf moves-remaining scalars (N,). Returns (policies
        (N, NUM_ACTIONS) softmaxed float32, values (N,) float32 in [0,1]).
        Empty input -> empty arrays."""
        n = planes_batch.shape[0]
        if n == 0:
            return (
                np.zeros((0, self.net.policy_conv.out_channels * 64), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        x = torch.from_numpy(planes_batch).to(self.device)
        deadline = torch.tensor(
            np.asarray(deadlines, dtype=np.float32).reshape(-1, 1) / DEADLINE_SCALE,
            dtype=torch.float32,
            device=self.device,
        )
        logits, value = self.net(x, deadline)
        policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        values = value.squeeze(1).cpu().numpy().astype(np.float32)
        return policies, values

    @torch.no_grad()
    def evaluate_one_goal(self, board: chess.Board, goal, remaining: int, protagonist: bool):
        """Single-leaf goal-conditioned evaluation (init_tree / advance path).
        Returns (policy (NUM_ACTIONS,) softmaxed, P(achieve) in [0,1])."""
        board_planes = to_model_input(encode_board(board))
        goal_planes, _ = encode_goal(goal, remaining, protagonist)
        planes = np.concatenate([board_planes, goal_planes.astype(np.float32)], axis=0)[None]
        policies, values = self.evaluate_planes(planes, np.asarray([remaining], dtype=np.float32))
        return policies[0], float(values[0])
