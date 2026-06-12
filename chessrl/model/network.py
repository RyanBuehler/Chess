"""Policy-value ResNet (AlphaZero-style, configurable size)."""
import chess
import numpy as np
import torch
import torch.nn as nn

from chessrl.chess_env.encoding import NUM_PLANES, encode_board, to_model_input
from chessrl.config.config import NetworkConfig


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
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        ch = cfg.filters
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(ch) for _ in range(cfg.blocks)])
        self.policy_conv = nn.Conv2d(ch, 73, 1)  # AZ-style conv head, NOT flatten->FC
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

    def forward(self, x):
        h = self.tower(self.stem(x))
        # (B,73,rank,file) -> (B,rank,file,73) -> flat, so index = square*73 + type
        logits = self.policy_conv(h).permute(0, 2, 3, 1).flatten(1)
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
