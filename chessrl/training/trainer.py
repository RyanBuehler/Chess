"""SGD loop with pacing budget, AMP, and checkpointing."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessrl.config.config import TrainingConfig
from chessrl.training.buffer import ReplayBuffer


class Trainer:
    def __init__(self, net: nn.Module, cfg: TrainingConfig, run_dir: str | Path):
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.net = net.to(self.device)
        self.opt = torch.optim.Adam(
            net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        self.step = 0
        self.run_dir = Path(run_dir)

    def allowed_steps(self, total_positions: int) -> int:
        """Pacing: total step budget so far minus steps already taken."""
        budget = int(total_positions * self.cfg.samples_per_position / self.cfg.batch_size)
        return max(0, budget - self.step)

    def train_steps(self, buffer: ReplayBuffer, n: int, rng: np.random.Generator) -> dict:
        self.net.train()
        lp_sum = lv_sum = 0.0
        for _ in range(n):
            x, p, v = buffer.sample(self.cfg.batch_size, rng)
            xt = torch.from_numpy(x).to(self.device)
            pt = torch.from_numpy(p).to(self.device)
            vt = torch.from_numpy(v).to(self.device)
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                logits, value = self.net(xt)
                loss_p = -(pt * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss_v = F.mse_loss(value.squeeze(1), vt)
                loss = loss_p + loss_v
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.step += 1
            lp_sum += loss_p.detach().item()
            lv_sum += loss_v.detach().item()
        n = max(n, 1)
        return {"policy_loss": lp_sum / n, "value_loss": lv_sum / n, "step": self.step}

    def save_checkpoint(self) -> Path:
        d = self.run_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"ckpt_{self.step:08d}.pt"
        torch.save(
            {"step": self.step, "model": self.net.state_dict(), "optimizer": self.opt.state_dict()},
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.step = ck["step"]
