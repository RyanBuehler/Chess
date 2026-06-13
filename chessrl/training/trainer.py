"""SGD loop with pacing budget, AMP, and checkpointing."""
import os
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

    def train_steps_goal(self, buffer, n: int, rng: np.random.Generator) -> dict:
        """Goal-conditioned SGD step (spec sec 8, 11; plan Task 3.3).

        Value loss is **BCE** on the sigmoid achievement head, per-sample weighted
        (search-laundered/negatives up, raw HER positives down). Policy loss is
        cross-entropy on the **active-goal** visit-count targets only, masked to
        the rows that carry a policy target (HER future/negative rows do not).
        The deadline scalar is scaled to match the evaluators' DEADLINE_SCALE.
        """
        from chessrl.model.network import DEADLINE_SCALE

        self.net.train()
        lp_sum = lv_sum = 0.0
        for _ in range(n):
            x, deadline, p, p_mask, v, vw = buffer.sample(self.cfg.batch_size, rng)
            xt = torch.from_numpy(x).to(self.device)
            dt = torch.from_numpy(deadline / DEADLINE_SCALE).to(self.device)
            pt = torch.from_numpy(p).to(self.device)
            pmask = torch.from_numpy(p_mask).to(self.device)
            vt = torch.from_numpy(v).to(self.device)
            vwt = torch.from_numpy(vw).to(self.device)
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                logits, value = self.net(xt, dt)
                # Masked CE policy loss on active-goal rows only.
                ce = -(pt * F.log_softmax(logits, dim=1)).sum(dim=1)
                denom = pmask.sum().clamp_min(1.0)
                loss_p = (ce * pmask).sum() / denom
                # Weighted BCE value loss (sigmoid achievement probability).
                val = value.squeeze(1).clamp(1e-6, 1.0 - 1e-6)
                bce = F.binary_cross_entropy(val, vt, reduction="none")
                loss_v = (bce * vwt).sum() / vwt.sum().clamp_min(1e-6)
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
        # torch.save is not atomic; write to a temp name and os.replace so
        # concurrent readers (workers, evaluator) never see a half-written file.
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {"step": self.step, "model": self.net.state_dict(), "optimizer": self.opt.state_dict()},
            tmp,
        )
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.step = ck["step"]
