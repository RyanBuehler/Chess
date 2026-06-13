"""Shared provenance helper used by both loop.py and parallel_loop.py.

Every curve must be traceable to exactly the code and toolchain that produced
it. The network sub-dict captures architecture and parameter count so that
archived runs are self-describing even if the default config changes.
"""
from __future__ import annotations

import subprocess

import torch

from chessrl.config.config import NetworkConfig, RunConfig
from chessrl.model.network import PolicyValueNet


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or None
        )
    except OSError:
        return None


def _network_info(net_cfg: NetworkConfig) -> dict:
    """Build the network provenance sub-dict.

    Instantiates a throw-away PolicyValueNet on CPU solely to count parameters,
    then discards it.  Never touches CUDA so it's safe in any context.
    """
    net = PolicyValueNet(net_cfg)
    params = sum(p.numel() for p in net.parameters())
    del net
    blocks = net_cfg.blocks
    filters = net_cfg.filters
    return {
        "blocks": blocks,
        "filters": filters,
        "params": params,
        "archetype": f"resnet-{blocks}x{filters}",
    }


def build_provenance(cfg: RunConfig) -> dict:
    """Return the full provenance dict for a new run."""
    return {
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "network": _network_info(cfg.network),
        "goal": _goal_info(cfg.goal),
    }


def _goal_info(goal_cfg) -> dict:
    """Build the goal provenance sub-dict so goal arms are self-describing."""
    return {
        "goal_mode": goal_cfg.goal_mode,
        "win_floor": goal_cfg.win_floor,
        "lp_window": goal_cfg.lp_window,
        "novelty_beta": goal_cfg.novelty_beta,
        "min_attempts_for_lp": goal_cfg.min_attempts_for_lp,
        "deadline_max": goal_cfg.deadline_max,
    }
