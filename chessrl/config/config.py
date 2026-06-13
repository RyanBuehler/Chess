"""Experiment configuration. A run is fully described by one RunConfig."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NetworkConfig:
    blocks: int = 6
    filters: int = 64


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.3
    temperature_moves: int = 30   # sample proportionally to visits for this many plies, then argmax
    leaves_per_tree: int = 1      # M5: K leaves selected per tree per batching round (virtual loss). K=1 == reference.


@dataclass(frozen=True)
class SelfPlayConfig:
    ply_cap: int = 512
    resign_threshold: float = -0.95
    resign_consecutive: int = 2          # consecutive own moves below threshold before resigning
    resign_playout_fraction: float = 0.1 # fraction of games where resignation is disabled (false-positive measurement)
    games_per_iteration: int = 10
    workers: int = 4                     # M5: number of self-play worker processes
    concurrent_games: int = 32           # M5: concurrent game trees per worker batch
    feed_port: int = 0                   # M7: live-feed base PUB port (0 = disabled, no zmq). Worker w binds feed_port + w.
    worker_heartbeat_seconds: float = 0.0  # Task 5.2: hung-worker detection, OFF by default. Restarts a worker alive-but-producing-no-new-game within this window. WARNING: must exceed the time to complete one self-play batch -- from-scratch games run concurrent_games at a time and take many minutes, so a too-small window kills healthy workers mid-batch and craters throughput. 0 disables (recommended); dead-worker restart stays active regardless.


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    buffer_size: int = 500_000
    samples_per_position: float = 2.0    # pacing: total SGD samples allowed per generated position
    device: str = "cuda"                 # trainer device; falls back to cpu if cuda unavailable
    seed: int = 0
    checkpoint_every_steps: int = 1000   # M5: trainer saves a checkpoint each time this many steps are crossed
    selfplay_device: str = "cuda"        # M5: device workers use; falls back to cpu in worker if unavailable


@dataclass(frozen=True)
class EvalConfig:
    every_n_checkpoints: int = 5         # evaluate every Nth checkpoint per run
    games_per_rung: int = 4              # games vs each ladder rung; MUST be even (both colors equally)
    agent_simulations: int = 200         # MCTS sims/move for the agent player in evaluation
    max_plies: int = 256                 # ply cap; a capped game is adjudicated a draw
    stockfish_path: str = ""             # "" disables all Stockfish rungs (floors-only ladder)
    stockfish_movetime_ms: int = 100     # per-move think time for movetime-limited Stockfish rungs
    poll_seconds: float = 10.0           # daemon poll interval for new checkpoints / inbox


@dataclass(frozen=True)
class GoalConfig:
    goal_mode: str = "none"          # none | always_win | random | lp
    win_floor: float = 0.2           # min fraction of games assigned g=win
    lp_window: int = 200             # attempts in the LP window
    novelty_beta: float = 1.0        # weight of the novelty bonus
    min_attempts_for_lp: int = 20    # gate LP on attempt count
    deadline_max: int = 60           # cap on goal deadline horizon (plies)

    def __post_init__(self):
        if self.goal_mode not in ("none", "always_win", "random", "lp"):
            raise ValueError(f"bad goal_mode {self.goal_mode}")


@dataclass(frozen=True)
class RunConfig:
    run_name: str = "default"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> "RunConfig":
        def build(klass, key):
            return klass(**(raw.get(key) or {}))
        return cls(
            run_name=raw.get("run_name", "default"),
            network=build(NetworkConfig, "network"),
            mcts=build(MCTSConfig, "mcts"),
            selfplay=build(SelfPlayConfig, "selfplay"),
            training=build(TrainingConfig, "training"),
            eval=build(EvalConfig, "eval"),
            goal=build(GoalConfig, "goal"),
        )

    @classmethod
    def from_yaml(cls, path) -> "RunConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_json(cls, path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
