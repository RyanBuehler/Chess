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


@dataclass(frozen=True)
class SelfPlayConfig:
    ply_cap: int = 512
    resign_threshold: float = -0.95
    resign_consecutive: int = 2          # consecutive own moves below threshold before resigning
    resign_playout_fraction: float = 0.1 # fraction of games where resignation is disabled (false-positive measurement)
    games_per_iteration: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    buffer_size: int = 500_000
    samples_per_position: float = 2.0    # pacing: total SGD samples allowed per generated position
    device: str = "cuda"                 # falls back to cpu if cuda unavailable
    seed: int = 0


@dataclass(frozen=True)
class RunConfig:
    run_name: str = "default"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

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
        )

    @classmethod
    def from_yaml(cls, path) -> "RunConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_json(cls, path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
