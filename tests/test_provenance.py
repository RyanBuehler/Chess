"""Unit tests for the shared provenance helper and the /api/runs/{id}/provenance
REST endpoint (Task 1)."""
import json

import pytest
from fastapi.testclient import TestClient

from chessrl.config.config import NetworkConfig, RunConfig
from chessrl.training.provenance import build_provenance
from server.app import create_app


# ---------------------------------------------------------------------------
# Unit tests for build_provenance()
# ---------------------------------------------------------------------------

def test_provenance_keys():
    cfg = RunConfig()
    prov = build_provenance(cfg)
    assert "git_commit" in prov
    assert "torch_version" in prov
    assert "cuda_version" in prov
    assert "network" in prov


def test_provenance_network_keys():
    cfg = RunConfig()
    net = build_provenance(cfg)["network"]
    assert "blocks" in net
    assert "filters" in net
    assert "params" in net
    assert "archetype" in net


def test_provenance_params_positive():
    cfg = RunConfig()
    net = build_provenance(cfg)["network"]
    assert net["params"] > 0


def test_provenance_archetype_string_format():
    cfg = RunConfig(network=NetworkConfig(blocks=6, filters=64))
    prov = build_provenance(cfg)
    assert prov["network"]["archetype"] == "resnet-6x64"


def test_provenance_archetype_10x128():
    cfg = RunConfig(network=NetworkConfig(blocks=10, filters=128))
    prov = build_provenance(cfg)
    net = prov["network"]
    assert net["archetype"] == "resnet-10x128"
    assert net["blocks"] == 10
    assert net["filters"] == 128
    # 10x128 net should have notably more params than 6x64
    cfg_small = RunConfig(network=NetworkConfig(blocks=6, filters=64))
    small_params = build_provenance(cfg_small)["network"]["params"]
    assert net["params"] > small_params


def test_provenance_network_values_match_config():
    cfg = RunConfig(network=NetworkConfig(blocks=4, filters=32))
    net = build_provenance(cfg)["network"]
    assert net["blocks"] == 4
    assert net["filters"] == 32
    assert net["archetype"] == "resnet-4x32"


# ---------------------------------------------------------------------------
# REST endpoint tests: GET /api/runs/{run_id}/provenance
# ---------------------------------------------------------------------------

def _make_run(runs_root, run_id="r1", with_provenance=True):
    run = runs_root / run_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "games").mkdir(parents=True)
    cfg = RunConfig(run_name=run_id)
    (run / "config.json").write_text(cfg.to_json())
    if with_provenance:
        prov = {
            "git_commit": "abc123",
            "torch_version": "2.0.0",
            "cuda_version": "12.0",
            "network": {
                "blocks": 6,
                "filters": 64,
                "params": 1234567,
                "archetype": "resnet-6x64",
            },
        }
        (run / "provenance.json").write_text(json.dumps(prov))
    return run


def test_provenance_endpoint_returns_200(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _make_run(runs_root, "r1")
    c = TestClient(create_app(runs_root))
    r = c.get("/api/runs/r1/provenance")
    assert r.status_code == 200


def test_provenance_endpoint_content(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _make_run(runs_root, "r1")
    c = TestClient(create_app(runs_root))
    data = c.get("/api/runs/r1/provenance").json()
    assert data["git_commit"] == "abc123"
    assert data["torch_version"] == "2.0.0"
    assert data["network"]["archetype"] == "resnet-6x64"
    assert data["network"]["params"] == 1234567


def test_provenance_endpoint_missing_file_is_404(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _make_run(runs_root, "r1", with_provenance=False)
    c = TestClient(create_app(runs_root))
    r = c.get("/api/runs/r1/provenance")
    assert r.status_code == 404


def test_provenance_endpoint_unknown_run_is_404(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _make_run(runs_root, "r1")
    c = TestClient(create_app(runs_root))
    r = c.get("/api/runs/nope/provenance")
    assert r.status_code == 404
