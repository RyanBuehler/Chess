import numpy as np
from chessrl.goals.reservoir import Reservoir


def test_fills_then_caps():
    r = Reservoir(capacity=100, dim=4, rng=np.random.default_rng(0))
    for i in range(50):
        r.add(np.full(4, float(i)))
    assert len(r) == 50 and r.seen == 50
    assert r.array().shape == (50, 4)
    for i in range(200):
        r.add(np.full(4, 1000.0 + i))
    assert len(r) == 100 and r.seen == 250
    assert r.array().shape == (100, 4)


def test_uniform_sampling_keeps_a_mix():
    # After overflow, the reservoir should contain some early and some late items
    # (not exclusively the last `capacity`). With a fixed seed this is deterministic.
    r = Reservoir(capacity=10, dim=1, rng=np.random.default_rng(42))
    for i in range(1000):
        r.add(np.array([float(i)]))
    vals = r.array().ravel()
    assert len(vals) == 10
    assert vals.min() < 990  # at least one item from before the final window survived


def test_dtype_is_float32():
    r = Reservoir(capacity=5, dim=3, rng=np.random.default_rng(1))
    r.add(np.ones(3, dtype=np.float64))
    assert r.array().dtype == np.float32
