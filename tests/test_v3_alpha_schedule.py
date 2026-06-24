import numpy as np
import chess

from chessrl.config.config import MCTSConfig
from chessrl.mcts.batched import BatchedMCTS
from tests.test_meansend_selfplay import FakeVectorEval


def test_per_tree_alpha_overrides_cfg():
    ev = FakeVectorEval()
    mcts = BatchedMCTS(ev, MCTSConfig(simulations=4, leaves_per_tree=1, meansend_alpha=0.25),
                       rng=np.random.default_rng(0), meansend=True)
    t = mcts.init_tree_for_meansend(chess.Board(), np.full(4, -1.0, np.float32), 60, add_noise=False)
    assert t.meansend_alpha is None          # default: falls back to cfg
    t.meansend_alpha = 0.0                    # pure win-value leaf
    mcts.run(t)                               # must not raise; uses per-tree alpha
    assert mcts.visit_counts(t)


def test_alpha_schedule_shape():
    from chessrl.selfplay.meansend_chained import alpha_schedule
    assert abs(alpha_schedule(0.0, 0.25, 0.6, 0, 512, 20) - 0.25) < 1e-6   # unclear -> full
    assert alpha_schedule(0.6, 0.25, 0.6, 0, 512, 20) == 0.0               # decisive (won)
    assert alpha_schedule(-0.6, 0.25, 0.6, 0, 512, 20) == 0.0              # decisive (lost)
    assert alpha_schedule(0.0, 0.25, 0.6, 500, 512, 20) == 0.0             # near ply cap
