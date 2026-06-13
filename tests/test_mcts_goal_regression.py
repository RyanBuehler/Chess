"""Task 2.4 — THE REGRESSION GATE (hard blocker).

With ``g = WIN_GOAL`` and no sub-goals, the goal-conditioned protagonist-frame
minimax search must reproduce the legacy negamax search's visit distribution
EXACTLY (tol=0), on fixed positions, at the same seed.

These are *different value representations* — legacy negamax is tanh-scale
v in [-1,1]; the goal head is sigmoid p in [0,1]. So this gate does NOT compare
two trained networks (that would prove nothing). It tests the SEARCH ALGEBRA:

  (a) ``legacy_negamax_search`` below is a FROZEN snapshot of reference.py's
      negamax search at the time of the redesign (copied verbatim), immune to
      later edits of reference.py.
  (b) Both searches are driven by the SAME deterministic evaluator. The legacy
      evaluator returns (policy, v) with v in [-1,1]; the goal evaluator returns
      (policy, p) with p = (v + 1) / 2, i.e. the affine map v = 2p - 1. The goal
      search's selection converts back via 2p - 1, so protagonist-max /
      opponent-min is bit-identical to negamax.

Then we assert visits_goal == visits_legacy exactly.
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import NUM_ACTIONS, index_to_move, move_to_index
from chessrl.config.config import MCTSConfig
from chessrl.goals.templates import WIN_GOAL
from chessrl.mcts.reference import GoalReferenceMCTS, Node


# --------------------------------------------------------------------------
# (a) FROZEN snapshot of the legacy negamax search (verbatim copy of the
#     ReferenceMCTS.search/_select/_expand at the time of the Stage-2 redesign).
#     Do NOT refactor to call reference.py — the point is immunity to edits.
# --------------------------------------------------------------------------
class _LegacyNegamaxMCTS:
    def __init__(self, evaluator, cfg, rng):
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = rng

    def search(self, board, add_noise=False):
        root = Node(0.0)
        value = self._expand(root, board)
        root.visit_count += 1
        root.value_sum += value
        if add_noise and root.children:
            self._add_dirichlet(root)
        for _ in range(self.cfg.simulations):
            b = board.copy()
            node, path = root, [root]
            while node.children:
                idx, node = self._select(node)
                b.push(index_to_move(idx, b.turn == chess.BLACK, b))
                path.append(node)
            value = self._expand(node, b)
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value
        visits = {i: c.visit_count for i, c in root.children.items() if c.visit_count > 0}
        return visits, root.q()

    def _select(self, node):
        sqrt_n = node.visit_count ** 0.5
        fpu = node.q() - self.cfg.fpu_reduction
        best_idx, best_child, best_score = -1, None, -1e18
        for idx, ch in node.children.items():
            q = -ch.q() if ch.visit_count else fpu
            score = q + self.cfg.c_puct * ch.prior * sqrt_n / (1 + ch.visit_count)
            if score > best_score:
                best_idx, best_child, best_score = idx, ch, score
        return best_idx, best_child

    def _expand(self, node, board):
        term = terminal_value(board)
        if term is not None:
            return term
        policy, value = self.evaluator.evaluate(board)
        flip = board.turn == chess.BLACK
        idxs = [move_to_index(m, flip) for m in board.legal_moves]
        priors = np.asarray([policy[i] for i in idxs], dtype=np.float64)
        total = priors.sum()
        priors = priors / total if total > 0 else np.full(len(idxs), 1.0 / len(idxs))
        for i, idx in enumerate(idxs):
            node.children[idx] = Node(float(priors[i]))
        return value

    def _add_dirichlet(self, root):
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        noise = self.rng.dirichlet([alpha] * len(root.children))
        for n, ch in zip(noise, root.children.values()):
            ch.prior = (1 - eps) * ch.prior + eps * float(n)


# --------------------------------------------------------------------------
# (b) Shared deterministic evaluator. A fixed pseudo-random policy + value per
#     position (seeded by the FEN), returned on BOTH scales by the two adapter
#     evaluators below. Same position -> same (policy, v) every call.
# --------------------------------------------------------------------------
class _DeterministicCore:
    def _pv(self, board):
        # Stable across processes (Python's hash() is salted per run).
        import hashlib

        digest = hashlib.sha256(board.fen().encode()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal(NUM_ACTIONS)
        policy = np.exp(logits - logits.max())
        policy = policy / policy.sum()
        v = float(np.tanh(rng.standard_normal()))  # value in (-1, 1)
        return policy.astype(np.float64), v


class LegacyEvaluator(_DeterministicCore):
    """Negamax-scale evaluator: value v in [-1,1] (side-to-move frame)."""

    def evaluate(self, board):
        policy, v = self._pv(board)
        return policy, v


class GoalEvaluator(_DeterministicCore):
    """Goal-scale evaluator: the SAME (policy, v) — but the goal value head is
    PROTAGONIST-FRAME (it always reports P(protagonist achieves goal), since the
    real net is conditioned on the protagonist via the input planes). The legacy
    ``v`` is SIDE-TO-MOVE-frame; convert to protagonist frame, then to a
    probability p = (v_proto + 1)/2. The goal search treats this as protagonist
    frame (no flip), so 2p - 1 = v_proto and the algebra is bit-identical to
    negamax."""

    def evaluate(self, board, goal, remaining, protagonist):
        policy, v = self._pv(board)
        v_proto = v if board.turn == protagonist else -v
        p = (v_proto + 1.0) / 2.0
        return policy, p


_POSITIONS = [
    chess.Board(),  # startpos
    chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"),
]


def _run_pair(board, sims, seed):
    cfg = MCTSConfig(simulations=sims)
    legacy = _LegacyNegamaxMCTS(LegacyEvaluator(), cfg, rng=np.random.default_rng(seed))
    goal = GoalReferenceMCTS(GoalEvaluator(), cfg, rng=np.random.default_rng(seed))
    visits_legacy, q_legacy = legacy.search(board.copy(), add_noise=False)
    visits_goal, v_goal = goal.search(board.copy(), goal=WIN_GOAL, protagonist=board.turn, add_noise=False)
    return visits_legacy, visits_goal, q_legacy, v_goal


def test_win_goal_matches_negamax_visit_distribution():
    for board in _POSITIONS:
        visits_legacy, visits_goal, _, _ = _run_pair(board, sims=64, seed=0)
        assert visits_goal == visits_legacy, f"divergence on {board.fen()}"


def test_win_goal_matches_negamax_at_multiple_seeds_and_sims():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    for seed in (0, 1, 7):
        for sims in (16, 64, 200):
            vl, vg, _, _ = _run_pair(board, sims=sims, seed=seed)
            assert vg == vl, f"divergence seed={seed} sims={sims}"


def test_root_value_matches_under_affine_map():
    """root_q (negamax, [-1,1]) and root_v (goal, [0,1]) relate by v = (q+1)/2."""
    for board in _POSITIONS:
        _, _, q_legacy, v_goal = _run_pair(board, sims=64, seed=0)
        assert abs((2.0 * v_goal - 1.0) - q_legacy) < 1e-9, board.fen()
