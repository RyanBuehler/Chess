"""Reference (sequential) PUCT search - the permanent correctness baseline.

Sign convention: Node.value_sum is from the perspective of the side to move
at that node. A parent evaluates child quality as -child.q(). Slow by design;
the batched implementation (M5) is diffed against this one.
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move, move_to_index
from chessrl.config.config import MCTSConfig


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}

    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class ReferenceMCTS:
    def __init__(self, evaluator, cfg: MCTSConfig, rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    def search(self, board: chess.Board, add_noise: bool = False):
        """Returns ({action_index: visit_count}, root_q). root_q is from the
        side to move's perspective (used for resignation)."""
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

    def _select(self, node: Node):
        sqrt_n = node.visit_count ** 0.5
        fpu = node.q() - self.cfg.fpu_reduction  # first-play urgency, parent's perspective
        best_idx, best_child, best_score = -1, None, -1e18
        for idx, ch in node.children.items():
            q = -ch.q() if ch.visit_count else fpu
            score = q + self.cfg.c_puct * ch.prior * sqrt_n / (1 + ch.visit_count)
            if score > best_score:
                best_idx, best_child, best_score = idx, ch, score
        return best_idx, best_child

    def _expand(self, node: Node, board: chess.Board) -> float:
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

    def _add_dirichlet(self, root: Node) -> None:
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        noise = self.rng.dirichlet([alpha] * len(root.children))
        for n, ch in zip(noise, root.children.values()):
            ch.prior = (1 - eps) * ch.prior + eps * float(n)
