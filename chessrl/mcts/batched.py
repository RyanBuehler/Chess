"""Batched PUCT search over many concurrent game trees.

Diffed against ReferenceMCTS: at leaves_per_tree (K) == 1, a single tree, and
add_noise=False, this produces the EXACT same visit-count dict as the reference
on the same position. That equivalence is the milestone's correctness gate.

Key differences from the reference (all behavior-preserving at K=1):
  * descent uses board.push / board.pop on each tree's own board instead of a
    full Board.copy per simulation (the spec's CPU-bound mitigation);
  * non-terminal leaves across all active trees in one round are evaluated in a
    single evaluate_many call (the GPU batch);
  * virtual loss diversifies the K>1 selections within one tree per round, and
    is a no-op at K=1.

Sign convention is identical to the reference: Node.value_sum is from the
perspective of the side to move at that node; a parent reads child quality as
-child.q(); backup flips sign each level leaf->root.
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move, move_to_index
from chessrl.config.config import MCTSConfig
from chessrl.mcts.reference import Node


class SearchTree:
    """One independent game tree: its root, a working board (kept at the root
    position between rounds), and how many simulations have been credited."""

    __slots__ = ("root", "board", "sims_done")

    def __init__(self, board: chess.Board):
        self.root = Node(0.0)
        self.board = board
        self.sims_done = 0


class BatchedMCTS:
    def __init__(self, evaluator_many, cfg: MCTSConfig, rng: np.random.Generator | None = None):
        self.evaluator = evaluator_many
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---- public API -----------------------------------------------------

    def init_tree(self, board: chess.Board, add_noise: bool = False) -> SearchTree:
        tree = SearchTree(board.copy())
        policy, value = self._evaluate_one(tree.board)
        self._expand(tree.root, tree.board, policy, value, is_terminal=False)
        tree.root.visit_count += 1            # initial expansion counts as one visit (matches reference)
        tree.root.value_sum += value
        if add_noise and tree.root.children:
            self._add_dirichlet(tree.root)
        return tree

    def run(self, tree: SearchTree) -> None:
        """Advance a single tree to cfg.simulations (drives K-leaf rounds)."""
        while tree.sims_done < self.cfg.simulations:
            self.step_round([tree])

    def step_round(self, trees: list) -> None:
        """Advance every not-yet-finished tree by one batching round. All
        non-terminal leaves selected this round (across all trees) are evaluated
        in a single evaluate_many call."""
        k = self.cfg.leaves_per_tree
        parked = []   # (tree, path, board_at_leaf) awaiting GPU evaluation
        for tree in trees:
            if tree.sims_done >= self.cfg.simulations:
                continue
            for _ in range(k):
                if tree.sims_done >= self.cfg.simulations:
                    break
                path = self._select_leaf(tree)
                leaf = path[-1]
                self._apply_virtual_loss(path)
                tree.sims_done += 1
                term = terminal_value(tree.board)
                if term is not None:
                    self._backup(path, term)
                    self._pop_to_root(tree, path)
                else:
                    leaf_board = tree.board.copy()
                    self._pop_to_root(tree, path)
                    parked.append((tree, path, leaf_board))
        if parked:
            policies, values = self.evaluator.evaluate_many([p[2] for p in parked])
            for (tree, path, leaf_board), policy, value in zip(parked, policies, values):
                self._expand(path[-1], leaf_board, policy, float(value), is_terminal=False)
                self._backup(path, float(value))
        for tree in trees:
            self._clear_virtual_loss(tree.root)

    def visit_counts(self, tree: SearchTree) -> dict:
        return {i: c.visit_count for i, c in tree.root.children.items() if c.visit_count > 0}

    def root_q(self, tree: SearchTree) -> float:
        return tree.root.q()

    def advance(self, tree: SearchTree, action_index: int) -> None:
        """Re-root at the chosen child (subtree reuse): discard siblings, keep
        the chosen subtree's accumulated statistics, and push the move on the
        tree's board so the working board sits at the new root."""
        child = tree.root.children.get(action_index)
        tree.board.push(index_to_move(action_index, tree.board.turn == chess.BLACK, tree.board))
        if child is None:
            tree.root = Node(0.0)
            policy, value = self._evaluate_one(tree.board)
            self._expand(tree.root, tree.board, policy, value, is_terminal=False)
            tree.root.visit_count += 1
            tree.root.value_sum += value
            tree.sims_done = tree.root.visit_count
            return
        tree.root = child
        if not child.children and child.visit_count == 0:
            policy, value = self._evaluate_one(tree.board)
            self._expand(tree.root, tree.board, policy, value, is_terminal=False)
            tree.root.visit_count += 1
            tree.root.value_sum += value
        elif child.children:
            # When this node was first visited as a leaf its own visit_count got
            # +1 from backup but its children got 0 visits. Add 1 visit (at the
            # same q-value, so q() is preserved) to the most-visited grandchild
            # to restore the invariant: children_sum == root.visit_count - 1.
            # This "ghost visit" accounts for the initial leaf-expansion sim and
            # ensures run() tops up children_sum to exactly cfg.simulations.
            gc = max(child.children.values(), key=lambda n: n.visit_count)
            if gc.visit_count > 0:
                gc.value_sum += gc.q()  # preserves q() after visit_count += 1
            gc.visit_count += 1
        tree.sims_done = tree.root.visit_count

    def add_root_noise(self, tree: SearchTree) -> None:
        if tree.root.children:
            self._add_dirichlet(tree.root)

    # ---- internals ------------------------------------------------------

    def _select_leaf(self, tree: SearchTree) -> list:
        """Descend from root to an unexpanded/terminal leaf, pushing moves on
        tree.board. Returns the path (root..leaf). Caller must pop back."""
        node = tree.root
        path = [node]
        while node.children:
            idx, node = self._select(node)
            tree.board.push(index_to_move(idx, tree.board.turn == chess.BLACK, tree.board))
            path.append(node)
        return path

    def _select(self, node: Node):
        # effective visits/value include virtual loss; vloss adds visits valued
        # -1 from the parent's perspective.
        eff_parent_n = node.visit_count + node.vloss
        sqrt_n = eff_parent_n ** 0.5
        parent_q = (node.value_sum - node.vloss) / eff_parent_n if eff_parent_n else 0.0
        fpu = parent_q - self.cfg.fpu_reduction
        best_idx, best_child, best_score = -1, None, -1e18
        for idx, ch in node.children.items():
            ch_n = ch.visit_count + ch.vloss
            if ch_n:
                child_q = (ch.value_sum - ch.vloss) / ch_n
                q = -child_q
            else:
                q = fpu
            score = q + self.cfg.c_puct * ch.prior * sqrt_n / (1 + ch_n)
            if score > best_score:
                best_idx, best_child, best_score = idx, ch, score
        return best_idx, best_child

    def _apply_virtual_loss(self, path: list) -> None:
        for n in path:
            n.vloss += 1

    def _backup(self, path: list, value: float) -> None:
        # remove the virtual loss this path added, then apply the real value.
        v = value
        for n in reversed(path):
            n.vloss -= 1
            n.visit_count += 1
            n.value_sum += v
            v = -v

    def _clear_virtual_loss(self, node: Node) -> None:
        # safety net: after a completed round every vloss should already be 0
        # (each applied loss is removed in _backup). This re-zeroes defensively.
        stack = [node]
        while stack:
            n = stack.pop()
            if n.vloss:
                n.vloss = 0
            stack.extend(n.children.values())

    def _pop_to_root(self, tree: SearchTree, path: list) -> None:
        for _ in range(len(path) - 1):
            tree.board.pop()

    def _evaluate_one(self, board: chess.Board):
        policies, values = self.evaluator.evaluate_many([board])
        return policies[0], float(values[0])

    def _expand(self, node: Node, board: chess.Board, policy, value: float, is_terminal: bool) -> None:
        flip = board.turn == chess.BLACK
        idxs = [move_to_index(m, flip) for m in board.legal_moves]
        if not idxs:
            return
        priors = np.asarray([policy[i] for i in idxs], dtype=np.float64)
        total = priors.sum()
        priors = priors / total if total > 0 else np.full(len(idxs), 1.0 / len(idxs))
        for i, idx in enumerate(idxs):
            node.children[idx] = Node(float(priors[i]))

    def _add_dirichlet(self, root: Node) -> None:
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        noise = self.rng.dirichlet([alpha] * len(root.children))
        for n, ch in zip(noise, root.children.values()):
            ch.prior = (1 - eps) * ch.prior + eps * float(n)
