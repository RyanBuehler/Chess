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

Goal-conditioned mode (spec sec 8, Task 3.1)
--------------------------------------------
When constructed with a ``goal`` + ``protagonist``, the search becomes the
batched analogue of ``GoalReferenceMCTS``: a protagonist-frame **minimax** (NOT
negamax) over ``V(s,g) = P(protagonist achieves g) in [0,1]`` with exact goal
terminals. ``Node.value_sum`` then stores the protagonist-frame achievement
probability summed over visits (no per-level sign flip on backup); selection
converts to the parent-mover's exploitation view via ``q = sign*(2*V - 1)``
(sign = +1 if the parent mover is the protagonist else -1), exactly as the
reference. The copy-free leaf-parking path additionally encodes the goal planes
and records the per-leaf ``remaining`` deadline scalar, evaluated together by a
goal-conditioned ``evaluate_planes(planes_batch, deadlines)``.

At leaves_per_tree (K) == 1 with a single tree and add_noise=False, this
reproduces ``GoalReferenceMCTS`` EXACTLY (the Task 3.1 equivalence gate). The
vanilla path (``goal=None``) is left UNCHANGED and still matches ReferenceMCTS.
"""
import chess
import numpy as np

from chessrl.chess_env.encoding import encode_board, to_model_input
from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move, move_to_index
from chessrl.config.config import MCTSConfig
from chessrl.goals.encoding import encode_goal
from chessrl.goals.features import board_features
from chessrl.mcts.reference import Node, goal_terminal_value


class SearchTree:
    """One independent game tree: its root, a working board (kept at the root
    position between rounds), and how many simulations have been credited.

    In goal-conditioned mode ``baseline`` holds the BoardFeatures at the search
    root (for count-delta goal terminals); it is None in vanilla mode.

    ``goal``/``protagonist`` are the PER-TREE goal context. They are None for the
    single-goal modes (vanilla, or a goal ``BatchedMCTS`` built with a fixed
    goal/protagonist on the instance), in which case the instance-level
    ``self.goal``/``self.protagonist`` apply unchanged — so the validated K=1
    equivalence path is byte-for-byte identical. They are set only by the
    goal-aware CONCURRENT driver, which batches heterogeneous goals/protagonists
    across many in-flight games into one shared evaluator call: each tree carries
    its own goal, protagonist, and baseline, so a leaf encodes its OWN goal planes
    + deadline and the per-game minimax/terminal algebra uses its OWN context."""

    __slots__ = ("root", "board", "sims_done", "baseline", "goal", "protagonist")

    def __init__(self, board: chess.Board, baseline=None, goal=None, protagonist=None):
        self.root = Node(0.0)
        self.board = board
        self.sims_done = 0
        self.baseline = baseline
        self.goal = goal
        self.protagonist = protagonist


class BatchedMCTS:
    def __init__(
        self,
        evaluator_many,
        cfg: MCTSConfig,
        rng: np.random.Generator | None = None,
        goal=None,
        protagonist: bool | None = None,
        goal_mode: bool | None = None,
    ):
        self.evaluator = evaluator_many
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        # Goal-conditioned mode: protagonist-frame minimax + goal terminals,
        # threaded through the leaf-parking path. None -> vanilla negamax.
        #
        # Two ways to enter goal mode:
        #   * fixed goal+protagonist on the INSTANCE (the original single-game /
        #     equivalence path): goal != None, every tree uses this same context;
        #   * goal_mode=True with goal=None (the CONCURRENT goal driver): the
        #     instance carries NO goal; each tree carries its own goal context
        #     (SearchTree.goal/protagonist/baseline) so heterogeneous, switching
        #     goals batch together. Use init_tree_for_goal() to build such trees.
        self.goal = goal
        self.protagonist = protagonist
        self.goal_mode = (goal is not None) if goal_mode is None else goal_mode
        if goal is not None and protagonist is None:
            raise ValueError("goal-conditioned BatchedMCTS requires a protagonist")

    def _ctx(self, tree: "SearchTree"):
        """(goal, protagonist) for this tree: the per-tree context if set
        (concurrent driver), else the instance-level fixed context (equivalence
        path). Validated K=1 path leaves tree.goal None -> returns self.goal."""
        if tree.goal is not None:
            return tree.goal, tree.protagonist
        return self.goal, self.protagonist

    # ---- public API -----------------------------------------------------

    def init_tree(self, board: chess.Board, add_noise: bool = False) -> SearchTree:
        b = board.copy()
        baseline = board_features(b) if self.goal_mode else None
        tree = SearchTree(b, baseline=baseline)
        value = self._expand_leaf(tree, tree.root, plies_from_root=0)
        tree.root.visit_count += 1            # initial expansion counts as one visit (matches reference)
        tree.root.value_sum += value
        if add_noise and tree.root.children:
            self._add_dirichlet(tree.root)
        return tree

    def init_tree_for_goal(
        self, board: chess.Board, goal, protagonist: bool, add_noise: bool = False
    ) -> SearchTree:
        """Build a goal-mode tree carrying its OWN goal/protagonist/baseline (the
        concurrent goal driver path). The instance must be in goal mode but need
        not have a fixed goal. The baseline is the count-delta origin captured at
        THIS search root, exactly as GoalReferenceMCTS recomputes per search."""
        if not self.goal_mode:
            raise ValueError("init_tree_for_goal requires a goal-mode BatchedMCTS")
        b = board.copy()
        tree = SearchTree(
            b, baseline=board_features(b), goal=goal, protagonist=protagonist
        )
        value = self._expand_leaf(tree, tree.root, plies_from_root=0)
        tree.root.visit_count += 1
        tree.root.value_sum += value
        if add_noise and tree.root.children:
            self._add_dirichlet(tree.root)
        return tree

    def run(self, tree: SearchTree) -> None:
        """Advance a single tree to cfg.simulations (drives K-leaf rounds)."""
        while tree.root.visit_count < self.cfg.simulations + 1:
            self.step_round([tree])

    def step_round(self, trees: list) -> None:
        """Advance every not-yet-finished tree by one batching round. All
        non-terminal leaves selected this round (across all trees) are evaluated
        in a single evaluate_planes call."""
        k = self.cfg.leaves_per_tree
        # parked entries: (tree, path, planes_float32, legal_idxs, deadline)
        # planes and legal_idxs are computed at park time, before popping, so
        # no Board.copy() is needed. ``deadline`` is the per-leaf remaining-to-
        # deadline scalar (None in vanilla mode).
        parked = []
        # dirty_nodes: accumulate every node touched by virtual loss so we can
        # zero exactly those nodes after all backups (no full-tree DFS needed).
        dirty_nodes = []
        for tree in trees:
            if tree.root.visit_count >= self.cfg.simulations + 1:
                continue
            for _ in range(k):
                if tree.root.visit_count >= self.cfg.simulations + 1:
                    break
                path = self._select_leaf(tree)
                self._apply_virtual_loss(path)
                dirty_nodes.extend(path)
                tree.sims_done += 1
                plies = len(path) - 1
                term = self._terminal_value(tree, plies)
                if term is not None:
                    self._backup(path, term)
                    self._pop_to_root(tree, path)
                else:
                    # Encode board and collect legal indices HERE, while the
                    # working board is at the leaf — avoids Board.copy().
                    flip = tree.board.turn == chess.BLACK
                    legal_idxs = [move_to_index(m, flip) for m in tree.board.legal_moves]
                    if self.goal_mode:
                        goal, protagonist = self._ctx(tree)
                        remaining = goal.deadline - plies
                        goal_planes, _ = encode_goal(goal, remaining, protagonist)
                        board_planes = to_model_input(encode_board(tree.board))
                        planes = np.concatenate(
                            [board_planes, goal_planes.astype(np.float32)], axis=0
                        )
                        deadline = remaining
                    else:
                        planes = to_model_input(encode_board(tree.board))
                        deadline = None
                    self._pop_to_root(tree, path)
                    parked.append((tree, path, planes, legal_idxs, deadline))
        if parked:
            planes_batch = np.stack([p[2] for p in parked])
            if self.goal_mode:
                deadlines = np.asarray([p[4] for p in parked], dtype=np.float32)
                policies, values = self.evaluator.evaluate_planes(planes_batch, deadlines)
            else:
                policies, values = self.evaluator.evaluate_planes(planes_batch)
            for (tree, path, _planes, legal_idxs, _deadline), policy, value in zip(
                parked, policies, values
            ):
                self._expand_from_idxs(path[-1], legal_idxs, policy, float(value))
                self._backup(path, float(value))
        # Clear virtual loss on exactly the nodes that were dirtied this round.
        # _backup already decremented vloss for backed-up paths; this handles
        # any remaining residual (safety net, mirrors the old DFS behaviour but
        # without traversing the whole tree).
        for n in dirty_nodes:
            if n.vloss:
                n.vloss = 0

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
            value = self._expand_leaf(tree, tree.root, plies_from_root=0)
            tree.root.visit_count += 1
            tree.root.value_sum += value
            tree.sims_done = tree.root.visit_count
            return
        tree.root = child
        if not child.children and child.visit_count == 0:
            value = self._expand_leaf(tree, tree.root, plies_from_root=0)
            tree.root.visit_count += 1
            tree.root.value_sum += value
        # No ghost visit: child's true statistics are kept intact.
        # sims_done is derived from root.visit_count so run() tops up correctly.
        tree.sims_done = tree.root.visit_count - 1

    def add_root_noise(self, tree: SearchTree) -> None:
        if tree.root.children:
            self._add_dirichlet(tree.root)

    # ---- internals ------------------------------------------------------

    def _select_leaf(self, tree: SearchTree) -> list:
        """Descend from root to an unexpanded/terminal leaf, pushing moves on
        tree.board. Returns the path (root..leaf). Caller must pop back."""
        _, protagonist = self._ctx(tree)
        node = tree.root
        path = [node]
        while node.children:
            idx, node = self._select(node, mover=tree.board.turn, protagonist=protagonist)
            tree.board.push(index_to_move(idx, tree.board.turn == chess.BLACK, tree.board))
            path.append(node)
        return path

    def _select(self, node: Node, mover: bool, protagonist: bool | None = None):
        if self.goal_mode:
            return self._select_goal(node, mover, protagonist)
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

    def _select_goal(self, node: Node, mover: bool, protagonist: bool | None = None):
        """Protagonist-frame minimax selection (batched analogue of
        ``GoalReferenceMCTS._select``). The parent mover's exploitation view on
        the negamax [-1,1] scale is ``q = sign*(2*p - 1)`` with sign = +1 if the
        mover is the protagonist else -1. Virtual loss adds pseudo-visits valued
        at the worst achievement probability FOR THE PARENT MOVER (p=0 when the
        mover is the protagonist, p=1 when the mover is the opponent), so vloss
        pulls q toward -1 just like the vanilla path. At K=1 (vloss==0) this is
        bit-identical to the reference.

        ``protagonist`` is the per-tree protagonist (concurrent driver); when
        None it falls back to the instance ``self.protagonist`` (equivalence
        path), leaving that path byte-for-byte unchanged."""
        if protagonist is None:
            protagonist = self.protagonist
        sign = 1.0 if mover == protagonist else -1.0
        p_vloss = 0.0 if sign > 0 else 1.0  # worst achievement prob for this mover

        eff_parent_n = node.visit_count + node.vloss
        sqrt_n = eff_parent_n ** 0.5
        if eff_parent_n:
            parent_p = (node.value_sum + p_vloss * node.vloss) / eff_parent_n
            parent_q = sign * (2.0 * parent_p - 1.0)
        else:
            parent_q = 0.0
        fpu = parent_q - self.cfg.fpu_reduction
        best_idx, best_child, best_score = -1, None, -1e18
        for idx, ch in node.children.items():
            ch_n = ch.visit_count + ch.vloss
            if ch_n:
                ch_p = (ch.value_sum + p_vloss * ch.vloss) / ch_n
                q = sign * (2.0 * ch_p - 1.0)
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
        # Goal mode: protagonist-frame, NO per-level sign flip (every node
        # accumulates the same achievement probability). Vanilla: negamax flip.
        v = value
        for n in reversed(path):
            n.vloss -= 1
            n.visit_count += 1
            n.value_sum += v
            if not self.goal_mode:
                v = -v

    def _pop_to_root(self, tree: SearchTree, path: list) -> None:
        for _ in range(len(path) - 1):
            tree.board.pop()

    def _terminal_value(self, tree: SearchTree, plies_from_root: int):
        """Terminal value at the tree's current (leaf) board. Vanilla: zero-sum
        ``terminal_value``. Goal mode: exact protagonist-frame goal terminal
        (achieved 1 / expired 0 / game-over evaluate-g), with ``remaining =
        deadline - plies_from_root``."""
        if not self.goal_mode:
            return terminal_value(tree.board)
        goal, protagonist = self._ctx(tree)
        remaining = goal.deadline - plies_from_root
        return goal_terminal_value(
            tree.board, goal, protagonist, remaining, tree.baseline
        )

    def _expand_leaf(self, tree: SearchTree, node: Node, plies_from_root: int) -> float:
        """Terminal-or-evaluate a single leaf at the tree's current board (used
        by init_tree / advance, which are not part of the batch). Returns the
        backup value; mirrors the reference ``_expand``."""
        term = self._terminal_value(tree, plies_from_root)
        if term is not None:
            return term
        board = tree.board
        flip = board.turn == chess.BLACK
        idxs = [move_to_index(m, flip) for m in board.legal_moves]
        if self.goal_mode:
            goal, protagonist = self._ctx(tree)
            remaining = goal.deadline - plies_from_root
            policy, value = self.evaluator.evaluate_one_goal(
                board, goal, remaining, protagonist
            )
        else:
            policies, values = self.evaluator.evaluate_many([board])
            policy, value = policies[0], float(values[0])
        self._expand_from_idxs(node, idxs, policy, value)
        return value

    def _expand_from_idxs(self, node: Node, idxs: list, policy, value: float) -> None:
        """Expand node using pre-computed legal move indices and a policy vector."""
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
