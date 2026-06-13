"""Reference (sequential) PUCT search - the permanent correctness baseline.

Sign convention: Node.value_sum is from the perspective of the side to move
at that node. A parent evaluates child quality as -child.q(). Slow by design;
the batched implementation (M5) is diffed against this one.

This module hosts two searches:

* ``ReferenceMCTS`` — the original zero-sum negamax search (vanilla arm). Left
  intact; it is the permanent equivalence baseline.
* ``GoalReferenceMCTS`` — the goal-conditioned search (spec sec 8): a
  protagonist-frame **minimax** (NOT negamax) over a value ``V(s,g) =
  P(protagonist achieves g by deadline) in [0,1]``, with exact goal terminals.
  When ``g = WIN_GOAL`` it reduces EXACTLY to ``ReferenceMCTS`` under the affine
  map ``v = 2p - 1`` (regression gate, Task 2.4).
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move, move_to_index
from chessrl.config.config import MCTSConfig
from chessrl.goals import templates as T
from chessrl.goals.features import board_features


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "vloss")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.vloss = 0      # virtual loss (batched MCTS only; reference leaves it at 0)

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
        root.visit_count += 1  # the initial expansion counts: root ends at simulations + 1
        root.value_sum += value
        if add_noise and root.children:
            self._add_dirichlet(root)
        for _ in range(self.cfg.simulations):
            b = board.copy()  # full copy per simulation: intentionally simple/slow; M5 optimizes
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


def goal_terminal_value(
    board: chess.Board,
    goal: T.GoalTemplate,
    protagonist: bool,
    remaining: int,
    baseline,
):
    """Exact goal terminal in the protagonist frame (spec sec 8). Returns a
    probability in {0.0, 0.5, 1.0} when the node is a goal-terminal, else None.

    Precedence (per spec): achieved -> 1; deadline expired -> 0; real game-over
    -> evaluate the goal (win goal: win=1, draw=0.5, loss=0; any other goal not
    achieved at a real game-over is a 0). ``remaining`` is plies left to the
    deadline AT this board (deadline - plies_from_root). ``baseline`` is the
    BoardFeatures at the search root (for count deltas).
    """
    achieved = _goal_achieved(board, goal, protagonist, baseline)
    if achieved:
        return 1.0
    over = board.is_game_over(claim_draw=True)
    if not over and remaining <= 0:
        # Deadline expired without achievement.
        return 0.0
    if over:
        if goal.is_win():
            outcome = board.outcome(claim_draw=True)
            if outcome.winner is None:
                return 0.5
            return 1.0 if outcome.winner == protagonist else 0.0
        # Sub-goal not achieved and the game ended: cannot be achieved anymore.
        return 0.0
    return None


def _goal_achieved(board, goal: T.GoalTemplate, protagonist: bool, baseline) -> bool:
    """Does ``goal`` hold at ``board`` for the protagonist, vs the root baseline?

    State-predicate goals (capture, check, win) are read from the current board;
    move-predicate goals (castle, promote, reach_*) are read from the last move
    on the board's move stack (which is the move that produced this node)."""
    kind = goal.kind
    opponent = not protagonist

    if kind == T.CAPTURE:
        pt = goal.param("piece_type")
        feats = board_features(board)
        return feats.counts[(pt, opponent)] < baseline.counts[(pt, opponent)]

    if kind == T.CHECK:
        return board.is_check() and board.turn == opponent

    if kind == T.WIN:
        if not board.is_game_over(claim_draw=True):
            return False
        outcome = board.outcome(claim_draw=True)
        return outcome.winner == protagonist

    # Move-predicate goals: inspect the move that produced this board.
    if not board.move_stack:
        return False
    move = board.peek()
    board.pop()
    try:
        mover_is_protagonist = board.turn == protagonist
        if not mover_is_protagonist:
            return False
        if kind == T.CASTLE:
            return board.is_castling(move)
        if kind == T.PROMOTE:
            return move.promotion is not None
        if kind == T.REACH_RANK:
            pt = goal.param("piece_type")
            target = goal.param("rank") if protagonist == chess.WHITE else 7 - goal.param("rank")
            mover = board.piece_at(move.from_square)
            return (
                mover is not None
                and mover.piece_type == pt
                and chess.square_rank(move.to_square) == target
            )
        if kind == T.REACH_SQUARE:
            pt = goal.param("piece_type")
            target = (
                goal.param("square")
                if protagonist == chess.WHITE
                else chess.square_mirror(goal.param("square"))
            )
            mover = board.piece_at(move.from_square)
            return mover is not None and mover.piece_type == pt and move.to_square == target
        return False
    finally:
        board.push(move)


class GoalReferenceMCTS:
    """Goal-conditioned reference search (spec sec 8, sec 10).

    Protagonist-frame minimax over ``V(s,g) = P(protagonist achieves g) in
    [0,1]``: protagonist-to-move nodes maximize child V, opponent-to-move nodes
    MINIMIZE it (no negamax sign flip into a shared accumulator). Exact goal
    terminals (achieved 1 / expired 0 / game-over evaluate-g).

    ``Node.value_sum`` here stores the **protagonist-frame** achievement
    probability summed over visits (NOT a side-to-move negamax value). The
    selection converts to the parent-mover's exploitation view explicitly.

    Equivalence to ReferenceMCTS for g=win: with the affine map ``v = 2p - 1``,
    the protagonist-frame minimax reproduces negamax's visit distribution
    exactly (Task 2.4). The selection math below is written so the ordering and
    PUCT scores are bit-identical.
    """

    def __init__(self, evaluator, cfg: MCTSConfig, rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    def search(self, board: chess.Board, goal: T.GoalTemplate, protagonist: bool, add_noise: bool = False):
        """Returns ({action_index: visit_count}, root_value). root_value is the
        protagonist-frame achievement probability at the root."""
        self.goal = goal
        self.protagonist = protagonist
        self.baseline = board_features(board)
        self.root_ply = 0  # plies are counted relative to the search root

        root = Node(0.0)
        value = self._expand(root, board, plies_from_root=0)
        root.visit_count += 1
        root.value_sum += value
        if add_noise and root.children:
            self._add_dirichlet(root)
        for _ in range(self.cfg.simulations):
            b = board.copy()
            node, path = root, [root]
            depth = 0
            while node.children:
                idx, node = self._select(node, mover=b.turn)
                b.push(index_to_move(idx, b.turn == chess.BLACK, b))
                path.append(node)
                depth += 1
            value = self._expand(node, b, plies_from_root=depth)
            # Backup in protagonist frame: NO sign flip — every node accumulates
            # the same protagonist-frame achievement probability.
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
        visits = {i: c.visit_count for i, c in root.children.items() if c.visit_count > 0}
        return visits, root.q()

    def _select(self, node: Node, mover: bool):
        """Pick the child the current mover prefers, in protagonist frame.

        Exploitation in the parent-mover's view: protagonist-to-move maximizes
        child V; opponent-to-move minimizes it. We express this on the negamax
        [-1,1] scale via ``q = sign * (2*V - 1)`` where ``sign = +1`` if the
        mover is the protagonist else ``-1`` — identical magnitude/ordering to
        negamax's ``-ch.q()`` per-level alternation, so g=win is bit-identical.
        """
        proto_to_move = (mover == self.protagonist)
        sign = 1.0 if proto_to_move else -1.0
        sqrt_n = node.visit_count ** 0.5
        # FPU: parent's exploitation value from the mover's view (negamax scale).
        parent_q = sign * (2.0 * node.q() - 1.0)
        fpu = parent_q - self.cfg.fpu_reduction
        best_idx, best_child, best_score = -1, None, -1e18
        for idx, ch in node.children.items():
            if ch.visit_count:
                q = sign * (2.0 * ch.q() - 1.0)
            else:
                q = fpu
            score = q + self.cfg.c_puct * ch.prior * sqrt_n / (1 + ch.visit_count)
            if score > best_score:
                best_idx, best_child, best_score = idx, ch, score
        return best_idx, best_child

    def _expand(self, node: Node, board: chess.Board, plies_from_root: int) -> float:
        remaining = self.goal.deadline - plies_from_root
        term = goal_terminal_value(board, self.goal, self.protagonist, remaining, self.baseline)
        if term is not None:
            return term
        policy, value = self.evaluator.evaluate(board, self.goal, remaining, self.protagonist)
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
