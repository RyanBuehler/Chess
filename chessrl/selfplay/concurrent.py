"""Concurrent self-play: many games advanced in lockstep through one batched
MCTS, so every search round produces a single shared GPU batch.

Per-game logic mirrors selfplay/play.py exactly (search with root noise,
temperature then argmax, ply cap, resignation with playout fraction), plus
false-positive resignation tracking in the returned meta dict. Subtree reuse
(advance) carries statistics across moves; Dirichlet noise is re-applied at
each new root.
"""
import chess
import numpy as np
from dataclasses import dataclass

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move
from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.goals.assignment import GoalAssigner
from chessrl.mcts.batched import BatchedMCTS
from chessrl.selfplay.feed import NullPublisher
from chessrl.selfplay.play import _SideGoal, _maybe_switch_to_win
from chessrl.selfplay.records import GameRecord, RecordBuilder


class _Game:
    """Mutable per-game state for one slot in the concurrent batch."""

    __slots__ = (
        "tree", "builder", "board", "allow_resign", "resign_streak",
        "ply", "done", "z", "resigned", "would_resign_side", "game_id",
    )

    def __init__(self, tree, board: chess.Board, allow_resign: bool):
        self.tree = tree
        self.builder = RecordBuilder()
        self.board = board
        self.allow_resign = allow_resign
        self.resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.ply = 0
        self.done = False
        self.z = 0
        self.resigned = False
        self.would_resign_side = None   # chess.Color of side that would resign, or None
        self.game_id = ""               # M7: stable per-game topic for the live feed


def play_games_concurrent(
    evaluator_many,
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    rng: np.random.Generator,
    num_games: int,
    publisher=None,
    game_id_prefix: str = "",
) -> list:
    """Returns list[(GameRecord, final_board, z, meta)] of length num_games,
    in slot order. z is from White's perspective (+1/0/-1). If `publisher` is
    given, every applied move is published to the live feed under the per-game
    topic f"{game_id_prefix}{slot}"; a terminal done=True frame is published when
    a game ends."""
    publisher = publisher or NullPublisher()
    mcts = BatchedMCTS(evaluator_many, mcts_cfg, rng)

    games: list[_Game] = []
    for slot in range(num_games):
        board = chess.Board()
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        tree = mcts.init_tree(board, add_noise=True)
        g = _Game(tree, board, allow_resign)
        g.game_id = f"{game_id_prefix}{slot}"          # NEW: stable per-game topic
        games.append(g)

    # Resolve any game that is already terminal / over the cap before searching.
    for g in games:
        _check_pre_move_termination(g, sp_cfg)

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        # Run a full search for every active tree (each tree tops up to
        # mcts_cfg.simulations; step_round shares one GPU batch across trees).
        # Use visit_count-based condition matching batched.py's internal API.
        trees = [g.tree for g in active]
        while any(t.root.visit_count < mcts_cfg.simulations + 1 for t in trees):
            mcts.step_round(trees)

        for g in active:
            _play_one_move(g, mcts, mcts_cfg, sp_cfg, rng, publisher)

    results = []
    for g in games:
        rec = g.builder.finalize(g.z)
        meta = {
            "plies": g.ply,
            "z": g.z,
            "resigned": g.resigned,
            "playout": not g.allow_resign,
            "would_resign": g.would_resign_side is not None,
            "fp": _is_false_positive(g),
        }
        results.append((rec, g.board, g.z, meta))
    return results


def _check_pre_move_termination(g: _Game, sp_cfg: SelfPlayConfig) -> None:
    term = terminal_value(g.board)
    if term is not None:
        g.z = int(term) if g.board.turn == chess.WHITE else -int(term)
        g.done = True
    elif g.ply >= sp_cfg.ply_cap:
        g.z = 0
        g.done = True


def _play_one_move(
    g: _Game, mcts: BatchedMCTS, mcts_cfg: MCTSConfig, sp_cfg: SelfPlayConfig,
    rng: np.random.Generator, publisher,
) -> None:
    visits = mcts.visit_counts(g.tree)
    root_q = mcts.root_q(g.tree)
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.fromiter(visits.values(), dtype=np.float64)
    if g.ply < mcts_cfg.temperature_moves:
        choice = int(rng.choice(idxs, p=counts / counts.sum()))
    else:
        choice = int(idxs[counts.argmax()])

    # Record before the resign check (the triggering search is a valid example).
    g.builder.add(g.board, idxs.astype(np.int32), counts.astype(np.int32), choice)

    # Decode the chosen move BEFORE committing (need the pre-move board context),
    # and build top-5 (uci, visit_frac) from the root visit distribution.
    flip = g.board.turn == chess.BLACK
    chosen_move = index_to_move(choice, flip, g.board)
    total = float(counts.sum())
    order = np.argsort(counts)[::-1][:5]
    top_moves = [
        [index_to_move(int(idxs[k]), flip, g.board).uci(), float(counts[k] / total)]
        for k in order
    ]

    if root_q < sp_cfg.resign_threshold:
        g.resign_streak[g.board.turn] += 1
        if g.resign_streak[g.board.turn] >= sp_cfg.resign_consecutive:
            if g.would_resign_side is None:
                g.would_resign_side = g.board.turn
            if g.allow_resign:
                g.z = -1 if g.board.turn == chess.WHITE else 1
                g.resigned = True
                g.done = True
                _publish_move(publisher, g, chosen_move, root_q, top_moves)
                return
    else:
        g.resign_streak[g.board.turn] = 0

    # Commit the move via subtree reuse (advance pushes the move on tree.board),
    # then keep g.board in sync, re-apply root noise, and check termination.
    mcts.advance(g.tree, choice)
    g.board = g.tree.board
    g.ply += 1
    mcts.add_root_noise(g.tree)
    _check_pre_move_termination(g, sp_cfg)
    _publish_move(publisher, g, chosen_move, root_q, top_moves)


def _publish_move(publisher, g: _Game, chosen_move, root_q: float, top_moves: list) -> None:
    publisher.publish(g.game_id, {
        "game_id": g.game_id,
        "fen": g.board.fen(),
        "last_move_uci": chosen_move.uci(),
        "ply": g.ply,
        "root_q": float(root_q),
        "top_moves": top_moves,
        "done": bool(g.done),
        "z": int(g.z) if g.done else None,
    })


def _is_false_positive(g: _Game) -> bool:
    """A playout game where resignation WOULD have fired but the would-be
    resigner did not actually lose -> a false positive. The would-be resigner
    is whoever was to move when the streak reached the threshold; we approximate
    it conservatively as: playout AND would_resign AND the game was not a loss
    for both sides being impossible -> use the recorded result. Since a resign
    abandons the game as a loss for the side to move, a false positive is a
    playout game that hit the criterion yet ended in a draw or a win for the
    would-be resigner."""
    if g.allow_resign or g.would_resign_side is None:
        return False
    # z is always from White's perspective.
    # A false positive is: the would-be resigner did NOT actually lose.
    # White would-resigner: fp when z >= 0 (draw or White win -> White didn't lose).
    # Black would-resigner: fp when z <= 0 (draw or Black win -> Black didn't lose).
    if g.would_resign_side == chess.WHITE:
        return g.z >= 0
    else:
        return g.z <= 0


# ===========================================================================
# Cluster side-goal helpers for means-end (emergent goal-space) self-play.
# Used by Stage 4b+; the assigner is called once per side at game start and
# the deadline helper is called after every move (idempotent).
# ===========================================================================


@dataclass
class _ClusterSideGoal:
    """One side's cluster goal for a means-end game. ``active_cluster < 0`` means
    the side is pursuing the terminal/extrinsic objective (vec = the net's
    win_vector)."""
    assigned_cluster: int
    assigned_vec: np.ndarray
    deadline: int
    start_ply: int
    active_cluster: int
    active_vec: np.ndarray
    explore: bool = False

    def is_terminal(self) -> bool:
        return self.active_cluster < 0


def assign_cluster_goal(goalspace, win_vector, goal_cfg, rng, curriculum=None) -> _ClusterSideGoal:
    """Pick one side's goal with epsilon-explore + optional win-valued curriculum.

    - If goalspace not ready: terminal.
    - With prob epsilon: uniform-random cluster, explore=True (interventional).
    - Else if curriculum given: curriculum.sample(rng) (-1 -> terminal, else cluster), explore=False.
    - Else (no curriculum): win_floor then uniform cluster, explore=False.
    """
    win_vector = np.asarray(win_vector, np.float32)
    ready = getattr(goalspace, "ready", False) and getattr(goalspace, "centroids", None) is not None
    if not ready:
        return _ClusterSideGoal(-1, win_vector, goal_cfg.deadline_max, 0, -1, win_vector, explore=False)

    def terminal():
        return _ClusterSideGoal(-1, win_vector, goal_cfg.deadline_max, 0, -1, win_vector, explore=False)

    def subgoal(c, explore):
        vec = np.asarray(goalspace.centroid(c), np.float32)
        return _ClusterSideGoal(c, vec, goal_cfg.goal_window, 0, c, vec, explore=explore)

    eps = getattr(goal_cfg, "epsilon", 0.0)
    if rng.random() < eps:                                  # interventional: uniform-random cluster
        return subgoal(int(rng.integers(goalspace.n_clusters)), explore=True)
    if curriculum is not None:                              # win-valued curriculum
        c = curriculum.sample(rng)
        return terminal() if c < 0 else subgoal(c, explore=False)
    # no curriculum: win-floor then uniform
    if rng.random() < goal_cfg.win_floor:
        return terminal()
    return subgoal(int(rng.integers(goalspace.n_clusters)), explore=False)


def maybe_switch_cluster_to_terminal(side: _ClusterSideGoal, ply: int, win_vector, deadline_max: int) -> None:
    """Deadline-based switch: once the goal window elapses, pursue the terminal
    objective for the rest of the game (idempotent)."""
    if side.is_terminal():
        return
    if ply - side.start_ply >= side.deadline:
        side.active_cluster = -1
        side.active_vec = np.asarray(win_vector, np.float32)
        side.deadline = deadline_max


# ===========================================================================
# Goal-conditioned concurrent self-play (spec sec 7, 10).
#
# This is the batched analogue of selfplay/play.py:play_goal_game. It advances
# ``num_games`` goal games in lockstep so that, per ply-round, the leaf
# evaluations across ALL in-flight games are batched into ONE
# BatchedGoalNetEvaluator.evaluate_planes call — each leaf already carries its
# OWN goal planes + deadline (BatchedMCTS leaf-parking), so games pursuing
# DIFFERENT, mid-game-SWITCHING goals batch together with no special-casing.
#
# Per-game semantics mirror play_goal_game EXACTLY:
#   * per-side goal assignment at game start (assigner.assign() per side);
#   * pure pursuit of the side-to-move's ACTIVE goal as protagonist;
#   * temperature sampling before mcts_cfg.temperature_moves, argmax after;
#   * switch-to-win on resolution (_maybe_switch_to_win, shared with play.py);
#   * resignation ONLY while the active goal is the win-goal, on the SAME
#     root-Q-mapped-(2p-1) threshold and consecutive-streak rule;
#   * the SAME record fields (protagonist / assigned / active / visit counts).
#
# Like play_goal_game, each ply rebuilds a FRESH search root with a fresh
# baseline and fresh Dirichlet noise (no subtree reuse): GoalReferenceMCTS
# recomputes the count-delta baseline at every search() call, and the active
# goal can change between plies, so a fresh per-ply tree is the exact mirror.
# ===========================================================================


class _GoalGame:
    """Mutable per-game state for one slot in the concurrent GOAL batch.

    Mirrors play_goal_game's locals: a board, a record builder, the two sides'
    goal state, the verifier state snapshots, the resign streak, and the result.
    """

    __slots__ = (
        "builder", "board", "states", "sides", "allow_resign", "resign_streak",
        "ply", "done", "z", "tree", "game_id", "last_pv",
    )

    def __init__(self, board: chess.Board, sides: dict, allow_resign: bool):
        self.builder = RecordBuilder()
        self.board = board
        self.states = [board.copy()]
        self.sides = sides
        self.allow_resign = allow_resign
        self.resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.ply = 0
        self.done = False
        self.z = 0
        self.tree = None        # the fresh per-ply search tree (rebuilt each round)
        self.game_id = ""        # M7: stable per-game topic for the live feed
        # Each side's most recent root P(achieve), so the live feed can show BOTH
        # sides every frame (the non-moving side's value is its last search).
        self.last_pv = {chess.WHITE: None, chess.BLACK: None}


def play_goal_games_concurrent(
    evaluator_goal,
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    goal_cfg: GoalConfig,
    rng: np.random.Generator,
    num_games: int,
    assigner: GoalAssigner,
    publisher=None,
    game_id_prefix: str = "",
) -> list:
    """Play ``num_games`` goal-conditioned games concurrently, batching leaf
    evaluations across all games into ``evaluator_goal`` (a
    BatchedGoalNetEvaluator). Returns list[(GameRecord, final_board, z, meta)] in
    slot order, the SAME shape as play_games_concurrent, with goal diagnostics in
    meta (win_ply_fraction). Reproduces play_goal_game per game exactly. If
    `publisher` is given, every applied move is published to the live feed under
    the per-game topic f"{game_id_prefix}{slot}"; a terminal done=True frame is
    published when a game ends (a pure side effect — RNG and decisions untouched)."""
    publisher = publisher or NullPublisher()
    # One goal-mode MCTS with NO fixed goal; each tree carries its own context.
    mcts = BatchedMCTS(evaluator_goal, mcts_cfg, rng, goal_mode=True)

    games: list[_GoalGame] = []
    for slot in range(num_games):
        board = chess.Board()
        # RNG draw order matches play_goal_game exactly (allow_resign, then the
        # White-side goal, then the Black-side goal) so a single-game run is
        # bit-comparable to the sequential reference under a matched seed.
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        sides = {
            chess.WHITE: _SideGoal(assigner.assign()),
            chess.BLACK: _SideGoal(assigner.assign()),
        }
        g = _GoalGame(board, sides, allow_resign)
        g.game_id = f"{game_id_prefix}{slot}"          # stable per-game topic
        games.append(g)

    # Resolve any game that is already terminal / over the cap before searching.
    for g in games:
        _goal_check_pre_move_termination(g, sp_cfg)

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        # Build a FRESH per-ply tree for every active game, each with the
        # side-to-move's ACTIVE goal as protagonist (pure pursuit) + root noise.
        for g in active:
            protagonist = g.board.turn
            side = g.sides[protagonist]
            g.tree = mcts.init_tree_for_goal(
                g.board, side.active, protagonist, add_noise=True
            )
        # Run all trees to cfg.simulations, sharing one GPU batch per round.
        trees = [g.tree for g in active]
        while any(t.root.visit_count < mcts_cfg.simulations + 1 for t in trees):
            mcts.step_round(trees)

        for g in active:
            _play_one_goal_move(g, mcts, mcts_cfg, sp_cfg, rng, publisher)

    results = []
    for g in games:
        rec = g.builder.finalize(g.z)
        meta = {
            "plies": len(rec),
            "z": g.z,
            "resigned": False,
            "playout": not g.allow_resign,
            "would_resign": False,
            "fp": False,
            "win_ply_fraction": rec.win_ply_fraction(),
        }
        results.append((rec, g.board, g.z, meta))
    return results


def _goal_check_pre_move_termination(g: _GoalGame, sp_cfg: SelfPlayConfig) -> None:
    """Mirror of play_goal_game's loop-top termination (real result / ply cap)."""
    term = terminal_value(g.board)
    if term is not None:
        g.z = int(term) if g.board.turn == chess.WHITE else -int(term)
        g.done = True
    elif g.ply >= sp_cfg.ply_cap:
        g.z = 0
        g.done = True


def _play_one_goal_move(
    g: _GoalGame, mcts: BatchedMCTS, mcts_cfg: MCTSConfig, sp_cfg: SelfPlayConfig,
    rng: np.random.Generator, publisher,
) -> None:
    """One move for one game: pick (temperature/argmax), record with goal
    columns, apply the win-goal-only resignation gate, push the move, then
    re-check the mover's goal resolution. Exact mirror of play_goal_game's body.
    Publishing is a PURE SIDE EFFECT mirroring _play_one_move's publish points."""
    protagonist = g.board.turn
    side = g.sides[protagonist]
    searched_goal = side.active   # the active goal as searched for THIS move

    visits = mcts.visit_counts(g.tree)
    root_v = mcts.root_q(g.tree)        # protagonist-frame achievement prob at root
    g.last_pv[protagonist] = float(root_v)   # remember for the both-sides live feed
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.fromiter(visits.values(), dtype=np.float64)
    if g.ply < mcts_cfg.temperature_moves:
        choice = int(rng.choice(idxs, p=counts / counts.sum()))
    else:
        choice = int(idxs[counts.argmax()])

    g.builder.add(
        g.board, idxs.astype(np.int32), counts.astype(np.int32), choice,
        protagonist=protagonist,
        assigned_goal=side.assigned,
        active_goal=side.active,
    )

    # Decode the chosen move BEFORE committing (need the pre-move board context),
    # and build top-5 (uci, visit_frac) from the root visit distribution — the
    # same computation as the vanilla _play_one_move.
    flip = g.board.turn == chess.BLACK
    chosen_move = index_to_move(choice, flip, g.board)
    total = float(counts.sum())
    order = np.argsort(counts)[::-1][:5]
    top_moves = [
        [index_to_move(int(idxs[k]), flip, g.board).uci(), float(counts[k] / total)]
        for k in order
    ]

    # Resignation gate: ONLY under the win-goal (mirror of play.py; a hard
    # sub-goal's tiny achievement prob must not resign the GAME).
    if side.active.is_win():
        root_q = 2.0 * root_v - 1.0
        if root_q < sp_cfg.resign_threshold:
            g.resign_streak[protagonist] += 1
            if g.allow_resign and g.resign_streak[protagonist] >= sp_cfg.resign_consecutive:
                g.z = -1 if protagonist == chess.WHITE else 1
                g.done = True
                _publish_goal_move(publisher, g, chosen_move, root_v, top_moves,
                                   protagonist)
                return
        else:
            g.resign_streak[protagonist] = 0
    else:
        g.resign_streak[protagonist] = 0

    g.board.push(index_to_move(choice, protagonist == chess.BLACK, g.board))
    g.ply += 1
    g.states.append(g.board.copy())

    # After the move lands, re-check the mover's goal resolution (achieved or
    # deadline expired -> switch active goal to WIN for the rest of the game).
    _maybe_switch_to_win(side, g.states, protagonist, g.ply)

    _goal_check_pre_move_termination(g, sp_cfg)
    _publish_goal_move(publisher, g, chosen_move, root_v, top_moves, protagonist)


def _goal_aux(g, protagonist):
    """Structured two-side `aux` for a goal-game frame: BOTH sides' assigned goal,
    phase, and last P(achieve), with the side-to-move marked -- so the live view
    shows White and Black at once instead of flip-flopping each ply (the published
    goal used to swap every half-move). The renderer lays out cols x rows
    generically and knows nothing about goals; the meaning lives here only.

    Returns a dict ``{cols, to_move, rows}`` (rows = [label, white_val,
    black_val]); ``[]`` defensively if a side is missing. Phase is abbreviated to
    "pursuing"/"resolved" -- going for the win after a sub-goal resolves is
    implicit, so it is not spelled out; win-floor games show "—"."""
    w = g.sides.get(chess.WHITE)
    b = g.sides.get(chess.BLACK)
    if w is None or b is None:
        return []

    def phase(side):
        if side.assigned.is_win():
            return "—"
        return "pursuing" if not side.active.is_win() else "resolved"

    def pv(color):
        v = g.last_pv.get(color)
        return f"{float(v):.2f}" if v is not None else "—"

    return {
        "cols": ["White", "Black"],
        "to_move": 0 if protagonist == chess.WHITE else 1,
        "rows": [
            ["goal", w.assigned.describe(), b.assigned.describe()],
            ["phase", phase(w), phase(b)],
            ["P(achieve)", pv(chess.WHITE), pv(chess.BLACK)],
        ],
    }


def _publish_goal_move(publisher, g: _GoalGame, chosen_move, root_q: float, top_moves: list,
                       protagonist=None) -> None:
    """Publish one goal-game frame mirroring _publish_move's payload schema, plus
    a structured `aux` describing BOTH sides' assigned goal / phase / P(achieve)
    with the side-to-move marked (see _goal_aux)."""
    publisher.publish(g.game_id, {
        "game_id": g.game_id,
        "fen": g.board.fen(),
        "last_move_uci": chosen_move.uci(),
        "ply": g.ply,
        "root_q": float(root_q),
        "top_moves": top_moves,
        "done": bool(g.done),
        "z": int(g.z) if g.done else None,
        "aux": _goal_aux(g, protagonist) if protagonist is not None else [],
    })


def _meansend_aux(g, protagonist, estimator=None, cluster_labels=None):
    """Structured two-side ``aux`` for a means-end game frame: BOTH sides'
    cluster goal, phase, and optional win-value estimate with the side-to-move
    marked. When ``cluster_labels`` (the post-hoc characterizer's
    {cluster_id: {label, features, n}} map) is available, the goal cell shows the
    chess-feature label inline and ``tips`` carries the per-side delta fingerprint
    for the LIVE view's hover tooltip. Returns ``{cols, to_move, rows, tips}`` or
    ``[]`` defensively if a side is missing."""
    w = g.sides.get(chess.WHITE)
    b = g.sides.get(chess.BLACK)
    if w is None or b is None:
        return []
    labels = cluster_labels or {}

    def goal_str(side):
        if side.is_terminal():
            return "win"
        c = side.active_cluster
        lab = labels.get(c, {}).get("label")
        return f"cluster {c} — {lab}" if lab else f"cluster {c}"

    def phase_str(side):
        return "terminal" if side.is_terminal() else "pursuing"

    def win_val_str(side):
        if estimator is not None and not side.is_terminal():
            return f"{estimator.win_value(side.active_cluster):+.2f}"
        return "—"  # em dash

    def tip(side):
        if side.is_terminal():
            return None
        info = labels.get(side.active_cluster)
        if not info:
            return None
        return {"cluster": side.active_cluster, "label": info.get("label"),
                "features": info.get("features", {})}

    return {
        "cols": ["White", "Black"],
        "to_move": 0 if protagonist == chess.WHITE else 1,
        "rows": [
            ["goal", goal_str(w), goal_str(b)],
            ["phase", phase_str(w), phase_str(b)],
            ["win-value", win_val_str(w), win_val_str(b)],
        ],
        "tips": {"White": tip(w), "Black": tip(b)},
    }


def _publish_meansend_move(publisher, g, chosen_move, root_q: float, top_moves: list,
                           protagonist, estimator=None, cluster_labels=None) -> None:
    """Publish one means-end game frame mirroring _publish_goal_move's payload
    schema, with a structured ``aux`` describing BOTH sides' cluster goal /
    phase / win-value with the side-to-move marked (see _meansend_aux)."""
    publisher.publish(g.game_id, {
        "game_id": g.game_id,
        "fen": g.board.fen(),
        "last_move_uci": chosen_move.uci(),
        "ply": g.ply,
        "root_q": float(root_q),
        "top_moves": top_moves,
        "done": bool(g.done),
        "z": int(g.z) if g.done else None,
        "aux": _meansend_aux(g, protagonist, estimator, cluster_labels),
    })


# ===========================================================================
# Means-end concurrent self-play (v2, Stage 4b).
#
# Advances num_games games in lockstep under a single means-end BatchedMCTS
# (Plan 4a). Each side is assigned a cluster goal (or terminal) at game start
# via assign_cluster_goal; the active goal switches to the terminal objective
# after the deadline elapses (maybe_switch_cluster_to_terminal). Records carry
# Plan 3 cluster columns (cluster_active, cluster_assigned, active_vec).
# ===========================================================================


class _MeansEndGame:
    __slots__ = ("builder", "board", "sides", "explore", "allow_resign",
                 "resign_streak", "ply", "done", "z", "tree", "game_id", "last_pv")

    def __init__(self, board, sides, explore, allow_resign):
        self.builder = RecordBuilder()
        self.board = board
        self.sides = sides
        self.explore = explore
        self.allow_resign = allow_resign
        self.resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.ply = 0
        self.done = False
        self.z = 0
        self.tree = None
        self.game_id = ""
        self.last_pv = {chess.WHITE: None, chess.BLACK: None}


def play_meansend_games_concurrent(
    evaluator_vector, mcts_cfg, sp_cfg, goal_cfg, goalspace, win_vector, rng,
    num_games, explore: bool = False, publisher=None, game_id_prefix: str = "",
    curriculum=None, estimator=None, cluster_labels=None,
) -> list:
    """Means-end concurrent self-play (v2). Each side pursues a discovered cluster
    goal (or the terminal objective) under the Plan 4a means-end MCTS; the switch
    to terminal pursuit is deadline-based. Writes Plan 3 cluster records. Returns
    list[(GameRecord, final_board, z, meta)] in slot order."""
    publisher = publisher or NullPublisher()
    win_vector = np.asarray(win_vector, np.float32)
    if estimator is None and curriculum is not None:
        estimator = getattr(curriculum, "est", None)
    mcts = BatchedMCTS(evaluator_vector, mcts_cfg, rng, meansend=True)

    games: list[_MeansEndGame] = []
    for slot in range(num_games):
        board = chess.Board()
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        sides = {
            chess.WHITE: assign_cluster_goal(goalspace, win_vector, goal_cfg, rng, curriculum),
            chess.BLACK: assign_cluster_goal(goalspace, win_vector, goal_cfg, rng, curriculum),
        }
        g = _MeansEndGame(board, sides, explore, allow_resign)
        g.game_id = f"{game_id_prefix}{slot}"
        games.append(g)

    for g in games:
        _goal_check_pre_move_termination(g, sp_cfg)

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        for g in active:
            side = g.sides[g.board.turn]
            if side.is_terminal():
                remaining = max(1, sp_cfg.ply_cap - g.ply)
            else:
                remaining = max(1, side.deadline - (g.ply - side.start_ply))
            g.tree = mcts.init_tree_for_meansend(g.board, side.active_vec, remaining, add_noise=True)
        trees = [g.tree for g in active]
        while any(t.root.visit_count < mcts_cfg.simulations + 1 for t in trees):
            mcts.step_round(trees)
        for g in active:
            _play_one_meansend_move(g, mcts, mcts_cfg, sp_cfg, goal_cfg, win_vector, rng, publisher, estimator, cluster_labels)

    results = []
    for g in games:
        rec = g.builder.finalize(g.z)
        wpf = float((rec.active_cluster == -1).mean()) if rec.has_cluster_goals() and len(rec) else 0.0
        meta = {"plies": len(rec), "z": g.z, "resigned": False,
                "playout": not g.allow_resign, "would_resign": False, "fp": False,
                "win_ply_fraction": wpf}
        results.append((rec, g.board, g.z, meta))
    return results


def _play_one_meansend_move(g, mcts, mcts_cfg, sp_cfg, goal_cfg, win_vector, rng, publisher, estimator=None, cluster_labels=None) -> None:
    protagonist = g.board.turn
    side = g.sides[protagonist]
    visits = mcts.visit_counts(g.tree)
    root_q = mcts.root_q(g.tree)
    g.last_pv[protagonist] = float(root_q)
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.fromiter(visits.values(), dtype=np.float64)
    if g.ply < mcts_cfg.temperature_moves:
        choice = int(rng.choice(idxs, p=counts / counts.sum()))
    else:
        choice = int(idxs[counts.argmax()])

    g.builder.add(
        g.board, idxs.astype(np.int32), counts.astype(np.int32), choice,
        protagonist=protagonist,
        cluster_active=side.active_cluster, cluster_assigned=side.assigned_cluster,
        active_vec=side.active_vec, explore=side.explore,
    )

    flip = g.board.turn == chess.BLACK
    chosen_move = index_to_move(choice, flip, g.board)
    total = float(counts.sum())
    order = np.argsort(counts)[::-1][:5]
    top_moves = [[index_to_move(int(idxs[k]), flip, g.board).uci(), float(counts[k] / total)] for k in order]

    if side.is_terminal():
        if root_q < sp_cfg.resign_threshold:
            g.resign_streak[protagonist] += 1
            if g.allow_resign and g.resign_streak[protagonist] >= sp_cfg.resign_consecutive:
                g.z = -1 if protagonist == chess.WHITE else 1
                g.done = True
                _publish_meansend_move(publisher, g, chosen_move, root_q, top_moves, protagonist, estimator, cluster_labels)
                return
        else:
            g.resign_streak[protagonist] = 0
    else:
        g.resign_streak[protagonist] = 0

    g.board.push(index_to_move(choice, protagonist == chess.BLACK, g.board))
    g.ply += 1
    maybe_switch_cluster_to_terminal(side, g.ply, win_vector, goal_cfg.deadline_max)
    _goal_check_pre_move_termination(g, sp_cfg)
    _publish_meansend_move(publisher, g, chosen_move, root_q, top_moves, protagonist, estimator, cluster_labels)
