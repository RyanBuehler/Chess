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
        "ply", "done", "z", "tree",
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


def play_goal_games_concurrent(
    evaluator_goal,
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    goal_cfg: GoalConfig,
    rng: np.random.Generator,
    num_games: int,
    assigner: GoalAssigner,
) -> list:
    """Play ``num_games`` goal-conditioned games concurrently, batching leaf
    evaluations across all games into ``evaluator_goal`` (a
    BatchedGoalNetEvaluator). Returns list[(GameRecord, final_board, z, meta)] in
    slot order, the SAME shape as play_games_concurrent, with goal diagnostics in
    meta (win_ply_fraction). Reproduces play_goal_game per game exactly."""
    # One goal-mode MCTS with NO fixed goal; each tree carries its own context.
    mcts = BatchedMCTS(evaluator_goal, mcts_cfg, rng, goal_mode=True)

    games: list[_GoalGame] = []
    for _ in range(num_games):
        board = chess.Board()
        # RNG draw order matches play_goal_game exactly (allow_resign, then the
        # White-side goal, then the Black-side goal) so a single-game run is
        # bit-comparable to the sequential reference under a matched seed.
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        sides = {
            chess.WHITE: _SideGoal(assigner.assign()),
            chess.BLACK: _SideGoal(assigner.assign()),
        }
        games.append(_GoalGame(board, sides, allow_resign))

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
            _play_one_goal_move(g, mcts, mcts_cfg, sp_cfg, rng)

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
    rng: np.random.Generator,
) -> None:
    """One move for one game: pick (temperature/argmax), record with goal
    columns, apply the win-goal-only resignation gate, push the move, then
    re-check the mover's goal resolution. Exact mirror of play_goal_game's body."""
    protagonist = g.board.turn
    side = g.sides[protagonist]

    visits = mcts.visit_counts(g.tree)
    root_v = mcts.root_q(g.tree)        # protagonist-frame achievement prob at root
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

    # Resignation gate: ONLY under the win-goal (mirror of play.py; a hard
    # sub-goal's tiny achievement prob must not resign the GAME).
    if side.active.is_win():
        root_q = 2.0 * root_v - 1.0
        if root_q < sp_cfg.resign_threshold:
            g.resign_streak[protagonist] += 1
            if g.allow_resign and g.resign_streak[protagonist] >= sp_cfg.resign_consecutive:
                g.z = -1 if protagonist == chess.WHITE else 1
                g.done = True
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
