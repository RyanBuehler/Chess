"""One self-play game: search -> record -> (maybe resign) -> move.

Two paths:

* ``play_game``       — the legacy vanilla path (negamax ReferenceMCTS), kept
                        byte-for-byte unchanged for ``goal_mode == "none"``.
* ``play_goal_game``  — goal-conditioned self-play (spec sec 7, 10): each side
                        is assigned a goal at game start; the side to move
                        searches its *active* goal as protagonist (pure pursuit);
                        when that goal resolves (achieved or deadline expired)
                        the side switches its active goal to WIN_GOAL for the
                        rest of the game (sequential, not blended). Play always
                        runs to a real chess result / ply cap regardless.
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move
from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.goals.assignment import GoalAssigner
from chessrl.goals.templates import WIN_GOAL, GoalTemplate
from chessrl.goals.verifier import achieved_by_deadline
from chessrl.mcts.reference import GoalReferenceMCTS, ReferenceMCTS
from chessrl.selfplay.records import GameRecord, RecordBuilder


def play_game(evaluator, mcts_cfg: MCTSConfig, sp_cfg: SelfPlayConfig,
              rng: np.random.Generator) -> tuple[GameRecord, chess.Board, int]:
    """Returns (record, final board, z) with z the result from White's
    perspective (+1/0/-1). Resignation per spec: threshold on root Q for
    `resign_consecutive` own moves; a `resign_playout_fraction` of games
    ignores resignation to measure false positives."""
    board = chess.Board()
    builder = RecordBuilder()
    mcts = ReferenceMCTS(evaluator, mcts_cfg, rng)
    allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
    resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
    ply = 0
    while True:
        term = terminal_value(board)
        if term is not None:
            z = int(term) if board.turn == chess.WHITE else -int(term)
            break
        if ply >= sp_cfg.ply_cap:
            z = 0
            break
        visits, root_q = mcts.search(board, add_noise=True)
        idxs = np.fromiter(visits.keys(), dtype=np.int64)
        counts = np.fromiter(visits.values(), dtype=np.float64)
        if ply < mcts_cfg.temperature_moves:
            choice = int(rng.choice(idxs, p=counts / counts.sum()))
        else:
            choice = int(idxs[counts.argmax()])
        # Record before the resign check: the search that triggered resignation
        # is still a valid training example for this position.
        builder.add(board, idxs.astype(np.int32), counts.astype(np.int32), choice)
        if root_q < sp_cfg.resign_threshold:
            resign_streak[board.turn] += 1
            if allow_resign and resign_streak[board.turn] >= sp_cfg.resign_consecutive:
                z = -1 if board.turn == chess.WHITE else 1
                break
        else:
            resign_streak[board.turn] = 0
        board.push(index_to_move(choice, board.turn == chess.BLACK, board))
        ply += 1
    return builder.finalize(z), board, z


class _SideGoal:
    """Per-side goal state for goal-conditioned self-play.

    ``assigned`` is the goal drawn at game start (immutable). ``active`` starts
    equal to ``assigned`` and switches to WIN_GOAL once resolved. ``start_ply``
    is the side's deadline origin (0: both sides begin pursuit at game start)."""

    __slots__ = ("assigned", "active", "start_ply", "resolved")

    def __init__(self, assigned: GoalTemplate):
        self.assigned = assigned
        self.active = assigned
        self.start_ply = 0
        self.resolved = assigned.is_win()  # a win-goal is never "switched" away


def _maybe_switch_to_win(side: _SideGoal, states, protagonist: bool, plies_so_far: int) -> None:
    """If ``side``'s active goal has resolved (achieved by deadline, or deadline
    expired), switch the active goal to WIN_GOAL for the rest of the game.

    ``states`` is the list of board snapshots [start .. current]; ``plies_so_far``
    is how many half-moves the protagonist has had available since start_ply."""
    if side.resolved:
        return
    ok, _ = achieved_by_deadline(states, side.active, protagonist, side.start_ply)
    if ok:
        side.active = WIN_GOAL
        side.resolved = True
        return
    # Deadline expired without achievement: the verifier window has fully
    # elapsed (we have seen >= deadline plies past start_ply) and it was not
    # achieved -> switch to win.
    if plies_so_far - side.start_ply >= side.active.deadline:
        side.active = WIN_GOAL
        side.resolved = True


def play_goal_game(
    evaluator,
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    goal_cfg: GoalConfig,
    rng: np.random.Generator,
    assigner: GoalAssigner,
) -> tuple[GameRecord, chess.Board, int]:
    """Goal-conditioned self-play game (spec sec 7, 10).

    ``evaluator`` must expose ``evaluate(board, goal, remaining, protagonist) ->
    (policy, value)`` (the GoalNetEvaluator interface). Returns (record, final
    board, z) with z from White's perspective. Resignation uses the same root-Q
    threshold; root Q here is the protagonist-frame achievement probability
    mapped to [-1,1] via ``2p-1`` so the existing threshold semantics hold for
    the win-goal (the only goal eval depends on)."""
    board = chess.Board()
    builder = RecordBuilder()
    mcts = GoalReferenceMCTS(evaluator, mcts_cfg, rng)
    allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
    resign_streak = {chess.WHITE: 0, chess.BLACK: 0}

    sides = {
        chess.WHITE: _SideGoal(assigner.assign()),
        chess.BLACK: _SideGoal(assigner.assign()),
    }
    states = [board.copy()]   # board snapshots for the verifier (index == ply)
    ply = 0
    while True:
        term = terminal_value(board)
        if term is not None:
            z = int(term) if board.turn == chess.WHITE else -int(term)
            break
        if ply >= sp_cfg.ply_cap:
            z = 0
            break

        protagonist = board.turn
        side = sides[protagonist]
        # The side searches its currently-active goal as protagonist (pure
        # pursuit: only this goal is searched).
        visits, root_v = mcts.search(board, side.active, protagonist, add_noise=True)
        idxs = np.fromiter(visits.keys(), dtype=np.int64)
        counts = np.fromiter(visits.values(), dtype=np.float64)
        if ply < mcts_cfg.temperature_moves:
            choice = int(rng.choice(idxs, p=counts / counts.sum()))
        else:
            choice = int(idxs[counts.argmax()])

        builder.add(
            board, idxs.astype(np.int32), counts.astype(np.int32), choice,
            protagonist=protagonist,
            assigned_goal=side.assigned,
            active_goal=side.active,
        )

        # Resignation uses the protagonist-frame achievement prob mapped to
        # negamax scale (2p-1) so the legacy threshold semantics carry over.
        root_q = 2.0 * root_v - 1.0
        if root_q < sp_cfg.resign_threshold:
            resign_streak[protagonist] += 1
            if allow_resign and resign_streak[protagonist] >= sp_cfg.resign_consecutive:
                z = -1 if protagonist == chess.WHITE else 1
                break
        else:
            resign_streak[protagonist] = 0

        board.push(index_to_move(choice, protagonist == chess.BLACK, board))
        ply += 1
        states.append(board.copy())

        # After the move lands, re-check the mover's goal resolution against the
        # accumulated states (achieved or deadline expired -> switch to win).
        _maybe_switch_to_win(side, states, protagonist, ply)

    return builder.finalize(z), board, z
