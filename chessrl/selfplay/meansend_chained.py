"""v3-zenith: chained means-end self-play. ENTIRELY SEGREGATED from v2's
selfplay/concurrent.py — v2 (goal_mode 'emergent') never imports this module.

Each side always holds an active cluster sub-goal, re-selected on achieve-or-expire
via a greedy curriculum_weight(g)*v_goal(s,g) score; goal-influence (the means-end
leaf alpha) fades to 0 as the position becomes decisive (alpha_schedule). 'Win as
apex' emerges from alpha->0; there is no discrete terminal goal during play."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess

from chessrl.chess_env.encoding import encode_board, to_model_input


def alpha_schedule(v_win: float, alpha_max: float, win_ramp: float,
                   ply: int, ply_cap: int, endgame_margin: int) -> float:
    """Goal-influence weight for the means-end leaf. Full (alpha_max) in unclear
    positions, -> 0 as |v_win| -> win_ramp (decisive either way) or near ply_cap."""
    if ply >= ply_cap - endgame_margin:
        return 0.0
    decisiveness = min(1.0, abs(float(v_win)) / max(win_ramp, 1e-6))
    return float(alpha_max) * (1.0 - decisiveness)


def select_next_goal(board, goalspace, curriculum, evaluator, goal_cfg, rng):
    """Greedy state-dependent next sub-goal: softmax over
    curriculum_weight(g) * v_goal(board, centroid_g). With prob epsilon, inject a
    uniform-random cluster (interventional). Requires goalspace.ready (caller
    guards). Returns (cluster_id >= 0, centroid_vec, explored) where ``explored``
    is THE SAME single epsilon draw that decides the pick — so the recorded
    explore flag matches the actual exploration decision (the win-value estimator
    is only de-confounded if these coincide)."""
    K = goalspace.n_clusters
    cents = np.asarray(goalspace.centroids, np.float32)            # (K, d)
    if rng.random() < getattr(goal_cfg, "epsilon", 0.0):
        c = int(rng.integers(K))
        return c, cents[c].astype(np.float32), True
    planes = to_model_input(encode_board(board))
    planes_batch = np.repeat(planes[None, ...], K, axis=0)
    deadlines = np.full(K, goal_cfg.goal_window, np.float32)
    _, _v_win, v_goal = evaluator.evaluate_planes(planes_batch, cents, deadlines)
    w = np.array([curriculum.weight(c) for c in range(K)], np.float64)
    score = w * np.asarray(v_goal, np.float64)
    temp = max(float(goal_cfg.goal_select_temp), 1e-6)
    logits = score / temp
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    c = int(rng.choice(K, p=p))
    return c, cents[c].astype(np.float32), False


@dataclass
class _ChainSideGoal:
    active_cluster: int
    active_vec: np.ndarray
    start_ply: int
    start_emb: np.ndarray      # frozen-encoder embedding of the state where this goal began
    explore: bool = False

    def is_terminal(self) -> bool:
        # active_cluster < 0 = pre-fit bootstrap (pure win pursuit until the goalspace
        # fits). When fit, the chain always holds a real cluster; win emerges via alpha->0.
        return self.active_cluster < 0


def assign_chain_goal(board, ply, goalspace, curriculum, evaluator, goal_cfg, rng) -> _ChainSideGoal:
    """Single entry that selects the next sub-goal, stamps the explore flag from
    the SAME draw that made the pick (so the win-value estimator credits the truly
    interventional segments), and caches the start-state embedding."""
    c, vec, explored = select_next_goal(board, goalspace, curriculum, evaluator, goal_cfg, rng)
    start_emb = np.asarray(evaluator.embed_boards([board])[0], np.float32)
    return _ChainSideGoal(c, vec, ply, start_emb, explored)


def goal_achieved_live(side, board, goalspace, evaluator) -> bool:
    if side.active_cluster < 0:
        return False   # terminal bootstrap is never "achieved"
    emb = np.asarray(evaluator.embed_boards([board])[0], np.float32)
    return bool(goalspace.achieved((emb - side.start_emb).astype(np.float32), side.active_cluster))


def maybe_reassign_goal(side, board, ply, goalspace, curriculum, evaluator, goal_cfg, rng):
    """Reassign (new _ChainSideGoal) when the active sub-goal is achieved (live)
    OR has been pursued for goal_window plies; else return the same side-goal.
    A terminal bootstrap side (pre-fit, cluster < 0) stays terminal."""
    if side.is_terminal() or curriculum is None or not getattr(goalspace, "ready", False):
        return side
    expired = (ply - side.start_ply) >= goal_cfg.goal_window
    achieved = goal_achieved_live(side, board, goalspace, evaluator)
    if expired or achieved:
        return assign_chain_goal(board, ply, goalspace, curriculum, evaluator, goal_cfg, rng)
    return side


def _terminal_side(win_vector, ply) -> _ChainSideGoal:
    """Pre-fit bootstrap: pure win pursuit (cluster -1) until the goalspace fits."""
    d = int(np.asarray(win_vector).shape[0])
    return _ChainSideGoal(-1, np.asarray(win_vector, np.float32), ply,
                          np.zeros(d, np.float32), explore=False)


def play_meansend_chained_games_concurrent(
    evaluator_vector, mcts_cfg, sp_cfg, goal_cfg, goalspace, win_vector, rng,
    num_games, *, curriculum=None, explore: bool = False, publisher=None,
    game_id_prefix: str = "", estimator=None, cluster_labels=None,
) -> list:
    """v3-zenith CHAINED means-end self-play (segregated from v2). Each side chains
    cluster sub-goals (achieve-or-expire reassignment, greedy selection); the
    means-end leaf alpha follows alpha_schedule(root v_win) so goal-influence fades
    to pure win-pursuit as the position becomes decisive. Returns the same shape as
    play_meansend_games_concurrent: list[(GameRecord, board, z, meta)]."""
    from chessrl.selfplay.concurrent import _MeansEndGame, NullPublisher
    from chessrl.mcts.batched import BatchedMCTS

    publisher = publisher or NullPublisher()
    win_vector = np.asarray(win_vector, np.float32)
    if estimator is None and curriculum is not None:
        estimator = getattr(curriculum, "est", None)
    mcts = BatchedMCTS(evaluator_vector, mcts_cfg, rng, meansend=True)
    ready = bool(getattr(goalspace, "ready", False)) and curriculum is not None

    def make_side(board, ply):
        if not ready:
            return _terminal_side(win_vector, ply)
        return assign_chain_goal(board, ply, goalspace, curriculum, evaluator_vector, goal_cfg, rng)

    games: list = []
    for slot in range(num_games):
        board = chess.Board()
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        sides = {chess.WHITE: make_side(board, 0), chess.BLACK: make_side(board, 0)}
        g = _MeansEndGame(board, sides, explore, allow_resign)
        g.game_id = f"{game_id_prefix}{slot}"
        games.append(g)

    from chessrl.selfplay.concurrent import _goal_check_pre_move_termination
    for g in games:
        _goal_check_pre_move_termination(g, sp_cfg)

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        # Per-ply alpha = alpha_schedule(root v_win). Batch the v_win eval over ALL
        # active non-terminal games in one call (review I1) rather than one-per-game.
        rem_by_id, alpha_by_id = {}, {}
        nonterm = []
        for g in active:
            side = g.sides[g.board.turn]
            if side.is_terminal():
                rem_by_id[id(g)] = max(1, sp_cfg.ply_cap - g.ply)
                alpha_by_id[id(g)] = 0.0
            else:
                rem = max(1, goal_cfg.goal_window - (g.ply - side.start_ply))
                rem_by_id[id(g)] = rem
                nonterm.append((g, rem))
        if nonterm:
            planes_batch = np.stack([to_model_input(encode_board(g.board)) for g, _ in nonterm])
            rems = np.asarray([rem for _, rem in nonterm], np.float32)
            vwins = evaluator_vector.win_value(planes_batch, rems)
            for (g, _rem), vwin in zip(nonterm, vwins):
                alpha_by_id[id(g)] = alpha_schedule(
                    float(vwin), mcts_cfg.meansend_alpha, goal_cfg.win_ramp,
                    g.ply, sp_cfg.ply_cap, goal_cfg.endgame_margin)
        for g in active:
            side = g.sides[g.board.turn]
            g.tree = mcts.init_tree_for_meansend(
                g.board, side.active_vec, rem_by_id[id(g)], add_noise=True,
                meansend_alpha=alpha_by_id[id(g)])
        trees = [g.tree for g in active]
        while any(t.root.visit_count < mcts_cfg.simulations + 1 for t in trees):
            mcts.step_round(trees)
        for g in active:
            _play_one_chained_move(g, mcts, mcts_cfg, sp_cfg, goal_cfg, goalspace,
                                   curriculum, win_vector, evaluator_vector, rng,
                                   publisher, estimator, cluster_labels)

    results = []
    for g in games:
        rec = g.builder.finalize(g.z)
        wpf = float((rec.active_cluster == -1).mean()) if rec.has_cluster_goals() and len(rec) else 0.0
        meta = {"plies": len(rec), "z": g.z, "resigned": False,
                "playout": not g.allow_resign, "would_resign": False, "fp": False,
                "win_ply_fraction": wpf}
        results.append((rec, g.board, g.z, meta))
    return results


def _play_one_chained_move(g, mcts, mcts_cfg, sp_cfg, goal_cfg, goalspace, curriculum,
                           win_vector, evaluator, rng, publisher, estimator, cluster_labels):
    from chessrl.selfplay.concurrent import _publish_meansend_move, _goal_check_pre_move_termination
    from chessrl.chess_env.moves import index_to_move

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
        cluster_active=side.active_cluster, cluster_assigned=side.active_cluster,
        active_vec=side.active_vec, explore=side.explore,
    )

    flip = g.board.turn == chess.BLACK
    chosen_move = index_to_move(choice, flip, g.board)
    total = float(counts.sum())
    order = np.argsort(counts)[::-1][:5]
    top_moves = [[index_to_move(int(idxs[k]), flip, g.board).uci(), float(counts[k] / total)] for k in order]

    # Resign only when effectively in win-pursuit (alpha near 0 = decisive position).
    alpha = g.tree.meansend_alpha if g.tree.meansend_alpha is not None else mcts_cfg.meansend_alpha
    if alpha <= goal_cfg.alpha_resign_gate:
        if root_q < sp_cfg.resign_threshold:
            g.resign_streak[protagonist] += 1
            if g.allow_resign and g.resign_streak[protagonist] >= sp_cfg.resign_consecutive:
                g.z = -1 if protagonist == chess.WHITE else 1
                g.done = True
                _publish_meansend_move(publisher, g, chosen_move, root_q, top_moves,
                                       protagonist, estimator, cluster_labels)
                return
        else:
            g.resign_streak[protagonist] = 0
    else:
        g.resign_streak[protagonist] = 0

    g.board.push(index_to_move(choice, protagonist == chess.BLACK, g.board))
    g.ply += 1
    g.sides[protagonist] = maybe_reassign_goal(
        side, g.board, g.ply, goalspace, curriculum, evaluator, goal_cfg, rng)
    _goal_check_pre_move_termination(g, sp_cfg)
    _publish_meansend_move(publisher, g, chosen_move, root_q, top_moves,
                           protagonist, estimator, cluster_labels)
