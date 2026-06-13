import chess

from chessrl.goals.templates import GoalTemplate, WIN_GOAL
from chessrl.goals.verifier import achieved_by_deadline


# --------------------------------------------------------------------------
# replay helpers: build a board-state list (index 0 == start position, each
# subsequent index == the board after one half-move).
# --------------------------------------------------------------------------
def replay(pgn_moves: str, start: chess.Board | None = None):
    """Build a state list from a compact SAN string like '1.e4 d5 2.exd5'."""
    board = (start or chess.Board()).copy()
    states = [board.copy()]
    for tok in pgn_moves.split():
        # strip move numbers like '1.' or '12.'
        if "." in tok:
            tok = tok.split(".", 1)[1]
        if not tok:
            continue
        board.push_san(tok)
        states.append(board.copy())
    return states


def replay_uci(ucis, start: chess.Board | None = None):
    board = (start or chess.Board()).copy()
    states = [board.copy()]
    for u in ucis:
        board.push(chess.Move.from_uci(u))
        states.append(board.copy())
    return states


def replay_promo():
    # White pawn one step from promotion; promote to a queen reaching rank 8.
    start = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    return replay_uci(["a7a8q"], start=start)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_capture_within_deadline():
    states = replay("1.e4 d5 2.exd5")            # helper builds board list
    g = GoalTemplate.capture(chess.PAWN, deadline=3)
    ok, ply = achieved_by_deadline(states, g, protagonist=chess.WHITE, start_ply=0)
    # exd5 captures Black's pawn; first visible at state index 3 (0-based).
    # (Plan comment said 4; that is an off-by-one -- the capture is at index 3.)
    assert ok and ply == 3


def test_missed_deadline_is_failure():
    g = GoalTemplate.capture(chess.QUEEN, deadline=2)
    ok, _ = achieved_by_deadline(replay("1.e4 d5 2.exd5"), g, chess.WHITE, 0)
    assert ok is False                            # no queen captured at all


def test_capture_after_deadline_window_is_failure():
    # The pawn capture happens at index 3; a deadline of 2 plies from start
    # must reject it (capture is outside the window).
    states = replay("1.e4 d5 2.exd5")
    g = GoalTemplate.capture(chess.PAWN, deadline=2)
    ok, ply = achieved_by_deadline(states, g, chess.WHITE, 0)
    assert ok is False and ply is None


def test_promotion_reaches_rank8():
    g = GoalTemplate.reach_rank(chess.PAWN, rank=7, deadline=99)  # rank index 7 == 8th rank
    ok, _ = achieved_by_deadline(replay_promo(), g, chess.WHITE, 0)
    assert ok


def test_protagonist_relative_capture():
    # Black captures White's queen; the goal "White captures a pawn" must NOT
    # fire for the Black-side delta, and the Black-protagonist capture goal must.
    states = replay("1.e4 d5 2.Nf3 dxe4")        # ...dxe4: Black captures White's e-pawn
    g_white = GoalTemplate.capture(chess.PAWN, deadline=10)
    ok_w, _ = achieved_by_deadline(states, g_white, protagonist=chess.WHITE, start_ply=0)
    assert ok_w is False                          # White lost a pawn, did not capture one
    ok_b, ply_b = achieved_by_deadline(states, g_white, protagonist=chess.BLACK, start_ply=0)
    assert ok_b and ply_b == 4                    # Black captured White's pawn at index 4


def test_check_goal():
    states = replay("1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7")  # Qxf7 is checkmate -> in check
    g = GoalTemplate.check(deadline=20)
    ok, _ = achieved_by_deadline(states, g, protagonist=chess.WHITE, start_ply=0)
    assert ok


def test_castle_goal():
    states = replay("1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.O-O")  # White castles kingside
    g = GoalTemplate.castle(deadline=20)
    ok, _ = achieved_by_deadline(states, g, protagonist=chess.WHITE, start_ply=0)
    assert ok
    ok_b, _ = achieved_by_deadline(states, g, protagonist=chess.BLACK, start_ply=0)
    assert ok_b is False                          # Black has not castled


def test_win_goal_on_checkmate():
    # Scholar's mate: White checkmates Black.
    states = replay("1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7")
    ok, _ = achieved_by_deadline(states, WIN_GOAL, protagonist=chess.WHITE, start_ply=0)
    assert ok
    ok_b, _ = achieved_by_deadline(states, WIN_GOAL, protagonist=chess.BLACK, start_ply=0)
    assert ok_b is False                          # Black lost


def test_start_ply_offset():
    # Deadline measured from start_ply: capture at index 3, start_ply=2 -> elapsed 1.
    states = replay("1.e4 d5 2.exd5")
    g = GoalTemplate.capture(chess.PAWN, deadline=1)
    ok, ply = achieved_by_deadline(states, g, chess.WHITE, start_ply=2)
    assert ok and ply == 3
    # but from start_ply=0 a deadline of 1 is too tight
    ok2, _ = achieved_by_deadline(states, g, chess.WHITE, start_ply=0)
    assert ok2 is False
