"""Game-termination helper shared by MCTS and self-play."""
import chess


def terminal_value(board: chess.Board) -> float | None:
    """None if the game is ongoing; otherwise the outcome from the
    perspective of the side to move (+1 win, 0 draw, -1 loss).
    Claimable draws (threefold, fifty-move) count as terminal draws."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
