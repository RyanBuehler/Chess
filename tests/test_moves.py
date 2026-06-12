import chess
import pytest

from chessrl.chess_env.moves import (
    NUM_ACTIONS,
    index_to_move,
    legal_move_mask,
    move_to_index,
)

ROUND_TRIP_FENS = [
    chess.STARTING_FEN,
    # white promotions incl. captures and underpromotions
    "1n4k1/P7/8/8/8/8/8/4K3 w - - 0 1",
    # black promotions (flip path)
    "4k3/8/8/8/8/8/p7/1N4K1 b - - 0 1",
    # both castling rights, both sides
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    # middlegame with sliding pieces in all directions
    "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P1B2/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
]


def test_num_actions():
    assert NUM_ACTIONS == 4672


@pytest.mark.parametrize("fen", ROUND_TRIP_FENS)
def test_round_trip_all_legal_moves(fen):
    board = chess.Board(fen)
    flip = board.turn == chess.BLACK
    for mv in board.legal_moves:
        idx = move_to_index(mv, flip)
        assert 0 <= idx < NUM_ACTIONS
        assert index_to_move(idx, flip, board) == mv


def test_en_passant_round_trip():
    board = chess.Board()
    for uci in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        board.push(chess.Move.from_uci(uci))
    ep = chess.Move.from_uci("e5d6")
    assert ep in board.legal_moves
    idx = move_to_index(ep, flip=False)
    assert index_to_move(idx, False, board) == ep


@pytest.mark.parametrize("fen", ROUND_TRIP_FENS)
def test_indices_unique_per_position(fen):
    board = chess.Board(fen)
    flip = board.turn == chess.BLACK
    idxs = [move_to_index(m, flip) for m in board.legal_moves]
    assert len(idxs) == len(set(idxs))


@pytest.mark.parametrize("fen", ROUND_TRIP_FENS)
def test_legal_move_mask(fen):
    board = chess.Board(fen)
    mask = legal_move_mask(board)
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.sum() == board.legal_moves.count()
