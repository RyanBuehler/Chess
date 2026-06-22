import numpy as np, chess
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import move_to_index
from chessrl.goals.winvalue import WinValueEstimator
from chessrl.training.parallel_loop import update_winvalue_from_record


def _explore_game(white_cluster=2, black_cluster=5, z_white=1, n=4):
    b = RecordBuilder(); board = chess.Board()
    wv = np.zeros(4, np.float32)
    for ply in range(n):
        mv = list(board.legal_moves)[0]; idx = move_to_index(mv, board.turn == chess.BLACK)
        c = white_cluster if board.turn == chess.WHITE else black_cluster
        b.add(board, [idx], [1], idx, protagonist=board.turn,
              cluster_active=c, cluster_assigned=c, active_vec=wv, explore=True)
        board.push(mv)
    return b.finalize(z_white=z_white)


def test_update_winvalue_credits_winner_side():
    est = WinValueEstimator()
    update_winvalue_from_record(est, _explore_game(white_cluster=2, black_cluster=5, z_white=1))
    # White won (z=1): White's cluster 2 gets a win; Black's cluster 5 gets a loss
    assert est.attempts(2) == 1 and est.attempts(5) == 1
    assert est.win_value(2) > est.win_value(5)


def test_update_skips_non_explore():
    est = WinValueEstimator()
    b = RecordBuilder(); board = chess.Board()
    mv = list(board.legal_moves)[0]; idx = move_to_index(mv, False)
    b.add(board, [idx], [1], idx, protagonist=chess.WHITE, cluster_active=1, cluster_assigned=1,
          active_vec=np.zeros(4, np.float32), explore=False)
    update_winvalue_from_record(est, b.finalize(z_white=1))
    assert est.attempts(1) == 0   # non-explore -> no update
