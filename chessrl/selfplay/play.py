"""One self-play game: search -> record -> (maybe resign) -> move."""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move
from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.mcts.reference import ReferenceMCTS
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
