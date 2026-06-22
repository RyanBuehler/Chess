import chess
from chessrl.config.config import NetworkConfig
from chessrl.model.network import PolicyValueNet
from chessrl.training.trainer import Trainer
from chessrl.config.config import TrainingConfig


def _save_vector_ckpt(tmp_path):
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16, goal_cond="vector"), goal_conditioned=True)
    tr = Trainer(net, TrainingConfig(device="cpu"), run_dir=str(tmp_path))
    return tr.save_checkpoint(), NetworkConfig(blocks=2, filters=16, goal_cond="vector")


def test_vector_player_plays_legal_move(tmp_path):
    from chessrl.evaluation.players import VectorGoalMCTSPlayer
    ckpt, ncfg = _save_vector_ckpt(tmp_path)
    p = VectorGoalMCTSPlayer("v2@0", ckpt, ncfg, simulations=8, device="cpu")
    board = chess.Board()
    mv = p.play(board)
    assert mv in board.legal_moves
    assert hasattr(p, "_last_root_q") and isinstance(p._last_thoughts, list)
