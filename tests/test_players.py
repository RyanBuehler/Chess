# tests/test_players.py
from pathlib import Path

import chess
import pytest

from chessrl.evaluation.players import (
    GreedyMaterialPlayer,
    MinimaxPlayer,
    RandomPlayer,
    StockfishPlayer,
    default_stockfish_path,
)


def _legal(player, board):
    mv = player.play(board)
    assert mv in board.legal_moves
    return mv


def test_random_player_plays_legal_and_is_seeded():
    b = chess.Board()
    p1 = RandomPlayer(seed=0)
    p2 = RandomPlayer(seed=0)
    assert p1.name == "random"
    assert _legal(p1, b) == _legal(p2, b)   # same seed -> same move


def test_greedy_takes_free_queen():
    # White to move; Black queen on d5 is hanging to the pawn on e4 (exd5).
    b = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    p = GreedyMaterialPlayer(seed=0)
    assert p.name == "greedy"
    assert p.play(b) == chess.Move.from_uci("e4d5")


def test_greedy_prefers_mate_in_one():
    # Back-rank mate: Ra8#.
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    p = GreedyMaterialPlayer(seed=0)
    assert p.play(b) == chess.Move.from_uci("a1a8")


def test_minimax_takes_free_queen_depth2():
    b = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    p = MinimaxPlayer(depth=2, seed=0)
    assert p.name == "minimax2"
    assert p.play(b) == chess.Move.from_uci("e4d5")


def test_minimax_avoids_hanging_its_queen_depth2():
    # White Qd1; if White plays Qd5?? then exd5 wins the queen. A depth-2 search
    # must NOT walk the queen onto d5 (the pawn defends d5). Any non-blunder is fine;
    # assert the move is legal and is not the self-hanging Qd5.
    b = chess.Board("4k3/8/8/8/4p3/8/8/3QK3 w - - 0 1")
    p = MinimaxPlayer(depth=2, seed=0)
    mv = p.play(b)
    assert mv in b.legal_moves
    assert mv != chess.Move.from_uci("d1d5")


def test_minimax_is_seeded_deterministic():
    b = chess.Board()
    a = MinimaxPlayer(depth=2, seed=3).play(b)
    c = MinimaxPlayer(depth=2, seed=3).play(b)
    assert a == c


# ---- Stockfish (skipped when the binary is absent) -------------------------

def _stockfish_available() -> bool:
    return default_stockfish_path() is not None


@pytest.mark.skipif(not _stockfish_available(), reason="stockfish binary not provisioned")
def test_stockfish_plays_legal_and_records_conditions():
    path = default_stockfish_path()
    p = StockfishPlayer(str(path), elo=1320, movetime_ms=50, name="sf1320")
    try:
        b = chess.Board()
        mv = p.play(b)
        assert mv in b.legal_moves
        cond = p.conditions()
        assert cond["Threads"] == 1
        assert cond["UCI_Elo"] == 1320
        assert "engine_id" in cond and cond["engine_id"]
    finally:
        p.close()


@pytest.mark.skipif(not _stockfish_available(), reason="stockfish binary not provisioned")
def test_stockfish_nodes_rung_plays_legal():
    path = default_stockfish_path()
    p = StockfishPlayer(str(path), nodes=100, name="sf_nodes100")
    try:
        mv = p.play(chess.Board())
        assert mv in chess.Board().legal_moves
        assert p.conditions()["nodes"] == 100
    finally:
        p.close()
