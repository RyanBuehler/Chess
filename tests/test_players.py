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


# ---- Hang-stub regression: nodes rung must not block forever ----------------

def test_nodes_rung_does_not_hang(tmp_path):
    """A nodes-only Limit must be time-bounded. This test uses a stub engine that
    speaks just enough UCI to pass the handshake but never replies to 'go', verifying
    that StockfishPlayer raises (rather than hanging forever) within a tight wall-clock
    bound.
    """
    import sys
    import time

    # Write a minimal UCI stub: handles uci + isready, ignores everything else.
    stub = tmp_path / "hang_stub.py"
    stub.write_text(
        # Watchdog: a hung engine ignores 'quit' too, so python-chess cannot
        # shut it down and Windows never reaps the orphan (it then wedges any
        # shell pipeline waiting on the process tree). Self-destruct instead.
        "import os, sys, threading\n"
        "threading.Timer(15, lambda: os._exit(0)).start()\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if line == 'uci':\n"
        "        print('id name HangStub', flush=True)\n"
        "        print('uciok', flush=True)\n"
        "    elif line == 'isready':\n"
        "        print('readyok', flush=True)\n"
        "    # 'go' and 'quit' are silently ignored — engine hangs forever on go\n"
        "os._exit(0)\n",
        encoding="utf-8",
    )

    timeout_s = 0.5
    p = StockfishPlayer(
        [sys.executable, str(stub)],
        nodes=1,
        timeout_s=timeout_s,
        name="hang_stub",
    )
    start = time.monotonic()
    try:
        with pytest.raises(Exception):
            p.play(chess.Board())
    finally:
        # close() may also raise if the stub ignores 'quit' — that's fine,
        # the engine subprocess will be reaped by the OS. Suppress so the
        # timing assertion is always reached.
        try:
            p.close()
        except Exception:
            pass

    elapsed = time.monotonic() - start
    # One play attempt times out, triggers one auto-restart (quit + popen_uci +
    # second play), all bounded by timeout_s each. Total wall time is bounded;
    # 10 seconds is far below "hang forever" but generous enough for slow CI.
    assert elapsed < 10.0, f"play() took {elapsed:.1f}s — engine may have hung"
