# tests/test_match.py
import chess

from chessrl.evaluation.match import MatchResult, play_pairing, play_single
from chessrl.evaluation.players import GreedyMaterialPlayer, RandomPlayer


def test_play_single_returns_z_and_pgn():
    white = GreedyMaterialPlayer(seed=0)
    black = RandomPlayer(seed=1)
    z, pgn = play_single(white, black, opening_idx=0, max_plies=60)
    assert z in (-1, 0, 1)
    assert isinstance(pgn, str)
    assert "[Result " in pgn


def test_play_single_ply_cap_is_a_draw():
    # Two random players capped very low almost never checkmate; cap -> draw z=0.
    white = RandomPlayer(seed=0)
    black = RandomPlayer(seed=0)
    z, pgn = play_single(white, black, opening_idx=3, max_plies=4)
    assert z == 0
    assert '[Result "1/2-1/2"]' in pgn


def test_play_pairing_requires_even_games():
    import pytest

    with pytest.raises(ValueError):
        play_pairing(RandomPlayer(0), RandomPlayer(1), games=3, openings_start=0, max_plies=10)


def test_play_pairing_alternates_colors_evenly():
    a = GreedyMaterialPlayer(seed=0)
    b = RandomPlayer(seed=1)
    results = play_pairing(a, b, games=4, openings_start=0, max_plies=40)
    assert len(results) == 4
    a_white = sum(1 for r in results if r.white_name == a.name)
    b_white = sum(1 for r in results if r.white_name == b.name)
    assert a_white == b_white == 2
    # Same opening shared by each color-swapped pair.
    assert results[0].opening_idx == results[1].opening_idx
    assert results[2].opening_idx == results[3].opening_idx
    assert results[0].opening_idx != results[2].opening_idx
    for r in results:
        assert isinstance(r, MatchResult)
        assert r.z in (-1, 0, 1)
        assert "[Result " in r.pgn


def test_play_pairing_structural_validity_over_more_games():
    # Greedy vs Random over 8 games: assert structural validity and equal colors
    # (strength ordering is asserted in the ratings integration test, not here, to
    # avoid flakiness from a handful of games).
    a = GreedyMaterialPlayer(seed=0)
    b = RandomPlayer(seed=2)
    results = play_pairing(a, b, games=8, openings_start=5, max_plies=60)
    assert len(results) == 8
    assert sum(r.white_name == a.name for r in results) == 4
    assert sum(r.white_name == b.name for r in results) == 4
    assert {r.opening_idx for r in results} == {5, 6, 7, 8}   # 4 openings, each twice
