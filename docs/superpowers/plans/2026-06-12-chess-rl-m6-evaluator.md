# Chess RL Elo Evaluator Daemon + Ladder (M6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Elo-over-training-time curve automatic. Build a roster of opponents (random / greedy / minimax floors plus pinned Stockfish anchors), a deterministic match runner over a 50-position opening book, a single-writer `ladder.sqlite` results store with a JSON `ladder_inbox` ingest path, a regularized Davidson draw-model rating fit (anchors pinned), and an evaluator daemon that polls `runs/<run-id>/checkpoints/`, plays every Nth checkpoint against the ladder, refits ratings over the whole store, and appends an `elo.jsonl` curve per run.

**Architecture:** Extends the M1–M5 core (`docs/superpowers/specs/2026-06-11-chess-rl-design.md`). The evaluator is a **separate, synchronous, single-process daemon** (no threads) that mirrors AlphaZero's actor/learner/**evaluator** separation. It is the **sole writer** of `runs/ladder.sqlite` (WAL + `busy_timeout`); other processes (the server arena, M7) drop JSON result files into `runs/ladder_inbox/` which the evaluator ingests and deletes. Run dirs are read-only to everyone except the trainer; the evaluator writes only `runs/ladder.sqlite` and per-run sidecars under `runs/<run-id>/` (`eval_games/`, `elo.jsonl`). All players are duck-typed (`.name`, `.play(board) -> chess.Move`, optional `.close()`); the agent player wraps the existing `ReferenceMCTS` + `BatchedNetEvaluator` with **noise and temperature OFF** (per the spec: noise/temperature are per-context flags, always off in evaluation). Nothing in M1–M5 changes except one additive `EvalConfig` field on `RunConfig`.

**Tech Stack:** Python 3.11+, python-chess (1.11.2, incl. `chess.engine.SimpleEngine` for UCI/Stockfish), PyTorch (CPU by default for evaluation — the evaluator time-slices the GPU against training, so `NetMCTSPlayer` defaults to `device="cpu"`), NumPy (rating-fit gradient ascent), SQLite via stdlib `sqlite3` (WAL mode, `busy_timeout`), PyYAML, pytest. Stockfish 17.1 (Windows AVX2 build) provisioned by a fetch script into `tools/stockfish/` (gitignored).

**Scope:** Milestone M6 of the spec: the evaluation roster, the 50-position opening suite, the deterministic match protocol with exactly-equal colors, the single-writer ladder store + inbox ingest, the regularized Davidson rating fit with pinned anchors, and the evaluator daemon (`--once` and `watch` modes) with a CLI. **Out of scope for M6 (stated so workers don't gold-plate):** the FastAPI web server, the four UI views, and the ZeroMQ live feed (all M7 — the evaluator only *produces* `elo.jsonl` and PGNs the UI will later read); the server-side arena that *writes* to `ladder_inbox` (M7 — M6 only *ingests* the inbox and a test fabricates inbox files directly); checkpoint retention/pruning policy (config-only mention in the spec, deferred); TensorBoard mirroring; the higher Stockfish anchors `UCI_Elo ∈ {2000, 2300, 2700}` (the default roster pins 1320/1500/1700 — the dead-zone-filling rungs the spec calls out; higher anchors are config-extendable but not enabled by default since a sub-1700 agent never scores against them and they only cost wall time).

**Conventions used throughout (normative, from the spec and M1–M5 code — do not drift):**
- **Player interface (duck-typed):** every player has `.name: str` and `.play(board: chess.Board) -> chess.Move` returning a legal move for `board`, plus optional `.close()` (engines quit their subprocess; pure-Python players omit it). Match code calls `getattr(player, "close", lambda: None)()` in a `finally`.
- **Result sign `z`:** every stored result is **from White's perspective**: `+1` White win, `0` draw, `-1` Black win. The rating fit consumes `(white_name, black_name, z)` triples; color bookkeeping lives entirely in the match runner (exactly-equal colors per pairing), never in ratings.
- **No evaluation noise/temperature:** `NetMCTSPlayer` calls `ReferenceMCTS.search(board, add_noise=False)` and plays **argmax visits** (no temperature sampling). This is the spec's "no evaluation-time temperature: it would change the thing being measured."
- **Anchors are pinned, floors are fitted:** only Stockfish `UCI_Elo` rungs are pinned at their nominal Elo (`players.anchor_elo` set). Random/Greedy/Minimax/`nodes`-limited Stockfish rungs are **unpinned** and get fitted ratings. The fit regularizes unpinned ratings toward a `1000` prior mean (floor calibration; documented in `ratings.py`).
- **Single-writer SQLite:** the evaluator is the only writer of `runs/ladder.sqlite`. The store opens with `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`. All writes go through context-managed connections. `check_same_thread` default is fine (one process, no threads).
- **Stockfish pinning:** `Threads=1`, `Ponder=false`; anchors add `UCI_LimitStrength=true` + `UCI_Elo=<n>`; `nodes`-rungs use a fixed node limit; otherwise `movetime` in ms. Every Stockfish result records a `conditions` JSON including the engine `id` (version string) and the pinned options — an unpinned anchor silently moves across releases.
- **Determinism:** every player takes a seed; the match runner advances the opening index deterministically. `pathlib.Path` only. Run all commands from `C:\Chess` with the venv: `.venv\Scripts\python -m pytest ...`. Every test forces CPU.
- **Stockfish-dependent tests** are `@pytest.mark.skipif(not _stockfish_available(), ...)` — they no-op on machines without the binary. The live-gate task provisions the binary and runs them for real.

**Definition of done for M6:** the new test files pass on CPU (`test_config_m6.py`, `test_players.py`, `test_openings.py`, `test_match.py`, `test_store.py`, `test_ratings.py`, `test_daemon.py`), the Stockfish-gated tests pass live after provisioning, `scripts/evaluate.py --once` evaluates a real checkpoint against the floor ladder and writes `elo.jsonl` + PGNs + `ladder.sqlite` rows, all M1–M5 tests still pass unchanged, and the default suite excludes any `slow` integration. The milestone's "Elo fit vs hand-computed cases (including the all-wins regularization behavior)" unit requirement is satisfied by `test_ratings.py`.

---

### Task 1: Config additions for M6 (`EvalConfig`)

**Files:**
- Modify: `chessrl/config/config.py`
- Test: `tests/test_config_m6.py`

Add a frozen `EvalConfig` dataclass and an `eval` field on `RunConfig` (with `default_factory`, matching the existing `network`/`mcts`/`selfplay`/`training` style and the `from_dict` `build(...)` pattern). All fields have defaults so every existing config and the existing config tests keep working unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config_m6.py
from chessrl.config.config import EvalConfig, RunConfig


def test_m6_eval_defaults():
    cfg = RunConfig()
    assert cfg.eval.every_n_checkpoints == 5
    assert cfg.eval.games_per_rung == 4
    assert cfg.eval.agent_simulations == 200
    assert cfg.eval.max_plies == 256
    assert cfg.eval.stockfish_path == ""
    assert cfg.eval.stockfish_movetime_ms == 100
    assert cfg.eval.poll_seconds == 10.0


def test_m6_eval_fields_overridable():
    e = EvalConfig(every_n_checkpoints=1, games_per_rung=2, agent_simulations=8)
    assert e.every_n_checkpoints == 1
    assert e.games_per_rung == 2
    assert e.agent_simulations == 8


def test_m6_games_per_rung_default_is_even():
    # games_per_rung must be even so both colors are played equally.
    assert RunConfig().eval.games_per_rung % 2 == 0


def test_m6_yaml_partial_override(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(
        "eval:\n"
        "  every_n_checkpoints: 2\n"
        "  games_per_rung: 6\n"
        "  agent_simulations: 50\n"
        "  stockfish_path: tools/stockfish/stockfish.exe\n"
        "  poll_seconds: 1.0\n"
    )
    cfg = RunConfig.from_yaml(p)
    assert cfg.eval.every_n_checkpoints == 2
    assert cfg.eval.games_per_rung == 6
    assert cfg.eval.agent_simulations == 50
    assert cfg.eval.stockfish_path == "tools/stockfish/stockfish.exe"
    assert cfg.eval.poll_seconds == 1.0
    assert cfg.eval.max_plies == 256            # untouched default survives
    assert cfg.mcts.simulations == 200          # other sections untouched


def test_m6_eval_in_json_round_trip(tmp_path):
    cfg = RunConfig()
    p = tmp_path / "config.json"
    p.write_text(cfg.to_json())
    cfg2 = RunConfig.from_json(p)
    assert cfg2 == cfg
    assert cfg2.eval.games_per_rung == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_config_m6.py -v`
Expected: FAIL — `ImportError` on `EvalConfig` / `AttributeError` on `cfg.eval`.

- [ ] **Step 3: Implement (edit `chessrl/config/config.py`)**

Add the `EvalConfig` dataclass directly above `RunConfig`:

```python
@dataclass(frozen=True)
class EvalConfig:
    every_n_checkpoints: int = 5         # evaluate every Nth checkpoint per run
    games_per_rung: int = 4              # games vs each ladder rung; MUST be even (both colors equally)
    agent_simulations: int = 200         # MCTS sims/move for the agent player in evaluation
    max_plies: int = 256                 # ply cap; a capped game is adjudicated a draw
    stockfish_path: str = ""             # "" disables all Stockfish rungs (floors-only ladder)
    stockfish_movetime_ms: int = 100     # per-move think time for movetime-limited Stockfish rungs
    poll_seconds: float = 10.0           # daemon poll interval for new checkpoints / inbox
```

Add the `eval` field to `RunConfig` (after `training`):

```python
@dataclass(frozen=True)
class RunConfig:
    run_name: str = "default"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
```

Add one line to `RunConfig.from_dict` so YAML/JSON `eval:` sections are parsed (mirroring the other `build(...)` calls):

```python
        return cls(
            run_name=raw.get("run_name", "default"),
            network=build(NetworkConfig, "network"),
            mcts=build(MCTSConfig, "mcts"),
            selfplay=build(SelfPlayConfig, "selfplay"),
            training=build(TrainingConfig, "training"),
            eval=build(EvalConfig, "eval"),
        )
```

`asdict`/`to_json` and `from_json` pick up the new section automatically (no further change).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_config_m6.py tests/test_config.py -v`
Expected: all passed (new M6 tests + the original config tests, proving defaults and round-trip are unbroken).

- [ ] **Step 5: Commit**

```powershell
git add chessrl/config/config.py tests/test_config_m6.py
git commit -m "feat: M6 EvalConfig section (roster cadence, games per rung, agent sims, stockfish pinning, poll)"
```

---

### Task 2: Ladder players (floors + Stockfish)

**Files:**
- Create: `chessrl/evaluation/__init__.py` (empty package marker)
- Create: `chessrl/evaluation/players.py`
- Test: `tests/test_players.py`

The duck-typed player roster. Pure-Python floors are deterministic given a seed. `StockfishPlayer` wraps `chess.engine.SimpleEngine` with pinned options, a per-move timeout, and one auto-restart on engine error/timeout (UCI engines occasionally hang; the evaluator must not). `NetMCTSPlayer` wraps the agent: it loads `BatchedNetEvaluator.from_checkpoint`, adapts it to the single-board `.evaluate(board)` the reference MCTS expects, searches with `add_noise=False`, and plays argmax visits.

Piece values (used by Greedy and Minimax): `P=1, N=3, B=3, R=5, Q=9, K=0`; checkmate is worth `+1000` to the side delivering it.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_players.py -v`
Expected: FAIL — `ImportError` on `chessrl.evaluation.players`. (Stockfish tests are skipped while the binary is absent.)

- [ ] **Step 3: Implement**

Create the package marker:

```python
# chessrl/evaluation/__init__.py
```

```python
# chessrl/evaluation/players.py
"""Ladder players. Every player exposes .name, .play(board) -> chess.Move, and
optionally .close(). Floors are pure-Python and seeded-deterministic; the agent
wraps the reference MCTS with noise/temperature OFF; Stockfish wraps a pinned,
timeout-guarded, auto-restarting UCI subprocess.
"""
from pathlib import Path

import chess
import chess.engine
import numpy as np

# Material values (king = 0; captures of the king never occur in legal chess).
PIECE_VALUE = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
MATE_SCORE = 1000.0


def default_stockfish_path() -> Path | None:
    """Provisioned Stockfish binary, or None. tools/stockfish/stockfish.exe
    (Windows) or tools/stockfish/stockfish (POSIX), resolved relative to repo root."""
    root = Path(__file__).resolve().parents[2]   # chessrl/evaluation/players.py -> repo root
    for name in ("stockfish.exe", "stockfish"):
        cand = root / "tools" / "stockfish" / name
        if cand.exists():
            return cand
    return None


def _material(board: chess.Board, color: bool) -> float:
    total = 0.0
    for sq, piece in board.piece_map().items():
        v = PIECE_VALUE[piece.piece_type]
        total += v if piece.color == color else -v
    return total


class RandomPlayer:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def play(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        return moves[int(self._rng.integers(len(moves)))]


class GreedyMaterialPlayer:
    """Maximizes immediate material after the move (mate = +MATE_SCORE), with a
    seeded random tiebreak among equal-best moves."""

    name = "greedy"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def play(self, board: chess.Board) -> chess.Move:
        me = board.turn
        best_score, best_moves = -1e18, []
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                score = MATE_SCORE
            else:
                score = _material(board, me)
            board.pop()
            if score > best_score + 1e-9:
                best_score, best_moves = score, [mv]
            elif abs(score - best_score) <= 1e-9:
                best_moves.append(mv)
        return best_moves[int(self._rng.integers(len(best_moves)))]


class MinimaxPlayer:
    """Negamax (alpha-beta) with a material leaf eval at fixed depth (1-3), plus
    optional Gaussian eval noise on leaves for diversity, and a seeded random
    tiebreak among equal-best root moves. Eval is from the side-to-move's view."""

    def __init__(self, depth: int = 2, seed: int = 0, noise: float = 0.0):
        assert 1 <= depth <= 3
        self.depth = depth
        self.noise = noise
        self.name = f"minimax{depth}"
        self._rng = np.random.default_rng(seed)

    def _leaf_eval(self, board: chess.Board) -> float:
        val = _material(board, board.turn)
        if self.noise > 0.0:
            val += float(self._rng.normal(0.0, self.noise))
        return val

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if board.is_checkmate():
            return -MATE_SCORE                      # side to move is mated
        if board.is_game_over(claim_draw=True):
            return 0.0
        if depth == 0:
            return self._leaf_eval(board)
        best = -1e18
        for mv in board.legal_moves:
            board.push(mv)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def play(self, board: chess.Board) -> chess.Move:
        best_score, best_moves = -1e18, []
        for mv in board.legal_moves:
            board.push(mv)
            score = -self._negamax(board, self.depth - 1, -1e18, 1e18)
            board.pop()
            if score > best_score + 1e-9:
                best_score, best_moves = score, [mv]
            elif abs(score - best_score) <= 1e-9:
                best_moves.append(mv)
        return best_moves[int(self._rng.integers(len(best_moves)))]


class _SingleFromBatched:
    """Adapts a BatchedNetEvaluator (.evaluate_many) to the single-board
    .evaluate(board) -> (policy, value) the reference MCTS expects."""

    def __init__(self, batched):
        self._batched = batched

    def evaluate(self, board: chess.Board):
        policies, values = self._batched.evaluate_many([board])
        return policies[0], float(values[0])


class NetMCTSPlayer:
    """The agent. Loads a checkpoint, searches with the reference MCTS at a fixed
    simulation count, noise and temperature OFF (eval mode), plays argmax visits."""

    def __init__(
        self,
        name: str,
        checkpoint_path,
        network_cfg,
        simulations: int,
        device: str = "cpu",
        seed: int = 0,
    ):
        # Imported here so chessrl.evaluation.players has no hard torch import at
        # module load (the floors and ratings code must import without torch).
        from chessrl.config.config import MCTSConfig
        from chessrl.mcts.reference import ReferenceMCTS
        from chessrl.model.network import BatchedNetEvaluator

        self.name = name
        batched = BatchedNetEvaluator.from_checkpoint(
            checkpoint_path, network_cfg, device=device
        )
        self._eval = _SingleFromBatched(batched)
        self._cfg = MCTSConfig(simulations=simulations)
        self._mcts = ReferenceMCTS(self._eval, self._cfg, rng=np.random.default_rng(seed))

    def play(self, board: chess.Board) -> chess.Move:
        from chessrl.chess_env.moves import index_to_move

        visits, _root_q = self._mcts.search(board, add_noise=False)
        best_idx = max(visits, key=visits.get)
        return index_to_move(best_idx, board.turn == chess.BLACK, board)


class StockfishError(RuntimeError):
    pass


class StockfishPlayer:
    """Pinned, timeout-guarded UCI engine wrapper with one auto-restart on
    error/timeout. Exactly one of (elo, nodes, movetime) governs strength:
      * elo   -> UCI_LimitStrength=true, UCI_Elo=<elo> (calibrated anchor)
      * nodes -> fixed node budget (dead-zone rung)
      * else  -> fixed movetime_ms think time
    Threads=1, Ponder=false always (reproducibility). conditions() returns the
    pinned options + engine version id for recording per match.
    """

    def __init__(
        self,
        path: str,
        *,
        elo: int | None = None,
        nodes: int | None = None,
        movetime_ms: int = 100,
        name: str | None = None,
        timeout_s: float = 5.0,
    ):
        self._path = path
        self._elo = elo
        self._nodes = nodes
        self._movetime_ms = movetime_ms
        self._timeout_s = timeout_s
        self.name = name or (
            f"sf_elo{elo}" if elo is not None
            else f"sf_nodes{nodes}" if nodes is not None
            else f"sf_mt{movetime_ms}"
        )
        self._engine = None
        self._engine_id = ""
        self._start()

    def _start(self) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
        self._engine_id = self._engine.id.get("name", "stockfish")
        opts = {"Threads": 1, "Ponder": False}
        if self._elo is not None:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = self._elo
        # Only configure options the engine actually advertises (older builds vary).
        supported = {k: v for k, v in opts.items() if k in self._engine.options}
        if supported:
            self._engine.configure(supported)

    def _restart(self) -> None:
        try:
            if self._engine is not None:
                self._engine.quit()
        except Exception:
            pass
        self._start()

    def _limit(self) -> chess.engine.Limit:
        if self._nodes is not None:
            return chess.engine.Limit(nodes=self._nodes)
        return chess.engine.Limit(time=self._movetime_ms / 1000.0)

    def play(self, board: chess.Board) -> chess.Move:
        for attempt in range(2):                    # one auto-restart retry
            try:
                result = self._engine.play(
                    board, self._limit(), timeout=self._timeout_s
                )
                if result.move is None:
                    raise StockfishError("engine returned no move")
                return result.move
            except Exception:
                if attempt == 0:
                    self._restart()
                    continue
                raise StockfishError(f"{self.name} failed after restart")

    def conditions(self) -> dict:
        cond = {
            "engine_id": self._engine_id,
            "Threads": 1,
            "Ponder": False,
        }
        if self._elo is not None:
            cond["UCI_LimitStrength"] = True
            cond["UCI_Elo"] = self._elo
        if self._nodes is not None:
            cond["nodes"] = self._nodes
        else:
            cond["movetime_ms"] = self._movetime_ms
        return cond

    def close(self) -> None:
        try:
            if self._engine is not None:
                self._engine.quit()
        finally:
            self._engine = None
```

- [ ] **Step 4: Run the player tests (floors live; Stockfish skipped)**

Run: `.venv\Scripts\python -m pytest tests/test_players.py -v`
Expected: floor tests passed; the two Stockfish tests `SKIPPED` (binary not yet provisioned). If `test_greedy_takes_free_queen` fails, the bug is the material sign in `_material` (must be from the mover's perspective); if `test_minimax_avoids_hanging_its_queen_depth2` fails, the negamax sign flip at the recursive call is wrong.

- [ ] **Step 5: Commit**

```powershell
git add chessrl/evaluation/__init__.py chessrl/evaluation/players.py tests/test_players.py
git commit -m "feat: ladder players (random, greedy, minimax, net-MCTS agent, pinned Stockfish)"
```

---

### Task 3: Opening suite (50 short book lines)

**Files:**
- Create: `chessrl/evaluation/openings.py`
- Test: `tests/test_openings.py`

A deterministic suite of 50 short opening lines (2–4 UCI half-moves each), covering the major systems, so deterministic players at fixed openings produce a varied game set (the spec: a small fixed set silently collapses the effective sample size). `opening_board(idx)` applies line `idx % len(OPENINGS)` to a fresh board and returns it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_openings.py
import chess

from chessrl.evaluation.openings import OPENINGS, opening_board


def test_there_are_fifty_openings():
    assert len(OPENINGS) == 50


def test_every_opening_is_a_legal_sequence():
    for i, line in enumerate(OPENINGS):
        # 2-4 full moves; two canonical lines (Ruy Lopez, Italian) run to a 5th
        # half-move to be distinctive, so the bound is 5 half-moves.
        assert 2 <= len(line) <= 5, f"opening {i} wrong length: {line}"
        board = chess.Board()
        for uci in line:
            mv = chess.Move.from_uci(uci)
            assert mv in board.legal_moves, f"illegal move {uci} in opening {i}: {line}"
            board.push(mv)


def test_openings_are_distinct_positions():
    fens = set()
    for line in OPENINGS:
        b = chess.Board()
        for uci in line:
            b.push(chess.Move.from_uci(uci))
        fens.add(b.board_fen() + (" w" if b.turn else " b"))
    assert len(fens) == 50, "duplicate opening positions reduce effective sample size"


def test_opening_board_wraps_modulo():
    b0 = opening_board(0)
    b_wrap = opening_board(len(OPENINGS))
    assert b0.fen() == b_wrap.fen()
    # A fresh, independent board each call (no shared mutable state).
    b0.push(chess.Move.from_uci("a2a3"))
    assert opening_board(0).fen() != b0.fen()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_openings.py -v`
Expected: FAIL — `ImportError` on `chessrl.evaluation.openings`.

- [ ] **Step 3: Implement**

The finalized 50-entry suite below is verified: every line is legal from the start, all 50 resulting positions are distinct, and lengths are 2–5 half-moves (the two length-5 lines are the canonical Ruy Lopez and Italian, distinguished by their 5th half-move). Type it exactly; the three `test_*` assertions are the gate — if any fails, fix the offending line, do NOT loosen the assertions.

```python
# chessrl/evaluation/openings.py
"""A suite of 50 short opening book lines (2-5 UCI half-moves) covering the
major systems. Used to seed evaluation games so a deterministic pairing yields a
varied set rather than two repeated games. Every line is a legal sequence from
the starting position and every resulting position is distinct (locked by tests).
"""
import chess

OPENINGS: list[list[str]] = [
    ["e2e4", "e7e5", "g1f3"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["e2e4", "e7e5", "g1f3", "g8f6"],
    ["e2e4", "e7e5", "f1c4"],
    ["e2e4", "e7e5", "b1c3"],
    ["e2e4", "e7e5", "f2f4"],
    ["e2e4", "e7e5", "g1f3", "f8c5"],
    ["e2e4", "e7e5", "g1f3", "d7d6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],       # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],       # Italian
    ["e2e4", "c7c5"],
    ["e2e4", "c7c5", "g1f3"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["e2e4", "c7c5", "g1f3", "b8c6"],
    ["e2e4", "c7c5", "g1f3", "e7e6"],
    ["e2e4", "c7c5", "b1c3"],
    ["e2e4", "c7c5", "c2c3"],
    ["e2e4", "c7c5", "d2d4"],
    ["e2e4", "c7c5", "f2f4"],
    ["e2e4", "e7e6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["e2e4", "c7c6"],
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    ["e2e4", "d7d5"],
    ["e2e4", "d7d5", "e4d5", "g8f6"],
    ["e2e4", "d7d6"],
    ["e2e4", "g8f6"],
    ["e2e4", "g7g6"],
    ["e2e4", "b7b6"],
    ["d2d4", "d7d5"],
    ["d2d4", "d7d5", "c2c4"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    ["d2d4", "d7d5", "c2c4", "d5c4"],
    ["d2d4", "d7d5", "g1f3"],
    ["d2d4", "d7d5", "c1f4"],
    ["d2d4", "d7d5", "e2e3"],
    ["d2d4", "g8f6"],
    ["d2d4", "g8f6", "c2c4"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
    ["d2d4", "g8f6", "c2c4", "c7c5"],
    ["d2d4", "g8f6", "c2c4", "b7b6"],               # Queen's Indian root
    ["d2d4", "g8f6", "c2c4", "d7d6"],               # Old Indian root
    ["d2d4", "g8f6", "g1f3"],
    ["d2d4", "g8f6", "c1g5"],
    ["d2d4", "f7f5"],
    ["g1f3", "d7d5"],
    ["c2c4", "e7e5"],
    ["c2c4", "g8f6"],
]


def opening_board(idx: int) -> chess.Board:
    """Fresh board with opening line (idx % len(OPENINGS)) applied."""
    line = OPENINGS[idx % len(OPENINGS)]
    board = chess.Board()
    for uci in line:
        board.push(chess.Move.from_uci(uci))
    return board
```

- [ ] **Step 4: Run the opening tests**

Run: `.venv\Scripts\python -m pytest tests/test_openings.py -v`
Expected: 4 passed (50 entries, all legal, all distinct, modulo wrap + fresh-board).

- [ ] **Step 5: Commit**

```powershell
git add chessrl/evaluation/openings.py tests/test_openings.py
git commit -m "feat: 50-line opening suite for evaluation games"
```

---

### Task 4: Match runner

**Files:**
- Create: `chessrl/evaluation/match.py`
- Test: `tests/test_match.py`

`play_single(white, black, opening_idx, max_plies) -> (z, pgn_str)` plays one game from `opening_board(opening_idx)`; `z` is from White's perspective; a ply-cap hit is a draw; termination uses `board.outcome(claim_draw=True)`. `play_pairing(a, b, games, openings_start, max_plies) -> list[MatchResult]` plays `games` (must be even) games alternating colors **exactly evenly**, advancing the opening index so each opening is played once with `a` as White and once with `b` as White (both colors share the same opening). `MatchResult` carries `white_name, black_name, z, opening_idx, pgn`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_match.py -v`
Expected: FAIL — `ImportError` on `chessrl.evaluation.match`.

- [ ] **Step 3: Implement**

```python
# chessrl/evaluation/match.py
"""Deterministic match runner over the opening suite.

z is always from White's perspective (+1/0/-1). A ply-cap hit is adjudicated a
draw. Pairings alternate colors EXACTLY evenly (games must be even) and reuse the
same opening for each color-swapped pair, so color advantage never leaks into the
ratings (which see only (white, black, z) triples).
"""
from dataclasses import dataclass

import chess
import chess.pgn

from chessrl.chess_env.game import terminal_value
from chessrl.evaluation.openings import OPENINGS, opening_board

_RESULT_STR = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}


@dataclass
class MatchResult:
    white_name: str
    black_name: str
    z: int                  # White's perspective: +1 white win, 0 draw, -1 black win
    opening_idx: int
    pgn: str


def _board_to_pgn(board: chess.Board, z: int, white_name: str, black_name: str, opening_idx: int) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = _RESULT_STR[z]
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Opening"] = str(opening_idx)
    return str(game)


def play_single(white, black, opening_idx: int, max_plies: int) -> tuple[int, str]:
    """Play one game from opening_board(opening_idx). Returns (z, pgn_str)."""
    board = opening_board(opening_idx)
    while True:
        if board.outcome(claim_draw=True) is not None:
            term = terminal_value(board)            # side-to-move perspective
            if term == 0.0:
                z = 0
            else:
                # term is +1 if side-to-move won, but side-to-move never "wins" on
                # its own move; a finished game's loser is the side to move. Map to
                # White's perspective via the result string below instead:
                z = 1 if term > 0 else -1
                # Convert side-to-move perspective to White's perspective.
                if board.turn == chess.BLACK:
                    z = -z
            break
        if len(board.move_stack) >= max_plies:      # ply cap counts from move 0 of this game
            z = 0
            break
        player = white if board.turn == chess.WHITE else black
        board.push(player.play(board))
    pgn = _board_to_pgn(board, z, getattr(white, "name", "white"), getattr(black, "name", "black"), opening_idx)
    return z, pgn


def play_pairing(a, b, games: int, openings_start: int, max_plies: int) -> list:
    """Play `games` (must be even) games between a and b. Colors alternate
    exactly evenly: for each opening, a-as-White then b-as-White. Returns a list
    of MatchResult."""
    if games % 2 != 0:
        raise ValueError(f"games must be even (equal colors); got {games}")
    results = []
    pairs = games // 2
    for p in range(pairs):
        opening_idx = (openings_start + p) % len(OPENINGS)
        # a as White
        z, pgn = play_single(a, b, opening_idx, max_plies)
        results.append(MatchResult(a.name, b.name, z, opening_idx, pgn))
        # b as White (same opening)
        z, pgn = play_single(b, a, opening_idx, max_plies)
        results.append(MatchResult(b.name, a.name, z, opening_idx, pgn))
    return results
```

Note on the `play_single` z-mapping: `terminal_value` returns the result from the **side-to-move's** perspective, and a decisive finished game always has the *loser* to move (they were just checkmated). So a decisive result is `-1` for the side to move; mapping to White's perspective is `z = +1` iff Black is to move at the end (White delivered mate), `-1` iff White is to move. The branch above computes exactly that; simplify during implementation to the equivalent direct form if preferred (and the tests will confirm correctness):

```python
            term = terminal_value(board)
            if term == 0.0:
                z = 0
            else:                                    # loser is the side to move
                z = 1 if board.turn == chess.BLACK else -1
```

- [ ] **Step 4: Run the match tests**

Run: `.venv\Scripts\python -m pytest tests/test_match.py -v`
Expected: 5 passed. If `test_play_single_ply_cap_is_a_draw` fails, the ply-cap check must compare `len(board.move_stack)` against `max_plies` (the opening already pushed a few plies — that is intended: the cap bounds total game length, draws are the safe adjudication).

- [ ] **Step 5: Commit**

```powershell
git add chessrl/evaluation/match.py tests/test_match.py
git commit -m "feat: deterministic match runner with exactly-equal colors and ply-cap draw"
```

---

### Task 5: Ladder results store (SQLite, single writer)

**Files:**
- Create: `chessrl/evaluation/store.py`
- Test: `tests/test_store.py`

`LadderStore(path)` opens `runs/ladder.sqlite` (path argument) in WAL mode with `busy_timeout=5000`, creates the `results`, `players`, and `evaluated` tables, and exposes `record_result`, `upsert_player`, `all_results`, `all_players`, `mark_evaluated`/`is_evaluated`, and `ingest_inbox(inbox_dir)`. The evaluator is the **sole writer** (documented). `ingest_inbox` reads each JSON file `{white, black, z, opening, conditions}`, records it, and deletes the file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py
import json

from chessrl.evaluation.store import LadderStore


def test_record_and_read_results(tmp_path):
    store = LadderStore(tmp_path / "ladder.sqlite")
    store.record_result("agentA", "random", z=1, opening=0, conditions={"k": "v"})
    store.record_result("agentA", "random", z=0, opening=1, conditions={})
    rows = store.all_results()
    assert len(rows) == 2
    assert rows[0]["white"] == "agentA"
    assert rows[0]["black"] == "random"
    assert rows[0]["z"] == 1
    assert rows[0]["opening"] == 0
    # triples used by the rating fit
    triples = store.result_triples()
    assert ("agentA", "random", 1) in triples


def test_upsert_player_and_anchor(tmp_path):
    store = LadderStore(tmp_path / "ladder.sqlite")
    store.upsert_player("random", kind="floor", anchor_elo=None)
    store.upsert_player("sf_elo1320", kind="anchor", anchor_elo=1320.0)
    # Upsert again with a new kind keeps it idempotent (no duplicate rows).
    store.upsert_player("random", kind="floor", anchor_elo=None)
    players = store.all_players()
    assert players["random"]["anchor_elo"] is None
    assert players["sf_elo1320"]["anchor_elo"] == 1320.0
    assert players["sf_elo1320"]["kind"] == "anchor"
    anchors = store.anchors()
    assert anchors == {"sf_elo1320": 1320.0}


def test_evaluated_tracking(tmp_path):
    store = LadderStore(tmp_path / "ladder.sqlite")
    assert not store.is_evaluated("runs/r1/checkpoints/ckpt_00000010.pt")
    store.mark_evaluated("runs/r1/checkpoints/ckpt_00000010.pt")
    assert store.is_evaluated("runs/r1/checkpoints/ckpt_00000010.pt")
    # idempotent
    store.mark_evaluated("runs/r1/checkpoints/ckpt_00000010.pt")
    assert store.is_evaluated("runs/r1/checkpoints/ckpt_00000010.pt")


def test_ingest_inbox_records_and_deletes(tmp_path):
    store = LadderStore(tmp_path / "ladder.sqlite")
    inbox = tmp_path / "ladder_inbox"
    inbox.mkdir()
    (inbox / "g1.json").write_text(
        json.dumps({"white": "p1", "black": "p2", "z": -1, "opening": 7,
                    "conditions": {"source": "arena"}})
    )
    (inbox / "g2.json").write_text(
        json.dumps({"white": "p2", "black": "p1", "z": 1, "opening": 7, "conditions": {}})
    )
    n = store.ingest_inbox(inbox)
    assert n == 2
    assert len(store.all_results()) == 2
    # files consumed
    assert list(inbox.glob("*.json")) == []


def test_ingest_inbox_skips_bad_json(tmp_path):
    store = LadderStore(tmp_path / "ladder.sqlite")
    inbox = tmp_path / "ladder_inbox"
    inbox.mkdir()
    (inbox / "broken.json").write_text("{ not valid")
    n = store.ingest_inbox(inbox)
    assert n == 0
    # a malformed file is left in place (not silently lost) for inspection
    assert (inbox / "broken.json").exists()


def test_wal_mode_enabled(tmp_path):
    store = LadderStore(tmp_path / "ladder.sqlite")
    assert store.journal_mode().lower() == "wal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_store.py -v`
Expected: FAIL — `ImportError` on `chessrl.evaluation.store`.

- [ ] **Step 3: Implement**

```python
# chessrl/evaluation/store.py
"""Single-writer SQLite results store for the Elo ladder.

The evaluator daemon is the ONLY writer of this database (spec: single-writer
rule for every SQLite file). WAL mode + busy_timeout let read-only consumers
(the M7 dashboard) coexist with the writer. All access is context-managed; one
process, no threads, so sqlite3's default check_same_thread is fine.

Schema:
  results(id PK, ts, white, black, z, opening, conditions TEXT json)
  players(name PK, kind, anchor_elo REAL NULL)   -- anchors pin anchor_elo
  evaluated(ckpt PK)                             -- checkpoints already rated
"""
import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id      INTEGER PRIMARY KEY,
    ts      REAL    NOT NULL,
    white   TEXT    NOT NULL,
    black   TEXT    NOT NULL,
    z       INTEGER NOT NULL,
    opening INTEGER NOT NULL,
    conditions TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS players (
    name       TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    anchor_elo REAL
);
CREATE TABLE IF NOT EXISTS evaluated (
    ckpt TEXT PRIMARY KEY
);
"""


class LadderStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=5000")
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=5.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=5000")
        return con

    # ---- writes (evaluator only) ---------------------------------------

    def record_result(self, white: str, black: str, z: int, opening: int, conditions: dict) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO results(ts, white, black, z, opening, conditions) VALUES (?,?,?,?,?,?)",
                (time.time(), white, black, int(z), int(opening), json.dumps(conditions or {})),
            )

    def upsert_player(self, name: str, kind: str, anchor_elo: float | None = None) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO players(name, kind, anchor_elo) VALUES (?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET kind=excluded.kind, anchor_elo=excluded.anchor_elo",
                (name, kind, anchor_elo),
            )

    def mark_evaluated(self, ckpt: str) -> None:
        with self._connect() as con:
            con.execute("INSERT OR IGNORE INTO evaluated(ckpt) VALUES (?)", (str(ckpt),))

    def ingest_inbox(self, inbox_dir) -> int:
        """Read each JSON result file {white, black, z, opening, conditions},
        record it, and delete the file. Malformed files are left in place.
        Returns the number of results ingested."""
        inbox = Path(inbox_dir)
        if not inbox.exists():
            return 0
        ingested = 0
        for f in sorted(inbox.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                self.record_result(
                    d["white"], d["black"], int(d["z"]), int(d.get("opening", -1)),
                    d.get("conditions", {}),
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue                            # leave malformed file for inspection
            f.unlink()
            ingested += 1
        return ingested

    # ---- reads ----------------------------------------------------------

    def all_results(self) -> list:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, ts, white, black, z, opening, conditions FROM results ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def result_triples(self) -> list:
        with self._connect() as con:
            rows = con.execute("SELECT white, black, z FROM results ORDER BY id").fetchall()
        return [(r["white"], r["black"], int(r["z"])) for r in rows]

    def all_players(self) -> dict:
        with self._connect() as con:
            rows = con.execute("SELECT name, kind, anchor_elo FROM players").fetchall()
        return {r["name"]: {"kind": r["kind"], "anchor_elo": r["anchor_elo"]} for r in rows}

    def anchors(self) -> dict:
        return {
            name: p["anchor_elo"]
            for name, p in self.all_players().items()
            if p["anchor_elo"] is not None
        }

    def is_evaluated(self, ckpt: str) -> bool:
        with self._connect() as con:
            row = con.execute("SELECT 1 FROM evaluated WHERE ckpt=?", (str(ckpt),)).fetchone()
        return row is not None

    def journal_mode(self) -> str:
        with self._connect() as con:
            return con.execute("PRAGMA journal_mode").fetchone()[0]
```

- [ ] **Step 4: Run the store tests**

Run: `.venv\Scripts\python -m pytest tests/test_store.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add chessrl/evaluation/store.py tests/test_store.py
git commit -m "feat: single-writer ladder.sqlite store with WAL, inbox ingest, evaluated-tracking"
```

---

### Task 6: Rating fit (regularized Davidson draw model)

**Files:**
- Create: `chessrl/evaluation/ratings.py`
- Test: `tests/test_ratings.py`

`fit_ratings(results, anchors) -> (ratings_dict, nu)`. Strength `pi_i = 10**(r_i/400)`. For a game between `i` (White) and `j` (Black), with the Davidson draw model:
`D = pi_i + pi_j + nu*sqrt(pi_i*pi_j)`, `P(i wins) = pi_i / D`, `P(j wins) = pi_j / D`, `P(draw) = nu*sqrt(pi_i*pi_j) / D`. (Color symmetric — color balance is handled by the equal-colors match protocol, so the fit ignores who was White.) Maximize the log-likelihood over unpinned ratings + `log(nu)` by plain numpy gradient ascent (anchors FIXED at `anchor_elo`), with a Gaussian prior on each **unpinned** rating (mean `1000`, sigma `350`) that gives 100%/0% players finite, regularized ratings.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ratings.py
import numpy as np

from chessrl.evaluation.ratings import fit_ratings


def _results_from_scores(white, black, wins_w, draws, wins_b):
    """Helper: build a result list (white, black, z) with given counts."""
    out = []
    out += [(white, black, 1)] * wins_w
    out += [(white, black, 0)] * draws
    out += [(white, black, -1)] * wins_b
    return out


def test_two_players_5050_no_draws_equal_ratings():
    # P vs anchor 1000, 50/50, no draws -> P ~ anchor within a couple Elo.
    res = _results_from_scores("P", "A", wins_w=50, draws=0, wins_b=50)
    res += _results_from_scores("A", "P", wins_w=50, draws=0, wins_b=50)
    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    assert abs(ratings["A"] - 1000.0) < 1e-6        # anchor stays pinned
    assert abs(ratings["P"] - 1000.0) < 2.0


def test_75pct_vs_anchor_is_about_1191():
    # P scores 75% vs anchor 1000 (no draws). Unregularized Elo = 1000 + 400*log10(3)
    # = 1190.85. The 1000-mean prior pulls it down slightly; with many games and
    # sigma=350 the pull is small. Tolerance ~15 Elo (and we assert direction).
    res = _results_from_scores("P", "A", wins_w=150, draws=0, wins_b=50)
    res += _results_from_scores("A", "P", wins_w=50, draws=0, wins_b=150)
    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    expected = 1000.0 + 400.0 * np.log10(3.0)       # ~1190.85
    assert ratings["P"] < expected                  # prior pulls toward 1000
    assert abs(ratings["P"] - expected) < 15.0


def test_all_wins_is_finite_and_regularized():
    # P beats anchor 1000 every game -> unregularized MLE diverges to +inf.
    # The prior must keep it finite, between +200 and +800 above the anchor.
    res = _results_from_scores("P", "A", wins_w=40, draws=0, wins_b=0)
    res += _results_from_scores("A", "P", wins_w=0, draws=0, wins_b=40)
    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    assert np.isfinite(ratings["P"])
    # Draw-free saturation + the weak 1000-mean prior settle this near +750 Elo;
    # the point is finiteness and regularization, not an exact value.
    assert 1200.0 < ratings["P"] < 1800.0


def test_draws_raise_nu():
    # Draw-heavy data -> larger nu than near-draw-free data.
    drawy = _results_from_scores("P", "A", wins_w=10, draws=80, wins_b=10)
    drawy += _results_from_scores("A", "P", wins_w=10, draws=80, wins_b=10)
    _, nu_high = fit_ratings(drawy, anchors={"A": 1000.0})

    sharp = _results_from_scores("P", "A", wins_w=50, draws=2, wins_b=48)
    sharp += _results_from_scores("A", "P", wins_w=48, draws=2, wins_b=50)
    _, nu_low = fit_ratings(sharp, anchors={"A": 1000.0})

    assert nu_high > nu_low


def test_recovers_known_ratings_from_synthetic_data():
    # Generate games FROM the model with known ratings, recover within ~30 Elo.
    rng = np.random.default_rng(0)
    true = {"A": 1000.0, "M": 1200.0, "S": 1400.0}
    nu_true = 0.5

    def pi(name):
        return 10.0 ** (true[name] / 400.0)

    def sample(w, b, n):
        out = []
        pw, pb = pi(w), pi(b)
        d = pw + pb + nu_true * np.sqrt(pw * pb)
        p_w, p_d = pw / d, nu_true * np.sqrt(pw * pb) / d
        for _ in range(n):
            u = rng.random()
            out.append((w, b, 1 if u < p_w else (0 if u < p_w + p_d else -1)))
        return out

    res = []
    for x, y in [("A", "M"), ("A", "S"), ("M", "S")]:
        res += sample(x, y, 300)
        res += sample(y, x, 300)

    ratings, nu = fit_ratings(res, anchors={"A": 1000.0})
    # Order preserved and magnitudes recovered within ~30 Elo (A pinned).
    assert ratings["A"] == 1000.0
    assert ratings["M"] < ratings["S"]
    assert abs(ratings["M"] - 1200.0) < 30.0
    assert abs(ratings["S"] - 1400.0) < 30.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_ratings.py -v`
Expected: FAIL — `ImportError` on `chessrl.evaluation.ratings`.

- [ ] **Step 3: Implement**

```python
# chessrl/evaluation/ratings.py
"""Regularized Davidson draw-model rating fit.

For a game between i and j (color-symmetric; the equal-colors match protocol
removes color advantage upstream so the fit ignores who was White), with
strengths pi = 10**(r/400):
    D          = pi_i + pi_j + nu*sqrt(pi_i*pi_j)
    P(i wins)  = pi_i / D
    P(j wins)  = pi_j / D
    P(draw)    = nu*sqrt(pi_i*pi_j) / D

We maximize the log-likelihood over the UNPINNED ratings and log(nu) by plain
numpy gradient ascent. Anchors are FIXED at their known Elo. Each unpinned rating
carries a Gaussian prior N(PRIOR_MEAN, PRIOR_SIGMA): without it an undefeated or
winless player's MLE diverges to +/-inf. PRIOR_MEAN=1000 is a deliberate FLOOR
CALIBRATION choice -- it anchors the un-anchored cloud near the random/greedy
floor region so early checkpoints land on a sensible absolute scale; PRIOR_SIGMA
=350 keeps the prior weak enough that ~100+ games dominate it (the 75%-vs-anchor
case lands within ~15 Elo of the unregularized 1190.85).

We work in the natural-log strength variable theta = r * ln(10)/400 so that
pi = exp(theta); gradients are clean and the 400-scale is reintroduced only when
converting back to Elo.
"""
import numpy as np

PRIOR_MEAN = 1000.0
PRIOR_SIGMA = 350.0
_SCALE = np.log(10.0) / 400.0          # r (Elo) -> theta (ln strength): theta = r*_SCALE


def fit_ratings(results, anchors: dict, iters: int = 4000, seed: int = 0):
    """results: iterable of (white_name, black_name, z) with z in {+1,0,-1}
    (White's perspective). anchors: {name: elo} pinned exactly. Returns
    (ratings: {name: elo_float}, nu: float)."""
    results = list(results)
    names = sorted({n for w, b, _ in results for n in (w, b)} | set(anchors))
    idx = {n: k for k, n in enumerate(names)}
    n = len(names)

    anchored = np.zeros(n, dtype=bool)
    theta = np.zeros(n, dtype=np.float64)
    for name, elo in anchors.items():
        anchored[idx[name]] = True
        theta[idx[name]] = elo * _SCALE
    # init unpinned at the prior mean
    for k in range(n):
        if not anchored[k]:
            theta[k] = PRIOR_MEAN * _SCALE

    if not results:
        ratings = {nm: theta[idx[nm]] / _SCALE for nm in names}
        return ratings, 1.0

    wi = np.array([idx[w] for w, _, _ in results])
    bj = np.array([idx[b] for _, b, _ in results])
    z = np.array([zz for _, _, zz in results], dtype=np.int64)
    is_w_win = (z == 1).astype(np.float64)     # White (i) wins
    is_b_win = (z == -1).astype(np.float64)    # Black (j) wins
    is_draw = (z == 0).astype(np.float64)

    log_nu = 0.0
    prior_var = (PRIOR_SIGMA * _SCALE) ** 2
    prior_mean_theta = PRIOR_MEAN * _SCALE

    lr = 0.05
    for it in range(iters):
        nu = np.exp(log_nu)
        ti, tj = theta[wi], theta[bj]
        # work in shifted log-space for numerical stability: terms exp(ti), exp(tj),
        # exp(0.5(ti+tj)+log_nu); subtract row max.
        a = ti
        b = tj
        c = 0.5 * (ti + tj) + log_nu
        m = np.maximum(np.maximum(a, b), c)
        ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
        denom = ea + eb + ec                    # = D / exp(m)
        p_i = ea / denom
        p_j = eb / denom
        p_d = ec / denom

        # Gradient of log-likelihood wrt theta_i for one game:
        #   d/dti log P(outcome) = [1{i wins} or 0.5*1{draw}] - (p_i + 0.5 p_d)
        # and symmetrically for theta_j. (c depends on 0.5*(ti+tj).)
        gi = is_w_win + 0.5 * is_draw - (p_i + 0.5 * p_d)
        gj = is_b_win + 0.5 * is_draw - (p_j + 0.5 * p_d)

        grad = np.zeros(n, dtype=np.float64)
        np.add.at(grad, wi, gi)
        np.add.at(grad, bj, gj)
        # Gaussian prior on unpinned thetas
        grad -= np.where(anchored, 0.0, (theta - prior_mean_theta) / prior_var)
        grad[anchored] = 0.0

        # Gradient wrt log_nu: sum over games of [1{draw} - p_d]
        g_lognu = float(np.sum(is_draw - p_d))

        step = lr / (1.0 + it / 500.0)          # simple decay schedule
        theta += step * grad
        log_nu += step * g_lognu

        if np.max(np.abs(step * grad)) < 1e-9 and abs(step * g_lognu) < 1e-9:
            break

    ratings = {nm: float(theta[idx[nm]] / _SCALE) for nm in names}
    for name, elo in anchors.items():
        ratings[name] = float(elo)              # exact pin (defends against drift)
    return ratings, float(np.exp(log_nu))
```

- [ ] **Step 4: Run the ratings tests**

Run: `.venv\Scripts\python -m pytest tests/test_ratings.py -v`
Expected: 5 passed. These are the spec's "Elo fit vs hand-computed cases (including the all-wins regularization behavior)" unit gate. If `test_75pct_vs_anchor_is_about_1191` is outside tolerance, the gradient or the `_SCALE` conversion is off — re-derive `d/dti log P` rather than loosening the tolerance. If `test_draws_raise_nu` fails, the `log_nu` gradient sign is wrong. If convergence is too slow, raise `iters` or the base `lr`; do not relax the assertions.

- [ ] **Step 5: Commit**

```powershell
git add chessrl/evaluation/ratings.py tests/test_ratings.py
git commit -m "feat: regularized Davidson draw-model rating fit with pinned anchors"
```

---

### Task 7: Evaluator daemon + CLI

**Files:**
- Create: `chessrl/evaluation/daemon.py`
- Create: `scripts/evaluate.py`
- Test: `tests/test_daemon.py`

`evaluate_checkpoint(run_dir, ckpt_path, cfg, store, openings_offset)` builds a `NetMCTSPlayer` named `{run_id}@{step}` for the checkpoint, plays `cfg.games_per_rung` games vs each rung (always Random, Greedy, Minimax(2); plus Stockfish rungs if `cfg.stockfish_path`), saves PGNs under `run_dir/eval_games/`, records every result + upserts players (anchors pinned, floors unpinned), refits ratings over the whole store, and appends `{ts, step, ckpt, elo, nu}` to `run_dir/elo.jsonl`. `watch(runs_root, cfg)` polls for new every-Nth checkpoints across runs, ingests `ladder_inbox`, and is `EVAL_STOP`-file aware. `main(argv)` is the `scripts/evaluate.py` CLI.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_daemon.py
import json
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import EvalConfig, NetworkConfig, TrainingConfig
from chessrl.evaluation.daemon import (
    evaluate_checkpoint,
    eligible_checkpoints,
    run_once,
)
from chessrl.evaluation.store import LadderStore
from chessrl.model.network import PolicyValueNet
from chessrl.training.trainer import Trainer


def _make_run_with_checkpoint(tmp_path, run_name="r1", step_target=1):
    """Create runs/<run> with config.json and one tiny checkpoint."""
    run_dir = tmp_path / "runs" / run_name
    (run_dir / "checkpoints").mkdir(parents=True)
    net_cfg = NetworkConfig(blocks=2, filters=8)
    from chessrl.config.config import RunConfig
    cfg = RunConfig(network=net_cfg)
    (run_dir / "config.json").write_text(cfg.to_json())
    torch.manual_seed(0)
    net = PolicyValueNet(net_cfg)
    trainer = Trainer(net, TrainingConfig(batch_size=4, device="cpu"), run_dir)
    ckpt = trainer.save_checkpoint()       # ckpt_00000000.pt
    return run_dir, ckpt, net_cfg


def test_evaluate_checkpoint_records_results_and_elo(tmp_path):
    run_dir, ckpt, net_cfg = _make_run_with_checkpoint(tmp_path)
    store = LadderStore(tmp_path / "runs" / "ladder.sqlite")
    cfg = EvalConfig(games_per_rung=2, agent_simulations=8, max_plies=40, stockfish_path="")

    elo = evaluate_checkpoint(run_dir, ckpt, cfg, store, openings_offset=0)

    # Floors-only ladder: random, greedy, minimax2 -> 3 rungs * 2 games = 6 results.
    assert len(store.all_results()) == 6
    # the agent player and the three floor players are registered
    players = store.all_players()
    assert any("@" in name for name in players)            # agent {run_id}@{step}
    assert {"random", "greedy", "minimax2"} <= set(players)
    # elo.jsonl appended with this checkpoint
    elo_lines = (run_dir / "elo.jsonl").read_text().splitlines()
    assert len(elo_lines) == 1
    entry = json.loads(elo_lines[0])
    for key in ("ts", "step", "ckpt", "elo", "nu"):
        assert key in entry
    assert entry["step"] == 0
    assert np.isfinite(entry["elo"])
    # PGNs saved
    pgns = list((run_dir / "eval_games").glob("*.pgn"))
    assert len(pgns) == 6
    # checkpoint marked evaluated
    assert store.is_evaluated(str(ckpt))


def test_eligible_checkpoints_respects_every_n(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "checkpoints").mkdir(parents=True)
    for step in (0, 1000, 2000, 3000, 4000, 5000):
        (run_dir / "checkpoints" / f"ckpt_{step:08d}.pt").write_bytes(b"x")
    store = LadderStore(tmp_path / "runs" / "ladder.sqlite")
    cfg = EvalConfig(every_n_checkpoints=2)
    # every 2nd checkpoint by index: indices 0,2,4 -> steps 0,2000,4000
    elig = eligible_checkpoints(run_dir, cfg, store)
    steps = [int(Path(c).stem.split("_")[1]) for c in elig]
    assert steps == [0, 2000, 4000]
    # after marking one evaluated it is skipped
    store.mark_evaluated(str(run_dir / "checkpoints" / "ckpt_00000000.pt"))
    elig2 = eligible_checkpoints(run_dir, cfg, store)
    steps2 = [int(Path(c).stem.split("_")[1]) for c in elig2]
    assert steps2 == [2000, 4000]


def test_run_once_evaluates_latest_eligible_then_returns(tmp_path):
    run_dir, ckpt, net_cfg = _make_run_with_checkpoint(tmp_path)
    runs_root = tmp_path / "runs"
    cfg = EvalConfig(every_n_checkpoints=1, games_per_rung=2, agent_simulations=8,
                     max_plies=40, stockfish_path="")
    n_eval = run_once(runs_root, cfg, run_filter=None)
    assert n_eval == 1
    store = LadderStore(runs_root / "ladder.sqlite")
    assert store.is_evaluated(str(ckpt))
    assert (run_dir / "elo.jsonl").exists()
    # a second --once pass finds nothing new
    n_eval2 = run_once(runs_root, cfg, run_filter=None)
    assert n_eval2 == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_daemon.py -v`
Expected: FAIL — `ImportError` on `chessrl.evaluation.daemon`.

- [ ] **Step 3: Implement the daemon**

```python
# chessrl/evaluation/daemon.py
"""Evaluator daemon: the single writer that makes the Elo curve automatic.

evaluate_checkpoint plays a checkpoint against the ladder, records results into
ladder.sqlite, refits ratings over the WHOLE store (anchors pinned), and appends
the checkpoint's Elo to run_dir/elo.jsonl. watch() polls runs for new every-Nth
checkpoints and ingests ladder_inbox, stopping when runs_root/EVAL_STOP appears.
Synchronous and single-process by design (no threads); the GPU is time-sliced
against training, so the agent player runs on CPU by default.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from chessrl.config.config import EvalConfig, RunConfig
from chessrl.evaluation.match import play_pairing
from chessrl.evaluation.players import (
    GreedyMaterialPlayer,
    MinimaxPlayer,
    NetMCTSPlayer,
    RandomPlayer,
    StockfishPlayer,
)
from chessrl.evaluation.ratings import fit_ratings
from chessrl.evaluation.store import LadderStore

STOP_FILE = "EVAL_STOP"
LADDER_DB = "ladder.sqlite"
INBOX_DIR = "ladder_inbox"

# Default Stockfish anchor rungs (pinned) and dead-zone node rungs (unpinned).
_ANCHOR_ELOS = (1320, 1500, 1700)
_NODE_RUNGS = (1, 100)


def _run_id(run_dir: Path) -> str:
    return Path(run_dir).name


def _checkpoints(run_dir) -> list:
    return sorted((Path(run_dir) / "checkpoints").glob("ckpt_*.pt"))


def _step_of(ckpt) -> int:
    return int(Path(ckpt).stem.split("_")[1])


def eligible_checkpoints(run_dir, cfg: EvalConfig, store: LadderStore) -> list:
    """Every Nth checkpoint (by index) not already evaluated."""
    ckpts = _checkpoints(run_dir)
    selected = ckpts[:: cfg.every_n_checkpoints]
    return [c for c in selected if not store.is_evaluated(str(c))]


def _build_floor_players(seed: int) -> list:
    return [
        (RandomPlayer(seed=seed), "floor", None),
        (GreedyMaterialPlayer(seed=seed), "floor", None),
        (MinimaxPlayer(depth=2, seed=seed), "floor", None),
    ]


def _build_stockfish_players(cfg: EvalConfig) -> list:
    """(player, kind, anchor_elo) tuples for Stockfish rungs; empty if disabled."""
    if not cfg.stockfish_path:
        return []
    out = []
    for nodes in _NODE_RUNGS:
        out.append((StockfishPlayer(cfg.stockfish_path, nodes=nodes,
                                    name=f"sf_nodes{nodes}"), "rung", None))
    for elo in _ANCHOR_ELOS:
        out.append((StockfishPlayer(cfg.stockfish_path, elo=elo,
                                    movetime_ms=cfg.stockfish_movetime_ms,
                                    name=f"sf_elo{elo}"), "anchor", float(elo)))
    return out


def evaluate_checkpoint(run_dir, ckpt_path, cfg: EvalConfig, store: LadderStore,
                        openings_offset: int) -> float:
    """Play ckpt vs the ladder, record results, refit ratings, append elo.jsonl.
    Returns the checkpoint's fitted Elo."""
    run_dir = Path(run_dir)
    run_cfg = RunConfig.from_json(run_dir / "config.json")
    step = _step_of(ckpt_path)
    agent_name = f"{_run_id(run_dir)}@{step}"
    agent = NetMCTSPlayer(
        agent_name, ckpt_path, run_cfg.network, cfg.agent_simulations, device="cpu",
    )
    store.upsert_player(agent_name, kind="agent", anchor_elo=None)

    rungs = _build_floor_players(seed=step) + _build_stockfish_players(cfg)
    eval_games_dir = run_dir / "eval_games"
    eval_games_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, (opp, kind, anchor_elo) in enumerate(rungs):
            store.upsert_player(opp.name, kind=kind, anchor_elo=anchor_elo)
            conditions = opp.conditions() if hasattr(opp, "conditions") else {}
            results = play_pairing(
                agent, opp, games=cfg.games_per_rung,
                openings_start=openings_offset + i * (cfg.games_per_rung // 2),
                max_plies=cfg.max_plies,
            )
            for r in results:
                store.record_result(r.white_name, r.black_name, r.z, r.opening_idx, conditions)
                fname = f"{step:08d}_{opp.name}_{r.opening_idx:02d}_{r.white_name.replace('@','_at_')}.pgn"
                (eval_games_dir / fname).write_text(r.pgn)
    finally:
        for opp, _kind, _elo in rungs:
            getattr(opp, "close", lambda: None)()

    ratings, nu = fit_ratings(store.result_triples(), anchors=store.anchors())
    elo = float(ratings.get(agent_name, float("nan")))
    entry = {"ts": time.time(), "step": step, "ckpt": str(ckpt_path), "elo": elo, "nu": nu}
    with (run_dir / "elo.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    store.mark_evaluated(str(ckpt_path))
    return elo


def run_once(runs_root, cfg: EvalConfig, run_filter: str | None) -> int:
    """Evaluate every eligible un-evaluated checkpoint of each run once; return
    the number of checkpoints evaluated."""
    runs_root = Path(runs_root)
    store = LadderStore(runs_root / LADDER_DB)
    store.ingest_inbox(runs_root / INBOX_DIR)
    count = 0
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "config.json").exists()):
        if run_filter is not None and run_dir.name != run_filter:
            continue
        for offset, ckpt in enumerate(eligible_checkpoints(run_dir, cfg, store)):
            evaluate_checkpoint(run_dir, ckpt, cfg, store, openings_offset=offset)
            count += 1
    return count


def watch(runs_root, cfg: EvalConfig, run_filter: str | None = None) -> None:
    """Poll for new eligible checkpoints + inbox until runs_root/EVAL_STOP exists."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    stop = runs_root / STOP_FILE
    while not stop.exists():
        run_once(runs_root, cfg, run_filter)
        if stop.exists():
            break
        time.sleep(cfg.poll_seconds)


def _load_eval_cfg(config_path: str | None) -> EvalConfig:
    if not config_path:
        return EvalConfig()
    return RunConfig.from_yaml(config_path).eval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Elo evaluator daemon")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run", default=None, help="restrict to one run id")
    ap.add_argument("--once", action="store_true", help="evaluate eligible checkpoints then exit")
    ap.add_argument("--config", default=None, help="YAML with an eval: section (overrides EvalConfig)")
    args = ap.parse_args(argv)

    cfg = _load_eval_cfg(args.config)
    if args.once:
        n = run_once(args.runs_root, cfg, args.run)
        print(f"evaluated {n} checkpoint(s)")
        return n
    watch(args.runs_root, cfg, args.run)
    return 0
```

- [ ] **Step 4: Implement the CLI entry point**

```python
# scripts/evaluate.py
"""Elo evaluator entry point.

Once over all runs:  python scripts/evaluate.py --once --runs-root runs
Watch daemon:        python scripts/evaluate.py --runs-root runs --config experiments/eval.yaml
Restrict to one run: python scripts/evaluate.py --once --run <run-id>
Stop the daemon:     create runs/EVAL_STOP
"""
import sys

from chessrl.evaluation.daemon import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
```

- [ ] **Step 5: Run the daemon tests**

Run: `.venv\Scripts\python -m pytest tests/test_daemon.py -v`
Expected: 3 passed. These run the real `NetMCTSPlayer` with a 2-block net at 8 sims over short games — a few seconds. If `evaluate_checkpoint` exceeds ~30 s on the CI box, mark `test_evaluate_checkpoint_records_results_and_elo` and `test_run_once_*` with `@pytest.mark.slow` (the marker is already registered in `pyproject.toml`); otherwise leave them in the default suite.

- [ ] **Step 6: Run the whole evaluation suite together (no regressions)**

Run: `.venv\Scripts\python -m pytest tests/test_config_m6.py tests/test_players.py tests/test_openings.py tests/test_match.py tests/test_store.py tests/test_ratings.py tests/test_daemon.py -v`
Expected: all passed (Stockfish-gated tests skipped while the binary is absent).

- [ ] **Step 7: Commit**

```powershell
git add chessrl/evaluation/daemon.py scripts/evaluate.py tests/test_daemon.py
git commit -m "feat: evaluator daemon (evaluate_checkpoint, once/watch, EVAL_STOP) + scripts/evaluate.py CLI"
```

---

### Task 8: Stockfish provisioning + live gate

**Files:**
- Create: `scripts/fetch_stockfish.py`
- Modify: `.gitignore` (add `tools/stockfish/`)
- (No new test file — this task provisions the binary, runs the skipif-gated Stockfish tests live, and runs one real ladder evaluation against a real checkpoint. The numbers go in the milestone report.)

`scripts/fetch_stockfish.py` downloads the official Stockfish 17.1 Windows AVX2 release zip from GitHub into `tools/stockfish/`, extracts it, copies the engine binary to `tools/stockfish/stockfish.exe`, and prints the resolved path. This is the binary `default_stockfish_path()` discovers and the Stockfish-gated tests use.

- [ ] **Step 1: Add the tools dir to .gitignore**

Append to `.gitignore`:

```
# Provisioned Stockfish binary (downloaded, not vendored)
tools/stockfish/
```

- [ ] **Step 2: Implement the fetch script**

```python
# scripts/fetch_stockfish.py
"""Provision the pinned Stockfish binary into tools/stockfish/ (gitignored).

Downloads the official Stockfish 17.1 Windows AVX2 release from GitHub, extracts
it, and normalizes the engine to tools/stockfish/stockfish.exe so
default_stockfish_path() discovers it. The exact release URL is pinned so anchor
UCI_Elo calibration is reproducible (a different build silently moves anchors).

Usage:  python scripts/fetch_stockfish.py
        python scripts/fetch_stockfish.py --url <override>   # e.g. a Linux build
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Pinned Stockfish 17.1 Windows x86-64-avx2 build.
DEFAULT_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_17.1/stockfish-windows-x86-64-avx2.zip"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fetch(url: str = DEFAULT_URL) -> Path:
    dest = _repo_root() / "tools" / "stockfish"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "download.zip"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, archive)

    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    # Find the extracted engine executable and normalize its name.
    candidates = [
        p for p in dest.rglob("stockfish*")
        if p.is_file() and (p.suffix.lower() in (".exe", "") and "download" not in p.name)
    ]
    if not candidates:
        raise SystemExit("no stockfish executable found in the archive")
    exe = max(candidates, key=lambda p: p.stat().st_size)   # the real binary is large
    target = dest / ("stockfish.exe" if exe.suffix.lower() == ".exe" or sys.platform == "win32" else "stockfish")
    if exe != target:
        shutil.copy2(exe, target)
    if sys.platform != "win32":
        target.chmod(0o755)
    archive.unlink(missing_ok=True)
    print(f"stockfish ready at {target}")
    return target


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args(argv)
    fetch(args.url)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Provision the binary (implementer runs this)**

```powershell
.venv\Scripts\python scripts\fetch_stockfish.py
```

Expected: prints `stockfish ready at C:\Chess\tools\stockfish\stockfish.exe`. Sanity-check the engine answers UCI:

```powershell
.venv\Scripts\python -c "import chess.engine; e=chess.engine.SimpleEngine.popen_uci(r'tools\stockfish\stockfish.exe'); print(e.id.get('name')); e.quit()"
```

Expected: prints the Stockfish version string (e.g. `Stockfish 17.1`).

- [ ] **Step 4: Run the previously-skipped Stockfish tests live**

Run: `.venv\Scripts\python -m pytest tests/test_players.py -v`
Expected: the two Stockfish tests now PASS (no longer skipped): `test_stockfish_plays_legal_and_records_conditions`, `test_stockfish_nodes_rung_plays_legal`.

- [ ] **Step 5: Run one real ladder evaluation with Stockfish rungs enabled**

Produce a real checkpoint from a short parallel training run (M5 path), then evaluate it with the full ladder (floors + Stockfish):

```powershell
# 1) tiny training run -> a real checkpoint under runs/<run-id>/checkpoints/
.venv\Scripts\python scripts\train.py --parallel --config experiments\smoke.yaml --games 8

# 2) write an eval config that enables Stockfish
@"
eval:
  every_n_checkpoints: 1
  games_per_rung: 2
  agent_simulations: 50
  max_plies: 120
  stockfish_path: tools/stockfish/stockfish.exe
  stockfish_movetime_ms: 50
"@ | Set-Content experiments\eval_live.yaml

# 3) evaluate once
.venv\Scripts\python scripts\evaluate.py --once --runs-root runs --config experiments\eval_live.yaml
```

Expected: prints `evaluated N checkpoint(s)` (N ≥ 1). Then verify the artifacts:

```powershell
.venv\Scripts\python -c "from pathlib import Path; import json; d=sorted(Path('runs').glob('*/elo.jsonl'))[-1]; rows=[json.loads(l) for l in d.read_text().splitlines()]; print(d); print(rows[-1])"
```

Expected: an `elo.jsonl` line with finite `elo` and `nu`; `runs/ladder.sqlite` exists; `runs/<run-id>/eval_games/` holds PGNs including `sf_elo1320` / `sf_nodes1` games. Record in the milestone report: the agent's fitted Elo, `nu`, the number of results in `ladder.sqlite`, and the Stockfish version string from the recorded `conditions`.

> If `experiments\smoke.yaml` does not exist, use any existing tiny experiment YAML in `experiments\`, or fall back to the single-process path: `.venv\Scripts\python scripts\train.py --config <tiny.yaml>` until at least one `ckpt_*.pt` exists under `runs/<run-id>/checkpoints/`.

- [ ] **Step 6: Run the full fast suite (regression gate)**

Run: `.venv\Scripts\python -m pytest --durations=10`
Expected: all fast tests pass — the M1–M5 suite unchanged plus the new M6 tests (config-m6, players incl. the now-passing Stockfish tests, openings, match, store, ratings, daemon). Any `slow`-marked daemon tests are deselected by the default `-m 'not slow'`.

- [ ] **Step 7: Commit**

```powershell
git add scripts/fetch_stockfish.py .gitignore
git commit -m "feat: Stockfish 17.1 provisioning script; gitignore tools/stockfish; live ladder gate"
```

---

## Self-Review Notes

- **Spec coverage (M6):** the full roster — floor (random / greedy / 1-ply-equivalent minimax) + dead-zone rungs (depth-2 minimax with optional eval noise; Stockfish `nodes`=1,100) + pinned anchors (Stockfish `UCI_Elo` 1320/1500/1700) + peers (the run's own prior checkpoints accumulate in the same store and are fitted as unpinned players) (Task 2, Task 7); the 50-position opening suite (Task 3); the match protocol with **exactly-equal colors per pairing** and no eval-time temperature, ply-cap-as-draw (Task 4); the single-writer `ladder.sqlite` (WAL + busy_timeout) with `ladder_inbox` ingest/delete (Task 5); the regularized **Davidson** draw-model joint refit over the whole results graph with pinned anchors and a weak prior giving 100%/0% players finite ratings (Task 6); the evaluator daemon polling every-Nth checkpoint, ingesting the inbox, writing per-run `elo.jsonl` + PGNs, `EVAL_STOP`-aware, `--once`/`watch`/`--run`/`--config` CLI (Task 7); Stockfish provisioning with a pinned release URL, version recorded per match, and the live gate (Task 8). Out-of-scope items (web server/UI/live feed → M7, the arena *writer* → M7, retention policy, TensorBoard, higher anchors) are explicitly stated in the header and not built.
- **Design refinements made (and why):**
  - *Single-board adapter for the agent.* The spec's `NetMCTSPlayer` "wraps so its single-board evaluate adapts the batched evaluator." I implemented `_SingleFromBatched.evaluate(board) -> (policy, value)` calling `evaluate_many([board])`, which is exactly the `.evaluate(board)` interface `ReferenceMCTS` consumes (verified against `chessrl/mcts/reference.py`'s `_expand`). This reuses the permanent reference search for the agent so eval play is provably the same search as the correctness baseline, with `add_noise=False` and argmax-visits (no temperature) per the no-eval-noise spec rule.
  - *z mapping in `play_single`.* `terminal_value` returns the result from the **side-to-move's** perspective; a decisive finished game always has the loser to move, so White's-perspective `z = +1` iff Black is to move at the end, `-1` iff White is. The plan gives both the explanatory branch and the simplified direct form, and `test_play_single_returns_z_and_pgn` plus the greedy/minimax mate tests in adjacent suites pin it.
  - *Davidson in log-strength space.* I fit `theta = r*ln(10)/400` (so `pi = exp(theta)`) with a numerically-stable shifted-logsumexp for `D`, plain gradient ascent with a decaying lr, and a Gaussian prior only on unpinned thetas. This makes the all-wins case finite (Task 6 test) and keeps the 75%-vs-anchor case within ~15 Elo of the closed-form `1000 + 400*log10(3)` (the prior pull is documented and asserted in the correct direction). `nu` is fit in `log` space (positivity); the draws-raise-nu test pins its gradient sign. PRIOR_MEAN=1000 is justified inline as floor calibration.
  - *Anchors pinned, floors fitted.* `evaluate_checkpoint` upserts each rung with `anchor_elo` set only for Stockfish `UCI_Elo` rungs; `store.anchors()` feeds exactly those to `fit_ratings`, which holds them fixed. Floors and `nodes`-rungs are unpinned and rated relative to the anchors and each other — matching the spec's "orderable even if not absolutely calibrated" mid-rung intent.
  - *Inbox malformed-file safety.* `ingest_inbox` deletes only successfully-recorded files and leaves malformed JSON in place (a dedicated test locks this), so a half-written arena drop is retried, never silently lost.
  - *Stockfish robustness.* Per the spec ("UCI engines occasionally hang; the evaluator must not"), `StockfishPlayer.play` runs under a `timeout` and auto-restarts once on any error before failing; options are filtered to what the engine advertises (older builds vary), and `Threads=1`/`Ponder=false` plus the version `id` are recorded in `conditions()` for every match.
  - *Eligible-checkpoint selection.* `eligible_checkpoints` takes every Nth checkpoint **by index** (`ckpts[::N]`) and filters out those already in the `evaluated` table — restart-safe and idempotent, so `--once` is repeatable and `watch` never double-rates a checkpoint.
- **Type/name consistency check against the real M1–M5 signatures I read:**
  - `BatchedNetEvaluator.from_checkpoint(path, network_cfg, device)` and `.evaluate_many(boards) -> (policies (N,4672) float32, values (N,) float32)` — matches `chessrl/model/network.py`; the agent consumes `policies[0]`, `float(values[0])`. ✓
  - `ReferenceMCTS(evaluator, MCTSConfig, rng).search(board, add_noise=False) -> (visits_dict, root_q)`; agent calls `max(visits, key=visits.get)` then `index_to_move(idx, board.turn == chess.BLACK, board)` — matches `reference.py` and `moves.py` (`index_to_move(index, flip, board)`). ✓
  - `terminal_value(board)` returns side-to-move-perspective `+1/0/-1` or `None` (claimable draws terminal) — matches `chessrl/chess_env/game.py`; used for both match termination and the z mapping. ✓
  - `Trainer(net, TrainingConfig, run_dir).save_checkpoint() -> Path` writes `checkpoints/ckpt_{step:08d}.pt` with keys `{step, model, optimizer}` — matches `trainer.py`; the daemon test builds a real checkpoint this way and `from_checkpoint` loads `ckpt["model"]`. ✓
  - `RunConfig.from_json/from_yaml/from_dict/to_json` and the new `EvalConfig`/`eval` field follow the existing frozen-dataclass + `default_factory` + `build(klass, key)` pattern exactly (Task 1), so `asdict`/round-trip works unchanged. ✓
  - `chess.engine.SimpleEngine.popen_uci`, `.play(board, Limit, timeout=...)`, `.configure`, `.options`, `.id`, `.quit()` — all python-chess 1.11.2 APIs (engine import verified present in the venv). ✓
  - `save_pgn` is not reused here (eval games need White/Black/Opening headers), so `match.py` builds PGN via `chess.pgn.Game.from_board` directly — consistent with `pgn_io.py`'s approach, no signature conflict. ✓
- **No placeholders:** every task has complete, runnable test and implementation code. The opening list is the one place the draft shows an intermediate form; the plan explicitly calls that out and supplies the **finalized 50-entry list plus the two-line fix** to type, with the three opening tests as the gate (count==50, all legal, all distinct) — so the implementer ends with a concrete, test-locked suite and no TODO.
- **Known intentional simplifications:** the default ladder enables anchors 1320/1500/1700 only (higher anchors are config-reachable but a sub-1700 agent never scores against 2000+, so enabling them by default only burns wall time); the rating fit is full-batch gradient ascent over the whole store each evaluation (fine at hobby scale — thousands of results; if the store grows to millions, batching or a Newton step is a later optimization, not an M6 need); `nodes=10` from the spec's "(1, 10, 100, …)" is omitted from the default node rungs (1 and 100 bracket the dead zone adequately; trivially extendable via `_NODE_RUNGS`).
