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

    def upsert_player(self, name: str, kind: str, anchor_elo=None) -> None:
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
