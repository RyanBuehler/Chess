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
