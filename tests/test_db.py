import json
import sqlite3

import pytest

from graft.db import Store, render, validate
from graft.skills import install


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    install(s)
    s.create_timeline("o", None, ["clone"])
    return s


def row(rid, role, kind, content, owner="o", tool=None):
    return {"row_id": rid, "owner_timeline": owner, "role": role, "kind": kind, "tool_name": tool,
            "content": content if isinstance(content, str) else json.dumps(content)}


def use(rid, uid, owner="o"):
    return row(rid, "assistant", "tool_use", {"id": uid, "input": {}}, owner, "site_write")


def res(rid, uid, owner="o"):
    return row(rid, "user", "tool_result", {"tool_use_id": uid, "content": "ok"}, owner, "site_write")


def test_turns_and_context_are_append_only(store):
    rid = store.add_turn("o", "user", "text", "hi")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.con.execute("UPDATE turns SET content='x' WHERE row_id=?", (rid,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.con.execute("DELETE FROM turns")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.con.execute("DELETE FROM context")


def test_pointer_dedup_and_stored_once(store):
    rid = store.add_turn("o", "user", "text", "hi")
    store.commit_clone("o", "c1", [rid], ["website", "join"])
    assert store.append("c1", rid) is None  # already pointed: skipped
    assert [r["row_id"] for r in store.view("c1")] == [rid]
    assert store.con.execute("SELECT count(*) FROM turns").fetchone()[0] == 1
    # a join that re-selects a row the source already holds does not duplicate it
    store.commit_join("o", "[join c1]", [rid])
    assert [r["row_id"] for r in store.view("o")].count(rid) == 1


def test_render_merges_consecutive_roles():
    msgs = render([row(1, "user", "text", "a"), row(2, "user", "text", "b"), use(3, "u1"), res(4, "u1")])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert len(msgs[0]["content"]) == 2


def test_valid_view():
    rows = [row(1, "user", "text", "hi"), use(2, "u1"), res(3, "u1"), row(4, "user", "text", "next")]
    assert validate(rows, "o") == []


@pytest.mark.parametrize("rows,needle", [
    ([], "empty"),
    ([row(1, "assistant", "text", "x"), row(2, "user", "text", "y")], "start with a user"),
    ([row(1, "user", "text", "x"), use(2, "u1"), row(3, "user", "text", "y")], "has no tool_result"),
    ([row(1, "user", "text", "x"), res(2, "zz")], "has no tool_use"),
    ([row(1, "user", "text", "x"), use(2, "u1"), row(3, "user", "text", "y"), res(4, "u1")],
     "must come before text"),
    ([row(1, "user", "text", "x"), row(2, "assistant", "text", "y")], "end on a user turn"),
    ([row(1, "user", "text", "x"), row(2, "assistant", "thinking", {"thinking": "", "signature": "s"}, "c1"),
      row(3, "assistant", "text", "t"), row(4, "user", "text", "y")], "bound to its own conversation"),
    ([row(1, "user", "text", "x"), row(2, "assistant", "thinking", {"thinking": "", "signature": "s"}),
      row(3, "user", "text", "y")], "no text or tool_use"),
    ([row(1, "user", "text", "x"), row(1, "user", "text", "x")], "appears twice"),
])
def test_violations(rows, needle):
    v = validate(rows, "o")
    assert any(needle in x for x in v), v


def test_trigger_must_be_last():
    rows = [row(1, "user", "text", "trig"), row(2, "assistant", "text", "a"), row(3, "user", "text", "b")]
    assert any("trigger row 1" in x for x in validate(rows, "o", trigger=1))
    assert validate(rows, "o", trigger=3) == []


def test_running_leaf_and_descendants(store):
    rid = store.add_turn("o", "user", "text", "hi")
    assert store.running_leaf() == "o"
    store.commit_clone("o", "c1", [rid], ["website", "join"])
    assert store.running_leaf() == "c1"
    assert store.descendants("o") == ["c1"] and store.descendants("c1") == []
    store.set_status("c1", "joined")
    assert store.running_leaf() == "o"


def test_tools_only_through_skills(store):
    rid = store.add_turn("o", "user", "text", "hi")
    store.commit_clone("o", "c1", [rid], ["website", "join"])
    assert {t["name"] for t in store.tools_for("c1")} == {"site_write", "site_read", "join", "load_skill"}
    assert "site_write" not in {t["name"] for t in store.tools_for("o")}
