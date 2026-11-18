"""The cap on live background grafts is the harness's `max_live_background` (default 4): the
machines' clone rule reads it, the clone skill's text says it, and the cancel tool takes up to
twice it. The default leaves every prompt byte as it was."""

import asyncio
import json

import pytest

from _util import clone_in_turn, clones
from graft.db import Store
from graft.harness import Harness, cancel_ids
from graft.machine import MAX_LIVE_BACKGROUND
from graft.providers import FakeProvider
from graft.skills import CANCEL_TOOL, CATALOG, CLONE_INSTRUCTIONS, MAX_CANCEL_IDS, install
from test_parallel import REQUEST, clone_entry, clone_use, text


def make(tmp_path, cap=None):
    store = Store(str(tmp_path / "p.db"))
    install(store) if cap is None else install(store, max_live_background=cap)
    kw = {} if cap is None else {"max_live_background": cap}
    return store, Harness(store, FakeProvider(lambda *a: text("ok")), seeder=None, **kw)


def bg(n, skill="website"):
    return clone_use(*[clone_entry(skills=[skill], focus=False)] * n)


async def fill(h, store, first, then):
    """Clone `first` background grafts, then one more per entry of `then`; the result texts."""
    await h.start()
    store.add_turn("o", "user", "text", REQUEST)
    out = [await clone_in_turn(h=h, use=bg(first))]
    for skill in then:
        out.append(await clone_in_turn(h=h, use=bg(1, skill=skill)))
    await h.machines.stop()
    return out


def test_a_cap_of_five_takes_a_fifth_live_background_graft_across_clone_calls_and_refuses_a_sixth(tmp_path):
    store, h = make(tmp_path=tmp_path, cap=5)
    (four, _), (fifth, _), (sixth, msg) = asyncio.run(fill(h=h, store=store, first=4, then=["slides", "calendar"]))
    assert len(four) == 4 and fifth == ["c5"] and sixth is None
    assert "at most 5 background branches may be live" in msg and len(clones(store)) == 5


def test_the_default_cap_still_refuses_a_fifth_live_background_graft(tmp_path):
    store, h = make(tmp_path=tmp_path)
    (four, _), (fifth, msg) = asyncio.run(fill(h=h, store=store, first=4, then=["slides"]))
    assert len(four) == MAX_LIVE_BACKGROUND == 4 and fifth is None
    assert "at most 4 background branches may be live" in msg


def test_the_default_cap_leaves_the_clone_skill_and_cancel_tool_bytes_as_they_were(tmp_path):
    store, _ = make(tmp_path=tmp_path)
    clone = next(s for s in CATALOG if s.name == "branch")
    row = store.con.execute("SELECT instructions FROM skills WHERE name='branch'").fetchone()[0]
    assert row == CLONE_INSTRUCTIONS == clone.instructions
    assert "At most four background branches of yours may be live at a time." in row
    assert stored_cancel(store) == {**CANCEL_TOOL["input_schema"], "description": CANCEL_TOOL["description"]}
    assert stored_cancel(store)["properties"]["cancel"]["maxItems"] == MAX_CANCEL_IDS == 8


def stored_cancel(store):
    return json.loads(store.con.execute("SELECT input_schema FROM tools WHERE name='cancel'").fetchone()[0])


def test_a_configured_cap_is_said_in_the_clone_skill_and_bounds_the_cancel_call(tmp_path):
    store, h = make(tmp_path=tmp_path, cap=5)
    row = store.con.execute("SELECT instructions FROM skills WHERE name='branch'").fetchone()[0]
    assert "At most 5 background branches of yours may be live at a time" in row and "At most four" not in row
    ids = [f"c{i}" for i in range(1, 12)]
    assert cancel_ids({"cancel": ids[:10]}, limit=10)[1] == []
    assert cancel_ids({"cancel": ids}, limit=10)[1]  # past twice the cap: refused
    assert h.max_cancel_ids == 10 and stored_cancel(store)["properties"]["cancel"]["maxItems"] == 10


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "4", None])
def test_the_cap_must_be_a_whole_number_of_at_least_one(tmp_path, bad):
    store = Store(str(tmp_path / "p.db"))
    install(store)
    with pytest.raises((ValueError, TypeError)):
        Harness(store, FakeProvider(lambda *a: text("ok")), seeder=None, max_live_background=bad)


def test_a_cap_the_installed_clone_skill_does_not_say_is_refused(tmp_path):
    store = Store(str(tmp_path / "p.db"))
    install(store)  # the default text says four
    with pytest.raises(ValueError, match="max_live_background"):
        Harness(store, FakeProvider(lambda *a: text("ok")), seeder=None, max_live_background=5)


def test_a_cancel_tool_that_does_not_take_twice_the_cap_is_refused(tmp_path):
    store = Store(str(tmp_path / "p.db"))
    install(store)
    schema = stored_cancel(store)
    schema["properties"]["cancel"]["maxItems"] = 9
    store.con.execute("UPDATE tools SET input_schema=? WHERE name='cancel'", (json.dumps(schema),))
    with pytest.raises(ValueError, match="max_live_background"):
        Harness(store, FakeProvider(lambda *a: text("ok")), seeder=None)
