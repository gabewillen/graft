"""A branch's id (spec: Clone, Ids): given by the model or drawn by the harness, unique in the
conversation and never reused; every note, result and refusal names a graft by it, the branch is
told it, and `stop` resolves ids only through it, never a timeline id."""

import asyncio
import itertools

import pytest

from _util import clone_in_turn
from graft.db import Store
from graft.harness import MAX_NAME_DRAWS, Harness, new_names
from graft.providers import FakeProvider
from graft.skills import install
from test_parallel import REQUEST, text


def make(tmp_path, names=None, policy=None):
    store = Store(str(tmp_path / "ids.db"))
    install(store)
    h = Harness(store=store, provider=FakeProvider(policy or (lambda *a: text("ok"))), seeder=None, names=names)
    return store, h


def fork(*, current=None, detached=(), id="t"):
    inp = {"detached": list(detached)} if detached else {}
    if current is not None:
        inp["current"] = current
    return {"id": id, "name": "fork", "input": inp}


def test_the_default_source_draws_six_hex_characters():
    draw = new_names()  # imported before the tests' counted source replaces it
    ids = {draw() for _ in range(50)}
    assert all(len(i) == 6 and int(i, 16) >= 0 for i in ids) and len(ids) > 40


def test_ids_given_and_drawn_name_the_branch_everywhere_and_never_its_timeline_id(tmp_path):
    drawn = iter(["3fa9c1", "77aa00"])
    store, h = make(tmp_path, names=lambda: next(drawn))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        (c1, c2), msg = await clone_in_turn(h=h, use=fork(detached=[{"id": "site", "skills": ["website"]},
                                                                   {"skills": ["slides"]}]))
        assert (c1, c2) == ("c1", "c2")  # timeline ids stay graft's own
        assert msg == ("forked -> site (detached, running; skills ['website', 'detached']), "
                       "3fa9c1 (detached, running; skills ['slides', 'detached'])")
        assert store.name_of(c1) == "site" and store.timeline_named("3fa9c1") == c2
        assert "This branch of yours has the id site." in h.system(c1)
        assert "has the id" not in h.system("o")
        store.add_turn(c1, "assistant", "text", "Site built.")
        await h.do_join(child=c1, reason="done")
        await h.drain_and_wake("o")
        notes = [r["content"] for r in store.view("o") if r["harness"]]
        assert any(n.startswith("(harness note: your branch site merged, reason=done;") for n in notes), notes
        assert any("Still running: 3fa9c1 (skills: slides)." in n for n in notes), notes
        assert not any(__import__("re").search(r"\bc[12]\b", n) for n in notes), notes
        await h.machines.stop()

    asyncio.run(go())


def test_an_id_is_never_reused_and_a_colliding_draw_is_drawn_again(tmp_path):
    drawn = iter(["site", "site", "deck"])  # the first two collide with the given id
    store, h = make(tmp_path, names=lambda: next(drawn))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        (c1,), _ = await clone_in_turn(h=h, use=fork(detached=[{"id": "site", "skills": ["website"]}]))
        children, msg = await clone_in_turn(h=h, use=fork(detached=[{"id": "site", "skills": ["slides"]}]))
        assert children is None and "detached[0]: id 'site' is already used in this conversation" in msg
        (c2,), msg = await clone_in_turn(h=h, use=fork(detached=[{"skills": ["slides"]}]))
        assert store.name_of(c2) == "deck" and msg.startswith("forked -> deck")
        await h.do_join(child=c1, reason="done")
        await h.drain_and_wake("o")
        children, msg = await clone_in_turn(h=h, use=fork(detached=[{"id": "site", "skills": ["calendar"]}]))
        assert children is None and "already used" in msg  # merged: still never reused
        await h.machines.stop()

    asyncio.run(go())


@pytest.mark.parametrize("source", ["collides", "not an id"])
def test_an_id_source_that_cannot_give_an_id_refuses_the_call(tmp_path, source):
    """Draws are bounded: a source that keeps colliding, or gives something that is not an id, is
    an error of the preparation, so the call is refused and nothing is made."""
    store, h = make(tmp_path, names=(lambda: "site") if source == "collides" else (lambda: "Not An Id"))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        if source == "collides":
            await clone_in_turn(h=h, use=fork(detached=[{"id": "site", "skills": ["website"]}]))
        children, msg = await clone_in_turn(h=h, use=fork(detached=[{"skills": ["slides"]}]))
        assert children is None and msg.startswith("fork refused; no branch was started: internal error")
        assert (f"no free branch id after {MAX_NAME_DRAWS} draws" if source == "collides"
                else "which is not a branch id") in msg
        assert store.con.execute("SELECT count(*) FROM transactions WHERE kind='clone' AND status='open'"
                                 ).fetchone()[0] == 0
        await h.machines.stop()

    asyncio.run(go())


def test_stop_takes_branch_ids_and_a_timeline_id_names_nothing(tmp_path):
    """The model stops a branch by the id it knows; c1 is a timeline id it never saw (the branch is
    named `site`), so it names nothing."""
    gate = __import__("threading").Event()

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "fork" not in names:
            gate.wait(10)
            return text("Built.")
        if last["type"] == "text" and not last.get("harness") and last["text"] == REQUEST:
            return [{"type": "tool_use", "id": "f1", "name": "fork",
                     "input": {"detached": [{"id": "site", "skills": ["website"]}]}}]
        if last["type"] == "text" and not last.get("harness") and last["text"] == "Stop it.":
            return [{"type": "tool_use", "id": "s1", "name": "stop", "input": {"ids": ["c1", "site"]}}]
        return text("Okay.")

    store, h = make(tmp_path, policy=policy)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.user_says("Stop it.")
        gate.set()
        await h.settle()
        await h.close()

    asyncio.run(go())
    result = [r["content"] for r in store.view("o") if r["kind"] == "tool_result" and "stopped ->" in r["content"]]
    assert len(result) == 1 and '"stopped -> site (what you did in it' in result[0]
    assert "c1: not stopped: no such branch" in result[0]
    assert store.timeline("c1")["join_reason"] == "cancelled"
    notes = [r["content"] for r in store.view("o") if r["harness"]]
    assert any("site (stopped)" in n and "You stopped site" in n for n in notes), notes
