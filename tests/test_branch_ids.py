"""A branch's id (spec: Clone, Ids): given by the model or drawn by the harness, unique in the
conversation and never reused; every note, result and refusal names a graft by it, the branch is
told it, and `stop` resolves ids only through it, never a timeline id."""

import asyncio
import random
import re

import pytest

from _util import clone_in_turn
from graft.db import Store
from graft.harness import GRAFT_ID, MAX_NAME_DRAWS, Harness, new_names
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


def test_the_default_source_draws_six_lowercase_hex_characters():
    draw = new_names(rng=random.Random(7))  # imported before the tests' counted source replaces it
    ids = [draw() for _ in range(50)]
    assert all(re.fullmatch(r"[0-9a-f]{6}", i) and GRAFT_ID.fullmatch(i) for i in ids), ids
    assert ids == [new_names(rng=random.Random(7))() for _ in range(1)] + ids[1:]  # seeded: replayable
    assert len(set(ids)) > 45


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


def test_error_lines_name_a_branch_by_its_id_even_when_another_branchs_timeline_id_is_that_id(tmp_path):
    """One namespace for every audience: the user's harness error lines name a branch by the id the
    model knows (`site`, timeline c1), never by its timeline id, so an id relayed from an error line
    (here `c1`, the model's name for timeline c2) never names a different branch."""
    said = []
    store = Store(str(tmp_path / "err.db"))
    install(store)
    h = Harness(store=store, provider=FakeProvider(lambda *a: text("ok")), seeder=None, emit=said.append)

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        (site,), _ = await clone_in_turn(h=h, use=fork(detached=[{"id": "site", "skills": ["website"]}]))
        (named_c1,), _ = await clone_in_turn(h=h, use=fork(detached=[{"id": "c1", "skills": ["slides"]}]))
        assert (site, named_c1) == ("c1", "c2") and store.name_of("c2") == "c1"
        real, failed = store.commit_join, []

        def commit_join(*a, **kw):
            if not failed:
                failed.append(True)
                raise OSError("disk full")
            return real(*a, **kw)
        store.commit_join = commit_join
        store.add_turn(site, "assistant", "text", "Site built.")
        await h.do_join(child=site, reason="done")
        await h.drain_and_wake("o")
        await h.machines.stop()

    asyncio.run(go())
    lines = [line for line in said if line.startswith("(harness error: committing the merge")]
    assert lines and "branch site" in lines[0] and "c1" not in lines[0], said


def test_a_request_handed_back_any_number_of_times_may_still_be_forked_for(tmp_path):
    """No hand-back count: a current branch that hands the same request back ten times in a row
    is forked for again each time (only the host bounds this)."""
    rounds = 10

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "fork" not in names:
            return [{"type": "tool_use", "id": f"m{len(messages)}", "name": "merge", "input": {"reason": "out_of_scope"}}]
        forks = sum(1 for m in messages for b in m["content"] if b["type"] == "tool_use" and b["name"] == "fork")
        if forks >= rounds:
            return text("I could not route that.")
        if last["type"] == "tool_result":
            return text("ok")
        return [{"type": "tool_use", "id": f"f{forks}", "name": "fork", "input": {"current": {"skills": ["website"]}}}]

    store, h = make(tmp_path, policy=policy)

    async def go():
        await h.start()
        out = await h.user_says(REQUEST)
        assert out[-1] == "I could not route that."
        await h.close()

    asyncio.run(go())
    reasons = [store.timeline(f"c{i}")["join_reason"] for i in range(1, rounds + 1)]
    assert reasons == ["out_of_scope"] * rounds  # each hand-back was forked for again
    assert store.timeline(f"c{rounds + 1}") is None
