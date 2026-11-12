"""Parallel grafts offline: one clone call with several entries, background grafts that never
speak, the join queue that commits only while the source is at rest, and the wake run.

Model calls run in worker threads, so finish order is racy: the scripted policies block on
threading.Event gates, and assertions that span more than one graft are order-free unless a gate
fixes the order."""

import asyncio
import threading

import pytest

from _util import clone_in_turn
from graft.db import Store, validate
from graft.harness import Harness, clone_entries
from graft.providers import FakeProvider
from graft.skills import install

REQUEST = "Build a website and a slide deck for the dental practice."


def text(t):
    return [{"type": "text", "text": t}]


def entry(skills, focus, quotes=(REQUEST,), purpose="p"):
    return {"purpose": purpose, "skills": skills, "focus": focus, "seed_quotes": list(quotes)}


def clone_use(*entries, id="t"):
    return {"id": id, "name": "clone", "input": {"clone": list(entries)}}


def user_texts(messages):
    return [b["text"] for m in messages if m["role"] == "user" for b in m["content"] if b["type"] == "text"]


class Script:
    """A scripted original plus scripted grafts. The original clones `entries` for the request,
    says "Started." after the clone result, and reports on a wake note. A graft waits for its
    gate, then writes one text reply (which, in the background, joins it `done`)."""

    def __init__(self, entries, graft=None):
        self.entries = entries
        self.gates = {"website": threading.Event(), "slides": threading.Event(), "root": threading.Event()}
        self.graft = graft or (lambda skill, messages: text(f"{skill} finished"))
        self.root_gate_on_reply = False

    def open(self, *names):
        for n in names or self.gates:
            self.gates[n].set()

    def __call__(self, system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "clone" in names:
            if last["type"] == "tool_result":
                if self.root_gate_on_reply:
                    assert self.gates["root"].wait(5)
                return text("Started.")
            if last["text"].startswith("(harness note: background clone"):
                return text("Report: " + last["text"][:60])
            return [clone_use(*self.entries, id=f"c{len(messages)}") | {"type": "tool_use"}]
        skill = "website" if "site_write" in names else "slides"
        assert self.gates[skill].wait(5)
        return self.graft(skill, messages)


def make(tmp_path, policy, **kw):
    store = Store(str(tmp_path / "p.db"))
    install(store)
    said = []
    h = Harness(store, FakeProvider(policy), seeder=None, emit=said.append, **kw)
    return store, h, said


def join_records(store, tl="o"):
    return [r["content"] for r in store.view(tl) if r["content"].startswith("(harness note: clone ")]


# ------------------------------------------------------------------ the clone call is one batch
def test_clone_entries_shape():
    assert clone_entries({"purpose": "p", "skills": ["website"]})[1]  # not a list of entries
    assert clone_entries({"clone": []})[1]
    _, v = clone_entries({"clone": [{"purpose": "p", "skills": ["website"]}]})
    assert v == ["clone[0]: focus is required and must be true or false"]
    _, v = clone_entries({"clone": [entry(["website"], True), entry(["slides"], True)]})
    assert len(v) == 1 and "at most one entry may have focus=true" in v[0]
    es, v = clone_entries({"clone": [entry(["website"], True), entry(["slides"], False)]})
    assert v == [] and len(es) == 2


def test_refused_batch_is_atomic(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("ok"))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        for bad in (clone_use(entry(["website"], True), entry(["slides"], True)),          # two focus
                    clone_use(entry(["website"], False), {"purpose": "p", "skills": ["slides"],
                                                          "seed_quotes": [REQUEST]}),        # focus missing
                    clone_use(entry(["website"], False), entry(["nope"], False)),            # one bad entry
                    clone_use(entry(["website"], False), entry(["slides"], False, ["no such message"]))):
            children, msg = await clone_in_turn(h, bad, retry_left=True)
            assert children is None and "no clone was created" in msg and "Your messages" in msg
            assert list(h.machines.timelines) == ["o"] and h.machines.state("o") == "active"
            assert store.con.execute("SELECT count(*) FROM timelines").fetchone()[0] == 1
            assert not store.con.execute("SELECT 1 FROM transactions WHERE status='committed'").fetchone()
        assert "clone[1]: unknown skill 'nope'" in msg or "clone[1]: seed quote" in msg  # per-entry violations
        children, msg = await clone_in_turn(h, clone_use(entry(["website"], False), entry(["slides"], False)),
                                           retry_left=False)
        assert children == ["c1", "c2"]  # distinct ids for one batch
        assert "c1 (background, running" in msg and "c2 (background, running" in msg
        assert [h.machines.state(t) for t in ("o", "c1", "c2")] == ["active", "background", "background"]
        assert store.running_leaf() == "o"  # the conversation stays with the source
        h.machines.check()
        assert {t["name"] for t in store.tools_for("c1")} == {"load_skill", "join", "site_write", "site_read"}
        assert "background:" in h.system("c1") and "join:" not in h.system("c1")
        assert store.con.execute("SELECT count(*) FROM transactions WHERE kind='clone' AND "
                                 "status='committed'").fetchone()[0] == 2
        await h.machines.stop()

    asyncio.run(go())


def test_background_and_join_are_not_selectable_skills(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("ok"))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        children, msg = await clone_in_turn(h, clone_use(entry(["background"], False)), retry_left=False)
        assert children is None and "'background' is linked by the harness" in msg
        await h.machines.stop()

    asyncio.run(go())


# ------------------------------------------------------------------ background grafts end to end
def test_source_keeps_the_conversation_and_is_woken_when_grafts_join(tmp_path):
    script = Script([entry(["website"], False), entry(["slides"], False)])
    store, h, said = make(tmp_path, script)

    async def go():
        await h.start()
        out = await h.user_says(REQUEST)
        assert out == ["Started."]  # the source answered while both grafts are still out
        assert [h.machines.state(t) for t in ("o", "c1", "c2")] == ["active", "background", "background"]
        assert store.running_leaf() == "o" and h.machines.slot.holder == "o"
        h.machines.check()
        assert join_records(store) == []
        script.open()
        out = await h.settle()
        assert out and all(o.startswith("Report: (harness note: background clone") for o in out)
        assert sorted(c for w in h.wakes for c, _ in w["joined"]) == ["c1", "c2"]
        assert all(r == "done" for w in h.wakes for _, r in w["joined"])  # a text-only turn joined done
        assert [h.machines.state(t) for t in ("o", "c1", "c2")] == ["active", "joined", "joined"]
        assert len(join_records(store)) == 2
        await h.machines.stop()

    asyncio.run(go())
    view = store.view("o")
    assert validate(view, "o", require_user_end=False) == []
    replies = {r["content"]: r["owner_timeline"] for r in view if r["content"].endswith(" finished")}
    assert replies == {"website finished": "c1", "slides finished": "c2"}  # joined by pointer, with provenance
    # a background graft never speaks to the user: its text is in its timeline and the source's memory only
    assert said == h.out and not any("finished" in s for s in said)
    for c in ("c1", "c2"):
        assert store.timeline(c)["join_reason"] == "done" and store.timeline(c)["join_trigger"] is None
    # usage is still attributed per role although calls run in worker threads
    tags = h.provider.by_tag
    assert tags["original:clone_tx"]["calls"] == 1 and tags["clone:c1"]["calls"] == 1 and tags["clone:c2"]["calls"] == 1
    assert "untagged" not in tags


def test_joins_wait_for_the_source_to_be_at_rest_and_one_wake_covers_them(tmp_path):
    script = Script([entry(["website"], False), entry(["slides"], False)])
    script.root_gate_on_reply = True  # the source is mid-turn until the gate opens
    script.open("website", "slides")
    store, h, _ = make(tmp_path, script)

    async def go():
        await h.start()
        turn = asyncio.create_task(h.user_says(REQUEST))
        await h.join_queued("c1")
        await h.join_queued("c2")
        # both grafts are frozen and queued, but nothing was committed into the busy source
        assert h.machines.state("c1") == h.machines.state("c2") == "queued"
        assert store.timeline("c1")["status"] == "queued"  # mirrored: queued is not joined
        assert sorted(h.machines.queued("o")) == ["c1", "c2"] and h.machines.phase("o") == "turn"
        assert await h.machines.try_begin_drain("o") is None  # the machine refuses a commit mid-turn
        assert join_records(store) == [] and h.wakes == []
        assert store.view("o")[-1]["kind"] == "tool_result"  # mid-turn: between tool result and reply
        script.open("root")
        out = await turn
        assert out[0] == "Started." and len(out) == 2 and out[1].startswith("Report:")
        assert len(h.wakes) == 1 and sorted(c for c, _ in h.wakes[0]["joined"]) == ["c1", "c2"]
        texts = [r["content"] for r in store.view("o") if r["kind"] == "text"]
        assert texts.index("Started.") < min(texts.index(x) for x in join_records(store))  # after its turn
        assert await h.settle() == []
        await h.machines.stop()

    asyncio.run(go())
    assert validate(store.view("o"), "o", require_user_end=False) == []


def test_queued_joins_commit_before_the_next_user_message_without_a_wake(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Noted."))

    async def go():
        await h.start()
        req = store.add_turn("o", "user", "text", REQUEST)
        store.add_turn("o", "assistant", "text", "Started.")
        (c1,), _ = await clone_in_turn(h, clone_use(entry(["website"], False)), retry_left=False)
        done = store.add_turn(c1, "assistant", "text", "Site built.")
        await h.do_join(c1, "done")  # the graft's join event: frozen and queued, nothing committed
        assert h.machines.state(c1) == "queued" and h.machines.queued("o") == [c1]
        assert join_records(store) == []
        out = await h.user_says("Thanks. Anything else?")
        assert out == ["Noted."] and h.wakes == []
        assert h.machines.state(c1) == "joined" and store.timeline(c1)["status"] == "joined"
        assert h.machines.queued("o") == []
        rows = store.view("o")
        tail = [r["row_id"] for r in rows][-4:]
        assert tail[1] == done and rows[-2]["content"] == "Thanks. Anything else?"  # record, reply, message
        assert rows[-4]["content"].startswith("(harness note: clone c1 joined, reason=done")
        assert req in store.pointed("o") and [r["row_id"] for r in rows].count(req) == 1  # the seed is not re-joined
        await h.machines.stop()

    asyncio.run(go())


def test_background_joins_commit_before_a_focus_join_so_its_trigger_stays_last(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Routing that."))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        (c1, c2), msg = await clone_in_turn(h, clone_use(entry(["website"], True), entry(["slides"], False)),
                                           retry_left=False)
        assert "c1 (focus, active" in msg and "c2 (background, running" in msg
        assert [h.machines.state(t) for t in ("o", c1, c2)] == ["cloned", "active", "background"]
        assert store.running_leaf() == c1  # the focus graft, not the newer background graft
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": msg}, "clone")
        deck = store.add_turn(c2, "assistant", "text", "Deck built.")
        await h.do_join(c2, "done")
        assert h.machines.queued("o") == [c2] and join_records(store) == []  # the source is suspended
        store.add_turn(c1, "assistant", "text", "Site built.")
        trig = store.add_turn(c1, "user", "text", "Now email it to Bob.")
        await h.do_join(c1, "out_of_scope")
        rows = store.view("o")
        ids = [r["row_id"] for r in rows]
        assert [x.split()[3] for x in join_records(store)] == ["c2", "c1"]  # background first
        assert ids.index(deck) < ids.index(trig) and ids[-2] == trig  # trigger last, then the source's answer
        assert rows[-1]["content"] == "Routing that." and h.wakes == []  # the trigger woke it; no wake note
        assert h.machines.state("o") == "active" and h.machines.queued("o") == []
        await h.machines.stop()

    asyncio.run(go())
    assert validate(store.view("o"), "o", require_user_end=False) == []


def test_focus_done_with_queued_background_joins_wakes_the_source_once(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Both are finished."))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        (c1, c2), msg = await clone_in_turn(h, clone_use(entry(["website"], True), entry(["slides"], False)),
                                           retry_left=False)
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": msg}, "clone")
        store.add_turn(c2, "assistant", "text", "Deck built.")
        await h.do_join(c2, "done")
        store.add_turn(c1, "assistant", "text", "Site built.")
        await h.do_join(c1, "done")
        assert h.wakes == [{"source": "o", "joined": [(c2, "done")], "report": True}]  # to report, not to clone
        assert h.out == ["Both are finished."] and h.machines.phase("o") == "idle"
        await h.machines.stop()

    asyncio.run(go())


# ------------------------------------------------------------------ a background graft always joins
def test_background_step_budget_and_empty_response_join_failed(tmp_path):
    def graft(skill, messages):
        if skill == "slides":
            return []  # empty response
        return [{"type": "tool_use", "id": f"r{len(messages)}", "name": "site_read", "input": {"page": "x"}}]

    script = Script([entry(["website"], False), entry(["slides"], False)], graft)
    script.open()
    store, h, _ = make(tmp_path, script, max_steps=3)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.settle()
        assert store.timeline("c1")["join_reason"] == "failed" and store.timeline("c2")["join_reason"] == "failed"
        assert h.machines.background() == [] and h.machines.queued("o") == []
        assert len(h.reclone_block["o"]) == 1 and h.clone_blocked("o", 0)  # no re-clone for it
        errors = [r["content"] for r in store.view("o") if "failed; its timeline is frozen" in r["content"]]
        assert len(errors) == 2
        await h.user_says("Try again please.")  # a new message may clone again
        assert "c3" in h.machines.timelines
        script.open()
        await h.settle()
        await h.machines.stop()

    asyncio.run(go())
    assert validate(store.view("o"), "o", require_user_end=False) == []


def test_background_mechanical_error_joins_failed(tmp_path):
    def graft(skill, messages):
        raise ValueError("provider exploded")

    script = Script([entry(["website"], False)], graft)
    script.open()
    store, h, _ = make(tmp_path, script)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.settle()
        assert store.timeline("c1")["join_reason"] == "failed"
        assert h.background_errors == [("c1", "ValueError: provider exploded")]
        await h.machines.stop()

    asyncio.run(go())


def test_background_silent_done_is_refused_until_it_writes_its_result(tmp_path):
    steps = iter([
        [{"type": "tool_use", "id": "j1", "name": "join", "input": {"reason": "done"}}],
        [{"type": "text", "text": "Site built with defaults."},
         {"type": "tool_use", "id": "j2", "name": "join", "input": {"reason": "done"}}],
    ])
    script = Script([entry(["website"], False)], lambda skill, messages: next(steps))
    script.open()
    store, h, _ = make(tmp_path, script)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.settle()
        await h.machines.stop()

    asyncio.run(go())
    assert h.silent_done_refusals == ["c1"]
    refusal = [r for r in store.view("c1") if r["kind"] == "tool_result"][0]
    assert "you have not written your result yet" in refusal["content"]
    assert store.timeline("c1")["join_reason"] == "done"
    assert "Site built with defaults." in [r["content"] for r in store.view("o")]


def test_one_clone_call_per_message_and_no_clone_in_a_wake(tmp_path):
    """An original that always wants to clone: the second call in a turn and a call in the wake
    run are refused, so background work cannot loop."""
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "clone" in names:
            if last["type"] == "tool_result" and "not executed" in str(last["content"]):
                return text("Okay.")
            return [clone_use(entry(["website"], False), id=f"c{len(messages)}") | {"type": "tool_use"}]
        return text("site finished")

    store, h, _ = make(tmp_path, policy)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.settle()
        await h.machines.stop()

    asyncio.run(go())
    assert [t for t in h.machines.timelines] == ["o", "c1"]
    refusals = [r["content"] for r in store.view("o") if r["kind"] == "tool_result" and "not executed" in r["content"]]
    assert len(refusals) == 2 and "no clone calls left" in refusals[0] and "woken to report" in refusals[1]


def test_running_leaf_ignores_background_grafts(tmp_path):
    from graft.db import Graft
    store = Store(str(tmp_path / "l.db"))
    install(store)
    store.create_timeline("o", None, ["clone"])
    rid = store.add_turn("o", "user", "text", "hi")
    store.commit_clone("o", [Graft(child="c1", seed=[rid], skills=["website", "background"], focus=False)])
    assert store.timeline("c1")["status"] == "background" and store.running_leaf() == "o"
    store.commit_clone("o", [Graft(child="c2", seed=[rid], skills=["slides", "join"], focus=True), Graft(child="c3", seed=[rid], skills=["calendar", "background"], focus=False)])
    assert store.running_leaf() == "c2"
    assert store.next_clone_ids(3) == ["c4", "c5", "c6"]
    with pytest.raises(Exception):  # one transaction: a bad graft rolls the whole batch back
        store.commit_clone("o", [Graft(child="c7", seed=[rid], skills=["website", "join"], focus=True), Graft(child="c8", seed=[rid], skills=["nope"], focus=False)])
    assert store.timeline("c7") is None


def test_parallel_experiment_runs_offline(capsys):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import parallel

    async def go():
        return [await parallel.run_arm(arm, live=False, latency=0.0, trace=False)
                for arm in ("sequential", "parallel")]

    seq, par = asyncio.run(go())
    assert parallel.report([seq, par], live=False, latency=0.0)
    assert all(seq["checks"].values()) and all(par["checks"].values())
    assert (seq["focus"], seq["background"], seq["clone_calls"], seq["wakes"]) == (3, 0, 3, 0)
    assert (par["focus"], par["background"], par["clone_calls"]) == (0, 3, 1) and par["wakes"] >= 1
    assert [(s, j) for _, s, j in par["clones"]] == [("joined", "done")] * 3
    assert par["replies"][0].startswith("I have started") and len(par["replies"]) == 1 + par["wakes"]
    assert "offline numbers show the mechanics only" in capsys.readouterr().out


# ------------------------------------------------------------------ review defects
def test_a_failed_wake_is_visible_when_it_happens_and_still_raises_at_settle(tmp_path):
    script = Script([entry(["website"], False)])
    broken = {"on": True}

    def policy(system, messages, tools):
        last = messages[-1]["content"][-1]
        if broken["on"] and last["type"] == "text" and last["text"].startswith("(harness note: background clone"):
            raise ValueError("wake exploded")
        if last["type"] == "text" and last["text"] == "Is it done?":
            return text("It joined.")
        return script(system, messages, tools)

    store, h, said = make(tmp_path, policy)

    async def go():
        await h.start()
        assert await h.user_says(REQUEST) == ["Started."]
        script.open()
        await asyncio.wait([h.background_task("c1")])  # nobody called settle
        assert h.wake_errors == [("o", ["c1"], "ValueError: wake exploded")]
        assert len(said) == 2 and "wake exploded" in said[1] and "c1" in said[1]  # shown at once
        broken["on"] = False
        out = await h.user_says("Is it done?")
        assert "wake exploded" in out[0] and out[1:] == ["It joined."]  # and again with the next turn
        assert store.timeline("c1")["status"] == "joined" and len(join_records(store)) == 1  # the join held
        with pytest.raises(ValueError, match="wake exploded"):
            await h.settle()  # not swallowed
        assert await h.settle() == []  # raised once
        await h.machines.stop()

    asyncio.run(go())


def test_a_wake_reports_or_routes_according_to_its_own_batch(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Reporting."))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        store.add_turn("o", "assistant", "text", "Started.")
        (c1, c2), _ = await clone_in_turn(h, clone_use(entry(["website"], False), entry(["slides"], False)),
                                         retry_left=False)
        store.add_turn(c1, "assistant", "text", "Site built.")
        await h.do_join(c1, "done")
        await h.drain_and_wake("o")
        assert h.wakes[0]["report"] is True  # woken to report finished work: no clone in that run
        await h.do_join(c2, "out_of_scope")  # no user message in between
        await h.drain_and_wake("o")
        assert len(h.wakes) == 2 and h.wakes[1]["joined"] == [(c2, "out_of_scope")]
        assert h.wakes[1]["report"] is False and h.clone_blocked("o", 0) is None  # this one is routable
        await h.machines.stop()

    asyncio.run(go())


def test_a_background_failure_block_survives_an_unrelated_focus_trigger(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Routing that."))

    async def go():
        await h.start()
        req = store.add_turn("o", "user", "text", REQUEST)
        (c1, c2), msg = await clone_in_turn(h, clone_use(entry(["website"], True), entry(["slides"], False)),
                                           retry_left=False)
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": msg}, "clone")
        await h.do_join(c2, "failed")  # the background graft fails while the source is suspended
        store.add_turn(c1, "assistant", "text", "Site built.")
        trig = store.add_turn(c1, "user", "text", "Now email it to Bob.")
        await h.do_join(c1, "out_of_scope")  # an unrelated request reaches the source, first time
        assert req in h.reclone_block["o"]  # the failed request stays blocked ...
        assert trig not in h.reclone_block["o"] and h.clone_blocked("o", 0) is None  # ... the new one is not
        await h.machines.stop()

    asyncio.run(go())


def test_a_failed_background_request_is_not_recloned_until_a_new_message(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Reporting."))

    async def go():
        await h.start()
        req = store.add_turn("o", "user", "text", REQUEST)
        store.add_turn("o", "assistant", "text", "Started.")
        (c1, c2), _ = await clone_in_turn(h, clone_use(entry(["website"], False), entry(["slides"], False)),
                                         retry_left=False)
        await h.do_join(c1, "failed")
        await h.do_join(c2, "out_of_scope")
        await h.drain_and_wake("o")  # one wake for both; the out_of_scope join makes it a routing turn
        assert h.wakes[-1]["report"] is False
        assert h.reclone_block["o"] == {req} and "already failed" in h.clone_blocked("o", 0)
        await h.user_says("Please try the deck again.")
        assert h.clone_blocked("o", 0) is None  # a new message may clone again
        await h.machines.stop()

    asyncio.run(go())


# ------------------------------------------------------------------ scope: a graft does its own part
def test_skill_text_carries_the_own_part_rule():
    from graft.skills import BACKGROUND_INSTRUCTIONS, CATALOG, CLONE_INSTRUCTIONS, JOIN_INSTRUCTIONS
    desc = {s.name: " ".join(s.description.split()).lower() for s in CATALOG}
    for name, instr in (("join", JOIN_INSTRUCTIONS), ("background", BACKGROUND_INSTRUCTIONS)):
        flat = " ".join(instr.split()).lower()
        # the messages a graft starts with may ask for several things: it does only its own part
        assert "start with" in flat and "only the part" in flat and "other instances" in flat, name
        assert "do not attempt" in flat and "apologise" in flat and "mention" in flat, name
        assert "never a reason to join" in flat, name
        # the one-line description is what a graft sees before it loads the skill
        assert "only the part" in desc[name] and "other instances" in desc[name], name
    join = " ".join(JOIN_INSTRUCTIONS.split()).lower()
    assert "after you started" in join and "after you started" in desc["join"]  # whole-message rule: later messages
    background = " ".join(BACKGROUND_INSTRUCTIONS.split()).lower()
    assert "only if you could not do your own part" in background
    clone = " ".join(CLONE_INSTRUCTIONS.split()).lower()
    assert "only the part its skills cover" in clone and "one entry per part" in clone
    assert "still running" in clone and "do not clone again for the parts" in clone


def test_a_focus_graft_seeded_with_a_multi_part_message_does_its_part(tmp_path):
    """The seed asks for a website and slides. The website focus graft builds the site; only a
    later message for something else sends it back out of scope."""
    from graft.fake import bench_graft_policy
    multi = "We offer cleaning services. Please build our website and a slide deck for the open house."

    def policy(system, messages, tools):
        if "clone" not in {t["name"] for t in tools}:
            return bench_graft_policy(system, messages, tools)
        last = messages[-1]["content"][-1]
        if last["type"] == "text" and last["text"] == multi:
            return [clone_use(entry(["website"], True, [multi]), entry(["slides"], False, [multi]),
                              id="k1") | {"type": "tool_use"}]
        return text("Okay.")

    store, h, _ = make(tmp_path, policy)

    async def go():
        await h.start()
        out = await h.user_says(multi)
        assert h.machines.state("c1") == "active" and store.timeline("c1")["join_reason"] is None
        assert "services" in h.world.site["home"] and out and out[-1].startswith("Done: wrote page home")
        await h.settle()
        assert store.timeline("c2")["join_reason"] == "done" and h.world.decks
        await h.user_says("Now book an appointment on 2026-12-05 at 10:00.")  # arrives after it started
        assert store.timeline("c1")["join_reason"] == "out_of_scope"
        await h.machines.stop()

    asyncio.run(go())


def test_pivot_policy_website_graft_keeps_a_multi_part_seed(tmp_path):
    from graft.fake import pivot_policy
    store = Store(str(tmp_path / "v.db"))
    install(store)
    h = Harness(store, FakeProvider(pivot_policy), seeder=None)

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", "I need a website and later a slide show. Our services: cleaning.")
        (c1,), msg = await clone_in_turn(h, clone_use(entry(["website"], True, ["I need a website"])),
                                        retry_left=False)
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": msg}, "clone")
        await h.run(c1)  # the seed also mentions slides: that is another instance's part
        assert h.machines.state("c1") == "active" and "cleaning" in h.world.site["home"]  # did its part
        await h.user_says("Instead let's build a slide show.")  # a later message: out of scope
        assert store.timeline("c1")["join_reason"] == "out_of_scope"
        await h.machines.stop()

    asyncio.run(go())


# ------------------------------------------------------------------ notes name the grafts still running
def test_wake_note_and_join_record_name_the_grafts_still_running(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Reporting."))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        store.add_turn("o", "assistant", "text", "Started.")
        (c1, c2), _ = await clone_in_turn(h, clone_use(entry(["website"], False), entry(["slides"], False)),
                                         retry_left=False)
        store.add_turn(c1, "assistant", "text", "Site built.")
        await h.do_join(c1, "done")
        await h.drain_and_wake("o")
        notes = [r["content"] for r in store.view("o") if r["content"].startswith("(harness note")]
        record, wake = notes[-2:]
        assert record.startswith("(harness note: clone c1 joined, reason=done")
        assert "Still running: c2 (skills: slides)" in record
        assert wake.startswith("(harness note: background clone c1 (done) joined")
        assert "Still running: c2 (skills: slides)" in wake
        assert "only" in wake and "not describe still-running work as finished" in wake
        store.add_turn(c2, "assistant", "text", "Deck built.")
        await h.do_join(c2, "done")
        await h.drain_and_wake("o")
        notes = [r["content"] for r in store.view("o") if r["content"].startswith("(harness note")]
        record, wake = notes[-2:]
        assert "Still running" not in record and "No other clone of yours is still running" in record
        assert "Still running" not in wake and "No other clone of yours is still running" in wake
        await h.machines.stop()

    asyncio.run(go())
    from graft import seeder
    from graft.fake import woken
    for r in store.view("o"):  # every note is still recognised as a harness note
        if "still running" in r["content"].lower():
            assert not seeder.is_candidate(r) and r["content"].startswith("(harness note")
    wake_msgs = [{"role": "user", "content": [{"type": "text", "text": store.view("o")[-2]["content"]}]}]
    assert woken(wake_msgs)


def test_focus_join_record_names_live_background_grafts(tmp_path):
    seen = {}

    def policy(system, messages, tools):
        seen["notes"] = [b["text"] for m in messages if m["role"] == "user" for b in m["content"]
                         if b["type"] == "text" and b["text"].startswith("(harness note")]
        return text("Routing that.")

    store, h, _ = make(tmp_path, policy)

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        (c1, c2, c3), msg = await clone_in_turn(h, clone_use(
            entry(["website"], True), entry(["slides"], False), entry(["calendar"], False)), retry_left=False)
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": msg}, "clone")
        await h.do_join(c1, "out_of_scope")  # bounces at once, while both background grafts are live
        assert len(seen["notes"]) == 1  # the source ran for the trigger and saw the record
        assert seen["notes"][0].startswith("(harness note: clone c1 joined, reason=out_of_scope")
        assert "Still running: c2 (skills: slides), c3 (skills: calendar)" in seen["notes"][0]
        await h.machines.stop()

    asyncio.run(go())


# ------------------------------------------------------------------ round-1 review repairs
def test_out_of_scope_background_joins_cannot_loop_without_a_user_message(tmp_path):
    """The model picks a seed that ends on an older message and every background graft hands
    the request back: re-cloning must stop on its own."""
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "clone" not in names:
            return [{"type": "tool_use", "id": f"j{len(messages)}", "name": "join",
                     "input": {"reason": "out_of_scope"}}]
        if last["type"] == "tool_result":
            return text("Refused." if last.get("is_error") else "Started.")
        if last["text"] == "Earlier note.":
            return text("Noted.")
        if policy.calls > 12:  # the probe's stop: without a bound this never ends
            return text("Giving up.")
        policy.calls += 1
        return [clone_use(entry(["website"], False, ["Earlier note."]), id=f"k{policy.calls}") | {"type": "tool_use"}]

    policy.calls = 0
    store, h, _ = make(tmp_path, policy)

    async def go():
        await h.start()
        await h.user_says("Earlier note.")
        await h.user_says(REQUEST)
        await h.settle()
        await h.machines.stop()

    asyncio.run(go())
    clones = [t for t in h.machines.timelines if t != "o"]
    assert len(clones) <= 2, clones  # the first clone, and at most one re-route
    assert [w["report"] for w in h.wakes] == [False, True]  # one routing wake, then reports only
    assert h.wake_clone_calls == {"o": 1}


def test_an_exception_inside_clone_handling_ends_as_a_refused_tool_result(tmp_path):
    class Boom:
        name = "boom"

        async def score(self, purpose, cands, trigger):
            raise RuntimeError("seeder exploded")

    def policy(system, messages, tools):
        last = messages[-1]["content"][-1]
        if last["type"] == "tool_result":
            return text("Could not start that.")
        return [clone_use(entry(["website"], False), id="k1") | {"type": "tool_use"}]

    store = Store(str(tmp_path / "e.db"))
    install(store)
    h = Harness(store, FakeProvider(policy), seeder=Boom())

    async def go():
        await h.start()
        out = await h.user_says(REQUEST)
        assert out == ["Could not start that."]
        await h.machines.stop()

    asyncio.run(go())
    assert validate(store.view("o"), "o", require_user_end=False) == []  # no orphan tool_use
    result = [r for r in store.view("o") if r["kind"] == "tool_result"][0]
    assert '"is_error": true' in result["content"] and "seeder exploded" in result["content"]
    assert "no clone was created" in result["content"]
    statuses = [r[0] for r in store.con.execute("SELECT status FROM transactions")]
    assert statuses and set(statuses) == {"rolled_back"}  # nothing left open
    assert list(h.machines.timelines) == ["o"]


def test_a_failed_wake_at_the_end_of_a_user_turn_still_returns_the_turn(tmp_path):
    script = Script([entry(["website"], False)])
    script.root_gate_on_reply = True
    script.open("website")

    def policy(system, messages, tools):
        last = messages[-1]["content"][-1]
        if last["type"] == "text" and last["text"].startswith("(harness note: background clone"):
            raise ValueError("wake exploded")
        return script(system, messages, tools)

    store, h, said = make(tmp_path, policy)

    async def go():
        await h.start()
        turn = asyncio.create_task(h.user_says(REQUEST))
        await h.join_queued("c1")  # the graft finished while the source was mid-turn
        script.open("root")
        out = await turn  # the wake at the end of the turn fails: the turn's replies still come back
        assert out[0] == "Started." and len(out) == 2 and "wake exploded" in out[1] and "c1" in out[1]
        assert said == out and len(join_records(store)) == 1  # told at once; the join stays committed
        assert await h.user_says("Is it there?") != []  # the source is usable again
        await h.machines.stop()

    asyncio.run(go())


def test_settle_raises_every_failed_background_task(tmp_path):
    script = Script([entry(["website"], False), entry(["slides"], False)])

    def policy(system, messages, tools):
        last = messages[-1]["content"][-1]
        if last["type"] == "text" and last["text"].startswith("(harness note: background clone"):
            raise ValueError(f"wake exploded {len(h.wake_errors) + 1}")
        return script(system, messages, tools)

    store, h, _ = make(tmp_path, policy)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        script.open("website")
        await asyncio.wait([h.background_task("c1")])  # its wake has failed
        script.open("slides")
        await asyncio.wait([h.background_task("c2")])
        with pytest.raises(ExceptionGroup) as group:
            await h.settle()
        assert sorted(str(e) for e in group.value.exceptions) == ["wake exploded 1", "wake exploded 2"]
        lines = await h.settle()  # raised once; the two error lines the user was shown are returned
        assert len(lines) == 2 and all("reporting the result failed" in x for x in lines)
        await h.machines.stop()

    asyncio.run(go())


def test_wake_note_does_not_point_at_rows_the_size_backstop_dropped(tmp_path):
    store, h, _ = make(tmp_path, lambda *a: text("Reporting."), window_tokens=120)

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        store.add_turn("o", "assistant", "text", "Started.")
        (c1,), _ = await clone_in_turn(h, clone_use(entry(["website"], False)), retry_left=False)
        store.add_turn(c1, "assistant", "text", "Site built. " + "x" * 4000)
        await h.do_join(c1, "done")
        await h.drain_and_wake("o")
        await h.machines.stop()

    asyncio.run(go())
    assert h.join_fallbacks and h.join_fallbacks[0]["tier"] == 2  # nothing of c1 fits
    wake = [r["content"] for r in store.view("o") if r["content"].startswith("(harness note: background")][0]
    assert "in the rows above" not in wake
    assert "c1" in wake and "did not fit" in wake and "frozen timeline" in wake


def test_clone_entries_validates_model_written_fields_and_keeps_raw_indexes():
    from graft.skills import CATALOG, MAX_CLONE_ENTRIES
    es, v = clone_entries({"clone": ["junk", {"purpose": "p", "skills": "website", "focus": True},
                                     {"purpose": "p", "skills": ["website", "website"], "focus": False,
                                      "seed_quotes": [1], "seed_row_ids": ["7", True]}]})
    assert [e.index for e in es] == [1, 2]  # the index in the model's list, not in the filtered one
    assert v == ["clone[0]: must be an object",
                 "clone[1]: skills must be a list of strings",
                 "clone[2]: skills lists ['website'] more than once",
                 "clone[2]: seed_quotes must be a list of strings",
                 "clone[2]: seed_row_ids must be a list of integers"]
    assert es[0].skills == [] and es[1].seed_quotes == [] and es[1].seed_row_ids == []  # nothing unusable kept
    _, v = clone_entries({"clone": [entry(["website"], False)] * (MAX_CLONE_ENTRIES + 1)})
    assert v == [f"a clone call takes at most {MAX_CLONE_ENTRIES} entries (got {MAX_CLONE_ENTRIES + 1})"]
    assert clone_entries("not a dict")[1] and clone_entries({"clone": {"purpose": "p"}})[1]
    schema = next(t for s in CATALOG if s.name == "clone" for t in s.tools if t["name"] == "clone")["input_schema"]
    assert schema["properties"]["clone"]["maxItems"] == MAX_CLONE_ENTRIES


def test_clone_call_refuses_bad_entries_by_raw_index_and_caps_live_background_grafts(tmp_path):
    from graft.machine import MAX_LIVE_BACKGROUND
    store, h, _ = make(tmp_path, lambda *a: text("ok"))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        children, msg = await clone_in_turn(h, clone_use("junk", entry(["nope"], False)), retry_left=True)
        assert children is None and "clone[0]: must be an object" in msg and "clone[1]: unknown skill 'nope'" in msg
        children, msg = await clone_in_turn(h, clone_use(*[entry(["website"], False)] * MAX_LIVE_BACKGROUND))
        assert len(children) == MAX_LIVE_BACKGROUND
        children, msg = await clone_in_turn(h, clone_use(entry(["slides"], False)))
        assert children is None and f"at most {MAX_LIVE_BACKGROUND} background clones may be live" in msg
        assert "c1, c2, c3, c4" in msg and len(h.machines.timelines) == 1 + MAX_LIVE_BACKGROUND
        (focus,), msg = await clone_in_turn(h, clone_use(entry(["slides"], True)))  # a focus clone is not capped
        assert focus == "c5"
        statuses = {r[0] for r in store.con.execute("SELECT status FROM transactions")}
        assert statuses == {"rolled_back", "committed"}
        await h.machines.stop()

    asyncio.run(go())


def test_a_failed_commit_of_the_clone_batch_is_a_refusal(tmp_path, monkeypatch):
    import sqlite3
    store, h, _ = make(tmp_path, lambda *a: text("ok"))

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)

        def broken(source, grafts):
            raise sqlite3.IntegrityError("UNIQUE constraint failed: timelines.timeline_id")

        monkeypatch.setattr(store, "commit_clone", broken)
        children, msg = await clone_in_turn(h, clone_use(entry(["website"], False), entry(["slides"], True)),
                                            retry_left=True)
        assert children is None and "no clone was created" in msg and "IntegrityError" in msg
        assert list(h.machines.timelines) == ["o"] and h.machines.phase("o") == "idle"
        assert [r[0] for r in store.con.execute("SELECT status FROM transactions")] == ["rolled_back"] * 2
        await h.machines.stop()

    asyncio.run(go())


async def _two_queued(store, h):
    """o at rest with two finished background grafts queued on it (no task is running them)."""
    await h.start()
    store.add_turn("o", "user", "text", REQUEST)
    store.add_turn("o", "assistant", "text", "Started.")
    (c1, c2), _ = await clone_in_turn(h, clone_use(entry(["website"], False), entry(["slides"], False)))
    store.add_turn(c1, "assistant", "text", "Site built.")
    store.add_turn(c2, "assistant", "text", "Deck built.")
    await h.do_join(c1, "done")
    await h.do_join(c2, "done")
    return c1, c2


def test_a_queued_commit_that_raises_is_reported_kept_and_retried(tmp_path, monkeypatch):
    store, h, said = make(tmp_path, lambda *a: text("Reporting."))

    async def go():
        c1, c2 = await _two_queued(store, h)
        real = h.default_join

        def flaky(src, child, reason, trigger):
            if child == c2:
                raise OSError("disk full")
            return real(src, child, reason, trigger=trigger)

        monkeypatch.setattr(h, "default_join", flaky)
        await h.drain_and_wake("o")  # c1 commits and is reported; c2's commit raises
        assert h.commit_errors == [("o", [c2], "OSError: disk full")]
        line = [x for x in said if x.startswith("(harness error")][0]
        assert "committing the join of background clone c2 failed" in line and "disk full" in line
        assert (h.machines.state("o"), h.machines.phase("o")) == ("active", "idle")  # usable again
        assert h.machines.state(c1) == "joined" and h.machines.state(c2) == "queued"  # not lost
        assert h.machines.queued("o") == [c2] and store.timeline(c2)["status"] == "queued"
        assert [w["joined"] for w in h.wakes] == [[(c1, "done")]]
        monkeypatch.setattr(h, "default_join", real)
        out = await h.user_says("And the deck?")  # the next drain point retries it
        assert out[0] == line and out[-1] == "Reporting."
        assert h.machines.state(c2) == "joined" and h.machines.queued("o") == []
        assert "Deck built." in [r["content"] for r in store.view("o")]
        await h.machines.stop()

    asyncio.run(go())
    assert validate(store.view("o"), "o", require_user_end=False) == []


def test_a_rolled_back_queued_join_drives_the_rollback_events(tmp_path, monkeypatch):
    store, h, _ = make(tmp_path, lambda *a: text("Reporting."))
    events = []

    async def go():
        c1, c2 = await _two_queued(store, h)
        real = h.default_join
        monkeypatch.setattr(h, "default_join", lambda src, child, reason, trigger:
                            False if child == c1 else real(src, child, reason, trigger=trigger))
        commit_queued, finish_join = h.machines.commit_queued, h.machines.finish_join

        async def spy_commit(child, committed):
            events.append(("queued", child, committed))
            await commit_queued(child, committed=committed)

        async def spy_finish(source, committed):
            events.append(("source", source, committed))
            await finish_join(source, committed=committed)

        monkeypatch.setattr(h.machines, "commit_queued", spy_commit)
        monkeypatch.setattr(h.machines, "finish_join", spy_finish)
        await h.drain_and_wake("o")
        assert events == [("queued", c1, False), ("queued", c2, True), ("source", "o", False)]
        assert h.machines.state(c1) == h.machines.state(c2) == "joined" and h.machines.queued("o") == []
        assert h.commit_errors == [] and len(h.wakes) == 1
        await h.machines.stop()

    asyncio.run(go())


def _wake_raises(system, messages, tools):
    last = messages[-1]["content"][-1]
    if last["type"] == "text" and last["text"].startswith("(harness note: background clone"):
        raise ValueError("wake exploded")
    return text("ok")


def test_a_failed_wake_from_settle_is_shown_and_raised(tmp_path):
    store, h, said = make(tmp_path, _wake_raises)

    async def go():
        c1, c2 = await _two_queued(store, h)
        with pytest.raises(ValueError, match="wake exploded"):
            await h.settle()
        assert h.wake_errors == [("o", [c1, c2], "ValueError: wake exploded")]
        assert len(said) == 1 and "reporting the result failed" in said[0]
        assert len(join_records(store)) == 2 and h.machines.phase("o") == "idle"  # committed; usable
        assert await h.settle() == said  # raised once; the line is returned
        await h.machines.stop()

    asyncio.run(go())


def test_a_failed_wake_after_a_focus_done_join_is_contained(tmp_path):
    store, h, said = make(tmp_path, _wake_raises)

    async def go():
        await h.start()
        store.add_turn("o", "user", "text", REQUEST)
        (c1, c2), msg = await clone_in_turn(h, clone_use(entry(["website"], True), entry(["slides"], False)))
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": msg}, "clone")
        store.add_turn(c2, "assistant", "text", "Deck built.")
        await h.do_join(c2, "done")
        store.add_turn(c1, "assistant", "text", "Site built.")
        await h.do_join(c1, "done")  # the wake that follows fails; the joins hold and nothing propagates
        assert h.wake_errors == [("o", [c2], "ValueError: wake exploded")] and len(said) == 1
        assert len(join_records(store)) == 2 and h.machines.phase("o") == "idle"
        await h.machines.stop()

    asyncio.run(go())


def test_close_cancels_running_background_grafts_and_joins_them_failed(tmp_path):
    script = Script([entry(["website"], False), entry(["slides"], False)])  # gates closed: both hang
    store, h, _ = make(tmp_path, script)

    async def go():
        await h.start()
        assert await h.user_says(REQUEST) == ["Started."]
        tasks = [h.background_task("c1"), h.background_task("c2")]
        await h.close()
        assert all(t.cancelled() for t in tasks)
        assert h.wakes == []  # nothing is run on shutdown
        assert h.machines.state("o") == "stopped"

    asyncio.run(go())
    script.open()  # let the abandoned model-call threads end
    assert [store.timeline(c)["join_reason"] for c in ("c1", "c2")] == ["failed", "failed"]
    assert [store.timeline(c)["status"] for c in ("c1", "c2")] == ["joined", "joined"]
    records = [x for x in join_records(store) if " joined, " in x]  # without the two error records
    assert len(records) == 2 and all("reason=failed" in x for x in records)
    assert validate(store.view("o"), "o", require_user_end=False) == []


def test_usage_state_is_carried_into_a_worker_thread():
    provider = FakeProvider(lambda *a: text("ok"))
    mine = provider.thread_usage_reset()
    with provider.usage_tag("outer"):
        state = provider.thread_state()
    assert state == {"tag": "outer", "usage": mine}
    seen = {}

    def work():
        with provider.adopt(state):
            with provider.usage_tag("clone:c1"):
                provider.complete("s", [], [])
            seen["last"] = provider.last_call()
        seen["after"] = (getattr(provider._local, "tag", None), getattr(provider._local, "usage", None))

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert mine["calls"] == 1 and provider.by_tag["clone:c1"]["calls"] == 1  # counted for the caller
    assert seen["last"]["tag"] == "clone:c1" and seen["after"] == (None, None)  # restored in the worker
    assert provider.last_call() is None  # the caller's own last call is untouched


def test_multi_entry_clone_call_with_an_async_seeder_and_one_fallback(tmp_path):
    from graft import seeder as sd

    class Seeder:
        name = "fake"

        def __init__(self):
            self.live = self.peak = 0

        async def score(self, purpose, cands, trigger):
            self.live += 1
            self.peak = max(self.peak, self.live)
            await asyncio.sleep(0)  # entries are seeded concurrently
            self.live -= 1
            if "deck" in purpose:
                return sd.SeedDecision(error="TimeoutError: seeder timeout", model="fake-1")
            return sd.SeedDecision(scores={c["row_id"]: 0.9 for c in cands}, model="fake-1", requests=1)

    store = Store(str(tmp_path / "m.db"))
    install(store)
    seeder = Seeder()
    h = Harness(store, FakeProvider(lambda *a: text("ok")), seeder=seeder)

    async def go():
        await h.start()
        fact = store.add_turn("o", "user", "text", "We are Okafor Dental.")
        store.add_turn("o", "assistant", "text", "Noted.")
        req = store.add_turn("o", "user", "text", REQUEST)
        (c1, c2), msg = await clone_in_turn(h, clone_use(
            entry(["website"], False, quotes=(), purpose="the site"),
            entry(["slides"], False, quotes=(REQUEST,), purpose="the deck")))
        assert seeder.peak == 2
        assert [r["row_id"] for r in store.view(c1)][0] == fact and store.view(c1)[-1]["row_id"] == req
        assert [r["row_id"] for r in store.view(c2)] == [req]  # the model's own seed: the fallback
        assert sorted(str(x["fallback"]) for x in h.seeder_log) == ["None", "model_seed"]
        audits = [__import__("json").loads(r[0])["seeder"] for r in store.con.execute(
            "SELECT scratch FROM transactions WHERE kind='clone' AND status='committed' ORDER BY tx_id")]
        assert [a["fallback"] for a in audits] == [None, "model_seed"] and "seeder timeout" in audits[1]["error"]
        await h.machines.stop()

    asyncio.run(go())


# ------------------------------------------------------------------ graft chat
_CHAT = '''
import sys, threading
import graft.cli as cli
from graft.providers import FakeProvider

def policy(system, messages, tools):
    names = {t["name"] for t in tools}
    last = messages[-1]["content"][-1]
    if "clone" not in names:
        threading.Event().wait()  # a background clone whose model call never returns
    if last["type"] == "tool_result":
        return [{"type": "text", "text": "Started."}]
    return [{"type": "tool_use", "id": "k1", "name": "clone", "input": {"clone": [
        {"purpose": "p", "skills": ["website"], "focus": False, "seed_quotes": [last["text"]]}]}}]

cli.make_provider = lambda: FakeProvider(policy)
sys.argv = ["graft", "chat", "--db", sys.argv[1]]
cli.main()
'''


def _chat(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return subprocess.Popen([sys.executable, "-u", "-c", _CHAT, str(tmp_path / "chat.db")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_until(proc, needle):
    seen = ""
    while needle not in seen:
        ch = proc.stdout.read(1)
        assert ch, f"chat ended before {needle!r}: {seen!r} {proc.stderr.read()!r}"
        seen += ch
    return seen


def test_chat_ctrl_c_exits_promptly_without_a_traceback(tmp_path, monkeypatch):
    """Ctrl-C at the prompt, while a background clone's model call is hanging and the input
    thread is blocked reading: the process must not be pinned by either."""
    import signal
    proc = _chat(tmp_path, monkeypatch)
    try:
        _read_until(proc, "you> ")
        proc.stdin.write("Build me a website.\n")
        proc.stdin.flush()
        _read_until(proc, "assistant> Started.")
        _read_until(proc, "you> ")
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=10)
    finally:
        proc.kill()
    assert proc.returncode == 130 and "Traceback" not in err and "Task exception" not in err, err
    store = Store(str(tmp_path / "chat.db"))
    assert store.timeline("c1")["join_reason"] == "failed" and store.timeline("c1")["status"] == "joined"  # closed


def test_chat_end_of_input_exits_cleanly(tmp_path, monkeypatch):
    proc = _chat(tmp_path, monkeypatch)
    try:
        _read_until(proc, "you> ")
        out, err = proc.communicate(timeout=10)  # closes stdin
    finally:
        proc.kill()
    assert proc.returncode == 0 and "Traceback" not in err, err
