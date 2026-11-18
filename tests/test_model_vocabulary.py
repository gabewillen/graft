"""What the model reads calls a graft a branch of itself and a join a merge (spec: What the model is
told: branch and merge). No model-facing text names a clone, a join, a graft, the original, an
instance, a subagent, a worker, a helper or a delegate, or says the assistant will remember something for a graft.

Model-facing text is what the harness writes for a model to read: the system prompt of every
timeline, every tool's name, description and schema, harness notes, and tool results (of the
built-in tools, and the harness's "not executed" results). It is collected two ways: what the
scripted model was actually sent in a set of runs chosen to reach most of the harness's notes,
results and refusals (each run asserts the texts it is meant to reach were reached), and every
harness row and tool result left in the store after them. Paths these runs do not reach (a few
internal-error refusals, shutdown races) are covered only by review of their text. The
user's own messages and the model's own replies are not the harness's words and are left out;
the scripted user and model avoid the banned words, so the capability tools' results (which echo
them) can be scanned too."""

import asyncio
import json
import re
import threading

import pytest

from graft.db import Store
from graft.harness import (CANCELLED_BY_THE_TOOL, CANCELLED_WHILE_RUNNING, FAILED_AFTER_INVOKED,
                           NOT_RUN_BESIDE_CLONE, NOT_RUN_INTERRUPTED, NOT_RUN_JOINING, WITHDRAWN_SEED,
                           Harness)
from graft.providers import FakeProvider
from graft import seeder as seeder_mod
from graft.skills import BACKGROUND_INSTRUCTIONS, BUILTIN, CATALOG, CLONE_INSTRUCTIONS, JOIN_INSTRUCTIONS, \
    catalog, install

# Stems, so every inflection is caught (clone, clones, cloned, cloning; join, joins, joined, ...).
BANNED = re.compile(r"\bclon|\bjoin|\binstances?\b|\bsub-?agents?\b|\bworkers?\b|\bhelpers?\b|\bdelegat"
                    r"|\bgrafts?\b|\boriginal\b|the assistant will", re.IGNORECASE)
REQUEST = "Build a website and a slide deck for the dental practice."
GATE_TIMEOUT = 10


def banned(text: str) -> list[str]:
    return sorted({m.group().lower() for m in BANNED.finditer(text)})


def text(t):
    return [{"type": "text", "text": t}]


def use(tool, use_id, **args):
    return {"type": "tool_use", "id": use_id, "name": tool, "input": args}


def entry(skills, focus, quotes=(REQUEST,), **extra):
    return {"purpose": "p", "skills": skills, "focus": focus, "seed_quotes": list(quotes), **extra}


class Recorder:
    """A scripted model that keeps what it was sent: the system prompt, the tools, and every
    block the harness wrote (harness notes and tool results), never the user's own text."""

    def __init__(self, policy):
        self.policy = policy
        self.seen: list[tuple[str, str]] = []  # (where, text)
        self.lock = threading.Lock()

    def __call__(self, system, messages, tools):
        with self.lock:
            self.seen.append(("system", system))
            self.seen.append(("tools", json.dumps(tools)))
            for m in messages:
                for b in m["content"]:
                    if b["type"] == "text" and b.get("harness"):
                        self.seen.append(("note", b["text"]))
                    elif b["type"] == "tool_result":
                        self.seen.append(("result", str(b["content"])))
        return self.policy(system=system, messages=messages, tools=tools)


def store_texts(store: Store) -> list[tuple[str, str]]:
    """Every harness row and every tool result in the store, whoever owns it."""
    out = []
    for r in store.con.execute("SELECT kind, harness, content FROM turns"):
        kind, harness, content = r
        if kind == "text" and harness:
            out.append(("stored note", content))
        elif kind == "tool_result":
            out.append(("stored result", str(json.loads(content)["content"])))
    return out


def last_block(messages):
    return messages[-1]["content"][-1]


def make(tmp_path, policy, name, cap=None, **kw):
    store = Store(str(tmp_path / f"{name}.db"))
    if cap is None:
        install(store)
    else:
        install(store=store, max_live_background=cap)
        kw["max_live_background"] = cap
    rec = Recorder(policy)
    h = Harness(store=store, provider=FakeProvider(policy=rec), seeder=None, **kw)
    return store, h, rec


def check(found: list[tuple[str, str]]):
    bad = [(where, banned(t), t[:200]) for where, t in found if banned(t)]
    assert not bad, "model-facing text uses a banned word:\n" + "\n".join(map(str, bad))


# ------------------------------------------------------------------ the text itself
def test_the_built_in_skills_and_tools_say_branch_and_merge():
    names = {s.name for s in CATALOG}
    assert {"branch", "merge", "background"} <= names and not {"clone", "join"} & names
    assert set(BUILTIN) == {"branch", "merge", "background"}
    tools = {t["name"] for s in CATALOG for t in s.tools}
    assert {"branch", "merge", "cancel", "load_skill"} <= tools and not {"clone", "join"} & tools
    branch_tool = next(t for s in CATALOG for t in s.tools if t["name"] == "branch")
    assert set(branch_tool["input_schema"]["properties"]) == {"branch"}
    merge_tool = next(t for s in CATALOG for t in s.tools if t["name"] == "merge")
    assert merge_tool["input_schema"]["properties"]["reason"]["enum"] == ["done", "out_of_scope", "failed"]
    # the first-person framing: a branch is you, and what you did there is your memory
    focus, background, original = (" ".join(t.split()) for t in (JOIN_INSTRUCTIONS, BACKGROUND_INSTRUCTIONS,
                                                                   CLONE_INSTRUCTIONS))
    assert "in a branch of yourself" in focus and "in a background branch of yourself" in background
    assert "other branches of yourself" in focus and "other branches of yourself" in background
    assert "what you will remember of this work when the branch merges" in background
    assert "branching yourself: a branch is you" in original and "in the first person" in original


@pytest.mark.parametrize("cap", [1, 4, 7])
def test_no_built_in_skill_text_or_tool_uses_a_banned_word(cap):
    found = []
    for s in catalog(max_live_background=cap):
        if s.name not in BUILTIN:
            continue
        found += [("name", s.name), ("description", s.description), ("instructions", s.instructions)]
        found += [("tool", json.dumps(t)) for t in s.tools]
    check(found)


def test_the_harness_constant_results_and_the_seeder_question_use_no_banned_word():
    check([("const", str(x)) for x in (NOT_RUN_BESIDE_CLONE, NOT_RUN_INTERRUPTED, NOT_RUN_JOINING,
                                       CANCELLED_WHILE_RUNNING, CANCELLED_BY_THE_TOOL, FAILED_AFTER_INVOKED,
                                       WITHDRAWN_SEED)]
          + [("seeder", seeder_mod.NOUL_TRUE), ("seeder", seeder_mod.NOUL_FALSE),
             ("seeder", json.dumps(seeder_mod.reduced_question(
                 candidates=[{"row_id": 1, "role": "user", "text": "t"}], i=0)))])


def test_the_banned_words_are_caught():
    """The check itself: each word the framing rules out is caught, in any inflection."""
    for t in ("a clone", "Cloned ->", "joined", "join(reason", "another instance", "the subagent",
              "a sub-agent", "workers", "a helper", "delegates", "the assistant will remember", "a graft",
              "background grafts", "the original"):
        assert banned(t), t
    for t in ("your branch c2 merged", "cancel", "out_of_scope", "the same assistant", "rejoice", "grafted",
              "originally"):
        assert not banned(t), t


# ------------------------------------------------------------------ representative runs
def test_background_parts_cancel_and_refusals_say_branch_and_merge(tmp_path):
    """Two background parts, a refused call (bad entries and seeds, a tool beside it), a graft
    that tries to cancel and to merge silently, a cancel by the original (bad ids among them), the
    wake that reports it, and a re-branch of the cancelled part, which is refused."""
    slides_gate = threading.Event()
    website_gate = threading.Event()  # c1 is still running when the original cancels: it is not named
    waits = []  # each gate wait's outcome, asserted in the test body (a failure on a model-call thread is a failed run)

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        if "branch" in names:
            if last["type"] == "text" and not last.get("harness") and last["text"] == REQUEST:
                return [use(tool="branch", use_id="b1", branch=[
                            entry(skills=["merge"], focus=False), entry(skills=["branch"], focus=False), {"purpose": "p", "skills": ["website"]},
                            entry(skills=["slides"], focus=False, quotes=("no such words anywhere",), seed_row_ids=[999])]),
                        use(tool="load_skill", use_id="l1", name="website")]
            if last["type"] == "tool_result" and "b1" == last["tool_use_id"]:
                return [use(tool="branch", use_id="b2", branch=[entry(skills=["website"], focus=False), entry(skills=["slides"], focus=False)])]
            if last["type"] == "text" and not last.get("harness") and last["text"] == "Stop the slides.":
                return [use(tool="cancel", use_id="x1", cancel=["c2", "c9", "o", 5])]
            if last["type"] == "tool_result" and last["tool_use_id"] == "x1":
                website_gate.set()
                return text("Okay.")
            if last["type"] == "text" and last.get("harness") and "c2 (cancelled)" in last["text"]:
                return [use(tool="branch", use_id="b3", branch=[entry(skills=["slides"], focus=False, quotes=("Stop the slides.",))])]
            return text("Okay.")
        if "site_write" in names:
            if not any(b["type"] == "tool_result" for m in messages for b in m["content"]):
                return [use(tool="cancel", use_id="g1", cancel=["c2"]), use(tool="merge", use_id="g2", reason="done")]
            waits.append(("website", website_gate.wait(GATE_TIMEOUT)))
            return [{"type": "text", "text": "Website built."}, use(tool="site_write", use_id="g3", page="home", content="hi"),
                    use(tool="merge", use_id="g4", reason="done")]
        waits.append(("slides", slides_gate.wait(GATE_TIMEOUT)))
        return text("Slides built.")

    store, h, rec = make(tmp_path=tmp_path, policy=policy, name="a")

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.user_says("Stop the slides.")
        await h.settle()
        slides_gate.set()
        await h.close()

    asyncio.run(go())
    assert ("website", True) in waits and all(released for _, released in waits), waits
    results = [t for w, t in rec.seen if w == "result"]
    assert any(t.startswith("branch refused") and "'merge' is linked by the harness" in t
               and "'branch' cannot be held by a branch" in t for t in results), results
    assert any(t.startswith("branched -> ") for t in results), results
    cancel = sorted({t for t in results if t.startswith("cancelled -> ")})  # seen once per later model call
    assert len(cancel) == 1 and cancel[0].startswith("cancelled -> c2 (stopped;"), results
    assert "c9: not cancelled: no such branch" in cancel[0] and "o: not cancelled:" in cancel[0], cancel
    assert "cancel[3]: not cancelled: a int is not a branch id" in cancel[0], cancel
    assert any("already failed, was cancelled" in t for t in results), results  # the re-branch block
    assert any("merge(reason='done') refused" in t for t in results), results
    notes = [t for w, t in rec.seen if w == "note"]
    assert any(t.startswith("(harness note: your background branch") and "c2 (cancelled)" in t
               and "You cancelled c2" in t for t in notes), notes
    check(rec.seen + store_texts(store))


def test_focus_hand_back_and_failure_say_branch_and_merge(tmp_path):
    """A focus branch that asks, then hands a message back; a focus branch that fails."""
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        user = last["type"] == "text" and not last.get("harness")
        if "branch" in names:
            if user and last["text"] == "Build me a website.":
                return [use(tool="branch", use_id="f1", branch=[entry(skills=["website"], focus=True, quotes=("Build me a website.",))])]
            if user and last["text"] == "Make slides.":
                return [use(tool="branch", use_id="f2", branch=[entry(skills=["slides"], focus=True, quotes=("Make slides.",))])]
            return text("Okay.")
        if "site_write" in names:
            if user and last["text"] == "Build me a website.":
                return text("Which colors?")
            return [use(tool="merge", use_id="m1", reason="out_of_scope")]
        return [use(tool="merge", use_id="m2", reason="failed")]

    store, h, rec = make(tmp_path=tmp_path, policy=policy, name="b")

    async def go():
        await h.start()
        await h.user_says("Build me a website.")
        await h.user_says("Blue. Also send an email to Bob.")
        await h.user_says("Make slides.")
        await h.settle()
        await h.close()

    asyncio.run(go())
    notes = [t for w, t in rec.seen if w == "note"]
    assert any("merged, reason=out_of_scope" in t for t in notes), notes
    assert any("branch c2 failed" in t for t in notes), notes
    check(rec.seen + store_texts(store))


def test_caps_and_spent_calls_say_branch(tmp_path):
    """The live cap (a harness with another cap), the entries limit, and a run with no calls left."""
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        if "branch" in names:
            n = sum(1 for m in messages for b in m["content"] if b["type"] == "tool_result")
            if n == 0:
                return [use(tool="branch", use_id="k1", branch=[entry(skills=["website"], focus=False), entry(skills=["slides"], focus=False)])]
            if n == 1:
                return [use(tool="branch", use_id="k2", branch=[entry(skills=["website"], focus=False)] * 5)]
            if n == 2:
                return [use(tool="branch", use_id="k3", branch=[entry(skills=["website"], focus=False)])]
            return text("Okay.")
        return text("Done here.")

    store, h, rec = make(tmp_path=tmp_path, policy=policy, name="c", cap=1)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.settle()
        await h.close()

    asyncio.run(go())
    results = [t for w, t in rec.seen if w == "result"]
    assert any("at most 1 background branches may be live" in t for t in results), results
    assert any("a branch call takes at most 4 entries" in t for t in results), results
    assert any("no branch calls left" in t for t in results), results
    check(rec.seen + store_texts(store))


def test_the_per_message_cap_says_branch(tmp_path):
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        if "branch" in names:
            if last["type"] == "tool_result" and ("not executed" in str(last["content"])
                                                  or "branch refused" in str(last["content"])):
                return text("Okay.")
            return [use(tool="branch", use_id=f"p{len(messages)}", branch=[entry(skills=["website"], focus=False)])]
        return text("Site finished.")

    store, h, rec = make(tmp_path=tmp_path, policy=policy, name="d")

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await h.settle()
        await h.close()

    asyncio.run(go())
    assert any("are used up" in t for w, t in rec.seen if w == "result")
    check(rec.seen + store_texts(store))


def test_a_merge_too_large_for_the_context_says_branch(tmp_path):
    """The size backstop's records and the wake note that says a result did not fit."""
    from _util import clone_in_turn
    store, h, rec = make(tmp_path=tmp_path, policy=lambda **_: text(t="Reporting."), name="e", window_tokens=120)

    async def go():
        await h.start()
        store.add_turn(tl="o", role="user", kind="text", content=REQUEST)
        store.add_turn(tl="o", role="assistant", kind="text", content="Started.")
        (c1,), _ = await clone_in_turn(h=h, use={"id": "t", "name": "branch",
                                                 "input": {"branch": [entry(skills=["website"], focus=False)]}})
        store.add_turn(tl=c1, role="assistant", kind="text", content="Site built. " + "x" * 4000)
        await h.do_join(child=c1, reason="done")
        await h.drain_and_wake("o")
        await h.machines.stop()

    asyncio.run(go())
    assert h.join_fallbacks and h.join_fallbacks[0]["tier"] == 2
    notes = [t for w, t in store_texts(store) if w == "stored note"]
    assert any("did not fit" in t for t in notes), notes
    check(rec.seen + store_texts(store))


def test_the_old_tool_names_run_nothing(tmp_path):
    """A model that still calls `clone` or `join` is answered that the tool is not there: no graft
    is made or ended, and no capability tool runs."""
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        if "branch" in names:
            if last["type"] == "tool_result" and last["tool_use_id"] == "old":
                return [use(tool="branch", use_id="new", branch=[entry(skills=["website"], focus=True,
                                                                     quotes=("Build me a website.",))])]
            if last["type"] == "tool_result":
                return text("Okay.")
            return [use(tool="clone", use_id="old", clone=[entry(skills=["website"], focus=True,
                                                               quotes=("Build me a website.",))])]
        if last["type"] == "tool_result" and last["tool_use_id"] == "j":
            return text("Which colors?")
        return [use(tool="join", use_id="j", reason="done")]

    store, h, rec = make(tmp_path=tmp_path, policy=policy, name="f")

    async def go():
        await h.start()
        await h.user_says("Build me a website.")
        await h.close()

    asyncio.run(go())
    results = [t for w, t in store_texts(store) if w == "stored result"]
    assert results.count("tool 'clone' is not available to you") == 1, results
    assert results.count("tool 'join' is not available to you") == 1, results
    made = [r[0] for r in store.con.execute("SELECT timeline_id FROM timelines WHERE source_id IS NOT NULL")]
    assert made == ["c1"]  # only the `branch` call made a branch
    assert store.timeline("c1")["join_reason"] is None  # `join` did not end it
    assert h.world.log == []
    echoes = {"tool 'clone' is not available to you", "tool 'join' is not available to you"}  # the model's own words
    check([(w, t) for w, t in rec.seen + store_texts(store) if t not in echoes])


def test_a_merge_whose_view_would_be_invalid_says_branch(tmp_path, monkeypatch):
    """The record a merge falls back to when its view would be invalid (it cannot happen for
    well-formed rows, so validation is forced to fail for that one check)."""
    import graft.harness as harness_mod
    from _util import clone_in_turn
    real = harness_mod.validate
    forced = []

    def failing(rows, viewer, trigger=None, require_user_end=True):
        if viewer == "o" and require_user_end is False and trigger is None \
                and any(r.get("owner_timeline") == "c1" for r in rows):
            forced.append(viewer)
            return ["forced"]
        return real(rows=rows, viewer=viewer, trigger=trigger, require_user_end=require_user_end)

    store, h, rec = make(tmp_path=tmp_path, policy=lambda **_: text(t="Reporting."), name="g")

    async def go():
        await h.start()
        store.add_turn(tl="o", role="user", kind="text", content=REQUEST)
        store.add_turn(tl="o", role="assistant", kind="text", content="Started.")
        (c1,), _ = await clone_in_turn(h=h, use={"id": "t", "name": "branch",
                                                 "input": {"branch": [entry(skills=["website"], focus=False)]}})
        store.add_turn(tl=c1, role="assistant", kind="text", content="Site built.")
        monkeypatch.setattr(harness_mod, "validate", failing)
        await h.do_join(child=c1, reason="done")
        monkeypatch.setattr(harness_mod, "validate", real)
        await h.machines.stop()

    asyncio.run(go())
    notes = [t for w, t in store_texts(store) if w == "stored note"]
    assert forced, "the merge's validation was never forced to fail"
    assert any("its rows were skipped (invalid view)" in t for t in notes), notes
    check(rec.seen + store_texts(store))
