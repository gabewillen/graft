"""What the model reads calls a graft a branch of itself and a join a merge (spec: What the model is
told: branch, merge and stop). No model-facing text names a clone, a join, a graft, the original, an
instance, a subagent, a worker, a helper or a delegate, says the assistant will remember something
for a graft, or says background or cancel (the model's words are detached and stop).

Model-facing text is what the harness writes for a model to read: the system prompt of every
timeline, every tool's name, description and schema, harness notes, and tool results (of the
built-in tools, and the harness's "not executed" results). It is collected two ways: what the
scripted model was actually sent in a set of runs chosen to reach most of the harness's notes,
results and refusals (each run asserts the texts it is meant to reach were reached), and every
harness row and tool result left in the store after them. Paths these runs do not reach (a few
internal-error refusals, shutdown races) are covered only by review of their text. The
user's own messages and the model's own replies are not the harness's words and are left out;
the scripted user and model avoid the banned words, so the capability tools' results (which echo
them) can be scanned too. A static check covers the paths no run reaches: the string constants of
every function that writes model-facing text (PRODUCERS) are scanned, and each producer is shown
to be scanned by a mutation of its own text that the check catches."""

import ast
import asyncio
import inspect
import json
import re
import textwrap
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
                    r"|\bgrafts?\b|\boriginal\b|the assistant will|\bbackground|\bcancel", re.IGNORECASE)
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
    assert {"branch", "merge", "detached"} <= names and not {"clone", "join"} & names
    assert set(BUILTIN) == {"branch", "merge", "detached"}
    tools = {t["name"] for s in CATALOG for t in s.tools}
    assert {"branch", "merge", "stop", "load_skill"} <= tools and not {"clone", "join"} & tools
    branch_tool = next(t for s in CATALOG for t in s.tools if t["name"] == "branch")
    assert set(branch_tool["input_schema"]["properties"]) == {"branch"}
    merge_tool = next(t for s in CATALOG for t in s.tools if t["name"] == "merge")
    assert merge_tool["input_schema"]["properties"]["reason"]["enum"] == ["done", "out_of_scope", "failed"]
    # the first-person framing: a branch is you, and what you did there is your memory
    focus, background, original = (" ".join(t.split()) for t in (JOIN_INSTRUCTIONS, BACKGROUND_INSTRUCTIONS,
                                                                   CLONE_INSTRUCTIONS))
    assert "in a branch of yourself" in focus and "in a detached branch of yourself" in background
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
              "background grafts", "the original", "cancel", "Cancelled", "a background branch", "cancelling"):
        assert banned(t), t
    for t in ("your branch c2 merged", "stop", "a detached branch", "the current branch", "out_of_scope", "the same assistant", "rejoice", "grafted",
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
                return [use(tool="stop", use_id="x1", ids=["c2", "c9", "o", 5])]
            if last["type"] == "tool_result" and last["tool_use_id"] == "x1":
                website_gate.set()
                return text("Okay.")
            if last["type"] == "text" and last.get("harness") and "c2 (stopped)" in last["text"]:
                return [use(tool="branch", use_id="b3", branch=[entry(skills=["slides"], focus=False, quotes=("Stop the slides.",))])]
            return text("Okay.")
        if "site_write" in names:
            if not any(b["type"] == "tool_result" for m in messages for b in m["content"]):
                return [use(tool="stop", use_id="g1", ids=["c2"]), use(tool="merge", use_id="g2", reason="done")]
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
    cancel = sorted({t for t in results if t.startswith("stopped -> ")})  # seen once per later model call
    assert len(cancel) == 1 and cancel[0].startswith("stopped -> c2 (what you did in it"), results
    assert "c9: not stopped: no such branch" in cancel[0] and "o: not stopped:" in cancel[0], cancel
    assert "ids[3]: not stopped: a int is not a branch id" in cancel[0], cancel
    assert any("already failed, was stopped" in t for t in results), results  # the re-branch block
    assert any("merge(reason='done') refused" in t for t in results), results
    notes = [t for w, t in rec.seen if w == "note"]
    assert any(t.startswith("(harness note: your detached branch") and "c2 (stopped)" in t
               and "You stopped c2" in t for t in notes), notes
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
    assert any("at most 1 detached branches may be live" in t for t in results), results
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


# ------------------------------------------------------------------ the producers, statically
from graft import compaction as compaction_mod, db as db_mod, harness as harness_mod, machine as machine_mod, \
    skills as skills_mod

# Every function or constant that writes text a model reads. A producer that is renamed or removed
# fails the check (it must be listed again), so a new place that words a note or a refusal is
# added here, not missed.
PRODUCERS = {
    harness_mod: ["said", "merge_head", "check_skill_link", "clone_entries", "_typed_list", "cancel_ids",
                  "Harness.cancel_text", "Harness.system", "Harness.load", "Harness._names", "Harness._running_note",
                  "Harness.wake_note", "Harness.default_join_rows", "Harness.fit_join", "Harness.default_join",
                  "Harness.resolve_seed", "Harness.seed_index", "Harness.prepare_clone", "Harness.refuse_clone",
                  "Harness._prepare_clone", "Harness._loop", "Harness._missing_results",
                  "NOT_RUN_INTERRUPTED", "NOT_RUN_JOINING", "NOT_RUN_BESIDE_CLONE", "CANCELLED_WHILE_RUNNING",
                  "CANCELLED_BY_THE_TOOL", "FAILED_AFTER_INVOKED", "WITHDRAWN_SEED"],
    machine_mod: ["clone_refusal", "decline_why", "clone_why", "cancel_refusal", "_cancel_rows", "refuse_clone_call",
                  "refuse_cancel_call", "clone_refusal_not_written", "decline", "decline_failed_step",
                  "preparation_crashed", "drop_clone", "answer_pending", "_cut_outcome", "ToolCall"],
    db_mod: ["WITHDRAWN_NOTE"],
    compaction_mod: ["COMPACT_TOOL", "EXTRACT_INSTRUCTION", "WITHDRAWN_INDEX", "index_text", "extract_request",
                     "SUMMARY_HEADER", "COMPACT_INSTRUCTION", "summary_turn"],
    seeder_mod: ["NOUL_TRUE", "NOUL_FALSE", "reduced_state", "reduced_question", "jev_request"],
    skills_mod: ["LOAD_SKILL_TOOL", "JOIN_TOOL", "CANCEL_TOOL", "CLONE_INSTRUCTIONS", "JOIN_INSTRUCTIONS",
                 "BACKGROUND_INSTRUCTIONS", "load_skill", "skill_listing"],
}
# graft's own identifiers (store and transaction kinds, a graft's kind, its reason, the keys of the
# harness's logs) and its audit records: compared and stored, never shown to a model.
INTERNAL = {"clone", "join", "joined", "background", "focus", "cancelled", ":clone_tx", "DEFAULT JOIN ", "CLONE ",
            "BACKGROUND", "FOCUS"}
# Calls whose text is never shown to a model: the host's trace and the lines the user's callbacks get.
NOT_MODEL_FACING = {"trace", "_report", "_report_failure", "emit", "DecisionFault", "Aborted", "RuntimeError",
                    "TransitionRefused", "AssertionError", "ValueError", "TypeError"}


def _lookup(module, name: str):
    obj = module
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _strings(node: ast.AST) -> list[str]:
    """The string constants of a function's or a value's source, but its docstring and the text of
    calls that never reach a model (NOT_MODEL_FACING)."""
    out: list[str] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.Call):
            f = n.func
            called = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if called in NOT_MODEL_FACING:
                return
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            for child in n.body[1:]:
                visit(child)
            for child in [*getattr(n, "decorator_list", []), getattr(n, "args", None)]:
                if child is not None:
                    visit(child)
            return
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in INTERNAL:
            out.append(n.value)
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return out


def _source_node(module, name: str) -> ast.AST:
    obj = _lookup(module, name)
    if callable(obj) and not isinstance(obj, str):
        return ast.parse(textwrap.dedent(inspect.getsource(obj)))
    tree = ast.parse(inspect.getsource(module))
    for n in tree.body:  # a module-level value: its assignment
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return n
    raise LookupError(f"{module.__name__}.{name} is not assigned at module level")


def scanned(source: ast.AST) -> list[tuple[str, list[str]]]:
    return [(t, banned(t)) for t in _strings(source) if banned(t)]


@pytest.mark.parametrize("module,name", [(m, n) for m, names in PRODUCERS.items() for n in names],
                         ids=lambda x: x if isinstance(x, str) else x.__name__.split(".")[-1])
def test_every_model_facing_producer_uses_no_banned_word(module, name):
    """The string constants of each producer (its notes, results, refusals and prompts, whether or
    not a run reaches them), and a mutation of its own text is caught: the producer is scanned."""
    source = _source_node(module, name)
    strings = _strings(source)
    assert strings, f"{name}: no text found to scan"
    assert scanned(source) == [], f"{name} uses a banned word"
    target = next(t for t in strings if t.strip())
    mutated = ast.parse(ast.unparse(source).replace(repr(target), repr(target + " a clone"), 1)) \
        if repr(target) in ast.unparse(source) else None
    if mutated is None:  # an f-string part: mutate the constant node itself
        mutated = ast.parse(ast.unparse(source))
        for n in ast.walk(mutated):
            if isinstance(n, ast.Constant) and n.value == target:
                n.value = target + " a clone"
                break
    assert scanned(mutated), f"{name}: a banned word put in its text was not caught"


# ------------------------------------------------------------------ wordings no other run reaches
def test_the_merge_record_of_a_branch_that_wrote_nothing_and_plural_stops_say_branch(tmp_path):
    """A branch that brings back no rows the conversation lacks, and a stop of two branches (its
    result and the wake note that reports them)."""
    from _util import clone_in_turn
    store, h, rec = make(tmp_path=tmp_path, policy=lambda **_: text(t="Reporting."), name="h")

    async def go():
        await h.start()
        store.add_turn(tl="o", role="user", kind="text", content=REQUEST)
        store.add_turn(tl="o", role="assistant", kind="text", content="Started.")
        (c1,), _ = await clone_in_turn(h=h, use={"id": "t", "name": "branch",
                                                 "input": {"branch": [entry(skills=["website"], focus=False)]}})
        await h.do_join(child=c1, reason="done")  # it wrote nothing
        await h.drain_and_wake("o")
        await h.machines.stop()

    asyncio.run(go())
    notes = [t for w, t in store_texts(store) if w == "stored note"]
    assert any("it brings back no rows you do not already hold" in t for t in notes), notes
    check(rec.seen + store_texts(store))

    from graft.machine import CANCELLED, CancelCall
    stopped, _ = h.cancel_text(call=CancelCall(use={"id": "x"}, ids=["c1", "c2"], problems=[]),
                               outcomes={"c1": CANCELLED, "c2": CANCELLED})
    assert stopped.startswith("stopped -> c1, c2 (what you did in them so far merges"), stopped
    store.add_turn(tl="o", role="assistant", kind="text", content="Okay.")
    h.wake_note(src="o", joined=[("c1", CANCELLED), ("c2", CANCELLED)], handed_back=False, running=[], background=True)
    note = store.view("o")[-1]["content"]
    assert "detached branches c1 (stopped), c2 (stopped) merged" in note and "You stopped c1, c2" in note
    assert "they were stopped" in note and "The parts you were doing" in note, note
    check([("result", stopped), ("note", note)])
