"""What the model reads calls a graft a branch of itself and a join a merge (spec: What the model is
told: fork, merge and stop). No model-facing text names a clone, a join, a graft, the original, an
instance, a subagent, a worker, a helper or a delegate, says the assistant will remember something
for a graft, says focus, background or cancel (the model's words are current, detached and stop), or
uses `branch` as the tool (the tool is `fork`; a branch is what it makes).

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
from _util import ran, started, woke
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
    install

# Stems, so every inflection is caught (clone, clones, cloned, cloning; join, joins, joined, ...).
BANNED = re.compile(r"\bclon|\bjoin|\binstances?\b|\bsub-?agents?\b|\bworkers?\b|\bhelpers?\b|\bdelegat"
                    r"|\bgrafts?\b|\boriginal\b|the assistant will|\bbackground|\bcancel|\bfocus"
                    # `branch` as the tool (the model's tool is `fork`); the noun, a branch, stays
                    r"|\bbranch\(|`branch`|\"name\": \"branch\"|\bbranch(?:ed|ing)\b"
                    r"|\b(?:call|may|you|not|to) branch\b|\bbranch (?:again|calls?|refused|not executed|once more|on its own)\b"
                    # a join reason (graft's own; the model's merge takes none)
                    r"|\bout_of_scope\b|reason=['\"]?(?:done|failed)\b|\(done\)|\(failed\)",
                    re.IGNORECASE)
REQUEST = "Build a website and a slide deck for the dental practice."
GATE_TIMEOUT = 10


def banned(text: str) -> list[str]:
    return sorted({m.group().lower() for m in BANNED.finditer(text)})


def text(t):
    return [{"type": "text", "text": t}]


def use(tool, use_id, **args):
    return {"type": "tool_use", "id": use_id, "name": tool, "input": args}


def entry(skills, focus, **extra):
    return {"skills": skills, "focus": focus, **extra}


def branched(*entries):
    """A fork call's input from entries (the one with focus=True is `current`)."""
    from _util import branch_input
    return branch_input(*entries)


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


def make(tmp_path, policy, name, **kw):
    store = Store(str(tmp_path / f"{name}.db"))
    install(store)
    rec = Recorder(policy)
    h = Harness(store=store, provider=FakeProvider(policy=rec), seeder=None, **kw)
    return store, h, rec


def check(found: list[tuple[str, str]]):
    bad = [(where, banned(t), t[:200]) for where, t in found if banned(t)]
    assert not bad, "model-facing text uses a banned word:\n" + "\n".join(map(str, bad))


# ------------------------------------------------------------------ the text itself
def test_the_built_in_skills_and_tools_say_branch_and_merge():
    names = {s.name for s in CATALOG}
    assert {"fork", "merge", "detached"} <= names and not {"clone", "join"} & names
    assert set(BUILTIN) == {"fork", "merge", "detached"}
    tools = {t["name"] for s in CATALOG for t in s.tools}
    assert {"fork", "merge", "stop", "load_skill"} <= tools and not {"clone", "join"} & tools
    branch_tool = next(t for s in CATALOG for t in s.tools if t["name"] == "fork")
    assert set(branch_tool["input_schema"]["properties"]) == {"current", "detached"}
    merge_tool = next(t for s in CATALOG for t in s.tools if t["name"] == "merge")
    assert set(merge_tool["input_schema"]["properties"]) == {"reason"} and merge_tool["input_schema"]["required"] == []
    assert merge_tool["input_schema"]["properties"]["reason"]["type"] == "string"  # its own words: no enum
    assert "enum" not in merge_tool["input_schema"]["properties"]["reason"]
    assert set(branch_tool["input_schema"]["properties"]["current"]["properties"]) == {"id", "skills", "messages", "reason"}
    # the first-person framing: a branch is you, and what you did there is your memory
    focus, background, original = (" ".join(t.split()) for t in (JOIN_INSTRUCTIONS, BACKGROUND_INSTRUCTIONS,
                                                                   CLONE_INSTRUCTIONS))
    assert "on a branch of yourself" in focus and "on a detached branch of yourself" in background
    assert "yours on other branches or on main" in focus and "yours on other branches or on main" in background
    assert "the reason you wrote for this branch" in focus and "the reason you wrote for this branch" in background
    assert "what you will remember of this work when the branch merges" in background
    assert "forking yourself: each branch you fork is you" in original and "in the first person" in original
    # main stays main: `current` names a new branch, never main itself
    assert "Forking never moves you off main" in original and "never main itself" in original


def test_no_built_in_skill_text_or_tool_uses_a_banned_word():
    found = []
    for s in CATALOG:
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
              "background grafts", "the original", "cancel", "Cancelled", "a background branch", "cancelling", "focus: true", "the focus branch",
              "call branch", "branch(current=", "`branch`", "branched ->", "branch refused", "you branch only",
              "no branch calls left", "branching is not possible", "out_of_scope", "merge(reason='done')",
              "merge(reason=\"failed\")", "c2 (done)", "c2 (failed)"):
        assert banned(t), t
    for t in ("your branch c2 merged", "stop", "a detached branch", "fork", "forked ->", "each branch you fork",
              "a branch never stops another branch", "the current branch", "the same assistant", "rejoice", "grafted",
              "the work is done", "it failed", "site merged after it failed",
              "originally"):
        assert not banned(t), t


# ------------------------------------------------------------------ representative runs
def test_background_parts_cancel_and_refusals_say_branch_and_merge(tmp_path):
    """Two background parts, a refused call (bad entries, a tool beside it), a graft that tries to
    cancel and then merges with a reason, a cancel by the original (bad ids among them), the merge
    result that reports it, and a new branch for the stopped part, which is allowed (no harness limit)."""
    slides_gate = threading.Event()
    website_gate = threading.Event()  # c1 is still running when the original cancels: it is not named
    waits = []  # each gate wait's outcome, asserted in the test body (a failure on a model-call thread is a failed run)

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        if "fork" in names:
            if last["type"] == "text" and not woke(last) and last["text"] == REQUEST:
                return [use(tool="fork", use_id="b1", **branched(
                            entry(skills=["merge"], focus=False), entry(skills=["fork"], focus=False),
                            entry(skills=["website"], focus=False, purpose="p"), entry(skills=["slides"], focus=False, id="Bad Id"))),
                        use(tool="load_skill", use_id="l1", name="website")]
            if ran(last) and "b1" == last["tool_use_id"]:
                return [use(tool="fork", use_id="b2", **branched(entry(skills=["website"], focus=False), entry(skills=["slides"], focus=False)))]
            if last["type"] == "text" and not woke(last) and last["text"] == "Stop the slides.":
                return [use(tool="stop", use_id="x1", ids=["c2", "c9", "o", 5])]
            if ran(last) and last["tool_use_id"] == "x1":
                website_gate.set()
                slides_gate.set()  # c2 is stopped: the slides branch forked again later finishes
                return text("Okay.")
            if woke(last) and any("c2 was stopped before it finished" in str(b.get("content"))
                                  for b in messages[-1]["content"]):
                return [use(tool="fork", use_id="b3", **branched(entry(skills=["slides"], focus=False)))]
            return text("Okay.")
        if "site_write" in names:
            if not any(ran(b) for m in messages for b in m["content"]):
                return [use(tool="stop", use_id="g1", ids=["c2"])]
            waits.append(("website", website_gate.wait(GATE_TIMEOUT)))
            return [{"type": "text", "text": "Website built."}, use(tool="site_write", use_id="g3", page="home", content="hi"),
                    use(tool="merge", use_id="g4", reason="The home page is written; nothing else is left.")]
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
    assert any(t.startswith("fork refused") and "'merge' is linked by the harness" in t
               and "'fork' cannot be held by a branch" in t for t in results), results
    assert any(t.startswith("forked -> ") for t in results), results
    cancel = sorted({t for t in results if t.startswith("stopped -> ")})  # seen once per later model call
    assert len(cancel) == 1 and cancel[0].startswith("stopped -> c2 (what you did in it"), results
    assert "c9: not stopped: no such branch" in cancel[0] and "o: not stopped:" in cancel[0], cancel
    assert "ids[3]: not stopped: a int is not a branch id" in cancel[0], cancel
    assert len({t for t in results if t.startswith("forked -> ")}) == 2, results  # a stopped part may be forked again
    assert any(t.startswith("c2 was stopped before it finished.") and "you had run there: no tools" in t
               for t in results), results
    check(rec.seen + store_texts(store))


def test_focus_merges_hand_back_and_failure_say_branch_and_merge(tmp_path):
    """A current branch that asks and merges at its turn's end; one that merges at once with a
    reason (it hands the request back); one whose run fails. Each merge result is main's."""
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        user = last["type"] == "text" and not woke(last)
        if "fork" in names:
            if user and last["text"] == "Build me a website.":
                return [use(tool="fork", use_id="f1", **branched(entry(skills=["website"], focus=True)))]
            if user and last["text"] == "Book a room.":
                return [use(tool="fork", use_id="f2", **branched(entry(skills=["calendar"], focus=True)))]
            if user and last["text"] == "Make slides.":
                return [use(tool="fork", use_id="f3", **branched(entry(skills=["slides"], focus=True)))]
            return [] if woke(last) else text("Okay.")
        if "site_write" in names:
            return text("Which colors?")
        if "calendar_book" in names:
            return [use(tool="merge", use_id="m1", reason="I cannot tell which room; ask on main.")]
        raise ValueError("provider exploded")

    store, h, rec = make(tmp_path=tmp_path, policy=policy, name="b")

    async def go():
        await h.start()
        await h.user_says("Build me a website.")
        await h.user_says("Blue. Also send an email to Bob.")
        await h.user_says("Book a room.")
        await h.user_says("Make slides.")
        await h.settle()
        await h.close()

    asyncio.run(go())
    results = [t for w, t in rec.seen if w == "result"]
    assert any(t.startswith("c1 merged. Its replies above already reached the user") for t in results), results
    assert any(t.startswith("c2 merged without answering the user's last message") for t in results), results
    assert any(t.startswith("c3 merged after it failed") for t in results), results
    check(rec.seen + store_texts(store))


def test_a_merge_too_large_for_the_context_says_branch(tmp_path):
    """The size backstop's records and the wake note that says a result did not fit."""
    from _util import clone_in_turn
    store, h, rec = make(tmp_path=tmp_path, policy=lambda **_: text(t="Reporting."), name="e", window_tokens=120)

    async def go():
        await h.start()
        store.add_turn(tl="o", role="user", kind="text", content=REQUEST)
        store.add_turn(tl="o", role="assistant", kind="text", content="Started.")
        (c1,), _ = await clone_in_turn(h=h, use={"id": "t", "name": "fork",
                                                 "input": branched(entry(skills=["website"], focus=False))})
        store.add_turn(tl=c1, role="assistant", kind="text", content="Site built. " + "x" * 4000)
        await h.do_join(child=c1, reason="done")
        await h.drain_and_wake("o")
        await h.machines.stop()

    asyncio.run(go())
    assert h.join_fallbacks and h.join_fallbacks[0]["tier"] == 2
    results = [t for w, t in store_texts(store) if w == "stored result"]
    assert any("did not fit in your context" in t for t in results), results
    check(rec.seen + store_texts(store))


def test_the_old_tool_names_run_nothing(tmp_path):
    """A model that still calls `clone` or `join` is answered that the tool is not there: no graft
    is made or ended, and no capability tool runs."""
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        if "fork" in names:
            if ran(last) and last["tool_use_id"] == "old":
                return [use(tool="fork", use_id="new", **branched(entry(skills=["website"], focus=True)))]
            if ran(last) or woke(last):
                return text("Okay.")
            return [use(tool="clone", use_id="old", clone=[{"skills": ["website"]}])]
        if ran(last) and last["tool_use_id"] == "j":
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
    assert made == ["c1"]  # only the `fork` call made a branch
    # `join` did not end it: its turn ended on its reply, and the harness merged it
    assert store.timeline("c1")["join_reason"] == "done"
    assert [r["tool_name"] for r in store.view("c1")][-2:] == ["merge", "merge"]
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
        if viewer == "o" and trigger is None and any(r.get("row_id") is None for r in rows) \
                and any(r.get("owner_timeline") == "c1" for r in rows):  # the merge's pending view
            forced.append(viewer)
            return ["forced"]
        return real(rows=rows, viewer=viewer, trigger=trigger, require_user_end=require_user_end)

    store, h, rec = make(tmp_path=tmp_path, policy=lambda **_: text(t="Reporting."), name="g")

    async def go():
        await h.start()
        store.add_turn(tl="o", role="user", kind="text", content=REQUEST)
        store.add_turn(tl="o", role="assistant", kind="text", content="Started.")
        (c1,), _ = await clone_in_turn(h=h, use={"id": "t", "name": "fork",
                                                 "input": branched(entry(skills=["website"], focus=False))})
        store.add_turn(tl=c1, role="assistant", kind="text", content="Site built.")
        monkeypatch.setattr(harness_mod, "validate", failing)
        await h.do_join(child=c1, reason="done")
        monkeypatch.setattr(harness_mod, "validate", real)
        await h.machines.stop()

    asyncio.run(go())
    notes = [t for w, t in store_texts(store) if w == "stored note"]
    assert forced, "the merge's validation was never forced to fail"
    assert any("site merged, but its messages were skipped" in t or "c1 merged, but its messages were skipped" in t
               for t in notes), notes
    check(rec.seen + store_texts(store))


# ------------------------------------------------------------------ the producers, statically
from graft import compaction as compaction_mod, db as db_mod, harness as harness_mod, machine as machine_mod, \
    skills as skills_mod

# Every function or constant that writes text a model reads. A producer that is renamed or removed
# fails the check (it must be listed again), so a new place that words a note or a refusal is
# added here, not missed.
PRODUCERS = {
    harness_mod: ["merge_result", "SKIPPED_NOTE", "MERGED", "check_skill_link", "clone_entries", "_strings", "_messages", "cancel_ids",
                  "fork_result", "Harness.resolve_messages", "Harness.message_index", "ENTRY_SHAPE",
                  "Harness.cancel_text", "Harness.system", "Harness.load", "Harness._names", "Harness._running_note",
                  "Harness.merge_call", "Harness.fit_join", "Harness.default_join",
                  "Harness.task_of", "Harness._stop_call", "_entry", "NOT_A_DRAWN_ID", "NO_FREE_ID", "Harness.prepare_clone", "Harness.refuse_clone",
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
    skills_mod: ["LOAD_SKILL_TOOL", "JOIN_TOOL", "CANCEL_TOOL", "MESSAGES", "REASON", "CLONE_INSTRUCTIONS", "JOIN_INSTRUCTIONS",
                 "BACKGROUND_INSTRUCTIONS", "load_skill", "skill_listing"],
}
# graft's own identifiers (store and transaction kinds, a graft's kind, its reason, the keys of the
# harness's logs) and its audit records: compared and stored, never shown to a model.
INTERNAL = {"clone", "join", "joined", "background", "focus", "cancelled", "out_of_scope", "FOCUS ", ":clone_tx", "DEFAULT JOIN ", "CLONE ",
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
def test_the_merge_result_of_a_branch_that_wrote_nothing_and_plural_stops_say_branch(tmp_path):
    """A branch that brings back no rows the conversation lacks, a stop of two branches, and every
    merge result wording (each reason, focus or not, each size tier)."""
    from _util import clone_in_turn
    from graft.harness import merge_result
    store, h, rec = make(tmp_path=tmp_path, policy=lambda **_: text(t="Reporting."), name="h")

    async def go():
        await h.start()
        store.add_turn(tl="o", role="user", kind="text", content=REQUEST)
        store.add_turn(tl="o", role="assistant", kind="text", content="Started.")
        (c1,), _ = await clone_in_turn(h=h, use={"id": "t", "name": "fork",
                                                 "input": branched(entry(skills=["website"], focus=False))})
        await h.do_join(child=c1, reason="done")  # it wrote nothing
        await h.drain_and_wake("o")
        await h.machines.stop()

    asyncio.run(go())
    results = [t for w, t in store_texts(store) if w == "stored result"]
    assert any(t.startswith("c1 merged; it brings back no messages you do not already have.") for t in results), results
    check(rec.seen + store_texts(store))

    from graft.machine import CANCELLED, CancelCall
    stopped, _ = h.cancel_text(call=CancelCall(use={"id": "x"}, ids=["c1", "c2"], problems=[]),
                               outcomes={"c1": CANCELLED, "c2": CANCELLED})
    assert stopped.startswith("stopped -> c1, c2 (what you did in them so far merges"), stopped
    worded = [merge_result(name="site", reason=reason, focus=focus, ran=["site_write (returned an error)"],
                           dropped=dropped, running="No other branch of yours is still running.", empty=empty)
              for reason in ("done", "out_of_scope", "failed", CANCELLED) for focus in (True, False)
              for empty in (True, False) for dropped in (None, "What you did in site did not fit.")]
    assert all(t.endswith("end your turn without saying anything.") for t in worded)
    check([("result", stopped)] + [("result", t) for t in worded])
