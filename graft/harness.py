"""The Graft harness loop.

1. Read a timeline's context view (validated).
2. Resolve its tools through timeline_skills -> skill_tools.
3. Call the model (in a daemon worker thread, so background grafts run concurrently).
4. Insert the new turns once and append their pointers; execute tools.

Clone is one call, `clone(clone=[{purpose, skills, focus, seed_quotes | seed_row_ids}, ...])`, and
one atomic batch: the harness resolves every entry's seed to pointers, validates, and commits all
of them or none (on violations the source gets them back for one retry; any error inside clone
handling is a refusal too, never a half-made clone). At most one entry has focus=true: that
graft takes the conversation and the source is suspended until it joins. The other grafts run
in the background as asyncio tasks, never speak to the user, and the source keeps the
conversation.

Join is deterministic (no model call): the harness appends pointers to the text rows of the
clone's working view that the source lacks. A focus graft's join commits at once. A background
graft freezes when it finishes (`queued`) and its join waits on its source's queue, in finish
order, until the state machine accepts a drain: the source holds the conversation and is at rest
(graft/machine.py). That happens before a new user message is appended, at the end of the
source's turn, when a graft finishes while the source is idle, and, when a focus graft joins
back, before that focus join. After a drain that leaves no user message to answer, the source is
woken once with a closing harness note: to report (clone refused), or, if a request was handed
back, to route it.

The lifecycle is the state machine's (graft/machine.py): which grafts exist and of what kind,
which have raised their join event, what is queued, and what may happen now. The harness keeps
no copy of that; it dispatches and is told accepted or refused. Every join, a focus graft's as
much as a background graft's, is a queue entry carrying its reason and trigger: the graft is
`queued` until its join has been written to the source and only then `joined`, so a commit that
fails loses nothing and is retried at the next commit point.

The conversation lock (`Harness._conv`) is serialisation only: it keeps a user turn and a wake
run from interleaving their awaits. It decides nothing. Whether a queued join may be applied is
the machine's `idle`/`busy` answer to `drain_begin`, asked while the lock is held. All store and World access stays on the event-loop thread; there is no locking on external
state. Commits are recorded in `transactions`.
A clone may not join `done` before it has replied to the user's last message.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from . import callsites, compaction
from . import seeder as seeder_mod
from .db import Graft, Store, render, validate
from .machine import Machines, QueuedJoin, TransitionRefused
from .providers import Provider, prefix_nonce, window_tokens_for
from .skills import BASE_SYSTEM, BUILTIN, MAX_CLONE_ENTRIES, World, load_skill, skill_listing

CLONE_ATTEMPTS = 2  # the first clone call plus one retry after violations
# Clone calls a source may commit in wake runs between two user messages. With the per-request
# re-clone block this is a second, unconditional bound on clone/wake loops.
MAX_WAKE_CLONE_CALLS = 1
# Joins that may hand one request back to a timeline (any part, any reason) before the timeline
# stops cloning for that request altogether. A last bound under the per-part blocks.
MAX_REQUEST_HANDBACKS = 8
# Clone calls committed for one user message, by any timeline, across the whole chain of
# hand-backs and wakes it sets off. Reset only by a user message. This bound depends on no key:
# whatever a model seeds and however a hand-back is attributed, cloning stops here.
MAX_CLONE_CALLS_PER_MESSAGE = 4
# A join must leave the source's working view under this fraction of the model window:
# estimate_tokens is chars / 4 (rough), and the reply needs room too.
JOIN_WINDOW_FRACTION = 0.85
_NONCE_TAG = re.compile(r"^\[conversation [^\]]*\]\s*")


def check_skill_link(store: Store, source: str, skills: list[str]) -> list[str]:
    """The one place skill linking is checked. Grafts are one level deep: only the original
    clones, so `clone` is never linked to a clone."""
    v = []
    known = set(store.skill_names())
    if not skills:
        v.append("select at least one skill")
    for s in skills:
        if s not in known:
            v.append(f"unknown skill '{s}'")
        elif s in ("join", "background"):
            v.append(f"'{s}' is linked by the harness (join to a focus clone, background to a "
                     "background clone); do not select it")
        elif s == "clone":
            v.append("'clone' cannot be linked to a clone (only the original clones)")
    return v


@dataclass
class CloneEntry:
    """One validated entry of a clone call. `index` is its position in the model's list (the one
    violations name); the list fields are empty when the model's value was not usable."""
    index: int
    raw: dict
    purpose: str
    skills: list[str]
    focus: bool
    seed_quotes: list[str]
    seed_row_ids: list[int]


def _typed_list(entry: dict, *, name: str, kind: type, index: int, violations: list[str]) -> list:
    value = entry.get(name)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, kind) and not isinstance(x, bool) for x in value):
        what = "strings" if kind is str else "integers"
        violations.append(f"clone[{index}]: {name} must be a list of {what}")
        return []
    return list(value)


def clone_entries(args: object) -> tuple[list[CloneEntry], list[str]]:
    """The entries of a clone call's input and its shape violations. `clone` is a list of one to
    MAX_CLONE_ENTRIES objects, each with a purpose, an explicit boolean focus, `skills` (strings,
    no duplicates) and optional `seed_quotes` (strings) and `seed_row_ids` (integers); at most
    one focus=true. The input is model-written: nothing about its shape is assumed. Violations
    name the entry by its index in the model's list."""
    raw = args.get("clone") if isinstance(args, dict) else None
    if not isinstance(raw, list) or not raw:
        return [], ["`clone` must be a list of one or more entries, each "
                    "{purpose, skills, focus, seed_quotes or seed_row_ids}"]
    if len(raw) > MAX_CLONE_ENTRIES:  # refused whole, before any per-entry work
        return [], [f"a clone call takes at most {MAX_CLONE_ENTRIES} entries (got {len(raw)})"]
    v: list[str] = []
    entries: list[CloneEntry] = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            v.append(f"clone[{i}]: must be an object")
            continue
        purpose = e.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            v.append(f"clone[{i}]: purpose is required")
        if not isinstance(e.get("focus"), bool):
            v.append(f"clone[{i}]: focus is required and must be true or false")
        skills = _typed_list(e, name="skills", kind=str, index=i, violations=v)
        repeated = sorted({k for k in skills if skills.count(k) > 1})
        if repeated:
            v.append(f"clone[{i}]: skills lists {repeated} more than once")
        entries.append(CloneEntry(
            index=i, raw=e, purpose=purpose if isinstance(purpose, str) else "", skills=skills,
            focus=e.get("focus") is True,
            seed_quotes=_typed_list(e, name="seed_quotes", kind=str, index=i, violations=v),
            seed_row_ids=_typed_list(e, name="seed_row_ids", kind=int, index=i, violations=v)))
    focus = [e.index for e in entries if e.focus]
    if len(focus) > 1:
        v.append(f"at most one entry may have focus=true (entries {focus} do): only one clone can "
                 "take the conversation")
    return entries, v


async def in_daemon_thread(fn: Callable[[], object]) -> object:
    """Run fn in a daemon thread and await its result. A model call must not pin the process: a
    daemon thread is abandoned at exit, where an executor thread would be joined."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def deliver(result: object, error: BaseException | None) -> None:
        if future.cancelled():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def work() -> None:
        result, error = None, None
        try:
            result = fn()
        except StopIteration as e:  # cannot be set on a future
            error = RuntimeError(f"StopIteration in a model call: {e}")
        except BaseException as e:
            error = e
        try:
            loop.call_soon_threadsafe(deliver, result, error)
        except RuntimeError:  # the loop is closed: nobody is waiting
            pass

    threading.Thread(target=work, daemon=True, name="graft-model-call").start()
    return await future


def _reason(reason: str) -> str:
    return reason if reason in ("done", "out_of_scope", "failed") else "failed"


class Harness:
    def __init__(self, store: Store, provider: Provider, world: World | None = None,
                 trace: Callable[[str], None] | None = None, max_steps: int = 10,
                 persona: str = BASE_SYSTEM, nonce: str | None = None,
                 compact_tokens: int | None = None, compaction_mode: str = "turns", seeder=...,
                 window_tokens: int | None = ..., emit: Callable[[str], None] | None = None):
        self.store = store
        self.persona = persona  # user-provided; identical, verbatim, on every timeline
        self.provider = provider
        self.world = world or World()
        self.machines = Machines()
        self.trace = trace or (lambda s: None)
        self.max_steps = max_steps
        self.silent_done_refusals: list[str] = []  # clones whose join(done) was refused
        # Seed actor (graft/seeder.py): None = the original names the seed itself (GRAFT_SEEDER=model)
        self.seeder = seeder_mod.make_seeder() if seeder is ... else seeder
        self.seeder_log: list[dict] = []  # per clone: Jev latency, tokens, scores, selection, fallback
        self.out: list[str] = []  # user-facing texts: only the conversation holder's
        self.emit = emit or (lambda s: None)  # called with each user-facing text as it is said
        self._mark = 0  # self.out[:_mark] was already returned by user_says / settle
        # Background grafts. Their lifecycle and the join queue are the state machine's; these are
        # the harness's own handles on its tasks. The conversation lock serialises user turns and
        # wake runs and decides nothing (see the module docstring).
        self._conv = asyncio.Lock()
        self._queued: dict[str, asyncio.Event] = {}  # background graft -> set when its join is queued
        self._tasks: dict[str, asyncio.Task] = {}  # background graft -> its task
        # per wake run: {source, joined: [(clone, reason)], report: True if clone was not allowed}
        self.wakes: list[dict] = []
        self.wake_clone_calls: dict[str, int] = {}  # source -> clone calls in wake runs since the last user message
        self.join_dropped: set[str] = set()  # clones none of whose rows fit in the source (size backstop)
        # (source, clones, error): a wake run that raised, a join commit that raised, and a
        # background task that ended with an error nobody had reported
        self.wake_errors: list[tuple[str, list[str], str]] = []
        self.commit_errors: list[tuple[str, list[str], str]] = []
        self.task_errors: list[tuple[str, list[str], str]] = []
        self._failures: list[Exception] = []  # the same errors, as raised, in order
        self.graft_errors: list[tuple[str, str]] = []  # (clone, error): its run raised; it joined failed
        # Per-run conversation nonce, prefixed to the first message of every request at render
        # time (never stored). None = no nonce (cache experiments: --cache-scope global).
        self.nonce = nonce
        # Compaction threshold (estimated tokens of a working view); None = never compact.
        self.compact_tokens = compact_tokens
        if compaction_mode not in ("turns", "summary"):
            raise ValueError(f"compaction_mode must be turns or summary, got {compaction_mode!r}")
        # turns: kept rows + compressed windows (default); summary: one summary row (ablation)
        self.compaction_mode = compaction_mode
        self.compactions: list[str] = []  # timelines that compacted, in order
        self.compaction_log: list[dict] = []  # per compaction: tokens before/after, kept, notes
        self.compaction_failures: list[str] = []  # timelines whose selection was refused twice
        self.compaction_failure_calls: list[dict] = []  # cost of refused-twice compactions
        self._after_compaction: dict[str, dict] = {}  # tl -> log event awaiting its next model call
        # Model context window (tokens); None = joins are not size-checked. Default (...): the
        # GRAFT_WINDOW_TOKENS override, else the provider model's known window (providers.CONTEXT_WINDOWS).
        self.window_tokens = window_tokens_for(provider) if window_tokens is ... else window_tokens
        self.join_fallbacks: list[dict] = []  # joins reduced to fit: {clone, tier, before, after}
        # Re-clone blocks, per part. A part is one graft's share of a request: the request the
        # graft was working on and the graft's catalog skills. The request is the user message a
        # focus graft hands back (its trigger), or else the source's current request when the
        # graft was cloned, recorded then. timeline -> {(request row or None, skills)} that failed
        # or were handed back twice: no clone entry with any of those skills while that request is
        # the timeline's current one. Cleared by a user message to the timeline.
        self.request_at_clone: dict[str, int | None] = {}  # graft -> its source's current request at clone time
        self.reclone_block: dict[str, set[tuple[int | None, frozenset[str]]]] = {}
        # (timeline, request, skills) -> joins that handed that part back
        self.join_arrivals: dict[tuple[str, int | None, frozenset[str]], int] = {}
        self.request_handbacks: dict[tuple[str, int | None], int] = {}  # (timeline, request) -> all hand-backs
        self.clone_calls_this_message = 0  # clone calls committed since the last user message, by anyone

    async def complete(self, tag: str, system: str, messages: list[dict], tools: list[dict], **kw
                       ) -> tuple[list[dict], dict | None]:
        """Every model call goes through here: applies the run nonce, the usage-role tag, and the
        call site's reasoning effort and model (graft/callsites.py). The whole call runs in a
        daemon worker thread so timelines run concurrently; the provider's usage accounting is
        per-thread, so the caller's thread state is carried over and the call-log entry comes
        back with the blocks. Returns (blocks, call-log entry or None)."""
        cs = callsites.settings(tag, messages)
        provider = self.provider
        carry = getattr(provider, "thread_state", None)
        state = carry() if carry else None
        sent = prefix_nonce(messages, self.nonce)

        def call() -> tuple[list[dict], dict | None]:
            tagger = getattr(provider, "usage_tag", None)
            with provider.adopt(state) if state is not None else contextlib.nullcontext():
                with tagger(tag) if tagger else contextlib.nullcontext():
                    blocks = provider.complete(system, sent, tools, **kw, **cs)
                return blocks, getattr(provider, "last_call", lambda: None)()

        return await in_daemon_thread(call)

    def role(self, tl: str) -> str:
        return "original" if self.store.timeline(tl)["source_id"] is None else f"clone:{tl}"

    # ------------------------------------------------------------ setup
    async def start(self, root: str = "o") -> None:
        self.store.create_timeline(root, None, ["clone"])
        await self.machines.start_root(root)
        self._mirror()

    def _mirror(self) -> None:
        for t in self.machines.timelines:
            self.store.set_status(t, self.machines.state(t))

    def system(self, tl: str) -> str:
        """Persona (same on every timeline) + skill listing (the only per-timeline part)."""
        names = [s["name"] for s in self.store.skills_for(tl)]
        parts = [self.persona, skill_listing(
            self.store, names,
            "Your skills (call load_skill(name) to read a skill's full instructions before using it):")]
        if "clone" in names:
            catalog = [r[0] for r in self.store.con.execute(
                "SELECT name FROM skills ORDER BY skill_id") if r[0] not in BUILTIN]
            parts.append(skill_listing(self.store, catalog, "Skill catalog (available to clones):"))
        return "\n\n".join(parts)

    def load(self, tl: str, name: str) -> tuple[str, bool]:
        return load_skill(self.store, {s["name"] for s in self.store.skills_for(tl)}, name)

    def is_background(self, tl: str) -> bool:
        """Was tl created as a background graft? The machine's record of the clone call."""
        return self.machines.kind(tl) == "background"

    def say(self, tl: str, text: str) -> None:
        """Only the branch holding the conversation speaks to the user; a background graft's text
        stays in its timeline."""
        if self.is_background(tl):
            self.trace(f"[{tl}] (background, not shown to the user) assistant: {text}")
            return
        self.out.append(text)
        self.emit(text)
        self.trace(f"[{tl}] assistant: {text}")

    def _take(self) -> list[str]:
        texts, self._mark = self.out[self._mark:], len(self.out)
        return texts

    def _names(self, clones: list[str], named: bool = True) -> str:
        """'background clones c1, c2' / 'clone c1'; with named=False just the noun."""
        if not clones:
            return "a clone"
        word = "background clone" if all(self.is_background(c) for c in clones) else "clone"
        word += "s" if len(clones) != 1 else ""
        return f"{word} {', '.join(clones)}" if named else word

    def _report_failure(self, *, kind: str, src: str, clones: list[str], error: Exception) -> None:
        """The one place a failed wake run, a failed join commit, or an unreported background
        task error is recorded and shown. The user is told at once (the line is user-facing
        output and goes through `emit`); the error itself is kept for `settle()` and for the
        background task that hit it."""
        err = f"{type(error).__name__}: {error}"
        names = self._names(clones)
        if kind == "wake":
            self.wake_errors.append((src, clones, err))
            line = f"(harness error: {names} joined, but reporting the result failed: {err})"
        elif kind == "commit":
            self.commit_errors.append((src, clones, err))
            line = (f"(harness error: committing the join of {names} failed: {err}; "
                    "the result is kept and the join will be retried)")
        else:
            self.task_errors.append((src, clones, err))
            line = f"(harness error: {names} stopped with an error: {err})"
        self._failures.append(error)
        self.trace(f"[{src}] {kind.upper()} FAILED for {clones}: {err}")
        self.out.append(line)
        self.emit(line)

    # ------------------------------------------------------------ user entry
    async def user_says(self, text: str) -> list[str]:
        """One user turn, given to the branch holding the conversation. Returns the user-facing
        texts emitted since the last return: this turn's replies, plus wake output that arrived
        while the conversation was idle. A wake or join-commit failure after the turn is part of
        that output (a harness error line), not an exception."""
        async with self._conv:
            leaf = self.store.running_leaf()
            await self._drain(leaf)  # queued background joins land before the message
            await self.machines.begin_turn(leaf)  # refused unless leaf is at rest
            rid = self.store.add_turn(leaf, "user", "text", text)
            self.reclone_block.pop(leaf, None)  # a new request may clone again
            self.clone_calls_this_message = 0
            self.wake_clone_calls.pop(leaf, None)
            self.trace(f"[{leaf}] user (row {rid}): {text}")
            failed_commits = len(self.commit_errors)
            await self._turn(leaf)
            if len(self.commit_errors) == failed_commits:  # a commit that just failed is not retried at once
                await self._drain_waiting()
            return self._take()

    async def settle(self) -> list[str]:
        """Wait until no background graft is running and every join that can commit has (joins
        queued on a suspended source wait for its focus graft). Returns the user-facing texts
        emitted since the last return. Raises what failed meanwhile in background tasks and in
        its own drain: the one error, or an ExceptionGroup of all of them; each is raised once."""
        errors: list[Exception] = []
        while True:
            live = [t for t in self._tasks.values() if not t.done()]
            if live:
                await asyncio.wait(live)
                continue
            done, self._tasks = self._tasks, {}
            for t in done.values():
                error = None if t.cancelled() else t.exception()
                if isinstance(error, ExceptionGroup):
                    errors += [e for e in error.exceptions if e not in errors]
                elif isinstance(error, Exception) and error not in errors:
                    errors.append(error)
            async with self._conv:
                before = len(self._failures)
                await self._drain_waiting()
                errors += [e for e in self._failures[before:] if e not in errors]
            if not self._tasks:
                break
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("background grafts failed", errors)
        return self._take()

    async def close(self) -> None:
        """Shut down. Cancel the running background grafts; every background graft that has not
        raised its join event then joins `failed`, including one whose task never ran a step.
        One whose join event cannot be raised ends `abandoned`. A task error nobody reported is
        reported now. The joins that can be applied without running anything are applied, and
        the state machines stop. No model call is started here."""
        live = [t for t in self._tasks.values() if not t.done()]
        for t in live:
            t.cancel()
        if live:
            await asyncio.wait(live)
        for child, t in self._tasks.items():
            error = None if t.cancelled() else t.exception()
            if isinstance(error, Exception) and not self._reported(error):
                self._report_failure(kind="task", src=self.machines.source(child), clones=[child], error=error)
        self._tasks.clear()
        for child in self.machines.background():
            try:
                await self._join_background(child, reason="failed")
            except Exception as e:  # its join event cannot be raised: a terminal state, and say so
                await self.machines.abandon(child)
                self._report_failure(kind="task", src=self.machines.source(child), clones=[child], error=e)
        self._mirror()
        async with self._conv:
            for src in self.machines.waiting():
                await self._drain(src)  # refused for a suspended source: its joins stay queued
        await self.machines.stop()

    def _reported(self, error: BaseException) -> bool:
        parts = error.exceptions if isinstance(error, BaseExceptionGroup) else [error]
        return all(any(p is f for f in self._failures) for p in parts)

    def join_queued(self, child: str) -> Coroutine[Any, Any, bool]:
        """Awaitable: background graft `child` has finished and its join is queued."""
        return self._queued.setdefault(child, asyncio.Event()).wait()

    def background_task(self, child: str) -> asyncio.Task:
        """The task running background graft `child` (done once it has also offered its join)."""
        return self._tasks[child]

    # ------------------------------------------------------------ main loop
    def _store_results(self, tl: str, results: list[tuple[dict, str, bool]]) -> None:
        for use, content, err in results:
            self.store.add_turn(tl, "user", "tool_result",
                                {"tool_use_id": use["id"], "content": content, "is_error": err},
                                tool_name=use["name"])

    def _store_blocks(self, tl: str, blocks: list[dict]) -> None:
        for b in blocks:
            if b["type"] == "text":
                self.store.add_turn(tl, "assistant", "text", b["text"])
            elif b["type"] == "tool_use":
                self.store.add_turn(tl, "assistant", "tool_use", {"id": b["id"], "input": b["input"]},
                                    tool_name=b["name"])
            elif b["type"] == "thinking":
                payload = {k: v for k, v in b.items() if k != "type"}
                self.store.add_turn(tl, "assistant", "thinking", payload)

    async def maybe_compact(self, tl: str) -> None:
        """Compaction (spec: Compaction; graft/compaction.py). When the working view, ending in
        a fresh user message, is over the threshold, one `compact` call partitions the prefix into
        kept rows and compressed windows; the new view (items in order + the fresh message) is
        validated (partition, then turn structure like a seed or join), with one retry on
        violations, then recorded with Store.mark_compaction. The full timeline is unchanged."""
        rows = self.store.working_view(tl)
        if len(rows) < 2 or not compaction.needs_compaction(render(rows), self.compact_tokens):
            return
        prefix, fresh = rows[:-1], rows[-1]
        if self.compaction_mode == "summary":
            await self._summary_compact(tl, rows)
            return
        violations: list[str] | None = None
        calls: list[dict] = []
        for attempt in range(2):
            budget = int(self.compact_tokens * compaction.BUDGET_FRACTION)
            blocks, last = await self.complete(
                f"{self.role(tl)}:compaction", self.system(tl),
                compaction.extract_request(render(prefix), prefix, violations, budget,
                                           compaction.estimate_tokens(render(rows))),
                [compaction.COMPACT_TOOL], max_tokens=4000)
            if last:
                calls.append(last)
            items = compaction.parse_items(blocks)
            if items is None:
                violations = ["call the compact tool (no tool call was made)"]
                continue
            new_rows, problems = compaction.partition(prefix, items)
            for it in new_rows:
                if it["row_id"] is None:
                    it["owner_timeline"] = tl
            violations = problems or validate(new_rows + [fresh], tl)
            if not violations:
                size = compaction.estimate_tokens(render(new_rows + [fresh]))
                if size > budget:
                    violations = [f"the compacted conversation is about {size} tokens, over the budget of "
                                  f"{budget}; compress more rows into windows"]
            if violations:
                self.trace(f"[{tl}] compaction refused (attempt {attempt + 1}): {violations}")
                continue
            view_ids = [it["row_id"] if it["row_id"] is not None else
                        self.store.add_compressed_turn(tl, it["role"], it["content"], *it["window"])
                        for it in new_rows] + [fresh["row_id"]]
            self.store.mark_compaction(tl, view_ids)
            self._log_compaction(tl, rows, view_ids, attempt + 1, calls)
            self.trace(f"[{tl}] COMPACTED: {len(new_rows)} items for {len(prefix)} rows")
            return
        self.compaction_failures.append(tl)
        self.compaction_failure_calls.append({"timeline": tl, "output_tokens": sum(c["output"] for c in calls),
                                              "calls": len(calls)})
        self.trace(f"[{tl}] compaction abandoned after 2 attempts: {violations}")

    async def _summary_compact(self, tl: str, rows: list[dict]) -> None:
        """Ablation (--compaction summary): one summary of everything before the fresh message,
        stored as a user-role compressed row whose window is the whole prefix; view = [summary,
        fresh message]."""
        prefix, fresh = rows[:-1], rows[-1]
        blocks, last = await self.complete(f"{self.role(tl)}:compaction", self.system(tl),
                                           compaction.request(render(prefix)), [], max_tokens=2000)
        sid = self.store.add_compressed_turn(tl, "user", compaction.summary_turn(blocks),
                                             prefix[0]["row_id"], prefix[-1]["row_id"])
        view_ids = [sid, fresh["row_id"]]
        self.store.mark_compaction(tl, view_ids)
        self._log_compaction(tl, rows, view_ids, 1, [last] if last else [])
        self.trace(f"[{tl}] COMPACTED (summary) -> row {sid}")

    def _log_compaction(self, tl: str, rows: list[dict], view_ids: list[int], attempts: int,
                        calls: list[dict] | None = None) -> None:
        """Record a committed compaction for metrics: tokens before/after, kept rows, each
        compressed message with its window, the compaction calls' output tokens, the kept head
        (prefix rows kept unchanged before the first window), and, filled in by run(), the cached
        input tokens of the first model call after it."""
        prefix = rows[:-1]
        ids = [r["row_id"] for r in prefix]
        windows = []
        for rid in view_ids:
            # only messages written by this compaction; an earlier compressed row kept now is a keep
            w = self.store.compression_window(rid) if rid not in set(ids) else None
            if w:
                covered = ids[ids.index(w[0]):ids.index(w[1]) + 1]
                windows.append({"row_id": rid, "begin": w[0], "end": w[1], "covers": covered,
                                "text": self.store.turn(rid)["content"]})
        self.compactions.append(tl)
        self.compaction_log.append({
            "timeline": tl, "before_tokens": compaction.estimate_tokens(render(rows)),
            "after_tokens": compaction.estimate_tokens(render(self.store.working_view(tl))),
            "rows_before": len(rows), "kept_ids": [i for i in view_ids[:-1] if i in set(ids)],
            "windows": windows, "prefix": [(r["row_id"], r["content"]) for r in prefix],
            "fresh": rows[-1]["content"], "attempts": attempts,
            "output_tokens": sum(c["output"] for c in calls or []), "calls": len(calls or []),
            "kept_head": next((i for i, rid in enumerate(view_ids[:-1]) if rid not in set(ids)),
                              len(view_ids) - 1),
            "first_call_after": None})
        self._after_compaction[tl] = self.compaction_log[-1]

    async def run(self, tl: str) -> str:
        """One turn of a conversation-holding timeline that is at rest: idle -> turn, run, and
        back to idle unless the turn handed the conversation on. Returns how it ended (`_loop`)."""
        await self.machines.begin_turn(tl)
        return await self._turn(tl)

    async def _turn(self, tl: str, wake: bool = False, report: bool = False) -> str:
        """Run tl, which the caller has just put in `turn` or `reporting`, and close the turn. A
        focus graft whose run raises joins `failed` without the model (spec: Join), like a
        background graft: the error does not leave the conversation stuck on a dead clone."""
        try:
            end = await self._loop(tl, wake=wake, report=report)
        except Exception as e:
            await self.machines.try_end_turn(tl)  # leave tl usable; refused if it handed off
            self._mirror()
            if self.machines.kind(tl) != "focus":
                raise
            try:
                await self._join_focus(tl, reason="failed")
            except TransitionRefused:  # it had already raised its join event: not its own failure
                raise e from None
            self.graft_errors.append((tl, f"{type(e).__name__}: {e}"))
            self.trace(f"[{tl}] run failed -> join failed: {type(e).__name__}: {e}")
            return "joined"
        except BaseException:
            await self.machines.try_end_turn(tl)
            self._mirror()
            raise
        if end == "ended":
            await self.machines.end_turn(tl)
            self._mirror()
        return end

    async def _loop(self, tl: str, wake: bool = False, report: bool = False) -> str:
        """The model loop for tl. Returns "ended" (tl ended its turn and still holds the
        conversation), "handed_off" (it cloned a focus graft), or "joined" (it raised its join
        event). The conversation holder ends a turn on a text-only reply (it waits for the user).
        A background graft has nobody to wait for: a text-only reply joins it `done`, an empty
        response or an exhausted step budget joins it `failed`. `wake`: this run was started by
        the harness for background joins; `report`: and it may only report (no clone)."""
        clone_attempts = 0
        background = self.is_background(tl)
        await self.maybe_compact(tl)
        for _ in range(self.max_steps):
            rows = self.store.working_view(tl)
            violations = validate(rows, tl)
            if violations:
                raise RuntimeError(f"{tl} view invalid before model call: {violations}")
            tools = self.store.tools_for(tl)
            blocks, last = await self.complete(self.role(tl), self.system(tl), render(rows), tools)
            pending = self._after_compaction.pop(tl, None)
            if pending is not None and last:
                pending["first_call_after"] = {k: last[k] for k in ("input", "cached", "cache_write", "uncached")}
            if not any(b["type"] in ("text", "tool_use") for b in blocks):
                self.trace(f"[{tl}] (empty response)")
                if background:
                    await self._join_background(tl, reason="failed")
                    return "joined"
                return "ended"
            self._store_blocks(tl, blocks)
            for b in blocks:
                if b["type"] == "text":
                    self.say(tl, b["text"])
            uses = [b for b in blocks if b["type"] == "tool_use"]
            if not uses:
                if background:  # its reply is its result; nobody will answer it
                    await self._join_background(tl, reason="done")
                    return "joined"
                return "ended"
            allowed = {t["name"] for t in tools}
            join_use = next((u for u in uses if u["name"] == "join" and "join" in allowed), None)
            clone_use = next((u for u in uses if u["name"] == "clone" and "clone" in allowed), None)
            refused_join = None
            if join_use and str(join_use["input"].get("reason")) == "done" and not self.replied(tl):
                refused_join, join_use = join_use, None
                self.silent_done_refusals.append(tl)
                self.trace(f"[{tl}] join(done) refused: no reply since the last user message")
            if join_use:
                results = [(u, "not executed: this task is being joined", True) for u in uses if u is not join_use]
                results.append((join_use, "joined", False))
                self._store_results(tl, results)
                reason = _reason(str(join_use["input"].get("reason", "failed")))
                if background:
                    await self._join_background(tl, reason=reason)
                else:
                    await self._join_focus(tl, reason=reason)
                return "joined"
            if clone_use:
                # this call's role was routing into a clone: account it as the clone transaction
                retag = getattr(self.provider, "retag", None)
                if retag:
                    retag(last, f"{self.role(tl)}:clone_tx")
                others = [(u, "not executed: call clone on its own", True) for u in uses if u is not clone_use]
                blocked = self.clone_blocked(tl, clone_attempts, report=report)
                if blocked:
                    self.trace(f"[{tl}] clone not executed: {blocked}")
                    self._store_results(tl, others + [(clone_use, f"clone not executed: {blocked}", True)])
                    continue
                clone_attempts += 1
                children, focus, text = await self.clone_call(tl, clone_use,
                                                              retry_left=clone_attempts < CLONE_ATTEMPTS)
                self._store_results(tl, others + [(clone_use, text, children is None)])
                if children:
                    self.clone_calls_this_message += 1
                    if wake:
                        self.wake_clone_calls[tl] = self.wake_clone_calls.get(tl, 0) + 1
                    self._start_background(children)
                    if focus is not None:
                        await self.run(focus)
                        return "handed_off"
                    clone_attempts = CLONE_ATTEMPTS  # one clone call per message: now tell the user
                continue
            results = []
            for u in uses:
                if u is refused_join:
                    if background:
                        msg = ("join(reason='done') refused: you have not written your result yet. First "
                               "write your final reply saying what you did; then call join(reason='done').")
                    else:
                        msg = ("join(reason='done') refused: you have not replied to the user since "
                               "their last message. First tell the user, in one or two sentences, what "
                               "you did; then call join(reason='done').")
                    results.append((u, msg, True))
                elif u["name"] not in allowed:
                    results.append((u, f"tool '{u['name']}' is not available to you", True))
                elif u["name"] == "load_skill":
                    content, err = self.load(tl, str(u["input"].get("name", "")))
                    self.trace(f"[{tl}] load_skill({u['input'].get('name')}) -> {'error' if err else 'ok'}")
                    results.append((u, content, err))
                else:
                    content, err = self.world.execute(u["name"], u["input"])
                    self.trace(f"[{tl}] tool {u['name']}({json.dumps(u['input'])[:120]}) -> {content[:80]}")
                    results.append((u, content, err))
            self._store_results(tl, results)
        if background:
            self.trace(f"[{tl}] step budget exhausted -> join failed")
            await self._join_background(tl, reason="failed")
            return "joined"
        if self.machines.kind(tl) == "focus":
            self.trace(f"[{tl}] step budget exhausted -> join failed")
            await self._join_focus(tl, reason="failed")
            return "joined"
        return "ended"

    # ------------------------------------------------------------ background grafts
    def _start_background(self, children: list[str]) -> None:
        for c in children:
            if self.is_background(c) and c not in self._tasks:
                self._tasks[c] = asyncio.create_task(self._run_background(c), name=f"graft-{c}")

    async def _run_background(self, child: str) -> None:
        """A background graft's whole life: run to its join event, then offer the join to its
        source. If its run raises or is cancelled (see `close`) it joins `failed`; the machine
        refuses that if it had already raised its join event. A wake run or a join commit that
        fails here was already shown to the user; the task still ends with that error, so
        `settle()` raises it."""
        try:
            await self._loop(child)
        except asyncio.CancelledError:
            if await self._join_background(child, reason="failed"):
                self.trace(f"[{child}] background run cancelled -> join failed")
            raise
        except Exception as e:  # a mechanical failure joins `failed` without the model (spec: Join)
            if await self._join_background(child, reason="failed"):
                self.graft_errors.append((child, f"{type(e).__name__}: {e}"))
                self.trace(f"[{child}] background run failed -> join failed: {type(e).__name__}: {e}")
            else:
                raise
        src = self.machines.source(child)
        async with self._conv:  # waits while the source is mid-turn
            before = len(self._failures)
            await self.drain_and_wake(src)
            failed = self._failures[before:]
        if len(failed) == 1:
            raise failed[0]
        if failed:
            raise ExceptionGroup(f"joining background clone {child} failed", failed)

    async def _join_background(self, child: str, reason: str) -> bool:
        """A background graft's join event: ask the machine first. Accepted, it freezes now
        (`queued`) and waits on its source's queue with its reason; only then is the reason
        written to the store. Nothing is written to the source until the machine accepts a
        drain. False: the machine refused (it had already raised its join event) and nothing
        was written."""
        src = await self.machines.finish_background(child, reason=reason)
        if src is None:
            return False
        self.store.set_join(child, reason, trigger=None)
        self._mirror()
        self._queued.setdefault(child, asyncio.Event()).set()
        self.trace(f"JOIN EVENT {{clone: {child}, reason: {reason}, trigger: None}} -> queued on {src}")
        return True

    def part_of(self, child: str) -> frozenset[str]:
        """The part a graft was given: its linked catalog skills (harness-known, deterministic)."""
        return frozenset(k["name"] for k in self.store.skills_for(child) if k["name"] not in BUILTIN)

    def _note_arrival(self, tl: str, join: QueuedJoin) -> None:
        """A join handed its part of a request back to tl (out_of_scope or failed). Count it
        under the request the graft was working on and the graft's skills, and block re-cloning
        of that part if it failed or came back twice. A sibling's part is not touched. The
        request is harness-owned (`request_of`). A request of None (tl had no user message when
        it cloned) is counted like any other: a bound is never skipped."""
        if join.reason == "done":
            return
        req = self.request_of(join)
        part = self.part_of(join.child)
        self.request_handbacks[(tl, req)] = self.request_handbacks.get((tl, req), 0) + 1
        n = self.join_arrivals[(tl, req, part)] = self.join_arrivals.get((tl, req, part), 0) + 1
        if join.reason == "failed" or n >= 2:
            self.reclone_block.setdefault(tl, set()).add((req, part))

    def part_blocked(self, tl: str, skills: list[str]) -> bool:
        """Would a clone entry with these skills re-clone a part of tl's current request that
        already failed or was handed back twice? Any shared skill counts, so adding a skill to a
        blocked part does not get around the block."""
        req = self.current_request(tl)
        return any(r == req and part & set(skills) for r, part in self.reclone_block.get(tl, ()))

    async def _commit_joining(self, src: str, queue: list[QueuedJoin]) -> tuple[list[QueuedJoin], bool]:
        """src is `joining` and `queue` is what the machine handed over, in order: apply each
        join to src. Each graft becomes `joined` and leaves the queue only after its join was
        written. If one cannot be applied, it and those behind it stay `queued` with their
        reason and trigger, the user is told, and they are retried at the next commit point.
        src always leaves `joining`: through join_commit if every join was committed, else
        through join_rollback. Returns (the joins applied, whether that was all of them)."""
        applied: list[QueuedJoin] = []
        outcomes: list[bool] = []
        complete = False
        try:
            for join in queue:
                committed = self.default_join(src, join.child, join.reason, trigger=join.trigger)
                await self.machines.commit_queued(join.child, committed=committed)
                self._note_arrival(src, join)
                applied.append(join)
                outcomes.append(committed)
            complete = True
        except Exception as e:
            self._report_failure(kind="commit", src=src, clones=[j.child for j in queue[len(applied):]], error=e)
        finally:
            await self.machines.finish_join(src, committed=complete and all(outcomes))
            self._mirror()
        return applied, complete

    async def _drain(self, src: str) -> tuple[list[QueuedJoin], bool] | None:
        """Caller holds the conversation lock. Ask the machine to start applying the joins
        waiting on src. None: it refused (src is mid-turn or suspended, or nothing is queued).
        Otherwise what `_commit_joining` returns."""
        queue = await self.machines.try_begin_drain(src)
        if queue is None:
            return None
        self._mirror()
        return await self._commit_joining(src, queue)

    async def _drain_waiting(self) -> None:
        """Caller holds the conversation lock. Offer a drain to every source with queued joins;
        the machine refuses those that are suspended or mid-turn."""
        for src in self.machines.waiting():
            await self.drain_and_wake(src)

    async def drain_and_wake(self, src: str) -> None:
        """Caller holds the conversation lock. While the machine accepts a drain for src: apply
        its queued joins and run it once for the batch. Nothing raised by that run leaves here:
        the joins stay committed and the user is told."""
        while True:
            drained = await self._drain(src)
            if drained is None:
                return
            applied, complete = drained
            if applied:
                await self._after_commit(src, applied, contain=True)
            if not complete:
                return

    def _ends_on_user_message(self, tl: str) -> bool:
        rows = self.store.working_view(tl)
        return bool(rows) and rows[-1]["role"] == "user" and seeder_mod.is_candidate(rows[-1])

    async def _after_commit(self, src: str, applied: list[QueuedJoin], contain: bool) -> None:
        """Joins were applied to src and no user message is waiting: decide whether src runs.
        It runs if a focus graft handed its request back or failed (the user must be answered,
        whether or not the graft had already spoken), or if background grafts joined (their work
        is news). A focus graft's `done` join alone needs nothing: the user saw that
        conversation. If a handed-back user message is now src's last row, src answers it in a
        turn; otherwise src is woken with a closing harness note. `contain`: this is a retry or
        an idle commit point, with no caller to hand an error to, so a failure of the turn is
        reported like a failed wake; a focus graft joining inside a user's turn passes False and
        the error reaches the user's call as that turn's own failure."""
        news = [(j.child, j.reason) for j in applied if self.is_background(j.child) or j.reason != "done"]
        if not news:
            return
        if not self._ends_on_user_message(src):
            await self._wake(src, news)
        elif not contain:
            await self.run(src)
        else:
            try:
                await self.run(src)
            except Exception as e:
                self._report_failure(kind="wake", src=src, clones=[c for c, _ in news], error=e)

    def still_running(self, src: str) -> str:
        """Harness-generated sentence for join records and wake notes: which of src's grafts are
        still running, with their skills so the source can tell which part each is doing."""
        live = []
        for c in self.machines.background(src):
            skills = [k["name"] for k in self.store.skills_for(c) if k["name"] not in BUILTIN]
            live.append(f"{c} (skills: {', '.join(skills)})")
        return ("Still running: " + ", ".join(live) + "." if live
                else "No other clone of yours is still running.")

    async def _wake(self, src: str, joined: list[tuple[str, str]]) -> None:
        """Joins were applied and src's view does not end on a user message: close it with one
        harness note (a user row, so src can continue) and run src once for all of them. If a
        request was handed back and src has wake clone calls left, the run is a turn (it may
        route the request); otherwise it is a report, in which the machine refuses a clone. A
        failure of this run never propagates: the joins stay committed and the user is told."""
        names = ", ".join(f"{c} ({r})" for c, r in joined)
        dropped = [c for c, _ in joined if c in self.join_dropped]
        kept = [c for c, _ in joined if c not in self.join_dropped]
        where = "what they did is in the rows above." if kept else ""
        if dropped:
            where = (where + " " if where else "") + (
                f"The result of {', '.join(dropped)} did not fit in your context and stays in the frozen "
                f"timeline of {'that clone' if len(dropped) == 1 else 'those clones'}; say so if asked.")
        # A focus graft that hands back or fails leaves the user waiting on this run: it is a turn,
        # as it is when the handed-back message is the last row. Background hand-backs are routed
        # only while the source has a wake clone call left; otherwise the run only reports.
        focus_back = any(r != "done" and not self.is_background(c) for c, r in joined)
        routing = focus_back or (any(r == "out_of_scope" for _, r in joined)
                                 and self.wake_clone_calls.get(src, 0) < MAX_WAKE_CLONE_CALLS)
        task = ("A request came back: answer the user, and route what is still to do unless a "
                "still-running clone is already doing it. " if routing else "Do not clone now. ")
        self.store.add_turn(src, "user", "text", harness=True, content=(
                            f"(harness note: {self._names([c for c, _ in joined], named=False)} {names} joined; "
                            f"{where} {self.still_running(src)} You hold the conversation: tell the user, "
                            f"in your own words, only the outcome of the clones that have joined. {task}"
                            "Do not describe still-running work as finished.)"))
        self.wakes.append({"source": src, "joined": joined, "report": not routing})
        self.trace(f"[{src}] WAKE ({'route' if routing else 'report'}) for joins: {names}")
        try:
            if routing:
                await self.machines.begin_turn(src)
            else:
                await self.machines.begin_report(src)
            self._mirror()
            await self._turn(src, wake=not focus_back, report=not routing)
        except Exception as e:
            self._report_failure(kind="wake", src=src, clones=[c for c, _ in joined], error=e)

    # ------------------------------------------------------------ join event
    def replied(self, tl: str) -> bool:
        """Has tl produced user-facing assistant text since the last user message in its working
        view? A harness note is not a user message."""
        answered = False
        for r in self.store.working_view(tl):
            if r["role"] == "user" and seeder_mod.is_candidate(r):
                answered = False
            elif r["role"] == "assistant" and r["kind"] == "text":
                answered = True
        return answered

    def current_request(self, tl: str) -> int | None:
        """The request tl is answering: the last user message in its working view (harness notes
        are not requests)."""
        users = [r["row_id"] for r in self.store.working_view(tl)
                 if r["role"] == "user" and seeder_mod.is_candidate(r)]
        return users[-1] if users else None

    def clone_blocked(self, tl: str, attempts: int, report: bool = False) -> str | None:
        """Why a whole clone call must not run, or None: tl is only reporting (the machine would
        refuse the clone too); the clone calls for this user message are used up (a cap that
        depends on no key); tl's current request has been handed back too many times; or this
        run's clone attempts are used up. Re-cloning one part that failed or bounced is refused
        per entry, in `clone_call` (`part_blocked`). Without these bounds a source whose clone
        keeps failing or handing the same request back re-clones it forever."""
        if report:
            return ("you were woken to report background clones that just finished. Do not clone "
                    "now: tell the user the outcome, and wait for their next message.")
        if self.clone_calls_this_message >= MAX_CLONE_CALLS_PER_MESSAGE:
            return (f"the {MAX_CLONE_CALLS_PER_MESSAGE} clone calls for this user message are used up. Do not "
                    "clone again: answer the user yourself, briefly saying what could not be done.")
        if self.request_handbacks.get((tl, self.current_request(tl)), 0) >= MAX_REQUEST_HANDBACKS:
            return ("clones for this message have failed or handed it back too many times. Do not clone "
                    "for it again: answer the user yourself, briefly saying what could not be done.")
        if attempts >= CLONE_ATTEMPTS:
            return "no clone calls left for this message; answer the user instead."
        return None

    def compute_trigger(self, child: str, reason: str) -> int | None:
        """The user message child did not handle, if any: the last user message in its working
        view, when it is handing it back or never answered it. A harness note is never a user
        message, so it is never a trigger."""
        rows = self.store.working_view(child)
        last_user = None
        answered = False
        for r in rows:
            if r["role"] == "user" and seeder_mod.is_candidate(r):
                last_user, answered = r["row_id"], False
            elif r["role"] == "assistant" and r["kind"] == "text":
                answered = True
        if last_user is None:
            return None
        if reason == "out_of_scope" or not answered:
            return last_user
        return None

    def request_of(self, join: QueuedJoin) -> int | None:
        """The request a joining graft was working on, as the harness knows it. If the graft hands
        back a user message it received after it started (a row of its own, not a harness note),
        that message. Otherwise the source's current request when the graft was cloned, recorded
        then. Never a row the model chose to seed: a trigger that is merely the last user row of
        a model-written seed says nothing about which request this is."""
        if join.trigger is not None:
            row = self.store.turn(join.trigger)
            if row is not None and row["owner_timeline"] == join.child and not row["harness"]:
                return join.trigger
        return self.request_at_clone.get(join.child)

    async def do_join(self, child: str, reason: str) -> None:
        """Raise child's join event from outside its own loop (tests, the replay experiment).
        TransitionRefused, with nothing written, if the machine refuses it: child is the
        original, or has already raised its join event."""
        if self.is_background(child):
            if not await self._join_background(child, reason=_reason(reason)):
                raise TransitionRefused(f"{child} has already raised its join event")
        else:
            await self._join_focus(child, reason=_reason(reason))

    async def _join_focus(self, child: str, reason: str) -> None:
        """A focus graft's join: deterministic, no model call. The machine is asked first; it
        puts the join, with its reason and trigger, last on the source's queue, behind any
        background joins that were waiting, so the trigger stays the last row. The graft is
        `queued` until its join has been written. If the write fails the join stays queued, the
        source takes the conversation back, the user is told, and the next commit point retries
        it. TransitionRefused, with nothing written, if the machine refuses the join event."""
        trigger = self.compute_trigger(child, reason)
        accepted = await self.machines.join_focus(child, reason=reason, trigger=trigger)
        if accepted is None:
            raise TransitionRefused(f"{child} cannot join: it is the original, or has already "
                                    "raised its join event")
        src, queue = accepted
        self.store.set_join(child, reason, trigger=trigger)
        self._mirror()
        self.trace(f"JOIN EVENT {{clone: {child}, reason: {reason}, trigger: {trigger}}} -> {src} joining")
        applied, _ = await self._commit_joining(src, queue)
        if applied:
            await self._after_commit(src, applied, contain=False)

    # ------------------------------------------------------------ default join (deterministic)
    def default_join_rows(self, src: str, child: str, reason: str, trigger: int | None
                          ) -> tuple[list[int], str | None]:
        """Spec 'Default join': text rows (user + assistant) of the clone's working view that the
        source lacks; out_of_scope ends with the trigger; failed adds a harness error record.
        Returns (row ids to append, error record text or None)."""
        have = self.store.pointed(src)
        rows = [r["row_id"] for r in self.store.working_view(child)
                if r["kind"] == "text" and r["row_id"] not in have and r["row_id"] != trigger]
        error = (f"(harness note: clone {child} failed; its timeline is frozen)"
                 if reason == "failed" else None)
        if trigger is not None and trigger not in have:
            rows.append(trigger)
        return rows, error

    def _join_tail(self, src: str, record: str, rows: list[int], error: str | None,
                   trigger: int | None) -> list[dict]:
        """The rows a join appends to src, in order: record, remembered rows, error, trigger."""
        tail = [self._virtual(src, "user", "text", record)]
        tail += [self.store.turn(r) for r in rows if trigger is None or r != trigger]
        if error:
            tail.append(self._virtual(src, "user", "text", error))
        if trigger is not None and trigger in rows:
            tail.append(self.store.turn(trigger))
        return tail

    def fit_join(self, src: str, child: str, reason: str, *, rows: list[int], error: str | None,
                 trigger: int | None) -> tuple[list[int], str]:
        """Join size check (spec: Open Questions, join size). The joined working view of src must
        stay under JOIN_WINDOW_FRACTION of the window: compaction cannot rescue an over-window
        view, since its own request carries the whole prefix. Tiers, first that fits: (0) every
        remembered row; (1) the clone's last assistant reply + trigger; (2) trigger only. The
        dropped rows stay in the clone's frozen timeline. Returns (rows to append, record)."""
        listed = ", ".join(map(str, rows)) or "none"
        live = self.still_running(src)
        full = (rows, f"(harness note: clone {child} joined, reason={reason}; remembered rows {listed} "
                      f"follow. {live})")
        if self.window_tokens is None:
            return full
        limit = int(self.window_tokens * JOIN_WINDOW_FRACTION)
        base = self.store.working_view(src)
        trig = [trigger] if trigger is not None and trigger in rows else []
        last = next((r for r in reversed(rows) if r != trigger and self.store.turn(r)["role"] == "assistant"), None)
        frozen = (f"(harness note: clone {child} joined, reason={reason}; its rows {listed} are too large "
                  f"for this context, so only {{kept}} follow; the rest stay in {child}'s frozen timeline. "
                  f"{live})")
        tiers = [full]
        if last is not None:
            tiers.append(([last] + trig, frozen.format(kept=f"its last reply (row {last})"
                                                       + (" and the trigger" if trig else ""))))
        tiers.append((trig, frozen.format(kept="the trigger" if trig else "this note")))
        before = compaction.estimate_tokens(render(base + self._join_tail(src, full[1], rows, error, trigger)))
        for tier, (kept, record) in enumerate(tiers):
            size = compaction.estimate_tokens(render(base + self._join_tail(src, record, kept, error, trigger)))
            if size <= limit or tier == len(tiers) - 1:
                if tier:
                    self.join_fallbacks.append({"clone": child, "tier": tier, "before": before, "after": size})
                    self.trace(f"[{src}] join of {child} is about {before} tokens, over {limit}: "
                               f"tier {tier} keeps rows {kept} ({size} tokens)")
                return kept, record
        raise AssertionError("unreachable")

    def default_join(self, src: str, child: str, reason: str, trigger: int | None) -> bool:
        """Apply the default join of child to src, once. True: committed. False: the resulting
        view was invalid and only the record (and trigger) were appended (rolled back). If the
        write fails, nothing of the join is in the store, its audit record says `failed`, and
        the error is raised."""
        applied = self.store.join_outcome(child)
        if applied is not None:
            # The join is already in the store: an earlier attempt wrote it and then failed before
            # the machine recorded it. The store commit and the machine step are one unit, so the
            # retry finishes the machine step and writes nothing.
            return applied == "committed"
        wanted, error = self.default_join_rows(src, child, reason, trigger)
        rows, record = self.fit_join(src, child, reason, rows=wanted, error=error, trigger=trigger)
        if [r for r in wanted if r != trigger] and not [r for r in rows if r != trigger]:
            self.join_dropped.add(child)  # the size backstop kept none of its rows
        pending = self.store.view(src) + self._join_tail(src, record, rows, error, trigger)
        trig = trigger if trigger is not None and trigger not in self.store.pointed(src) else None
        v = validate(pending, src, trigger=trig, require_user_end=trig is not None)
        tx = self.store.begin_tx(src, "join", target=child)
        sql = f"DEFAULT JOIN {child} reason={reason} rows={rows}"
        try:
            if v:
                # Cannot happen for well-formed text rows; keep the source continuable anyway.
                self.trace(f"[{src}] default join invalid: {v}; appending record + trigger only")
                self.store.commit_join(src, record.replace("follow", "skipped (invalid view)"), [], error=None,
                                       trigger=trig, tx=tx, status="rolled_back", committed_sql=None,
                                       scratch={"default": True, "violations": v})
                return False
            # record, pointers, error row, trigger and the audit outcome: one SQL transaction
            self.store.commit_join(src, record, [r for r in rows if r != trigger], error=error, trigger=trig,
                                   tx=tx, status="committed", committed_sql=sql, scratch={"default": True})
        except Exception as e:  # nothing of the join was written; the audit record says so
            self.store.end_tx(tx, status="failed", committed_sql=None,
                              scratch={"default": True, "error": f"{type(e).__name__}: {e}"})
            raise
        self.trace(f"[{src}] {sql}")
        return True

    @staticmethod
    def _virtual(owner: str, role: str, kind: str, content: str) -> dict:
        return {"row_id": None, "owner_timeline": owner, "role": role, "kind": kind,
                "tool_name": None, "content": content, "harness": 1}

    # ------------------------------------------------------------ one-shot clone
    def resolve_seed(self, src: str, quotes: list, row_ids: list) -> tuple[list[int], list[str]]:
        """Seed items -> row ids (pointers). A seed may only point at rows the source itself holds
        in its context: its own rows and what was joined into it. A descendant's other rows (a
        running sibling's work, tool output that never joined) are not reachable through a seed.
        A quote must match exactly one text row in the source's view (exact text first, then
        substring). Order: as given, ids then quotes,
        re-sorted by position in the source's view so the seed reads in conversation order."""
        view = self.store.view(src)
        held = {r["row_id"] for r in view}
        texts = [r for r in view if r["kind"] == "text"]
        out: list[int] = []
        problems: list[str] = []
        for x in row_ids:
            try:
                rid = int(x)
            except (TypeError, ValueError):
                problems.append(f"seed_row_ids item {x!r} is not an integer")
                continue
            if rid not in held:
                problems.append(f"row {rid} is not in your conversation; seed only messages you hold")
            else:
                out.append(rid)
        for q in quotes:
            qs = _NONCE_TAG.sub("", str(q)).strip().strip('"').strip()
            if not qs:
                continue
            exact = [r for r in texts if r["content"].strip() == qs]
            hits = exact or [r for r in texts if qs in r["content"]]
            if len(hits) == 1:
                out.append(hits[0]["row_id"])
            elif not hits:
                problems.append(f"seed quote {qs[:60]!r} matches no message")
            else:
                problems.append(f"seed quote {qs[:60]!r} matches {len(hits)} messages "
                                f"(rows {[r['row_id'] for r in hits]}); quote more or use seed_row_ids")
        pos = {r["row_id"]: r["pos"] for r in view}
        seen: set[int] = set()
        uniq = [r for r in out if not (r in seen or seen.add(r))]
        return sorted(uniq, key=lambda r: pos.get(r, 1 << 30)), problems

    def seed_index(self, src: str) -> str:
        lines = [f"Your messages (row_id role: text):"]
        for r in self.store.view(src):
            if r["kind"] == "text":
                lines.append(f"  {r['row_id']} {r['role']}: {r['content'].replace(chr(10), ' ')[:100]}")
        return "\n".join(lines)

    async def seed_with_actor(self, src: str, purpose: str, model_seed: list[int],
                              model_problems: list[str]) -> tuple[list[int], list[str], dict]:
        """GRAFT_SEEDER=jev: the seed actor picks the seed (graft/seeder.py); the original's own seed
        fields are ignored except as the fallback. Candidates are the text rows of the source's
        working view (harness notes excluded); rows scoring >= the threshold are seeded in view
        order, the trigger (latest user message) always. If the seed would start with an assistant
        row, the nearest earlier user text row is added so the clone's view starts with a user turn.
        On a seeder error: the original's own seed if it resolved, else every candidate row."""
        view = self.store.working_view(src)
        cands = [{"row_id": r["row_id"], "role": r["role"], "text": r["content"]}
                 for r in view if seeder_mod.is_candidate(r)]
        users = [c for c in cands if c["role"] == "user"]
        trigger = users[-1]["row_id"] if users else None
        trig_text = users[-1]["text"] if users else ""
        threshold = seeder_mod.seed_threshold()
        d = await self.seeder.score(purpose, cands, trig_text) if cands else seeder_mod.SeedDecision(error="no candidates")
        record = {"seeder": getattr(self.seeder, "name", "jev"), "model": d.model, "threshold": threshold,
                  "scores": {str(k): round(v, 4) for k, v in d.scores.items()}, "latency": d.latency,
                  "input_tokens": d.input_tokens, "requests": d.requests, "truncated": d.truncated,
                  "error": d.error, "candidates": [c["row_id"] for c in cands]}
        problems: list[str] = []
        if d.error:
            if model_seed and not model_problems:
                seed, record["fallback"] = list(model_seed), "model_seed"
            else:
                seed, record["fallback"] = [c["row_id"] for c in cands], "all_candidates"
        else:
            chosen = {rid for rid, p in d.scores.items() if p >= threshold}
            if trigger is not None:
                chosen.add(trigger)
            order = [c["row_id"] for c in cands]
            seed = [rid for rid in order if rid in chosen]
            if seed and self.store.turn(seed[0])["role"] != "user":
                first = order.index(seed[0])
                lead = next((c["row_id"] for c in reversed(cands[:first]) if c["role"] == "user"), None)
                if lead is not None:
                    seed.insert(0, lead)
                    record["added_lead_user_row"] = lead
            record["fallback"] = None
        record["selected"] = seed
        self.seeder_log.append(record)
        self.trace(f"[{src}] seeder {record['seeder']}: {len(seed)}/{len(cands)} rows, "
                   f"{d.latency:.2f}s{', fallback ' + record['fallback'] if record['fallback'] else ''}")
        return seed, problems, record

    async def clone_call(self, src: str, use: dict, retry_left: bool
                         ) -> tuple[list[str] | None, str | None, str]:
        """One clone call, one atomic batch: resolve, seed, validate and commit every entry, or
        refuse them all with the violations per entry. No model call of its own (the seeder actor
        runs once per entry, concurrently). Returns (new timeline ids in entry order, the focus
        graft's id or None, result text), or (None, None, refusal). Background grafts are created
        in `background` but not started; the caller starts them (`_start_background`).

        The input is model-written and anything can fail on it. Whatever is raised in here up to
        and including the commit (the seeder, a machine refusal, the SQL transaction) ends as a
        refusal with every transaction `rolled_back`: the caller always gets a tool result, and
        no clone exists in the store or in the state machine."""
        txs: list[int] = []
        state = {"committed": False}
        try:
            return await self._clone_call(src, use, retry_left=retry_left, txs=txs, state=state)
        except Exception as e:
            if state["committed"]:
                raise  # the grafts exist in the store: not a refusal, a harness bug
            err = f"{type(e).__name__}: {e}"
            if not txs:
                txs.append(self.store.begin_tx(src, "clone"))
            for tx in txs:
                if self.store.tx_status(tx) == "open":
                    self.store.end_tx(tx, status="rolled_back", committed_sql=None,
                                      scratch={"args": use.get("input"), "error": err})
            self.trace(f"[{src}] clone refused: internal error: {err}")
            tail = "Call clone once more." if retry_left \
                else "No retries left for this message; answer the user instead."
            return None, None, f"clone refused; no clone was created: internal error ({err}).\n{tail}"

    async def _clone_call(self, src: str, use: dict, *, retry_left: bool, txs: list[int], state: dict
                          ) -> tuple[list[str] | None, str | None, str]:
        args = use["input"]
        entries, shape = clone_entries(args)
        ids = self.store.next_clone_ids(len(entries))  # distinct before anything commits
        txs += [self.store.begin_tx(src, "clone", target=c) for c in ids]
        pairs = [(child, e.focus) for child, e in zip(ids, entries)]
        if not shape:
            refusal = self.machines.clone_refusal(src, pairs)  # depth, the focus rule, the live cap
            if refusal:
                shape.append(f"clone is not possible now: {refusal}; use fewer background entries "
                             "or wait for them to join")
            shape += [f"clone[{e.index}]: a clone with skills {e.skills} for this message already failed or "
                      "handed it back twice. Do not clone for that part again: leave this entry out and "
                      "tell the user, briefly, what could not be done"
                      for e in entries if self.part_blocked(src, e.skills)]
        resolved = [self.resolve_seed(src, e.seed_quotes, e.seed_row_ids) for e in entries]
        decisions: list[dict | None] = [None] * len(entries)
        if self.seeder is not None and not shape:
            try:  # all entries, or none left running: a failure cancels and awaits the others
                async with asyncio.TaskGroup() as group:
                    calls = [group.create_task(self.seed_with_actor(src, e.purpose, seed, problems))
                             for e, (seed, problems) in zip(entries, resolved)]
            except ExceptionGroup as failed:
                raise failed.exceptions[0] from None
            seeded = [c.result() for c in calls]
            resolved = [(seed, problems) for seed, problems, _ in seeded]
            decisions = [d for _, _, d in seeded]
        per_entry: list[list[str]] = []
        for child, e, (seed, problems) in zip(ids, entries, resolved):
            v = problems + check_skill_link(self.store, src, e.skills)
            rows = [t for t in (self.store.turn(r) for r in seed) if t]
            v += validate(rows, child) if rows else ["the seed is empty: seed at least the user message "
                                                     "that asks for the task"]
            per_entry.append(v)
        scratch = []
        for e, (seed, _), v, decision in zip(entries, resolved, per_entry, decisions):
            sc = {"args": e.raw, "seed": seed, "violations": v, "batch": ids, "focus": e.focus}
            if decision is not None:
                sc["seeder"] = decision
            scratch.append(sc)
        violations = shape + [f"clone[{e.index}]: {x}" for e, v in zip(entries, per_entry) for x in v]
        if violations:
            for tx, sc in zip(txs, scratch):
                self.store.end_tx(tx, status="rolled_back", committed_sql=None,
                                  scratch=dict(sc, batch_violations=violations))
            if not txs:  # no usable entry: one audit record of the refused call
                self.store.end_tx(self.store.begin_tx(src, "clone"), status="rolled_back", committed_sql=None,
                                  scratch={"violations": violations})
            self.trace(f"[{src}] clone refused: {violations}")
            tail = ("Fix the entries and call clone once more.\n" + self.seed_index(src)) if retry_left \
                else "No retries left for this message; answer the user instead."
            return None, None, "clone refused; no clone was created:\n- " + "\n- ".join(violations) + "\n" + tail
        grafts = [Graft(child=child, seed=seed, skills=e.skills + ["join" if e.focus else "background"],
                        focus=e.focus)
                  for child, e, (seed, _) in zip(ids, entries, resolved)]
        parts, audits = [], []
        for tx, sc, g in zip(txs, scratch, grafts):
            mode = "FOCUS" if g.focus else "BACKGROUND"
            audits.append((tx, f"CLONE {g.child} {mode} SEED {g.seed} SKILLS {g.skills}", sc))
            parts.append(f"{g.child} ({'focus, active' if g.focus else 'background, running'}; "
                         f"seed rows {g.seed}; skills {g.skills})")
        # The machine transition and the store commit are one unit. The machine goes first (it may
        # refuse, and then nothing was written); if the single SQL transaction then fails, the
        # machine step is undone. Nothing awaits in between, so no one observes the gap.
        await self.machines.clone(src, pairs)
        try:
            self.store.commit_clone(src, grafts, audits=audits)
        except BaseException:
            await self.machines.unclone(src, pairs)
            raise
        state["committed"] = True
        request = self.current_request(src)
        for g in grafts:
            self.request_at_clone[g.child] = request  # the request this graft is working on
            if not g.focus:
                self._queued.setdefault(g.child, asyncio.Event())
            self.trace(f"[{src}] COMMIT clone -> {g.child} ({'focus' if g.focus else 'background'}) "
                       f"seed={g.seed} skills={g.skills[:-1]}")
        self._mirror()
        focus = next((g.child for g in grafts if g.focus), None)
        return ids, focus, "cloned -> " + ", ".join(parts)
