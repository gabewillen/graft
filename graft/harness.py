"""The Graft harness: the state machines' host (graft/machine.py).

1. Read a timeline's context view (validated).
2. Resolve its tools through timeline_skills -> skill_tools.
3. Call the model (in a daemon worker thread, so background grafts run concurrently).
4. Insert the new turns once and append their pointers; execute tools.

Who decides what. The state machines decide everything that is coordination: what may happen,
who waits, and in what order. A run of the model loop (`_loop`) is the activity of a machine
state, so leaving that state is what ends it; a user message that arrives mid-run, a join that
waits for its source, a cancel that waits for a graft to stop, `settle` and `close` are all
events and states of the machines. The harness does only I/O for them: model calls, store reads
and writes (the ones that belong to a machine step are made by that step's effect, through the
host methods below), tool execution, emitting text, building prompts. It holds no lock, no task
of its own, and no future except a reply carried in an event's data. It never reads a machine:
what it needs to know about a timeline (its kind, its source's grafts still running) comes in
the host call the machine's activity makes.

Clone is one call, `clone(clone=[{purpose, skills, focus, seed_quotes | seed_row_ids}, ...])`, and
one atomic batch, put to the original's machine as the model made it. The machine decides
whether the call may be tried at all (a run makes at most two: one, and one retry after a
refusal); the harness then resolves every entry's seed to pointers and validates (the activity
of the machine's `preparing` state; on violations the source gets them back); the machine then
decides the call and commits it, all of it or none. At most one entry has focus=true: that
graft takes the conversation and the source is suspended until it joins. The other grafts run in
the background, never speak to the user, and the source keeps the conversation.

Join is deterministic (no model call): the harness appends pointers to the text rows of the
clone's working view that the source lacks. Every join is a queue entry on the original's
machine, applied only when the original holds the conversation and is at rest: before a new user
message (which waits on the machine meanwhile), at the end of a run, when a graft finishes while
the original is idle, and, when a focus graft joins back, behind the background joins that were
waiting. After joins that bring news and leave no user message to answer, the original is woken
once with a closing harness note: to report (clone refused), or, if a request was handed back, to
route it.

Cancel is the original's alone: `cancel(cancel=[graft ids])`, while it runs. The cancel calls of
one response are one request to its machine, which cancels each graft the graph allows (leaving
`background` ends its run before anything else) and waits until each one's join, reason
`cancelled`, is queued; only then are the calls' real results stored, so an interrupted turn
never records a cancel that ran as not run. In a response that also carries a clone call, the
cancel calls run first and the clone call after them, so a part can be replaced in one response;
any other tool call beside a clone call is answered without being run. Nothing a graft did
outside the conversation is undone. Every other outcome (a malformed call, an unknown id, a focus
graft, a graft that already finished, a caller that is not the original) is a refusal in the
tool result, and nothing is touched.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Callable

from . import callsites, compaction
from . import seeder as seeder_mod
from .db import Graft, Store, render, validate
from .machine import (CANCELLED, MAX_JOIN_ATTEMPTS, Cancel, CancelCall, Clone, Machines, Mode, Prepared,
                      QueuedJoin, RunOutcome, Unwritten)
from .providers import Provider, prefix_nonce, window_tokens_for
from .skills import (BASE_SYSTEM, BUILTIN, MAX_CANCEL_IDS, MAX_CLONE_ENTRIES, World, load_skill,
                     skill_listing)

# Seconds one model call may take before it is abandoned as a mechanical failure (the harness's
# `call_timeout`; None disables it). Without it a hung call keeps a graft live for ever.
MODEL_CALL_TIMEOUT = 300.0
# A join must leave the source's working view under this fraction of the model window:
# estimate_tokens is chars / 4 (rough), and the reply needs room too.
JOIN_WINDOW_FRACTION = 0.85
# What a graft id may look like in a cancel call: the harness's own ids are `c<number>`. Anything
# else is malformed input and is never looked up or echoed in full.
GRAFT_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,15}")
ECHO_CHARS = 24  # characters of a malformed model-written value shown back, as a repr


class NotRun(str):
    """The text of a result the harness gives a tool call it did not run. `_store_results` marks
    such a result in the stored block (`not_run`), and that mark, never the wording, is what
    says a call was not executed: a tool whose own output happens to read the same is not
    mistaken for one. The mark is not rendered to the model."""


class Ran(str):
    """The result of a tool call the harness executed (a capability tool or load_skill).
    `_store_results` marks it in the stored block (`ran`); the cancel record lists only calls so
    marked, so no pairing by tool-use id is needed. The mark is not rendered to the model."""


NOT_RUN_INTERRUPTED = NotRun("not executed: the turn was interrupted")
NOT_RUN_JOINING = NotRun("not executed: this task is being joined")
NOT_RUN_BESIDE_CLONE = NotRun("not executed: call clone on its own")
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


def _echo(value: object) -> str:
    """A model-written value for a tool result or the trace: a string as a repr, cut short, and
    anything else by its type only (its repr could be of any size or depth)."""
    if not isinstance(value, str):
        return f"a {type(value).__name__}"
    return repr(value) if len(value) <= ECHO_CHARS else repr(value[:ECHO_CHARS]) + "..."


def cancel_ids(args: object) -> tuple[list[str], list[str]]:
    """The graft ids of a cancel call's input, in order and without repeats, and what is wrong
    with the rest. `cancel` is a list of one to MAX_CANCEL_IDS ids, each shaped like a graft id
    (GRAFT_ID). The input is model-written: nothing about its shape is assumed. A call that is
    not such a list names no graft at all; an item that is not shaped like an id is reported by
    its index, shown as a short repr, and the others stand."""
    raw = args.get("cancel") if isinstance(args, dict) else None
    if not isinstance(raw, list) or not raw:
        return [], ["`cancel` must be a list of one or more clone ids, for example [\"c2\"]"]
    if len(raw) > MAX_CANCEL_IDS:
        return [], [f"a cancel call takes at most {MAX_CANCEL_IDS} clone ids (got {len(raw)})"]
    ids: list[str] = []
    problems: list[str] = []
    for i, x in enumerate(raw):
        if not isinstance(x, str) or not GRAFT_ID.fullmatch(x):
            problems.append(f"cancel[{i}]: not cancelled: {_echo(x)} is not a clone id (ids look like \"c2\")")
        elif x not in ids:
            ids.append(x)
    return ids, problems


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


class ModelCallTimeout(TimeoutError):
    """A model call did not return within the harness's `call_timeout`."""


def _reason(reason: str) -> str:
    return reason if reason in ("done", "out_of_scope", "failed") else "failed"



@dataclass
class CloneBatch:
    """The harness's record of a clone call the original's machine is to decide: the grafts to
    commit, their audit records, the result text, and whether a wake run made it."""
    grafts: list[Graft]
    audits: list[tuple[int, str, dict]]
    text: str
    txs: list[int]
    retry_left: bool


class Harness:
    def __init__(self, store: Store, provider: Provider, world: World | None = None,
                 trace: Callable[[str], None] | None = None, max_steps: int = 10,
                 persona: str = BASE_SYSTEM, nonce: str | None = None,
                 compact_tokens: int | None = None, compaction_mode: str = "turns", seeder=...,
                 window_tokens: int | None = ..., emit: Callable[[str], None] | None = None,
                 call_timeout: float | None = MODEL_CALL_TIMEOUT):
        self.store = store
        self.persona = persona  # user-provided; identical, verbatim, on every timeline
        self.provider = provider
        self.world = world or World()
        self.machines = Machines(self)
        self.trace = trace or (lambda s: None)
        self.max_steps = max_steps
        # Seconds a model call may take (None: no limit). A call that exceeds it raises
        # ModelCallTimeout, which is handled like any error of the run it belongs to.
        if call_timeout is not None and call_timeout <= 0:
            raise ValueError(f"call_timeout must be positive or None, got {call_timeout!r}")
        self.call_timeout = call_timeout
        self.silent_done_refusals: list[str] = []  # clones whose join(done) was refused
        # Seed actor (graft/seeder.py): None = the original names the seed itself (GRAFT_SEEDER=model)
        self.seeder = seeder_mod.make_seeder() if seeder is ... else seeder
        self.seeder_log: list[dict] = []  # per clone: Jev latency, tokens, scores, selection, fallback
        self.out: list[str] = []  # user-facing texts: only the conversation holder's
        self.emit = emit or (lambda s: None)  # called with each user-facing text as it is said
        self._mark = 0  # self.out[:_mark] was already returned by user_says / settle
        # per wake run: {source, joined: [(clone, reason)], report: True if clone was not allowed}
        self.wakes: list[dict] = []
        self.join_dropped: set[str] = set()  # clones none of whose rows fit in the source (size backstop)
        # Records of what failed, for the experiments and tests (the session machine keeps the
        # errors `settle` raises): (source, clones, error) per failed wake run, failed join commit,
        # and other harness failure; (clone, error) per graft whose run raised and joined failed.
        self.wake_errors: list[tuple[str, list[str], str]] = []
        self.commit_errors: list[tuple[str, list[str], str]] = []
        self.task_errors: list[tuple[str, list[str], str]] = []
        self.graft_errors: list[tuple[str, str]] = []
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

        if self.call_timeout is None:
            return await in_daemon_thread(call)
        try:
            return await asyncio.wait_for(in_daemon_thread(call), timeout=self.call_timeout)
        except TimeoutError:  # the thread is a daemon: its late result, if any, is dropped
            raise ModelCallTimeout(f"the model call for {tag} did not return within "
                                   f"{self.call_timeout:g}s") from None

    def role(self, tl: str) -> str:
        return "original" if self.store.timeline(tl)["source_id"] is None else f"clone:{tl}"


    # ------------------------------------------------------------ setup
    async def start(self, root: str = "o") -> None:
        self.store.create_timeline(root, None, ["clone"])
        await self.machines.start_root(root)

    def _report_failure(self, *, kind: str, src: str | None, clones: list[str], error: BaseException,
                        quiet: bool, background: bool) -> None:
        """The one place a failure is recorded and shown (it is user-facing output and goes
        through `emit`). The errors `settle` raises are the session machine's; this is the line
        the user sees at once, and the record kept for the experiments. `background`: the clones
        are all background grafts (the machine says so: it made them)."""
        err = f"{type(error).__name__}: {error}"
        record = (src, clones, err)
        names = self._names(clones=clones, background=background)
        if kind == "wake":
            self.wake_errors.append(record)
            line = f"(harness error: {names} joined, but reporting the result failed: {err})"
            if not clones:
                line = f"(harness error: a wake run failed: {err})"
        elif kind == "commit":
            self.commit_errors.append(record)
            line = (f"(harness error: committing the join of {names} failed: {err}; "
                    "the result is kept and the join will be retried)")
        elif kind == "abandon":
            line = (f"(harness error: the join of {names} could not be applied after {MAX_JOIN_ATTEMPTS} attempts "
                    f"({err}); it is abandoned. Its work stays in its frozen timeline and is not part of the "
                    "conversation's memory.)")
        elif kind == "cancel":
            self.task_errors.append(record)
            line = (f"(harness error: {names} was cancelled and stopped, but recording that failed: {err}; "
                    "its join is queued and will still be applied)")
        elif kind == "cancel_result":
            self.task_errors.append(record)
            what = (f"{names} was cancelled and stopped by that call" if clones
                    else "no clone was stopped by that call")
            line = f"(harness error: writing the result of a cancel call failed: {err}; {what})"
        else:  # task, machine, mirror
            self.task_errors.append(record)
            line = (f"(harness error: {names} stopped with an error: {err})" if clones
                    else f"(harness error: {err})")
        if quiet:  # a repeat of something the user was already told
            line = None
        if line is not None:
            self.out.append(line)
        # The caller's callbacks: each is called whatever the other raised (the line is kept in
        # `out` either way); what they raise is the scribe's, which keeps it for `settle`.
        failed: list[Exception] = []
        for callback, text in ((self.trace, f"[{src}] {kind.upper()} FAILED for {clones}: {err}"),
                               (self.emit, line)):
            if text is None:
                continue
            try:
                callback(text)
            except Exception as e:  # noqa: BLE001 - raised below, after the other callback ran
                failed.append(e)
        if failed:
            raise failed[0]

    # ------------------------------------------------------------ entry points
    async def user_says(self, text: str) -> list[str]:
        """One user message, put to the original's machine: it answers it, forwards it to its
        focus graft, or keeps it until the run in progress and the joins waiting are done. Returns
        the user-facing texts emitted since the last return, once the conversation is at rest. If
        the call is cancelled the run answering it is interrupted. A wake or join-commit failure
        after the turn is part of that output (a harness error line), not an exception."""
        root = self.machines.root
        reply = self.machines.future()  # the reply, and the request's identity
        await self.machines.send(tl=root, name="user_message", data=(text, reply))
        try:
            await reply
        except asyncio.CancelledError:
            await self.machines.send(tl=root, name="interrupt", data=reply)  # this request's run, or its place in line
            raise
        return self._take()

    async def settle(self) -> list[str]:
        """Wait until nothing is in flight: no background graft running, no join waiting that can
        be applied (joins queued on a suspended original wait for its focus graft), no wake run.
        Returns the user-facing texts emitted since the last return. Raises what failed meanwhile
        (in background grafts, wakes, join commits): the one error, or an ExceptionGroup of all of
        them; each is raised once."""
        reply = self.machines.future()
        await self.machines.send(tl="session", name="settle_request", data=reply)
        try:
            await reply
        except asyncio.CancelledError:  # the caller went away: the session stops waiting to answer it
            await self.machines.send(tl="session", name="settle_withdrawn", data=reply)
            raise
        return self._take()

    async def close(self) -> None:
        """Shut down: the session machine shuts every running background graft down (its run ends,
        and it joins `failed`), the original applies the joins it can without running anything,
        and closes; every later request is refused by the closed machines. No model call is
        started here; a second `close` is answered when the first is done."""
        await self.machines.ask(tl="session", name="shutdown", build=lambda reply: reply)

    async def run(self, tl: str) -> str:
        """One run of tl, which must be at rest: a turn on what its view ends with. Returns how it
        ended: "ended" (it is at rest again), "handed_off" (it cloned a focus graft) or "joined"."""
        return await self.machines.ask(tl=tl, name="turn_request", build=lambda reply: (Mode(name="turn", user=True), reply))

    async def until_rest(self, tl: str) -> str:
        """Wait until tl is no longer running; answered like `run`."""
        return await self.machines.ask(tl=tl, name="rest_request", build=lambda reply: reply)

    async def begin_turn(self, tl: str) -> None:
        """A turn of tl driven by the caller (tests, the replay experiment): tl is busy, and may
        clone, until `end_turn`."""
        await self.machines.ask(tl=tl, name="turn_request", build=lambda reply: (Mode(name="external"), reply))

    async def end_turn(self, tl: str) -> None:
        """End a turn begun with `begin_turn` (refused, naming the state, if tl is not in one: a
        model-loop run is never ended this way)."""
        await self.machines.ask(tl=tl, name="driven_done", build=lambda reply: reply)

    async def drain_and_wake(self, src: str) -> None:
        """Ask src to apply the joins it has queued now (it does only from rest), and wait until
        it is at rest again, after the wake those joins set off."""
        await self.machines.ask(tl=src, name="drain_request", build=lambda reply: reply)

    def join_queued(self, child: str):
        """Awaitable: background graft `child` has raised its join event (it is queued)."""
        return self.machines.ask(tl=child, name="when_queued", build=lambda reply: reply)

    async def do_join(self, child: str, reason: str) -> None:
        """Raise child's join event from outside its own loop (tests, the replay experiment). One
        event, which child's machine handles by its kind: a background graft's run is ended first,
        so nothing runs in a timeline after it is frozen, and this returns once its join is queued;
        a focus graft's join is applied by its source at once (it is suspended), and this returns
        when the source is at rest again. TransitionRefused, with nothing written, if the machine
        refuses: child is the original, or has already raised its join event."""
        join = self.join_of(tl=child, reason=_reason(reason))
        await self.machines.ask(tl=child, name="join", build=lambda reply: (join, reply))

    # ------------------------------------------------------------ host: I/O for the machines
    async def run_busy(self, tl: str, mode: Mode, kind: str) -> RunOutcome:
        return await self._loop(tl=tl, kind=kind, wake=mode.wake, report=mode.report)

    async def run_background(self, tl: str) -> RunOutcome:
        return await self._loop(tl=tl, kind="background")

    def mirror(self, tl: str, status: str) -> None:
        self.store.set_status(tl=tl, status=status)

    def begin_request(self, tl: str, text: str) -> int:
        rid = self.store.add_turn(tl=tl, role="user", kind="text", content=text)
        self.trace(f"[{tl}] user (row {rid}): {text}")
        return rid

    def commit_clone(self, src: str, clone: Clone) -> str:
        """The clone call's store commit: one SQL transaction (made by the original's machine in
        the step that decides the call)."""
        batch: CloneBatch = clone.data
        self.store.commit_clone(source=src, grafts=batch.grafts, audits=batch.audits)
        for g in batch.grafts:
            self.trace(f"[{src}] COMMIT clone -> {g.child} ({'focus' if g.focus else 'background'}) "
                       f"seed={g.seed} skills={g.skills[:-1]}")
        return batch.text

    def store_result(self, tl: str, use: dict, content: str, is_error: bool) -> None:
        self._store_results(tl=tl, results=[(use, content, is_error)])

    def close_tool_uses(self, tl: str, known: dict[str, tuple[str, bool]]) -> None:
        self._close_tool_uses(tl=tl, known=known)

    def record_join(self, child: str, reason: str) -> None:
        self.store.set_join(tl=child, reason=reason, trigger=None)

    def apply_join(self, src: str, join: QueuedJoin, cloned_for: int | None, running: list[str]
                   ) -> tuple[bool, int | None, bool]:
        """One join applied to src (the activity of its machine's `applying` state). `cloned_for`:
        src's request when the graft was cloned; `running`: src's grafts still running (both
        from src's machine). Returns (committed, the request the graft was working on, whether
        src's view now ends on a user message it must answer)."""
        self.trace(f"JOIN {{clone: {join.child}, reason: {join.reason}, trigger: {join.trigger}}} -> {src}")
        self.store.set_join(tl=join.child, reason=join.reason, trigger=join.trigger)
        committed = self.default_join(src=src, child=join.child, reason=join.reason, trigger=join.trigger,
                                      running=running)
        return committed, self.request_of(join=join, at_clone=cloned_for), self._ends_on_user_message(src)

    def report(self, kind: str, src: str | None, clones: list[str], error: BaseException, quiet: bool,
               background: bool) -> None:
        self._report_failure(kind=kind, src=src, clones=clones, error=error, quiet=quiet, background=background)

    def graft_failed(self, tl: str, error: BaseException) -> None:
        self.graft_errors.append((tl, f"{type(error).__name__}: {error}"))
        self.trace(f"[{tl}] run failed -> join failed: {type(error).__name__}: {error}")

    def abandon_clone(self, src: str, grafts: list[str]) -> None:
        """A committed clone call whose machines could not all start: its grafts are marked
        abandoned (they exist in the store, but never run and never join)."""
        for g in grafts:
            self.store.set_status(tl=g, status="abandoned")
        self.trace(f"[{src}] clone undone: {grafts} abandoned")

    def join_of(self, tl: str, reason: str) -> QueuedJoin:
        return QueuedJoin(child=tl, reason=reason, trigger=self.compute_trigger(child=tl, reason=reason))

    def cancel_text(self, call: CancelCall, outcomes: dict[str, str]) -> tuple[str, bool]:
        """A cancel call's result: what became of each id (cancelled, already finished, refused),
        and what was wrong with the rest of its input. An error only if nothing was cancelled."""
        stopped = [c for c in call.ids if outcomes.get(c) == CANCELLED]
        lines = list(call.problems)
        for c in call.ids:
            outcome = outcomes.get(c)
            if outcome == CANCELLED:
                continue
            if outcome in ("done", "failed", "out_of_scope"):
                outcome = "it had already finished; nothing to cancel"
            lines.append(f"{c}: not cancelled: {outcome}")
        if stopped:
            head = (f"cancelled -> {', '.join(stopped)} (stopped; "
                    f"{'its' if len(stopped) == 1 else 'their'} work so far joins when this turn ends, "
                    "and nothing already done outside this conversation is undone)")
            lines = [head, *lines]
        self.trace(f"cancel {call.ids}: stopped {stopped}")
        return "\n".join(lines) or "nothing to cancel", not stopped

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

    def say(self, tl: str, text: str, background: bool) -> None:
        """Only the branch holding the conversation speaks to the user; a background graft's text
        stays in its timeline."""
        if background:
            self.trace(f"[{tl}] (background, not shown to the user) assistant: {text}")
            return
        self.out.append(text)
        self.emit(text)
        self.trace(f"[{tl}] assistant: {text}")

    def _take(self) -> list[str]:
        texts, self._mark = self.out[self._mark:], len(self.out)
        return texts

    @staticmethod
    def _names(clones: list[str], background: bool, named: bool = True) -> str:
        """'background clones c1, c2' / 'clone c1'; with named=False just the noun. `background`:
        all of them are background grafts."""
        if not clones:
            return "a clone"
        word = "background clone" if background else "clone"
        word += "s" if len(clones) != 1 else ""
        return f"{word} {', '.join(clones)}" if named else word


    # ------------------------------------------------------------ main loop
    def _store_results(self, tl: str, results: list[tuple[dict, str, bool]]) -> None:
        for use, content, err in results:
            block = {"tool_use_id": use["id"], "content": content, "is_error": err}
            if isinstance(content, NotRun):
                block["not_run"] = True  # structural: the harness did not run this call
            if isinstance(content, Ran):
                block["ran"] = True  # structural: the harness ran this call
            self.store.add_turn(tl=tl, role="user", kind="tool_result", content=block, tool_name=use["name"])

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


    def _close_tool_uses(self, tl: str, known: dict[str, tuple[str, bool]] | None = None) -> None:
        """tl's run ended before every tool call of its last assistant turn had a result: give
        each such call one, so the view stays valid for the next model call. `known`: the real
        results of calls that ran but whose result row could not be written (kept by tl's
        machine); every other call gets one saying it was not executed. A clone or cancel call's
        result is stored by the machine step that ran it, before the run can end, so this never
        stands in for a call that ran. Calls of that turn that already have their result are left
        alone, whatever was stored after them."""
        rows = self.store.view(tl)
        last = max((i for i, r in enumerate(rows) if r["role"] == "assistant"), default=None)
        if last is None:
            return
        first = last
        while first > 0 and rows[first - 1]["role"] == "assistant":
            first -= 1
        answered = [json.loads(r["content"]).get("tool_use_id") for r in rows[last + 1:] if r["kind"] == "tool_result"]
        missing = []
        for r in rows[first:last + 1]:
            if r["kind"] != "tool_use":
                continue
            use_id = json.loads(r["content"])["id"]
            if use_id in answered:
                answered.remove(use_id)  # one result answers one call, also when an id repeats
                continue
            content, err = (known or {}).get(use_id, (NOT_RUN_INTERRUPTED, True))
            missing.append(({"id": use_id, "name": r["tool_name"]}, content, err))
        if missing:
            self._store_results(tl=tl, results=missing)

    async def _loop(self, tl: str, kind: str, wake: bool = False, report: bool = False) -> RunOutcome:
        """The model loop for tl: the activity of its machine's `busy` (or `background`) state.
        Returns how it ended, and nothing more: a reply with no tool call, an empty response, a
        join (with its reason), or a spent step budget. What follows is the machine's to decide
        (a background graft joins with a reason, a focus graft that joins hands back, the
        original rests). A clone that hands the conversation to a focus graft ends the run from
        the machine (the state is left). `wake`: the run was started for joins; `report`: it may
        only report (no clone). `kind`: tl's kind, as its machine says; it decides only whether
        tl's text reaches the user."""
        background = kind == "background"
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
                return RunOutcome(kind="empty")
            self._store_blocks(tl, blocks)
            for b in blocks:
                if b["type"] == "text":
                    self.say(tl=tl, text=b["text"], background=background)
            uses = [b for b in blocks if b["type"] == "tool_use"]
            if not uses:
                return RunOutcome(kind="replied")  # a background graft's reply is its result
            allowed = {t["name"] for t in tools}
            join_use = next((u for u in uses if u["name"] == "join" and "join" in allowed), None)
            clone_use = next((u for u in uses if u["name"] == "clone" and "clone" in allowed), None)
            refused_join = None
            if join_use and str(join_use["input"].get("reason")) == "done" and not self.replied(tl):
                refused_join, join_use = join_use, None
                self.silent_done_refusals.append(tl)
                self.trace(f"[{tl}] join(done) refused: no reply since the last user message")
            if join_use:
                results = [(u, NOT_RUN_JOINING, True) for u in uses if u is not join_use]
                results.append((join_use, "joined", False))
                self._store_results(tl, results)
                return RunOutcome(kind="join", reason=_reason(str(join_use["input"].get("reason", "failed"))))
            if clone_use:
                # this call's role was routing into a clone: account it as the clone transaction
                retag = getattr(self.provider, "retag", None)
                if retag:
                    retag(last, f"{self.role(tl)}:clone_tx")
                # Beside a clone call only cancel calls run, and they run first, wherever the model
                # put them: a cancelled graft's place is free, and its part blocked, before the
                # clone call is decided. Every other call is answered without being run.
                cancels = [u for u in uses if u is not clone_use and u["name"] == "cancel"]
                others = [(u, NOT_RUN_BESIDE_CLONE, True) for u in uses
                          if u is not clone_use and u["name"] != "cancel"]
                self._store_results(tl=tl, results=others)
                # one request carries the response's cancels and its clone: the machine runs the
                # cancels first, then decides the clone
                cancel = self._cancel_request(tl=tl, uses=cancels) if cancels else None
                await self.clone_call(src=tl, use=clone_use, store=True, cancels=cancel)
                if cancel is not None:
                    await self._settle_cancel(tl=tl, uses=cancels, reply=cancel.reply)
                continue
            results = []
            cancels = []
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
                elif u["name"] == "cancel":  # whoever calls it, whatever tools it was shown: the machine decides
                    cancels.append(u)
                elif u["name"] not in allowed:
                    results.append((u, NotRun(f"tool '{u['name']}' is not available to you"), True))
                elif u["name"] == "load_skill":
                    content, err = self.load(tl, str(u["input"].get("name", "")))
                    self.trace(f"[{tl}] load_skill({u['input'].get('name')}) -> {'error' if err else 'ok'}")
                    results.append((u, content, err))
                else:
                    content, err = self.world.execute(u["name"], u["input"])
                    self.trace(f"[{tl}] tool {u['name']}({json.dumps(u['input'])[:120]}) -> {content[:80]}")
                    results.append((u, Ran(content), err))
            self._store_results(tl, results)
            # The cancel calls run after every other call's result is stored: one request to the
            # machine, which stores each call's real result once the grafts have stopped.
            if cancels:
                await self._cancels(tl=tl, uses=cancels)
        self.trace(f"[{tl}] step budget exhausted")
        return RunOutcome(kind="exhausted")

    def _cancel_request(self, tl: str, uses: list[dict]) -> Cancel:
        """The cancel calls of one response, as one request for tl's machine. It cancels what the
        graph allows, waits for those grafts' joins to be queued, and stores each call's real
        result. A timeline that may not cancel at all answers with the reason; the results are
        then stored here (they are this run's own tool results)."""
        calls = []
        for u in uses:
            ids, problems = cancel_ids(u.get("input"))
            calls.append(CancelCall(use=u, ids=ids, problems=problems))
        return Cancel(calls=calls, reply=self.machines.future(), request=self.current_request(tl))

    async def _cancels(self, tl: str, uses: list[dict]) -> None:
        cancel = self._cancel_request(tl=tl, uses=uses)
        await self.machines.send(tl=tl, name="cancel_request", data=cancel)
        await self._settle_cancel(tl=tl, uses=uses, reply=cancel.reply)

    async def _settle_cancel(self, tl: str, uses: list[dict], reply: asyncio.Future) -> None:
        answer = await reply
        if isinstance(answer, str):  # refused whole: every id gets the reason; ours to store
            self.trace(f"[{tl}] cancel refused: {answer}")
            results = []
            for u in uses:
                ids, problems = cancel_ids(u.get("input"))
                content, err = self.cancel_text(call=CancelCall(use=u, ids=ids, problems=problems),
                                                outcomes={c: answer for c in ids})
                results.append((u, content, err))
            self._store_results(tl=tl, results=results)

    # ------------------------------------------------------------ cancel record, parts, blocks
    def tools_run(self, child: str) -> list[str]:
        """The capability tools child executed itself, in order: the results the harness marked
        as run (`Ran`) where it executed the tool; the mark is read, never the result's wording,
        and no pairing by tool-use id is needed, so a repeated id hides nothing. A tool that
        returned an error was still executed and may have acted before it failed, so it is
        listed, marked "(returned an error)". Names only: tool input and output never leave a
        graft's timeline."""
        ran = []
        for r in self.store.view(child):
            if r["owner_timeline"] != child or r["kind"] != "tool_result":
                continue
            data = json.loads(r["content"])
            if data.get("ran"):
                ran.append(r["tool_name"] + (" (returned an error)" if data.get("is_error") else ""))
        return ran

    def part_of(self, child: str) -> frozenset[str]:
        """The part a graft was given: its linked catalog skills (harness-known, deterministic)."""
        return frozenset(k["name"] for k in self.store.skills_for(child) if k["name"] not in BUILTIN)

    # ------------------------------------------------------------ after joins
    def _ends_on_user_message(self, tl: str) -> bool:
        rows = self.store.working_view(tl)
        return bool(rows) and rows[-1]["role"] == "user" and seeder_mod.is_candidate(rows[-1])


    def _running_note(self, running: list[str]) -> str:
        """Harness-generated sentence for join records and wake notes: which of the source's grafts
        are still running (`running`, from its machine), with their skills so the source can tell
        which part each is doing."""
        live = []
        for c in running:
            skills = [k["name"] for k in self.store.skills_for(c) if k["name"] not in BUILTIN]
            live.append(f"{c} (skills: {', '.join(skills)})")
        return ("Still running: " + ", ".join(live) + "." if live
                else "No other clone of yours is still running.")


    def wake_note(self, src: str, joined: list[tuple[str, str]], routing: bool, running: list[str],
                  background: bool) -> None:
        """Close src's view with one harness note (a user row, so src can continue) for the joins
        its machine is waking it for (the activity of its `noting` state). Whether the run may
        route a handed-back request or only report is the machine's choice: `routing`. `running`:
        src's grafts still running; `background`: the joined grafts are all background grafts."""
        names = ", ".join(f"{c} ({r})" for c, r in joined)
        dropped = [c for c, _ in joined if c in self.join_dropped]
        kept = [c for c, _ in joined if c not in self.join_dropped]
        where = "what they did is in the rows above." if kept else ""
        if dropped:
            where = (where + " " if where else "") + (
                f"The result of {', '.join(dropped)} did not fit in your context and stays in the frozen "
                f"timeline of {'that clone' if len(dropped) == 1 else 'those clones'}; say so if asked.")
        task = ("A request came back: answer the user, and route what is still to do unless a "
                "still-running clone is already doing it. " if routing else "Do not clone now. ")
        cancelled = [c for c, r in joined if r == CANCELLED]
        if cancelled:
            task += (f"You cancelled {', '.join(cancelled)}: tell the user it was stopped and what it had "
                     "already done, and do not clone for that part again. ")
        self.store.add_turn(tl=src, role="user", kind="text", harness=True, content=(
                            f"(harness note: {self._names(clones=[c for c, _ in joined], background=background, named=False)} "
                            f"{names} joined; {where} {self._running_note(running)} "
                            "You hold the conversation: tell the user, "
                            f"in your own words, only the outcome of the clones that have joined. {task}"
                            "Do not describe still-running work as finished.)"))
        self.wakes.append({"source": src, "joined": joined, "report": not routing})
        self.trace(f"[{src}] WAKE ({'route' if routing else 'report'}) for joins: {names}")

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

    def request_of(self, join: QueuedJoin, at_clone: int | None) -> int | None:
        """The request a joining graft was working on, as the harness knows it. If the graft hands
        back a user message it received after it started (a row of its own, not a harness note),
        that message. Otherwise the source's current request when the graft was cloned, recorded
        then. Never a row the model chose to seed: a trigger that is merely the last user row of
        a model-written seed says nothing about which request this is."""
        if join.trigger is not None:
            row = self.store.turn(join.trigger)
            if row is not None and row["owner_timeline"] == join.child and not row["harness"]:
                return join.trigger
        return at_clone


    # ------------------------------------------------------------ default join (deterministic)
    def default_join_rows(self, src: str, child: str, reason: str, trigger: int | None
                          ) -> tuple[list[int], str | None]:
        """Spec 'Default join': text rows (user + assistant) of the clone's working view that the
        source lacks; out_of_scope ends with the trigger; failed adds a harness error record, and
        cancelled a harness record of the tools the graft had run.
        Returns (row ids to append, error record text or None)."""
        have = self.store.pointed(src)
        rows = [r["row_id"] for r in self.store.working_view(child)
                if r["kind"] == "text" and r["row_id"] not in have and r["row_id"] != trigger]
        error = (f"(harness note: clone {child} failed; its timeline is frozen)"
                 if reason == "failed" else None)
        if reason == CANCELLED:
            ran = self.tools_run(child)
            error = (f"(harness note: clone {child} was cancelled by you before it finished; its timeline is frozen. "
                     f"Before it was stopped it had run: {', '.join(ran) if ran else 'no tools'}. "
                     "Nothing it did outside this conversation was undone; a tool that returned an error "
                     "may still have acted before it failed.)")
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
                 trigger: int | None, running: list[str]) -> tuple[list[int], str]:
        """Join size check (spec: Open Questions, join size). The joined working view of src must
        stay under JOIN_WINDOW_FRACTION of the window: compaction cannot rescue an over-window
        view, since its own request carries the whole prefix. Tiers, first that fits: (0) every
        remembered row; (1) the clone's last assistant reply + trigger; (2) trigger only. The
        dropped rows stay in the clone's frozen timeline. `running`: src's grafts still running, for
        the record. Returns (rows to append, record)."""
        listed = ", ".join(map(str, rows)) or "none"
        live = self._running_note(running)
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

    def default_join(self, src: str, child: str, reason: str, trigger: int | None, running: list[str]) -> bool:
        """Apply the default join of child to src, once. True: committed. False: the resulting
        view was invalid and only the record (and trigger) were appended (rolled back). If the
        write fails, nothing of the join is in the store, its audit record says `failed`, and
        the error is raised. `running`: src's grafts still running, named in the record."""
        applied = self.store.join_outcome(child)
        if applied is not None:
            # The join is already in the store: an earlier attempt wrote it and then failed before
            # the machine recorded it. The store commit and the machine step are one unit, so the
            # retry finishes the machine step and writes nothing.
            return applied == "committed"
        wanted, error = self.default_join_rows(src=src, child=child, reason=reason, trigger=trigger)
        rows, record = self.fit_join(src=src, child=child, reason=reason, rows=wanted, error=error, trigger=trigger,
                                     running=running)
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
    def resolve_seed(self, src: str, *, quotes: list, row_ids: list) -> tuple[list[int], list[str]]:
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


    async def clone_call(self, src: str, use: dict, store: bool = False, cancels: Cancel | None = None
                         ) -> tuple[list[str] | None, str | None, str]:
        """One clone call, put to src's machine as the model made it, with the cancel calls of the
        same response. The machine runs the cancels first, then decides whether the call may be
        tried at all, has it prepared (`prepare_clone`), and decides and commits it, all of it or
        none. Returns (new timeline ids in entry order, the focus graft's id or None, result
        text), or (None, None, refusal). `store`: the machine stores the call's result row, as the
        model loop needs."""
        children, focus, text = await self.machines.ask(tl=src, name="clone_request", build=lambda reply: Clone(
            use=use, reply=reply, store=store, request=self.current_request(src), cancels=cancels))
        if isinstance(text, Unwritten) and store:  # refused by a timeline that is not running: ours to write
            self._store_results(tl=src, results=[(use, str(text), True)])
        return children, focus, text

    async def prepare_clone(self, src: str, clone: Clone, retry_left: bool) -> Prepared | str:
        """The clone call's preparation (the activity of src's machine's `preparing` state): the
        harness resolves, seeds (the seeder actor runs once per entry, concurrently) and validates
        every entry. Returns the prepared call, or the text of its refusal. `retry_left`: the run
        may make one more clone call after this one (the machine counts them).

        The input is model-written and anything can fail on it. Whatever is raised in here (the
        seeder, a store write) ends as a refusal with every transaction `rolled_back`; if the run
        is ended meanwhile, the transactions are closed and the end goes on."""
        txs: list[int] = []
        try:
            prepared = await self._prepare_clone(src=src, use=clone.use, retry_left=retry_left, txs=txs)
        except BaseException as e:  # an error, or the run was ended while the call was being prepared
            err = f"{type(e).__name__}: {e}"
            self._roll_back(src=src, txs=txs, scratch={"args": clone.use.get("input"), "error": err})
            self.trace(f"[{src}] clone refused: internal error: {err}")
            if not isinstance(e, Exception):
                raise  # ended: the transactions are closed; the machine closes the tool call
            tail = "Call clone once more." if retry_left \
                else "No retries left for this message; answer the user instead."
            return f"clone refused; no clone was created: internal error ({err}).\n{tail}"
        if isinstance(prepared, str):
            return prepared
        batch, pairs, parts = prepared
        return Prepared(pairs=pairs, parts=parts, data=batch)

    def _roll_back(self, src: str, txs: list[int], scratch: dict) -> None:
        if not txs:
            txs.append(self.store.begin_tx(tl=src, kind="clone"))
        for tx in txs:
            if self.store.tx_status(tx) == "open":
                self.store.end_tx(tx_id=tx, status="rolled_back", committed_sql=None, scratch=scratch)

    def refuse_clone(self, src: str, clone: Clone, why: str, error: BaseException | None) -> str:
        """The machine refused a prepared clone call (or its commit failed): every transaction is
        rolled back. Returns the result text."""
        batch: CloneBatch = clone.data
        detail = f"internal error ({type(error).__name__}: {error})" if error is not None else why
        self._roll_back(src=src, txs=batch.txs, scratch={"refused": detail})
        self.trace(f"[{src}] clone refused: {detail}")
        tail = "Call clone once more." if batch.retry_left \
            else "No retries left for this message; answer the user instead."
        return f"clone refused; no clone was created: {detail}.\n{tail}"

    async def _prepare_clone(self, src: str, use: dict, *, retry_left: bool, txs: list[int]
                             ) -> str | tuple[CloneBatch, list[tuple[str, bool]], list]:
        args = use["input"]
        entries, shape = clone_entries(args)
        ids = self.store.next_clone_ids(len(entries))  # distinct before anything commits
        txs += [self.store.begin_tx(src, "clone", target=c) for c in ids]
        pairs = [(child, e.focus) for child, e in zip(ids, entries)]
        # depth, the focus rule, the live cap and the re-clone bounds are the machine's guards
        resolved = [self.resolve_seed(src, quotes=e.seed_quotes, row_ids=e.seed_row_ids) for e in entries]
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
            return "clone refused; no clone was created:\n- " + "\n- ".join(violations) + "\n" + tail
        grafts = [Graft(child=child, seed=seed, skills=e.skills + ["join" if e.focus else "background"],
                        focus=e.focus)
                  for child, e, (seed, _) in zip(ids, entries, resolved)]
        parts, audits = [], []
        for tx, sc, g in zip(txs, scratch, grafts):
            mode = "FOCUS" if g.focus else "BACKGROUND"
            audits.append((tx, f"CLONE {g.child} {mode} SEED {g.seed} SKILLS {g.skills}", sc))
            parts.append(f"{g.child} ({'focus, active' if g.focus else 'background, running'}; "
                         f"seed rows {g.seed}; skills {g.skills})")
        batch = CloneBatch(grafts=grafts, audits=audits, text="cloned -> " + ", ".join(parts), txs=list(txs),
                           retry_left=retry_left)
        return batch, pairs, [(e.index, frozenset(e.skills)) for e in entries]
