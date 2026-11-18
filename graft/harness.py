"""The Graft harness: the state machines' host (graft/machine.py).

1. Read a timeline's context view (validated).
2. Resolve its tools through timeline_skills -> skill_tools.
3. Call the model (in a daemon worker thread, so background grafts run concurrently).
4. Insert the new turns once and append their pointers; each host tool call is made by the
   timeline's machine, as the activity of its `calling` state (awaited if `execute` is async).

Who decides what. The state machines decide everything that is coordination: what may happen,
who waits, and in what order. A run of the model loop (`_loop`) is the activity of a machine
state, so leaving that state is what ends it; a user message that arrives mid-run, a join that
waits for its source, a cancel that waits for a graft to stop, `settle` and `close` are all
events and states of the machines. The harness does only I/O for them: model calls, store reads
and writes (the ones that belong to a machine step are made by that step's effect, through the
host methods below), tool execution, emitting text, building prompts. It holds no lock, no task
of its own, and no future except a reply carried in an event's data, whose answer it never
inspects. It never reads a machine: what it needs to know about a timeline (its kind, its
source's grafts still running) comes in the host call the machine's activity makes, and whether a
run may take its next step is the machine's answer to the run's `may_continue` request. A user's
call is a request the original's machine owns (see `user_says`); a cancelled call withdraws it.

Clone is one call, `fork(current={id?, skills}, detached=[{id?, skills}, ...])` as the model is
told it, and one atomic batch, put to the original's machine as the model made it. The machine
decides whether the call may be tried at all (from the original, in a run whose request is live,
while not closing; how many calls a run makes is not bounded); the harness then draws the ids the call left out, seeds every entry (the conversation so
far, or a decisions client's pick of it) and validates (the activity of the machine's `preparing`
state; on violations the source gets them back); the machine then decides the call and commits it,
all of it or none. The `current` graft takes the conversation and the source is suspended until it
joins. The `detached` grafts run in the background, never speak to the user, and the source keeps
the conversation. A graft is known to the model by its id (its name), never by its timeline id.

Join is deterministic (no model call): the harness appends pointers to the text rows of the
clone's working view that the source lacks. Every join is a queue entry on the original's
machine, applied only when the original holds the conversation and is at rest: before a new user
message (which waits on the machine meanwhile), at the end of a run, when a graft finishes while
the original is idle, and, when a focus graft joins back, behind the background joins that were
waiting. After joins that bring news and leave no user message to answer, the original is woken
once with a closing harness note, for a full run: it tells the user the outcome and may go on
(clone, cancel, answer a request that was handed back), as any turn.

Cancel is the original's alone: `stop(ids=[branch ids])` as the model is told it, while it runs. The cancel calls of
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
import inspect
import json
import random
import re
import threading
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

from . import callsites, compaction
from . import seeder as seeder_mod
from .db import Graft, Store, render, validate
from .machine import (CANCELLED, MAX_JOIN_ATTEMPTS, Cancel, CancelCall, Clone, Machines, Mode,
                      Prepared, QueuedJoin, RunOutcome, ToolCall, TransitionRefused, Unwritten, described)
from .providers import Provider, prefix_nonce, window_tokens_for
from .skills import BASE_SYSTEM, BRANCH_ID_PATTERN, BUILTIN, World, load_skill, skill_listing

# Seconds one model call may take before it is abandoned as a mechanical failure (the harness's
# `call_timeout`; None disables it). Without it a hung call keeps a graft live for ever.
MODEL_CALL_TIMEOUT = 300.0
# A join must leave the source's working view under this fraction of the model window:
# estimate_tokens is chars / 4 (rough), and the reply needs room too.
JOIN_WINDOW_FRACTION = 0.85
# What a branch id (a graft's name, as the model knows it) looks like. Anything else is malformed
# input and is never looked up or echoed in full.
GRAFT_ID = re.compile(BRANCH_ID_PATTERN)
# Draws of a random id for one entry before the call is refused: a source that keeps colliding, or
# returns something that is not an id, is an error of the call's preparation, never a loop.
MAX_NAME_DRAWS = 16
# The line the original holds, as the model is told it (git's main): never a branch's id.
MAIN = "main"


# What `draw_names` raises (an error of the call's preparation, so the call is refused): its text
# reaches the model in the refusal.
NOT_A_DRAWN_ID = "the id source returned {got}, which is not a branch id"
NO_FREE_ID = "no free branch id after {draws} draws"


def new_names(rng: random.Random | None = None) -> Callable[[], str]:
    """The harness's default source of branch ids: 6 random lowercase hex characters per draw
    (`rng`: a seeded generator, for tests and experiments)."""
    rng = rng or random.Random()
    return lambda: f"{rng.getrandbits(24):06x}"
ERROR_CHARS = 200  # characters of a tool's own error shown in its result
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


class ToolResult(NamedTuple):
    """One tool call's result as the harness stores it: the model's tool call, the result's
    content (a NotRun or Ran marks it), and whether it is an error."""
    use: dict
    content: str
    is_error: bool


NOT_RUN_INTERRUPTED = NotRun("not executed: the turn was interrupted")
NOT_RUN_JOINING = NotRun("not executed: this branch is merging")
NOT_RUN_BESIDE_CLONE = NotRun("not executed: call fork on its own")
# A host tool call cut short while it ran (its run was left: a cancel, a withdrawal, shutdown).
# It may have acted, or may still act (cancellation is cooperative: work it handed to a thread, or
# shielded, can go on after the cancel), so it is recorded as run, with an error.
CANCELLED_WHILE_RUNNING = Ran("stopped while running: it may have acted, or may still act")
# A host tool call whose tool was cancelled by something other than its graft (the host cancelled a
# future or task the tool awaited, while the graft still ran): it ran, with an error.
CANCELLED_BY_THE_TOOL = Ran("interrupted by the tool: it may have acted")
# A host tool call that failed after the tool was invoked in a way that gave no result of its own.
FAILED_AFTER_INVOKED = Ran("failed after it was invoked: it may have acted")
_NONCE_TAG = re.compile(r"^\[conversation [^\]]*\]\s*")
# Why a seed row is refused when the user withdrew the message it holds (spec: State machine).
WITHDRAWN_SEED = "row {row} is a message the user withdrew; it is not a request and cannot be seeded"


def said(reason: str) -> str:
    """A join reason as the model is told it (the one place it is worded): the harness's own
    `cancelled` is said `stopped`, the model's word for graft's cancel; the others are the
    `merge` tool's own reasons."""
    return "stopped" if reason == CANCELLED else reason


def merge_head(*, child: str, reason: str) -> str:
    """The head of every merge record a source is given (the one place it is worded)."""
    return f"(harness note: your branch {child} merged, reason={said(reason)}; "


def check_skill_link(store: Store, source: str, skills: list[str]) -> list[str]:
    """The one place skill linking is checked. Grafts are one level deep: only the original
    clones, so the `fork` skill (graft's clone skill, as the model is told it) is never linked
    to a graft."""
    v = []
    known = set(store.skill_names())
    if not skills:
        v.append("select at least one skill")
    for s in skills:
        if s not in known:
            v.append(f"unknown skill '{s}'")
        elif s in ("merge", "detached"):
            v.append(f"'{s}' is linked by the harness (merge to a current branch, detached to a "
                     "detached branch); do not select it")
        elif s == "fork":
            v.append("'fork' cannot be held by a branch (you fork only from main)")
    return v


@dataclass
class CloneEntry:
    """One validated entry of a clone call. `label` names it in violations (`current`, or
    `detached[i]` by its index in the model's list); `name` is the id the model gave, or None (one
    is drawn); the skills list is empty when the model's value was not usable. The current entry
    comes first, then the detached ones in order: timeline ids are drawn in that order."""
    label: str
    raw: dict
    name: str | None
    skills: list[str]
    focus: bool


ENTRY_FIELDS = ("id", "skills")


def _strings(entry: dict, *, name: str, label: str, violations: list[str]) -> list[str]:
    value = entry.get(name)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        violations.append(f"{label}: {name} must be a list of strings")
        return []
    return list(value)


def _entry(e: object, label: str, focus: bool, v: list[str]) -> CloneEntry | None:
    if not isinstance(e, dict):
        v.append(f"{label}: must be an object {{id?, skills}}")
        return None
    extra = sorted(str(k) for k in e if k not in ENTRY_FIELDS)
    if extra:
        v.append(f"{label}: unknown field{'s' if len(extra) > 1 else ''} {', '.join(map(_echo, extra))} "
                 "(a branch has only `id` and `skills`)")
    name = e.get("id")
    if name is not None and not (isinstance(name, str) and GRAFT_ID.fullmatch(name)):
        v.append(f"{label}: id {_echo(name)} is not a branch id (lowercase letters, digits and -, "
                 "starting with a letter or digit, at most 32 characters)")
        name = None
    elif name == MAIN:
        v.append(f"{label}: id 'main' is the line you hold, not a branch; give the branch another id")
        name = None
    skills = _strings(e, name="skills", label=label, violations=v)
    repeated = sorted({k for k in skills if skills.count(k) > 1})
    if repeated:
        v.append(f"{label}: skills lists {repeated} more than once")
    return CloneEntry(label=label, raw=e, name=name, skills=skills, focus=focus)


def clone_entries(args: object) -> tuple[list[CloneEntry], list[str]]:
    """The entries of a clone call's input and its shape violations. The input (the model's
    `fork` call) has an optional `current`, one entry, and an optional `detached`, a list of
    entries; at least one entry in all. An entry is {id?, skills}:
    `skills` strings without repeats, `id` a branch id, and nothing else. Two entries with the same
    set of skills need ids. The input is model-written: nothing about its shape is assumed.
    Violations name the entry by its label."""
    if not isinstance(args, dict):
        return [], ["`fork` takes `current` (one branch), `detached` (a list of branches), or both; "
                    "each branch is {id?, skills}"]
    extra = sorted(str(k) for k in args if k not in ("current", "detached"))
    current, detached = args.get("current"), args.get("detached")
    if detached is not None and not isinstance(detached, list):
        return [], ["`detached` must be a list of branches, each {id?, skills}"]
    raw = ([("current", current, True)] if current is not None else []) \
        + [(f"detached[{i}]", e, False) for i, e in enumerate(detached or [])]
    if not raw:
        return [], ["`fork` needs `current`, a non-empty `detached`, or both; each branch is {id?, skills}"]
    v: list[str] = [f"unknown field {_echo(k)} (`fork` takes `current` and `detached`)" for k in extra]
    entries = [e for e in (_entry(e, label, focus, v) for label, e, focus in raw) if e is not None]
    names = [e.name for e in entries if e.name is not None]
    v += [f"id {_echo(n)} is given to more than one branch" for n in sorted(set(names)) if names.count(n) > 1]
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if a.skills and set(a.skills) == set(b.skills) and (a.name is None or b.name is None):
                v.append(f"{a.label} and {b.label} hold the same skills: give each an id to tell them apart")
    return entries, v


def _echo(value: object) -> str:
    """A model-written value for a tool result or the trace: a string as a repr, cut short, and
    anything else by its type only (its repr could be of any size or depth)."""
    if not isinstance(value, str):
        return f"a {type(value).__name__}"
    return repr(value) if len(value) <= ECHO_CHARS else repr(value[:ECHO_CHARS]) + "..."


def _shown(error: BaseException) -> str:
    """A tool's own error, shown after its type: ': message' cut short, or nothing if it has none or
    its message cannot be had (its __str__ raises)."""
    try:
        text = str(error)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 - its message cannot be had: its type alone is shown
        return ""
    return f": {text[:ERROR_CHARS]}" if text else ""


def cancel_ids(args: object) -> tuple[list[str], list[str]]:
    """The branch ids of a cancel call's input (the model's `stop`), in order and without repeats,
    and what is wrong with the rest. `ids` is a list of one or more ids, each shaped like a branch id (GRAFT_ID). They are names, resolved to timeline
    ids by the run (`Harness._cancel_request`). The input is model-written: nothing about its shape is assumed. A call that is
    not such a list names no graft at all; an item that is not shaped like an id is reported by
    its index, shown as a short repr, and the others stand."""
    raw = args.get("ids") if isinstance(args, dict) else None
    if not isinstance(raw, list) or not raw:
        return [], ["`ids` must be a list of one or more branch ids, for example [\"c2\"]"]
    ids: list[str] = []
    problems: list[str] = []
    for i, x in enumerate(raw):
        if not isinstance(x, str) or not GRAFT_ID.fullmatch(x):
            problems.append(f"ids[{i}]: not stopped: {_echo(x)} is not a branch id (ids look like \"site\")")
        elif x not in ids:
            ids.append(x)
    return ids, problems


async def in_daemon_thread(fn: Callable[[], object]) -> object:
    """Run fn in a daemon thread and await its result. A model call must not pin the process: a
    daemon thread is abandoned at exit, where an executor thread would be joined."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def deliver(result: object, error: BaseException | None) -> None:
        if future.done():
            return
        if error is None:
            future.set_result(result)
            return
        try:
            future.set_exception(error)
        except TypeError:  # an error a future cannot carry: carried as one it can
            future.set_exception(RuntimeError(f"in a model call: {described(error)}"))

    def work() -> None:
        """The call, and its outcome handed to the loop whatever it raised: nothing that goes wrong
        here (an error whose message raises included) leaves the future unset."""
        result, error = None, None
        try:
            result = fn()
        except BaseException as e:  # noqa: BLE001 - every outcome is the caller's to have
            error = e
        if isinstance(error, StopIteration):  # cannot be set on a future
            error = RuntimeError(f"in a model call: {described(error)}")
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
    """The harness's record of a prepared clone call, which the original's machine decides:
    `grafts`, the grafts to commit; `audits`, each graft's (transaction id, audit text, scratch)
    written with the commit; `text`, the call's result if it commits; `txs`, the open audit
    transactions, rolled back if it is refused."""
    grafts: list[Graft]
    audits: list[tuple[int, str, dict]]
    text: str
    txs: list[int]


def _described(error: BaseException) -> str:
    """An error as one line tells it: its type and text, and each error of a group, in order."""
    if isinstance(error, BaseExceptionGroup):
        return f"{type(error).__name__}: {error.message}: " + "; ".join(_described(e) for e in error.exceptions)
    return described(error)  # guarded: a message that raises shows the type alone


class Harness:
    def __init__(self, store: Store, provider: Provider, world: World | None = None,
                 trace: Callable[[str], None] | None = None, max_steps: int = 10,
                 persona: str = BASE_SYSTEM, nonce: str | None = None,
                 compact_tokens: int | None = None, compaction_mode: str = "turns", seeder=...,
                 window_tokens: int | None = ..., emit: Callable[[str], None] | None = None,
                 call_timeout: float | None = MODEL_CALL_TIMEOUT,
                 names: Callable[[], str] | None = None):
        self.store = store
        # The source of the ids a fork call leaves out (seeded in tests).
        self.names = names or new_names()
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
        self.silent_done_refusals: list[str] = []  # grafts whose merge(done) call was refused
        # Decisions client (graft/seeder.py) that picks a graft's seed; None: the conversation so far
        self.seeder = seeder_mod.make_seeder() if seeder is ... else seeder
        self.seeder_log: list[dict] = []  # per clone: Jev latency, tokens, scores, selection, fallback
        self.out: list[str] = []  # user-facing texts: only the conversation holder's
        self.emit = emit or (lambda s: None)  # called with each user-facing text as it is said
        self._mark = 0  # self.out[:_mark] was already returned by user_says / settle
        # per wake run: {source, joined: [(clone, reason)]}
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
        self.store.create_timeline(tl=root, source=None, skills=["fork"])
        await self.machines.start_root(root)

    def _report_failure(self, *, kind: str, src: str | None, clones: list[str], error: BaseException,
                        quiet: bool, background: bool) -> None:
        """The one place a failure is recorded and shown (it is user-facing output and goes
        through `emit`). The errors `settle` raises are the session machine's; this is the line
        the user sees at once, and the record kept for the experiments. `background`: the clones
        are all background grafts (the machine says so: it made them)."""
        err = _described(error)
        record = (src, clones, err)
        # the user knows a graft by the id the model gave or drew, never by its timeline id; the
        # failure told may be the store's own, so a name that cannot be read is left out
        try:
            shown = [self.shown(c) for c in clones]
        except Exception:  # noqa: BLE001 - the line is still told, without the ids
            shown = []
        names = self._names(clones=shown, background=background)
        if kind == "wake":
            self.wake_errors.append(record)
            line = f"(harness error: {names} merged, but reporting the result failed: {err})"
            if not clones:
                line = f"(harness error: a wake run failed: {err})"
        elif kind == "commit":
            self.commit_errors.append(record)
            line = (f"(harness error: committing the merge of {names} failed: {err}; "
                    "the result is kept and the merge will be retried)")
        elif kind == "abandon":
            line = (f"(harness error: the merge of {names} could not be applied after {MAX_JOIN_ATTEMPTS} attempts "
                    f"({err}); it is abandoned. What was done there stays in its frozen timeline and is not part of the "
                    "conversation's memory.)")
        elif kind == "cancel":
            self.task_errors.append(record)
            line = (f"(harness error: {names} was cancelled and stopped, but recording that failed: {err}; "
                    "its merge is queued and will still be applied)")
        elif kind == "cancel_result":
            self.task_errors.append(record)
            what = (f"{names} was cancelled and stopped by that call" if clones
                    else "no branch was stopped by that call")
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
        """One user message, put to the original's machine as a request (an id, the text and the
        reply future): the machine answers it, forwards it to its focus graft, or keeps it until
        the run in progress and the joins waiting are done. Returns the user-facing texts emitted
        since the last return, once the conversation is at rest. A wake or join-commit failure
        after the turn is part of that output (a harness error line), not an exception.

        The request is put on the machine without waiting, so the call's only await is its reply:
        a cancel, wherever the request is in its lifecycle, cancels that reply at once and the
        caller withdraws the request by id. The machine then never starts a run, writes a row or
        lets a graft act for it; a run already answering it is ended by its machine before
        its next step, and a row already written for it is marked withdrawn. A cancel that lands
        after the machine answered but before this call took the answer wins too: CancelledError
        is raised, and the withdrawal stops the background grafts made for the request and marks
        its row (the machine honours it until it hears the call is done, `call_done`). If a step
        the message sets off fails, the machine answers the reply with that error, raised here."""
        root = self.machines.root
        request = self.machines.request(text)
        self.machines.post(tl="session", name="call_began", data=request.id)
        try:
            self.machines.post(tl=root, name="user_message", data=request)
            try:
                await request.reply
            except asyncio.CancelledError:  # the reply it awaited is cancelled with it
                with contextlib.suppress(TransitionRefused):  # closed meanwhile: nothing left to withdraw from
                    self.machines.post(tl=root, name="withdraw", data=request.id)
                raise
            return self._take()
        finally:  # a failure told while this call was in flight is this call's: settle does not raise it
            with contextlib.suppress(TransitionRefused):  # its answer is taken, or it was withdrawn
                self.machines.post(tl=root, name="call_done", data=request.id)
            with contextlib.suppress(TransitionRefused):  # closed meanwhile: nothing is kept any more
                self.machines.post(tl="session", name="call_done", data=request.id)

    async def settle(self) -> list[str]:
        """Wait until nothing is in flight: no background graft running, no join waiting that can
        be applied (joins queued on a suspended original wait for its focus graft), no wake run.
        Returns the user-facing texts emitted since the last return. Raises what failed meanwhile
        (in background grafts, wakes, join commits): the one error, or an exception group (a BaseExceptionGroup if one is not an Exception) of all of
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
        """Wait until tl, a timeline that can hold the conversation, is no longer running. Returns
        "ended" (it is at rest), "handed_off" (it is suspended under a focus graft) or "joined"
        (its run ended by joining, applied or not yet). TransitionRefused if tl does not rest: a
        background graft (wait for its join with `join_queued`), one whose join was abandoned,
        or a closed harness."""
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
    async def run_busy(self, tl: str, kind: str) -> RunOutcome:
        return await self._loop(tl=tl, kind=kind)

    async def run_background(self, tl: str) -> RunOutcome:
        return await self._loop(tl=tl, kind="background")

    def trace_line(self, text: str) -> None:
        """A trace line the machines write through the scribe (its failure is the scribe's to tell)."""
        self.trace(text)

    def mirror(self, tl: str, status: str) -> None:
        self.store.set_status(tl=tl, status=status)

    def begin_request(self, tl: str, text: str) -> int:
        rid = self.store.add_turn(tl=tl, role="user", kind="text", content=text)
        self.trace(f"[{tl}] user (row {rid}): {text}")
        return rid

    def mark_withdrawn(self, timelines: list[str], rows: list[int]) -> None:
        """User messages whose callers withdrew them after their rows were written: marked in the
        views of `timelines` (the machine names them), so those views show them as withdrawn and
        nothing reads them there as a request again."""
        self.store.mark_withdrawn(timelines=timelines, rows=rows)
        self.trace(f"[{timelines[0]}] withdrawn: rows {rows}")

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
        self._store_results(tl=tl, results=[ToolResult(use=use, content=content, is_error=is_error)])

    async def call_tool(self, tl: str, use: dict) -> tuple[Ran, bool]:
        """A host tool call (the activity of tl's machine's `calling` state). A World whose
        `execute` returns an awaitable (an `async def execute`) is awaited here, so other
        timelines go on while it waits; a cancel of the activity reaches the tool where it waits.
        A synchronous result is used as it is. Once the tool is invoked it has run, whatever
        follows: a tool that raises, or returns anything but (text, is_error), gets a result
        marked run, with an error naming what happened (it may have acted), never a failure of
        the run. Its trace line is the machine's (through the scribe), off this result's path."""
        try:
            outcome = self.world.execute(use["name"], use["input"])
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise  # a cancel is the activity's to tell apart (it has the state's context)
        except BaseException as error:  # noqa: BLE001 - it was invoked: it ran, with an error
            return Ran(f"raised {type(error).__name__}{_shown(error)}; it may have acted"), True
        if not (type(outcome) is tuple and tuple.__len__(outcome) == 2 and isinstance(outcome[0], str)
                and isinstance(outcome[1], bool)):
            return Ran(f"returned a {type(outcome).__name__}, not (text, is_error); it may have acted"), True
        content, err = outcome
        return Ran(str.__str__(content)), err

    def close_tool_uses(self, tl: str, known: dict[str, tuple[str, bool]]) -> None:
        self._close_tool_uses(tl=tl, known=known)

    def record_join(self, child: str, reason: str) -> None:
        self.store.set_join(tl=child, reason=reason, trigger=None)

    def apply_join(self, src: str, join: QueuedJoin, running: list[str]) -> tuple[bool, bool]:
        """One join applied to src (the activity of its machine's `applying` state). `running`:
        src's grafts still running (from src's machine). Returns (committed, whether src's view
        now ends on a user message it must answer)."""
        self.trace(f"JOIN {{clone: {join.child}, reason: {join.reason}, trigger: {join.trigger}}} -> {src}")
        self.store.set_join(tl=join.child, reason=join.reason, trigger=join.trigger)
        committed = self.default_join(src=src, child=join.child, reason=join.reason, trigger=join.trigger,
                                      running=running, closing=join.closing)
        return committed, self._ends_on_user_message(src)

    def report(self, kind: str, src: str | None, clones: list[str], error: BaseException, quiet: bool,
               background: bool) -> None:
        self._report_failure(kind=kind, src=src, clones=clones, error=error, quiet=quiet, background=background)

    def graft_failed(self, tl: str, error: BaseException) -> None:
        self.graft_errors.append((tl, _described(error)))
        self.trace(f"[{tl}] run failed -> join failed: {_described(error)}")

    def abandon_clone(self, src: str, grafts: list[str]) -> None:
        """A committed clone call whose machines could not all start: its grafts are marked
        abandoned (they exist in the store, but never run and never join)."""
        for g in grafts:
            self.store.set_status(tl=g, status="abandoned")
        self.trace(f"[{src}] clone undone: {grafts} abandoned")

    def recover(self, tl: str, known: dict[str, tuple[str, bool]], close_calls: bool, viewers: list[str],
                rows: list[int], cancels: list[tuple[CancelCall, dict[str, str]]]) -> None:
        """What a step of tl's machine that failed left open is closed out (the activity of its
        `recovering` state): a clone call's open audit transactions are rolled back, the run's
        open tool calls get results (`known`: the real results of calls that ran; `cancels`: the
        cancel calls of a step cut short after they stopped grafts, each with what became of its
        ids, whose real result is written; every other call was not executed; `close_calls` False
        for a turn its caller drives, whose calls are its own), and the rows of withdrawn requests
        still to be marked are marked in the views of `viewers`."""
        txs = self.store.open_txs(tl=tl, kind="clone")
        if txs:
            self._roll_back(src=tl, txs=txs, scratch={"error": "a step of the machine failed while the call was open"})
            self.trace(f"[{tl}] clone not executed: a step of the machine failed")
        if close_calls:
            ran = {call.use["id"]: self.cancel_text(call=call, outcomes=outcomes) for call, outcomes in cancels}
            self._close_tool_uses(tl=tl, known=ran | known)
        if rows:
            self.mark_withdrawn(timelines=viewers, rows=rows)

    def join_of(self, tl: str, reason: str) -> QueuedJoin:
        return QueuedJoin(child=tl, reason=reason, trigger=self.compute_trigger(child=tl, reason=reason))

    def shown(self, tl: str) -> str:
        """A graft as the model knows it: its name (the id it was given or drawn), never its
        timeline id. A timeline without one (the original) is shown by its timeline id."""
        name = self.store.name_of(tl)
        return name if name is not None else tl

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
                outcome = "it had already finished; nothing to stop"
            lines.append(f"{self.shown(c)}: not stopped: {outcome}")
        if stopped:
            head = (f"stopped -> {', '.join(map(self.shown, stopped))} (what you did in {'it' if len(stopped) == 1 else 'them'} "
                    "so far merges when this turn ends, and nothing already done outside this conversation "
                    "is undone)")
            lines = [head, *lines]
        self.trace(f"cancel {call.ids}: stopped {stopped}")
        return "\n".join(lines) or "nothing to stop", not stopped

    def system(self, tl: str) -> str:
        """Persona (same on every timeline) + skill listing (the only per-timeline part)."""
        names = [s["name"] for s in self.store.skills_for(tl)]
        parts = [self.persona, skill_listing(
            self.store, names,
            "Your skills (call load_skill(name) to read a skill's full instructions before using it):")]
        name = self.store.name_of(tl)
        if name is not None:
            parts.append(f"This branch of yours has the id {name}.")
        if "fork" in names:
            catalog = [r[0] for r in self.store.con.execute(
                "SELECT name FROM skills ORDER BY skill_id") if r[0] not in BUILTIN]
            parts.append(skill_listing(store=self.store, names=catalog,
                                       heading="Skill catalog (the skills a branch of yours can hold):"))
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
        """'detached branches c1, c2' / 'branch c1'; with named=False just the noun. `background`:
        all of them are background grafts (detached branches, as the model is told them). Grafts are
        told to the model, and to the user, as branches, by their names (spec: What the model is told:
        fork, merge and stop)."""
        if not clones:
            return "a branch"
        word = "detached branch" if background else "branch"
        word += "es" if len(clones) != 1 else ""
        return f"{word} {', '.join(clones)}" if named else word  # clones: names, as the caller shows them


    # ------------------------------------------------------------ main loop
    @staticmethod
    def _result_block(result: ToolResult) -> dict:
        use, content, err = result
        block = {"tool_use_id": use["id"], "content": content, "is_error": err}
        if isinstance(content, NotRun):
            block["not_run"] = True  # structural: the harness did not run this call
        if isinstance(content, Ran):
            block["ran"] = True  # structural: the harness ran this call
        return block

    def _store_results(self, tl: str, results: list[ToolResult]) -> None:
        for result in results:
            self.store.add_turn(tl=tl, role="user", kind="tool_result", content=self._result_block(result),
                                tool_name=result[0]["name"])

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
        violations, then recorded with Store.mark_compaction. The full timeline is unchanged. A
        compaction is a step of tl's run: its machine is asked before each compaction call and again
        when it comes back, before anything of it is stored."""
        rows = self.store.working_view(tl)
        if len(rows) < 2 or not compaction.needs_compaction(render(rows), self.compact_tokens):
            return
        prefix, fresh = rows[:-1], rows[-1]
        if self.compaction_mode == "summary":
            await self.may_continue(tl)
            await self._summary_compact(tl, rows)
            return
        violations: list[str] | None = None
        calls: list[dict] = []
        for attempt in range(2):
            await self.may_continue(tl)
            budget = int(self.compact_tokens * compaction.BUDGET_FRACTION)
            blocks, last = await self.complete(
                f"{self.role(tl)}:compaction", self.system(tl),
                compaction.extract_request(render(prefix), prefix, violations, budget,
                                           compaction.estimate_tokens(render(rows))),
                [compaction.COMPACT_TOOL], max_tokens=4000)
            await self.may_continue(tl)  # nothing of a compaction that comes back after a cancel is stored
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
        fresh message]. A message the user withdrew is left out of what the summary call reads:
        the summary is a user-role row, and must not restate it as something the user asked."""
        prefix, fresh = rows[:-1], rows[-1]
        said = [r for r in prefix if not r["withdrawn"]]
        blocks, last = await self.complete(f"{self.role(tl)}:compaction", self.system(tl),
                                           compaction.request(render(said)), [], max_tokens=2000)
        await self.may_continue(tl)  # nothing of a summary that comes back after a cancel is stored
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
        missing = self._missing_results(tl=tl, known=known)
        if missing:
            self._store_results(tl=tl, results=missing)

    def _missing_results(self, tl: str, known: dict[str, tuple[str, bool]] | None) -> list[ToolResult]:
        """The results tl's last assistant turn still lacks (see `_close_tool_uses`), unwritten:
        a known result for a call that ran, else not executed. [] once every call has one."""
        rows = self.store.view(tl)
        last = max((i for i, r in enumerate(rows) if r["role"] == "assistant"), default=None)
        if last is None:
            return []
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
            missing.append(ToolResult(use={"id": use_id, "name": r["tool_name"]}, content=content, is_error=err))
        return missing

    async def may_continue(self, tl: str) -> None:
        """Ask tl's machine whether its run may take its next step (a model call, a tool, the next
        text). Returns when it may. A run whose request was withdrawn is not answered here:
        the machine leaves the run's state, which ends this run where it waits."""
        await self.machines.ask(tl=tl, name="may_continue", build=lambda reply: reply)

    async def _loop(self, tl: str, kind: str) -> RunOutcome:
        """The model loop for tl: the activity of its machine's `busy` (or `background`) state.
        Returns how it ended, and nothing more: a reply with no tool call, an empty response, a
        join (with its reason), or a spent step budget. What follows is the machine's to decide
        (a background graft joins with a reason, a focus graft that joins hands back, the
        original rests). A clone that hands the conversation to a focus graft ends the run from
        the machine (the state is left). `kind`: tl's kind, as its machine says; it decides only whether
        tl's text reaches the user.

        Whether the run may go on is its machine's decision, asked (`may_continue`) before every
        step that could act: each compaction call and model call, when either comes back (a
        response or a compaction is then not stored, said or acted on), after each text it says
        (nothing more of that response is said or run), and before each tool call and the
        response's cancel calls. A run whose request was withdrawn is ended by its machine
        there (the calls it stored but did not run are closed by the machine as not run).

        A host tool call is asked of the machine (`tool_request`) and made by it: the activity of
        its `calling` state runs the tool, awaiting it if it returns an awaitable (other timelines
        go on meanwhile), and writes its result row; the run waits for the reply, one call at a
        time, so a turn's results are stored in call order. Leaving the run while a tool runs
        cancels the tool where it waits, and the call is closed as run with an error."""
        background = kind == "background"
        # The run starts inside a machine step, so this first question is answered in that step
        # and the answer reaches the run only after it: a caller cancelled in that step has
        # withdrawn its request by then, and the machine has ended the run. Every later question
        # (before the first model call too) is answered in the step the run acts in.
        await self.may_continue(tl)
        await self.maybe_compact(tl)
        for _ in range(self.max_steps):
            await self.may_continue(tl)
            rows = self.store.working_view(tl)
            violations = validate(rows, tl)
            if violations:
                raise RuntimeError(f"{tl} view invalid before model call: {violations}")
            tools = self.store.tools_for(tl)
            blocks, last = await self.complete(self.role(tl), self.system(tl), render(rows), tools)
            await self.may_continue(tl)
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
                    await self.may_continue(tl)
            uses = [b for b in blocks if b["type"] == "tool_use"]
            if not uses:
                return RunOutcome(kind="replied")  # a background graft's reply is its result
            allowed = {t["name"] for t in tools}
            join_use = next((u for u in uses if u["name"] == "merge" and "merge" in allowed), None)
            clone_use = next((u for u in uses if u["name"] == "fork" and "fork" in allowed), None)
            refused_join = None
            if join_use and str(join_use["input"].get("reason")) == "done" and not self.replied(tl):
                refused_join, join_use = join_use, None
                self.silent_done_refusals.append(tl)
                self.trace(f"[{tl}] merge(done) refused: no reply since the last user message")
            if join_use:
                results = [ToolResult(use=u, content=NOT_RUN_JOINING, is_error=True) for u in uses if u is not join_use]
                results.append(ToolResult(use=join_use, content="merged", is_error=False))
                self._store_results(tl=tl, results=results)
                return RunOutcome(kind="join", reason=_reason(str(join_use["input"].get("reason", "failed"))))
            if clone_use:
                # this call's role was routing into a clone: account it as the clone transaction
                retag = getattr(self.provider, "retag", None)
                if retag:
                    retag(last, f"{self.role(tl)}:clone_tx")
                # Beside a clone call only cancel calls run, and they run first, wherever the model
                # put them: a cancelled graft is stopped before the clone call is decided. Every other call is answered without being run.
                cancels = [u for u in uses if u is not clone_use and u["name"] == "stop"]
                others = [ToolResult(use=u, content=NOT_RUN_BESIDE_CLONE, is_error=True) for u in uses
                          if u is not clone_use and u["name"] != "stop"]
                self._store_results(tl=tl, results=others)
                # one request carries the response's cancels and its clone: the machine runs the
                # cancels first, then decides the clone
                cancel = self._cancel_request(tl=tl, uses=cancels) if cancels else None
                await self.clone_call(src=tl, use=clone_use, store=True, cancels=cancel)
                if cancel is not None:
                    await self._settle_cancel(tl=tl, uses=cancels, reply=cancel.reply)
                continue
            cancels = []
            # Each result is stored as soon as its call returns, so a later call of the same
            # response that raises never leaves a call that ran recorded as not run.
            for u in uses:
                await self.may_continue(tl)
                if u is refused_join:
                    if background:
                        msg = ("merge(reason='done') refused: you have not written your result yet. First "
                               "write your final reply saying what you did; then call merge(reason='done').")
                    else:
                        msg = ("merge(reason='done') refused: you have not replied to the user since "
                               "their last message. First tell the user, in one or two sentences, what "
                               "you did; then call merge(reason='done').")
                    self._store_results(tl=tl, results=[ToolResult(use=u, content=msg, is_error=True)])
                elif u["name"] == "stop":  # graft's cancel: whoever calls it, whatever tools it was shown, the machine decides
                    cancels.append(u)
                elif u["name"] not in allowed:
                    absent = NotRun(f"tool '{u['name']}' is not available to you")
                    self._store_results(tl=tl, results=[ToolResult(use=u, content=absent, is_error=True)])
                elif u["name"] == "load_skill":
                    content, err = self.load(tl, str(u["input"].get("name", "")))
                    self.trace(f"[{tl}] load_skill({u['input'].get('name')}) -> {'error' if err else 'ok'}")
                    await self._store_ran(tl=tl, result=ToolResult(use=u, content=content, is_error=err))
                else:  # the activity of the machine's `calling` state; it also writes the result row
                    await self.machines.ask(tl=tl, name="tool_request", build=lambda reply, u=u: ToolCall(
                        use=u, reply=reply, interrupted=CANCELLED_WHILE_RUNNING, cancelled=CANCELLED_BY_THE_TOOL,
                        failed=FAILED_AFTER_INVOKED))
            # The cancel calls run after every other call's result is stored: one request to the
            # machine, which stores each call's real result once the grafts have stopped.
            if cancels:
                await self.may_continue(tl)
                await self._cancels(tl=tl, uses=cancels)
        self.trace(f"[{tl}] step budget exhausted")
        return RunOutcome(kind="exhausted")

    async def _store_ran(self, tl: str, result: ToolResult) -> None:
        """Store the result of a tool call that ran. If the row cannot be written, the real result
        is first handed to tl's machine (`result_unstored`), which keeps it for the run's closing,
        so the call is closed with what it returned, never as not run; then the error ends the run."""
        try:
            self._store_results(tl=tl, results=[result])
        except Exception:
            await self.machines.send(tl=tl, name="result_unstored", data=result)
            raise

    def _cancel_request(self, tl: str, uses: list[dict]) -> Cancel:
        """The cancel calls of one response, as one request for tl's machine. It cancels what the
        graph allows, waits for those grafts' joins to be queued, and stores each call's real
        result. A timeline that may not cancel at all answers with the reason; the results are
        then stored here (they are this run's own tool results)."""
        return Cancel(calls=[self._stop_call(u) for u in uses], reply=self.machines.future())

    def _stop_call(self, use: dict) -> CancelCall:
        """A stop call (graft's cancel) with its ids resolved to timeline ids, by the run, before
        it is put to the machine: an id resolves only through graft names, so a timeline id the
        model never saw names nothing, and an id no graft has is the call's own problem."""
        names, problems = cancel_ids(use.get("input"))
        ids = []
        for name in names:
            tl = self.store.timeline_named(name)
            if tl is None:
                problems.append(f"{name}: not stopped: no such branch")
            else:
                ids.append(tl)
        return CancelCall(use=use, ids=ids, problems=problems)

    async def _cancels(self, tl: str, uses: list[dict]) -> None:
        cancel = self._cancel_request(tl=tl, uses=uses)
        self.machines.post(tl=tl, name="cancel_request", data=cancel)  # the run's only await is the reply
        await self._settle_cancel(tl=tl, uses=uses, reply=cancel.reply)

    async def _settle_cancel(self, tl: str, uses: list[dict], reply: asyncio.Future) -> None:
        answer = await reply
        if isinstance(answer, str):  # refused whole: every id gets the reason; ours to store
            self.trace(f"[{tl}] cancel refused: {answer}")
            results = []
            for u in uses:
                call = self._stop_call(u)
                content, err = self.cancel_text(call=call, outcomes={c: answer for c in call.ids})
                results.append(ToolResult(use=u, content=content, is_error=err))
            self._store_results(tl=tl, results=results)

    # ------------------------------------------------------------ cancel record, parts, blocks
    def tools_run(self, child: str, closing: Sequence[ToolResult] = ()) -> list[str]:
        """The capability tools child executed itself, in order: the results the harness marked
        as run (`Ran`) where it executed the tool; the mark is read, never the result's wording,
        and no pairing by tool-use id is needed, so a repeated id hides nothing. A tool that
        returned an error was still executed and may have acted before it failed, so it is
        listed, marked "(returned an error)". Names only: tool input and output never leave a
        graft's timeline."""
        ran = []
        blocks = [(r["tool_name"], json.loads(r["content"])) for r in self.store.view(child)
                  if r["owner_timeline"] == child and r["kind"] == "tool_result"]
        # `closing`: the results its join writes first (a close-out it could not write as it froze)
        blocks += [(result[0]["name"], self._result_block(result)) for result in closing]
        for name, data in blocks:
            if data.get("ran"):
                ran.append(name + (" (returned an error)" if data.get("is_error") else ""))
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
            live.append(f"{self.shown(c)} (skills: {', '.join(skills)})")
        return ("Still running: " + ", ".join(live) + "." if live
                else "No other branch of yours is still running.")


    def wake_note(self, src: str, joined: list[tuple[str, str]], handed_back: bool, running: list[str],
                  background: bool) -> None:
        """Close src's view with one harness note (a user row, so src can continue) for the joins
        its machine is waking it for (the activity of its `noting` state). The note is factual and
        forbids nothing: the wake is a full run. `handed_back`: a focus graft failed or handed its
        request back, so that request is src's to answer (the machine says so). `running`: src's
        grafts still running; `background`: the joined grafts are all background grafts."""
        names = ", ".join(f"{self.shown(c)} ({said(r)})" for c, r in joined)
        dropped = [c for c, _ in joined if c in self.join_dropped]
        kept = [c for c, _ in joined if c not in self.join_dropped]
        where = "what you did there is in the rows above." if kept else ""
        if dropped:
            where = (where + " " if where else "") + (
                f"What you did in {', '.join(map(self.shown, dropped))} did not fit in your context and stays in the frozen "
                f"timeline of {'that branch' if len(dropped) == 1 else 'those branches'}; say so if asked.")
        task = ("A request came back: answer it. " if handed_back else "") + (
            "Then go on with what the user asked for that is still to do, unless a still-running "
            "branch of yours is already doing it. ")
        cancelled = [c for c, r in joined if r == CANCELLED]
        if cancelled:
            many = len(cancelled) > 1
            task = (f"You stopped {', '.join(map(self.shown, cancelled))}: tell the user {'they were' if many else 'it was'} "
                    f"stopped and what you had already done there. ") + task
        self.store.add_turn(tl=src, role="user", kind="text", harness=True, content=(
                            f"(harness note: your {self._names(clones=[c for c, _ in joined], background=background, named=False)} "
                            f"{names} merged; {where} {self._running_note(running)} "
                            "You hold the conversation: tell the user, in your own words and in the first "
                            f"person, only what you did in the branches that have merged. {task}"
                            "Do not describe still-running work as finished.)"))
        self.wakes.append({"source": src, "joined": joined})
        self.trace(f"[{src}] WAKE for joins: {names}")

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

    # ------------------------------------------------------------ default join (deterministic)
    def default_join_rows(self, src: str, child: str, reason: str, trigger: int | None,
                          closing: Sequence[ToolResult] = ()) -> tuple[list[int], str | None]:
        """Spec 'Default join': text rows (user + assistant) of the clone's working view that the
        source lacks; out_of_scope ends with the trigger; failed adds a harness error record, and
        cancelled a harness record of the tools the graft had run.
        Returns (row ids to append, error record text or None)."""
        have = self.store.pointed(src)
        rows = [r["row_id"] for r in self.store.working_view(child)
                if r["kind"] == "text" and r["row_id"] not in have and r["row_id"] != trigger]
        error = (f"(harness note: your branch {self.shown(child)} failed; its timeline is frozen)"
                 if reason == "failed" else None)
        if reason == CANCELLED:
            ran = self.tools_run(child, closing=closing)
            error = (f"(harness note: your branch {self.shown(child)} was stopped before it finished; its timeline is "
                     f"frozen. Before it was stopped you had run there: {', '.join(ran) if ran else 'no tools'}. "
                     "Nothing you did there outside this conversation was undone; a tool that returned an "
                     "error may still have acted before it failed.)")
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
        where = (f"what you did there is in the remembered rows {listed}, which follow." if rows
                 else "it brings back no rows you do not already hold.")
        full = (rows, merge_head(child=self.shown(child), reason=reason) + f"{where} {live})")
        if self.window_tokens is None:
            return full
        limit = int(self.window_tokens * JOIN_WINDOW_FRACTION)
        base = self.store.working_view(src)
        trig = [trigger] if trigger is not None and trigger in rows else []
        last = next((r for r in reversed(rows) if r != trigger and self.store.turn(r)["role"] == "assistant"), None)
        frozen = (merge_head(child=self.shown(child), reason=reason) + f"its rows {listed} are too large "
                  f"for this context, so only {{kept}} follow; the rest stay in {self.shown(child)}'s frozen timeline. "
                  f"{live})")
        tiers = [full]
        if last is not None:
            tiers.append(([last] + trig, frozen.format(kept=f"your last reply there (row {last})"
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

    def default_join(self, src: str, child: str, reason: str, trigger: int | None, running: list[str],
                     closing: dict[str, tuple[str, bool]] | None = None) -> bool:
        """Apply the default join of child to src, once. True: committed. False: the resulting
        view was invalid and only the record (and trigger) were appended (rolled back). If the
        write fails, nothing of the join is in the store, its audit record says `failed`, and
        the error is raised. `running`: src's grafts still running, named in the record.
        `closing`: the close-out child could not write as it froze (its machine carried it with the
        join): the results its last turn still lacks are written into child's timeline first, in
        the join's SQL transaction, so a retry after a failed write writes them once."""
        applied = self.store.join_outcome(child)
        if applied is not None:
            # The join is already in the store: an earlier attempt wrote it and then failed before
            # the machine recorded it. The store commit and the machine step are one unit, so the
            # retry finishes the machine step and writes nothing.
            return applied == "committed"
        closed = self._missing_results(tl=child, known=closing) if closing is not None else []
        closes = [(child, result[0]["name"], self._result_block(result)) for result in closed]
        if closes:
            self.trace(f"[{src}] join of {child} writes its close-out: {[c[2]['tool_use_id'] for c in closes]}")
        wanted, error = self.default_join_rows(src=src, child=child, reason=reason, trigger=trigger, closing=closed)
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
                skipped = (merge_head(child=self.shown(child), reason=reason)
                           + f"its rows were skipped (invalid view). {self._running_note(running)})")
                self.store.commit_join(source=src, record=skipped, rows=[], error=None,
                                       trigger=trig, tx=tx, status="rolled_back", committed_sql=None,
                                       scratch={"default": True, "violations": v}, closing=closes)
                return False
            # record, pointers, error row, trigger and the audit outcome: one SQL transaction
            self.store.commit_join(src, record, [r for r in rows if r != trigger], error=error, trigger=trig,
                                   tx=tx, status="committed", committed_sql=sql, scratch={"default": True},
                                   closing=closes)
        except Exception as e:  # nothing of the join was written; the audit record says so
            self.store.end_tx(tx, status="failed", committed_sql=None,
                              scratch={"default": True, "error": _described(e)})
            raise
        self.trace(f"[{src}] {sql}")
        return True

    @staticmethod
    def _virtual(owner: str, role: str, kind: str, content: str) -> dict:
        return {"row_id": None, "owner_timeline": owner, "role": role, "kind": kind,
                "tool_name": None, "content": content, "harness": 1}


    # ------------------------------------------------------------ one-shot clone
    def default_seed(self, src: str) -> list[int]:
        """A graft starts from the conversation, as a git branch starts from its parent: every
        seed candidate of the source's working view (a text row that is not a harness note or a
        message the user withdrew) up to and including the last user one, so the seed ends on the
        user message that asks for the work, and starts at the first user row. The source's tool
        calls, tool results and loaded skills are never seeded. [] if it holds no user message."""
        cands = [r for r in self.store.working_view(src) if seeder_mod.is_candidate(r)]
        users = [i for i, r in enumerate(cands) if r["role"] == "user"]
        return [r["row_id"] for r in cands[users[0]:users[-1] + 1]] if users else []

    def task_of(self, skills: list[str]) -> str:
        """The task a decisions client is asked about for one entry: the part of the latest user
        request its skills cover, said by their names and descriptions (an entry names no purpose)."""
        said = []
        for name in skills:
            row = self.store.con.execute("SELECT description FROM skills WHERE name=?", (name,)).fetchone()
            said.append(f"{name} ({row[0]})" if row else name)
        return "the part of the latest user request covered by: " + "; ".join(said)

    async def seed_with_actor(self, src: str, skills: list[str]) -> tuple[list[int], dict]:
        """With a decisions client (graft/seeder.py), it picks the seed: candidates are the text
        rows of the source's working view (harness notes and withdrawn messages excluded); rows
        scoring >= the threshold are seeded in view order, the trigger (latest user message)
        always. If the seed would start with an assistant row, the nearest earlier user text row
        is added so the clone's view starts with a user turn. On an error: the default seed."""
        view = self.store.working_view(src)
        cands = [{"row_id": r["row_id"], "role": r["role"], "text": r["content"]}
                 for r in view if seeder_mod.is_candidate(r)]
        users = [c for c in cands if c["role"] == "user"]
        trigger = users[-1]["row_id"] if users else None
        trig_text = users[-1]["text"] if users else ""
        threshold = seeder_mod.seed_threshold()
        task = self.task_of(skills)
        d = await self.seeder.score(task, cands, trig_text) if cands else seeder_mod.SeedDecision(error="no candidates")
        record = {"seeder": getattr(self.seeder, "name", "jev"), "model": d.model, "threshold": threshold,
                  "scores": {str(k): round(v, 4) for k, v in d.scores.items()}, "latency": d.latency,
                  "input_tokens": d.input_tokens, "requests": d.requests, "truncated": d.truncated,
                  "error": d.error, "candidates": [c["row_id"] for c in cands], "task": task}
        if d.error:
            seed, record["fallback"] = self.default_seed(src), "conversation"
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
        return seed, record

    def draw_names(self, entries: list[CloneEntry]) -> list[str]:
        """Each entry's id: the one the model gave, else one drawn from the harness's source,
        redrawn while it collides with a stored name or an id given or drawn earlier in this call.
        A candidate that is not a branch id, or a source that keeps colliding past MAX_NAME_DRAWS,
        raises: an error of the call's preparation, which refuses the call."""
        taken = self.store.names() | {e.name for e in entries if e.name is not None}
        out = []
        for e in entries:
            if e.name is not None:
                out.append(e.name)
                continue
            for _ in range(MAX_NAME_DRAWS):
                name = self.names()
                if not (isinstance(name, str) and GRAFT_ID.fullmatch(name)):
                    raise ValueError(NOT_A_DRAWN_ID.format(got=_echo(name)))
                if name not in taken:
                    break
            else:
                raise RuntimeError(NO_FREE_ID.format(draws=MAX_NAME_DRAWS))
            taken.add(name)
            out.append(name)
        return out

    async def clone_call(self, src: str, use: dict, store: bool = False, cancels: Cancel | None = None
                         ) -> tuple[list[str] | None, str | None, str]:
        """One clone call, put to src's machine as the model made it, with the cancel calls of the
        same response. The machine runs the cancels first, then decides whether the call may be
        tried at all, has it prepared (`prepare_clone`), and decides and commits it, all of it or
        none. Returns (new timeline ids in entry order, the focus graft's id or None, result
        text), or (None, None, refusal). `store`: the machine stores the call's result row, as the
        model loop needs."""
        children, focus, text = await self.machines.ask(tl=src, name="clone_request", build=lambda reply: Clone(
            use=use, reply=reply, store=store, cancels=cancels))
        if isinstance(text, Unwritten) and store:  # refused by a timeline that is not running: ours to write
            self._store_results(tl=src, results=[ToolResult(use=use, content=str(text), is_error=True)])
        return children, focus, text

    async def prepare_clone(self, src: str, clone: Clone) -> Prepared | str:
        """The clone call's preparation (the activity of src's machine's `preparing` state): the
        harness resolves, seeds (the seeder actor runs once per entry, concurrently) and validates
        every entry. Returns the prepared call, or the text of its refusal.

        The input is model-written and anything can fail on it. Whatever is raised in here (the
        seeder, a store write) ends as a refusal with every transaction `rolled_back`. If the run
        is ended meanwhile (this activity is cancelled), nothing more is done here: the machine
        rolls the call back in a state of its own (`roll_back_clone`)."""
        txs: list[int] = []
        try:
            prepared = await self._prepare_clone(src=src, use=clone.use, txs=txs)
        except asyncio.CancelledError:
            # the run left (the machine also rolls the call back, in a state of its own), or a cancel
            # that came from elsewhere (the machine takes it as the preparation's failure): either way
            # nothing of the call stays open; rolling back what is already rolled back changes nothing
            self._roll_back(src=src, txs=txs, scratch={"args": clone.use.get("input"), "error": "cancelled"})
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001 - an error, Exception or not, is a refusal of the call
            err = _described(e)
            self._roll_back(src=src, txs=txs, scratch={"args": clone.use.get("input"), "error": err})
            self.trace(f"[{src}] clone refused: internal error: {err}")
            return f"fork refused; no branch was started: internal error ({err}).\nCall fork once more."
        if isinstance(prepared, str):
            return prepared
        batch, pairs, names = prepared
        return Prepared(pairs=pairs, data=batch, names=names)

    def roll_back_clone(self, src: str) -> None:
        """A clone call whose preparation was cut short (the run was ended while it was being
        seeded): its open audit transactions are closed as `rolled_back` (the activity of the
        machine's `unpreparing` state; if this raises, the machine reports it)."""
        self._roll_back(src=src, txs=self.store.open_txs(tl=src, kind="clone"),
                        scratch={"error": "the turn ended while the call was prepared"})
        self.trace(f"[{src}] clone not executed: the turn ended while it was prepared")

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
        detail = f"internal error ({_described(error)})" if error is not None else why
        self._roll_back(src=src, txs=batch.txs, scratch={"refused": detail})
        self.trace(f"[{src}] clone refused: {detail}")
        return f"fork refused; no branch was started: {detail}.\nCall fork once more."

    async def _prepare_clone(self, src: str, use: dict, *, txs: list[int]
                             ) -> str | tuple[CloneBatch, list[tuple[str, bool]], dict[str, str]]:
        args = use["input"]
        entries, shape = clone_entries(args)
        ids = self.store.next_clone_ids(len(entries))  # distinct before anything commits
        txs += [self.store.begin_tx(src, "clone", target=c) for c in ids]
        pairs = [(child, e.focus) for child, e in zip(ids, entries)]
        # depth and the focus rule are the machine's guards
        used = self.store.names()
        given = [f"{e.label}: id {_echo(e.name)} is already used in this conversation; ids are never reused"
                 for e in entries if e.name is not None and e.name in used]
        seeds: list[list[int]] = [self.default_seed(src) for _ in entries]
        decisions: list[dict | None] = [None] * len(entries)
        if self.seeder is not None and not shape and not given:
            try:  # all entries, or none left running: a failure cancels and awaits the others
                async with asyncio.TaskGroup() as group:
                    calls = [group.create_task(self.seed_with_actor(src, e.skills)) for e in entries]
            except ExceptionGroup as failed:
                raise failed.exceptions[0] from None
            seeded = [c.result() for c in calls]
            seeds = [seed for seed, _ in seeded]
            decisions = [d for _, d in seeded]
        # the commit condition on every entry's final seed, whoever chose it (the default or the
        # decisions client): no row the user withdrew is seeded
        withdrawn = {r["row_id"] for r in self.store.view(src) if r["withdrawn"]}
        per_entry: list[list[str]] = []
        for child, e, seed in zip(ids, entries, seeds):
            v = [WITHDRAWN_SEED.format(row=r) for r in seed if r in withdrawn]
            v += check_skill_link(store=self.store, source=src, skills=e.skills)
            rows = [t for t in (self.store.turn(r) for r in seed) if t]
            v += validate(rows, child) if rows else ["there is no user message in the conversation to fork from"]
            per_entry.append(v)
        scratch = []
        for e, seed, v, decision in zip(entries, seeds, per_entry, decisions):
            sc = {"args": e.raw, "name": e.name, "seed": seed, "violations": v, "batch": ids, "focus": e.focus}
            if decision is not None:
                sc["seeder"] = decision
            scratch.append(sc)
        violations = shape + given + [f"{e.label}: {x}" for e, v in zip(entries, per_entry) for x in v]
        if violations:
            for tx, sc in zip(txs, scratch):
                self.store.end_tx(tx, status="rolled_back", committed_sql=None,
                                  scratch=dict(sc, batch_violations=violations))
            if not txs:  # no usable entry: one audit record of the refused call
                self.store.end_tx(self.store.begin_tx(src, "clone"), status="rolled_back", committed_sql=None,
                                  scratch={"violations": violations})
            self.trace(f"[{src}] clone refused: {violations}")
            return ("fork refused; no branch was started:\n- " + "\n- ".join(violations)
                    + "\nFix the branches and call fork once more.")
        names = self.draw_names(entries)  # only for a call that will be decided: no id is drawn for nothing
        for sc, name in zip(scratch, names):
            sc["name"] = name
        grafts = [Graft(child=child, seed=seed, skills=e.skills + ["merge" if e.focus else "detached"],
                        focus=e.focus, name=name)
                  for child, e, seed, name in zip(ids, entries, seeds, names)]
        parts, audits = [], []
        for tx, sc, g in zip(txs, scratch, grafts):
            mode = "FOCUS" if g.focus else "BACKGROUND"
            audits.append((tx, f"CLONE {g.child} NAME {g.name} {mode} SEED {g.seed} SKILLS {g.skills}", sc))
            parts.append(f"{g.name} ({'current' if g.focus else 'detached, running'}; skills {g.skills})")
        batch = CloneBatch(grafts=grafts, audits=audits, text="forked -> " + ", ".join(parts), txs=list(txs))
        return batch, pairs, dict(zip(ids, names))
