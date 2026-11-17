"""The internal-faults sweep (spec: Internal faults (e)-(h)). For each scenario a dry run counts the
calls of every function of graft.machine the graphs reach through the module (guards, effects,
entries, exits, the I/O work of `io` states and their helpers: `_rule` and the module's own calls
read each by name) and of every host call the machines' activities make. Then each of them raises
once, at its first, second, middle and last call, in a run of its own; and, two raises, a few such
faults whose recoveries together reach every target only a fault reaches are run again with each
of those targets raising at its first call after the fault. Each run is checked:

* nothing the fault gated acts: no clone commit and no graft model call beyond the dry run's,
  and no model call of the original beyond it but the one run a failed hand-back sets off;
* every call returns or raises within a bounded wait (none hangs), and the next `user_says` is
  answered, `settle` returns and `close` returns (or raises what it could not stop);
* the fault is reported once: raised by the one caller given it (and then nothing else reports
  it), or told as one line; a line that reached a user call's output is that call's (settle
  does not raise it), and one that reached no call's output is raised by settle, once; nothing
  reported at all only where the outputs and the store are as in the dry run (a guard whose
  branch was not the one to take). A failure of a host call (an activity's I/O) is handled by
  its state's error transition and reported at least once and at most twice (a line, and the
  caller that waited on that write);
* settle returns (the session's record is empty), one timeline holds the conversation, every view
  is valid, no clone transaction is left open, no withdrawal mark exists the user did not cause,
  and no user message goes to a focus graft or back more than once.

Every harness error line and every error a caller got counts: one not about an injected error
(nor a DecisionFault a raise caused) is a problem unless the dry run had it too. A line counts as
shown when some call's returned output has it; one shown by the call it went out during must not
also be raised by settle, one no output showed must be (a line that went out as a call raised, or as
it ended, is that call's or settle's: the session decides by when it heard the call was done, which
the test does not see). A background graft's run that raised the error (the host's `graft_failed`)
is its caller. A close that raised a fault of closing is followed by a third, which returns (spec:
"close returns, or raises what it could not stop"). The test observes only replies, host calls,
posts the registry refuses after close, and the store (the store's copy of states may lag a note
the scribe lost: STORE_COPY). A call not done in WAIT is slow (counted), not hung; hung is not done
after HANG more. A future failed with no one to retrieve it, or a coroutine never awaited, fails the
case. After every scenario, every target raised in some scenario but those UNREACHED names.

The sweep runs with `-m sweep` (it is left out of the default run); the check of what it leaves
out runs by default."""

import asyncio
import collections
import dataclasses
import gc
import inspect
import logging
import re
import threading
import warnings

import pytest

import graft.machine as machine
from graft.machine import DecisionFault, TransitionRefused
from test_parallel import (BAD, BG, FOCUS, STAY, STOP, FaultScript, clone_entry, clone_use, is_wake, make, text,
                           user_texts, violations)

WAIT = 3.0  # seconds a call may take before the case counts it as slow
# Seconds more a slow call is waited for before the case counts it as hung: a deadlock never ends,
# a call slowed by a loaded machine does (several sweeps at once can stall one past WAIT).
HANG = 30.0
SPIN = 4  # a message forwarded more often than this one way is a spin: the case stops it

# Functions of graft.machine the sweep does not make raise, and why: the plumbing that runs the
# behaviors (a raise there is not a behavior's fault: it is the wrapper itself), the builders of
# the graphs (run once, at import), and the async activities (reached as host calls instead).
PLUMBING = {"_rule", "_step", "_guard", "_holds", "_failed", "_signal", "_keep_fault", "_handling_failed", "_take_fault",
            "_held_back", "_pending", "checked", "Effect", "Call", "Recovers", "Entry", "Exit", "Guard", "_G"}
BUILDERS = {"mirrored", "into", "io", "Activity", "held_back_in", "_work", "_timeline", "ended_as", "restating",
            "_join_arrivals"}
EXCLUDED = PLUMBING | BUILDERS


def _work_functions() -> set[str]:
    """The I/O work of the states whose activity is `io(done, work)`: read from the graphs."""
    works = set()
    for model in (machine.TIMELINE, machine.BACKGROUND, machine.SESSION, machine.SCRIBE):
        for element in model.members.values():
            operation = getattr(element, "operation", None)
            code = getattr(operation, "__code__", None)
            for name, cell in zip(code.co_freevars if code else (), getattr(operation, "__closure__", None) or ()):
                if name == "work" and inspect.isfunction(cell.cell_contents):
                    works.add(cell.cell_contents.__name__)
    return works


# Functions an activity calls (its I/O, before or after the host call): a raise there is the
# activity's failure, handled by its state's error transition like a host call's.
ACTIVITY_HELPERS = {"_mark_pending", "_close_out"} | _work_functions()
# The host calls the machines make (every one from an activity, or the scribe's).
HOST = ("run_busy", "run_background", "mirror", "report", "graft_failed", "begin_request", "mark_withdrawn",
        "prepare_clone", "roll_back_clone", "commit_clone", "refuse_clone", "store_result", "close_tool_uses",
        "record_join", "apply_join", "wake_note", "cancel_text", "join_of", "abandon_clone", "recover",
        "call_tool")
# A graft that makes a host tool call (site_write) and then replies: in the background (TOOL), or
# holding the conversation (TOOL_FOCUS: it then joins done).
TOOL, TOOL_FOCUS = "TOOL: build a website with a tool.", "TOOL_FOCUS: build a website with a tool."
LONG = "LONG: " + "a detail of the request " * 60  # a message long enough that the next turn compacts

SCENARIOS = ["plain", "BG", "FOCUS", "STAY", "BAD", "STOP", "withdrawn", "late", "prepare", "AB", "close", "compaction",
             "spawn", "tool", "tool_focus"]


def functions() -> list[str]:
    return sorted(n for n, f in vars(machine).items()
                  if inspect.isfunction(f) and f.__module__ == machine.__name__ and n not in EXCLUDED
                  and not inspect.iscoroutinefunction(f))


TARGETS = functions()  # read once, before any run stands in for one of them


class Spin(BaseException):
    """A message bounced between a source and its focus graft past the bound."""


@dataclasses.dataclass
class Case:
    """One run: what each call returned or raised, the lines shown, and what was done."""
    calls: list[tuple[str, object]] = dataclasses.field(default_factory=list)  # (label, lines or error)
    # (label, first, last): the lines shown (indexes in `said`) while each call was in flight
    windows: list[tuple[str, int, int]] = dataclasses.field(default_factory=list)
    said: list[str] = dataclasses.field(default_factory=list)  # every line shown, in order
    acts: tuple[int, int, int] = (0, 0, 0)  # original model calls, graft model calls, commits
    counts: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    after: collections.Counter = dataclasses.field(default_factory=collections.Counter)  # calls after the fault
    forwards: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    withdrawals: int = 0
    problems: list[str] = dataclasses.field(default_factory=list)
    slow: list[tuple[str, float]] = dataclasses.field(default_factory=list)  # calls past WAIT that ended
    fired: bool = False
    fired_second: bool = False
    injected: set[str] = dataclasses.field(default_factory=set)  # the targets that raised
    late: dict[str, asyncio.Task] = dataclasses.field(default_factory=dict)  # the call the late scenario cancels
    seeding: asyncio.Event | None = None  # the prepare scenario: its clone call's seeder began
    graft_errors: list[tuple[str, str]] = dataclasses.field(default_factory=list)  # graft runs that raised (host)
    scenario: bool = True  # the scenario's calls are being made (only they are counted, and fault)


class Policy(FaultScript):
    """The fault script, a compaction call answered by compressing every row into one, and TOOL and
    TOOL_FOCUS: the original clones one website graft (background, or focus), which calls
    site_write and then replies (a focus graft then joins done)."""

    def __call__(self, system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        request = user_texts(messages)[-1]
        if request in (TOOL, TOOL_FOCUS) and names != {"compact"}:
            if "clone" in names:
                if last["type"] == "text" and last["text"] == request:
                    entry = clone_entry(skills=["website"], focus=request == TOOL_FOCUS, quotes=[request])
                    return [clone_use(entry, id=f"k{len(messages)}") | {"type": "tool_use"}]
                return text("Reported." if is_wake(last) else "Started.")
            if last["type"] == "tool_result" and request == TOOL_FOCUS:
                return [{"type": "text", "text": "site done"},
                        {"type": "tool_use", "id": f"j{len(messages)}", "name": "join", "input": {"reason": "done"}}]
            if last["type"] == "tool_result":
                return text("site done")
            return [{"type": "tool_use", "id": f"w{len(messages)}", "name": "site_write",
                     "input": {"page": "index", "content": "hi"}}]
        if names == {"compact"}:
            ids = [int(x) for x in re.findall(r"row_id=(\d+) ", messages[-1]["content"][-1]["text"])]
            return [{"type": "tool_use", "id": "compaction", "name": "compact", "input": {"items": [
                {"op": "compress", "role": "user", "text": "the user asked for a long thing",
                 "begin_row_id": ids[0], "end_row_id": ids[-1]}]}}]
        return super().__call__(system=system, messages=messages, tools=tools)


def acts(h, store) -> tuple[int, int, int]:
    graft = sum(1 for c in h.provider.calls if "clone" not in {t["name"] for t in c["tools"]}
                and {t["name"] for t in c["tools"]} != {"compact"})
    commits = store.con.execute("SELECT count(*) FROM transactions WHERE kind='clone' AND status='committed'"
                                ).fetchone()[0]
    return len(h.provider.calls) - graft, graft, commits


async def call(case: Case, label: str, coro) -> None:
    """One call, bounded: what it returned or raised, and the lines shown while it was in flight. A
    call not done in WAIT is slow; one not done in WAIT + HANG is hung (every task's stack is shown)."""
    first = len(case.said)
    task = asyncio.ensure_future(coro)
    began = asyncio.get_running_loop().time()
    done, _ = await asyncio.wait({task}, timeout=WAIT)
    if not done:
        done, _ = await asyncio.wait({task}, timeout=HANG)
        case.slow += [(label, asyncio.get_running_loop().time() - began)] if done else []
    if not done:
        for pending in asyncio.all_tasks():
            pending.print_stack(limit=8)
        task.cancel()
        case.problems.append(f"{label} hung")
        case.calls.append((label, "hung"))
    elif task.cancelled():
        case.calls.append((label, "cancelled"))
    elif task.exception() is not None:
        case.calls.append((label, task.exception()))
    else:
        case.calls.append((label, task.result()))
    case.windows.append((label, first, len(case.said)))


class _Entered(threading.Event):
    """A scripted model call began (set in its thread), seen by the loop too (`seen`)."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop, self.seen = loop, asyncio.Event()

    def set(self) -> None:
        super().set()
        self.loop.call_soon_threadsafe(self.seen.set)


def _entered(script, key: str) -> _Entered:
    script.gate(key)
    script.entered[key] = entered = _Entered(loop=asyncio.get_running_loop())
    return entered


async def _first_of(waiting, task: asyncio.Future) -> None:
    """Until `waiting` is done or `task` is (bounded by WAIT + HANG, like a call)."""
    other = asyncio.ensure_future(waiting)
    await asyncio.wait({other, task}, timeout=WAIT + HANG, return_when=asyncio.FIRST_COMPLETED)
    other.cancel()


async def drive(h, store, script, scenario: str, case: Case, stashed: asyncio.Event) -> None:
    """The scenario's calls, each bounded; then settle (twice at most: a kept error is raised once)."""
    def call_(label, coro):
        return call(case=case, label=label, coro=coro)

    await h.start()
    for key in (BG, FOCUS, STAY):
        script.gate(f"graft:{key}")
    if scenario in ("plain", "BG", "FOCUS", "BAD", "spawn", "tool", "tool_focus"):
        script.open()
        await call_(label="user_says", coro=h.user_says({"plain": "plain", "BG": BG, "FOCUS": FOCUS, "BAD": BAD,
                                              "spawn": BG, "tool": TOOL, "tool_focus": TOOL_FOCUS}[scenario]))
    elif scenario == "STAY":
        script.open()
        await call_(label="user_says", coro=h.user_says(STAY))
        await call_(label="user_says", coro=h.user_says("B to the focus graft"))
    elif scenario == "STOP":
        await call_(label="user_says", coro=h.user_says(BG))  # c1 waits at its model call
        await call_(label="user_says", coro=h.user_says(STOP))
        script.open()
    elif scenario == "withdrawn":
        entered = _entered(script=script, key="A")
        first = len(case.said)
        task = asyncio.create_task(h.user_says("A"))
        await _first_of(waiting=entered.seen.wait(), task=task)  # A's model call began, or the fault stopped A before it
        if entered.is_set():
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=WAIT + HANG)
        if not done:
            case.problems.append("user_says hung")
        outcome = ("hung" if not done else "cancelled" if task.cancelled() else task.exception() or task.result())
        case.calls.append(("user_says", outcome))
        case.windows.append(("user_says", first, len(case.said)))
        script.open()
    elif scenario == "late":  # the call is cancelled in the tick it is answered: the cancel wins
        first = len(case.said)
        task = case.late["task"] = asyncio.create_task(h.user_says(BG))
        done, _ = await asyncio.wait({task}, timeout=WAIT + HANG)
        if not done:
            case.problems.append("user_says hung")
        outcome = ("hung" if not done else "cancelled" if task.cancelled() else task.exception() or task.result())
        case.calls.append(("user_says", outcome))
        case.windows.append(("user_says", first, len(case.said)))
        script.open()  # c1's model call returns: it stops before acting on it
    elif scenario == "prepare":  # cancelled while its clone call is prepared (the seeder waits): cut short
        case.seeding = asyncio.Event()
        first = len(case.said)
        task = asyncio.create_task(h.user_says(BG))
        await _first_of(waiting=case.seeding.wait(), task=task)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=WAIT + HANG)
        if not done:
            case.problems.append("user_says hung")
        outcome = ("hung" if not done else "cancelled" if task.cancelled() else task.exception() or task.result())
        case.calls.append(("user_says", outcome))
        case.windows.append(("user_says", first, len(case.said)))
    elif scenario == "AB":
        entered = _entered(script=script, key="A")
        first = asyncio.create_task(call_(label="user_says", coro=h.user_says("A")))
        await _first_of(waiting=entered.seen.wait(), task=first)  # unless the fault stopped A before it
        second = asyncio.create_task(call_(label="user_says", coro=h.user_says("B behind A")))
        await _first_of(waiting=stashed.wait(), task=second)  # B is held by the original's machine, behind A, or done
        script.open()
        await asyncio.gather(first, second)
    elif scenario == "close":
        await call_(label="user_says", coro=h.user_says(BG))  # c1 waits at its model call; close shuts it down
        script.open()
        await call_(label="close", coro=h.close())
        await call_(label="close", coro=h.close())  # a second close returns too, and every machine is stopped
        case.scenario = False
        if isinstance(case.calls[-1][1], Exception):  # both raised a fault of the closing: a third returns
            await call_(label="close", coro=h.close())
        live = []
        for (tl,) in store.con.execute("SELECT timeline_id FROM timelines"):
            try:  # a stopped machine is refused at once; a live one would take the event
                h.machines.post(tl=tl, name="poke")
                live.append(tl)
            except TransitionRefused:
                pass
        await call_(label="settle", coro=h.settle())  # a closed session answers at once (one closing would wait)
        if live:
            case.problems.append(f"machines left running after close: {live}")
        return
    elif scenario == "compaction":
        script.open()
        await call_(label="user_says", coro=h.user_says(LONG))
        await call_(label="user_says", coro=h.user_says("plain"))
    script.open()
    for _ in range(2):
        await call_(label="settle", coro=h.settle())
        if not isinstance(case.calls[-1][1], Exception):
            break


# The functions on the way of a state's store copy: a raise there loses that copy (not retried:
# the next copy of the state overwrites it), so the store may lag the machines. (`_forward` and
# `_send` carry every send, a state's copy among them.)
STORE_COPY = {"_forward", "_send", "_note", "take_note", "_scribe_write", "keeps", "is_write", "kept_after_report",
              "told_after_write", "s_keep", "s_tell_kept", "s_report_write", "s_keep_lost_line", "scribe_fault", "host.mirror",
              "host.abandon_clone"}  # (an undone call's grafts not marked abandoned: told, and the store keeps them)


async def usable(h, store, case: Case) -> None:
    """After the scenario: one timeline holds the conversation (by the store's copy of the states),
    the next `user_says` is answered, `settle` returns (or raises the kept error, once: counted
    with the reports) and `close` returns. The session's record is observed by what it answers: a
    settle returns only once nothing is in flight."""
    holding = [tl for (tl,) in store.con.execute("SELECT timeline_id FROM timelines WHERE status IN ('active', 'joining')")]
    if len(holding) != 1 and not (case.injected & STORE_COPY):
        case.problems.append(f"holding the conversation: {holding}")
    await call(case=case, label="next", coro=h.user_says("next"))
    label, out = case.calls[-1]
    if not (isinstance(out, list) and out and out[-1] == "answer to next"):
        case.problems.append(f"next answered {out!r}")
    for _ in range(2):
        await call(case=case, label="settle", coro=h.settle())
        if not isinstance(case.calls[-1][1], Exception):
            break
    await call(case=case, label="close", coro=h.close())


def run(tmp_path, monkeypatch, scenario: str, target: str | None = None, nth: int = 0, count: bool = False,
        second: str | None = None) -> Case:
    """One run of `scenario`; `target` raises at its `nth` call (a machine function, or `host.<name>`),
    and `second`, if any, raises at its first call after that."""
    script = Policy()
    tmp_path.mkdir(parents=True)
    store, h, said = make(tmp_path=tmp_path, policy=script, **({"compact_tokens": 400} if scenario == "compaction"
                                                                  else {}))
    case = Case(said=said)
    error = RuntimeError(f"INJECTED {target}#{nth}")
    error2 = RuntimeError(f"INJECTED {second}#after")
    stashed = asyncio.Event()

    def fault_of(name) -> BaseException | None:
        """Counts a call of `name`; the error it is to raise now, if any."""
        if not case.scenario:
            return None
        case.counts[name] += 1
        case.after[name] += int(case.fired)
        if name == target and case.counts[name] == nth and not case.fired:
            case.fired = True
            case.injected.add(name)
            return error
        if name == second and case.fired and not case.fired_second and case.after[name] == 1:
            case.fired_second = True
            case.injected.add(name)
            return error2
        return None

    def counted(name, real):
        def wrapper(*args, **kw):
            if (raised := fault_of(name)) is not None:
                raise raised
            return real(*args, **kw)
        wrapper.__name__ = real.__name__
        return wrapper

    def forwarding(name, real):
        def wrapper(ctx, inst, event):
            if isinstance(event.data, machine.Request):
                case.forwards[(name, event.data.id)] += 1
                if case.forwards[(name, event.data.id)] > SPIN:
                    raise Spin(f"{name} sent request {event.data.id} again and again")
            return real(ctx=ctx, inst=inst, event=event)
        wrapper.__name__ = real.__name__
        return wrapper

    real_stash = machine.stash_message

    def stash_message(ctx, inst, event):
        stashed.set()
        return real_stash(ctx=ctx, inst=inst, event=event)
    monkeypatch.setattr(machine, "stash_message", stash_message)
    real_rest = machine.come_to_rest

    def come_to_rest(ctx, inst, event):  # the late scenario's caller is cancelled as it is answered
        real_rest(ctx=ctx, inst=inst, event=event)
        task = case.late.pop("task", None)
        if task is not None and inst.timeline_id == "o":
            task.cancel()
    monkeypatch.setattr(machine, "come_to_rest", come_to_rest)
    for name in ("forward_to_focus", "return_to_source"):
        monkeypatch.setattr(machine, name, forwarding(name=name, real=getattr(machine, name)))
    named = [t for t in (target, second) if t]
    names = TARGETS if count else [t for t in named if not t.startswith("host.")]
    for name in names:
        monkeypatch.setattr(machine, name, counted(name=name, real=getattr(machine, name)))
    hosts = HOST if count else [t.removeprefix("host.") for t in named if t.startswith("host.")]
    for name in hosts:
        real = getattr(h, name)
        key = f"host.{name}"
        if inspect.iscoroutinefunction(real):
            def async_wrapper(*args, _key=key, _real=real, **kw):
                async def body():
                    if (raised := fault_of(_key)) is not None:
                        raise raised  # inside the activity's coroutine, as its own failure would be
                    return await _real(*args, **kw)
                return body()
            setattr(h, name, async_wrapper)
        else:
            setattr(h, name, counted(name=key, real=real))
    if scenario == "prepare":
        class Waiting:  # a seeder that waits until the preparation is cut short
            name = "waiting"

            async def score(self, purpose, cands, trigger):
                case.seeding.set()
                await asyncio.get_running_loop().create_future()
        h.seeder = Waiting()
    if scenario == "spawn":
        real_started = machine.hsm.Started

        async def started_or_fail(ctx, sm, model, config=None):
            if getattr(sm, "timeline_id", None) == "c1":
                raise RuntimeError("cannot start")
            return await real_started(ctx, sm, model, config)
        monkeypatch.setattr(machine.hsm, "Started", started_or_fail)

    async def go():
        await drive(h=h, store=store, script=script, scenario=scenario, case=case, stashed=stashed)
        case.scenario = False  # what follows checks the harness is usable: nothing more faults
        case.acts = acts(h=h, store=store)
        case.withdrawals = store.con.execute("SELECT count(*) FROM withdrawals").fetchone()[0]
        case.graft_errors = list(h.graft_errors)
        if scenario != "close":
            await usable(h=h, store=store, case=case)

    lost: list[str] = []

    class Lost(logging.Handler):
        def emit(self, record):
            if "never retrieved" in record.getMessage():
                lost.append(record.getMessage().splitlines()[0])
    handler = Lost()
    logging.getLogger("asyncio").addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                asyncio.run(go())
            except Spin as e:
                case.problems.append(f"spin: {e}")
            finally:
                script.open()
            gc.collect()  # a future failed with no one to retrieve it, or a coroutine never awaited, is told now
        lost += [str(w.message) for w in caught if "never awaited" in str(w.message)]
    finally:
        logging.getLogger("asyncio").removeHandler(handler)
    if lost:
        case.problems.append(f"lost: {lost}")
    for tl in [r[0] for r in store.con.execute("SELECT timeline_id FROM timelines")]:
        if store.view(tl) and (v := violations(store=store, tl=tl)):  # (a fault may leave one unwritten)
            case.problems.append(f"view of {tl} invalid: {v}")
    if store.con.execute("SELECT count(*) FROM transactions WHERE status='open'").fetchone()[0]:
        case.problems.append("a transaction left open")
    bounced = {k: n for k, n in case.forwards.items() if n > 1}
    if bounced:
        case.problems.append(f"forwarded more than once: {bounced}")
    return case


DECISION = "held, but its branch was not taken"  # a DecisionFault's text: a guard's raise changed a decision


def _is_mine(error: BaseException, text_of: str, decisions: bool = True) -> bool:
    """Is this error the injected one (or, with `decisions`, a DecisionFault a raise caused)?"""
    if isinstance(error, BaseExceptionGroup):
        return any(_is_mine(error=e, text_of=text_of, decisions=decisions) for e in error.exceptions)
    return text_of in str(error) or (decisions and isinstance(error, DecisionFault))


def _harness_lines(case: Case) -> list[int]:
    return [i for i, line in enumerate(case.said) if "harness error" in line]


def _reported(case: Case, text_of: str, activity: bool, scenario: str, decisions: bool = True
              ) -> tuple[list[str], bool]:
    """How one injected error (or, `decisions`, a DecisionFault its raise caused) was reported: the
    checks of the module docstring for it, and whether it was reported at all."""
    problems = []
    errors = [(label, e) for label, e in case.calls if isinstance(e, Exception)]
    mine = [(label, e) for label, e in errors if _is_mine(error=e, text_of=text_of, decisions=decisions)]
    by_caller = [label for label, _ in mine if label != "settle"]
    by_caller += [f"run of {tl}" for tl, e in case.graft_errors  # a graft's run raised it
                  if text_of in e or (decisions and DECISION in e)]
    by_settle = [label for label, _ in mine if label == "settle"]
    lines = [i for i in _harness_lines(case) if text_of in case.said[i] or (decisions and DECISION in case.said[i])]
    # shown by the user's call it went out during: in the lines that call returned. (A line that
    # went out as a call raised, or as its call was ending, is that call's or settle's: the session
    # decides by when it heard the call was done, which the test cannot see; either is once.)
    outcomes = [outcome for label, outcome in case.calls]
    waited = any(first <= i < last and isinstance(outcome, list) and case.said[i] in outcome
                 for i in lines for (label, first, last), outcome in zip(case.windows, outcomes)
                 if label in ("user_says", "next"))
    shown = any(case.said[i] in outcome for i in lines for outcome in outcomes if isinstance(outcome, list))
    if len(by_caller) > 1 or len(by_settle) > 1 or len(lines) > 1 + int(activity):
        problems.append(f"{text_of} reported more than once: {len(lines)} lines, raised by {by_caller + by_settle}")
    if by_caller and by_settle:
        problems.append(f"{text_of} raised by {by_caller} and by settle")
    if by_caller and lines and not activity:
        problems.append(f"{text_of} raised by {by_caller} and also told")
    if lines and waited and by_settle:
        problems.append(f"{text_of}: a line shown while a user's call was in flight was raised by settle too")
    if lines and not shown and not by_settle and not by_caller and scenario != "close":  # nothing settles after close
        problems.append(f"{text_of}: a line no call's output showed was never raised by settle")
    return problems, bool(by_caller or by_settle or lines)


def judge(case: Case, dry: Case, target: str, scenario: str, second: str | None = None) -> list[str]:
    """The checks of the module docstring, for one fired fault (and the second, if it fired). Every
    harness error line and every error a caller got counts: one that is not about an injected error
    (a cascade of it) is a problem unless the dry run had it too."""
    problems = list(case.problems)
    injected = [(target, f"INJECTED {target}#")] + ([(second, f"INJECTED {second}#after")] if case.fired_second else [])
    texts = [text for _, text in injected]
    activity = any(t.startswith("host.") or t in ACTIVITY_HELPERS for t, _ in injected)
    reported = []
    two = len(injected) > 1  # a DecisionFault is then checked on its own: it cannot be told which raise caused it
    for name, text in injected:
        found, any_report = _reported(case=case, text_of=text, activity=name.startswith("host.") or name in ACTIVITY_HELPERS,
                                      scenario=scenario, decisions=not two)
        problems += found
        reported.append(any_report)
    decisions = {str(e) for _, e in case.calls if isinstance(e, DecisionFault)} | {
        line[line.index("DecisionFault: ") + len("DecisionFault: "):].split(";")[0].rstrip(")")
        for line in case.said if "DecisionFault: " in line} | {
        e.removeprefix("DecisionFault: ") for _, e in case.graft_errors if e.startswith("DecisionFault: ")}
    for text in sorted(decisions) if two else []:
        found, any_report = _reported(case=case, text_of=text, activity=False, scenario=scenario, decisions=False)
        problems += found
        reported.append(any_report)
    if not activity and not any(reported):
        scenario_calls = [c for c in case.calls if c[0] in ("user_says", "settle", "close")]
        dry_calls = [c for c in dry.calls if c[0] in ("user_says", "settle", "close")]
        if (scenario_calls, case.withdrawals) != (dry_calls, dry.withdrawals):
            problems.append("nothing reported, but the outcome differs from the dry run")
    others = collections.Counter(case.said[i] for i in _harness_lines(case)
                                 if not any(text in case.said[i] for text in texts) and DECISION not in case.said[i])
    cascade = others - collections.Counter(dry.said[i] for i in _harness_lines(dry))
    if cascade:
        problems.append(f"harness error lines not about the injected errors: {sorted(cascade)}")
    errors = [(label, e) for label, e in case.calls if isinstance(e, Exception)]
    strays = [(label, e) for label, e in errors if not any(_is_mine(error=e, text_of=text) for text in texts)
              and not isinstance(e, TransitionRefused)]
    if strays:
        problems.append(f"other errors reached callers: {[(label, repr(e)[:100]) for label, e in strays]}")
    original, graft, commits = case.acts
    d_original, d_graft, d_commits = dry.acts
    # a failed write may refuse a clone call, which the model then retries: only a fault of the
    # machines' own code is held to the dry run's model calls (plus the one run a hand-back sets off)
    if commits > d_commits or graft > d_graft or (not activity and original > d_original + 1):
        problems.append(f"acted beyond the dry run: {case.acts} > {dry.acts}")
    if case.withdrawals > dry.withdrawals:
        problems.append(f"{case.withdrawals} withdrawal marks, the user caused {dry.withdrawals}")
    return problems


# The targets (machine functions and host calls) each scenario made raise, first or second: their
# union must be every target, but those named here, each with the repo test that makes it raise.
REACHED: dict[str, set[str]] = {}
_DRIVEN = "a turn its caller drives, or a run or wait asked for from outside (begin_turn, run, until_rest, drain_and_wake): no scenario makes one"
_JOINED = "a join raised from outside (do_join) or a background graft whose model calls join: no scenario makes one"
_ABANDON = "a join whose commit fails three times (abandoned): no scenario makes one"
_TIMING = "a withdrawal or message that lands at that point of a step (the scenarios' calls are not timed to it)"
_THIRD = "reached only once a recovery has itself failed (a third raise; the sweep raises twice at most)"
_CLOSE = "a close edge no scenario makes (a graft made after closing began, a machine that cannot be stopped, a user callback that raises)"
UNREACHED: dict[str, str] = {
    **dict.fromkeys(["begin_driven", "end_driven", "is_external", "not_external", "begin_run",
                     "add_run_waiter", "answer_ended", "answer_handed_off", "answer_joined", "engage_with_waiter",
                     "has_joined", "no_queue", "interrupted_closing", "error_shown", "run_failed_after_it_ended"],
                    _DRIVEN),
    **dict.fromkeys(["forced", "ended_by_join", "focus_join", "add_watcher"], _JOINED),
    **dict.fromkeys(["abandon_head", "report_abandoned", "last_attempt", "last_commit_failed", "close_abandoned"],
                    _ABANDON),
    **dict.fromkeys(["drop_message", "request_dropped", "request_held", "withdraws_waiting", "withdraw_and_forward",
                     "stash_and_engage", "take_handed_and_begin", "take_handed_and_engage", "return_to_source",
                     "s_withdraw", "one_of_pending", "refuse_cancel_call", "refuse_clone_call", "drop_clone",
                     "keep_unstored", "bg_fault_noted", "close_and_release", "tool_row_failed_late",
                     "bg_run_failed_late"], _TIMING),
    **dict.fromkeys(["fold_fault", "give_up_fault", "fault_releasing", "recovery_failed", "restating_again",
                     "restate_through_focus_again", "_closing_out", "close_out_failed"], _THIRD),
    **dict.fromkeys(["s_shut_late", "s_stop_crashed", "s_take_stop_errors", "s_tell_crashed", "s_keep_lost_line"],
                    _CLOSE),
}


def _cover(sets: dict[tuple[str, int], set[str]]) -> list[tuple[str, int]]:
    """Few faults whose recoveries reach, together, every target any fault's recovery reaches."""
    left, chosen = set().union(*sets.values()) if sets else set(), []
    while left:
        job = max(sorted(sets), key=lambda j: len(sets[j] & left))
        chosen.append(job)
        left -= sets[job]
    return chosen


@pytest.mark.sweep
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_machine_function_and_host_call_that_raises_once_is_recovered_from(tmp_path, monkeypatch, scenario):
    """See the module docstring. One raise: every target the dry run calls raises at its first,
    second, middle and last call. Two raises: faults whose recoveries together reach every target
    only a fault reaches (its recovery, its failure branches, the host calls they make) are each
    run again with every target its recovery reaches raising at its first call after the fault.
    Fails with every case that broke a check, by target and call."""
    dry = run(tmp_path=tmp_path / "dry", monkeypatch=monkeypatch, scenario=scenario, count=True)
    monkeypatch.undo()
    assert dry.problems == [], dry.problems
    jobs = [(target, k) for target, n in sorted(dry.counts.items()) for k in sorted({1, 2, (n + 1) // 2, n})
            if 1 <= k <= n]  # its first, second, middle and last call in the scenario
    failures, injected, after, slow = [], set(), {}, []
    for index, (target, k) in enumerate(jobs):
        with pytest.MonkeyPatch.context() as patch:
            case = run(tmp_path=tmp_path / f"case{index}", monkeypatch=patch, scenario=scenario, target=target, nth=k,
                       count=True)
        if not case.fired:
            continue
        injected.add(target)
        slow += case.slow
        after[(target, k)] = {name for name, n in case.after.items() if n}
        problems = judge(case=case, dry=dry, target=target, scenario=scenario)
        if problems:
            failures.append(f"{target}#{k}: {problems}")
    recovery = {job: reached - set(dry.counts) for job, reached in after.items()}
    pairs = [(job, second) for job in _cover(recovery)
             for second in sorted(recovery[job] | {n for n in after[job] if n.startswith("host.")})]
    twice = 0
    for index, ((target, k), second) in enumerate(pairs):
        with pytest.MonkeyPatch.context() as patch:
            case = run(tmp_path=tmp_path / f"two{index}", monkeypatch=patch, scenario=scenario, target=target, nth=k,
                       second=second)
        if not case.fired_second:
            continue
        twice += 1
        slow += case.slow
        injected.add(second)
        problems = judge(case=case, dry=dry, target=target, scenario=scenario, second=second)
        if problems:
            failures.append(f"{target}#{k} then {second}: {problems}")
    REACHED[scenario] = injected
    print(f"\nsweep {scenario}: one raise {len(after)}/{len(jobs)} fired; two raises {twice}/{len(pairs)} fired; "
          f"{len(injected)} targets raised; {len(failures)} failed; {len(slow)} calls slow (past {WAIT}s) but done")
    assert not failures, "\n".join(failures)


@pytest.mark.sweep
def test_the_sweep_makes_every_target_raise_or_names_the_test_that_does():
    """Run after every scenario of the sweep (it is skipped otherwise): every machine function and
    host call the sweep counts raised in some scenario, but those in UNREACHED."""
    if set(REACHED) != set(SCENARIOS):
        pytest.skip("reads what every scenario of the sweep made raise")
    every = set(TARGETS) | {f"host.{name}" for name in HOST}
    print("\nnever made to raise:", sorted(every - set().union(*REACHED.values())))
    assert sorted(every - set().union(*REACHED.values())) == sorted(UNREACHED)


def test_every_target_the_sweep_does_not_make_raise_is_named_with_its_reason():
    """UNREACHED names only targets the sweep counts, each with the reason no scenario reaches it
    (that it is exactly what the sweep leaves out is checked after the sweep, above)."""
    every = set(TARGETS) | {f"host.{name}" for name in HOST}
    assert set(UNREACHED) <= every and all(UNREACHED.values())


def test_the_sweep_leaves_out_only_plumbing_builders_and_activities():
    """Every function of graft.machine is swept, or left out for a stated reason."""
    every = {n for n, f in vars(machine).items() if inspect.isfunction(f) and f.__module__ == machine.__name__}
    activities = {n for n in every if inspect.iscoroutinefunction(getattr(machine, n))}
    assert every - set(functions()) == EXCLUDED | activities
    assert activities == {"busy_activity", "background_activity", "prepare_activity", "spawn_activity",
                          "unspawn_activity", "s_stop_machines", "tool_activity",
                          "store_activity"}


# ---------------------------------------------------------------- regressions (run by default)
def _clean(case: Case) -> list[str]:
    """What a case found, judged without a dry run: its own problems, and every error a caller got
    or a harness line said that is not an injected one."""
    texts = ["INJECTED "]
    problems = list(case.problems)
    problems += [f"{label}: {e!r}"[:160] for label, e in case.calls if isinstance(e, Exception)
                 and not any(_is_mine(error=e, text_of=t) for t in texts) and not isinstance(e, TransitionRefused)]
    problems += [case.said[i] for i in _harness_lines(case) if "INJECTED " not in case.said[i]]
    return problems


@pytest.mark.parametrize("scenario,first", [("BG", "take_clone"), ("BG", "attempt_clone"), ("FOCUS", "take_clone")])
def test_a_close_out_that_cannot_be_written_is_written_before_the_next_row(tmp_path, monkeypatch, scenario, first):
    """A step fails, and so does its recovery's close-out (`host.recover`): what it could not write
    (the run's open calls, the clone call's rollback) is kept and written before anything next
    writes to the view, so the next user_says is answered and every view stays valid."""
    case = run(tmp_path=tmp_path / "c", monkeypatch=monkeypatch, scenario=scenario, target=first, nth=1,
               second="host.recover")
    assert case.fired and case.fired_second
    assert _clean(case) == []
    assert case.calls[0] == ("user_says", case.calls[0][1]) and "INJECTED" in str(case.calls[0][1])


@pytest.mark.parametrize("scenario,first,second", [
    ("BG", "take_clone", "begin_recovery"), ("STAY", "take_clone", "begin_recovery"),
    ("BG", "attempt_clone", "begin_recovery"), ("BG", "take_clone", "recovered"),
    ("plain", "take_message", "_requeue_strays"), ("plain", "take_message", "_errors_for_session"),
    ("plain", "request_written", "_errors_for_session"), ("BG", "spawned", "_errors_for_session"),
    ("STOP", "cancel_head", "_errors_for_session"), ("BG", "take_clone", "_restate_and_fail"),
    ("plain", "take_message", "_fail_all"), ("plain", "take_message", "closed_out"),
])
def test_a_raise_in_a_recovery_is_folded_into_the_fault_it_handles(tmp_path, monkeypatch, scenario, first, second):
    """A step fails, and a behavior handling it raises too (at once, before it took anything, or
    after it took the callers out): the fault is handled again with that error added, so the
    first fault's caller gets its error once, nothing hangs, and the harness is usable."""
    case = run(tmp_path=tmp_path / "c", monkeypatch=monkeypatch, scenario=scenario, target=first, nth=1, second=second)
    assert case.fired and case.fired_second
    assert _clean(case) == []
    firsts = [label for label, outcome in case.calls if isinstance(outcome, Exception) and f"INJECTED {first}#" in str(outcome)]
    told = [i for i in _harness_lines(case) if f"INJECTED {first}#" in case.said[i]]
    assert len(firsts) + len(told) >= 1 and len(firsts) <= 1


def test_a_clone_call_whose_step_failed_before_it_was_taken_is_reported_once(tmp_path, monkeypatch):
    """`take_clone` raises before it keeps the call: the gate refuses the call with that error,
    without working out its rules on a call that is not there, so the run's caller raises it and
    nothing else reports anything (no second line, no future failed with no one to take it)."""
    case = run(tmp_path=tmp_path / "c", monkeypatch=monkeypatch, scenario="BG", target="take_clone", nth=1)
    assert case.fired and case.problems == []
    assert [label for label, outcome in case.calls if isinstance(outcome, Exception)] == ["user_says"]
    assert _harness_lines(case) == []


def test_a_call_past_the_bound_that_ends_is_slow_and_one_that_never_ends_is_hung(monkeypatch):
    """The sweep's bound on a call: one not done in WAIT that then ends is slow, not hung (several
    sweeps at once can stall a call past WAIT), and one still not done after HANG more is hung."""
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "WAIT", 0.01)
    monkeypatch.setattr(module, "HANG", 0.5)

    async def go():
        case = Case()
        loop = asyncio.get_running_loop()
        ended = asyncio.Event()
        loop.call_later(0.05, ended.set)  # after WAIT, well within WAIT + HANG

        async def slow():
            await ended.wait()
            return "done"
        await call(case=case, label="slow", coro=slow())
        await call(case=case, label="never", coro=loop.create_future())
        return case

    case = asyncio.run(go())
    assert case.calls == [("slow", "done"), ("never", "hung")]
    assert [label for label, _ in case.slow] == ["slow"] and case.problems == ["never hung"]
