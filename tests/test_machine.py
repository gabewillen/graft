"""The state machines on their own, with a scripted host: every request is an event, every answer
comes from the graph. The host does the I/O (here it records it), and a run of the model loop is
an activity the test ends. A test waits on a reply future or on a call the machines make to the
host (`Host.next`), never on loop turns, and asserts on what the host was asked to do."""

import ast
import contextlib
import asyncio

import pytest

from graft.machine import (CANCELLED, MAX_JOIN_ATTEMPTS, Cancel, CancelCall, Clone, Prepared,
                           Request, RunOutcome, ToolCall,
                           Machines, Mode, QueuedJoin, TransitionRefused)

GUARD = 5  # seconds: a wait that should end at once fails the test instead of hanging it
HEAD = 0  # the first item of a list


class Host:
    """A host that records what the machines ask of it. A run waits until the test ends it.
    `next(*prefix)` is a future for the next call whose log entry starts with `prefix`."""

    def __init__(self):
        self.log: list[tuple] = []
        self.status: dict[str, str] = {}
        self.runs: dict[str, asyncio.Future] = {}
        self.reports: list[tuple[str, list[str], str]] = []
        self.failing_joins: set[str] = set()
        self.join_reasons: dict[str, str] = {}  # a focus graft -> the reason its next join is given
        self.fail_commit = False
        self.hold: asyncio.Future | None = None  # set: a cancelled background run waits for it to end
        self.hold_tl: str | None = None  # set: only that timeline's cancelled run waits for `hold`
        self.tool_gate: asyncio.Future | None = None  # set: a host tool call waits for it
        self.watches: list[tuple[tuple, asyncio.Future]] = []

    def record(self, *entry):
        self.log.append(entry)
        for watch in list(self.watches):
            prefix, seen = watch
            if entry[:len(prefix)] == prefix:
                self.watches.remove(watch)
                if not seen.done():
                    seen.set_result(entry)

    def next(self, *prefix) -> asyncio.Future:
        seen = asyncio.get_running_loop().create_future()
        self.watches.append((prefix, seen))
        return seen

    def logged(self, *, prefix: tuple[str, ...]) -> list[tuple]:
        return [e for e in self.log if e[:len(prefix)] == prefix]

    async def _run(self, tl):
        self.runs[tl] = asyncio.get_running_loop().create_future()
        try:
            return await self.runs[tl]
        except asyncio.CancelledError:
            self.record("run_cancelled", tl)
            if self.hold is not None and self.hold_tl in (None, tl):
                await self.hold
            raise

    async def run_busy(self, tl, kind, woken=None):
        # a wake run is told the joins it is for (`woken`); its view ends on their merge results,
        # so no note is written for it: ("wake", ...) logs that a wake run began
        if woken is not None:
            self.record("wake", tuple(c for c, _ in woken), False)
        self.record("run", tl, "wake" if woken is not None else "turn")
        return await self._run(tl)

    async def run_background(self, tl):
        self.record("run", tl, "background")
        return await self._run(tl)

    def end(self, tl, result=RunOutcome(kind="replied")):
        self.runs.pop(tl).set_result(result)

    def fail(self, tl, error):
        self.runs.pop(tl).set_exception(error)

    def mirror(self, tl, status):
        self.status[tl] = status
        self.record("status", tl, status)

    def begin_request(self, tl, text):
        self.record("request", tl, text)
        return len(self.log)

    def mark_withdrawn(self, timelines, rows):
        self.record("withdrawn", tuple(timelines), list(rows))

    async def prepare_clone(self, src, clone):
        self.record("prepare", src)
        return Prepared(pairs=clone.pairs, data=None)

    def commit_clone(self, src, clone):
        if self.fail_commit:
            raise OSError("disk full")
        self.record("commit", src, [g for g, _ in clone.pairs])
        return "cloned"

    def roll_back_clone(self, src):
        self.record("roll_back", src)

    def refuse_clone(self, src, clone, why, error):
        return f"refused: {why}"

    def store_result(self, tl, use, content, is_error):
        self.record("result", tl, use["id"], content, is_error)

    async def call_tool(self, tl, use):
        self.record("tool", tl, use["id"])
        if self.tool_gate is not None:
            await self.tool_gate
        return "ran", False

    def close_tool_uses(self, tl, known):
        self.record("close", tl, dict(known))

    def record_join(self, child, reason):
        self.record("reason", child, reason)

    def apply_join(self, src, join, running):
        if join.child in self.failing_joins:
            raise OSError(f"cannot apply {join.child}")
        self.record("apply", join.child, join.reason)
        return True, False

    def cancel_text(self, call, outcomes):
        return repr(sorted(outcomes.items())), not any(v == CANCELLED for v in outcomes.values())

    def report(self, kind, src, clones, error, quiet, background):
        self.reports.append((kind, clones, str(error)))
        self.record("report", kind, clones, str(error))

    def graft_failed(self, tl, error):
        self.record("graft_failed", tl, str(error))

    def abandon_clone(self, src, grafts):
        self.record("abandoned", tuple(grafts))

    def join_of(self, tl, reason, focus=False):
        # the harness words a focus graft's reason from its view; a test says it (`join_reasons`)
        return QueuedJoin(child=tl, reason=self.join_reasons.get(tl, reason), trigger=None)

    def recover(self, tl, known, close_calls, viewers, rows, cancels):
        self.record("recover", tl, dict(known), close_calls, list(rows))

    def trace(self, text):
        pass

    def trace_line(self, text):
        pass


def holders(host) -> list[str]:
    """The timelines holding the conversation, by the states the machines mirrored to the host."""
    return sorted(t for t, status in host.status.items() if status in ("active", "joining"))


async def started():
    host = Host()
    m = Machines(host)
    await m.start_root("o")
    return host, m


REQUESTS: dict[asyncio.Future, int] = {}  # a reply future -> the id of its request


async def say(m, text):
    """Put a user message to the original; returns its reply future (answered at rest)."""
    request = m.request(text)
    REQUESTS[request.reply] = request.id
    await m.send(tl="o", name="user_message", data=request)
    return request.reply


async def withdraw(m, reply):
    """The caller of the request with this reply withdraws it (by id)."""
    await m.send(tl="o", name="withdraw", data=REQUESTS[reply])


async def clone(m, pairs):
    """A clone call of the original."""
    return await m.ask(tl="o", name="clone_request", build=lambda reply: Clone(
        use={"id": "k"}, reply=reply, store=True, pairs=pairs))


async def cancel(m, tl, ids):
    call = CancelCall(use={"id": "x"}, ids=ids, problems=[])
    return await m.ask(tl=tl, name="cancel_request", build=lambda reply: Cancel(calls=[call], reply=reply))


async def at_rest(m, tl="o"):
    """Wait until tl is at rest (a rest request is answered then); GUARD fails a wait that does not end."""
    return await asyncio.wait_for(m.ask(tl=tl, name="rest_request", build=lambda r: r), timeout=GUARD)


def run(coro_fn):
    asyncio.run(coro_fn())


# ------------------------------------------------------------------ a run, and the user's call
def test_a_user_message_is_a_turn_and_its_caller_is_answered_at_rest():
    async def go():
        host, m = await started()
        assert holders(host) == ["o"]
        reply = await say(m=m, text="hi")
        assert "o" in host.runs and ("request", "o", "hi") in host.log and not reply.done()
        host.end("o")
        assert await reply == "ended" and "o" not in host.runs
        assert await at_rest(m) == "ended" and holders(host) == ["o"]
        await m.stop()
    run(go)


def test_a_message_that_arrives_mid_run_waits_on_the_machine_and_is_answered_after_it():
    async def go():
        host, m = await started()
        first = await say(m=m, text="one")
        second = await say(m=m, text="two")  # the run is in progress: the machine keeps it
        assert host.logged(prefix=("request",)) == [("request", "o", "one")]
        ran = host.next("run", "o")
        host.end("o")
        await ran
        assert host.logged(prefix=("request",)) == [("request", "o", "one"), ("request", "o", "two")]
        assert not first.done()  # answered when the conversation is at rest
        host.end("o")
        assert await first == await second == "ended"
        await m.stop()
    run(go)


def test_a_run_that_raises_gives_the_waiting_caller_its_error_and_closes_its_calls():
    async def go():
        host, m = await started()
        reply = await say(m=m, text="hi")
        host.fail(tl="o", error=ValueError("model down"))
        with pytest.raises(ValueError, match="model down"):
            await reply
        assert host.logged(prefix=("close", "o")) and await at_rest(m) == "ended"
        await m.stop()
    run(go)


def test_an_interrupt_ends_the_run_and_closes_its_open_calls():
    async def go():
        host, m = await started()
        reply = await say(m=m, text="hi")
        await withdraw(m=m, reply=reply)
        assert ("run_cancelled", "o") in host.log and host.logged(prefix=("close", "o"))
        assert await reply == "ended" and await at_rest(m) == "ended"
        await m.stop()
    run(go)


# ------------------------------------------------------------------ clone
async def driven_turn(m):
    """A turn of the original that the test drives: its clone tries start afresh."""
    await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))


async def end_driven_turn(m):
    await m.ask(tl="o", name="driven_done", build=lambda r: r)


def test_clone_is_decided_by_the_graph():
    async def go():
        host, m = await started()
        _, _, text = await clone(m=m, pairs=[("c1", False)])  # at rest: not in a turn
        assert "refused" in text and not host.logged(prefix=("commit",))
        for pairs, refusal in (([("c1", True), ("c2", True)], "at most one branch may carry the conversation"),):
            await driven_turn(m=m)
            _, _, text = await clone(m=m, pairs=pairs)
            assert refusal in text and not host.logged(prefix=("commit",))
            await end_driven_turn(m=m)
        await driven_turn(m=m)
        host.fail_commit = True
        _, _, text = await clone(m=m, pairs=[("c1", False)])  # the store commit fails inside the step: refused
        assert "internal error" in text and "c1" not in host.status
        host.fail_commit = False
        children, focus, _ = await clone(m=m, pairs=[("c1", False)])  # the retry
        assert children == ["c1"] and focus is None and host.status["c1"] == "background"
        assert ("result", "o", "k", "cloned", False) in host.log  # the result row is the step's own
        await end_driven_turn(m=m)
        await m.ask(tl="session", name="shutdown", build=lambda r: r)
        await m.stop()
    run(go)


def test_a_committed_clone_call_is_not_the_runs_last():
    """No bound on fork calls in a run: after a committed call the run may fork again."""
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        children, _, _ = await clone(m=m, pairs=[("c1", False)])
        again, _, text = await clone(m=m, pairs=[("c2", False)])
        assert children == ["c1"] and again == ["c2"] and text == "cloned"
        assert len(host.logged(prefix=("prepare",))) == 2
        await m.stop()
    run(go)


def test_a_graft_is_refused_a_clone_by_the_graph_however_it_asks():
    """Grafts are one level deep: a focus graft in its own turn is refused by the graph, before
    anything is prepared."""
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        _, focus, _ = await clone(m=m, pairs=[("c1", True)])
        assert focus == "c1" and "c1" in host.runs  # in its turn
        _, _, text = await m.ask(tl="c1", name="clone_request",
                                 build=lambda r: Clone(use={"id": "z"}, reply=r, pairs=[("c9", False)]))
        assert "you fork only from main" in text and not host.logged(prefix=("prepare", "c1"))
        assert "c9" not in host.status
        await m.stop()
    run(go)


def test_a_wake_run_after_a_done_join_may_clone():
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        waking = host.next("run", "o", "wake")
        host.end("o")
        host.end("c1")  # c1's join is applied, then a wake: a full run
        await waking
        _, _, text = await clone(m=m, pairs=[("c2", False)])
        assert text == "cloned" and host.status["c2"] == "background"
        await m.stop()
    run(go)


def test_a_focus_clone_hands_the_conversation_over_and_its_join_goes_last():
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        children, focus, _ = await clone(m=m, pairs=[("c1", False), ("c2", True)])
        assert focus == "c2" and host.status["o"] == "cloned" and holders(host) == ["c2"] and "c2" in host.runs
        assert ("run_cancelled", "o") in host.log  # leaving the turn ended the original's run
        queued = host.next("status", "c1", "queued")
        host.end("c1")  # a background join waits on the suspended original
        await queued
        assert not host.logged(prefix=("apply",))
        waking = host.next("run", "o", "wake")
        host.end("c2")  # the focus graft answers the user: its turn ends in its join
        await waking  # the original runs once after the merges: c1's work is reported
        assert [e[1] for e in host.logged(prefix=("apply",))] == ["c1", "c2"]  # the focus join last
        assert host.logged(prefix=("wake",)) == [("wake", ("c1", "c2"), False)]
        host.end("o")
        assert await reply == "ended"  # the user's call is answered once the original is at rest
        assert (host.status["o"], host.status["c1"], host.status["c2"]) == ("active", "joined", "joined")
        assert holders(host) == ["o"]
        await m.stop()
    run(go)


def test_an_interrupt_of_the_users_call_reaches_the_focus_graft_answering_it():
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        _, focus, _ = await clone(m=m, pairs=[("c1", True)])
        assert focus == "c1" and "c1" in host.runs  # the focus graft's first run answers the call
        closed = host.next("close", "c1")
        await withdraw(m=m, reply=reply)  # forwarded by the suspended original
        await closed  # its run ended early: its open calls are closed
        assert ("run_cancelled", "c1") in host.log
        assert await at_rest(m=m, tl="c1") == "ended"
        await m.stop()
    run(go)


def test_a_focus_graft_whose_run_raises_joins_failed():
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", True)])
        joined = host.next("status", "c1", "joined")
        host.fail(tl="c1", error=RuntimeError("boom"))
        await joined
        assert ("graft_failed", "c1", "boom") in host.log and ("apply", "c1", "failed") in host.log
        assert holders(host) == ["o"]
        await m.stop()
    run(go)


# ------------------------------------------------------------------ the join queue
def test_joins_wait_while_the_original_runs_and_are_applied_when_it_is_at_rest():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False), ("c2", False)])
        for child in ("c2", "c1"):  # this finish order
            queued = host.next("status", child, "queued")
            host.end(child)
            await queued
        assert "o" in host.runs and not host.logged(prefix=("apply",))
        waking = host.next("run", "o", "wake")
        host.end("o")
        await waking
        assert [e[1] for e in host.logged(prefix=("apply",))] == ["c2", "c1"]  # finish order
        assert ("wake", ("c2", "c1"), False) in host.log  # one wake for both
        host.end("o")
        assert await first == "ended" and await at_rest(m) == "ended"
        assert host.status["c1"] == host.status["c2"] == "joined"
        await m.stop()
    run(go)


def test_a_graft_that_finishes_while_the_original_is_idle_is_applied_and_wakes_it():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        host.end("o")
        assert await first == "ended"
        waking = host.next("run", "o", "wake")
        host.end("c1")
        await waking
        assert ("apply", "c1", "done") in host.log
        reported = host.next("report", "wake")
        host.fail(tl="o", error=ValueError("wake exploded"))
        assert await reported == ("report", "wake", ["c1"], "wake exploded")
        assert await at_rest(m) == "ended"
        await m.stop()
    run(go)


def test_a_join_that_cannot_be_applied_is_kept_retried_and_then_given_up():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False), ("c2", False)])
        for child in ("c1", "c2"):
            queued = host.next("status", child, "queued")
            host.end(child)
            await queued
        host.failing_joins = {"c1"}
        host.end("o")
        assert await first == "ended"  # not retried at once: the original is at rest, nothing applied
        assert not host.logged(prefix=("apply",)) and host.status["c1"] == host.status["c2"] == "queued"
        assert host.reports == [("commit", ["c1", "c2"], "cannot apply c1")]
        for _ in range(MAX_JOIN_ATTEMPTS - 2):  # a commit point that fails again
            assert await m.ask(tl="o", name="drain_request", build=lambda r: r) == "ended"
        waking = host.next("run", "o", "wake")  # the last attempt: c1 is given up, c2 applied
        draining = asyncio.ensure_future(m.ask(tl="o", name="drain_request", build=lambda r: r))
        await waking
        host.end("o")
        assert await draining == "ended"
        assert host.status["c1"] == "abandoned" and host.status["c2"] == "joined"
        assert [k for k, _, _ in host.reports].count("abandon") == 1
        await m.stop()
    run(go)


# ------------------------------------------------------------------ cancel
def test_the_original_cancels_a_running_background_graft_and_waits_for_its_join():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False), ("c2", True)])  # c2 holds the conversation
        waking = host.next("run", "o", "wake")
        host.end("c2")  # the focus graft answered: its turn ends in its join, and the original runs once
        await waking
        host.end("o")
        assert await first == "ended" and await at_rest(m) == "ended" and holders(host) == ["o"]
        reply = await say(m=m, text="stop it")
        results = await cancel(m=m, tl="o", ids=["c1"])
        log = host.log
        assert log.index(("run_cancelled", "c1")) < log.index(("reason", "c1", CANCELLED))  # ended, then queued
        assert results == [(repr([("c1", CANCELLED)]), False)] and host.status["c1"] == "queued"
        assert "o" in host.runs and not host.logged(prefix=("apply", "c1"))  # queued: applied at rest
        _, _, text = await clone(m=m, pairs=[("n9", False)])
        assert text == "cloned"  # its part may be forked again: no block
        waking = host.next("run", "o", "wake")
        host.end("o")
        await waking
        assert ("apply", "c1", CANCELLED) in host.log and ("wake", ("c1",), False) in host.log
        host.end("o")
        assert await reply == "ended" and host.status["c1"] == "joined"
        await m.stop()
    run(go)


def test_cancel_is_refused_for_anything_but_a_running_background_graft_of_the_original():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False), ("c2", False)])
        queued = host.next("status", "c2", "queued")
        host.end("c2")
        await queued
        results = await cancel(m=m, tl="o", ids=["o", "c9", "c2"])
        outcomes = dict(ast.literal_eval(results[0][0]))
        assert outcomes == {"o": "main cannot be stopped", "c9": "no such branch",
                            "c2": "it has already finished or was already stopped; nothing to stop"}
        assert results[0][1] and host.status["c1"] == "background"  # nothing was touched
        assert "you stop branches only from main" in await cancel(m=m, tl="c1", ids=["c2"])  # a graft never cancels
        waking = host.next("run", "o", "wake")
        host.end("o")
        await waking  # the wake that reports c2
        host.end("o")
        assert await first == "ended"
        answer = await cancel(m=m, tl="o", ids=["c1"])  # at rest: not while it runs
        assert "only during your own turn" in answer and host.status["c1"] == "background"
        await m.stop()
    run(go)


def test_an_interrupt_waits_while_the_original_waits_for_a_graft_it_cancelled():
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        host.hold = m.future()
        stopping = host.next("run_cancelled", "c1")
        asking = asyncio.ensure_future(cancel(m=m, tl="o", ids=["c1"]))
        await stopping  # c1's run is ending, held there
        assert not asking.done() and host.status["c1"] == "background"
        await withdraw(m=m, reply=reply)  # noted by the machine; the cancel lands first
        assert not asking.done() and not host.logged(prefix=("close", "o"))
        closed = host.next("close", "o")
        host.hold.set_result(None)
        assert await asking == [(repr([("c1", CANCELLED)]), False)]
        await closed  # then the withdrawal ended the run
        result = ("result", "o", "x", repr([("c1", CANCELLED)]), False)
        assert host.log.index(result) < host.log.index(host.logged(prefix=("close", "o"))[0])
        assert await reply == "ended" and host.status["c1"] == "joined"  # c1's join is applied at rest
        assert not host.logged(prefix=("run", "o", "wake"))  # and no wake runs for a withdrawn call
        await m.stop()
    run(go)


@pytest.mark.parametrize("withdrawn", [False, True])
def test_a_run_that_fails_during_a_cancel_step_waits_for_the_step(withdrawn):
    """A cancel request from outside the original's run is waiting for the graft it cancelled when
    the run raises. The failure is the run's and waits for the step: the cancel finishes with its
    real result, then the run's caller raises the failure. If the caller withdrew meanwhile, the
    step ends the run, and the failure, which no caller waits on, is reported once."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        host.hold = m.future()
        stopping = host.next("run_cancelled", "c1")
        asking = asyncio.ensure_future(cancel(m=m, tl="o", ids=["c1"]))
        await stopping  # o waits for c1's join in its cancel step
        host.fail(tl="o", error=RuntimeError("run failed"))
        if withdrawn:
            await withdraw(m=m, reply=reply)
        waking = host.next("run", "o", "wake")  # c1's join is applied and reported at rest
        host.hold.set_result(None)
        assert await asyncio.wait_for(asking, GUARD) == [(repr([("c1", CANCELLED)]), False)]
        if withdrawn:
            assert await asyncio.wait_for(reply, GUARD) == "ended"
            await at_rest(m)
            assert [r for r in host.reports if r[2] == "run failed"] == [("task", [], "run failed")]
        else:
            await asyncio.wait_for(waking, GUARD)
            host.end("o")
            with pytest.raises(RuntimeError, match="run failed"):
                await asyncio.wait_for(reply, GUARD)
            assert [r for r in host.reports if r[2] == "run failed"] == []  # its caller has it
        assert host.status["c1"] == "joined"
        await m.stop()
    run(go)


# ------------------------------------------------------------------ the session
def test_settle_waits_until_nothing_is_in_flight_and_raises_what_failed_once():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        host.end("o")
        assert await first == "ended"
        settling = m.future()
        await m.send(tl="session", name="settle_request", data=settling)
        assert not settling.done()  # c1 is still running
        waking = host.next("run", "o", "wake")
        host.end("c1")
        await waking
        assert not settling.done()  # the wake run is in flight
        host.fail(tl="o", error=ValueError("wake exploded"))
        with pytest.raises(ValueError, match="wake exploded"):
            await settling
        assert await m.ask(tl="session", name="settle_request", build=lambda r: r) is None  # raised once
        await m.stop()
    run(go)


def test_close_shuts_running_grafts_down_then_closes_the_original():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False), ("c2", False)])
        host.end("o")
        assert await first == "ended"
        await m.ask(tl="session", name="shutdown", build=lambda r: r)
        assert host.logged(prefix=("reason",)) == [("reason", "c1", "failed"), ("reason", "c2", "failed")]
        assert [e[1] for e in host.logged(prefix=("apply",))] == ["c1", "c2"]
        assert not host.logged(prefix=("run", "o", "wake"))
        assert host.status["c1"] == "joined"
        with pytest.raises(TransitionRefused, match="refused in state closed"):  # closed
            await (await say(m=m, text="hello"))
        await m.stop()
    run(go)


def test_a_failing_write_in_an_activity_is_handled_by_its_state_and_kept_for_settle():
    """Writing the cancel calls' results is the activity of `writing`; it raises here. The state
    handles the error event: the caller gets the error, the original is back in its turn (not
    stuck in `writing`), and with no user's call waiting the failure is the next settle's."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        host.end("o")
        await first
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))

        def broken(call, outcomes):
            raise RuntimeError("host bug")

        host.cancel_text = broken
        with pytest.raises(RuntimeError, match="host bug"):
            await cancel(m=m, tl="o", ids=["c1"])
        assert ("machine", [], "host bug") in host.reports and host.status["c1"] == "queued"
        waking = host.next("run", "o", "wake")
        # the turn goes on: its caller ends it (it would wait in `writing`)
        assert await asyncio.wait_for(m.ask(tl="o", name="driven_done", build=lambda r: r), timeout=GUARD) == "ended"
        await waking
        host.end("o")  # the wake that reports c1
        with pytest.raises(RuntimeError, match="host bug"):
            await m.ask(tl="session", name="settle_request", build=lambda r: r)
        await m.stop()
    run(go)


def test_requests_to_a_machine_that_does_not_exist_are_refused():
    async def go():
        host, m = await started()
        with pytest.raises(TransitionRefused):
            await m.send(tl="c9", name="join")
        await m.stop()
    run(go)


# ------------------------------------------------------------------ every write is a state's activity
def _failing(host, name, error):
    real = getattr(host, name)

    def call(*args, **kwargs):
        raise error
    setattr(host, name, call)
    return real


@pytest.mark.parametrize("fails", [1, 2])
def test_a_withdrawal_mark_that_cannot_be_written_is_written_before_the_view_is_read_again(fails):
    """A withdrawn request's row whose mark cannot be written (`withdrawing` fails) does not stay a
    live request for later readers: the failure is told once, and the row is kept to be marked,
    by the next run before it reads the view (and, if that fails too, that run fails and the row
    is marked by a later one)."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        row = host.log.index(("request", "o", "go")) + 1
        real, left = host.mark_withdrawn, [fails]

        def mark_withdrawn(timelines, rows):
            if left[0]:
                left[0] -= 1
                raise OSError("cannot mark")
            real(timelines=timelines, rows=rows)
        host.mark_withdrawn = mark_withdrawn
        told = host.next("report", "task")
        await withdraw(m=m, reply=first)
        assert (await asyncio.wait_for(told, GUARD))[3] == "cannot mark"
        await at_rest(m)
        assert host.logged(prefix=("withdrawn",)) == []
        second = await say(m=m, text="next")
        if fails == 2:  # the next run's mark fails too: that run fails, before it reads anything
            with pytest.raises(OSError, match="cannot mark"):
                await asyncio.wait_for(second, GUARD)
            second = await say(m=m, text="again")
        marked = host.logged(prefix=("withdrawn",))
        assert marked == [("withdrawn", ("o",), [row])]
        runs = [i for i, entry in enumerate(host.log) if entry[:2] == ("run", "o")]
        assert runs[-2] < host.log.index(marked[0]) < runs[-1]  # written as the run that reads it begins
        host.end("o")
        assert await asyncio.wait_for(second, GUARD) == "ended"
        assert len([r for r in host.reports if r[2] == "cannot mark"]) == 1  # the run's failure is its caller's
        await m.stop()
    run(go)


def _session_sends_fail(m, names):
    """Fault injection: each event named is raised once when the session is sent it (in order)."""
    real, left = m.session.dispatch, list(names)
    errors = {name: RuntimeError(f"cannot send {name}") for name in names}

    def dispatch(ctx, event):
        if left and event.name == left[0]:
            raise errors[left.pop(HEAD)]
        return real(ctx, event)
    m.session.dispatch = dispatch
    return errors, left



def test_a_fault_in_the_recovery_s_own_step_is_folded_into_it():
    """The original's `rested` send fails (a fault), and so does the first record its recovery
    sends: the recovery is entered again, closes out, and the two errors are told together, once;
    settle raises them, and the next message is answered."""
    async def go():
        host, m = await started()
        errors, left = _session_sends_fail(m=m, names=["rested", "resync"])
        reply = await say(m=m, text="hi")
        host.end("o")
        assert await asyncio.wait_for(reply, GUARD) == "ended"  # answered before the send failed
        assert await at_rest(m) == "ended" and left == []
        assert len(host.logged(prefix=("recover", "o"))) == 1
        with pytest.raises(ExceptionGroup) as kept:
            await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), GUARD)
        assert list(kept.value.exceptions) == [errors["rested"], errors["resync"]]
        assert await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), GUARD) is None
        nxt = await say(m=m, text="two")
        host.end("o")
        assert await asyncio.wait_for(nxt, GUARD) == "ended"
        await m.stop()
    run(go)


def test_a_message_whose_step_fails_while_a_clone_call_starts_its_grafts_fails_alone(monkeypatch):
    """A message arrives while the original starts a clone call's graft, and the step that takes
    it fails (keeping it behind the run raises): only that call took part in the step, so only its
    caller gets the error, at once. The clone call goes on (the graft starts and runs), the turn
    is answered normally, and the session's record holds the graft."""
    import graft.machine as machine_module
    real, error = machine_module.stash_message, RuntimeError("cannot keep the message")

    def stash_message(ctx, inst, event):
        monkeypatch.setattr(machine_module, "stash_message", real)  # once
        raise error

    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        starting, release = _held_start_of(monkeypatch=monkeypatch, child="c1")
        asking = asyncio.ensure_future(clone(m=m, pairs=[("c1", False)]))
        await asyncio.wait_for(starting, GUARD)
        monkeypatch.setattr(machine_module, "stash_message", stash_message)
        second = await say(m=m, text="two")
        with pytest.raises(RuntimeError, match="cannot keep the message"):
            await asyncio.wait_for(second, GUARD)
        ran = host.next("run", "c1")
        release.set_result(None)
        assert (await asyncio.wait_for(asking, GUARD))[0] == ["c1"]
        await asyncio.wait_for(ran, GUARD)
        assert m.session.grafts == {"c1"} and host.logged(prefix=("recover",)) == []
        host.end("o")
        assert await asyncio.wait_for(first, GUARD) == "ended"
        woken = host.next("run", "o")
        host.end("c1")
        await asyncio.wait_for(woken, GUARD)  # its join is applied, and the original reports it
        host.end("o")
        await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), GUARD)
        await m.stop()
    run(go)


def test_a_fault_after_a_focus_graft_hands_its_request_back_fails_that_call(monkeypatch):
    """A focus graft joins without answering (`out_of_scope`): the request is the original's to
    answer, by a wake run. The guard of that branch raises once: it does not hold (a guard
    writes nothing), the decision's safe branch finds it held and reports the broken decision, and
    the user's call gets that error, never "ended" with nothing answered."""
    import graft.machine as machine_module
    real, armed = machine_module.wakes_to_run, []

    def raising(*args, **kw):
        if armed:
            armed.clear()
            raise RuntimeError("guard bug")
        return real(*args, **kw)
    monkeypatch.setattr(machine_module, "wakes_to_run", raising)

    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", True)])
        applied = host.next("apply", "c1")
        armed.append("armed")  # the wake guard's next read raises
        host.join_reasons["c1"] = "out_of_scope"
        host.end(tl="c1", result=RunOutcome(kind="join", reason="done"))
        await asyncio.wait_for(applied, GUARD)
        with pytest.raises(machine_module.DecisionFault, match="wakes_to_run held"):
            await asyncio.wait_for(reply, GUARD)
        assert host.logged(prefix=("wake",)) == [] and host.logged(prefix=("run", "o")) == [("run", "o", "turn")]
        assert await at_rest(m) == "ended"
        await m.stop()
    run(go)


def test_a_request_row_that_cannot_be_written_fails_the_call_explicitly_and_runs_nothing():
    async def go():
        host, m = await started()
        _failing(host=host, name="begin_request", error=OSError("disk full"))
        reply = await say(m=m, text="hi")
        with pytest.raises(OSError, match="disk full"):
            await reply
        assert ("request", ["disk full"]) == (host.reports[-1][0], [host.reports[-1][2]])
        assert await at_rest(m) == "ended" and not host.logged(prefix=("run",))  # no run started
        await m.stop()
    run(go)


@pytest.mark.parametrize("name, refused", [("commit_clone", True), ("store_result", False)])
def test_a_clone_write_that_fails_ends_in_its_handled_state(name, refused):
    """The commit is `committing`'s activity: if it fails, the call is refused (`refusing`). The
    result row is `recording`'s: if it fails, the clone stands, the row is kept for the run's
    closing, and the call's caller (the run) gets the error, so the run does not go on over a
    view without that row. Either way the original is back in its run, never stuck in the
    writing state."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        _failing(host=host, name=name, error=OSError("disk full"))
        if refused:
            children, _, text = await clone(m=m, pairs=[("c1", False)])
            assert children is None
        else:
            with pytest.raises(OSError, match="disk full"):
                await clone(m=m, pairs=[("c1", False)])
        assert "o" in host.runs
        await withdraw(m=m, reply=first)  # the run goes on until it is ended
        assert ("run_cancelled", "o") in host.log and await first == "ended"
        kept = {} if refused else {"k": ("cloned", False)}  # the real result, written by the closing
        assert host.logged(prefix=("close", "o")) == [("close", "o", kept)]
        if not refused:  # the clone stands; the withdrawal of the request it was made for stops it
            if not host.logged(prefix=("reason", "c1")):
                await asyncio.wait_for(host.next("reason", "c1"), GUARD)
            assert host.logged(prefix=("reason", "c1")) == [("reason", "c1", CANCELLED)]
        await m.stop()
    run(go)


def test_a_refused_clone_call_whose_result_row_cannot_be_written_ends_its_run():
    """The commit fails, so the call is refused, and the refusal's result row cannot be written
    either: the row is kept for the run's closing, and the call's caller (the run) gets the error,
    so no model call reads a view without that row."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        host.fail_commit = True
        _failing(host=host, name="store_result", error=OSError("disk full"))
        with pytest.raises(OSError, match="disk full"):
            await clone(m=m, pairs=[("c1", False)])
        assert "o" in host.runs  # the run's own failure ends it
        await withdraw(m=m, reply=first)
        assert await first == "ended"
        (closed,) = host.logged(prefix=("close", "o"))
        assert closed[2]["k"][1] is True and "no branch was started" in closed[2]["k"][0]  # the refusal, kept
        await m.stop()
    run(go)


def test_a_focus_clone_whose_result_row_fails_has_the_kept_row_written_before_the_hand_over():
    """The row kept by `recording` is written by `flushing`, the step every hand-over passes,
    before the original is suspended."""
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        _failing(host=host, name="store_result", error=OSError("disk full"))
        _, focus, _ = await clone(m=m, pairs=[("c1", True)])
        assert focus == "c1" and host.status["o"] == "cloned"
        closed = host.logged(prefix=("close", "o"))
        assert closed == [("close", "o", {"k": ("cloned", False)})]  # the real result, kept and written
        assert host.log.index(closed[0]) < host.log.index(("status", "o", "cloned"))  # before the hand-over
        await m.stop()
    run(go)


@pytest.mark.parametrize("pairs", [[("c1", False)], [("c0", False), ("c1", False)], [("c0", True), ("c1", False)]])
def test_a_graft_machine_that_cannot_start_undoes_the_whole_clone_call(monkeypatch, pairs):
    """The machines are started before anything else learns of the grafts: if one cannot start,
    the ones made are stopped and forgotten, the committed grafts are marked abandoned, and the
    result is a refusal. A focus graft is started last, so one whose call could not start every
    background graft never ran and never told the session it was engaged.
    Settle, close and a cancel of the dead id all still return."""
    import graft.machine as machine_module
    real = machine_module.hsm.Started

    async def started_or_fail(ctx, sm, model, config=None):
        if getattr(sm, "timeline_id", None) == "c1":
            raise RuntimeError("cannot start")
        return await real(ctx, sm, model, config)

    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        monkeypatch.setattr(machine_module.hsm, "Started", started_or_fail)
        children, _, text = await clone(m=m, pairs=pairs)
        monkeypatch.setattr(machine_module.hsm, "Started", real)
        assert children is None and "refused" in text
        assert ("abandoned", tuple(g for g, _ in pairs)) in host.log
        if pairs[HEAD][1]:  # a focus graft is started last: one that could not start leaves it unmade
            assert host.logged(prefix=("run", "c0")) == [] and "c0" not in m.timelines and "c0" not in m.session.engaged
        for g, _ in pairs:  # forgotten
            with pytest.raises(TransitionRefused, match="no machine"):
                await m.send(tl=g, name="cancel")
        answer = await cancel(m=m, tl="o", ids=["c1"])
        assert "no such branch" in answer[0][0]
        host.end("o")
        assert await first == "ended"
        await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), timeout=GUARD)
        await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), timeout=GUARD)
        await m.stop()
    run(go)


def test_a_failing_closing_write_leaves_the_original_at_rest():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        _failing(host=host, name="close_tool_uses", error=OSError("disk full"))
        reported = host.next("report", "task")
        await withdraw(m=m, reply=first)  # the run ends early: its calls need results
        assert (await reported)[3] == "disk full"
        assert await at_rest(m) == "ended"
        await m.stop()
    run(go)


def test_a_focus_graft_whose_failure_cannot_be_recorded_still_joins_failed():
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", True)])
        _failing(host=host, name="join_of", error=OSError("disk full"))
        joined = host.next("status", "c1", "joined")
        host.fail(tl="c1", error=RuntimeError("boom"))
        await joined
        assert ("apply", "c1", "failed") in host.log and holders(host) == ["o"]
        assert ("task", ["c1"], "disk full") in host.reports
        await m.stop()
    run(go)


# ------------------------------------------------------------------ cancel races, driven turns, blocks per request
@pytest.mark.parametrize("order", ["cancel_first", "finish_first"])
def test_a_cancel_that_loses_the_race_to_a_finish_keeps_the_finished_reason(order):
    """The two orders a cancel and a graft's own join can land in, each forced: a cancel that
    stops the graft is told so; one that finds the join already raised keeps its reason. Neither
    blocks the part."""
    async def go():
        host, m = await started()
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))
        await clone(m=m, pairs=[("c1", False)])
        if order == "finish_first":
            await m.ask(tl="c1", name="join", build=lambda r: (QueuedJoin(child="c1", reason="done", trigger=None), r))
        answer = await asyncio.wait_for(cancel(m=m, tl="o", ids=["c1"]), timeout=GUARD)
        cancelled = answer[0][1] is False
        assert cancelled == (order == "cancel_first")
        waking = host.next("run", "o", "wake")
        await m.ask(tl="o", name="driven_done", build=lambda r: r)
        await waking  # c1's join is applied and reported
        host.end("o")
        assert await at_rest(m) == "ended"
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))  # the same request
        _, _, text = await clone(m=m, pairs=[("c2", False)])
        assert text == "cloned"  # no part is blocked, whichever landed first
        await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), timeout=GUARD)
        await m.stop()
    run(go)


def test_a_cancel_accepted_while_its_graft_ends_for_another_reason_keeps_that_reason():
    """The cancel is accepted (the graft has not raised its join yet), but the graft is already
    leaving its run, forced to join `done`: its join lands with that reason, and the cancel's
    result says it had finished."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        host.end(tl="o")
        assert await first == "ended"
        host.hold = m.future()  # c1's run, once cancelled, waits here before it ends
        stopping = host.next("run_cancelled", "c1")
        forcing = asyncio.ensure_future(m.ask(tl="c1", name="join", build=lambda r: (
            QueuedJoin(child="c1", reason="done", trigger=None), r)))
        await stopping  # c1 is leaving its run for the forced join, mid-step
        await driven_turn(m=m)
        asking = asyncio.ensure_future(cancel(m=m, tl="o", ids=["c1"]))
        reached = host.next("reason", "c1")
        host.hold.set_result(None)
        await reached
        answer = await asking
        await forcing
        assert answer == [(repr([("c1", "done")]), True)]  # accepted, and its join landed as done
        _, _, text = await clone(m=m, pairs=[("c2", False)])
        assert text == "cloned"
        await end_driven_turn(m=m)
        await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), timeout=GUARD)
        await m.stop()
    run(go)


def test_a_caller_driven_end_refuses_to_end_a_model_run():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        with pytest.raises(TransitionRefused, match="'driven_done' refused in state"):
            await m.ask(tl="o", name="driven_done", build=lambda r: r)
        assert not first.done()  # the model's run goes on
        host.end("o")
        assert await asyncio.wait_for(first, timeout=GUARD) == "ended"
        await m.stop()
    run(go)


def test_cancelling_a_held_graft_in_a_caller_driven_turn_returns():
    async def go():
        host, m = await started()
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))
        await clone(m=m, pairs=[("c1", False)])  # held: the caller drives it
        answer = await asyncio.wait_for(cancel(m=m, tl="o", ids=["c1"]), timeout=GUARD)
        assert answer == [(repr([("c1", CANCELLED)]), False)]
        await asyncio.wait_for(m.ask(tl="o", name="driven_done", build=lambda r: r), timeout=GUARD)
        await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), timeout=GUARD)
        await m.stop()
    run(go)


# ------------------------------------------------------------------ a request withdrawn wherever it is
def _withdraws_inside(host, m, method, reply, when=lambda **kw: True):
    """The caller withdraws the request with this reply while the host does `method` (the event is
    put on the machine then, and taken after the step in progress)."""
    real = getattr(host, method)

    def call(**kw):
        if when(**kw):
            m.post(tl="o", name="withdraw", data=REQUESTS[reply])
        return real(**kw)
    setattr(host, method, call)


def test_a_request_withdrawn_while_its_row_is_written_starts_no_run():
    async def go():
        host, m = await started()
        first = await say(m=m, text="A")
        second = await say(m=m, text="B")  # waits behind A's run
        _withdraws_inside(host=host, m=m, method="begin_request", reply=second, when=lambda tl, text: text == "B")
        host.end("o")  # A's run ends: B's row is written, and B is withdrawn meanwhile
        assert await first == "ended" and await at_rest(m) == "ended"
        assert ("request", "o", "B") in host.log and host.logged(prefix=("run",)) == [("run", "o", "turn")]  # A's only
        assert await second == "ended"  # answered at rest, nothing run for it
        await m.stop()
    run(go)


@pytest.mark.parametrize("method", ["apply_join"])
def test_no_wake_runs_for_a_turn_whose_request_was_withdrawn(method):
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        queued = host.next("status", "c1", "queued")
        host.end("c1")
        await queued  # its join waits on the original's queue
        _withdraws_inside(host=host, m=m, method=method, reply=first)
        host.end("o")  # the turn ends: the join is applied; the caller withdraws meanwhile
        assert await first == "ended" and await at_rest(m) == "ended"
        assert ("apply", "c1", "done") in host.log and not host.logged(prefix=("run", "o", "wake"))
        if method == "apply_join":
            assert not host.logged(prefix=("wake",))  # no wake run either
        await m.stop()
    run(go)


@pytest.mark.parametrize("method", ["commit_clone", "store_result"])
def test_a_request_withdrawn_during_a_focus_clone_never_lets_the_focus_graft_act(method):
    """Withdrawn while the clone call commits (before the graft is made), the focus graft never
    starts a run; withdrawn while its result row is written (after the graft started), the
    withdrawal is forwarded and the graft's run ends. Either way the clone call is carried out
    whole, the conversation is handed over, and the request's written row is marked in the
    graft's view and its source's before the graft is at rest."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        row = host.log.index(("request", "o", "go")) + 1  # the row the host wrote for it
        _withdraws_inside(host=host, m=m, method=method, reply=reply)
        _, focus, _ = await clone(m=m, pairs=[("c1", True)])
        assert focus == "c1" and host.status["o"] == "cloned"
        assert await at_rest(m=m, tl="c1") == "ended"
        if method == "commit_clone":
            assert not host.logged(prefix=("run", "c1"))  # it never started a run
            assert await asyncio.wait_for(reply, GUARD) == "ended"  # and its caller is answered at its rest
        else:
            assert ("run_cancelled", "c1") in host.log  # its run was ended
        assert host.logged(prefix=("withdrawn",)) == [("withdrawn", ("c1", "o"), [row])]
        await m.stop()
    run(go)


def test_a_message_whose_caller_has_gone_before_it_is_taken_is_never_written():
    async def go():
        host, m = await started()
        request = m.request("gone")
        request.reply.cancel()  # its caller went away; its withdrawal has not arrived
        await m.send(tl="o", name="user_message", data=request)
        assert not host.logged(prefix=("request",)) and not host.logged(prefix=("run",))
        reply = await say(m=m, text="next")
        assert host.logged(prefix=("request",)) == [("request", "o", "next")]
        host.end("o")
        assert await reply == "ended"
        await m.stop()
    run(go)


def test_a_waiting_message_whose_caller_has_gone_is_dropped_before_its_withdrawal_arrives():
    async def go():
        host, m = await started()
        first = await say(m=m, text="A")
        second = await say(m=m, text="B")  # waits behind A
        second.cancel()  # the caller went away; its withdrawal has not arrived
        host.end("o")
        assert await first == "ended" and await at_rest(m) == "ended"
        assert host.logged(prefix=("request",)) == [("request", "o", "A")] and len(host.logged(prefix=("run",))) == 1
        await m.stop()
    run(go)


def test_a_withdrawn_waiting_message_is_answered_at_once_and_never_taken():
    async def go():
        host, m = await started()
        first = await say(m=m, text="A")
        second = await say(m=m, text="B")
        await withdraw(m=m, reply=second)
        assert await asyncio.wait_for(second, GUARD) == "ended" and "o" in host.runs  # A's run goes on
        host.end("o")
        assert await first == "ended" and host.logged(prefix=("request",)) == [("request", "o", "A")]
        await m.stop()
    run(go)


def test_a_clone_call_from_a_run_whose_caller_has_gone_is_not_tried():
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        reply.cancel()  # the caller went away; its withdrawal has not arrived
        children, _, text = await clone(m=m, pairs=[("c1", False)])
        assert children is None and "withdrawn" in text and not host.logged(prefix=("prepare",))
        await m.stop()
    run(go)


def test_handed_messages_go_ahead_of_those_waiting_and_joins_queued_are_applied_first():
    """`inbox_handed` carries messages that arrived before any the receiver holds: they go first.
    At rest with joins queued (a commit that failed), the joins are tried before the message."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="A")
        later = await say(m=m, text="C")  # waits on o
        earlier = m.request("B")
        await m.send(tl="o", name="inbox_handed", data=([earlier], set()))
        ran = host.next("run", "o")
        host.end("o")
        await ran
        assert [e[2] for e in host.logged(prefix=("request",))] == ["A", "B"]  # B, handed, before C
        ran = host.next("run", "o")
        host.end("o")
        await ran
        host.end("o")
        assert await first == await earlier.reply == await later == "ended"
        await clone_in_turn_then_fail_join(host=host, m=m)
        handed = m.request("D")
        host.failing_joins = set()
        ran = host.next("run", "o")
        await m.send(tl="o", name="inbox_handed", data=([handed], set()))
        await asyncio.wait_for(ran, GUARD)
        log = host.log
        assert log.index(("apply", "c1", "done")) < log.index(("request", "o", "D"))  # the join first
        host.end("o")
        assert await handed.reply == "ended"
        await m.stop()
    run(go)


async def clone_in_turn_then_fail_join(host, m):
    """o at rest with c1's join queued and held (its commit failed)."""
    reply = await say(m=m, text="go")
    await clone(m=m, pairs=[("c1", False)])
    queued = host.next("status", "c1", "queued")
    host.end("c1")
    await queued
    host.failing_joins = {"c1"}
    host.end("o")
    assert await reply == "ended" and host.status["c1"] == "queued"


def test_a_withdrawal_that_comes_while_the_focus_graft_is_started_ends_its_run_at_the_hand_over(monkeypatch):
    """The withdrawal reaches the original while the focus graft's machine is being started (made
    without it, and not yet able to take events): the withdrawals handed over with the
    conversation end the graft's run."""
    import graft.machine as machine_module
    real = machine_module.hsm.Started

    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        starting, release = m.future(), m.future()

        async def held_start(ctx, sm, model, config=None):
            if getattr(sm, "timeline_id", None) == "c1":
                starting.set_result(None)
                await release
            return await real(ctx, sm, model, config)

        monkeypatch.setattr(machine_module.hsm, "Started", held_start)
        asking = asyncio.ensure_future(clone(m=m, pairs=[("c1", True)]))
        await asyncio.wait_for(starting, GUARD)
        await withdraw(m=m, reply=reply)  # by id only: the reply itself is not cancelled
        ended = host.next("run_cancelled", "c1")
        release.set_result(None)
        assert (await asyncio.wait_for(asking, GUARD))[1] == "c1"
        await asyncio.wait_for(ended, GUARD)
        assert await at_rest(m=m, tl="c1") == "ended"
        await m.stop()
    run(go)


def test_a_focus_graft_whose_request_was_withdrawn_still_answers_the_next_message_handed_to_it():
    async def go():
        host, m = await started()
        first = await say(m=m, text="A")
        second = await say(m=m, text="B")  # waits on the original
        _withdraws_inside(host=host, m=m, method="commit_clone", reply=first)
        ran = host.next("run", "c1")
        _, focus, _ = await clone(m=m, pairs=[("c1", True)])
        assert focus == "c1"
        await asyncio.wait_for(ran, GUARD)  # c1 started at rest, then took B
        assert host.logged(prefix=("request", "c1")) == [("request", "c1", "B")]
        assert len(host.logged(prefix=("run", "c1"))) == 1
        waking = host.next("run", "o", "wake")
        host.end("c1")  # its turn on B ends in its join: the original runs once after it
        await asyncio.wait_for(waking, GUARD)
        host.end("o")
        assert await second == "ended" and await first == "ended"
        await m.stop()
    run(go)


def test_a_background_clone_carried_out_while_its_request_is_withdrawn_starts_no_graft_run_and_wakes_nothing():
    """The withdrawal lands while the clone call commits: the step finishes whole (the call ran),
    but the graft it makes never runs: it joins `cancelled` at once. The run then ends, and the
    join is applied with no wake: nothing is run or said for the withdrawn request."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        _withdraws_inside(host=host, m=m, method="commit_clone", reply=reply)
        children, _, _ = await clone(m=m, pairs=[("c1", False)])
        assert children == ["c1"]  # the step finished whole
        assert await reply == "ended" and ("run_cancelled", "o") in host.log  # then the run ended
        assert await at_rest(m) == "ended"
        assert host.logged(prefix=("close", "o")) and ("reason", "c1", CANCELLED) in host.log
        assert ("apply", "c1", CANCELLED) in host.log and host.status["c1"] == "joined"
        assert host.logged(prefix=("run", "c1")) == [] and host.logged(prefix=("wake",)) == []  # it never ran; no wake
        assert host.logged(prefix=("run", "o")) == [("run", "o", "turn")]
        await m.stop()
    run(go)


def test_a_background_graft_stops_when_the_request_it_was_made_for_is_withdrawn_and_wakes_nothing():
    """The graft runs; then the caller of the request its turn answered goes away (the reply is
    cancelled, as a cancelled `user_says` does, before the withdrawal arrives): at its next step
    the graft stops and joins `cancelled`, and its join wakes nothing. A graft made for a request
    that was answered keeps running after its turn and wakes its source when it joins."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        ran = host.next("run", "c1")
        await clone(m=m, pairs=[("c1", False)])
        await asyncio.wait_for(ran, GUARD)
        reply.cancel()
        stopped = host.next("run_cancelled", "c1")
        await m.send(tl="c1", name="may_continue", data=m.future())  # its run asks before its next step
        await asyncio.wait_for(stopped, GUARD)
        await m.send(tl="o", name="withdraw", data=REQUESTS[reply])
        assert await at_rest(m) == "ended"
        assert ("reason", "c1", CANCELLED) in host.log and ("apply", "c1", CANCELLED) in host.log
        assert host.logged(prefix=("wake",)) == [] and host.logged(prefix=("run", "o")) == [("run", "o", "turn")]
        # a graft made for a request that its turn answered goes on and wakes its source
        second = await say(m=m, text="again")
        ran = host.next("run", "c2")
        await clone(m=m, pairs=[("c2", False)])
        host.end("o")
        assert await second == "ended"
        await asyncio.wait_for(ran, GUARD)
        assert host.status["c2"] == "background"  # still running
        waking = host.next("run", "o", "wake")
        host.end("c2")
        await asyncio.wait_for(waking, GUARD)
        host.end("o")
        assert await at_rest(m) == "ended" and ("wake", ("c2",), False) in host.log
        await m.stop()
    run(go)


def _held_start_of(monkeypatch, child, fails=False):
    """hsm.Started holds `child`'s machine until released (or raises for it): (starting, release)."""
    import graft.machine as machine_module
    real = machine_module.hsm.Started
    loop = asyncio.get_running_loop()
    starting, release = loop.create_future(), loop.create_future()

    async def held_start(ctx, sm, model, config=None):
        if getattr(sm, "timeline_id", None) == child:
            starting.set_result(None)
            await release
            if fails:
                raise RuntimeError(f"cannot start {child}")
        return await real(ctx, sm, model, config)

    monkeypatch.setattr(machine_module.hsm, "Started", held_start)
    return starting, release


def test_a_background_graft_made_before_a_withdrawal_mid_spawn_stops_and_the_next_never_runs(monkeypatch):
    """Two background grafts; the caller goes away while the second one's machine is started: the
    first, already running, stops at its next step and the second never runs; both join
    `cancelled`, and nothing is woken."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        starting, release = _held_start_of(monkeypatch=monkeypatch, child="c2")
        ran = host.next("run", "c1")
        asking = asyncio.ensure_future(clone(m=m, pairs=[("c1", False), ("c2", False)]))
        await asyncio.wait_for(starting, GUARD)
        await asyncio.wait_for(ran, GUARD)
        reply.cancel()  # the caller goes away: its reply first, then its withdrawal
        m.post(tl="o", name="withdraw", data=REQUESTS[reply])
        stopped = host.next("run_cancelled", "c1")
        await m.send(tl="c1", name="may_continue", data=m.future())  # c1's run asks before its next step
        await asyncio.wait_for(stopped, GUARD)
        release.set_result(None)
        assert (await asyncio.wait_for(asking, GUARD))[0] == ["c1", "c2"]
        assert await at_rest(m) == "ended"
        assert ("reason", "c1", CANCELLED) in host.log and ("reason", "c2", CANCELLED) in host.log
        assert host.logged(prefix=("run", "c2")) == [] and host.logged(prefix=("wake",)) == []
        await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), GUARD)
        await m.stop()
    run(go)


@pytest.mark.parametrize("reply_cancelled", [True, False])
def test_a_withdrawal_mid_spawn_reaches_the_focus_graft_made_after_it(monkeypatch, reply_cancelled):
    """A background graft and a focus graft; the request is withdrawn while the background graft's
    machine is started, before the focus graft is made. However the withdrawal reaches the focus
    graft (its request's reply already cancelled, as a cancelled `user_says` does, or by id alone,
    forwarded once the grafts are made), the focus graft never answers it: its run is ended or
    never started, and the request's written row is marked once, in its view and the original's."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        row = host.log.index(("request", "o", "go")) + 1
        starting, release = _held_start_of(monkeypatch=monkeypatch, child="c1")
        asking = asyncio.ensure_future(clone(m=m, pairs=[("c1", False), ("c2", True)]))
        await asyncio.wait_for(starting, GUARD)
        if reply_cancelled:
            reply.cancel()
        m.post(tl="o", name="withdraw", data=REQUESTS[reply])
        release.set_result(None)
        _, focus, _ = await asyncio.wait_for(asking, GUARD)
        assert focus == "c2" and await at_rest(m=m, tl="c2") == "ended"
        assert host.logged(prefix=("withdrawn",)) == [("withdrawn", ("c2", "o"), [row])]
        if reply_cancelled:
            assert host.logged(prefix=("run", "c2")) == []  # it never started a run
        else:
            assert host.logged(prefix=("run_cancelled", "c2")) == [("run_cancelled", "c2")]  # its run was ended
        await m.stop()
    run(go)


def test_a_graft_that_joined_at_once_in_a_clone_call_undone_at_its_start_is_never_applied(monkeypatch):
    """The call is withdrawn as it commits, so its first graft joins `cancelled` at once; its
    second graft's machine cannot be started, so the call is undone and refused. The first
    graft's join, held for the step, is then not applied to the original, and settle returns."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        _withdraws_inside(host=host, m=m, method="commit_clone", reply=reply)
        starting, release = _held_start_of(monkeypatch=monkeypatch, child="c2", fails=True)
        asking = asyncio.ensure_future(clone(m=m, pairs=[("c1", False), ("c2", False)]))
        await asyncio.wait_for(starting, GUARD)
        release.set_result(None)
        assert (await asyncio.wait_for(asking, GUARD))[0] is None  # refused: the call was undone
        assert await at_rest(m) == "ended"
        assert host.logged(prefix=("apply",)) == [] and ("abandoned", ("c1", "c2")) in host.log
        assert host.logged(prefix=("run", "c1")) == [] and host.logged(prefix=("wake",)) == []
        await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), GUARD)
        await m.stop()
    run(go)


def test_messages_waiting_on_the_original_go_to_its_focus_graft_in_order():
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        second = await say(m=m, text="B")  # waits on the original while it runs
        await clone(m=m, pairs=[("c1", True)])
        third = await say(m=m, text="C")  # forwarded to the focus graft
        assert ("request", "c1", "B") not in host.log
        ran = host.next("request", "o", "B")
        host.end("c1")  # its turn ends in its join: B and C come back to the original, in order
        await asyncio.wait_for(ran, GUARD)  # a waiting message is answered by its own turn, joins in view
        assert host.logged(prefix=("request", "c1")) == []
        host.end("o")
        await asyncio.wait_for(host.next("request", "o", "C"), GUARD)
        host.end("o")
        assert await first == await second == await third == "ended"
        assert [e[2] for e in host.logged(prefix=("request", "o"))] == ["go", "B", "C"]
        await m.stop()
    run(go)


@pytest.mark.parametrize("by_id_only", [False, True])
def test_a_waiting_message_withdrawn_while_a_hand_over_is_made_is_answered_and_settle_still_raises(by_id_only):
    """A (the turn's request) and B (waiting) are withdrawn while the clone call that hands the
    conversation to c1 commits; C waits too. c1 starts at rest (its request was withdrawn), drops
    B and takes C. B is answered at once and leaves the session's count of waiting calls, so a
    later failure is still kept for, and raised by, settle."""
    async def go():
        host, m = await started()
        a = await say(m=m, text="A")
        b = await say(m=m, text="B")  # waits on o
        real = host.commit_clone

        def commit(**kw):  # both callers go away while the clone call commits
            if not by_id_only:
                a.cancel()
                b.cancel()
            m.post(tl="o", name="withdraw", data=REQUESTS[a])
            m.post(tl="o", name="withdraw", data=REQUESTS[b])
            return real(**kw)
        host.commit_clone = commit
        c = await say(m=m, text="C")  # waits on o too
        ran = host.next("run", "c1")
        _, focus, _ = await clone(m=m, pairs=[("c1", True)])
        assert focus == "c1"
        await asyncio.wait_for(ran, GUARD)
        assert host.logged(prefix=("request", "c1")) == [("request", "c1", "C")]  # B was dropped, C taken
        if by_id_only:
            assert await asyncio.wait_for(b, GUARD) == "ended"
        waking = host.next("run", "o", "wake")
        host.end("c1")  # its turn on C ends in its join: the original runs once after it
        await asyncio.wait_for(waking, GUARD)
        host.end("o")
        assert await asyncio.wait_for(c, GUARD) == "ended"
        assert await at_rest(m) == "ended"
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c2", False)])
        host.end("o")
        assert await first == "ended"
        settling = m.future()
        await m.send(tl="session", name="settle_request", data=settling)
        waking = host.next("run", "o", "wake")
        host.end("c2")
        await asyncio.wait_for(waking, GUARD)
        host.fail(tl="o", error=ValueError("wake exploded"))
        with pytest.raises(ValueError, match="wake exploded"):
            await asyncio.wait_for(settling, GUARD)
        await m.stop()
    run(go)


def test_a_withdrawal_of_the_runs_request_and_of_another_id_agree_on_ending_the_run():
    """One rule decides whether a withdrawal ends a run: a withdrawal of the request it answers
    ends it (and the guard asked later, on `may_continue`, agrees); one of another request does
    not, and leaves the run's `may_continue` answered."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="A")
        second = await say(m=m, text="B")  # waits
        await withdraw(m=m, reply=second)
        assert await asyncio.wait_for(second, GUARD) == "ended" and "o" in host.runs  # A's run goes on
        assert await asyncio.wait_for(m.ask(tl="o", name="may_continue", build=lambda r: r), GUARD) is True
        ended = host.next("run_cancelled", "o")
        await withdraw(m=m, reply=first)
        await asyncio.wait_for(ended, GUARD)
        assert await first == "ended" and await at_rest(m) == "ended"
        assert host.logged(prefix=("request", "o")) == [("request", "o", "A")]
        await m.stop()
    run(go)


async def say_as(m, text, id):
    """A user message whose request reuses `id`. The original's record of withdrawals is observed
    by what it decides: the message is dropped unwritten (answered "ended") while that id is
    recorded as withdrawn, and taken (its row written) once it is not."""
    request = Request(id=id, text=text, reply=m.future())
    await m.send(tl="o", name="user_message", data=request)
    return request.reply


async def answered(host, m, reply, tl="o"):
    """A message that was taken: its run is ended and its caller answered."""
    if tl not in host.runs:
        await asyncio.wait_for(host.next("run", tl), GUARD)
    host.end(tl)
    assert await asyncio.wait_for(reply, GUARD) == "ended"


def test_the_withdrawal_record_holds_only_requests_still_held():
    """Withdrawn ids are kept only while their requests are held (or a background graft made for
    one has not joined): once a withdrawn request has left, its id decides nothing, however many
    calls were withdrawn; an id kept for a graft's join is dropped once that join is applied."""
    async def go():
        host, m = await started()
        for n in range(5):
            reply = await say(m=m, text=f"m{n}")
            waiting = await say(m=m, text=f"w{n}")
            await withdraw(m=m, reply=waiting)
            await withdraw(m=m, reply=reply)
            assert await at_rest(m) == "ended"
            await m.send(tl="o", name="withdraw", data=999 + n)  # unknown: recorded nowhere
            for k, gone in enumerate((REQUESTS[reply], REQUESTS[waiting], 999 + n)):
                again = await say_as(m=m, text=f"again {n}.{k}", id=gone)
                assert ("request", "o", f"again {n}.{k}") in host.log, (n, k)  # not read as withdrawn
                await answered(host=host, m=m, reply=again)
        reply = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        host.hold, host.hold_tl = asyncio.get_running_loop().create_future(), "c1"  # c1 stops slowly
        stopping, applied = host.next("run_cancelled", "c1"), host.next("apply", "c1")
        await withdraw(m=m, reply=reply)  # c1, made for it, is stopped at once (it joins cancelled)
        assert await at_rest(m) == "ended"
        await asyncio.wait_for(stopping, GUARD)
        kept = await say_as(m=m, text="kept", id=REQUESTS[reply])  # c1 was made for it and has not joined
        assert await asyncio.wait_for(kept, GUARD) == "ended" and not host.logged(prefix=("request", "o", "kept"))
        host.hold.set_result(None)  # its run ends now, and it joins
        await asyncio.wait_for(applied, GUARD)
        assert await at_rest(m) == "ended"
        assert host.logged(prefix=("wake",)) == []  # its join wakes nothing: its request was withdrawn
        again = await say_as(m=m, text="after the join", id=REQUESTS[reply])
        assert ("request", "o", "after the join") in host.log  # the id was dropped with the join
        await answered(host=host, m=m, reply=again)
        await m.stop()
    run(go)


def test_a_suspended_original_records_no_withdrawal_while_its_focus_graft_holds_the_conversation():
    """While a focus graft holds the conversation the original holds no request: a withdrawal
    forwarded through it is recorded only by the graft that holds the request, which drops the
    waiting message at once, so neither record keeps the id: a later message reusing it is taken
    by the graft, and by the original once the conversation is back."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", True)])
        ids = []
        for n in range(4):
            waiting = await say(m=m, text=f"w{n}")  # forwarded to c1, which is still running
            await withdraw(m=m, reply=waiting)
            assert await asyncio.wait_for(waiting, GUARD) == "ended"
            await m.send(tl="o", name="withdraw", data=999 + n)  # unknown: recorded nowhere
            ids.append(REQUESTS[waiting])
        later = [await say_as(m=m, text=f"again {n}", id=i) for n, i in enumerate(ids)]  # wait on c1
        host.end("c1")  # its turn ends in its join: the messages waiting on it come back to the original

        async def taken(n):
            while not (host.logged(prefix=("request", "o", f"again {n}")) and "o" in host.runs):
                await asyncio.sleep(0.005)
        for n in range(len(later)):  # the original takes each in turn: none of the ids is recorded
            await asyncio.wait_for(taken(n), GUARD)
            host.end("o")
        for reply in [first, *later]:  # answered once the conversation is back at rest
            assert await asyncio.wait_for(reply, GUARD) == "ended"
        assert await at_rest(m) == "ended"
        assert host.logged(prefix=("request", "c1")) == []
        back = await say_as(m=m, text="back", id=ids[0])
        assert ("request", "o", "back") in host.log  # the original never recorded it either
        await answered(host=host, m=m, reply=back)
        await m.stop()
    run(go)


# ------------------------------------------------------------------ a clone call cut short in its preparation
class Preparing(Host):
    """A host whose preparation never ends on its own; its rollback may fail."""
    rollback_error: BaseException | None = None

    async def prepare_clone(self, src, clone):
        self.record("prepare", src)
        await asyncio.get_running_loop().create_future()

    def roll_back_clone(self, src):
        self.record("roll_back", src)
        if self.rollback_error is not None:
            raise self.rollback_error


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("how", ["driven turn ended", "request withdrawn"])
def test_a_clone_call_cut_short_in_its_preparation_is_rolled_back_in_its_own_state(how, fails):
    async def go():
        host = Preparing()
        host.rollback_error = OSError("rollback failed") if fails else None
        m = Machines(host)
        await m.start_root("o")
        if how == "driven turn ended":
            await driven_turn(m)
        else:
            reply = await say(m=m, text="go")
        asking = asyncio.ensure_future(clone(m=m, pairs=[("c1", False)]))
        await asyncio.wait_for(host.next("prepare", "o"), GUARD)
        if how == "driven turn ended":
            await end_driven_turn(m)
        else:
            await withdraw(m=m, reply=reply)
        assert (await asyncio.wait_for(asking, GUARD))[0] is None  # the call is answered: not executed
        assert await at_rest(m) == "ended" and ("roll_back", "o") in host.log
        assert bool(host.logged(prefix=("close", "o"))) == (how == "request withdrawn")  # a driven turn's caller closes its own
        settling = m.ask(tl="session", name="settle_request", build=lambda r: r)
        if fails:
            assert ("task", [], "rollback failed") in host.reports
            with pytest.raises(OSError, match="rollback failed"):
                await asyncio.wait_for(settling, GUARD)
        else:
            assert await asyncio.wait_for(settling, GUARD) is None
        await m.stop()
    run(go)


# ------------------------------------------------------------------ close
def test_a_machine_that_cannot_be_stopped_is_reported_and_the_rest_are_still_stopped(monkeypatch):
    import graft.machine as machine_module
    real = machine_module.hsm.Stop

    def stop(sm, ctx=None):
        if getattr(sm, "timeline_id", None) == "o":
            async def fails():
                raise RuntimeError("cannot stop o")
            return fails()
        return real(sm, ctx)

    async def go():
        host, m = await started()
        reply = await say(m=m, text="hi")
        host.end("o")
        await reply
        monkeypatch.setattr(machine_module.hsm, "Stop", stop)
        with pytest.raises(RuntimeError, match="cannot stop o"):
            await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), GUARD)
        monkeypatch.setattr(machine_module.hsm, "Stop", real)
        assert ("machine", [], "cannot stop o") in host.reports
        assert not m.scribe.state()  # stopped although o could not be
        await m.stop()
    run(go)


def test_every_stop_failure_is_told_when_telling_one_raises_and_close_raises_that_too(monkeypatch):
    """Two machines cannot be stopped and the user's callback raises while the first is told:
    the second is still told, and close raises both stop failures and the callback's error."""
    import graft.machine as machine_module
    real = machine_module.hsm.Stop
    callback_error = RuntimeError("emit failed")

    class Reporter(Host):
        def report(self, kind, src, clones, error, quiet, background):
            super().report(kind=kind, src=src, clones=clones, error=error, quiet=quiet, background=background)
            if len(self.reports) == 1:
                raise callback_error

    async def go():
        host = Reporter()
        m = Machines(host)
        await m.start_root("o")

        def stop(sm, ctx=None):
            if getattr(sm, "timeline_id", None) == "o" or sm is m.scribe:
                async def fails():
                    raise RuntimeError(f"cannot stop {getattr(sm, 'timeline_id', 'scribe')}")
                return fails()
            return real(sm, ctx)

        reply = await say(m=m, text="hi")
        host.end("o")
        await reply
        monkeypatch.setattr(machine_module.hsm, "Stop", stop)
        with pytest.raises(ExceptionGroup) as raised:
            await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), GUARD)
        monkeypatch.setattr(machine_module.hsm, "Stop", real)
        assert [r for r in host.reports if r[0] == "machine"] == [
            ("machine", [], "cannot stop o"), ("machine", [], "cannot stop scribe")]
        assert [str(e) for e in raised.value.exceptions] == ["cannot stop o", "cannot stop scribe", "emit failed"]
        await m.stop()
    run(go)


def test_the_original_stays_open_until_a_late_grafts_join_is_applied(monkeypatch):
    """Close lands while a clone call is in flight: its graft is made after closing began and is
    shut down late (its run is slow to end). The original, shut down meanwhile, waits at rest for
    that graft's join, applies it, and only then closes."""
    import graft.machine as machine_module
    real = machine_module.hsm.Started

    class SlowGraft(Host):
        async def _run(self, tl):  # only c1's run waits for `hold` once it is ended
            self.runs[tl] = asyncio.get_running_loop().create_future()
            try:
                return await self.runs[tl]
            except asyncio.CancelledError:
                self.record("run_cancelled", tl)
                if tl == "c1" and self.hold is not None:
                    await self.hold
                raise

    async def go():
        host = SlowGraft()
        m = Machines(host)
        await m.start_root("o")
        await say(m=m, text="go")
        starting, release = m.future(), m.future()

        async def held_start(ctx, sm, model, config=None):
            if getattr(sm, "timeline_id", None) == "c1":
                starting.set_result(None)
                await release
            return await real(ctx, sm, model, config)

        monkeypatch.setattr(machine_module.hsm, "Started", held_start)
        asking = asyncio.ensure_future(clone(m=m, pairs=[("c1", False)]))
        await asyncio.wait_for(starting, GUARD)  # the clone call is in flight
        closing = asyncio.ensure_future(m.ask(tl="session", name="shutdown", build=lambda r: r))
        host.hold = m.future()  # c1's run, once shut down, is slow to end
        stopping = host.next("run_cancelled", "c1")
        release.set_result(None)
        await asyncio.wait_for(stopping, GUARD)
        await asyncio.wait_for(asking, GUARD)
        with contextlib.suppress(TransitionRefused):
            await at_rest(m)  # the original has done what it can before c1's join comes
        host.hold.set_result(None)
        await asyncio.wait_for(closing, GUARD)
        assert ("apply", "c1", "failed") in host.log and host.status["c1"] == "joined"
        await m.stop()
    run(go)


# ------------------------------------------------------------------ outcomes do not depend on declaration order
ORDERS = ["declared", "reversed"]


def _in_order(monkeypatch, order):
    """Install the timeline graph for the machines this test makes: as declared, or rebuilt with
    the guarded branches of every choice and every state in the reverse of the order they are
    declared in (each unguarded default still after them). A scenario that behaves the same both
    ways has an outcome that no declaration order decides."""
    import hsm
    import graft.machine as gm
    if order == "declared":
        return

    def guarded(element):
        return type(element).__name__ == "RedefinableTransition" and any(
            type(e).__name__ == "RedefinableConstraint" for e in element.owned_elements)

    def reorder(elements):
        others = [e for e in elements if type(e).__name__ != "RedefinableTransition"]
        branches = [e for e in elements if guarded(element=e)]
        defaults = [e for e in elements if type(e).__name__ == "RedefinableTransition" and not guarded(element=e)]
        return [*others, *reversed(branches), *defaults]

    builders = {name: getattr(hsm, name) for name in ("Define", "State", "Choice")}

    def rebuilt(build):
        def make(name, *elements):
            return build(name, *reorder(elements=elements))
        return make

    for name, build in builders.items():
        monkeypatch.setattr(hsm, name, rebuilt(build=build))
    model = gm._timeline("Timeline")
    for name, build in builders.items():
        monkeypatch.setattr(hsm, name, build)
    monkeypatch.setattr(gm, "TIMELINE", model)


@pytest.mark.parametrize("order", ORDERS)
def test_a_withdrawn_hand_back_wakes_nothing_whatever_order_the_branches_are_declared_in(monkeypatch, order):
    """A focus graft hands its request back, and the request is withdrawn as it does: after the
    joins, no run answers it (its row is marked instead), with the
    branches of the choices in either order."""
    async def go():
        _in_order(monkeypatch=monkeypatch, order=order)
        host, m = await started()
        reply = await say(m=m, text="go")
        row = host.log.index(("request", "o", "go")) + 1
        real = host.join_of

        def join_of(tl, reason, focus=False):  # the caller goes away as the graft hands the request back
            reply.cancel()
            m.post(tl="o", name="withdraw", data=REQUESTS[reply])
            return real(tl=tl, reason=reason, focus=focus)
        host.join_of = join_of
        host.join_reasons["c1"] = "out_of_scope"
        await clone(m=m, pairs=[("c1", True)])
        host.end(tl="c1", result=RunOutcome(kind="join", reason="done"))
        await asyncio.wait_for(host.next("apply", "c1"), GUARD)
        assert await at_rest(m) == "ended"
        assert host.logged(prefix=("wake",)) == [] and host.logged(prefix=("run", "o")) == [("run", "o", "turn")]
        assert host.logged(prefix=("withdrawn",)) == [("withdrawn", ("o",), [row])]
        await m.stop()
    run(go)


@pytest.mark.parametrize("order", ORDERS)
def test_a_message_whose_caller_has_gone_starts_no_join_commit_whatever_order_the_branches_are_declared_in(
        monkeypatch, order):
    """The original is at rest with a join it could not commit; a message whose caller has gone
    arrives: it is dropped at once, and no commit point is opened for it, with the branches of the
    choices in either order."""
    async def go():
        _in_order(monkeypatch=monkeypatch, order=order)
        host, m = await started()
        await clone_in_turn_then_fail_join(host=host, m=m)
        reports = list(host.reports)
        request = m.request("gone")
        request.reply.cancel()  # its caller went away; its withdrawal has not arrived
        await m.send(tl="o", name="user_message", data=request)
        assert await at_rest(m) == "ended"
        assert host.reports == reports and not host.logged(prefix=("request", "o", "gone"))
        await m.stop()
    run(go)


@pytest.mark.parametrize("order", ORDERS)
def test_a_call_refused_in_its_preparation_gets_the_harness_refusal_whatever_order_the_branches_are_declared_in(
        monkeypatch, order):
    """The harness refuses a clone call while preparing it (no grafts were made): its result is
    that refusal, never the graph's own, with the branches of the choices in either order."""
    class Refusing(Host):
        async def prepare_clone(self, src, clone):
            self.record("prepare", src)
            return "refused by the harness"

    async def go():
        _in_order(monkeypatch=monkeypatch, order=order)
        host = Refusing()
        m = Machines(host)
        await m.start_root("o")
        reply = await say(m=m, text="go")
        await clone(m=m, pairs=[])
        assert host.logged(prefix=("result", "o", "k")) == [("result", "o", "k", "refused by the harness", True)]
        host.end("o")
        assert await reply == "ended"
        await m.stop()
    run(go)


# ------------------------------------------------------------------ a behavior's events: one, or its fault
def _counting(sm, names, seen=None):
    """Record every event sent to sm's machine whose name is in `names` (into `seen`, if given)."""
    real, seen = sm.dispatch, ([] if seen is None else seen)

    def dispatch(ctx, event):
        if event.name in names:
            seen.append(event.name)
        return real(ctx, event)
    sm.dispatch = dispatch
    return seen


@pytest.mark.parametrize("how", ["sends, then raises", "sends twice"])
def test_a_behavior_that_fails_sends_nothing_but_its_one_fault(monkeypatch, how):
    """An effect's sends are made when it returns. One that raises after its send, or that sends
    twice, sends none of them: its one event is the fault (BH-06), and the machine recovers."""
    import graft.machine as machine_module
    real, armed = machine_module.come_to_rest, [True]

    def come_to_rest(ctx, inst, event):
        if not armed:
            return real(ctx=ctx, inst=inst, event=event)
        armed.clear()
        machine_module._session(inst=inst, name="rested", data=(inst.timeline_id, []))
        if how == "sends, then raises":
            raise RuntimeError("after the send")
        machine_module._session(inst=inst, name="rested", data=(inst.timeline_id, []))

    async def go():
        host, m = await started()
        reply = await say(m=m, text="hi")
        seen = _counting(sm=m.session, names={"rested"})
        _counting(sm=m.timelines["o"], names={machine_module.FAULT, machine_module.CALL_FAULT}, seen=seen)
        monkeypatch.setattr(machine_module, "come_to_rest", come_to_rest)
        host.end("o")
        await asyncio.wait_for(host.next("recover", "o"), GUARD)
        # the failed step's event is its fault alone; the one `rested` is the recovery's, at rest
        assert seen == [machine_module.FAULT, "rested"], seen
        assert await asyncio.wait_for(reply, GUARD) == "ended"
        await m.stop()
    run(go)


def test_no_generated_behavior_shadows_a_rule_of_the_module():
    """Guards and behaviors are read through the module by name when they run (`_rule`): a
    generated one (a closure) must not share its name with another definition there."""
    import graft.machine as machine_module
    for model in (machine_module.TIMELINE, machine_module.BACKGROUND, machine_module.SCRIBE, machine_module.SESSION):
        for element in model.members.values():
            run_ = getattr(element, "operation", None) or getattr(element, "expression", None)
            inner = getattr(run_, "__wrapped__", None)
            if inner is not None:
                found = vars(machine_module).get(inner.__name__)
                assert found is None or found is inner, inner.__name__


def test_a_raising_guard_of_a_join_from_a_graft_it_does_not_hold_is_reported_and_drops_it(monkeypatch):
    """A join from a graft the original does not hold (one of a clone call undone at its start)
    reaches it while the guard that tells it so raises: no branch holds, the safe branch keeps the
    join and reports the broken decision, and the recovery drops it (it is not one of its own)."""
    import graft.machine as machine_module
    real, armed = machine_module.foreign_graft, [True]

    def raising(**kw):
        if armed:
            armed.clear()
            raise RuntimeError("guard bug")
        return real(**kw)
    monkeypatch.setattr(machine_module, "foreign_graft", raising)

    async def go():
        host, m = await started()
        told = host.next("report", "machine")
        await m.send(tl="o", name="graft_queued", data=QueuedJoin(child="cX", reason="done", trigger=None))
        assert "foreign_graft held" in (await asyncio.wait_for(told, GUARD))[3]
        assert host.logged(prefix=("apply",)) == [] and await at_rest(m) == "ended"
        await m.stop()
    run(go)


def test_a_raising_guard_of_a_held_graft_s_start_stops_it_and_reports(monkeypatch):
    """A background graft made in a turn its caller drives starts `held`. If that guard raises it
    does not hold: the safe branch stops the graft (it never runs, and joins failed) and reports
    the broken decision."""
    import graft.machine as machine_module
    real, armed = machine_module.is_held, [True]

    def raising(**kw):
        if armed:
            armed.clear()
            raise RuntimeError("guard bug")
        return real(**kw)
    monkeypatch.setattr(machine_module, "is_held", raising)

    async def go():
        host, m = await started()
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))
        told = host.next("report", "task")
        await clone(m=m, pairs=[("c1", False)])
        assert "is_held held" in (await asyncio.wait_for(told, GUARD))[3]
        assert host.logged(prefix=("run", "c1")) == [] and ("reason", "c1", "failed") in host.log
        await m.stop()
    run(go)


def test_a_raising_recovery_guard_is_reported_and_its_fault_handled_again(monkeypatch):
    """The guard that lets a timeline recover raises (in the step that handles a fault): it does
    not hold, so the timeline gives up, and the safe branch reports the decision the raise changed
    (a DecisionFault, part of the fault it handles); the fault is handled again, and recovered
    from. Both are told together, once, and the timeline goes on."""
    import graft.machine as machine_module
    real, armed = machine_module.may_recover, [True]

    def raising(**kw):
        if armed:
            armed.clear()
            raise RuntimeError("guard bug")
        return real(**kw)
    monkeypatch.setattr(machine_module, "may_recover", raising)

    async def go():
        host, m = await started()
        errors, _ = _session_sends_fail(m=m, names=["rested"])
        reply = await say(m=m, text="hi")
        host.end("o")
        assert await asyncio.wait_for(reply, GUARD) == "ended"
        with pytest.raises(ExceptionGroup) as kept:
            await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), GUARD)
        rested, decision = kept.value.exceptions
        assert rested is errors["rested"] and isinstance(decision, machine_module.DecisionFault)
        assert "may_recover held" in str(decision)
        assert len(host.logged(prefix=("recover", "o"))) == 1  # handled again: closed out once
        nxt = await say(m=m, text="two")
        host.end("o")
        assert await asyncio.wait_for(nxt, GUARD) == "ended"
        await m.stop()
    run(go)


def _mark_fails_once(host):
    real, failed = host.mark_withdrawn, []

    def mark_withdrawn(timelines, rows):
        if not failed:
            failed.append(rows)
            raise OSError("cannot mark")
        real(timelines=timelines, rows=rows)
    host.mark_withdrawn = mark_withdrawn
    return failed


def test_a_row_still_to_be_marked_is_marked_before_a_driven_turn_s_clone_call_reads_the_view():
    """The mark failed; the next reader is a clone call in a turn its caller drives (no run starts
    it): the call's preparation marks the row before it resolves any seed."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        row = host.log.index(("request", "o", "go")) + 1
        failed = _mark_fails_once(host=host)
        await withdraw(m=m, reply=first)
        await at_rest(m)
        assert failed == [[row]]
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))
        await clone(m=m, pairs=[("c1", False)])
        marked = host.log.index(("withdrawn", ("o",), [row]))
        assert marked < host.log.index(("prepare", "o"))
        await m.stop()
    run(go)


def test_a_row_a_focus_graft_could_not_mark_is_marked_before_its_join_is_copied():
    """A focus graft's run is cut short by a withdrawal and the row's mark fails: the row goes back
    with its join, and the original marks it before the join copies the graft's rows into its view."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        row = host.log.index(("request", "o", "go")) + 1
        await clone(m=m, pairs=[("c1", True)])  # the original is suspended; c1 runs
        failed = _mark_fails_once(host=host)
        await withdraw(m=m, reply=first)  # reaches the focus graft: its run ends, the mark fails
        assert await at_rest(m=m, tl="c1") == "ended" and failed == [[row]]
        applied, waking = host.next("apply", "c1"), host.next("run", "o", "wake")
        joining = asyncio.ensure_future(m.ask(tl="c1", name="join", build=lambda r: (
            QueuedJoin(child="c1", reason="done", trigger=None), r)))
        await asyncio.wait_for(applied, GUARD)
        assert host.log.index(("withdrawn", ("o",), [row])) < host.log.index(("apply", "c1", "done"))
        await asyncio.wait_for(waking, GUARD)  # the original runs once after the merge
        host.end("o")
        await asyncio.wait_for(joining, GUARD)
        await m.stop()
    run(go)


# ---------------------------------------------------------------- the graphs' shape
def _cell(fn, name):
    for free, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if free == name:
            return cell.cell_contents
    return None


def _guard_name(model, guard):
    expression = model.members[guard].expression
    return (getattr(expression, "__wrapped__", None) or _cell(fn=expression, name="expression") or expression).__name__


def _behavior(model, effect):
    operation = getattr(model.members[effect], "operation", None)
    return (_cell(fn=operation, name="behavior") or operation) if operation is not None else None


def test_every_decision_has_one_safe_branch_that_re_checks_exactly_its_guards():
    """In all four graphs, every choice and every group of one event's transitions from one state
    that has a guarded branch has exactly one unguarded branch at the same level, whose effect
    re-checks exactly the guards of that decision (`checked`). So a guard that raises never lets
    the decision fall through to an ancestor's transitions, and the decision it changed is
    reported."""
    import collections
    import hsm
    import graft.machine as machine_module
    problems = []
    for name in ("TIMELINE", "BACKGROUND", "SESSION", "SCRIBE"):
        model = getattr(machine_module, name)
        groups = collections.defaultdict(list)
        for element in model.members.values():
            if isinstance(element, hsm.hsm.TransitionElement):
                groups[(element.source, tuple(sorted(element.events)))].append(element)
        for (source, events), transitions in sorted(groups.items()):
            guards = {_guard_name(model=model, guard=t.guard) for t in transitions if t.guard}
            if not guards:
                continue
            defaults = [t for t in transitions if not t.guard]
            if len(defaults) != 1:
                problems.append(f"{name} {source} {events}: {len(defaults)} unguarded branches for {sorted(guards)}")
                continue
            behaviors = [_behavior(model=model, effect=e) for e in defaults[0].effect]
            rechecked = [{g.__name__ for g in _cell(fn=b, name="guarded")} for b in behaviors
                         if b is not None and b.__name__.startswith("checked_")]
            if set().union(*rechecked) != guards:
                problems.append(f"{name} {source} {events}: guards {sorted(guards)}, re-checked "
                                f"{sorted(set().union(*rechecked))}")
    assert problems == []


def _reads(function_name):
    import ast
    import inspect
    import graft.machine as machine_module
    tree = ast.parse(inspect.getsource(getattr(machine_module, function_name)).lstrip())
    return tree.body[0]


def test_no_behavior_or_activity_decides_by_whether_its_step_failed():
    """Whether the step in progress has failed is read only by the plumbing (guards, the safe
    branch's re-check, the step's sends, the activities' hold-back) and by the guards that let a
    decision tell a failed step apart (`step_intact`); the safe branch a failed step takes names
    its error (`_pending`). No effect, entry, exit or activity reads the pending fault to choose
    what is sent, answered or done next."""
    import ast
    import inspect
    import graft.machine as machine_module
    allowed = {"_failed", "_held_back", "_take_fault", "_pending", "_keep_fault", "_signal", "_handling_failed",
               "_guard", "_holds", "_step", "Activity", "io", "step_intact", "decline_failed_step",
               "spawn_not_started"}
    readers = []
    for name, function in vars(machine_module).items():
        if not inspect.isfunction(function) or function.__module__ != machine_module.__name__ or name in allowed:
            continue
        for node in ast.walk(_reads(name)):
            reads_fault = isinstance(node, ast.Attribute) and node.attr == "fault" and isinstance(node.ctx, ast.Load)
            calls = isinstance(node, ast.Name) and node.id in ("_failed", "_pending", "_held_back")
            if reads_fault or calls:
                readers.append(name)
                break
    assert readers == []


def test_no_activity_reads_its_machine_s_data_after_it_first_awaits():
    """An activity takes its inputs before its first await (inside the step that enters its state):
    after it, it only dispatches its outcome (AC-06)."""
    import ast
    import inspect
    import graft.machine as machine_module
    late = []
    for name, function in vars(machine_module).items():
        if not inspect.iscoroutinefunction(function) or function.__module__ != machine_module.__name__:
            continue
        body = _reads(name)
        awaits = [n for n in ast.walk(body) if isinstance(n, ast.Await)]
        loops = [loop for loop in ast.walk(body) if isinstance(loop, (ast.For, ast.While))
                 and any(isinstance(a, ast.Await) for a in ast.walk(loop))]  # a body that awaits runs again
        looped = {id(n) for loop in loops for statement in loop.body for n in ast.walk(statement)}
        for node in ast.walk(body):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "inst"):
                continue
            if node.attr in ("dispatch", "context"):
                continue
            after = any((node.lineno, node.col_offset) > (a.end_lineno, a.end_col_offset) for a in awaits)
            if after or id(node) in looped:
                late.append(f"{name}: inst.{node.attr} (line {node.lineno})")
        # nor through a name bound to its data before the await: a container of the machine's
        # data read or written by subscript after it is the machine's data all the same
        aliases = {target.id for n in ast.walk(body) if isinstance(n, ast.Assign)
                   for target in (t for t in n.targets for t in (t.elts if isinstance(t, ast.Tuple) else [t]))
                   if isinstance(target, ast.Name)}
        bound = set()
        for n in ast.walk(body):
            if isinstance(n, ast.Assign):
                values = n.value.elts if isinstance(n.value, ast.Tuple) else [n.value]
                targets = [t for t in n.targets for t in (t.elts if isinstance(t, ast.Tuple) else [t])]
                for target, value in zip(targets, values):
                    if (isinstance(target, ast.Name) and isinstance(value, ast.Attribute)
                            and isinstance(value.value, ast.Name) and value.value.id == "inst"):
                        bound.add(target.id)
        for node in ast.walk(body):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in bound
                    and node.value.id in aliases
                    and any((node.lineno, node.col_offset) > (a.end_lineno, a.end_col_offset) for a in awaits)):
                late.append(f"{name}: {node.value.id}[...] (line {node.lineno}), the machine's data")
    assert late == []


def test_a_withdrawal_stops_every_graft_made_for_it_even_when_one_machine_is_gone():
    """The stops go one per step; a graft whose machine is gone is skipped, and the next is stopped."""
    async def go():
        host, m = await started()
        reply = await say(m=m, text="go")
        await clone(m=m, pairs=[("g0", False), ("g1", False), ("g2", False)])
        stopped = host.next("run_cancelled", "g2")
        m.timelines.pop("g0")  # forgotten: a stop sent to it is lost
        await withdraw(m=m, reply=reply)
        await asyncio.wait_for(stopped, GUARD)
        assert host.logged(prefix=("run_cancelled", "g1")) and host.logged(prefix=("run_cancelled", "g2"))
        await m.stop()
    run(go)


def test_a_background_run_that_fails_while_its_tool_waits_is_taken_as_the_run_s_failure():
    """The run's failure is its own event: a `calling` state never takes it for its tool's."""
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        await clone(m=m, pairs=[("g0", False)])
        if "g0" not in host.runs:
            await asyncio.wait_for(host.next("run", "g0"), GUARD)
        host.tool_gate = asyncio.get_running_loop().create_future()
        calling = host.next("tool", "g0")
        m.post(tl="g0", name="tool_request", data=ToolCall(use={"id": "t1", "name": "x", "input": {}},
                                                           reply=m.future(), interrupted="cut short"))
        await asyncio.wait_for(calling, GUARD)
        recorded = host.next("reason", "g0")
        host.fail("g0", RuntimeError("run down"))
        assert await asyncio.wait_for(recorded, GUARD) == ("reason", "g0", "failed")
        assert host.logged(prefix=("close", "g0")) == [("close", "g0", {"t1": ("cut short", True)})]
        await m.stop()
    run(go)


def test_a_background_run_failure_that_comes_after_the_graft_stopped_is_told():
    """A run's failure that lands once the graft has left `background` (it was stopped meanwhile)
    is told, once, not dropped."""
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        await clone(m=m, pairs=[("g0", False)])
        if "g0" not in host.runs:
            await asyncio.wait_for(host.next("run", "g0"), GUARD)
        queued = m.ask(tl="g0", name="when_queued", build=lambda r: r)
        await cancel(m=m, tl="o", ids=["g0"])
        await asyncio.wait_for(queued, GUARD)
        told = host.next("report")
        m.post(tl="g0", name="run_failed", data=RuntimeError("late failure"))
        assert (await asyncio.wait_for(told, GUARD))[3] == "late failure"
        await m.stop()
    run(go)


def test_a_focus_graft_whose_turn_a_fault_ended_joins_failed(monkeypatch):
    """A focus graft holds the conversation for one turn and never rests after one (spec: Internal
    faults): a fault in the step that ends its turn is recovered from, and the recovery goes on to
    its join (`resumed` -> `handing_back`, reason failed), so its source takes the conversation
    back. Its caller already has the error, so the source's run after the merge is a wake for no
    request."""
    import graft.machine as machine_module
    real, armed = machine_module.end_run, [True]

    def end_run(ctx, inst, event):
        if inst.timeline_id == "c1" and armed:
            armed.clear()
            raise RuntimeError("end_run bug")
        return real(ctx=ctx, inst=inst, event=event)

    async def go():
        monkeypatch.setattr(machine_module, "end_run", end_run)
        host, m = await started()
        reply = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", True)])
        applied, waking = host.next("apply", "c1"), host.next("run", "o", "wake")
        host.end("c1")
        assert (await asyncio.wait_for(applied, GUARD))[2] == "failed"
        with pytest.raises(RuntimeError, match="end_run bug"):
            await asyncio.wait_for(reply, GUARD)
        await asyncio.wait_for(waking, GUARD)
        host.end("o")
        assert await at_rest(m) == "ended" and holders(host) == ["o"] and host.status["c1"] == "joined"
        await m.stop()
    run(go)


def test_a_focus_graft_that_never_ran_rests_after_a_fault_and_does_not_join(monkeypatch):
    """A fault while a focus graft never ran (its request was withdrawn before it could act) ended no
    turn: the recovery leaves it at rest holding the conversation, as without the fault, and no run
    of its source starts."""
    import graft.machine as machine_module
    real, armed = machine_module.rows_marked, [True]

    def rows_marked(ctx, inst, event):
        if inst.timeline_id == "c1" and armed:
            armed.clear()
            raise RuntimeError("rows_marked bug")
        return real(ctx=ctx, inst=inst, event=event)

    async def go():
        monkeypatch.setattr(machine_module, "rows_marked", rows_marked)
        host, m = await started()
        first = await say(m=m, text="A")
        _withdraws_inside(host=host, m=m, method="commit_clone", reply=first)
        recovered = host.next("recover", "c1")
        await clone(m=m, pairs=[("c1", True)])
        await asyncio.wait_for(recovered, GUARD)
        assert await at_rest(m=m, tl="c1") == "ended" and host.status["c1"] == "active"
        assert not host.logged(prefix=("apply", "c1")) and not host.logged(prefix=("run", "o", "wake"))
        await m.stop()
    run(go)


def test_a_focus_graft_whose_turn_a_withdrawal_ended_rests_after_a_fault(monkeypatch):
    """The withdrawal ended its turn, not the fault: a fault while that run's close-out is written
    leaves it at rest, as without the fault, and no run of its source starts."""
    import graft.machine as machine_module
    real, armed = machine_module.rows_marked, [True]

    def rows_marked(ctx, inst, event):
        if inst.timeline_id == "c1" and armed:
            armed.clear()
            raise RuntimeError("rows_marked bug")
        return real(ctx=ctx, inst=inst, event=event)

    async def go():
        monkeypatch.setattr(machine_module, "rows_marked", rows_marked)
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", True)])
        recovered = host.next("recover", "c1")
        await withdraw(m=m, reply=first)  # reaches c1: its run ends, its row is marked (that step faults)
        await asyncio.wait_for(recovered, GUARD)
        assert await at_rest(m=m, tl="c1") == "ended" and host.status["c1"] == "active"
        assert not host.logged(prefix=("apply", "c1")) and not host.logged(prefix=("run", "o", "wake"))
        await m.stop()
    run(go)
