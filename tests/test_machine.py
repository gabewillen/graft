"""The state machines on their own, with a scripted host: every request is an event, every answer
comes from the graph. The host does the I/O (here it records it), and a run of the model loop is
an activity the test ends. A test waits on a reply future or on a call the machines make to the
host (`Host.next`), never on loop turns, and asserts on what the host was asked to do."""

import ast
import asyncio

import pytest

from graft.machine import (CANCELLED, MAX_JOIN_ATTEMPTS, MAX_LIVE_BACKGROUND, Cancel, CancelCall, Clone, Prepared,
                           RunOutcome,
                           Machines, Mode, QueuedJoin, TransitionRefused)

GUARD = 5  # seconds: a wait that should end at once fails the test instead of hanging it


class Host:
    """A host that records what the machines ask of it. A run waits until the test ends it.
    `next(*prefix)` is a future for the next call whose log entry starts with `prefix`."""

    def __init__(self):
        self.log: list[tuple] = []
        self.status: dict[str, str] = {}
        self.runs: dict[str, asyncio.Future] = {}
        self.reports: list[tuple[str, list[str], str]] = []
        self.failing_joins: set[str] = set()
        self.fail_commit = False
        self.hold: asyncio.Future | None = None  # set: a cancelled background run waits for it to end
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

    def logged(self, *prefix) -> list[tuple]:
        return [e for e in self.log if e[:len(prefix)] == prefix]

    async def _run(self, tl):
        self.runs[tl] = asyncio.get_running_loop().create_future()
        try:
            return await self.runs[tl]
        except asyncio.CancelledError:
            self.record("run_cancelled", tl)
            if self.hold is not None:
                await self.hold
            raise

    async def run_busy(self, tl, mode, kind):
        self.record("run", tl, mode.name)
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

    async def prepare_clone(self, src, clone, retry_left):
        self.record("prepare", src, retry_left)
        return Prepared(pairs=clone.pairs, parts=clone.parts, data=None)

    def commit_clone(self, src, clone):
        if self.fail_commit:
            raise OSError("disk full")
        self.record("commit", src, [g for g, _ in clone.pairs])
        return "cloned"

    def refuse_clone(self, src, clone, why, error):
        return f"refused: {why}"

    def store_result(self, tl, use, content, is_error):
        self.record("result", tl, use["id"], content, is_error)

    def close_tool_uses(self, tl, known):
        self.record("close", tl, dict(known))

    def record_join(self, child, reason):
        self.record("reason", child, reason)

    def apply_join(self, src, join, cloned_for, running):
        if join.child in self.failing_joins:
            raise OSError(f"cannot apply {join.child}")
        self.record("apply", join.child, join.reason)
        return True, cloned_for, False

    def wake_note(self, src, joined, routing, running, background):
        self.record("note", tuple(c for c, _ in joined), routing)

    def cancel_text(self, call, outcomes):
        return repr(sorted(outcomes.items())), not any(v == CANCELLED for v in outcomes.values())

    def report(self, kind, src, clones, error, quiet, background):
        self.reports.append((kind, clones, str(error)))
        self.record("report", kind, clones, str(error))

    def graft_failed(self, tl, error):
        self.record("graft_failed", tl, str(error))

    def abandon_clone(self, src, grafts):
        self.record("abandoned", tuple(grafts))

    def join_of(self, tl, reason):
        return QueuedJoin(child=tl, reason=reason, trigger=None)

    def trace(self, text):
        pass


def holders(host) -> list[str]:
    """The timelines holding the conversation, by the states the machines mirrored to the host."""
    return sorted(t for t, status in host.status.items() if status in ("active", "joining"))


async def started():
    host = Host()
    m = Machines(host)
    await m.start_root("o")
    return host, m


async def say(m, text):
    """Put a user message to the original; returns its reply future (answered at rest)."""
    reply = m.future()
    await m.send(tl="o", name="user_message", data=(text, reply))
    return reply


async def clone(m, pairs, parts=None, request=None):
    """A clone call of the original. `parts`: each graft's catalog skills (default: none);
    `request`: the original's current request, as the harness reads it from the store."""
    parts = [(i, frozenset(p)) for i, p in enumerate(parts or [()] * len(pairs))]
    return await m.ask(tl="o", name="clone_request", build=lambda reply: Clone(
        use={"id": "k"}, reply=reply, store=True, pairs=pairs, parts=parts, request=request))


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
        assert host.logged("request") == [("request", "o", "one")]
        ran = host.next("run", "o")
        host.end("o")
        await ran
        assert host.logged("request") == [("request", "o", "one"), ("request", "o", "two")]
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
        assert host.logged("close", "o") and await at_rest(m) == "ended"
        await m.stop()
    run(go)


def test_an_interrupt_ends_the_run_and_closes_its_open_calls():
    async def go():
        host, m = await started()
        reply = await say(m=m, text="hi")
        await m.send(tl="o", name="interrupt", data=reply)
        assert ("run_cancelled", "o") in host.log and host.logged("close", "o")
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
        assert "refused" in text and not host.logged("commit")
        for pairs, refusal in (([("c1", True), ("c2", True)], "at most one graft may take the conversation"),
                               ([(f"g{i}", False) for i in range(MAX_LIVE_BACKGROUND + 1)],
                                f"at most {MAX_LIVE_BACKGROUND} background clones")):
            await driven_turn(m=m)
            _, _, text = await clone(m=m, pairs=pairs)
            assert refusal in text and not host.logged("commit")
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


def test_a_run_tries_one_clone_call_and_one_retry_and_seeds_no_third():
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        _, _, first = await clone(m=m, pairs=[("c1", True), ("c2", True)])  # refused: two focus entries
        _, _, second = await clone(m=m, pairs=[("c1", True), ("c2", True)])  # the one retry: refused again
        _, _, third = await clone(m=m, pairs=[("c1", False)])
        assert [e[2] for e in host.logged("prepare")] == [True, False]  # the retry is the last one prepared
        assert "may take the conversation" in first and "may take the conversation" in second
        assert third == "clone not executed: no clone calls left for this message; answer the user instead"
        assert ("result", "o", "k", third, True) in host.log  # its row, and nothing seeded or committed
        assert not host.logged("commit")
        await m.stop()
    run(go)


def test_a_committed_clone_call_is_the_runs_last():
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        children, _, _ = await clone(m=m, pairs=[("c1", False)])
        _, _, text = await clone(m=m, pairs=[("c2", False)])
        assert children == ["c1"] and text.startswith("clone not executed: no clone calls left")
        assert len(host.logged("prepare")) == 1  # the second was not prepared
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
        assert "only the original clones" in text and not host.logged("prepare", "c1")
        assert "c9" not in host.status
        await m.stop()
    run(go)


def test_a_queued_graft_counts_toward_the_live_cap_until_its_join_is_applied():
    async def go():
        host, m = await started()
        await driven_turn(m=m)
        await clone(m=m, pairs=[(f"g{i}", False) for i in range(MAX_LIVE_BACKGROUND)])
        await end_driven_turn(m=m)
        await driven_turn(m=m)
        await m.ask(tl="g0", name="join", build=lambda r: (QueuedJoin(child="g0", reason="done", trigger=None), r))
        assert host.status["g0"] == "queued" and not host.logged("apply")  # finished, not joined
        _, _, text = await clone(m=m, pairs=[("n1", False)])
        assert f"at most {MAX_LIVE_BACKGROUND} background clones" in text  # g0 still counts
        reporting = host.next("run", "o", "report")
        await end_driven_turn(m=m)
        await reporting  # g0's join is applied
        host.end(tl="o")
        assert await at_rest(m=m) == "ended" and host.status["g0"] == "joined"
        await driven_turn(m=m)
        _, _, text = await clone(m=m, pairs=[("n1", False)])
        assert text == "cloned"  # its place is free once it has joined
        await end_driven_turn(m=m)
        await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), timeout=GUARD)
        await m.stop()
    run(go)


def test_a_report_run_may_not_clone():
    async def go():
        host, m = await started()
        await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        reporting = host.next("run", "o", "report")
        host.end("o")
        host.end("c1")  # c1's join is applied, then a report run
        await reporting
        _, _, text = await clone(m=m, pairs=[("c2", False)])
        assert "do not clone now" in text and "c2" not in host.status
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
        assert not host.logged("apply")
        host.end("c2")  # the focus graft answers the user: the user's call is answered by it
        assert await reply == "ended" and await at_rest(m=m, tl="c2") == "ended"
        reporting = host.next("run", "o", "report")
        await m.send(tl="c2", name="join", data=(QueuedJoin(child="c2", reason="done", trigger=None), None))
        await reporting  # c1's work is reported
        assert [e[1] for e in host.logged("apply")] == ["c1", "c2"]  # the focus join last
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
        await m.send(tl="o", name="interrupt", data=reply)  # forwarded by the suspended original
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
        assert "o" in host.runs and not host.logged("apply")
        reporting = host.next("run", "o", "report")
        host.end("o")
        await reporting
        assert [e[1] for e in host.logged("apply")] == ["c2", "c1"]  # finish order
        assert ("note", ("c2", "c1"), False) in host.log  # one wake for both
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
        reporting = host.next("run", "o", "report")
        host.end("c1")
        await reporting
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
        assert not host.logged("apply") and host.status["c1"] == host.status["c2"] == "queued"
        assert host.reports == [("commit", ["c1", "c2"], "cannot apply c1")]
        for _ in range(MAX_JOIN_ATTEMPTS - 2):  # a commit point that fails again
            assert await m.ask(tl="o", name="drain_request", build=lambda r: r) == "ended"
        reporting = host.next("run", "o", "report")  # the last attempt: c1 is given up, c2 applied
        draining = asyncio.ensure_future(m.ask(tl="o", name="drain_request", build=lambda r: r))
        await reporting
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
        await clone(m=m, pairs=[("c1", False), ("c2", True)], parts=[{"web"}, {"talk"}])  # c2 holds the conversation
        host.end("c2")
        assert await first == "ended"  # the focus graft answered
        await m.send(tl="c2", name="join", data=(QueuedJoin(child="c2", reason="done", trigger=None), None))
        assert await at_rest(m) == "ended" and holders(host) == ["o"]
        reply = await say(m=m, text="stop it")
        results = await cancel(m=m, tl="o", ids=["c1"])
        log = host.log
        assert log.index(("run_cancelled", "c1")) < log.index(("reason", "c1", CANCELLED))  # ended, then queued
        assert results == [(repr([("c1", CANCELLED)]), False)] and host.status["c1"] == "queued"
        assert "o" in host.runs and not host.logged("apply", "c1")  # queued: applied at rest
        _, _, text = await clone(m=m, pairs=[("n9", False)], parts=[{"web"}])
        assert "was cancelled" in text  # its part is blocked for this request
        _, _, text = await clone(m=m, pairs=[(f"n{i}", False) for i in range(MAX_LIVE_BACKGROUND)])
        assert text == "cloned"  # its place under the cap was free at once
        reporting = host.next("run", "o", "report")
        host.end("o")
        await reporting
        assert ("apply", "c1", CANCELLED) in host.log and ("note", ("c1",), False) in host.log
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
        assert outcomes == {"o": "the original cannot be cancelled", "c9": "no such clone",
                            "c2": "it has already finished or was already cancelled; nothing to cancel"}
        assert results[0][1] and host.status["c1"] == "background"  # nothing was touched
        assert "only the original cancels" in await cancel(m=m, tl="c1", ids=["c2"])  # a graft never cancels
        reporting = host.next("run", "o", "report")
        host.end("o")
        await reporting  # the wake that reports c2
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
        await m.send(tl="o", name="interrupt", data=reply)  # noted by the machine; the cancel lands first
        assert not asking.done() and not host.logged("close", "o")
        closed = host.next("close", "o")
        reporting = host.next("run", "o", "report")
        host.hold.set_result(None)
        assert await asking == [(repr([("c1", CANCELLED)]), False)]
        await closed  # then the interrupt ended the run
        result = ("result", "o", "x", repr([("c1", CANCELLED)]), False)
        assert host.log.index(result) < host.log.index(host.logged("close", "o")[0])
        await reporting  # c1's join is applied and reported
        host.end("o")
        assert await reply == "ended" and host.status["c1"] == "joined"
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
        reporting = host.next("run", "o", "report")
        host.end("c1")
        await reporting
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
        assert host.logged("reason") == [("reason", "c1", "failed"), ("reason", "c2", "failed")]
        assert [e[1] for e in host.logged("apply")] == ["c1", "c2"]
        assert not host.logged("run", "o", "report")
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
        reporting = host.next("run", "o", "report")
        # the turn goes on: its caller ends it (it would wait in `writing`)
        assert await asyncio.wait_for(m.ask(tl="o", name="driven_done", build=lambda r: r), timeout=GUARD) == "ended"
        await reporting
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


def test_a_request_row_that_cannot_be_written_fails_the_call_explicitly_and_runs_nothing():
    async def go():
        host, m = await started()
        _failing(host=host, name="begin_request", error=OSError("disk full"))
        reply = await say(m=m, text="hi")
        with pytest.raises(OSError, match="disk full"):
            await reply
        assert ("request", ["disk full"]) == (host.reports[-1][0], [host.reports[-1][2]])
        assert await at_rest(m) == "ended" and not host.logged("run")  # no run started
        await m.stop()
    run(go)


@pytest.mark.parametrize("name, refused", [("commit_clone", True), ("store_result", False)])
def test_a_clone_write_that_fails_ends_in_its_handled_state(name, refused):
    """The commit is `committing`'s activity: if it fails, the call is refused (`refusing`). The
    result row is `recording`'s: if it fails, the clone stands and the row is kept for the run's
    closing. Either way the original is back in its run, never stuck in the writing state."""
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        _failing(host=host, name=name, error=OSError("disk full"))
        children, _, text = await clone(m=m, pairs=[("c1", False)])
        assert (children is None) == refused and "o" in host.runs
        await m.send(tl="o", name="interrupt", data=first)  # the run goes on until it is ended
        assert ("run_cancelled", "o") in host.log and await first == "ended"
        kept = {} if refused else {"k": ("cloned", False)}  # the real result, written by the closing
        assert host.logged("close", "o") == [("close", "o", kept)]
        if not refused:
            assert host.status["c1"] == "background"
        await m.stop()
    run(go)


@pytest.mark.parametrize("pairs", [[("c1", False)], [("c0", False), ("c1", False)]])
def test_a_graft_machine_that_cannot_start_undoes_the_whole_clone_call(monkeypatch, pairs):
    """The machines are started before anything else learns of the grafts: if one cannot start,
    the ones made are stopped and forgotten, the committed grafts are marked abandoned, and the
    result is a refusal. Settle, close and a cancel of the dead id all still return."""
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
        for g, _ in pairs:  # forgotten
            with pytest.raises(TransitionRefused, match="no machine"):
                await m.send(tl=g, name="cancel")
        answer = await cancel(m=m, tl="o", ids=["c1"])
        assert "no such clone" in answer[0][0]
        host.end("o")
        assert await first == "ended"
        await asyncio.wait_for(m.ask(tl="session", name="settle_request", build=lambda r: r), timeout=GUARD)
        await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), timeout=GUARD)
        await m.stop()
    run(go)


@pytest.mark.parametrize("name", ["close_tool_uses", "wake_note"])
def test_a_failing_closing_or_wake_write_leaves_the_original_at_rest(name):
    async def go():
        host, m = await started()
        first = await say(m=m, text="go")
        await clone(m=m, pairs=[("c1", False)])
        _failing(host=host, name=name, error=OSError("disk full"))
        if name == "close_tool_uses":
            reported = host.next("report", "task")
            await m.send(tl="o", name="interrupt", data=first)  # the run ends early: its calls need results
        else:
            host.end("o")
            assert await first == "ended"
            reported = host.next("report", "wake")
            host.end("c1")  # its join is applied and a wake note is due
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


# ------------------------------------------------------------------ round-2 regressions
@pytest.mark.parametrize("order", ["cancel_first", "finish_first"])
def test_a_cancel_that_loses_the_race_to_a_finish_blocks_nothing(order):
    """The two orders a cancel and a graft's own join can land in, each forced: a cancel that
    stops the graft blocks its part; one that finds the join already raised blocks nothing."""
    async def go():
        host, m = await started()
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))
        await clone(m=m, pairs=[("c1", False)], parts=[{"web"}])
        if order == "finish_first":
            await m.ask(tl="c1", name="join", build=lambda r: (QueuedJoin(child="c1", reason="done", trigger=None), r))
        answer = await asyncio.wait_for(cancel(m=m, tl="o", ids=["c1"]), timeout=GUARD)
        cancelled = answer[0][1] is False
        assert cancelled == (order == "cancel_first")
        reporting = host.next("run", "o", "report")
        await m.ask(tl="o", name="driven_done", build=lambda r: r)
        await reporting  # c1's join is applied and reported
        host.end("o")
        assert await at_rest(m) == "ended"
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))  # the same request
        _, _, text = await clone(m=m, pairs=[("c2", False)], parts=[{"web"}])
        assert ("was cancelled" in text) == cancelled  # a part is blocked only by a real cancel
        await asyncio.wait_for(m.ask(tl="session", name="shutdown", build=lambda r: r), timeout=GUARD)
        await m.stop()
    run(go)


@pytest.mark.parametrize("reason, blocks", [("failed", True), ("done", False)])
def test_a_failed_graft_blocks_its_part_for_the_request_it_was_cloned_for_only(reason, blocks):
    async def go():
        host, m = await started()
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))
        await clone(m=m, pairs=[("c1", False)], parts=[{"web"}], request=1)  # cloned for request 1
        await m.ask(tl="c1", name="join", build=lambda r: (QueuedJoin(child="c1", reason=reason, trigger=None), r))
        reporting = host.next("run", "o", "report")
        await m.ask(tl="o", name="driven_done", build=lambda r: r)
        await reporting  # c1's join is applied and reported
        host.end(tl="o")
        assert await at_rest(m=m) == "ended"
        await m.ask(tl="o", name="turn_request", build=lambda r: (Mode(name="external"), r))
        _, _, text = await clone(m=m, pairs=[("c2", False)], parts=[{"web"}], request=1)  # same part, same request
        assert ("already failed" in text if blocks else text == "cloned"), text
        if blocks:  # the same part for another request is not blocked (a refused call leaves the turn its clone)
            _, _, text = await clone(m=m, pairs=[("c3", False)], parts=[{"web"}], request=2)
            assert text == "cloned", text
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
