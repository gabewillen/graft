"""A host tool may return an awaitable: the call is the activity of its timeline's `calling` state,
awaited there while every other timeline goes on. Leaving that state (a cancel, a withdrawal,
shutdown) cancels the tool where it waits, and the call is recorded as run with an error, never
as not executed: it may have acted, or may still act."""

import asyncio
import threading

import pytest

from graft.skills import World
from test_parallel import (REQUEST, WAIT_TIMEOUT, clone_entry, clone_use, is_wake, make, text, tool_results,
                           violations)

STOP = "stop the website"
CANCELLED_WHILE_RUNNING = "cancelled while running: it may have acted, or may still act"


def use(id, name, **args):
    return {"type": "tool_use", "id": id, "name": name, "input": args}


class Meeting(World):
    """site_write (clone A) waits until slides_create (clone B) has run; each call is logged as it
    starts and ends, on the loop. `a_started` is a threading.Event: B's model call waits on it, so
    A's tool is already waiting when B's runs."""

    def __init__(self):
        super().__init__()
        self.ready = asyncio.Event()
        self.a_started = threading.Event()
        self.events: list[str] = []

    async def execute(self, name, args):
        if name == "site_write":
            self.events.append("a_start")
            self.a_started.set()
            await self.ready.wait()
            self.events.append("a_end")
            return "wrote after the deck was made", False
        if name == "slides_create":
            self.events.append("b_start")
            self.ready.set()
            self.events.append("b_end")
            return "deck made", False
        return super().execute(name=name, args=args)


def background_original(entries):
    """An original that clones `entries` in the background, says "Started." after the clone
    result, reports on a wake note, and cancels c1 when the user says STOP."""
    def respond(last, messages):
        if last["type"] == "tool_result":
            return text("Stopped." if last["tool_use_id"].startswith("x") else "Started.")
        if is_wake(last):
            return text("Report.")
        if last["text"] == STOP:
            return [use(f"x{len(messages)}", "cancel", cancel=["c1"])]
        return [clone_use(*entries, id=f"c{len(messages)}") | {"type": "tool_use"}]
    return respond


def test_a_tool_that_awaits_another_clones_tool_lets_both_run_and_finish(tmp_path):
    world = Meeting()
    original = background_original([clone_entry(skills=["website"], focus=False),
                                    clone_entry(skills=["slides"], focus=False)])

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "branch" in names:
            return original(last, messages)
        if last["type"] == "tool_result":
            return text("done")
        if "site_write" in names:
            return [use("w1", "site_write", page="index", content="hi")]
        assert world.a_started.wait(WAIT_TIMEOUT)  # B acts only once A's tool is waiting on it
        return [use("s1", "slides_create", title="Deck")]

    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(asyncio.gather(h.join_queued("c1"), h.join_queued("c2")), WAIT_TIMEOUT)
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert world.events == ["a_start", "b_start", "b_end", "a_end"]  # A's call was open while B's ran
    assert tool_results(store=store, tl="c1") == [("site_write", "wrote after the deck was made", False)]
    assert tool_results(store=store, tl="c2") == [("slides_create", "deck made", False)]
    assert [store.timeline(c)["join_reason"] for c in ("c1", "c2")] == ["done", "done"]
    assert h.tools_run("c1") == ["site_write"] and h.tools_run("c2") == ["slides_create"]


class Hanging(World):
    """site_write waits for ever; it records that it started and that it was cancelled."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.running = threading.Event()  # the same, for a scripted model call in a worker thread
        self.cancelled = False

    async def execute(self, name, args):
        if name == "site_write":
            self.started.set()
            self.running.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return super().execute(name=name, args=args)


def graft_writes(original):
    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "branch" in names:
            return original(last, messages)
        if last["type"] == "tool_result":
            return text("done")
        return [use("w1", "site_write", page="index", content="hi")]
    return policy


def test_a_background_graft_cancelled_while_its_tool_awaits_records_the_call_as_run(tmp_path):
    world = Hanging()
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        assert await h.user_says(REQUEST) == ["Started."]
        await asyncio.wait_for(world.started.wait(), WAIT_TIMEOUT)
        out = await asyncio.wait_for(h.user_says(STOP), WAIT_TIMEOUT)
        assert out[0] == "Stopped."
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert world.cancelled  # the tool saw the cancel where it waited
    assert store.timeline("c1")["join_reason"] == "cancelled"
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_WHILE_RUNNING, True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"]  # in the cancel record: it may have acted
    assert violations(store=store, tl="c1") == []


def test_a_focus_graft_whose_request_is_withdrawn_while_its_tool_awaits_records_the_call_as_run(tmp_path):
    world = Hanging()

    def original(last, messages):
        if last["type"] == "tool_result":
            return text("Started.")
        return [clone_use(clone_entry(skills=["website"], focus=True), id=f"c{len(messages)}") | {"type": "tool_use"}]

    store, h, _ = make(tmp_path=tmp_path, policy=graft_writes(original), world=world)

    async def go():
        await h.start()
        call = asyncio.create_task(h.user_says(REQUEST))
        await asyncio.wait_for(world.started.wait(), WAIT_TIMEOUT)
        call.cancel()  # the user withdraws the request the focus graft's run answers
        try:
            await call
        except asyncio.CancelledError:
            pass
        assert await asyncio.wait_for(h.until_rest("c1"), WAIT_TIMEOUT) == "ended"
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert world.cancelled
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_WHILE_RUNNING, True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"]
    assert violations(store=store, tl="c1") == []


class Slow(World):
    """Two awaited tools: the first takes longer. Each logs its start and end."""

    def __init__(self):
        super().__init__()
        self.events: list[str] = []

    async def execute(self, name, args):
        self.events.append(f"{name} start")
        await asyncio.sleep(0.05 if name == "site_write" else 0)
        self.events.append(f"{name} end")
        return f"{name} ok", False


def test_a_turns_awaited_tool_calls_run_one_at_a_time_and_are_stored_in_call_order(tmp_path):
    world = Slow()
    original = background_original([clone_entry(skills=["website"], focus=False)])

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "branch" in names:
            return original(last, messages)
        if last["type"] == "tool_result":
            return text("done")
        return [use("w1", "site_write", page="index", content="hi"), use("r1", "site_read", page="index")]

    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert world.events == ["site_write start", "site_write end", "site_read start", "site_read end"]
    assert tool_results(store=store, tl="c1") == [("site_write", "site_write ok", False),
                                                  ("site_read", "site_read ok", False)]
    assert h.tools_run("c1") == ["site_write", "site_read"]


def test_close_while_a_background_graft_s_tool_awaits_returns_and_records_the_call_as_run(tmp_path):
    world = Hanging()
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        assert await h.user_says(REQUEST) == ["Started."]
        await asyncio.wait_for(world.started.wait(), WAIT_TIMEOUT)
        await asyncio.wait_for(h.close(), WAIT_TIMEOUT)  # a tool that never returns does not hold close up

    asyncio.run(go())
    assert world.cancelled
    assert store.timeline("c1")["join_reason"] == "failed"  # shut down while it ran
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_WHILE_RUNNING, True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"]


def test_a_synchronous_tool_that_raises_ends_as_run_with_an_error(tmp_path):
    """A tool that was invoked has run, whatever it then raised: its call gets a result marked
    run, with an error naming what was raised, and the run goes on (the model sees it)."""
    class Raising(World):
        def execute(self, name, args):
            raise RuntimeError("backend down")

    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=Raising())

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)
        await h.settle()
        await h.close()

    asyncio.run(go())
    assert h.graft_errors == [] and store.timeline("c1")["join_reason"] == "done"
    assert tool_results(store=store, tl="c1") == [
        ("site_write", "raised RuntimeError: backend down; it may have acted", True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"]


# ------------------------------------------------------------------ a frozen graft's close-out that cannot be written
def fail_once(h, name, tl="c1"):
    """h's host call `name` for timeline `tl` raises once (the store write fails), then works."""
    real, fired = getattr(h, name), []

    def failing(**kw):
        if not fired and kw.get("tl") == tl:
            fired.append(name)
            raise OSError(f"INJECTED {name}")
        return real(**kw)
    setattr(h, name, failing)
    return fired


async def settled(h):
    """settle, which raises the injected error once (it was told); then nothing more."""
    try:
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
    except OSError:
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)


def test_a_background_graft_whose_close_out_cannot_be_written_carries_it_with_its_join(tmp_path):
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy)
    fail_once(h=h, name="store_result")  # its row cannot be written: the run ends, the real result kept
    fired = fail_once(h=h, name="close_tool_uses")

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)
        await settled(h)
        await h.close()

    asyncio.run(go())
    assert fired == ["close_tool_uses"] and store.timeline("c1")["status"] == "joined"
    assert tool_results(store=store, tl="c1") == [("site_write", "wrote page index (2 chars)", False)]
    assert h.tools_run("c1") == ["site_write"]  # the carried close-out holds the real result, marked run
    assert violations(store=store, tl="c1") == [] and violations(store=store, tl="o") == []


def test_a_cancelled_graft_whose_close_out_cannot_be_written_still_names_the_running_tool(tmp_path):
    world = Hanging()
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(world.started.wait(), WAIT_TIMEOUT)
        fail_once(h=h, name="close_tool_uses")
        await asyncio.wait_for(h.user_says(STOP), WAIT_TIMEOUT)
        await settled(h)
        await h.close()

    asyncio.run(go())
    assert world.cancelled and store.timeline("c1")["join_reason"] == "cancelled"
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_WHILE_RUNNING, True)]
    record = [r["content"] for r in store.view("o") if "was cancelled before it finished" in r["content"]]
    assert len(record) == 1 and "site_write (returned an error)" in record[0]
    assert violations(store=store, tl="c1") == [] and violations(store=store, tl="o") == []


def test_a_failed_focus_graft_whose_close_out_cannot_be_written_carries_it_with_its_join(tmp_path):
    def original(last, messages):
        if last["type"] == "tool_result" or is_wake(last) or last.get("harness"):
            return text("Back.")
        return [clone_use(clone_entry(skills=["website"], focus=True), id=f"c{len(messages)}") | {"type": "tool_use"}]

    store, h, _ = make(tmp_path=tmp_path, policy=graft_writes(original))
    fail_once(h=h, name="store_result")  # its row cannot be written: the run ends, the real result kept
    fired = fail_once(h=h, name="close_tool_uses")

    async def go():
        await h.start()
        try:
            await asyncio.wait_for(h.user_says(REQUEST), WAIT_TIMEOUT)
        except OSError:
            pass
        await settled(h)
        await h.close()

    asyncio.run(go())
    assert fired == ["close_tool_uses"] and store.timeline("c1")["status"] == "joined"
    assert store.timeline("c1")["join_reason"] == "failed"
    assert tool_results(store=store, tl="c1") == [("site_write", "wrote page index (2 chars)", False)]
    assert h.tools_run("c1") == ["site_write"]
    assert violations(store=store, tl="c1") == [] and violations(store=store, tl="o") == []


# ------------------------------------------------------------------ a withdrawal stops the background grafts made for it
def test_a_withdrawal_stops_a_background_graft_whose_tool_awaits_at_once(tmp_path):
    """The user's call is cancelled before it is answered (the original's run is still going):
    the background graft made for it stops at the withdrawal, not at its next step."""
    world, gate = Hanging(), threading.Event()

    def original(last, messages):
        if last["type"] == "tool_result":
            assert gate.wait(WAIT_TIMEOUT)
            return text("Started.")
        if is_wake(last):
            return text("Report.")
        return [clone_use(clone_entry(skills=["website"], focus=False), id="k1") | {"type": "tool_use"}]

    store, h, _ = make(tmp_path=tmp_path, policy=graft_writes(original), world=world)

    async def go():
        await h.start()
        call = asyncio.create_task(h.user_says(REQUEST))
        await asyncio.wait_for(world.started.wait(), WAIT_TIMEOUT)
        call.cancel()
        gate.set()
        try:
            await call
        except asyncio.CancelledError:
            pass
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)  # stopped without close
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert world.cancelled and store.timeline("c1")["join_reason"] == "cancelled"
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_WHILE_RUNNING, True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"] and h.wakes == []
    assert violations(store=store, tl="c1") == [] and violations(store=store, tl="o") == []


def test_a_withdrawal_after_the_answer_stops_a_background_graft_whose_tool_awaits_at_once(tmp_path, monkeypatch):
    """The user's call is cancelled in the tick it is answered (its call not yet done): the cancel
    wins, and the background graft made for it stops at once."""
    import graft.machine as machine
    world = Hanging()
    late: dict = {}
    real_rest = machine.come_to_rest

    def come_to_rest(ctx, inst, event):
        real_rest(ctx=ctx, inst=inst, event=event)
        task = late.pop("task", None)
        if task is not None and inst.timeline_id == "o":
            task.cancel()
    monkeypatch.setattr(machine, "come_to_rest", come_to_rest)

    def original(last, messages):
        if last["type"] == "tool_result":
            assert world.running.wait(WAIT_TIMEOUT)  # the answer comes once the graft's tool waits
            return text("Started.")
        if is_wake(last):
            return text("Report.")
        return [clone_use(clone_entry(skills=["website"], focus=False), id="k1") | {"type": "tool_use"}]

    store, h, _ = make(tmp_path=tmp_path, policy=graft_writes(original), world=world)

    async def go():
        await h.start()
        call = late["task"] = asyncio.create_task(h.user_says(REQUEST))
        try:
            await asyncio.wait_for(call, WAIT_TIMEOUT)
        except asyncio.CancelledError:
            pass
        assert call.cancelled() and world.started.is_set()
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)  # stopped without close
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert world.cancelled and store.timeline("c1")["join_reason"] == "cancelled"
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_WHILE_RUNNING, True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"] and h.wakes == []
    assert violations(store=store, tl="c1") == [] and violations(store=store, tl="o") == []


def test_a_tool_whose_row_fails_and_whose_run_is_then_withdrawn_keeps_its_real_result(tmp_path):
    """The tool returns, its row cannot be written, and a withdrawal is taken before the
    row write's outcome is taken (`storing`): the call is closed with what the tool returned, marked run,
    not as cut short."""
    def original(last, messages):
        if last["type"] == "tool_result" or last.get("harness"):
            return text("Back.")
        return [clone_use(clone_entry(skills=["website"], focus=True), id=f"c{len(messages)}") | {"type": "tool_use"}]

    store, h, said = make(tmp_path=tmp_path, policy=graft_writes(original))
    real, call, raised = h.store_result, {}, []

    def failing(**kw):
        if kw["tl"] == "c1" and not call.get("fired"):
            call["fired"] = True
            h.machines.post(tl="c1", name="withdraw", data=call["id"])  # taken before the outcome
            call["task"].cancel()
            raise OSError("INJECTED store_result")
        return real(**kw)
    h.store_result = failing

    async def go():
        await h.start()
        call["id"] = h.machines.requests + 1  # the id the next user request gets
        call["task"] = asyncio.create_task(h.user_says(REQUEST))
        try:
            await asyncio.wait_for(call["task"], WAIT_TIMEOUT)
        except (asyncio.CancelledError, OSError):
            pass
        await asyncio.wait_for(h.until_rest("c1"), WAIT_TIMEOUT)
        try:
            await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        except OSError as error:
            raised.append(error)
        await h.close()

    asyncio.run(go())
    assert call["fired"]
    # the row write's failure reached the machine after its run was left: it is still told, once
    told = [line for line in said if "INJECTED store_result" in line]
    assert len(told) == 1 and [str(e) for e in raised] in ([], ["INJECTED store_result"])
    assert tool_results(store=store, tl="c1") == [("site_write", "wrote page index (2 chars)", False)]
    assert h.tools_run("c1") == ["site_write"] and violations(store=store, tl="c1") == []


# ------------------------------------------------------------------ a tool cancelled by something other than its graft
CANCELLED_BY_THE_TOOL = "cancelled by the tool: it may have acted"


class SelfCancel(World):
    """site_write awaits a future the host cancels itself (its own deadline): CancelledError reaches
    the activity while its state is still active."""

    async def execute(self, name, args):
        if name == "site_write":
            fut = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(fut.cancel)
            await fut
        return super().execute(name=name, args=args)


def test_a_tool_that_is_cancelled_by_its_host_ends_as_run_with_an_error_and_its_run_goes_on(tmp_path):
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=SelfCancel())

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)  # not wedged in `calling`
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_BY_THE_TOOL, True)]
    assert store.timeline("c1")["join_reason"] == "done"  # its model saw the error and replied
    assert h.tools_run("c1") == ["site_write (returned an error)"]


def test_a_focus_graft_whose_tool_is_cancelled_by_its_host_goes_on(tmp_path):
    def original(last, messages):
        if last["type"] == "tool_result" or last.get("harness"):
            return text("Back.")
        return [clone_use(clone_entry(skills=["website"], focus=True), id=f"c{len(messages)}") | {"type": "tool_use"}]

    store, h, _ = make(tmp_path=tmp_path, policy=graft_writes(original), world=SelfCancel())

    async def go():
        await h.start()
        assert await asyncio.wait_for(h.user_says(REQUEST), WAIT_TIMEOUT) == ["done"]
        await h.close()

    asyncio.run(go())
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_BY_THE_TOOL, True)]


class SharedWork(World):
    """slides_create (c1) starts a task and awaits it; site_write (c2) awaits the same task. When c1
    is cancelled its await is cancelled, and so is the shared task c2 awaits: c2's state is not left."""

    def __init__(self):
        super().__init__()
        self.task: asyncio.Future | None = None
        self.both = asyncio.Event()

    async def execute(self, name, args):
        if name == "slides_create":
            self.task = asyncio.ensure_future(asyncio.Event().wait())
            await self.task
        if name == "site_write":
            while self.task is None:
                await asyncio.sleep(0.01)
            self.both.set()
            await self.task
        return super().execute(name=name, args=args)


def test_a_graft_whose_tool_awaits_work_a_cancelled_sibling_shared_goes_on(tmp_path):
    world = SharedWork()
    original = background_original([clone_entry(skills=["slides"], focus=False),
                                    clone_entry(skills=["website"], focus=False)])

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "branch" in names:
            return original(last, messages)
        if last["type"] == "tool_result":
            return text("done")
        if "site_write" in names:
            return [use("w1", "site_write", page="i", content="x")]
        return [use("s1", "slides_create", title="D")]

    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(world.both.wait(), WAIT_TIMEOUT)
        await asyncio.wait_for(h.user_says(STOP), WAIT_TIMEOUT)  # cancels c1
        await asyncio.wait_for(h.join_queued("c2"), WAIT_TIMEOUT)  # c2 is not wedged
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert tool_results(store=store, tl="c1") == [("slides_create", CANCELLED_WHILE_RUNNING, True)]
    assert tool_results(store=store, tl="c2") == [("site_write", CANCELLED_BY_THE_TOOL, True)]
    assert store.timeline("c2")["join_reason"] == "done"


# ------------------------------------------------------------------ a tool that was invoked is never closed as not run
def one_background(world, **kw):
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    return make(tmp_path=kw.pop("tmp_path"), policy=policy, world=world, **kw)


def run_one(h):
    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)
        try:
            await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        except Exception:  # noqa: BLE001 - a failure told meanwhile is raised once here
            pass
        await h.close()
    asyncio.run(go())


class ActsThenRaises(World):
    """site_write acts, then its own deadline raises."""

    def __init__(self):
        super().__init__()
        self.acted = False

    async def execute(self, name, args):
        if name == "site_write":
            self.acted = True
            raise TimeoutError("the host's own deadline passed")
        return super().execute(name=name, args=args)


def test_a_tool_that_raises_after_it_was_invoked_ends_as_run_with_an_error(tmp_path):
    world = ActsThenRaises()
    store, h, _ = one_background(world, tmp_path=tmp_path)
    run_one(h)
    assert world.acted
    assert tool_results(store=store, tl="c1") == [
        ("site_write", "raised TimeoutError: the host's own deadline passed; it may have acted", True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"]
    assert store.timeline("c1")["join_reason"] == "done" and h.graft_errors == []  # its run went on


class BadShape(World):
    def __init__(self, value):
        super().__init__()
        self.value = value

    async def execute(self, name, args):
        if name == "site_write":
            return self.value
        return super().execute(name=name, args=args)


@pytest.mark.parametrize("value, kind", [((123, False), "tuple"), ({"ok": True}, "dict"), (("text", "no"), "tuple")])
def test_a_tool_that_returns_a_bad_shape_ends_as_run_with_an_error(tmp_path, value, kind):
    store, h, _ = one_background(BadShape(value), tmp_path=tmp_path)
    run_one(h)
    assert tool_results(store=store, tl="c1") == [
        ("site_write", f"returned a {kind}, not (text, is_error); it may have acted", True)]
    assert h.tools_run("c1") == ["site_write (returned an error)"]


def test_a_trace_callback_that_raises_leaves_the_tool_s_result_alone(tmp_path):
    def trace(line):
        if "] tool site_write" in line:
            raise RuntimeError("trace sink down")

    store, h, _ = one_background(World(), tmp_path=tmp_path, trace=trace)
    run_one(h)
    assert tool_results(store=store, tl="c1") == [("site_write", "wrote page index (2 chars)", False)]
    assert h.tools_run("c1") == ["site_write"]


def test_a_graft_whose_join_is_abandoned_still_writes_the_close_out_it_carried(tmp_path):
    from graft.machine import MAX_JOIN_ATTEMPTS
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy)
    fail_once(h=h, name="store_result")
    fail_once(h=h, name="close_tool_uses")

    def never(**kw):
        raise OSError("bad row")
    h.default_join = never

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)
        for _ in range(MAX_JOIN_ATTEMPTS + 1):
            if store.timeline("c1")["status"] == "abandoned":
                break
            await asyncio.wait_for(h.drain_and_wake("o"), WAIT_TIMEOUT)
        try:
            await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        except Exception:  # noqa: BLE001 - the failures told meanwhile
            pass
        await h.close()

    asyncio.run(go())
    assert store.timeline("c1")["status"] == "abandoned"
    assert tool_results(store=store, tl="c1") == [("site_write", "wrote page index (2 chars)", False)]
    assert violations(store=store, tl="c1") == []


def test_a_join_retried_after_its_write_failed_writes_the_carried_close_out_once(tmp_path):
    world = Hanging()
    policy = graft_writes(background_original([clone_entry(skills=["website"], focus=False)]))
    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)
    real, fired = store.commit_join, []

    def failing(*args, **kw):
        if not fired and kw.get("closing"):
            fired.append(1)
            raise OSError("INJECTED commit_join")
        return real(*args, **kw)
    store.commit_join = failing

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(world.started.wait(), WAIT_TIMEOUT)
        fail_once(h=h, name="close_tool_uses")
        await asyncio.wait_for(h.user_says(STOP), WAIT_TIMEOUT)  # cancels c1; its join's write fails once
        if store.timeline("c1")["status"] != "joined":
            await asyncio.wait_for(h.drain_and_wake("o"), WAIT_TIMEOUT)  # the next commit point retries it
        try:
            await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        except OSError:
            pass
        await h.close()

    asyncio.run(go())
    assert fired and store.timeline("c1")["status"] == "joined"
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_WHILE_RUNNING, True)]
    assert violations(store=store, tl="c1") == [] and violations(store=store, tl="o") == []


class SeveralWaiting(World):
    """Every capability tool waits for ever; each records that it started and that it was cancelled."""

    def __init__(self):
        super().__init__()
        self.started: dict[str, asyncio.Event] = {}
        self.cancelled: set[str] = set()

    async def execute(self, name, args):
        self.started.setdefault(name, asyncio.Event()).set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.add(name)
            raise


def test_a_withdrawal_stops_several_background_grafts_whose_tools_await(tmp_path):
    world, gate = SeveralWaiting(), threading.Event()
    tools = {"website": use("w1", "site_write", page="i", content="x"),
             "slides": use("s1", "slides_create", title="D"),
             "calendar": use("k1", "calendar_book", what="x", date="2026-12-05", time="10:00")}

    def policy(system, messages, tool_list):
        names = {t["name"] for t in tool_list}
        last = messages[-1]["content"][-1]
        if "branch" in names:
            if last["type"] == "tool_result":
                assert gate.wait(WAIT_TIMEOUT)
                return text("Started.")
            if is_wake(last):
                return text("Report.")
            return [clone_use(*[clone_entry(skills=[k], focus=False) for k in tools], id="k1") | {"type": "tool_use"}]
        skill = "website" if "site_write" in names else "slides" if "slides_create" in names else "calendar"
        return [tools[skill]]

    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        call = asyncio.create_task(h.user_says(REQUEST))
        for name in ("site_write", "slides_create", "calendar_book"):
            while name not in world.started:
                await asyncio.sleep(0.01)
        call.cancel()
        gate.set()
        try:
            await call
        except asyncio.CancelledError:
            pass
        for c in ("c1", "c2", "c3"):
            await asyncio.wait_for(h.join_queued(c), WAIT_TIMEOUT)  # each stopped at the withdrawal
        await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert world.cancelled == {"site_write", "slides_create", "calendar_book"} and h.wakes == []
    for c in ("c1", "c2", "c3"):
        assert store.timeline(c)["join_reason"] == "cancelled"
        assert [content for _, content, _ in tool_results(store=store, tl=c)] == [CANCELLED_WHILE_RUNNING]


# ------------------------------------------------------------------ whatever a tool raises, once invoked it ran
FAILED_AFTER_INVOKED = "failed after it was invoked: it may have acted"


def drive_one(h, store):
    """One background graft's run; whether it joined, settle returned and close returned."""
    out = {}

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        for key, coro in (("joined", lambda: h.join_queued("c1")), ("settled", h.settle), ("closed", h.close)):
            try:
                await asyncio.wait_for(coro(), WAIT_TIMEOUT)
                out[key] = True
            except asyncio.TimeoutError:
                out[key] = False
            except Exception as error:  # noqa: BLE001 - a failure told meanwhile
                out[key] = f"raised {type(error).__name__}"
    asyncio.run(go())
    return out


def test_a_harness_tool_call_that_fails_after_the_tool_ran_keeps_the_call_as_run(tmp_path):
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    real = h.call_tool

    async def broken(**kw):
        await real(**kw)
        raise RuntimeError("after the tool")
    h.call_tool = broken
    assert drive_one(h=h, store=store)["joined"] is True
    assert tool_results(store=store, tl="c1") == [("site_write", FAILED_AFTER_INVOKED, True)]
    assert h.world.site and h.tools_run("c1") == ["site_write (returned an error)"]
    assert store.timeline("c1")["join_reason"] == "done"  # its run went on


class BadStr(Exception):
    def __str__(self):
        raise RuntimeError("no str")


class ActsThenBadStr(World):
    async def execute(self, name, args):
        if name == "site_write":
            raise BadStr()
        return super().execute(name=name, args=args)


def test_a_tool_whose_error_cannot_be_shown_still_ends_as_run_with_an_error(tmp_path):
    store, h, _ = one_background(ActsThenBadStr(), tmp_path=tmp_path)
    assert drive_one(h=h, store=store)["joined"] is True
    (name, content, is_error), = tool_results(store=store, tl="c1")
    assert (name, is_error) == ("site_write", True) and content.startswith("raised BadStr")
    assert h.tools_run("c1") == ["site_write (returned an error)"]


class Abort(BaseException):
    pass


class ActsThenAborts(World):
    async def execute(self, name, args):
        if name == "site_write":
            raise Abort("host abort")
        return super().execute(name=name, args=args)


def test_a_tool_that_raises_a_base_exception_ends_as_run_with_an_error(tmp_path):
    store, h, _ = one_background(ActsThenAborts(), tmp_path=tmp_path)
    assert drive_one(h=h, store=store) == {"joined": True, "settled": True, "closed": True}
    assert tool_results(store=store, tl="c1") == [("site_write", "raised Abort: host abort; it may have acted", True)]


def test_a_run_whose_model_call_raises_a_base_exception_fails_and_does_not_wedge(tmp_path):
    def policy(system, messages, tools):
        if "branch" in {t["name"] for t in tools}:
            return background_original([clone_entry(skills=["website"], focus=False)])(messages[-1]["content"][-1],
                                                                                         messages)
        raise Abort("provider abort")

    store, h, _ = make(tmp_path=tmp_path, policy=policy)
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True and out["settled"] in (True, "raised Abort")
    assert store.timeline("c1")["join_reason"] == "failed"


class CancelsOwnTask(World):
    """The host cancels the task its tool runs in (its own deadline), while the graft still runs."""

    async def execute(self, name, args):
        if name == "site_write":
            me = asyncio.current_task()
            asyncio.get_running_loop().call_soon(me.cancel)
            await asyncio.sleep(1)
        return super().execute(name=name, args=args)


def test_a_tool_whose_own_task_the_host_cancels_ends_as_run_with_an_error(tmp_path):
    store, h, _ = one_background(CancelsOwnTask(), tmp_path=tmp_path)
    assert drive_one(h=h, store=store) == {"joined": True, "settled": True, "closed": True}
    assert tool_results(store=store, tl="c1") == [("site_write", CANCELLED_BY_THE_TOOL, True)]
    assert store.timeline("c1")["join_reason"] == "done"



def test_a_withdrawal_stops_every_waiting_graft_even_when_one_stop_fails_once(tmp_path, monkeypatch):
    """The stops are the original's own steps (it ticks through them where it stands): one that
    fails is made again, and no graft's answer is needed for the next stop."""
    import graft.machine as machine
    real, fired = machine.stop_one, []

    def stop_once_failing(ctx, inst, event):
        if not fired:
            fired.append(inst.timeline_id)
            raise RuntimeError("INJECTED stop_one")
        return real(ctx=ctx, inst=inst, event=event)
    monkeypatch.setattr(machine, "stop_one", stop_once_failing)
    world, gate = SeveralWaiting(), threading.Event()
    tools = {"website": use("w1", "site_write", page="i", content="x"),
             "slides": use("s1", "slides_create", title="D"),
             "calendar": use("k1", "calendar_book", what="x", date="2026-12-05", time="10:00")}

    def policy(system, messages, tool_list):
        names = {t["name"] for t in tool_list}
        last = messages[-1]["content"][-1]
        if "branch" in names:
            if last["type"] == "tool_result":
                assert gate.wait(WAIT_TIMEOUT)
                return text("Started.")
            if is_wake(last):
                return text("Report.")
            return [clone_use(*[clone_entry(skills=[k], focus=False) for k in tools], id="k1") | {"type": "tool_use"}]
        skill = "website" if "site_write" in names else "slides" if "slides_create" in names else "calendar"
        return [tools[skill]]

    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=world)

    async def go():
        await h.start()
        call = asyncio.create_task(h.user_says(REQUEST))
        for name in ("site_write", "slides_create", "calendar_book"):
            while name not in world.started:
                await asyncio.sleep(0.01)
        call.cancel()
        gate.set()
        try:
            await call
        except asyncio.CancelledError:
            pass
        for c in ("c1", "c2", "c3"):
            await asyncio.wait_for(h.join_queued(c), WAIT_TIMEOUT)
        try:
            await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        except RuntimeError:
            pass  # the injected failure, told once
        await h.close()

    asyncio.run(go())
    assert fired == ["o"] and world.cancelled == {"site_write", "slides_create", "calendar_book"}


class ActsThenGenExit(World):
    async def execute(self, name, args):
        if name == "site_write":
            raise GeneratorExit()
        return super().execute(name=name, args=args)


def test_a_tool_that_raises_generator_exit_ends_as_run_with_an_error(tmp_path):
    """asyncio lets only KeyboardInterrupt and SystemExit out of the loop: a GeneratorExit a tool
    raises is the tool's outcome, like any other exception."""
    store, h, _ = one_background(ActsThenGenExit(), tmp_path=tmp_path)
    assert drive_one(h=h, store=store) == {"joined": True, "settled": True, "closed": True}
    assert tool_results(store=store, tl="c1") == [("site_write", "raised GeneratorExit; it may have acted", True)]


class ActsThenStop(World):
    async def execute(self, name, args):
        if name == "site_write":
            raise StopAsyncIteration()
        return super().execute(name=name, args=args)


def test_a_tool_that_raises_stop_async_iteration_ends_as_run_with_an_error(tmp_path):
    store, h, _ = one_background(ActsThenStop(), tmp_path=tmp_path)
    assert drive_one(h=h, store=store) == {"joined": True, "settled": True, "closed": True}
    (name, content, is_error), = tool_results(store=store, tl="c1")
    assert (name, is_error) == ("site_write", True) and content.endswith("it may have acted")


class Hostile(str):
    def __getitem__(self, key):
        raise RuntimeError("hostile slice")

    def __len__(self):
        raise RuntimeError("hostile len")


class ReturnsHostile(World):
    async def execute(self, name, args):
        if name == "site_write":
            return Hostile("ok"), False
        return super().execute(name=name, args=args)


def test_a_tool_that_returns_a_hostile_str_subclass_is_stored_as_plain_text(tmp_path):
    store, h, _ = one_background(ReturnsHostile(), tmp_path=tmp_path)
    assert drive_one(h=h, store=store) == {"joined": True, "settled": True, "closed": True}
    assert tool_results(store=store, tl="c1") == [("site_write", "ok", False)]


# ------------------------------------------------------------------ no activity ends without an outcome
def test_a_row_write_that_raises_a_base_exception_still_ends_the_call(tmp_path):
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    real = h.store_result

    def broken(**kw):
        if kw.get("tl") == "c1":
            raise Abort("store abort")
        return real(**kw)
    h.store_result = broken
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True
    assert tool_results(store=store, tl="c1") == [("site_write", "wrote page index (2 chars)", False)]  # the closing's


def test_a_join_record_write_that_raises_a_base_exception_still_queues_the_join(tmp_path):
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    real = h.record_join

    def broken(**kw):
        if kw.get("child") == "c1":
            raise Abort("record abort")
        return real(**kw)
    h.record_join = broken
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True


# ------------------------------------------------------------------ a stop that always fails is given up
def withdraw_three(tmp_path, monkeypatch, failing):
    """Three background grafts whose tools wait; the call that made them is withdrawn. `stop_one`
    raises every time for the grafts in `failing`. Returns (calls to stop_one, the world, store, h)."""
    import graft.machine as machine
    real, calls = machine.stop_one, []

    def stop(ctx, inst, event):
        child = inst.stop_try[1] if inst.stop_try and inst.stop_try[0] is event else machine._stop_target(inst)
        calls.append(child)
        if child in failing or failing == "all":
            raise RuntimeError(f"INJECTED stop_one {child}")
        return real(ctx=ctx, inst=inst, event=event)
    monkeypatch.setattr(machine, "stop_one", stop)
    world, gate = SeveralWaiting(), threading.Event()
    tools = {"website": use("w1", "site_write", page="i", content="x"),
             "slides": use("s1", "slides_create", title="D"),
             "calendar": use("k1", "calendar_book", what="x", date="2026-12-05", time="10:00")}

    def policy(system, messages, tool_list):
        names = {t["name"] for t in tool_list}
        last = messages[-1]["content"][-1]
        if "branch" in names:
            if last["type"] == "tool_result":
                assert gate.wait(WAIT_TIMEOUT)
                return text("Started.")
            if is_wake(last):
                return text("Report.")
            return [clone_use(*[clone_entry(skills=[k], focus=False) for k in tools], id="k1") | {"type": "tool_use"}]
        skill = "website" if "site_write" in names else "slides" if "slides_create" in names else "calendar"
        return [tools[skill]]

    store, h, said = make(tmp_path=tmp_path, policy=policy, world=world)
    out = {}

    async def go():
        await h.start()
        call = asyncio.create_task(h.user_says(REQUEST))
        for name in ("site_write", "slides_create", "calendar_book"):
            while name not in world.started:
                await asyncio.sleep(0.01)
        call.cancel()
        gate.set()
        try:
            await call
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)
        out["calls"] = len(calls)
        await asyncio.wait_for(h.close(), WAIT_TIMEOUT)

    asyncio.run(go())
    return out, calls, world, said


def test_a_stop_that_always_fails_is_given_up_after_a_bounded_number_of_attempts(tmp_path, monkeypatch):
    from graft.machine import MAX_STOP_ATTEMPTS
    out, calls, world, said = withdraw_three(tmp_path=tmp_path, monkeypatch=monkeypatch, failing="all")
    assert out["calls"] <= 3 * MAX_STOP_ATTEMPTS  # bounded: never a spin
    assert any("could not be stopped" in line for line in said)  # each give-up is told


def test_a_graft_whose_stop_always_fails_does_not_keep_the_others_from_stopping(tmp_path, monkeypatch):
    from graft.machine import MAX_STOP_ATTEMPTS
    out, calls, world, _ = withdraw_three(tmp_path=tmp_path, monkeypatch=monkeypatch, failing={"c1"})
    # every counted try is made, on its graft: c1's three, then c2's and c3's, never None
    assert calls[:MAX_STOP_ATTEMPTS] == ["c1"] * MAX_STOP_ATTEMPTS and calls.count("c1") == MAX_STOP_ATTEMPTS
    assert None not in calls
    assert {"slides_create", "calendar_book"} <= world.cancelled  # c2 and c3 were stopped at the withdrawal


class Abort2(BaseException):
    pass


class BadMessage(BaseException):
    """A non-Exception whose own message raises another non-Exception."""

    def __str__(self):
        raise Abort2("no message")


def test_an_io_write_failure_whose_message_raises_still_has_an_outcome(tmp_path):
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    real = h.record_join

    def broken(**kw):
        if kw.get("child") == "c1":
            raise BadMessage()
        return real(**kw)
    h.record_join = broken
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True


def test_a_row_write_failure_whose_message_raises_still_has_an_outcome(tmp_path):
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    real = h.store_result

    def broken(**kw):
        if kw["use"]["name"] == "site_write":
            raise BadMessage()
        return real(**kw)
    h.store_result = broken
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True


# ------------------------------------------------------------------ no step escapes the graph
class BadExc(Exception):
    """An Exception whose message raises a non-Exception."""

    def __str__(self):
        raise Abort2("no message")


def test_a_refusal_row_whose_write_error_has_a_bad_message_still_ends_the_run(tmp_path):
    def policy(system, messages, tools):
        last = messages[-1]["content"][-1]
        if last["type"] == "tool_result":
            return text("Refused.")
        return [clone_use(clone_entry(skills=["website"], focus=True), clone_entry(skills=["slides"], focus=True),
                          id="k1") | {"type": "tool_use"}]  # two focus entries: refused in its preparation

    store, h, _ = make(tmp_path=tmp_path, policy=policy)
    real = h.store_result

    def broken(**kw):
        if kw["use"]["id"] == "k1":
            raise BadExc()
        return real(**kw)
    h.store_result = broken
    out = {}

    async def go():
        await h.start()
        try:
            out["said"] = await asyncio.wait_for(h.user_says(REQUEST), WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            out["said"] = "TIMEOUT"
        except BaseException as error:  # noqa: BLE001 - the write's error reaches the caller
            out["said"] = type(error).__name__
        await asyncio.wait_for(h.close(), WAIT_TIMEOUT)
        out["closed"] = True

    asyncio.run(go())
    assert out["said"] != "TIMEOUT" and out["closed"] is True


def test_an_effect_that_raises_a_base_exception_is_a_fault_the_machine_recovers_from(tmp_path, monkeypatch):
    import graft.machine as machine
    real, fired = machine.take_message, []

    def raising(ctx, inst, event):
        if not fired:
            fired.append(1)
            raise Abort("an effect's abort")
        return real(ctx=ctx, inst=inst, event=event)
    monkeypatch.setattr(machine, "take_message", raising)
    store, h, _ = make(tmp_path=tmp_path, policy=lambda *a: text("ok"))
    out = {}

    async def go():
        await h.start()
        try:
            out["first"] = await asyncio.wait_for(h.user_says("hello"), WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            out["first"] = "TIMEOUT"
        except BaseException as error:  # noqa: BLE001 - the failed step's caller gets its error
            out["first"] = type(error).__name__
        out["second"] = await asyncio.wait_for(h.user_says("again"), WAIT_TIMEOUT)
        try:
            await asyncio.wait_for(h.settle(), WAIT_TIMEOUT)
        except BaseException:  # noqa: BLE001 - told once
            pass
        await asyncio.wait_for(h.close(), WAIT_TIMEOUT)
        out["closed"] = True

    asyncio.run(go())
    assert fired and out["first"] != "TIMEOUT" and out["second"] == ["ok"] and out["closed"] is True


class BadStop(StopIteration):
    def __str__(self):
        raise Abort2("no message")


def test_a_model_call_raising_a_stop_iteration_with_a_bad_message_fails_its_run_without_a_timeout(tmp_path):
    """The worker thread delivers an outcome whatever its call raises, so with no call timeout
    (call_timeout=None) nothing waits for ever."""
    def policy(system, messages, tools):
        if "branch" in {t["name"] for t in tools}:
            return background_original([clone_entry(skills=["website"], focus=False)])(messages[-1]["content"][-1],
                                                                                         messages)
        raise BadStop()

    store, h, _ = make(tmp_path=tmp_path, policy=policy, call_timeout=None)
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True
    assert store.timeline("c1")["join_reason"] == "failed"


def test_a_seeder_that_raises_a_base_exception_refuses_the_clone_call_and_leaves_no_audit_open(tmp_path):
    from graft.db import Store
    from graft.providers import FakeProvider
    from graft.harness import Harness
    from graft.skills import install

    class AbortingSeeder:
        name = "aborting"

        async def score(self, purpose, cands, trigger):
            raise Abort("seeder abort")

    def policy(system, messages, tools):
        last = messages[-1]["content"][-1]
        if last["type"] == "tool_result":
            return text("Could not start it.")
        return [clone_use(clone_entry(skills=["website"], focus=False), id="k1") | {"type": "tool_use"}]

    store = Store(str(tmp_path / "p.db"))
    install(store)
    h = Harness(store, FakeProvider(policy), seeder=AbortingSeeder())
    out = {}

    async def go():
        await h.start()
        try:
            out["said"] = await asyncio.wait_for(h.user_says(REQUEST), WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            out["said"] = "TIMEOUT"
        await asyncio.wait_for(h.close(), WAIT_TIMEOUT)

    asyncio.run(go())
    assert out["said"] != "TIMEOUT"
    assert store.con.execute("SELECT count(*) FROM transactions WHERE status='open'").fetchone()[0] == 0
    assert store.con.execute("SELECT count(*) FROM timelines WHERE source_id IS NOT NULL").fetchone()[0] == 0


# ------------------------------------------------------------------ a CancelledError is a leaving only when the state is left
def test_a_background_model_call_that_raises_cancelled_error_fails_its_run(tmp_path):
    def policy(system, messages, tools):
        if "branch" in {t["name"] for t in tools}:
            return background_original([clone_entry(skills=["website"], focus=False)])(messages[-1]["content"][-1],
                                                                                         messages)
        raise asyncio.CancelledError()  # a provider SDK's own cancellation surfacing

    store, h, _ = make(tmp_path=tmp_path, policy=policy)
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True
    assert store.timeline("c1")["join_reason"] == "failed"


def say_and_close(h):
    out = {}

    async def go():
        await h.start()
        try:
            out["said"] = await asyncio.wait_for(h.user_says(REQUEST), WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            out["said"] = "TIMEOUT"
        except BaseException as error:  # noqa: BLE001 - the failure reaches the caller
            out["said"] = type(error).__name__
        try:
            await asyncio.wait_for(h.close(), WAIT_TIMEOUT)
            out["closed"] = True
        except asyncio.TimeoutError:
            out["closed"] = "TIMEOUT"
    asyncio.run(go())
    return out


def test_an_original_model_call_that_raises_cancelled_error_fails_its_run(tmp_path):
    def policy(system, messages, tools):
        raise asyncio.CancelledError()

    store, h, _ = make(tmp_path=tmp_path, policy=policy)
    out = say_and_close(h)
    assert out["said"] != "TIMEOUT" and out["closed"] is True


def test_a_host_write_that_raises_cancelled_error_is_its_state_s_failure(tmp_path):
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    real = h.record_join

    def broken(**kw):
        if kw.get("child") == "c1":
            raise asyncio.CancelledError()
        return real(**kw)
    h.record_join = broken
    out = drive_one(h=h, store=store)
    assert out["joined"] is True and out["closed"] is True


def test_a_clone_preparation_that_raises_cancelled_error_is_refused(tmp_path):
    store, h, _ = one_background(World(), tmp_path=tmp_path)

    async def broken(**kw):
        raise asyncio.CancelledError()
    h.prepare_clone = broken
    out = say_and_close(h)
    assert out["said"] != "TIMEOUT" and out["closed"] is True


def test_a_seeder_that_raises_cancelled_error_leaves_no_audit_open(tmp_path):
    from graft.db import Store
    from graft.providers import FakeProvider
    from graft.harness import Harness
    from graft.skills import install

    class CancellingSeeder:
        name = "cancelling"

        async def score(self, purpose, cands, trigger):
            raise asyncio.CancelledError()

    def policy(system, messages, tools):
        if messages[-1]["content"][-1]["type"] == "tool_result":
            return text("Could not start it.")
        return [clone_use(clone_entry(skills=["website"], focus=False), id="k1") | {"type": "tool_use"}]

    store = Store(str(tmp_path / "p.db"))
    install(store)
    h = Harness(store, FakeProvider(policy), seeder=CancellingSeeder())
    out = say_and_close(h)
    assert out["said"] != "TIMEOUT" and out["closed"] is True
    assert store.con.execute("SELECT count(*) FROM transactions WHERE status='open'").fetchone()[0] == 0


def test_an_error_whose_type_name_cannot_be_had_is_described_as_an_error():
    from graft.machine import described

    class Nameless(type):
        @property
        def __name__(cls):
            raise RuntimeError("no name")

    class Odd(Exception, metaclass=Nameless):
        pass

    assert described(Odd("its message")) == "error: its message"


def test_a_stop_whose_counting_always_fails_is_still_bounded_and_stops_every_graft(tmp_path, monkeypatch):
    """The bound holds even when the behavior that counts a try fails: the stop counts it itself."""
    import graft.machine as machine
    counted = []

    def always(ctx, inst, event):
        counted.append(1)
        raise RuntimeError("INJECTED count_stop")
    monkeypatch.setattr(machine, "count_stop", always)
    out, calls, world, said = withdraw_three(tmp_path=tmp_path, monkeypatch=monkeypatch, failing=set())
    assert len(counted) == 3  # one try per graft, each stopped by it
    assert {"site_write", "slides_create", "calendar_book"} <= world.cancelled


def test_several_kept_errors_that_are_not_exceptions_are_raised_together(tmp_path, monkeypatch):
    """Errors kept for settle that are not Exceptions are raised in one group, never a TypeError."""
    import graft.machine as machine

    def always(ctx, inst, event):
        raise Abort("INJECTED abort")
    monkeypatch.setattr(machine, "tell_given_up", always)  # runs at every rest gate: kept again and again
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    out = {}

    async def go():
        await h.start()
        for key, coro in (("said", lambda: h.user_says(REQUEST)), ("joined", lambda: h.join_queued("c1")),
                          ("settled", h.settle), ("settled2", h.settle), ("closed", h.close)):
            try:
                await asyncio.wait_for(coro(), WAIT_TIMEOUT)
                out[key] = "ok"
            except asyncio.TimeoutError:
                out[key] = "TIMEOUT"
            except BaseException as error:  # noqa: BLE001 - what each call raised
                out[key] = type(error).__name__
    asyncio.run(go())
    assert "TypeError" not in out.values() and "TIMEOUT" not in out.values() and out["closed"] == "ok"


def test_an_effect_that_raises_stop_iteration_fails_its_callers_with_an_error_they_can_carry(tmp_path, monkeypatch):
    import graft.machine as machine

    def bad(ctx, inst, event):
        raise StopIteration("a bug")
    monkeypatch.setattr(machine, "take_message", bad)
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    out = {}

    async def go():
        await h.start()
        for key, coro in (("said", lambda: h.user_says(REQUEST)), ("settled", h.settle), ("closed", h.close)):
            try:
                await asyncio.wait_for(coro(), WAIT_TIMEOUT)
                out[key] = "ok"
            except asyncio.TimeoutError:
                out[key] = "TIMEOUT"
            except BaseException as error:  # noqa: BLE001 - what each call raised
                out[key] = type(error).__name__
    asyncio.run(go())
    assert out["said"] not in ("TIMEOUT", "ok") and "TIMEOUT" not in out.values() and out["closed"] == "ok"


def test_failing_every_caller_with_an_error_a_future_cannot_carry_gives_each_one_it_can():
    from graft.machine import _fail_all

    async def go():
        loop = asyncio.get_running_loop()
        replies = [loop.create_future(), loop.create_future()]
        assert _fail_all(replies=replies, error=StopIteration("a bug")) is True
        return [type(r.exception()).__name__ for r in replies]
    assert asyncio.run(go()) == ["RuntimeError", "RuntimeError"]


def test_a_rest_step_that_always_fails_still_answers_the_waiting_call(tmp_path, monkeypatch):
    """A fault on the way to rest that comes back every time: the user's call gets the error (it
    fails closed), and settle and close return."""
    import graft.machine as machine

    def always(ctx, inst, event):
        raise RuntimeError("INJECTED come_to_rest")
    monkeypatch.setattr(machine, "come_to_rest", always)
    store, h, _ = one_background(World(), tmp_path=tmp_path)
    out = {}

    async def go():
        await h.start()
        for key, coro in (("said", lambda: h.user_says(REQUEST)), ("settled", h.settle), ("closed", h.close)):
            try:
                await asyncio.wait_for(coro(), WAIT_TIMEOUT)
                out[key] = "ok"
            except asyncio.TimeoutError:
                out[key] = "TIMEOUT"
            except BaseException as error:  # noqa: BLE001 - what each call raised
                out[key] = type(error).__name__
    asyncio.run(go())
    assert out["said"] == "RuntimeError" and out["settled"] != "TIMEOUT" and out["closed"] == "ok"


def test_a_stop_whose_counting_and_stopping_both_always_fail_is_still_given_up(tmp_path, monkeypatch):
    """A double fault: the try is counted on the fault path, so the bound holds and each stop is
    given up after MAX_STOP_ATTEMPTS."""
    import graft.machine as machine
    from graft.machine import MAX_STOP_ATTEMPTS
    counted = []

    def count_always(ctx, inst, event):
        counted.append(1)
        raise RuntimeError("INJECTED count_stop")
    monkeypatch.setattr(machine, "count_stop", count_always)
    out, calls, world, said = withdraw_three(tmp_path=tmp_path, monkeypatch=monkeypatch, failing="all")
    assert len(calls) == 3 * MAX_STOP_ATTEMPTS and len(counted) == 3 * MAX_STOP_ATTEMPTS
    assert any("could not be stopped" in line for line in said)


def test_a_provider_cancel_reaches_the_caller_as_a_runtime_error_and_withdraws_nothing(tmp_path):
    calls = []

    def policy(system, messages, tools):
        calls.append(1)
        if len(calls) == 1:
            raise asyncio.CancelledError()  # the provider's own cancel, not the caller's
        return text("Hello again.")

    store, h, _ = make(tmp_path=tmp_path, policy=policy)
    out = {}

    async def go():
        await h.start()
        try:
            await asyncio.wait_for(h.user_says("hello"), WAIT_TIMEOUT)
            out["first"] = "ok"
        except BaseException as error:  # noqa: BLE001 - what the caller got
            out["first"] = type(error).__name__
        out["second"] = await asyncio.wait_for(h.user_says("hello again"), WAIT_TIMEOUT)
        await h.close()

    asyncio.run(go())
    assert out == {"first": "RuntimeError", "second": ["Hello again."]}
    assert store.con.execute("SELECT count(*) FROM withdrawals").fetchone()[0] == 0  # nothing withdrawn
