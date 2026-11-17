"""A host tool may return an awaitable: the call is the activity of its timeline's `calling` state,
awaited there while every other timeline goes on. Leaving that state (a cancel, a withdrawal,
shutdown) cancels the tool where it waits, and the call is recorded as run with an error, never
as not executed: it may have acted before it stopped."""

import asyncio
import threading

from graft.skills import World
from test_parallel import (REQUEST, WAIT_TIMEOUT, clone_entry, clone_use, is_wake, make, text, tool_results,
                           violations)

STOP = "stop the website"
CANCELLED_WHILE_RUNNING = "cancelled while running: it may have acted before it stopped"


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
        if "clone" in names:
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
        if "clone" in names:
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
        if "clone" in names:
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


def test_a_synchronous_tool_that_raises_is_still_closed_as_not_run(tmp_path):
    """The `calling` state's error: the run ends with the tool's error, and the call, which never
    returned, is closed as not executed (as before awaited tools)."""
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
    assert h.graft_errors == [("c1", "RuntimeError: backend down")]
    assert tool_results(store=store, tl="c1") == [("site_write", "not executed: the turn was interrupted", True)]
    assert h.tools_run("c1") == []


# ------------------------------------------------------------------ a frozen graft's close-out that cannot be written
class RaisingTool(World):
    def execute(self, name, args):
        raise RuntimeError("backend down")


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
    store, h, _ = make(tmp_path=tmp_path, policy=policy, world=RaisingTool())
    fired = fail_once(h=h, name="close_tool_uses")

    async def go():
        await h.start()
        await h.user_says(REQUEST)
        await asyncio.wait_for(h.join_queued("c1"), WAIT_TIMEOUT)
        await settled(h)
        await h.close()

    asyncio.run(go())
    assert fired == ["close_tool_uses"] and store.timeline("c1")["status"] == "joined"
    assert tool_results(store=store, tl="c1") == [("site_write", "not executed: the turn was interrupted", True)]
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

    store, h, _ = make(tmp_path=tmp_path, policy=graft_writes(original), world=RaisingTool())
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
    assert tool_results(store=store, tl="c1") == [("site_write", "not executed: the turn was interrupted", True)]
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
