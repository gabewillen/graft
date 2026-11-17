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
        self.cancelled = False

    async def execute(self, name, args):
        if name == "site_write":
            self.started.set()
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

