"""graft's host interface is A2A 1.0.1 (spec: Host Interface): every object it emits is an A2A object
(checked against the vendored schema and the proto's REQUIRED fields), a task is a user's request
with all the work it sets off, a blocking SendMessage returns when main's turn ends, CancelTask is
Stop, and everything is told through the event log."""

import asyncio
import itertools
import json
import threading

import pytest

from _util import conversation, holder, status
from a2a_schema import check, problems
from graft import a2a
from graft.agent import GraftAgent, GraftConfig, HarnessError, fold
from graft.db import Store
from graft.fake import FakeWorld
from graft.providers import FakeProvider

GATE = 5  # seconds a scripted model call waits for its gate before the test fails
REQUEST = "Build a website for the dental practice."


def text(t):
    return [{"type": "text", "text": t}]


FORK_IDS = itertools.count(1)


def fork(**inp):
    """A fork call (each with its own tool_use id)."""
    return [{"type": "tool_use", "id": f"f{next(FORK_IDS)}", "name": "fork", "input": inp}]


def last_block(messages):
    return messages[-1]["content"][-1]


def message(text_, message_id="m1", task_id=None, **extra):
    m = {"messageId": message_id, "role": "ROLE_USER", "parts": [{"text": text_}], **extra}
    if task_id is not None:
        m["taskId"] = task_id
    return {"message": m}


class Policy:
    """Main forks one branch for REQUEST (`current` or `detached`), says "Started." after a
    detached fork, and reports a merge; the branch waits for `gate` and replies."""

    def __init__(self, mode=None):
        self.mode = mode
        self.gate = threading.Event()
        self.branch_started = threading.Event()

    def __call__(self, system, messages, tools):
        names = {t["name"] for t in tools}
        last = last_block(messages)
        if "fork" in names:
            if last["type"] == "tool_result" and str(last.get("content", "")).startswith("forked"):
                return text("Started.")
            if last["type"] == "tool_result":  # a merge result: the wake
                return text("Report: done.") if self.mode == "detached" else []
            if self.mode is None or REQUEST not in json.dumps(messages[-1]):
                return text("Hello.")
            entry = {"skills": ["website"], "messages": [REQUEST], "reason": "Build the site."}
            return fork(**({"current": entry} if self.mode == "current" else {"detached": [entry]}))
        self.branch_started.set()
        assert self.gate.wait(GATE)
        return text("Which pages do you want?" if self.mode == "current" else "Site built.")


def make(*, tmp_path, policy, **config):
    store = Store(str(tmp_path / "a2a.db"))
    chat = conversation(store, policy, seeder=None, quote=False, **config)
    return store, chat.agent


async def until(event, what):
    assert await asyncio.to_thread(event.wait, GATE), f"timed out waiting for {what}"


def every_event_is_a2a(store):
    for _seq, event in store.events_after(0, limit=10 ** 6):
        check(event, "Stream Response")


# ------------------------------------------------------------------ shapes
def test_every_emitted_object_is_an_a2a_object(tmp_path):
    store, agent = make(tmp_path=tmp_path, policy=Policy(mode=None))

    async def go():
        async with agent:
            response = await agent.send(message("Hi"))
            check(response.to_dict(), "Send Message Response")
            check(agent.get_task({"id": response.task.id}).to_dict(), "Task")
            check(agent.list_tasks({}).to_dict(), "List Tasks Response")
            check(agent.card().to_dict(), "Agent Card")
            events = [e.to_dict() async for e in agent.stream(message("Again", message_id="m2"))]
            for e in events:
                check(e, "Stream Response")
            return response, events

    response, events = asyncio.run(go())
    assert response.task.status.state == a2a.COMPLETED
    assert [m.role for m in response.task.history] == [a2a.ROLE_USER, a2a.ROLE_AGENT]
    assert response.task.status.message.text == "Hello."
    assert "task" in events[0] and events[-1]["statusUpdate"]["status"]["state"] == a2a.COMPLETED
    every_event_is_a2a(store)


def test_the_shape_check_has_teeth():
    assert problems({}, "Task")  # the schema alone would take {}
    assert problems({"id": "t", "status": {"state": "TASK_STATE_UNSPECIFIED"}}, "Task")
    assert problems({"id": "t", "status": {"state": 3}}, "Task")  # an enum number, not its name
    assert problems({"id": "t", "context_id": "c", "status": {"state": "TASK_STATE_WORKING"}}, "Task")
    assert problems({"messageId": "m", "role": "ROLE_AGENT", "parts": []}, "Message")
    assert not problems({"id": "t", "status": {"state": "TASK_STATE_WORKING"}}, "Task")


def test_the_card_names_the_capability_skills_and_streaming(tmp_path):
    _, agent = make(tmp_path=tmp_path, policy=Policy())
    card = agent.card().to_dict()
    check(card, "Agent Card")
    assert {s["id"] for s in card["skills"]} >= {"website", "slides", "calendar"}
    assert not {"fork", "merge", "detached"} & {s["id"] for s in card["skills"]}
    assert card["capabilities"] == {"streaming": True, "pushNotifications": False, "extendedAgentCard": False}


# ------------------------------------------------------------------ requests are untrusted input
@pytest.mark.parametrize("bad, error", [
    ({}, a2a.InvalidParams),
    ({"message": {"role": "ROLE_USER", "parts": [{"text": "x"}]}}, a2a.InvalidParams),  # no messageId
    ({"message": {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": "x"}]}}, a2a.InvalidParams),
    ({"message": {"messageId": "m", "role": "user", "parts": [{"text": "x"}]}}, a2a.InvalidParams),
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": []}}, a2a.InvalidParams),
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [{"raw": "AAAA"}]}}, a2a.ContentTypeNotSupported),
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": "x" * (a2a.MAX_TEXT_CHARS + 1)}]}},
     a2a.InvalidParams),
    ({"message": {"messageId": "m" * (a2a.MAX_ID_CHARS + 1), "role": "ROLE_USER", "parts": [{"text": "x"}]}},
     a2a.InvalidParams),
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": "x"}] * (a2a.MAX_PARTS + 1)}},
     a2a.InvalidParams),
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": "x"}], "metadata": {"a": "x" * 20000}}},
     a2a.InvalidParams),
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": "x"}]},
      "configuration": {"taskPushNotificationConfig": {"url": "https://x"}}}, a2a.PushNotificationNotSupported),
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": "x"}], "contextId": "elsewhere"}},
     a2a.InvalidParams),
])
def test_a_malformed_request_is_refused_with_no_task_made(tmp_path, bad, error):
    store, agent = make(tmp_path=tmp_path, policy=Policy())

    async def go():
        async with agent:
            with pytest.raises(error):
                await agent.send(bad)

    asyncio.run(go())
    assert store.events_after(0) == []


def test_unknown_and_terminal_task_ids(tmp_path):
    store, agent = make(tmp_path=tmp_path, policy=Policy())

    async def go():
        async with agent:
            with pytest.raises(a2a.TaskNotFound):
                await agent.send(message("Hi", task_id="nope"))
            with pytest.raises(a2a.TaskNotFound):
                agent.get_task({"id": "nope"})
            with pytest.raises(a2a.TaskNotFound):
                await agent.cancel_task({"id": "nope"})
            done = (await agent.send(message("Hi"))).task
            assert done.status.state == a2a.COMPLETED
            with pytest.raises(a2a.UnsupportedOperation):  # a message to a terminal task
                await agent.send(message("More", message_id="m2", task_id=done.id))
            with pytest.raises(a2a.TaskNotCancelable):
                await agent.cancel_task({"id": done.id})
            with pytest.raises(a2a.UnsupportedOperation):
                async for _ in agent.subscribe({"id": done.id}):
                    pass

    asyncio.run(go())


# ------------------------------------------------------------------ turn end, tasks and branches
def test_a_turn_with_detached_work_out_returns_input_required_and_completes_when_it_merges(tmp_path):
    policy = Policy(mode="detached")
    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        async with agent:
            task = (await agent.send(message(REQUEST))).task
            assert task.status.state == a2a.INPUT_REQUIRED  # main's turn ended; the branch still works
            assert task.status.message.text == "Started."
            policy.gate.set()
            events = [e async for e in agent.subscribe({"id": task.id})]
            return task, events

    task, events = asyncio.run(go())
    states = [e.status_update.status.state for e in events[1:]]
    assert states[-1] == a2a.COMPLETED and a2a.WORKING in states
    final = fold(events=[e for _, e in store.task_events(task.id)])
    assert [m.text for m in final.history if m.role == a2a.ROLE_AGENT] == ["Started.", "Report: done."]
    assert store.timeline("c1")["join_reason"] == "done"
    every_event_is_a2a(store)


def test_a_new_message_while_a_task_is_input_required_is_a_new_task(tmp_path):
    policy = Policy(mode="detached")
    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        async with agent:
            first = (await agent.send(message(REQUEST))).task
            second = (await agent.send(message("Thanks!", message_id="m2"))).task
            assert second.id != first.id and second.context_id == first.context_id
            assert second.status.state == a2a.COMPLETED  # nothing of its own is out
            assert agent.get_task({"id": first.id}).status.state == a2a.INPUT_REQUIRED
            policy.gate.set()
            async for _ in agent.subscribe({"id": first.id}):
                pass
            return first

    first = asyncio.run(go())
    assert fold(events=[e for _, e in store.task_events(first.id)]).status.state == a2a.COMPLETED


def test_a_follow_up_with_the_task_id_continues_the_task(tmp_path):
    policy = Policy(mode="detached")
    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        async with agent:
            first = (await agent.send(message(REQUEST))).task
            follow = (await agent.send(message("And keep it formal.", message_id="m2", task_id=first.id))).task
            assert follow.id == first.id
            assert [m.message_id for m in follow.history if m.role == a2a.ROLE_USER] == ["m1", "m2"]
            assert follow.status.state == a2a.INPUT_REQUIRED  # the branch is still out
            policy.gate.set()
            async for _ in agent.subscribe({"id": first.id}):
                pass

    asyncio.run(go())
    every_event_is_a2a(store)


def test_return_immediately_returns_once_the_message_is_in_the_log(tmp_path):
    policy = Policy(mode="detached")
    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        async with agent:
            task = (await agent.send(message(REQUEST) | {"configuration": {"returnImmediately": True}})).task
            assert task.status.state == a2a.SUBMITTED
            policy.gate.set()
            async for _ in agent.subscribe({"id": task.id}):
                pass

    asyncio.run(go())


def test_list_tasks_pages_most_recent_first(tmp_path):
    store, agent = make(tmp_path=tmp_path, policy=Policy())

    async def go():
        async with agent:
            ids = [(await agent.send(message(f"Hi {i}", message_id=f"m{i}"))).task.id for i in range(5)]
            page = agent.list_tasks({"pageSize": 2})
            assert [t.id for t in page.tasks] == ids[::-1][:2] and page.total_size == 5 and page.next_page_token
            rest = agent.list_tasks({"pageSize": 10, "pageToken": page.next_page_token})
            assert [t.id for t in rest.tasks] == ids[::-1][2:] and rest.next_page_token == ""
            assert agent.list_tasks({"status": "TASK_STATE_WORKING"}).tasks == []
            assert agent.list_tasks({"contextId": "elsewhere"}).total_size == 0
            with pytest.raises(a2a.InvalidParams):
                agent.list_tasks({"pageToken": "zz"})

    asyncio.run(go())


# ------------------------------------------------------------------ cancel is Stop
def test_cancel_stops_a_detached_branch_which_merges_cancelled_and_wakes_nothing(tmp_path):
    policy = Policy(mode="detached")
    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        async with agent:
            task = (await agent.send(message(REQUEST))).task
            assert task.status.state == a2a.INPUT_REQUIRED
            await until(policy.branch_started, "the branch's model call")
            canceled = await agent.cancel_task({"id": task.id})
            assert canceled.status.state == a2a.CANCELED
            check(canceled.to_dict(), "Task")
            policy.gate.set()  # the abandoned model call returns; nothing is made of it
            again = (await agent.send(message("Hello?", message_id="m2"))).task
            return task, again

    task, again = asyncio.run(go())
    assert store.timeline("c1")["join_reason"] == "cancelled"
    events = store.task_events(task.id)
    assert fold(events=[e for _, e in events]).status.state == a2a.CANCELED
    assert sum(1 for _, e in events if "statusUpdate" in e and e["statusUpdate"]["status"]["state"] in a2a.TERMINAL) == 1
    assert again.status.state == a2a.COMPLETED
    every_event_is_a2a(store)


def test_cancel_merges_a_current_branch_and_the_conversation_comes_back_to_main(tmp_path):
    policy = Policy(mode="current")
    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        async with agent:
            sending = asyncio.create_task(agent.send(message(REQUEST)))
            await until(policy.branch_started, "the current branch's model call")
            assert holder(store) == "c1"
            canceled = await agent.cancel_task({"id": (await _first_task(agent)).id})
            assert canceled.status.state == a2a.CANCELED
            response = await sending
            assert response.task.status.state == a2a.CANCELED
            policy.gate.set()
            after = (await agent.send(message("Hi again", message_id="m2"))).task
            return after

    after = asyncio.run(go())
    assert store.timeline("c1")["join_reason"] == "cancelled"
    assert status(store=store, tl="c1") == "joined" and holder(store) == "o"  # main holds the conversation
    assert after.status.state == a2a.COMPLETED and after.status.message.text == "Hello."


async def _first_task(agent):
    for _ in range(100):
        page = agent.list_tasks({})
        if page.tasks:
            return page.tasks[-1]
        await asyncio.sleep(0)
    raise AssertionError("no task")


# ------------------------------------------------------------------ errors and shutdown
def test_a_harness_error_is_an_event_and_send_raises_it(tmp_path):
    def policy(system, messages, tools):
        raise ValueError("model exploded")

    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        async with agent:
            with pytest.raises(HarnessError, match="model exploded"):
                await agent.send(message("Hi"))
            return agent.list_tasks({}).tasks[0]

    task = asyncio.run(go())
    assert task.status.state == a2a.FAILED
    assert "model exploded" in task.status.message.text
    assert task.status.message.metadata == {"graft": {"error": True}}
    every_event_is_a2a(store)


def test_close_fails_every_open_task_and_refuses_later_requests(tmp_path):
    policy = Policy(mode="detached")
    store, agent = make(tmp_path=tmp_path, policy=policy)

    async def go():
        await agent.start()
        task = (await agent.send(message(REQUEST))).task
        assert task.status.state == a2a.INPUT_REQUIRED
        await agent.close()
        policy.gate.set()
        return task

    task = asyncio.run(go())
    final = fold(events=[e for _, e in store.task_events(task.id)])
    assert final.status.state == a2a.FAILED and "closed" in final.status.message.text
    every_event_is_a2a(store)
