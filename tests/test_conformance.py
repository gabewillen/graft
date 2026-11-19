"""Language-neutral conformance (spec: Host Interface, Conformance): each tests/conformance/*.json
file is JSON-RPC 2.0 requests in and the responses and stream events expected out, with the model's
replies scripted per speaker. A rebuild in another language, or a transport codec, replays the same
files.

A fixture: {"name", "script": {"main": [replies], "<skill>": [replies]}, "steps": [step]}. A reply is
a list of content blocks (text, tool_use); main's replies are taken in order by main's calls, a
skill's by the calls of the branch that holds it (a branch's call waits until a `release` step for
its skill, so the order of events is fixed). A step is {"request": <JSON-RPC request>, "expect":
<JSON-RPC response, or for SendStreamingMessage and SubscribeToTask the list of results>} or
{"release": "<skill>"}, or {"await_call": "<skill>"} (wait until the branch holding that skill has
called the model, so it exists and is running), or {"await_text": ["<task id>", "<agent text>"]} (wait
until the task's history holds that agent text). Ids: the context is "ctx", task ids are t1, t2, ... in order, and agent
message ids are "ctx-m<log position>". Set GRAFT_RECORD_CONFORMANCE=1 to write `expect` for steps
that have none (then review the file by hand)."""

import asyncio
import glob
import itertools
import json
import os
import threading

import pytest

from a2a_schema import check
from graft import a2a
from graft.agent import GraftAgent, GraftConfig
from graft.db import Store
from graft.fake import FakeWorld
from graft.providers import FakeProvider

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = sorted(glob.glob(os.path.join(HERE, "conformance", "*.json")))
GATE = 5


class Scripted:
    """The scripted model: main's replies in order; a branch's by the skill it holds, each call
    waiting for that skill's release."""

    def __init__(self, script):
        self.queues = {who: list(replies) for who, replies in script.items()}
        self.gates = {who: threading.Event() for who in script if who != "main"}
        self.called = {who: threading.Event() for who in script if who != "main"}

    def __call__(self, system, messages, tools):
        names = {t["name"] for t in tools}
        if "fork" in names:
            who = "main"
        else:
            who = next(w for w in self.gates if any(n.startswith({"website": "site_", "slides": "slides_",
                                                                  "calendar": "calendar_"}.get(w, w)) for n in names))
            self.called[who].set()
            assert self.gates[who].wait(GATE), f"{who} was never released"
        assert self.queues[who], f"the script has no reply left for {who}"
        return self.queues[who].pop(0)


async def dispatch(agent: GraftAgent, request: dict):
    """One JSON-RPC request: its response object, or for a streaming method the list of results."""
    rid, method, params = request.get("id"), request.get("method"), request.get("params", {})

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def error(e: a2a.A2AError):
        return {"jsonrpc": "2.0", "id": rid, "error": e.to_dict()}

    try:
        if method == "SendMessage":
            return ok((await agent.send(params)).to_dict())
        if method == "GetTask":
            return ok(agent.get_task(params).to_dict())
        if method == "ListTasks":
            return ok(agent.list_tasks(params).to_dict())
        if method == "CancelTask":
            return ok((await agent.cancel_task(params)).to_dict())
        if method == "GetAgentCard":
            return ok(agent.card().to_dict())
        if method in ("SendStreamingMessage", "SubscribeToTask"):
            stream = agent.stream(params) if method == "SendStreamingMessage" else agent.subscribe(params)
            return [ok(event.to_dict()) async for event in stream]
        if method in ("CreateTaskPushNotificationConfig", "GetTaskPushNotificationConfig",
                      "ListTaskPushNotificationConfigs", "DeleteTaskPushNotificationConfig"):
            raise a2a.PushNotificationNotSupported("push notifications are not supported")
        if method == "GetExtendedAgentCard":
            raise a2a.UnsupportedOperation("there is no extended agent card")
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
    except a2a.A2AError as e:
        return error(e)


def _results(out):
    for item in out if isinstance(out, list) else [out]:
        yield item


@pytest.mark.parametrize("path", FIXTURES, ids=[os.path.basename(p) for p in FIXTURES])
def test_conformance(tmp_path, path):
    fixture = json.load(open(path))
    model = Scripted(fixture["script"])
    world = FakeWorld()
    count = itertools.count(1)
    agent = GraftAgent(Store(str(tmp_path / "c.db")), FakeProvider(model), tools=world.tools(),
                       config=GraftConfig(seeder=None, context_id="ctx", ids=lambda: f"t{next(count)}"))
    recorded = False

    async def go():
        nonlocal recorded
        async with agent:
            for step in fixture["steps"]:
                if "release" in step:
                    model.gates[step["release"]].set()
                    continue
                if "await_text" in step:
                    task_id, text = step["await_text"]
                    for _ in range(GATE * 100):
                        if any(m.role == a2a.ROLE_AGENT and m.text == text
                               for m in agent.get_task({"id": task_id}).history):
                            break
                        await asyncio.sleep(0.01)
                    else:
                        raise AssertionError(step)
                    continue
                if "await_call" in step:
                    assert await asyncio.to_thread(model.called[step["await_call"]].wait, GATE), step
                    continue
                out = await dispatch(agent, step["request"])
                for item in _results(out):
                    result = item.get("result")
                    if result is None:
                        continue
                    kind = {"SendMessage": "Send Message Response", "GetTask": "Task", "CancelTask": "Task",
                            "ListTasks": "List Tasks Response", "GetAgentCard": "Agent Card"}.get(
                        step["request"]["method"], "Stream Response")
                    check(result, kind)
                if "expect" not in step and os.environ.get("GRAFT_RECORD_CONFORMANCE"):
                    step["expect"], recorded = out, True
                    continue
                assert out == step["expect"], (step["request"]["method"], json.dumps(out, indent=1)[:4000])

    asyncio.run(go())
    if recorded:
        with open(path, "w") as f:
            json.dump(fixture, f, indent=1)
            f.write("\n")
