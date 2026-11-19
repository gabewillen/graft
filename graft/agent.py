"""`GraftAgent`: graft's interface to its host, the A2A operations in-process (spec: Host Interface).

Each method is one A2A operation and takes and returns the A2A objects of `graft.a2a` (a dict in
their JSON form is taken too, validated whole): `send` (SendMessage), `stream`
(SendStreamingMessage), `get_task` (GetTask), `list_tasks` (ListTasks), `cancel_task`
(CancelTask), `subscribe` (SubscribeToTask) and `card` (the Agent Card). A transport codec maps
JSON-RPC (or HTTP+JSON) onto these one to one.

What the agent tells its host is its event log (the store's `a2a_events`), which only the session
machine writes; every method reads it from its own cursor, and waits for more by a request to the
session machine carrying the reply (`events_wait`). The agent holds no queue, lock or future of
its own: the machines decide everything, the agent reads what they wrote.

Lifetime: `await agent.start()` (or `async with agent:`) before the first operation, `await
agent.close()` at the end. One agent is one conversation (its `contextId`) on one store, used from
one event loop. A blocking `send` returns when its task reaches `TASK_STATE_INPUT_REQUIRED` or a
terminal state after its message (main's turn ended); a stream stays open until the task is
terminal. A harness error delivered on the task is raised (`HarnessError`) once delivered.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from . import a2a
from .a2a import (A2AError, AgentCard, AgentInterface, AgentSkill, CancelTaskRequest, GetTaskRequest,
                  InvalidParams, ListTasksRequest, ListTasksResponse, SendMessageRequest, SendMessageResponse,
                  StreamResponse, SubscribeToTaskRequest, Task, TaskNotCancelable, TaskNotFound,
                  UnsupportedOperation)
from .db import Store
from .harness import MODEL_CALL_TIMEOUT, Host
from .machine import Submission
from .providers import Provider
from .skills import BASE_SYSTEM, BUILTIN, CATALOG, install
from .tools import Tool

DEFAULT_PAGE_SIZE = 50
IN_PROCESS_URL = "urn:graft:in-process"
IN_PROCESS_BINDING = "GRAFT-INPROC"  # a custom binding (A2A §5.8): the Python API itself


class HarnessError(A2AError):
    """A harness error delivered on a task (its line is the message): raised by `send`, `stream`
    and `subscribe` once they have delivered it (fail closed)."""
    code, name = -32603, "InternalError"


@dataclass
class GraftConfig:
    """How the agent is made. Everything has a default; none of it reaches the wire but the card's
    fields."""
    persona: str = BASE_SYSTEM
    max_steps: int = 10
    compact_tokens: int | None = None
    compaction_mode: str = "turns"
    seeder: Any = ...  # ...: from the environment (graft/seeder.py); None: no seeder
    window_tokens: int | None | Any = ...  # ...: the provider model's known window
    call_timeout: float | None = MODEL_CALL_TIMEOUT
    names: Callable[[], str] | None = None  # the source of branch ids a fork leaves out
    nonce: str | None = None
    trace: Callable[[str], None] | None = None  # a debug hook for internal trace lines, not a channel
    skills: Sequence | None = None  # the skill catalog (default: graft.skills.CATALOG)
    ids: Callable[[], str] | None = None  # the source of context and task ids (default: uuid4)
    context_id: str | None = None
    name: str = "graft"
    description: str = "One assistant that works on several parts of a request at once."
    version: str = "0.1.0"
    interfaces: list[AgentInterface] = field(default_factory=lambda: [
        AgentInterface(url=IN_PROCESS_URL, protocol_binding=IN_PROCESS_BINDING)])


def _parse(cls, request):
    if isinstance(request, cls):
        return request
    if isinstance(request, dict):
        return cls.from_dict(request)
    raise InvalidParams(f"expected a {cls.__name__} or its JSON object")


def fold(events: list[dict], history_length: int | None = None) -> Task:
    """A task's state from its events: its `task` event, then each `statusUpdate` (whose message
    joins its history once)."""
    first = events[0]["task"]
    task = {"id": first["id"], "contextId": first["contextId"], "status": dict(first["status"]),
            "history": list(first.get("history", []))}
    seen = {m["messageId"] for m in task["history"]}
    for event in events[1:]:
        status = event["statusUpdate"]["status"]
        task["status"] = status
        message = status.get("message")
        if message is not None and message["messageId"] not in seen:
            seen.add(message["messageId"])
            task["history"].append(message)
    if history_length is not None:
        task["history"] = task["history"][-history_length:] if history_length else []
    if not task["history"]:
        task.pop("history")
    return Task.from_dict(task)


def _is_error(event: dict) -> str | None:
    """The harness error line an event carries, if it does."""
    body = event.get("statusUpdate", {}).get("status", {}).get("message") or event.get("message") or {}
    if (body.get("metadata") or {}).get("graft", {}).get("error"):
        return "".join(p.get("text", "") for p in body.get("parts", []))
    return None


class GraftAgent:
    """graft as an A2A agent, in-process (see the module docstring)."""

    def __init__(self, store: Store, provider: Provider, *, tools: Sequence[Tool] = (),
                 config: GraftConfig | None = None):
        self.config = config or GraftConfig()
        self.store = store
        self._ids = self.config.ids or (lambda: str(uuid.uuid4()))
        self.context_id = self.config.context_id or self._ids()
        tools = list(tools)
        if not store.skill_names():
            install(store, tools={t.name: t.definition() for t in tools}, skills=self.config.skills)
        self._host = Host(store, provider, tools=tools, trace=self.config.trace, max_steps=self.config.max_steps,
                          persona=self.config.persona, nonce=self.config.nonce,
                          compact_tokens=self.config.compact_tokens, compaction_mode=self.config.compaction_mode,
                          seeder=self.config.seeder, window_tokens=self.config.window_tokens,
                          call_timeout=self.config.call_timeout, names=self.config.names)

    # ------------------------------------------------------------ lifetime
    async def start(self) -> None:
        """Start the machines (once, before the first operation)."""
        await self._host.start(context_id=self.context_id, position=self.store.event_position())

    async def close(self) -> None:
        """Shut down: running branches stop, every task still open fails, the machines stop."""
        await self._host.close()

    async def __aenter__(self) -> "GraftAgent":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------ discovery
    def card(self) -> AgentCard:
        """The Agent Card: its skills are the capability skills main can fork a branch to hold."""
        rows = self.store.con.execute("SELECT name, description FROM skills ORDER BY skill_id").fetchall()
        skills = [AgentSkill(id=r[0], name=r[0], description=r[1], tags=[r[0]]) for r in rows if r[0] not in BUILTIN]
        return AgentCard(name=self.config.name, description=self.config.description, version=self.config.version,
                         supported_interfaces=list(self.config.interfaces), skills=skills)

    # ------------------------------------------------------------ reading the log
    def _task_events(self, task_id: str, upto: int | None = None) -> list[dict]:
        return [e for _, e in self.store.task_events(task_id, upto=upto)]

    def get_task(self, request: GetTaskRequest | dict) -> Task:
        """GetTask: the task as its events say."""
        req = _parse(GetTaskRequest, request)
        events = self._task_events(req.id)
        if not events:
            raise TaskNotFound(f"task {req.id[:64]!r} not found")
        return fold(events, history_length=req.history_length)

    def list_tasks(self, request: ListTasksRequest | dict | None = None) -> ListTasksResponse:
        """ListTasks: the context's tasks, most recently updated first, a page at a time."""
        req = _parse(ListTasksRequest, request if request is not None else {})
        if req.context_id is not None and req.context_id != self.context_id:
            tasks: list[tuple[int, Task]] = []
        else:
            tasks = []
            for tid in self.store.task_ids():
                events = self.store.task_events(tid)
                task = fold([e for _, e in events], history_length=req.history_length)
                if req.status is None or task.status.state == req.status:
                    tasks.append((events[-1][0], task))
        tasks.sort(key=lambda x: -x[0])
        size = req.page_size or DEFAULT_PAGE_SIZE
        try:
            start = int(req.page_token) if req.page_token else 0
        except ValueError:
            raise InvalidParams("pageToken is not one this agent gave") from None
        if start < 0 or start > len(tasks):
            raise InvalidParams("pageToken is not one this agent gave")
        page = [t for _, t in tasks[start:start + size]]
        end = start + len(page)
        return ListTasksResponse(tasks=page, next_page_token=str(end) if end < len(tasks) else "",
                                 page_size=size, total_size=len(tasks))

    async def _wait(self, cursor: int) -> int:
        """Wait for events past `cursor` (a request to the session, which answers when they are
        written). Returns the position written."""
        return await self._host.machines.ask(tl="session", name="events_wait", build=lambda reply: (cursor, reply))

    async def _events(self, cursor: int, task_id: str) -> AsyncIterator[tuple[int, dict]]:
        """The task's events after `cursor`, as they are written."""
        while True:
            rows = self.store.events_after(cursor)
            for seq, event in rows:
                cursor = seq
                body = event.get("task") or event.get("statusUpdate")
                if body is not None and (body.get("id") if "task" in event else body.get("taskId")) == task_id:
                    yield seq, event
            if not rows:
                await self._wait(cursor)

    # ------------------------------------------------------------ messages
    async def _submit(self, req: SendMessageRequest) -> tuple[str, str, int]:
        """The message goes to the session as a new task or a follow-up. Returns (task id, the
        message's id, the log position before it)."""
        message = req.message
        if message.context_id is not None and message.context_id != self.context_id:
            raise InvalidParams("message.contextId names another context")
        follow_up = message.task_id is not None
        task_id = message.task_id if follow_up else self._ids()
        sent = a2a.Message(message_id=message.message_id, role=message.role, parts=message.parts,
                           context_id=self.context_id, task_id=task_id, metadata=message.metadata,
                           extensions=message.extensions, reference_task_ids=message.reference_task_ids)
        cursor = self.store.event_position()
        request = self._host.machines.request(text=message.text, task=task_id)
        answer = await self._host.machines.ask(tl="session", name="submit", build=lambda reply: (
            Submission(task=task_id, follow_up=follow_up, message=sent.to_dict(), request=request), reply))
        if answer == "not_found":
            raise TaskNotFound(f"task {task_id[:64]!r} not found")
        if answer == "terminal":
            raise UnsupportedOperation(f"task {task_id[:64]!r} is in a terminal state")
        if answer != "accepted":
            raise UnsupportedOperation("the agent is closed")
        return task_id, message.message_id, cursor

    async def _updates(self, task_id: str, message_id: str, cursor: int,
                       until: frozenset[str]) -> AsyncIterator[tuple[int, dict, bool]]:
        """The task's events from its message on, until a state in `until` after it (each with
        whether it is the last). An error event is delivered, then raised at the end."""
        mine, error = False, None
        async for seq, event in self._events(cursor, task_id):
            if not mine:
                carried = event.get("task", {}).get("history", [{}])[-1] if "task" in event else \
                    event["statusUpdate"]["status"].get("message") or {}
                mine = carried.get("messageId") == message_id and carried.get("role") == a2a.ROLE_USER
                if not mine:
                    continue
            error = error or _is_error(event)
            state = (event.get("task") or event["statusUpdate"])["status"]["state"]
            last = state in until and "task" not in event and not (
                state == a2a.SUBMITTED and event["statusUpdate"]["status"].get("message", {}).get("messageId") == message_id)
            yield seq, event, last
            if last:
                break
        if error is not None:
            raise HarnessError(error)

    async def send(self, request: SendMessageRequest | dict) -> SendMessageResponse:
        """SendMessage: always a Task. Blocking (the default) returns at the task's next
        `TASK_STATE_INPUT_REQUIRED` or terminal state after this message; with
        `configuration.returnImmediately`, once its message is in the log."""
        req = _parse(SendMessageRequest, request)
        task_id, message_id, cursor = await self._submit(req)
        until = frozenset() if req.configuration.return_immediately else a2a.INTERRUPTED | a2a.TERMINAL
        upto = cursor
        async for seq, _event, _last in self._updates(task_id, message_id, cursor, until):
            upto = seq
            if req.configuration.return_immediately:
                break
        return SendMessageResponse(task=fold(self._task_events(task_id, upto=upto),
                                             history_length=req.configuration.history_length))

    async def stream(self, request: SendMessageRequest | dict) -> AsyncIterator[StreamResponse]:
        """SendStreamingMessage: the task, then its status updates until it is terminal."""
        req = _parse(SendMessageRequest, request)
        task_id, message_id, cursor = await self._submit(req)
        first = True
        async for seq, event, _last in self._updates(task_id, message_id, cursor, a2a.TERMINAL):
            if first:  # the task as its message left it, then each update after it
                first = False
                yield StreamResponse(task=fold(self._task_events(task_id, upto=seq)))
                continue
            yield StreamResponse.from_dict(event)

    async def subscribe(self, request: SubscribeToTaskRequest | dict) -> AsyncIterator[StreamResponse]:
        """SubscribeToTask: the task as it is now, then its status updates until it is terminal."""
        req = _parse(SubscribeToTaskRequest, request)
        events = self.store.task_events(req.id)
        if not events:
            raise TaskNotFound(f"task {req.id[:64]!r} not found")
        task = fold([e for _, e in events])
        if task.status.state in a2a.TERMINAL:
            raise UnsupportedOperation(f"task {req.id[:64]!r} is in a terminal state")
        yield StreamResponse(task=task)
        error = None
        async for _seq, event in self._events(events[-1][0], req.id):
            error = error or _is_error(event)
            yield StreamResponse.from_dict(event)
            if event["statusUpdate"]["status"]["state"] in a2a.TERMINAL:
                break
        if error is not None:
            raise HarnessError(error)

    async def cancel_task(self, request: CancelTaskRequest | dict) -> Task:
        """CancelTask: Stop. The task is CANCELED, its request withdrawn: its runs end, its
        branches stop and merge cancelled, a current branch holding the conversation for it merges
        too. Returns the task once CANCELED is in the log."""
        req = _parse(CancelTaskRequest, request)
        answer = await self._host.machines.ask(tl="session", name="cancel_task", build=lambda reply: (req.id, reply))
        if answer == "not_found":
            raise TaskNotFound(f"task {req.id[:64]!r} not found")
        if answer == "terminal":
            raise TaskNotCancelable(f"task {req.id[:64]!r} is in a terminal state")
        if answer != "accepted":
            raise UnsupportedOperation("the agent is closed")
        cursor = 0
        async for seq, event in self._events(cursor, req.id):
            body = event.get("statusUpdate")
            if body is not None and body["status"]["state"] == a2a.CANCELED:
                return fold(self._task_events(req.id, upto=seq))
        raise AssertionError("unreachable")  # _events never ends
