"""Scripted user simulator: reproducible conversations without a human, through the host interface
(`GraftAgent`, or the subagent baseline behind the same interface)."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Union

from . import a2a

Step = Union[str, Callable[[Any], "str | None"]]
_ids = itertools.count(1)


def user_message(text: str, task_id: str | None = None) -> dict:
    """A SendMessageRequest's JSON for one user text (a fresh messageId)."""
    message = {"messageId": f"sim-{next(_ids)}", "role": a2a.ROLE_USER, "parts": [{"text": text}]}
    if task_id is not None:
        message["taskId"] = task_id
    return {"message": message}


def replies(task: a2a.Task, after: str | None = None) -> list[str]:
    """The agent's texts in a task's history (after the message `after`, if given)."""
    out, seen = [], after is None
    for m in task.history:
        if m.message_id == after:
            seen = True
            continue
        if seen and m.role == a2a.ROLE_AGENT and not (m.metadata or {}).get("graft", {}).get("error"):
            out.append(m.text)
    return out


async def say(agent, text: str, task_id: str | None = None) -> tuple[a2a.Task, list[str]]:
    """One message, blocking (SendMessage): its task when main's turn ends, and the agent's texts
    on it after the message."""
    request = user_message(text, task_id)
    task = (await agent.send(request)).task
    return task, replies(task, after=request["message"]["messageId"])


async def follow(agent, task_ids: list[str]) -> list[str]:
    """Follow each task until it is terminal (SubscribeToTask): the agent's texts told meanwhile.
    An agent without streaming (the subagent baseline) has no task left open."""
    subscribe = getattr(agent, "subscribe", None)
    out: list[str] = []
    for task_id in task_ids if subscribe else []:
        if agent.get_task({"id": task_id}).status.state in a2a.TERMINAL:
            continue
        seen: set[str] = set()
        async for event in subscribe({"id": task_id}):
            if event.task is not None:
                seen |= {m.message_id for m in event.task.history}
                continue
            message = event.status_update.status.message
            if message is not None and message.message_id not in seen:
                seen.add(message.message_id)
                if message.role == a2a.ROLE_AGENT and not (message.metadata or {}).get("graft", {}).get("error"):
                    out.append(message.text)
    return out


async def run_script(agent, steps: list[Step], echo: bool = True, world: Any = None) -> list[tuple[str, str]]:
    """Each step is a user message, or a callable of `world` (the run's simulated tools' state)
    returning one (or None to skip). Each is sent as a
    new task; once the script is done, every task still open is followed until it is terminal, so
    background work is reported before scoring."""
    transcript: list[tuple[str, str]] = []

    def heard(texts: list[str]) -> None:
        for reply in texts:
            transcript.append(("assistant", reply))
            if echo:
                print(f"ASSISTANT: {reply}")

    tasks: list[str] = []
    for step in steps:
        text = step(world) if callable(step) else step
        if text is None:
            continue
        transcript.append(("user", text))
        if echo:
            print(f"\nUSER: {text}")
        task, said = await say(agent, text)
        heard(said)
        tasks.append(task.id)
    heard(await follow(agent, tasks))
    return transcript


async def ask(agent, text: str, task_id: str | None = None) -> list[str]:
    """Send one message and follow its task until it is terminal (SendStreamingMessage): the
    agent's texts in order, a harness error line included (it is raised after). An agent without
    streaming (the subagent baseline) answers within `send`."""
    stream = getattr(agent, "stream", None)
    request = user_message(text, task_id)
    if stream is None:
        return replies((await agent.send(request)).task, after=request["message"]["messageId"])
    out, seen = [], {request["message"]["messageId"]}
    async for event in stream(request):
        messages = event.task.history if event.task is not None else [event.status_update.status.message]
        for m in messages:
            if m is not None and m.role == a2a.ROLE_AGENT and m.message_id not in seen:
                seen.add(m.message_id)
                out.append(m.text)
    return out
