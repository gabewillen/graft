"""The `Machines` registry: starts, addresses, and stops the session, scribe, and timeline HSMs."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import hsm

from .scribe import SCRIBE, Scribe
from .session import SESSION, Session
from .timeline import TIMELINE
from .timeline_ops import Timeline
from .types import Host, Request, TransitionRefused


def _taken(reply: asyncio.Future) -> None:
    """Consume a request's answer marker exception so asyncio never logs it as unread."""
    if not reply.cancelled():
        reply.exception()


class Machines:
    """Registry of machines and event dispatch interface used by the harness."""

    def __init__(self, host: Host):
        self.host = host
        self.timelines: dict[str, Timeline] = {}
        self.session: Session | None = None
        self.scribe: Scribe | None = None
        self.root: str | None = None
        self.requests = 0

    async def start_root(self, tl: str, context_id: str = "context", position: int = 0) -> None:
        if self.root is not None:
            raise TransitionRefused("root already started")
        self.scribe = Scribe(self)
        await hsm.Started(None, self.scribe, SCRIBE, hsm.Config(ID="scribe"))
        self.session = Session(self, context_id=context_id, position=position)
        self.session.root = tl
        await hsm.Started(None, self.session, SESSION, hsm.Config(ID="session"))
        sm = Timeline(machines=self, timeline_id=tl, source_id=None, kind="original")
        self.timelines[tl] = sm
        self.root = tl
        await hsm.Started(None, sm, TIMELINE, hsm.Config(ID=tl))

    def future(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    def request(self, text: str, task: str | None = None, metadata: dict[str, Any] | None = None) -> Request:
        self.requests += 1
        reply = self.future()
        reply.add_done_callback(_taken)
        return Request(id=self.requests, text=text, reply=reply, task=task,
                       metadata=dict(metadata) if metadata else {})

    def _target(self, tl: str, name: str) -> hsm.Instance:
        sm = self.session if tl == "session" else self.timelines.get(tl)
        if sm is None:
            raise TransitionRefused(f"no machine for {tl}")
        if not sm.state():
            raise TransitionRefused(f"{tl}: '{name}' refused in state closed (the harness is closed)")
        return sm

    async def send(self, tl: str, name: str, data: Any = None) -> None:
        await hsm.Dispatch(None, self._target(tl=tl, name=name), hsm.Event(name=name, data=data))

    def post(self, tl: str, name: str, data: Any = None) -> None:
        hsm.Dispatch(None, self._target(tl=tl, name=name), hsm.Event(name=name, data=data))

    async def ask(self, tl: str, name: str, build: Callable[[asyncio.Future], Any]) -> Any:
        reply = self.future()
        self.post(tl=tl, name=name, data=build(reply))
        return await reply

    async def stop(self) -> None:
        for sm in list(self.timelines.values()):
            if sm.state():
                await hsm.Stop(sm)
        for sm in (self.session, self.scribe):
            if sm is not None and sm.state():
                await hsm.Stop(sm)
