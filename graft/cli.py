"""`uv run graft chat` (interactive) and `uv run graft dump DB`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile

from . import a2a
from .agent import GraftAgent, GraftConfig, HarnessError
from .db import Store
from .fake import FakeTools
from .harness import in_daemon_thread
from .providers import make_provider
from .sim import user_message


def _read_line() -> str | None:
    """One line of input (called in a daemon thread, like a model call); None at end of input."""
    try:
        return input()
    except EOFError:
        return None


def _prompt() -> None:
    print("\nyou> ", end="", flush=True)


def _show(message: a2a.Message | None, seen: set[str]) -> None:
    """An agent message, once: replies to stdout, harness errors to stderr."""
    if message is None or message.role != a2a.ROLE_AGENT or message.message_id in seen:
        return
    seen.add(message.message_id)
    if (message.metadata or {}).get("graft", {}).get("error"):
        print(message.text, file=sys.stderr)
    else:
        print(f"assistant> {message.text}")


async def _follow(agent: GraftAgent, stream, seen: set[str]) -> None:
    """Print a task's messages as they are told, until it is terminal."""
    try:
        async for event in stream:
            if event.status_update is not None:
                _show(event.status_update.status.message, seen)
    except HarnessError:
        pass  # its line was printed as it was told
    except a2a.A2AError as error:  # the agent closed before the task ended
        print(f"(task not followed to its end: {error})", file=sys.stderr)


async def chat(db_path: str, trace: bool) -> None:
    store = Store(db_path)
    world = FakeTools()  # the simulated skills' tools
    provider = make_provider()
    agent = GraftAgent(store, provider, tools=world.tools(), config=GraftConfig(trace=print if trace else None))
    await agent.start()
    print(f"graft chat ({provider.name}:{provider.model}); db={db_path}. Commands: /dump, /quit")
    seen: set[str] = set()
    followers: list[asyncio.Task] = []
    try:
        while True:
            # Each message is a task, streamed: its replies print as they are said, and a task whose
            # branches still work stays followed in the background (their reports print as they
            # come). The prompt comes back when main's turn ends.
            _prompt()
            line = await in_daemon_thread(_read_line)
            if line is None:
                break
            text = line.strip()
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/dump":
                print(store.dump())
                continue
            stream = agent.stream(user_message(text))
            turn_over = False
            try:
                async for event in stream:
                    if event.task is not None:
                        seen |= {m.message_id for m in event.task.history if m.role == a2a.ROLE_AGENT}
                        continue
                    _show(event.status_update.status.message, seen)
                    if event.status_update.status.state in a2a.INTERRUPTED:
                        turn_over = True
                        break
            except HarnessError:
                continue  # its line was printed as it was told; the task has failed
            except a2a.A2AError as error:
                print(f"(not sent: {error})", file=sys.stderr)
                continue
            if turn_over:
                followers.append(asyncio.create_task(_follow(agent, stream, seen)))
        if any(not f.done() for f in followers):
            print("waiting for background branches to merge...", file=sys.stderr)
        for outcome in await asyncio.gather(*followers, return_exceptions=True):
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                print(f"(a follower failed: {type(outcome).__name__}: {outcome})", file=sys.stderr)
    finally:
        # Also on Ctrl-C (the task is cancelled): running branches stop, open tasks fail.
        for f in followers:
            f.cancel()
        await agent.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="graft")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--db", default=None)
    c.add_argument("--trace", action="store_true", help="show branch/join internals")
    d = sub.add_parser("dump")
    d.add_argument("db")
    a = p.parse_args()
    if a.cmd == "chat":
        path = a.db or os.path.join(tempfile.mkdtemp(prefix="graft-"), "chat.db")
        try:
            asyncio.run(chat(path, a.trace))
        except KeyboardInterrupt:  # Ctrl-C: chat() has already closed the harness
            print()
            raise SystemExit(130) from None
    else:
        print(Store(a.db).dump())
