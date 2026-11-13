"""`uv run graft chat` (interactive) and `uv run graft dump DB`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import threading
from typing import Callable

from .db import Store
from .harness import Harness
from .providers import make_provider
from .skills import install


def _read_lines(loop: asyncio.AbstractEventLoop, deliver: Callable[[str | None], None]) -> None:
    """Runs in a daemon thread: read stdin line by line and hand each line to the event loop (the
    one thread hop, `call_soon_threadsafe`); None at end of input. A daemon thread blocked in
    input() does not keep the process alive."""
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            line = None
        try:
            loop.call_soon_threadsafe(deliver, line)
        except RuntimeError:  # the loop is closed
            return
        if line is None:
            return


FIRST = 0  # lines are taken in the order they were read


def _prompt() -> None:
    print("\nyou> ", end="", flush=True)


async def chat(db_path: str, trace: bool) -> None:
    store = Store(db_path)
    install(store)
    # Replies are printed as they are said: a background clone's result arrives between turns.
    h = Harness(store, make_provider(), trace=print if trace else None,
                emit=lambda reply: print(f"assistant> {reply}"))
    await h.start()
    print(f"graft chat ({h.provider.name}:{h.provider.model}); db={db_path}. "
          "Commands: /dump, /quit")
    loop = asyncio.get_running_loop()
    arrived: list[str | None] = []  # lines read and handed over, not yet taken (on the loop)
    waiting: list[asyncio.Future] = []  # the chat waiting for the next line: answered when one arrives

    def deliver(line: str | None) -> None:  # on the loop: the reader thread's hand-over
        arrived.append(line)
        for reply in waiting:
            if not reply.done():
                reply.set_result(None)
        waiting.clear()

    async def next_line() -> str | None:
        while not arrived:
            reply = loop.create_future()
            waiting.append(reply)
            await reply
        return arrived.pop(FIRST)

    # Input is read off the event loop, so background clones keep running while we wait; the
    # prompt is shown when the machine has answered the user's call.
    threading.Thread(target=_read_lines, args=(loop, deliver), daemon=True, name="graft-chat-input").start()
    try:
        while True:
            _prompt()
            line = await next_line()
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
            await h.user_says(text)
        # Status and errors are diagnostics: stderr. stdout carries the conversation.
        # The store's copy of each graft's state says which are still running.
        if store.con.execute("SELECT 1 FROM timelines WHERE status = 'background' LIMIT 1").fetchone():
            print("waiting for background clones to join...", file=sys.stderr)
        try:
            await h.settle()
        except Exception as e:  # already shown when it happened; do not end on a traceback
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        # Also on Ctrl-C (the task is cancelled): running clones are cancelled and joined failed.
        await h.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="graft")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--db", default=None)
    c.add_argument("--trace", action="store_true", help="show clone/join internals")
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
