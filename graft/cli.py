"""`uv run graft chat` (interactive) and `uv run graft dump DB`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import threading

from .db import Store
from .harness import Harness
from .providers import make_provider
from .skills import install


def _read_lines(loop: asyncio.AbstractEventLoop, lines: asyncio.Queue, ready: threading.Event) -> None:
    """Runs in a daemon thread: prompt when asked (`ready`), hand each line to the event loop,
    None at end of input. A daemon thread blocked in input() does not keep the process alive."""
    while True:
        ready.wait()
        ready.clear()
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            line = None
        try:
            loop.call_soon_threadsafe(lines.put_nowait, line)
        except RuntimeError:  # the loop is closed
            return
        if line is None:
            return


async def chat(db_path: str, trace: bool) -> None:
    store = Store(db_path)
    install(store)
    # Replies are printed as they are said: a background clone's result arrives between turns.
    h = Harness(store, make_provider(), trace=print if trace else None,
                emit=lambda reply: print(f"assistant> {reply}"))
    await h.start()
    print(f"graft chat ({h.provider.name}:{h.provider.model}); db={db_path}. "
          "Commands: /dump, /quit")
    lines: asyncio.Queue = asyncio.Queue()
    ready = threading.Event()
    # Input is read off the event loop, so background clones keep running while we wait.
    threading.Thread(target=_read_lines, args=(asyncio.get_running_loop(), lines, ready),
                     daemon=True, name="graft-chat-input").start()
    try:
        while True:
            ready.set()
            line = await lines.get()
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
        if h.machines.background():
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
