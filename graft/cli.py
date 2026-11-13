"""`uv run graft chat` (interactive) and `uv run graft dump DB`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile

from .db import Store
from .harness import Harness, in_daemon_thread
from .providers import make_provider
from .skills import install


def _read_line() -> str | None:
    """One line of input (called in a daemon thread, like a model call); None at end of input."""
    try:
        return input()
    except EOFError:
        return None


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
    try:
        while True:
            # The prompt is shown once the machine has answered the last call. Each line is read
            # off the event loop, one read per request, in a daemon thread whose result comes back
            # by the same hop as a model call's; background clones keep running meanwhile.
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
