"""`uv run graft chat` (interactive) and `uv run graft dump DB`."""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile

from .db import Store
from .harness import Harness
from .providers import make_provider
from .skills import install


async def chat(db_path: str, trace: bool) -> None:
    store = Store(db_path)
    install(store)
    h = Harness(store, make_provider(), trace=print if trace else None)
    await h.start()
    print(f"graft chat ({h.provider.name}:{h.provider.model}); db={db_path}. "
          "Commands: /dump, /quit")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/dump":
            print(store.dump())
            continue
        for reply in await h.user_says(text):
            print(f"assistant> {reply}")
    await h.machines.stop()


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
        asyncio.run(chat(path, a.trace))
    else:
        print(Store(a.db).dump())
