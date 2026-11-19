"""`uv run graft acp`, `uv run graft code [dir]`, `uv run graft chat`, and `uv run graft dump DB`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from . import a2a
from .acp import ACPServer
from .agent import ForkContext, GraftAgent, GraftConfig, HarnessError
from .coding import (CODING_SKILLS, CodingWorkspace, coding_persona, format_branches,
                     format_search, format_usage, load_env)
from .db import Store
from .fake import FakeTools
from .harness import in_daemon_thread
from .providers import make_provider
from .sim import user_message
from .skills import MIN_BRANCH_SKILLS_TO_FORK, load_skills


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


def _parse_min_branch_skills(raw: str | None) -> int | None:
    if raw is None:
        return MIN_BRANCH_SKILLS_TO_FORK
    if raw.strip().lower() in ("none", "off", "all"):
        return None
    val = int(raw)
    if val < 1:
        raise ValueError("--min-branch-skills must be >= 1 or 'none'")
    return val


async def code_repl(cwd: str, *, db_path: str | None, provider_name: str | None, model: str | None,
                    skills_dir: str | None, min_branch_skills: int | None, trace: bool) -> None:
    root = Path(cwd).resolve()
    load_env(root)
    if model:
        os.environ["GRAFT_MODEL"] = model
    if db_path is None:
        graft_dir = root / ".graft"
        try:
            graft_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(graft_dir / "code.db")
        except OSError:
            db_path = os.path.join(tempfile.mkdtemp(prefix="graft-code-"), "code.db")

    store = Store(db_path)
    provider = make_provider(provider_name)

    def on_tool_event(ev: dict) -> None:
        if ev["phase"] == "start":
            print(f"  -> {ev['title']}", file=sys.stderr, flush=True)
        elif ev.get("is_error"):
            print(f"  !! {ev['title']} (error)", file=sys.stderr, flush=True)

    def observe_fork(ctx: ForkContext) -> None:
        parts = [f"{e.name or '+'.join(e.skills)} ({'current' if e.focus else 'detached'}: {', '.join(e.skills)})"
                 for e in ctx.entries]
        print(f"  => fork: {', '.join(parts)}", file=sys.stderr, flush=True)
        return None

    workspace = CodingWorkspace(root, on_event=on_tool_event)
    skills = list(CODING_SKILLS) + (load_skills(skills_dir) if skills_dir else [])
    agent = GraftAgent(
        store,
        provider,
        tools=workspace.tools(),
        config=GraftConfig(
            persona=coding_persona(root),
            skills=skills,
            max_steps=25,
            seeder=None,
            before_fork=observe_fork,
            min_branch_skills_to_fork=min_branch_skills,
            trace=(lambda s: print(f"  [trace] {s}", file=sys.stderr, flush=True)) if trace else None,
        ),
    )
    await agent.start()
    print(f"graft code ({provider.name}:{provider.model}) cwd={root} db={db_path}")
    print("Commands: /branches, /usage, /search <query>, /dump, /quit")
    seen: set[str] = set()
    followers: list[asyncio.Task] = []
    try:
        while True:
            _prompt()
            line = await in_daemon_thread(_read_line)
            if line is None:
                break
            text = line.strip()
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/branches":
                print(format_branches(store))
                continue
            if text == "/usage":
                print(format_usage(provider))
                continue
            if text.startswith("/search"):
                print(format_search(store, text[7:]))
                continue
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
                continue
            except a2a.A2AError as error:
                print(f"(not sent: {error})", file=sys.stderr)
                continue
            if turn_over:
                followers.append(asyncio.create_task(_follow(agent, stream, seen)))
        if any(not f.done() for f in followers):
            print("waiting for background branches to merge...", file=sys.stderr)
        await asyncio.gather(*followers, return_exceptions=True)
    finally:
        for f in followers:
            f.cancel()
        await agent.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="graft")
    sub = p.add_subparsers(dest="cmd", required=True)

    a_p = sub.add_parser("acp", help="Run Graft coding agent as a stdio ACP server (for Toad, Zed, etc.)")
    a_p.add_argument("--provider", default=None, help="openai or anthropic")
    a_p.add_argument("--model", default=None, help="model name (overrides GRAFT_MODEL)")
    a_p.add_argument("--db", default=None, help="explicit SQLite DB path")
    a_p.add_argument("--skills-dir", default=None, help="optional directory of extra SKILL.md files")
    a_p.add_argument("--min-branch-skills", default=None, help="min child skills to gain fork (default 2, or 'none')")

    co = sub.add_parser("code", help="Interactive terminal coding agent")
    co.add_argument("dir", nargs="?", default=".", help="workspace directory (default: .)")
    co.add_argument("--provider", default=None, help="openai or anthropic")
    co.add_argument("--model", default=None, help="model name (overrides GRAFT_MODEL)")
    co.add_argument("--db", default=None, help="explicit SQLite DB path")
    co.add_argument("--skills-dir", default=None, help="optional directory of extra SKILL.md files")
    co.add_argument("--min-branch-skills", default=None, help="min child skills to gain fork (default 2, or 'none')")
    co.add_argument("--trace", action="store_true", help="show branch/join internals on stderr")

    c = sub.add_parser("chat", help="Interactive chat over simulated demo tools")
    c.add_argument("--db", default=None)
    c.add_argument("--trace", action="store_true", help="show branch/join internals")
    d = sub.add_parser("dump", help="Dump a SQLite session store")
    d.add_argument("db")

    a = p.parse_args()
    if a.cmd == "acp":
        min_bs = _parse_min_branch_skills(a.min_branch_skills)
        server = ACPServer(
            provider_name=a.provider,
            model=a.model,
            db_path=a.db,
            skills_dir=a.skills_dir,
            min_branch_skills_to_fork=min_bs,
        )
        try:
            asyncio.run(server.serve())
        except KeyboardInterrupt:
            raise SystemExit(130) from None
    elif a.cmd == "code":
        min_bs = _parse_min_branch_skills(a.min_branch_skills)
        try:
            asyncio.run(code_repl(
                a.dir,
                db_path=a.db,
                provider_name=a.provider,
                model=a.model,
                skills_dir=a.skills_dir,
                min_branch_skills=min_bs,
                trace=a.trace,
            ))
        except KeyboardInterrupt:
            print()
            raise SystemExit(130) from None
    elif a.cmd == "chat":
        path = a.db or os.path.join(tempfile.mkdtemp(prefix="graft-"), "chat.db")
        try:
            asyncio.run(chat(path, a.trace))
        except KeyboardInterrupt:  # Ctrl-C: chat() has already closed the harness
            print()
            raise SystemExit(130) from None
    else:
        print(Store(a.db).dump())
