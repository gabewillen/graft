"""Agent Client Protocol (ACP) stdio server for Graft (`uv run graft acp`).

Connects Graft's coding agent (`graft.coding`) to any ACP client — notably Toad
(`batrachianai/toad`) and Zed — over newline-delimited JSON-RPC 2.0 on stdin/stdout:

    uvx --from batrachian-toad toad acp "uv run graft acp" --project-dir . --title "Graft"

Design:
- Zero external runtime dependencies: speaks JSON-RPC 2.0 conforming to ACP v1
  (`agent-client-protocol==1.0.0`).
- Prompts run as background asyncio tasks so `session/cancel` (Esc in Toad) is processed
  immediately and mapped to A2A `CancelTask` (stopping running branches and killing active
  `Bash` process groups).
- Every branch fork and tool call (tagged with its branch name via `ToolContext`) streams live
  as ACP `tool_call` / `tool_call_update` cards, with native `diff` blocks for `Edit`/`Write`.
- Slash commands (`/branches`, `/usage`, `/search <query>`, `/dump`) run locally without an LLM call.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

from . import a2a
from .agent import ForkContext, GraftAgent, GraftConfig, HarnessError
from .coding import (CODING_SKILLS, CodingWorkspace, coding_persona, format_branches,
                     format_search, format_usage, load_env)
from .db import Store
from .providers import Provider, make_provider
from .sim import user_message
from .skills import MIN_BRANCH_SKILLS_TO_FORK, load_skills

ACP_PROTOCOL_VERSION = 1

AVAILABLE_COMMANDS = [
    {"name": "branches", "description": "Show main and all forked branches in this session"},
    {"name": "usage", "description": "Show token, prompt cache, and per-role usage stats"},
    {"name": "search", "description": "Search committed FTS5 history across merged branches",
     "input": {"hint": "search query"}},
    {"name": "dump", "description": "Dump the raw SQLite timelines and turns"},
]


def _extract_prompt_text(prompt: Any) -> str:
    """Flatten ACP prompt content blocks (text, resource, resource_link) into a user message."""
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return ""
    parts: list[str] = []
    for block in prompt:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = str(block.get("text", ""))
            if txt:
                parts.append(txt)
        elif btype == "resource":
            res = block.get("resource") or {}
            uri = str(res.get("uri") or "")
            txt = str(res.get("text") or "")
            if txt:
                header = f"[Resource: {uri}]\n" if uri else ""
                parts.append(f"{header}{txt}")
            elif uri:
                parts.append(f"[Resource: {uri}]")
        elif btype == "resource_link":
            uri = str(block.get("uri") or block.get("name") or "")
            if uri:
                parts.append(f"[File: {uri}]")
    return "\n\n".join(parts).strip()


@dataclass
class ACPSession:
    """One active ACP session backed by a `GraftAgent` and `CodingWorkspace`."""
    session_id: str
    cwd: Path
    store: Store
    provider: Provider
    workspace: CodingWorkspace
    agent: GraftAgent
    active_task_id: str | None = None
    active_prompt: asyncio.Task | None = None
    cancelled: bool = False
    fork_seq: int = 0
    seen_messages: set[str] = field(default_factory=set)


class ACPServer:
    """Async JSON-RPC 2.0 ACP server over newline-delimited streams."""

    def __init__(
        self,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        provider_factory: Callable[[], Provider] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        db_path: str | None = None,
        skills_dir: str | Path | None = None,
        min_branch_skills_to_fork: int | None = MIN_BRANCH_SKILLS_TO_FORK,
        max_steps: int = 25,
    ):
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._provider_factory = provider_factory
        self._provider_name = provider_name
        self._model = model
        self._db_path = db_path
        self._skills_dir = skills_dir
        self._min_branch_skills_to_fork = min_branch_skills_to_fork
        self._max_steps = max_steps
        self._write_lock = asyncio.Lock()
        self._sessions: dict[str, ACPSession] = {}
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ JSON-RPC I/O
    async def _send(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
        async with self._write_lock:
            self._stdout.write(line)
            self._stdout.flush()

    async def _reply(self, req_id: Any, result: Any) -> None:
        await self._send({"jsonrpc": "2.0", "id": req_id, "result": result})

    async def _error(self, req_id: Any, code: int, message: str, data: Any = None) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        await self._send({"jsonrpc": "2.0", "id": req_id, "error": err})

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _update(self, session_id: str, update: dict[str, Any]) -> None:
        await self._notify("session/update", {"sessionId": session_id, "update": update})

    async def _send_text(self, session_id: str, text: str) -> None:
        if not text:
            return
        await self._update(session_id, {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        })

    # ------------------------------------------------------------ session setup
    def _make_provider(self, cwd: Path) -> Provider:
        if self._provider_factory is not None:
            return self._provider_factory()
        load_env(cwd)
        if self._model:
            os.environ["GRAFT_MODEL"] = self._model
        return make_provider(self._provider_name)

    def _session_db(self, cwd: Path, session_id: str) -> str:
        if self._db_path:
            return self._db_path
        graft_dir = cwd / ".graft"
        try:
            graft_dir.mkdir(parents=True, exist_ok=True)
            return str(graft_dir / f"{session_id}.db")
        except OSError:
            return os.path.join(tempfile.mkdtemp(prefix="graft-acp-"), f"{session_id}.db")

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_ver = params.get("protocolVersion", ACP_PROTOCOL_VERSION)
        ver = ACP_PROTOCOL_VERSION if not isinstance(raw_ver, int) else min(raw_ver, ACP_PROTOCOL_VERSION)
        return {
            "protocolVersion": ver,
            "agentInfo": {"name": "graft", "version": "0.1.0"},
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": True,
                },
            },
            "authMethods": [],
        }

    async def _handle_session_new(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_cwd = params.get("cwd") or os.getcwd()
        cwd = Path(str(raw_cwd)).resolve()
        if not cwd.exists() or not cwd.is_dir():
            raise ValueError(f"cwd does not exist or is not a directory: {cwd}")
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        db_path = self._session_db(cwd, session_id)
        store = Store(db_path)
        provider = self._make_provider(cwd)

        async def on_tool_event(ev: dict[str, Any]) -> None:
            call_id = ev.get("tool_use_id") or f"tc-{uuid.uuid4().hex[:8]}"
            if ev["phase"] == "start":
                payload: dict[str, Any] = {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": ev["title"],
                    "name": ev["tool"],
                    "kind": ev["kind"],
                    "status": "in_progress",
                    "rawInput": ev.get("args"),
                }
                if ev.get("locations"):
                    payload["locations"] = ev["locations"]
                await self._update(session_id, payload)
                return
            diff = ev.get("diff")
            content: list[dict[str, Any]]
            if diff is not None:
                content = [{
                    "type": "diff",
                    "path": diff["path"],
                    "oldText": diff["oldText"],
                    "newText": diff["newText"],
                }]
            else:
                content = [{
                    "type": "content",
                    "content": {"type": "text", "text": str(ev.get("output", ""))},
                }]
            await self._update(session_id, {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "failed" if ev.get("is_error") else "completed",
                "content": content,
                "rawOutput": ev.get("output"),
            })

        session_holder: list[ACPSession] = []

        async def observe_fork(ctx: ForkContext) -> None:
            if not session_holder:
                return
            sess = session_holder[0]
            sess.fork_seq += 1
            fid = f"fork-{sess.fork_seq}"
            parts = []
            for e in ctx.entries:
                mode = "current" if e.focus else "detached"
                label = e.name or "+".join(e.skills)
                parts.append(f"{label} ({mode}: {', '.join(e.skills)})")
            title = f"Fork -> {', '.join(parts)}"
            details = "\n".join(
                f"- **{e.name or '+'.join(e.skills)}** (`{'current' if e.focus else 'detached'}`, "
                f"skills: `{', '.join(e.skills)}`, seed rows: `{list(seed)}`)"
                + (f": {e.reason}" if e.reason else "")
                for e, seed in zip(ctx.entries, ctx.seeds)
            )
            await self._update(session_id, {
                "sessionUpdate": "tool_call",
                "toolCallId": fid,
                "title": title,
                "name": "fork",
                "kind": "switch_mode",
                "status": "in_progress",
            })
            await self._update(session_id, {
                "sessionUpdate": "tool_call_update",
                "toolCallId": fid,
                "status": "completed",
                "content": [{"type": "content", "content": {"type": "text", "text": details}}],
            })
            return None

        workspace = CodingWorkspace(cwd, on_event=on_tool_event)
        skills = list(CODING_SKILLS)
        if self._skills_dir is not None:
            skills = skills + load_skills(self._skills_dir)

        agent = GraftAgent(
            store,
            provider,
            tools=workspace.tools(),
            config=GraftConfig(
                persona=coding_persona(cwd),
                skills=skills,
                max_steps=self._max_steps,
                seeder=None,
                before_fork=observe_fork,
                min_branch_skills_to_fork=self._min_branch_skills_to_fork,
            ),
        )
        await agent.start()
        sess = ACPSession(
            session_id=session_id,
            cwd=cwd,
            store=store,
            provider=provider,
            workspace=workspace,
            agent=agent,
        )
        session_holder.append(sess)
        self._sessions[session_id] = sess

        await self._update(session_id, {
            "sessionUpdate": "available_commands_update",
            "availableCommands": AVAILABLE_COMMANDS,
        })
        return {"sessionId": session_id}

    # ------------------------------------------------------------ slash commands & prompt
    async def _handle_slash(self, sess: ACPSession, text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return False
        cmd, _, rest = stripped[1:].partition(" ")
        cmd = cmd.strip().lower()
        if cmd == "branches":
            await self._send_text(sess.session_id, format_branches(sess.store))
            return True
        if cmd == "usage":
            await self._send_text(sess.session_id, format_usage(sess.provider))
            return True
        if cmd == "search":
            await self._send_text(sess.session_id, format_search(sess.store, rest))
            return True
        if cmd == "dump":
            await self._send_text(sess.session_id, f"```text\n{sess.store.dump()}\n```")
            return True
        return False

    async def _handle_session_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        sess = self._sessions.get(session_id)
        if sess is None:
            raise ValueError(f"unknown sessionId: {session_id!r}")
        text = _extract_prompt_text(params.get("prompt"))
        if not text:
            return {"stopReason": "end_turn"}
        if await self._handle_slash(sess, text):
            return {"stopReason": "end_turn"}

        sess.cancelled = False
        sess.active_task_id = None
        sent_chunks = 0
        try:
            async for event in sess.agent.stream(user_message(text)):
                if event.task is not None:
                    sess.active_task_id = event.task.id
                    sess.seen_messages |= {
                        m.message_id for m in event.task.history if m.role == a2a.ROLE_AGENT
                    }
                    continue
                upd = event.status_update
                if upd is None:
                    continue
                sess.active_task_id = upd.task_id
                msg = upd.status.message
                if msg is not None and msg.role == a2a.ROLE_AGENT and msg.message_id not in sess.seen_messages:
                    sess.seen_messages.add(msg.message_id)
                    body = msg.text
                    if body:
                        prefix = "\n\n" if sent_chunks > 0 else ""
                        await self._send_text(session_id, prefix + body)
                        sent_chunks += 1
                if upd.status.state == a2a.CANCELED:
                    sess.cancelled = True
        except HarnessError as err:
            await self._send_text(session_id, f"\n\n{err}")
        except asyncio.CancelledError:
            sess.cancelled = True
            raise
        finally:
            sess.active_task_id = None
        return {"stopReason": "cancelled" if sess.cancelled else "end_turn"}

    async def _handle_session_cancel(self, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        sess = self._sessions.get(session_id)
        if sess is None:
            return
        sess.cancelled = True
        task_id = sess.active_task_id
        if task_id is not None:
            try:
                await sess.agent.cancel_task(a2a.CancelTaskRequest(id=task_id))
            except a2a.A2AError:
                pass

    # ------------------------------------------------------------ dispatch loop
    async def _dispatch_request(self, req_id: Any, method: str, params: dict[str, Any]) -> None:
        try:
            if method == "initialize":
                res = await self._handle_initialize(params)
            elif method == "authenticate":
                res = {}
            elif method == "session/new":
                res = await self._handle_session_new(params)
            elif method == "session/prompt":
                res = await self._handle_session_prompt(params)
            else:
                await self._error(req_id, -32601, f"method not found: {method}")
                return
            await self._reply(req_id, res)
        except ValueError as err:
            await self._error(req_id, -32602, str(err))
        except Exception as err:  # noqa: BLE001
            await self._error(req_id, -32603, f"{type(err).__name__}: {err}")

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def handle_line(self, line: str) -> None:
        """Process one newline-delimited JSON-RPC 2.0 message."""
        raw = line.strip()
        if not raw:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as err:
            await self._error(None, -32700, f"parse error: {err}")
            return
        if not isinstance(msg, dict):
            await self._error(None, -32600, "invalid request: expected JSON object")
            return
        method = msg.get("method")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        has_id = "id" in msg and msg["id"] is not None
        if not isinstance(method, str):
            if has_id:
                await self._error(msg["id"], -32600, "invalid request: missing method")
            return
        if not has_id:
            if method == "session/cancel":
                self._spawn(self._handle_session_cancel(params))
            return
        req_id = msg["id"]
        if method == "session/prompt":
            self._spawn(self._dispatch_request(req_id, method, params))
        else:
            await self._dispatch_request(req_id, method, params)

    async def serve(self) -> None:
        """Read JSON-RPC lines from stdin until EOF, then shut down all sessions cleanly."""
        try:
            while True:
                line = await asyncio.to_thread(self._stdin.readline)
                if not line:
                    break
                await self.handle_line(line)
        finally:
            await self.close()

    async def close(self) -> None:
        """Cancel in-flight prompt tasks and close all active GraftAgent sessions."""
        for sess in list(self._sessions.values()):
            if sess.active_task_id is not None:
                try:
                    await sess.agent.cancel_task(a2a.CancelTaskRequest(id=sess.active_task_id))
                except Exception:  # noqa: BLE001
                    pass
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        for sess in list(self._sessions.values()):
            try:
                await sess.agent.close()
            except Exception:  # noqa: BLE001
                pass
        self._sessions.clear()
