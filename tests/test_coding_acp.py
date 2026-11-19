"""Tests for `graft.coding` (workspace tools & skills) and `graft.acp` (stdio ACP JSON-RPC server)."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import threading
import time
from pathlib import Path

from graft.acp import ACPServer
from graft.coding import CODING_SKILLS, CodingWorkspace, format_branches, format_search, format_usage
from graft.fake import _last_block, _row_ids, _text, _use, woken
from graft.providers import FakeProvider
from graft.tools import ToolContext


# ------------------------------------------------------------------ CodingWorkspace tools
def test_coding_workspace_file_search_git_and_confinement(tmp_path):
    """Read, Grep, Glob, GitRead, Edit, Write, and Bash work inside workspace and refuse escapes."""
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "app.py"
    target.write_text("def greet(name):\n    return f'hello {name}'\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    events: list[dict] = []
    ws = CodingWorkspace(tmp_path, on_event=events.append)
    tools = {t.name: t for t in ws.tools()}
    call = ToolContext(branch="c1", branch_name="coder", tool_use_id="tu-1")

    async def go():
        # Read
        r = await tools["Read"].fn({"file_path": "src/app.py", "offset": 1, "limit": 10}, call)
        assert not r.is_error and "def greet" in r.text()
        # Path escape refused
        esc = await tools["Read"].fn({"file_path": "../../etc/passwd"}, call)
        assert esc.is_error and "escapes workspace" in esc.text()
        # Write into .git refused
        git_w = await tools["Write"].fn({"file_path": ".git/hooks/x", "content": "bad"}, call)
        assert git_w.is_error and ".git" in git_w.text()
        # Grep & Glob
        g = await tools["Grep"].fn({"pattern": "def greet"}, call)
        assert not g.is_error and "src/app.py:1:" in g.text()
        gl = await tools["Glob"].fn({"pattern": "**/*.py"}, call)
        assert not gl.is_error and "src/app.py" in gl.text()
        # Edit emits diff event
        ed = await tools["Edit"].fn({
            "file_path": "src/app.py",
            "old_string": "hello",
            "new_string": "hi",
        }, call)
        assert not ed.is_error and "hi" in target.read_text()
        # Ambiguous edit refused
        (tmp_path / "dup.txt").write_text("foo\nfoo\n", encoding="utf-8")
        amb = await tools["Edit"].fn({"file_path": "dup.txt", "old_string": "foo", "new_string": "bar"}, call)
        assert amb.is_error and "occurs 2 times" in amb.text()
        # Write creates nested dirs + emits diff with oldText=None
        wr = await tools["Write"].fn({"file_path": "nested/new.txt", "content": "fresh\n"}, call)
        assert not wr.is_error and (tmp_path / "nested" / "new.txt").read_text() == "fresh\n"
        # Bash runs async
        b = await tools["Bash"].fn({"command": "echo hi_from_bash"}, call)
        assert not b.is_error and "hi_from_bash" in b.text() and "(exit 0)" in b.text()

    asyncio.run(go())

    edit_done = [e for e in events if e["phase"] == "done" and e["tool"] == "Edit" and not e["is_error"]]
    assert len(edit_done) == 1
    assert edit_done[0]["diff"]["oldText"].startswith("def greet")
    assert "hi {name}" in edit_done[0]["diff"]["newText"]
    write_done = [e for e in events if e["phase"] == "done" and e["tool"] == "Write" and not e["is_error"]]
    assert len(write_done) == 1 and write_done[0]["diff"]["oldText"] is None


def test_coding_workspace_git_read_and_bash_cancellation(tmp_path):
    """GitRead allows read-only subcommands and blocks writes/unsafe flags; cancelling Bash kills subprocess."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")

    ws = CodingWorkspace(tmp_path)
    tools = {t.name: t for t in ws.tools()}
    call = ToolContext(branch="c1", branch_name="sh", tool_use_id="tu-2")

    async def go():
        st = await tools["GitRead"].fn({"subcommand": "status", "args": "--short"}, call)
        assert not st.is_error and "README.md" in st.text()
        bad_sub = await tools["GitRead"].fn({"subcommand": "commit", "args": "-m nope"}, call)
        assert bad_sub.is_error and "unsupported git subcommand" in bad_sub.text()
        bad_flag = await tools["GitRead"].fn({"subcommand": "status", "args": "-c core.pager=id"}, call)
        assert bad_flag.is_error and "not allowed" in bad_flag.text()

        # Cancel a sleeping Bash command -> must cancel promptly without waiting 30s
        t0 = time.monotonic()
        task = asyncio.create_task(tools["Bash"].fn({"command": "sleep 30"}, call))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert time.monotonic() - t0 < 3.0

    asyncio.run(go())


# ------------------------------------------------------------------ ACPServer end-to-end
class _FakeIO:
    """In-memory line-oriented output collector for ACPServer."""

    def __init__(self):
        self.buf = io.StringIO()
        self.messages: list[dict] = []

    def write(self, s: str) -> int:
        self.buf.write(s)
        for line in s.splitlines():
            if line.strip():
                self.messages.append(json.loads(line))
        return len(s)

    def flush(self) -> None:
        pass


def test_acp_full_lifecycle_tool_cards_diffs_and_slash_commands(tmp_path):
    """initialize -> session/new -> session/prompt (fork + Edit diff + Read) -> slash commands
    (/branches, /usage, /search) -> clean close."""
    target = tmp_path / "hello.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "You are on main" in system:
            if woken(messages):
                return []
            return _use(tool="fork", current={
                "id": "fix-val", "skills": ["code"], "messages": _row_ids(messages)[-1:], "wake": False,
            })
        if "Edit" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "ed-1":
                return _text("Updated VALUE to 2 in hello.py.")
            if last["type"] == "tool_result" and last.get("tool_use_id") == "rd-1":
                return [{"type": "tool_use", "id": "ed-1", "name": "Edit",
                         "input": {"file_path": "hello.py", "old_string": "VALUE = 1", "new_string": "VALUE = 2"}}]
            return [{"type": "tool_use", "id": "rd-1", "name": "Read", "input": {"file_path": "hello.py"}}]
        return _text("ok")

    out = _FakeIO()
    server = ACPServer(
        stdout=out,
        provider_factory=lambda: FakeProvider(policy),
        db_path=str(tmp_path / "acp.db"),
    )

    async def go():
        await server.handle_line(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": 1, "clientInfo": {"name": "toad", "version": "0.1"}},
        }))
        await server.handle_line(json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "session/new",
            "params": {"cwd": str(tmp_path), "mcpServers": []},
        }))
        sess_reply = next(m for m in out.messages if m.get("id") == 2)
        sid = sess_reply["result"]["sessionId"]

        # Send prompt with embedded @file resource
        await server.handle_line(json.dumps({
            "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [
                    {"type": "text", "text": "Change VALUE to 2."},
                    {"type": "resource", "resource": {"uri": f"file://{target}", "text": "VALUE = 1\n"}},
                ],
            },
        }))
        # Wait for spawned prompt task to complete
        while not any(m.get("id") == 3 for m in out.messages):
            await asyncio.sleep(0.01)

        # Test slash commands (/branches, /usage, /search)
        for req_id, cmd in ((4, "/branches"), (5, "/usage"), (6, "/search VALUE")):
            await server.handle_line(json.dumps({
                "jsonrpc": "2.0", "id": req_id, "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": cmd}]},
            }))
            while not any(m.get("id") == req_id for m in out.messages):
                await asyncio.sleep(0.01)

        await server.close()

    asyncio.run(go())

    # Verify initialize response
    init_res = next(m["result"] for m in out.messages if m.get("id") == 1)
    assert init_res["protocolVersion"] == 1
    assert init_res["agentInfo"]["name"] == "graft"

    # Verify available_commands_update uses exact ACP key `availableCommands`
    updates = [m["params"]["update"] for m in out.messages if m.get("method") == "session/update"]
    cmd_upds = [u for u in updates if u["sessionUpdate"] == "available_commands_update"]
    assert cmd_upds and {c["name"] for c in cmd_upds[0]["availableCommands"]} >= {"branches", "usage", "search"}

    # Verify fork + Read + Edit tool_call and native diff tool_call_update streamed
    tool_starts = [u for u in updates if u["sessionUpdate"] == "tool_call"]
    assert any(u["name"] == "fork" for u in tool_starts)
    assert any(u["name"] == "Read" and "[fix-val]" in u["title"] for u in tool_starts)
    assert any(u["name"] == "Edit" and "[fix-val]" in u["title"] for u in tool_starts)

    tool_progs = [u for u in updates if u["sessionUpdate"] == "tool_call_update"]
    diff_blocks = [c for u in tool_progs for c in u.get("content", []) if c.get("type") == "diff"]
    assert len(diff_blocks) == 1
    assert diff_blocks[0]["oldText"] == "VALUE = 1\n"
    assert diff_blocks[0]["newText"] == "VALUE = 2\n"

    # Verify file was actually edited and prompt returned stopReason=end_turn
    assert target.read_text() == "VALUE = 2\n"
    p3 = next(m["result"] for m in out.messages if m.get("id") == 3)
    assert p3["stopReason"] == "end_turn"

    # Verify slash command outputs streamed as agent_message_chunk
    chunks = [u["content"]["text"] for u in updates if u["sessionUpdate"] == "agent_message_chunk"]
    joined = "\n".join(chunks)
    assert "Updated VALUE to 2" in joined
    assert "| `fix-val` |" in joined  # /branches table
    assert "Found" in joined and "VALUE" in joined  # /search FTS5 hit on dropped Read output!


def test_acp_session_cancel_stops_running_bash(tmp_path):
    """session/cancel notification cancels an in-flight Bash command and returns stopReason=cancelled."""
    started = threading.Event()

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        if "You are on main" in system:
            return _use(tool="fork", current={
                "id": "runner", "skills": ["shell"], "messages": _row_ids(messages)[-1:], "wake": False,
            })
        if "Bash" in names:
            started.set()
            return [{"type": "tool_use", "id": "sh-1", "name": "Bash", "input": {"command": "sleep 30"}}]
        return _text("ok")

    out = _FakeIO()
    server = ACPServer(
        stdout=out,
        provider_factory=lambda: FakeProvider(policy),
        db_path=str(tmp_path / "cancel.db"),
    )

    async def go():
        await server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}}))
        await server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": str(tmp_path)}}))
        sid = next(m for m in out.messages if m.get("id") == 2)["result"]["sessionId"]

        t0 = time.monotonic()
        await server.handle_line(json.dumps({
            "jsonrpc": "2.0", "id": 3, "method": "session/prompt",
            "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "sleep a while"}]},
        }))
        # Wait until Bash tool actually starts
        while not any(m.get("method") == "session/update" and m["params"]["update"].get("name") == "Bash"
                      for m in out.messages):
            await asyncio.sleep(0.01)

        await server.handle_line(json.dumps({
            "jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": sid},
        }))
        while not any(m.get("id") == 3 for m in out.messages):
            await asyncio.sleep(0.01)
        elapsed = time.monotonic() - t0
        await server.close()
        return elapsed

    elapsed = asyncio.run(go())
    assert elapsed < 5.0
    p3 = next(m["result"] for m in out.messages if m.get("id") == 3)
    assert p3["stopReason"] == "cancelled"
