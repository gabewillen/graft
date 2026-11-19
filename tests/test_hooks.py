"""Tests for host seams: ToolContext (branch + A2A metadata on Tool.fn) and before_fork hook."""

from __future__ import annotations

import asyncio
import uuid

from graft import a2a
from graft.agent import ForkContext, GraftAgent, GraftConfig
from graft.db import Skill, Store
from graft.fake import _last_block, _row_ids, _text, _use, _user_texts, woken
from graft.providers import FakeProvider
from graft.skills import BUILTIN, CATALOG
from graft.tools import Tool, ToolContext


_BUILTINS = [s for s in CATALOG if s.name in BUILTIN]


def _msg(text: str, *, task_id: str | None = None, metadata: dict | None = None) -> a2a.SendMessageRequest:
    return a2a.SendMessageRequest(
        message=a2a.Message(
            message_id=str(uuid.uuid4()),
            role=a2a.ROLE_USER,
            parts=[a2a.Part(text=text)],
            task_id=task_id,
            metadata=metadata,
        )
    )


def _skill(name: str, tool_names: list[str]) -> Skill:
    return Skill(
        name=name,
        description=f"{name} skill",
        instructions=f"Use {', '.join(tool_names)} for {name}.",
        tools=tool_names,
    )


def _schema() -> dict:
    return {"type": "object", "properties": {"val": {"type": "string"}}}


def test_tool_context_signatures_and_a2a_metadata(tmp_path):
    """Tool.fn supports 1-arg (args), 2-arg (args, call), async 2-arg, kw-only (*, call), and
    **kwargs. Each invocation sees the calling branch id, branch name, tool_use_id, and A2A
    message.metadata (updating per message without leaking stale metadata)."""
    seen: list[tuple[str, dict, ToolContext | None]] = []

    def t_legacy(args):
        seen.append(("legacy", dict(args), None))
        return "legacy ok"

    def t_pos(args, call):
        seen.append(("pos", dict(args), call))
        return f"pos:{call.branch}:{call.branch_name}"

    async def t_async(args, call):
        await asyncio.sleep(0)
        seen.append(("async", dict(args), call))
        return f"async:{call.branch}:{call.branch_name}"

    def t_kwonly(args, *, call):
        seen.append(("kwonly", dict(args), call))
        return f"kwonly:{call.branch}"

    def t_varkw(args, **kw):
        seen.append(("varkw", dict(args), kw.get("call")))
        return "varkw ok"

    tools = [
        Tool(name="t_legacy", input_schema=_schema(), fn=t_legacy),
        Tool(name="t_pos", input_schema=_schema(), fn=t_pos),
        Tool(name="t_async", input_schema=_schema(), fn=t_async),
        Tool(name="t_kwonly", input_schema=_schema(), fn=t_kwonly),
        Tool(name="t_varkw", input_schema=_schema(), fn=t_varkw),
    ]
    skills = _BUILTINS + [
        _skill("alpha", ["t_legacy", "t_pos", "t_async", "t_kwonly", "t_varkw"]),
    ]

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "fork" in names:
            if woken(messages) or last["type"] == "tool_result":
                return []
            texts = _user_texts(messages)
            bid = "b-first" if len(texts) == 1 else "b-second"
            return _use(tool="fork", current={
                "id": bid, "skills": ["alpha"], "messages": _row_ids(messages)[-1:], "wake": False,
            })
        if last["type"] == "tool_result" and last.get("tool_use_id", "").startswith("tc-"):
            return _text("all tools done")
        return [
            {"type": "tool_use", "id": "tc-1", "name": "t_legacy", "input": {"val": "a"}},
            {"type": "tool_use", "id": "tc-2", "name": "t_pos", "input": {"val": "b"}},
            {"type": "tool_use", "id": "tc-3", "name": "t_async", "input": {"val": "c"}},
            {"type": "tool_use", "id": "tc-4", "name": "t_kwonly", "input": {"val": "d"}},
            {"type": "tool_use", "id": "tc-5", "name": "t_varkw", "input": {"val": "e"}},
        ]

    async def go():
        store = Store(str(tmp_path / "ctx.db"))
        agent = GraftAgent(
            store,
            FakeProvider(policy),
            tools=tools,
            config=GraftConfig(skills=skills, seeder=None),
        )
        async with agent:
            r1 = await agent.send(_msg("first turn", metadata={"tenant": "acme", "turn": 1}))
            assert r1.task.status.state == a2a.COMPLETED
            # Second message has NO metadata -> must not inherit stale metadata from message 1
            r2 = await agent.send(_msg("second turn"))
            assert r2.task.status.state == a2a.COMPLETED

    asyncio.run(go())

    first_five = seen[:5]
    second_five = seen[5:10]
    assert [name for name, _, _ in first_five] == ["legacy", "pos", "async", "kwonly", "varkw"]
    assert first_five[0][2] is None  # 1-arg legacy tool did not receive ToolContext
    for idx, expected_uid in [(1, "tc-2"), (2, "tc-3"), (3, "tc-4"), (4, "tc-5")]:
        ctx = first_five[idx][2]
        assert isinstance(ctx, ToolContext)
        assert ctx.branch == "c1"
        assert ctx.branch_name == "b-first"
        assert ctx.tool_use_id == expected_uid
        assert ctx.metadata == {"tenant": "acme", "turn": 1}

    for idx in (1, 2, 3, 4):
        ctx = second_five[idx][2]
        assert isinstance(ctx, ToolContext)
        assert ctx.branch == "c2"
        assert ctx.branch_name == "b-second"
        assert ctx.metadata == {}


def test_tool_context_concurrent_detached_branches(tmp_path):
    """Concurrent detached branches each receive their own distinct branch id & name and the
    originating A2A message metadata."""
    records: list[tuple[str, str | None, dict]] = []

    async def work(args, call: ToolContext):
        await asyncio.sleep(0.01)
        records.append((call.branch, call.branch_name, dict(call.metadata)))
        return f"done {args.get('val')}"

    tools = [Tool(name="work", input_schema=_schema(), fn=work)]
    skills = _BUILTINS + [_skill("worker", ["work"])]

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "fork" in names:
            if woken(messages):
                return _text("both finished")
            if last["type"] == "tool_result":
                return _text("started both")
            return _use(tool="fork", detached=[
                {"id": "left", "skills": ["worker"], "messages": _row_ids(messages)[-1:], "reason": "left"},
                {"id": "right", "skills": ["worker"], "messages": _row_ids(messages)[-1:], "reason": "right"},
            ])
        if last["type"] == "tool_result" and last.get("tool_use_id", "").startswith("w-"):
            return _text("worker complete")
        return [{"type": "tool_use", "id": "w-1", "name": "work", "input": {"val": "go"}}]

    async def go():
        store = Store(str(tmp_path / "det.db"))
        agent = GraftAgent(
            store,
            FakeProvider(policy),
            tools=tools,
            config=GraftConfig(skills=skills, seeder=None),
        )
        async with agent:
            res = await agent.send(_msg("run parallel", metadata={"trace_id": "tr-42"}))
            task_id = res.task.id
            # Wait until detached branches finish and task completes
            async for _ev in agent.subscribe(a2a.SubscribeToTaskRequest(id=task_id)):
                pass

    asyncio.run(go())
    assert len(records) == 2
    by_name = {name: (branch, meta) for branch, name, meta in records}
    assert set(by_name) == {"left", "right"}
    assert by_name["left"][0] != by_name["right"][0]
    assert by_name["left"][1] == {"trace_id": "tr-42"}
    assert by_name["right"][1] == {"trace_id": "tr-42"}


def test_before_fork_per_message_cap_and_reset(tmp_path):
    """before_fork enforces a per-human-message fork cap, rolls back atomically on refusal,
    and resets forks_this_message to 0 on the next human message."""
    seen_contexts: list[dict] = []
    refusals: list[str] = []

    def cap_one_fork(ctx: ForkContext) -> str | None:
        seen_contexts.append({
            "source": ctx.source,
            "forks_this_message": ctx.forks_this_message,
            "live_branches": ctx.live_branches,
            "total_branches": ctx.total_branches,
            "metadata": dict(ctx.metadata),
            "entry_skills": [e.skills for e in ctx.entries],
        })
        if ctx.forks_this_message >= 1:
            return "fork limit reached: at most 1 fork per user message"
        return None

    tools = [Tool(name="do_a", input_schema=_schema(), fn=lambda a: "a ok")]
    skills = _BUILTINS + [_skill("step_a", ["do_a"])]
    stage = {"n": 0}

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "fork" in names:
            if last["type"] == "tool_result" and last.get("is_error"):
                refusals.append(str(last.get("content")))
                return _text("stopped after fork refusal")
            if woken(messages):
                # After first branch merges, attempt a SECOND fork on the SAME human message
                if stage["n"] == 1:
                    stage["n"] = 2
                    return _use(tool="fork", current={
                        "id": "second-same-msg", "skills": ["step_a"],
                        "messages": _row_ids(messages)[-1:], "wake": False,
                    })
                return _text("done")
            stage["n"] += 1
            return _use(tool="fork", current={
                "id": f"b-{stage['n']}", "skills": ["step_a"],
                "messages": _row_ids(messages)[-1:], "wake": True,
            })
        if last["type"] == "tool_result" and last.get("tool_use_id") == "a1":
            return _text("step_a finished")
        return [{"type": "tool_use", "id": "a1", "name": "do_a", "input": {"val": "x"}}]

    store = Store(str(tmp_path / "cap.db"))

    async def go():
        agent = GraftAgent(
            store,
            FakeProvider(policy),
            tools=tools,
            config=GraftConfig(skills=skills, seeder=None, before_fork=cap_one_fork),
        )
        async with agent:
            r1 = await agent.send(_msg("msg 1", metadata={"m": 1}))
            assert r1.task.status.state == a2a.COMPLETED
            r2 = await agent.send(_msg("msg 2", metadata={"m": 2}))
            assert r2.task.status.state == a2a.COMPLETED

    asyncio.run(go())

    # 3 fork attempts:
    # 1. msg 1 first fork -> forks_this_message == 0 -> allowed (c1 created)
    # 2. msg 1 wake second fork -> forks_this_message == 1 -> refused!
    # 3. msg 2 first fork -> forks_this_message == 0 -> allowed (c2 created)
    assert [c["forks_this_message"] for c in seen_contexts] == [0, 1, 0]
    assert [c["total_branches"] for c in seen_contexts] == [0, 1, 1]
    assert [c["metadata"] for c in seen_contexts] == [{"m": 1}, {"m": 1}, {"m": 2}]
    assert len(refusals) == 1
    assert "fork limit reached: at most 1 fork per user message" in refusals[0]
    # Only 2 branches actually committed (refused fork rolled back atomically)
    branch_rows = store.con.execute("SELECT timeline_id, name FROM timelines WHERE source_id IS NOT NULL").fetchall()
    assert len(branch_rows) == 2
    rolled_back = store.con.execute(
        "SELECT COUNT(*) FROM transactions WHERE kind='branch' AND status='rolled_back'"
    ).fetchone()[0]
    assert rolled_back == 1


def test_before_fork_async_seed_privacy_policy_and_retry(tmp_path):
    """Async before_fork inspects resolved seeds + store contents to block private rows from
    entering a public branch; the refused attempt does NOT count against forks_this_message, so
    the corrected retry succeeds."""
    refusals: list[str] = []
    seen_counts: list[int] = []

    async def no_secret_in_public(ctx: ForkContext) -> list[str]:
        await asyncio.sleep(0)
        seen_counts.append(ctx.forks_this_message)
        problems = []
        for entry, seed in zip(ctx.entries, ctx.seeds):
            if "public_post" in entry.skills:
                for rid in seed:
                    row = ctx.store.turn(rid)
                    if row and "SECRET_KEY" in row["content"]:
                        problems.append(f"{entry.label}: row {rid} contains private data and cannot be seeded to public_post")
        return problems

    tools = [Tool(name="post", input_schema=_schema(), fn=lambda a: "posted")]
    skills = _BUILTINS + [_skill("public_post", ["post"])]
    attempt = {"n": 0}

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "fork" in names:
            if woken(messages):
                return []
            if last["type"] == "tool_result" and last.get("is_error"):
                refusals.append(str(last.get("content")))
                # Retry cleanly without the secret row!
                return _use(tool="fork", current={
                    "id": "pub-clean", "skills": ["public_post"],
                    "messages": _row_ids(messages)[-1:], "wake": False,
                })
            texts = _user_texts(messages)
            if len(texts) == 1:
                return _text("Noted your secret.")
            attempt["n"] += 1
            # First attempt leaks row 1 (SECRET_KEY) into public_post
            return _use(tool="fork", current={
                "id": "pub-leaky", "skills": ["public_post"],
                "messages": _row_ids(messages), "wake": False,
            })
        if last["type"] == "tool_result" and last.get("tool_use_id") == "p1":
            return _text("published safely")
        return [{"type": "tool_use", "id": "p1", "name": "post", "input": {"val": "hi"}}]

    store = Store(str(tmp_path / "priv.db"))

    async def go():
        agent = GraftAgent(
            store,
            FakeProvider(policy),
            tools=tools,
            config=GraftConfig(skills=skills, seeder=None, before_fork=no_secret_in_public),
        )
        async with agent:
            await agent.send(_msg("My SECRET_KEY is 12345."))
            r2 = await agent.send(_msg("Publish a hello world post."))
            assert r2.task.status.state == a2a.COMPLETED

    asyncio.run(go())

    assert len(refusals) == 1
    assert "cannot be seeded to public_post" in refusals[0]
    # Both the refused attempt and the retry saw forks_this_message == 0
    assert seen_counts == [0, 0]
    names = [r[0] for r in store.con.execute("SELECT name FROM timelines WHERE source_id IS NOT NULL").fetchall()]
    assert names == ["pub-clean"]


def test_before_fork_skipped_when_shape_invalid(tmp_path):
    """before_fork is only invoked after shape/skill/seed validation succeeds."""
    called = []

    def hook(ctx: ForkContext):
        called.append(ctx)
        return None

    tools = [Tool(name="do_a", input_schema=_schema(), fn=lambda a: "ok")]
    skills = _BUILTINS + [_skill("step_a", ["do_a"])]
    tried_bad = {"done": False}

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "fork" in names:
            if last["type"] == "tool_result" and last.get("is_error"):
                return _text("handled bad fork")
            if not tried_bad["done"]:
                tried_bad["done"] = True
                return _use(tool="fork", current={"skills": ["nonexistent_skill"], "messages": _row_ids(messages)})
            return _text("ok")
        return _text("branch")

    async def go():
        store = Store(str(tmp_path / "skip.db"))
        agent = GraftAgent(
            store,
            FakeProvider(policy),
            tools=tools,
            config=GraftConfig(skills=skills, seeder=None, before_fork=hook),
        )
        async with agent:
            await agent.send(_msg("try bad skill"))

    asyncio.run(go())
    assert called == []
