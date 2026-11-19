"""Tests for hierarchical nested skills, explicit-path SKILL.md disk loading,
the >= 2 child fork-ability rule, depth-N current & detached branches, and FTS5 visibility."""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path

import pytest

from graft import a2a
from graft.agent import GraftAgent, GraftConfig
from graft.db import Skill, Store
from graft.fake import _last_block, _row_ids, _text, _use, woken
from graft.providers import FakeProvider
from graft.skills import install, load_skills
from graft.tools import Tool, ToolContext


def _msg(text: str, *, metadata: dict | None = None) -> a2a.SendMessageRequest:
    return a2a.SendMessageRequest(
        message=a2a.Message(
            message_id=str(uuid.uuid4()),
            role=a2a.ROLE_USER,
            parts=[a2a.Part(text=text)],
            metadata=metadata,
        )
    )


def _schema() -> dict:
    return {"type": "object", "properties": {"q": {"type": "string"}}}


def _write_skill(dir_path: Path, *, name: str, description: str, instructions: str,
                 tools: list[str] | None = None, meta_parent: str | None = None) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    if tools:
        lines.append("tools:")
        for t in tools:
            lines.append(f"  - {t}")
    if meta_parent is not None:
        lines.append("metadata:")
        lines.append(f"  parent: {meta_parent}")
    lines.append("---")
    lines.append("")
    lines.append(instructions)
    (dir_path / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ disk loader
def test_load_skills_nested_and_flat_metadata_parent(tmp_path):
    """load_skills(path) supports nested <parent>/skills/<child>/SKILL.md, flat metadata.parent,
    matching redundant metadata.parent, arbitrary extra files/dirs, and depth >= 2 trees."""
    root = tmp_path / "catalog"
    _write_skill(root / "ops", name="ops", description="Operations umbrella.", instructions="Inspect ops.", tools=["ping"])
    # Extra non-skill assets inside a skill dir must be ignored per Agent Skills spec
    (root / "ops" / "scripts").mkdir(parents=True)
    (root / "ops" / "scripts" / "check.sh").write_text("#!/bin/sh\necho ok\n")

    # Nested child (parent inferred from dir)
    _write_skill(root / "ops" / "skills" / "db-audit", name="db-audit",
                 description="Audit database.", instructions="Run db checks.", tools=["sql_check"])
    # Nested child with matching metadata.parent (allowed)
    _write_skill(root / "ops" / "skills" / "cache-audit", name="cache-audit",
                 description="Audit cache.", instructions="Run cache checks.", meta_parent="ops")
    # Flat child referencing parent via metadata.parent
    _write_skill(root / "net-audit", name="net-audit",
                 description="Audit network.", instructions="Run net checks.", meta_parent="ops")
    # Depth-2 grandchild under nested child
    _write_skill(root / "ops" / "skills" / "db-audit" / "skills" / "pg-explain", name="pg-explain",
                 description="Explain Postgres plans.", instructions="Run EXPLAIN ANALYZE.", tools=["pg_explain"])

    skills = load_skills(root)
    by_name = {s.name: s for s in skills}
    assert set(by_name) == {"ops", "db-audit", "cache-audit", "net-audit", "pg-explain"}
    assert by_name["ops"].parent is None
    assert by_name["ops"].tools == ["ping"]
    assert by_name["db-audit"].parent == "ops"
    assert by_name["cache-audit"].parent == "ops"
    assert by_name["net-audit"].parent == "ops"
    assert by_name["pg-explain"].parent == "db-audit"
    # Ordered parents before children
    order = [s.name for s in skills]
    assert order.index("ops") < order.index("db-audit") < order.index("pg-explain")


def test_load_skills_refuses_bad_catalogs_atomically(tmp_path):
    """Refuses missing path, name!=dir, conflicting dir vs metadata.parent, unknown parent,
    cycles, and duplicate names; GraftAgent writes zero skills to SQLite on failure."""
    # Missing dir
    with pytest.raises(ValueError, match="does not exist"):
        load_skills(tmp_path / "does_not_exist")

    # Name mismatch
    d1 = tmp_path / "mismatch"
    _write_skill(d1 / "folder-a", name="folder-b", description="bad", instructions="x")
    with pytest.raises(ValueError, match="does not match directory name"):
        load_skills(d1)

    # Dir parent disagrees with metadata.parent
    d2 = tmp_path / "conflict"
    _write_skill(d2 / "p1", name="p1", description="parent 1", instructions="x")
    _write_skill(d2 / "p2", name="p2", description="parent 2", instructions="x")
    _write_skill(d2 / "p1" / "skills" / "child", name="child", description="c", instructions="x", meta_parent="p2")
    with pytest.raises(ValueError, match="disagrees with metadata.parent"):
        load_skills(d2)

    # Unknown parent
    d3 = tmp_path / "unknown_parent"
    _write_skill(d3 / "orphan", name="orphan", description="o", instructions="x", meta_parent="ghost")
    with pytest.raises(ValueError, match="unknown parent"):
        load_skills(d3)

    # Cycle (a -> b -> a)
    d4 = tmp_path / "cycle"
    _write_skill(d4 / "a", name="a", description="a", instructions="x", meta_parent="b")
    _write_skill(d4 / "b", name="b", description="b", instructions="x", meta_parent="a")
    with pytest.raises(ValueError, match="cycle"):
        load_skills(d4)

    # Duplicate names (flat + nested)
    d5 = tmp_path / "dup"
    _write_skill(d5 / "p", name="p", description="p", instructions="x")
    _write_skill(d5 / "dup-child", name="dup-child", description="c1", instructions="x", meta_parent="p")
    _write_skill(d5 / "p" / "skills" / "dup-child", name="dup-child", description="c2", instructions="x")
    with pytest.raises(ValueError, match="duplicate skill name"):
        load_skills(d5)

    # GraftAgent fails closed before writing anything to SQLite
    store = Store(str(tmp_path / "fail_closed.db"))
    with pytest.raises(ValueError, match="disagrees"):
        GraftAgent(store, FakeProvider(lambda *a: _text("ok")), config=GraftConfig(skills_dir=d2, seeder=None))
    assert store.skill_names() == []


# ------------------------------------------------------------------ catalog hiding & threshold
def test_main_catalog_and_card_hide_children_and_refuse_direct_fork(tmp_path):
    """Child skills stay out of main's system prompt and Agent Card; main is refused if it tries
    to fork directly into a hidden child skill."""
    skills = [
        Skill("ops", "Operations root.", "Coordinate ops.", []),
        Skill("db", "Database checks.", "Inspect DB.", ["db_tool"], parent="ops"),
        Skill("net", "Network checks.", "Inspect net.", ["net_tool"], parent="ops"),
    ]
    tools = [
        Tool("db_tool", _schema(), lambda a: "db ok"),
        Tool("net_tool", _schema(), lambda a: "net ok"),
    ]
    refusals: list[str] = []

    def policy(system, messages, tool_defs):
        last = _last_block(messages)
        if last["type"] == "tool_result" and last.get("is_error"):
            refusals.append(str(last.get("content")))
            return _text("understood, will not bypass ops")
        return _use(tool="fork", current={"skills": ["db"], "messages": _row_ids(messages)})

    async def go():
        store = Store(str(tmp_path / "hide.db"))
        agent = GraftAgent(store, FakeProvider(policy), tools=tools, config=GraftConfig(skills=skills, seeder=None))
        # Agent Card only exposes top-level skills (ops, search), never db/net
        card_names = [s.name for s in agent.card().skills]
        assert "ops" in card_names and "search" in card_names
        assert "db" not in card_names and "net" not in card_names
        async with agent:
            sys_main = agent._host.system("o")
            assert "- ops:" in sys_main
            assert "- db:" not in sys_main and "- net:" not in sys_main
            await agent.send(_msg("check database"))

    asyncio.run(go())
    assert len(refusals) == 1
    assert "nested under 'ops'" in refusals[0]


def test_min_children_to_fork_two_vs_one_child(tmp_path):
    """With default min_children_to_fork=2:
    - parent with >= 2 children gains `fork` and sees its children in its child catalog
    - parent with 1 child does NOT gain `fork`; its lone child is auto-linked directly
    With min_children_to_fork=1:
    - parent with 1 child DOES gain `fork`."""
    skills = [
        Skill("multi", "Multi parent.", "Multi instr.", []),
        Skill("m1", "M1 child.", "M1 instr.", ["t_m1"], parent="multi"),
        Skill("m2", "M2 child.", "M2 instr.", ["t_m2"], parent="multi"),
        Skill("single", "Single parent.", "Single instr.", []),
        Skill("s1", "S1 lone child.", "S1 instr.", ["t_s1"], parent="single"),
        Skill("leaf", "Leaf skill.", "Leaf instr.", ["t_leaf"]),
    ]
    tools = [
        Tool("t_m1", _schema(), lambda a: "m1"),
        Tool("t_m2", _schema(), lambda a: "m2"),
        Tool("t_s1", _schema(), lambda a: "s1"),
        Tool("t_leaf", _schema(), lambda a: "leaf"),
    ]

    # Default threshold = 2
    store2 = Store(str(tmp_path / "t2.db"))
    install(store2, tools={t.name: t.definition() for t in tools}, skills=skills)
    agent2 = GraftAgent(store2, FakeProvider(lambda *a: _text("ok")), tools=tools,
                        config=GraftConfig(skills=skills, seeder=None))
    linked_multi, cf_multi = agent2._host._resolve_branch_skills(["multi"], focus=True)
    assert cf_multi is True and linked_multi == ["multi", "fork", "merge"]
    linked_single, cf_single = agent2._host._resolve_branch_skills(["single"], focus=True)
    assert cf_single is False and linked_single == ["single", "s1", "merge"]
    linked_leaf, cf_leaf = agent2._host._resolve_branch_skills(["leaf"], focus=True)
    assert cf_leaf is False and linked_leaf == ["leaf", "merge"]

    # Threshold = 1
    store1 = Store(str(tmp_path / "t1.db"))
    agent1 = GraftAgent(store1, FakeProvider(lambda *a: _text("ok")), tools=tools,
                        config=GraftConfig(skills=skills, seeder=None, min_children_to_fork=1))
    linked_s1, cf_s1 = agent1._host._resolve_branch_skills(["single"], focus=True)
    assert cf_s1 is True and linked_s1 == ["single", "fork", "merge"]


# ------------------------------------------------------------------ depth-2 current & detached
def test_depth_2_current_parent_forks_current_child_wake_and_quiet(tmp_path):
    """Depth-2 current -> current with wake=True (parent wakes and summarizes) and wake=False
    (clean reply propagates quietly back to main without extra wake calls)."""
    calls: list[tuple[str, str | None, dict]] = []

    def run_sql(args, call: ToolContext):
        calls.append((call.branch, call.branch_name, dict(call.metadata)))
        return "rows=42"

    skills = [
        Skill("ops", "Ops coordinator.", "Coordinate ops tasks.", []),
        Skill("db", "DB inspector.", "Inspect DB with run_sql.", ["run_sql"], parent="ops"),
        Skill("net", "Net inspector.", "Inspect net.", ["ping"], parent="ops"),
    ]
    tools = [
        Tool("run_sql", _schema(), run_sql),
        Tool("ping", _schema(), lambda a: "pong"),
    ]
    wakes_seen: list[str] = []

    def make_policy(child_wake: bool):
        def policy(system, messages, tool_defs):
            names = {t["name"] for t in tool_defs}
            last = _last_block(messages)
            # Root main
            if "You are on main" in system:
                if woken(messages):
                    wakes_seen.append("main")
                    return []
                return _use(tool="fork", current={
                    "id": "ops-b", "skills": ["ops"], "messages": _row_ids(messages)[-1:], "wake": False,
                })
            # Depth-1 parent branch (ops)
            if "fork" in names:
                assert "- db:" in system and "- net:" in system
                assert "- ops:" not in system  # already linked, not in child catalog
                if woken(messages):
                    wakes_seen.append("ops")
                    return _text("Ops summary: DB has 42 rows.")
                return _use(tool="fork", current={
                    "id": "db-b", "skills": ["db"], "messages": _row_ids(messages)[-1:], "wake": child_wake,
                })
            # Depth-2 leaf branch (db)
            if last["type"] == "tool_result" and last.get("tool_use_id") == "sql-1":
                return _text("Found 42 rows in DB.")
            return [{"type": "tool_use", "id": "sql-1", "name": "run_sql", "input": {"q": "count"}}]
        return policy

    # Case 1: child_wake=True -> ops wakes, main stays quiet (ops wake=False)
    async def run_case(db_name: str, child_wake: bool):
        store = Store(str(tmp_path / db_name))
        agent = GraftAgent(store, FakeProvider(make_policy(child_wake)), tools=tools,
                           config=GraftConfig(skills=skills, seeder=None))
        async with agent:
            res = await agent.send(_msg("how many rows?", metadata={"req": db_name}))
            assert res.task.status.state == a2a.COMPLETED
            return res.task

    wakes_seen.clear()
    calls.clear()
    task1 = asyncio.run(run_case("wake_true.db", True))
    assert wakes_seen == ["ops"]
    assert calls == [("c2", "db-b", {"req": "wake_true.db"})]
    assert "Ops summary: DB has 42 rows." in task1.status.message.text

    wakes_seen.clear()
    calls.clear()
    task2 = asyncio.run(run_case("wake_false.db", False))
    assert wakes_seen == []  # neither ops nor main needed to wake!
    assert calls == [("c2", "db-b", {"req": "wake_false.db"})]
    assert "Found 42 rows in DB." in task2.status.message.text


def test_depth_2_current_parent_forks_parallel_detached_children(tmp_path):
    """A current parent branch forks two detached child branches in parallel, waits for BOTH
    without premature INPUT_REQUIRED, wakes ONCE with both merged results, replies to the user,
    and merges back to main."""
    ran_tools: list[tuple[str, str | None]] = []

    async def db_tool(args, call: ToolContext):
        await asyncio.sleep(0.02)
        ran_tools.append(("db", call.branch_name))
        return "db_latency=12ms"

    async def net_tool(args, call: ToolContext):
        await asyncio.sleep(0.02)
        ran_tools.append(("net", call.branch_name))
        return "net_loss=0%"

    skills = [
        Skill("ops", "Ops root.", "Coordinate checks.", []),
        Skill("db", "DB check.", "Check DB.", ["db_tool"], parent="ops"),
        Skill("net", "Net check.", "Check net.", ["net_tool"], parent="ops"),
    ]
    tools = [
        Tool("db_tool", _schema(), db_tool),
        Tool("net_tool", _schema(), net_tool),
    ]
    ops_calls = {"n": 0}

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "You are on main" in system:
            if woken(messages):
                return []
            return _use(tool="fork", current={
                "id": "ops-cur", "skills": ["ops"], "messages": _row_ids(messages)[-1:], "wake": False,
            })
        if "fork" in names:
            ops_calls["n"] += 1
            if woken(messages):
                # Both db-c and net-c should have merged before ops woke!
                return _text("Health OK: db_latency=12ms and net_loss=0%.")
            if last["type"] == "tool_result" and str(last.get("content", "")).startswith("forked ->"):
                # End step quietly to wait for detached children
                return []
            return _use(tool="fork", detached=[
                {"id": "db-c", "skills": ["db"], "messages": _row_ids(messages)[-1:], "reason": "check db"},
                {"id": "net-c", "skills": ["net"], "messages": _row_ids(messages)[-1:], "reason": "check net"},
            ])
        if "db_tool" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "d1":
                return _text("DB latency is 12ms.")
            return [{"type": "tool_use", "id": "d1", "name": "db_tool", "input": {"q": "lat"}}]
        if "net_tool" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "n1":
                return _text("Network packet loss is 0%.")
            return [{"type": "tool_use", "id": "n1", "name": "net_tool", "input": {"q": "loss"}}]
        return _text("unexpected")

    async def go():
        store = Store(str(tmp_path / "par_det.db"))
        agent = GraftAgent(store, FakeProvider(policy), tools=tools, config=GraftConfig(skills=skills, seeder=None))
        async with agent:
            res = await agent.send(_msg("Check DB and network health."))
            # Must stay WORKING until both children finish and ops produces final reply -> COMPLETED!
            assert res.task.status.state == a2a.COMPLETED
            assert "12ms" in res.task.status.message.text and "0%" in res.task.status.message.text

    asyncio.run(go())
    assert set(ran_tools) == {("db", "db-c"), ("net", "net-c")}
    # ops ran: (1) fork detached=[db-c, net-c], (2) empty end step, (3) single wake after BOTH merged
    assert ops_calls["n"] == 3


def test_depth_2_detached_parent_forks_detached_children_and_refuses_current(tmp_path):
    """A detached parent branch cannot fork `current` (refused), CAN fork parallel `detached`
    children, wakes once they merge, synthesizes, and merges back into main."""
    refusals: list[str] = []

    skills = [
        Skill("ops", "Ops root.", "Coordinate ops.", []),
        Skill("db", "DB check.", "Check DB.", ["db_tool"], parent="ops"),
        Skill("net", "Net check.", "Check net.", ["net_tool"], parent="ops"),
    ]
    tools = [
        Tool("db_tool", _schema(), lambda a: "db=ok"),
        Tool("net_tool", _schema(), lambda a: "net=ok"),
    ]
    tried_current = {"done": False}

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "You are on main" in system:
            if woken(messages):
                return _text("Background ops finished: both DB and net are OK.")
            if last["type"] == "tool_result" and str(last.get("content", "")).startswith("forked ->"):
                return _text("Started background ops.")
            return _use(tool="fork", detached=[
                {"id": "ops-bg", "skills": ["ops"], "messages": _row_ids(messages)[-1:], "reason": "run ops"},
            ])
        if "fork" in names:
            if woken(messages):
                return _text("Synthesized: DB ok and net ok.")
            if last["type"] == "tool_result" and last.get("is_error"):
                refusals.append(str(last.get("content")))
                return _use(tool="fork", detached=[
                    {"id": "sub-db", "skills": ["db"], "messages": _row_ids(messages)[-1:], "reason": "db"},
                    {"id": "sub-net", "skills": ["net"], "messages": _row_ids(messages)[-1:], "reason": "net"},
                ])
            if last["type"] == "tool_result" and str(last.get("content", "")).startswith("forked ->"):
                return []
            if not tried_current["done"]:
                tried_current["done"] = True
                return _use(tool="fork", current={
                    "id": "bad-cur", "skills": ["db"], "messages": _row_ids(messages)[-1:],
                })
            return []
        if "db_tool" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "bd1":
                return _text("DB ok.")
            return [{"type": "tool_use", "id": "bd1", "name": "db_tool", "input": {"q": "db"}}]
        if "net_tool" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "bn1":
                return _text("Net ok.")
            return [{"type": "tool_use", "id": "bn1", "name": "net_tool", "input": {"q": "net"}}]
        return _text("unexpected")

    async def go():
        store = Store(str(tmp_path / "det_parent.db"))
        agent = GraftAgent(store, FakeProvider(policy), tools=tools, config=GraftConfig(skills=skills, seeder=None))
        async with agent:
            res = await agent.send(_msg("Run ops in background."))
            task_id = res.task.id
            final_text = ""
            async for ev in agent.subscribe(a2a.SubscribeToTaskRequest(id=task_id)):
                if ev.status_update and ev.status_update.status.message:
                    final_text = ev.status_update.status.message.text
            assert "both DB and net are OK" in final_text

    asyncio.run(go())
    assert len(refusals) == 1
    assert "detached branch can only fork detached branches" in refusals[0]


def test_depth_3_tree_and_fts5_merged_to_root_visibility(tmp_path):
    """3-level tree (ops -> db -> pg) loaded from disk via GraftConfig(skills_dir=...):
    FTS5 search MUST NOT see pg's output while ops is still unmerged, and MUST see it once ops
    merges into main."""
    sdir = tmp_path / "skills_tree"
    _write_skill(sdir / "ops", name="ops", description="Ops root.", instructions="Coordinate ops.")
    _write_skill(sdir / "ops" / "skills" / "db", name="db", description="DB root.", instructions="Coordinate DB.")
    _write_skill(sdir / "ops" / "skills" / "net", name="net", description="Net check.", instructions="Check net.", tools=["ping"])
    _write_skill(sdir / "ops" / "skills" / "db" / "skills" / "pg", name="pg",
                 description="Postgres.", instructions="Inspect PG.", tools=["pg_dump"])
    _write_skill(sdir / "ops" / "skills" / "db" / "skills" / "redis", name="redis",
                 description="Redis.", instructions="Inspect Redis.", tools=["redis_info"])

    mid_search_results: list[str] = []
    store = Store(str(tmp_path / "depth3.db"))

    tools = [
        Tool("ping", _schema(), lambda a: "pong"),
        Tool("pg_dump", _schema(), lambda a: "RAW_PG_LOG: PG_NEEDLE_7711 table=orders"),
        Tool("redis_info", _schema(), lambda a: "redis ok"),
    ]

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "You are on main" in system:
            if woken(messages):
                return []
            return _use(tool="fork", current={
                "id": "d1-ops", "skills": ["ops", "search"], "messages": _row_ids(messages)[-1:], "wake": False,
            })
        if "This branch of yours has the id d1-ops." in system:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "mid-s":
                mid_search_results.append(str(last.get("content")))
                return _text("Ops done after DB inspection.")
            if woken(messages):
                # pg merged into db, db merged into ops, BUT ops has NOT merged into main yet!
                return [{"type": "tool_use", "id": "mid-s", "name": "search", "input": {"query": "PG_NEEDLE_7711"}}]
            return _use(tool="fork", current={"id": "d2-db", "skills": ["db"], "messages": _row_ids(messages)[-1:], "wake": True})
        if "This branch of yours has the id d2-db." in system:
            if woken(messages):
                return _text("DB inspected Postgres.")
            return _use(tool="fork", current={"id": "d3-pg", "skills": ["pg"], "messages": _row_ids(messages)[-1:], "wake": True})
        if "pg_dump" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "pg1":
                return _text("Postgres inspection finished.")
            return [{"type": "tool_use", "id": "pg1", "name": "pg_dump", "input": {"q": "orders"}}]
        return _text("unexpected")

    async def go():
        agent = GraftAgent(store, FakeProvider(policy), tools=tools,
                           config=GraftConfig(skills_dir=sdir, seeder=None))
        async with agent:
            res = await agent.send(_msg("Inspect Postgres orders table."))
            assert res.task.status.state == a2a.COMPLETED

    asyncio.run(go())

    # Before ops merged into main, FTS5 search saw ZERO hits for PG_NEEDLE_7711
    assert mid_search_results == ["No matching history found."]
    # After ops merged into main, FTS5 search sees pg's raw tool output!
    after_hits = store.search("PG_NEEDLE_7711")
    assert len(after_hits) == 1
    assert "PG_NEEDLE_7711" in after_hits[0]["snippet"]
    assert after_hits[0]["branch_name"] == "d3-pg"


def test_depth_2_clean_shutdown_while_detached_children_running(tmp_path):
    """Closing the agent while a depth-1 branch has running depth-2 detached children shuts down
    cleanly without hanging."""
    entered = threading.Event()
    release = threading.Event()

    async def slow_tool(args):
        entered.set()
        await asyncio.to_thread(release.wait, 5.0)
        return "slow done"

    skills = [
        Skill("ops", "Ops root.", "Coordinate ops.", []),
        Skill("db", "DB check.", "Check DB.", ["slow_tool"], parent="ops"),
        Skill("net", "Net check.", "Check net.", ["ping"], parent="ops"),
    ]
    tools = [
        Tool("slow_tool", _schema(), slow_tool),
        Tool("ping", _schema(), lambda a: "pong"),
    ]

    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        if "You are on main" in system:
            return _use(tool="fork", current={"id": "ops-s", "skills": ["ops"], "messages": _row_ids(messages)[-1:]})
        if "fork" in names:
            if last["type"] == "tool_result" and str(last.get("content", "")).startswith("forked ->"):
                return []
            return _use(tool="fork", detached=[
                {"id": "slow-db", "skills": ["db"], "messages": _row_ids(messages)[-1:], "reason": "slow"},
            ])
        if "slow_tool" in names:
            return [{"type": "tool_use", "id": "s1", "name": "slow_tool", "input": {"q": "wait"}}]
        return _text("ok")

    async def go():
        store = Store(str(tmp_path / "shutdown.db"))
        agent = GraftAgent(store, FakeProvider(policy), tools=tools, config=GraftConfig(skills=skills, seeder=None))
        await agent.start()
        send_task = asyncio.create_task(agent.send(_msg("start slow ops")))
        await asyncio.to_thread(entered.wait, 5.0)
        assert entered.is_set()
        release.set()
        await asyncio.wait_for(agent.close(), timeout=5.0)
        try:
            await send_task
        except Exception:
            pass

    asyncio.run(go())
