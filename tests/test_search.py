"""Phase 4: shipped `search` skill (`search(query)`) backed by SQLite FTS5.

Covers:
- main always holds `search` and can grant it per branch via fork(..., skills=[..., "search"])
- only holders of `search` can call `search`; non-holders are refused
- merged-rows-only scope: never unmerged/running branches, never withdrawn rows, never harness rows
- dropped tool_result outputs from merged branches ARE searchable after merge (and NOT before)
- compacted-away main rows remain searchable
- FTS5 punctuation / hyphenated identifiers (ERR-8492, us-east-2, #sev1-platform) never crash
- search results themselves are never indexed (no recursive snippet echoes)
"""

from __future__ import annotations

import asyncio
import json
import threading

from _util import chat_of, conversation, install_fake
from graft.db import Graft, Store
from graft.fake import _last_block, _row_ids, _text, _use, _user_texts
from graft.sim import run_script


def test_main_holds_history_and_only_granted_branches_get_search(tmp_path):
    store = Store(str(tmp_path / "h.db"))
    install_fake(store)
    h = conversation(store, lambda *a: _text("ok"), quote=True).h

    async def go():
        await chat_of(h).start()
        assert {"fork", "stop", "search"} == {t["name"] for t in store.tools_for("o")}
        assert "search(query" in h.system("o")
        rid = store.add_turn("o", "user", "text", "build site")
        store.commit_branch(source="o", grafts=[
            Graft(child="c1", seed=[rid], skills=["website", "merge"], focus=True),
            Graft(child="c2", seed=[rid], skills=["website", "search", "detached"], focus=False),
        ])
        assert "search" not in {t["name"] for t in store.tools_for("c1")}
        assert "search" in {t["name"] for t in store.tools_for("c2")}
        await h.machines.stop()

    asyncio.run(go())


def test_ungranted_branch_calling_search_is_refused(tmp_path):
    called = []

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        if "fork" in names:
            if last["type"] == "tool_result":
                return _text("done")
            return _use(tool="fork", current={"skills": ["website"], "messages": _row_ids(messages)[-1:],
                                              "reason": "update site"})
        if last["type"] == "tool_result" and last.get("tool_use_id") == "s1":
            called.append((last.get("content"), last.get("is_error")))
            return _text("finished without search")
        return [{"type": "tool_use", "id": "s1", "name": "search", "input": {"query": "secret"}}]

    store = Store(str(tmp_path / "refuse.db"))
    install_fake(store)
    h = conversation(store, policy, quote=True).h

    async def go():
        await chat_of(h).start()
        await run_script(chat_of(h).agent, ["Secret token is ALPHA-999.", "Update the site."], echo=False)
        await h.close()

    asyncio.run(go())
    assert called and called[0][1] is True and "not available" in str(called[0][0])


def test_merged_only_scope_and_dropped_tool_outputs_searchable_after_merge(tmp_path):
    """While a detached branch is still running, its text and tool_result rows MUST NOT appear in
    search(query). Once it merges, both its text AND its dropped tool_result (which is off main's
    context) become searchable; withdrawn and harness rows never appear."""
    gate = threading.Event()
    entered = threading.Event()
    mid_run_hits: list[str] = []
    post_merge_hits: list[str] = []

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        texts = _user_texts(messages)
        if "fork" in names:
            if last["type"] == "tool_result":
                if last.get("tool_use_id") == "q1":
                    mid_run_hits.append(str(last.get("content")))
                    return _text("checked mid-run")
                if last.get("tool_use_id") == "q2":
                    post_merge_hits.append(str(last.get("content")))
                    return _text("checked post-merge")
                return _text("started")
            cur = texts[-1] if texts else ""
            if "Start background" in cur:
                return _use(tool="fork", detached=[{
                    "id": "bg-logs", "skills": ["website"], "messages": _row_ids(messages)[-1:],
                    "reason": "read home page",
                }])
            if "Check mid-run" in cur:
                return [{"type": "tool_use", "id": "q1", "name": "search", "input": {"query": "ERR-8492"}}]
            if "Check post-merge" in cur:
                return [{"type": "tool_use", "id": "q2", "name": "search", "input": {"query": "ERR-8492"}}]
            return _text("ok")
        # detached branch: writes a tool_result containing ERR-8492, waits on gate, then replies
        if last["type"] == "tool_result" and last.get("tool_use_id") == "w1":
            entered.set()
            assert gate.wait(5.0)
            return _text("Checked the page; everything looks summarized.")
        return [{"type": "tool_use", "id": "w1", "name": "site_write",
                 "input": {"page": "logs", "content": "buried needle ERR-8492 upstream-auth-v3"}}]

    store = Store(str(tmp_path / "scope.db"))
    install_fake(store)
    h = conversation(store, policy, quote=True).h

    # Override site_write to return the buried needle inside the tool_result
    orig_exec = chat_of(h).world.execute

    def exec_with_needle(name, args):
        if name == "site_write":
            return "RAW LOG DUMP: 2026-11-19 ERROR code=ERR-8492 dep=upstream-auth-v3 latency_p99=1480ms"
        return orig_exec(name, args)

    chat_of(h).world.execute = exec_with_needle

    async def go():
        await chat_of(h).start()
        await chat_of(h).say("Start background work.")
        assert await asyncio.to_thread(entered.wait, 5.0)
        # While bg-logs is still running, ERR-8492 MUST NOT be searchable
        await chat_of(h).say("Check mid-run")
        gate.set()
        await chat_of(h).quiet()
        # After bg-logs merges, ERR-8492 (from its dropped tool_result!) MUST be searchable
        await chat_of(h).say("Check post-merge")
        await h.close()

    asyncio.run(go())
    assert mid_run_hits == ["No matching history found."]
    assert len(post_merge_hits) == 1
    assert "ERR-8492" in post_merge_hits[0] and "bg-logs" in post_merge_hits[0] and "tool:site_write" in post_merge_hits[0]
    # Confirm raw site_write tool_result is still NOT on main's context (context stayed clean!)
    assert not any(r["tool_name"] == "site_write" for r in store.view("o"))
    # Confirm search results themselves were NOT indexed in turns_fts (no recursive echoes)
    fts_rows = [r[0] for r in store.con.execute("SELECT content FROM turns_fts")]
    assert not any("tool:site_write" in c for c in fts_rows)


def test_withdrawn_harness_compacted_and_punctuation_queries(tmp_path):
    store = Store(str(tmp_path / "fts.db"))
    install_fake(store)
    store.create_timeline("o", source=None, skills=["fork", "search"])
    r1 = store.add_turn("o", "user", "text", "Escalation channel is #sev1-platform and region is us-east-2.")
    r2 = store.add_turn("o", "assistant", "text", "Noted #sev1-platform.")
    r_gone = store.add_turn("o", "user", "text", "Withdrawn secret ZEBRA-404.")
    r_harness = store.add_turn("o", "user", "text", "Harness internal marker HARNESS-999.", harness=True)
    r3 = store.add_turn("o", "user", "text", "Latest turn.")
    r4 = store.add_turn("o", "assistant", "text", "Latest reply.")

    # Withdraw r_gone -> immediately deleted from FTS5 and filtered out
    store.mark_withdrawn(["o"], [r_gone])
    assert store.search("ZEBRA-404") == []
    # Harness rows are never indexed or returned
    assert store.search("HARNESS-999") == []
    assert r_harness not in [h["row_id"] for h in store.search("marker")]

    # Compact away r1 & r2 from working_view("o") -> still searchable in FTS5!
    store.mark_compaction("o", [r3, r4])
    assert r1 not in [r["row_id"] for r in store.working_view("o")]
    visible = {r["row_id"] for r in store.working_view("o")}
    hits = store.search("#sev1-platform us-east-2", exclude_rows=visible)
    assert any(h["row_id"] == r1 for h in hits)
    # Visible rows in working_view("o") are excluded when exclude_rows is passed
    assert store.search("Latest turn", exclude_rows=visible) == []
    # Hyphenated / punctuation / invalid FTS syntax never crashes
    assert isinstance(store.search("ERR-8492:foo/bar (baz) \"unclosed"), list)


def test_search_excludes_rows_already_in_callers_working_view(tmp_path):
    """A search from main or a branch must NEVER echo rows already in the caller's working_view
    (especially the current user prompt that mentions the searched terms), so OR/BM25 reaches
    out-of-view compacted rows and merged branches' dropped tool_result outputs."""
    results: list[str] = []

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        texts = _user_texts(messages)
        if "fork" in names:
            cur = texts[-1] if texts else ""
            if "Fetch logs" in cur:
                return _use(tool="fork", current={"skills": ["website"], "messages": _row_ids(messages)[-1:],
                                                  "reason": "run tool"})
            if "Draft email" in cur:
                return _use(tool="fork", current={"skills": ["website", "search"], "messages": _row_ids(messages)[-1:],
                                                  "reason": "search dropped tool output"})
            return _text("ok")
        if "search" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "s1":
                results.append(str(last.get("content")))
                return _text("done")
            # Kitchen-sink query containing words from the visible user prompt ("Draft email pod cpu_pct")
            return [{"type": "tool_use", "id": "s1", "name": "search",
                     "input": {"query": "Draft email failing pod cpu_pct"}}]
        if last["type"] == "tool_result" and last.get("tool_use_id") == "w1":
            return _text("Summary only: ERR-8492.")
        return [{"type": "tool_use", "id": "w1", "name": "site_write", "input": {"page": "p", "content": "x"}}]

    store = Store(str(tmp_path / "out_of_view.db"))
    install_fake(store)
    h = conversation(store, policy, quote=True).h
    orig_exec = chat_of(h).world.execute
    chat_of(h).world.execute = lambda name, args: (
        "ERROR code=ERR-8492 pod=checkout-7d9f4c-x2k9 retries=3" if name == "site_write" else orig_exec(name, args)
    )

    async def go():
        await chat_of(h).start()
        await run_script(chat_of(h).agent, [
            "Fetch logs.",
            "Draft email with failing pod and cpu_pct.",
        ], echo=False)
        await h.close()

    asyncio.run(go())
    assert len(results) == 1
    assert "checkout-7d9f4c-x2k9" in results[0]
    assert "Draft email" not in results[0]  # visible user prompt was excluded!


def test_branch_granted_history_can_search_merged_history(tmp_path):
    found: list[str] = []

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        texts = _user_texts(messages)
        if "fork" in names:
            cur = texts[-1] if texts else ""
            if "Remember" in cur:
                return _text("Remembered.")
            # Fork website + history, seeding ONLY the last message (NOT the earlier secret/fact row!)
            return _use(tool="fork", current={
                "skills": ["website", "search"],
                "messages": _row_ids(messages)[-1:],
                "reason": "Find the escalation channel in history and write it to the status page.",
            })
        if last["type"] == "tool_result" and last.get("tool_use_id") == "s1":
            found.append(str(last.get("content")))
            return _text("Found it via history search.")
        return [{"type": "tool_use", "id": "s1", "name": "search", "input": {"query": "#sev1-platform"}}]

    store = Store(str(tmp_path / "grant.db"))
    install_fake(store)
    h = conversation(store, policy, quote=True).h

    async def go():
        await chat_of(h).start()
        await run_script(chat_of(h).agent, [
            "Remember that our escalation channel is #sev1-platform.",
            "Update the website with our escalation channel.",
        ], echo=False)
        await h.close()

    asyncio.run(go())
    assert found and "#sev1-platform" in found[0]
