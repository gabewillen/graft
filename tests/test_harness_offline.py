"""End-to-end harness run with the scripted FakeProvider (no API key needed)."""

import asyncio

import pytest

from _util import clone_in_turn, clones, holders, status, conversation, chat_of, install_fake
from graft.db import Graft, Store
from graft.fake import pivot_policy
from graft.harness import Host, check_skill_link
from graft.providers import FakeProvider
from graft.sim import run_script


def clone_use(skills, focus=True, id="t"):
    """A fork call (graft's clone call) with one entry."""
    entry = {"skills": skills, "messages": []}
    return {"id": id, "name": "fork", "input": {"current": entry} if focus else {"detached": [entry]}}


@pytest.fixture
def harness(tmp_path):
    store = Store(str(tmp_path / "h.db"))
    install_fake(store)
    return conversation(store, pivot_policy, quote=True).h


def test_pivot_flow_offline(harness):
    """Each message reaches the original, which forks a current website branch for it; each
    branch's turn ends in its merge, and the original runs once after it (a current branch's reply
    already reached the user, so it says nothing). On the pivot it forks a slides branch pointed at
    the user's messages: the same rows, by pointer, never copied."""
    h = harness

    async def go():
        await chat_of(h).start()
        await run_script(chat_of(h).agent, [
            "I need a website.", "It is for dentists in Ohio.", "Formal tone please.",
            "Our services: cleaning, implants, whitening.", "Instead let's build a slide show."], echo=False)
        assert holders(h.store) == ["o"]
        return h

    asyncio.run(go())
    s = h.store
    branches = clones(s)
    assert branches == ["c1", "c2", "c3", "c4", "c5"]
    assert [s.timeline(c)["join_reason"] for c in branches] == ["done"] * 5
    assert all(s.timeline(c)["status"] == "joined" for c in branches)
    assert [r[0] for r in s.con.execute("SELECT status FROM transactions WHERE kind='join'")] == ["committed"] * 5
    ohio = s.con.execute("SELECT row_id FROM turns WHERE content='It is for dentists in Ohio.'").fetchone()[0]
    assert s.turn(ohio)["owner_timeline"] == "o"  # every message reaches the original
    assert ohio in s.pointed("c5")  # the slides branch points at it: same row, never copied
    assert s.con.execute("SELECT count(*) FROM turns WHERE content='It is for dentists in Ohio.'").fetchone()[0] == 1
    # the default join is deterministic: no model call ran for it
    assert set(t for t, u in h.provider.by_tag.items() if u["calls"]) <= {
        "original", "original:clone_tx", *(f"clone:{c}" for c in branches)}
    tx = s.con.execute("SELECT committed_sql FROM transactions WHERE kind='join' AND target='c5'").fetchone()
    assert tx[0].startswith("DEFAULT JOIN c5 reason=done")
    assert h.provider.by_tag["original:clone_tx"]["calls"] == 5  # one fork call per message
    assert "ohio" in chat_of(h).world.deck_text().lower()
    c5_tools = {c["tools"][i]["name"] for c in h.provider.calls if "slides_create" in str(c["tools"])
                for i in range(len(c["tools"]))}
    assert "site_write" not in c5_tools and "fork" not in c5_tools


def test_one_shot_clone_refused_with_violations_then_retry(harness):
    h = harness

    async def go():
        await chat_of(h).start()
        s = h.store
        s.add_turn("o", "user", "text", "hello")
        s.add_turn("o", "assistant", "text", "hi")
        use = {"id": "t1", "name": "fork", "input": {"current": {"skills": ["fork"], "purpose": "p"}}}
        child, text = await clone_in_turn(h=h, use=use)
        assert child is None
        assert "cannot be held by a branch" in text and "unknown field 'purpose'" in text
        assert text.endswith("Fix the branches and call fork once more.")
        (child,), text = await clone_in_turn(h=h, use=dict(use, input={"current": {"skills": ["website"], "messages": []}}))
        assert child == "c1" and status(store=s, tl="c1") == "active" and status(store=s, tl="o") == "cloned"
        assert [r["content"] for r in s.view("c1")] == ["hello"]
        await h.machines.stop()

    asyncio.run(go())


def _clone_with_rows(h, seed_text="Make a deck."):
    s = h.store
    rid = s.add_turn("o", "user", "text", seed_text)
    return rid


def test_default_join_is_deterministic_text_rows_only(harness):
    h = harness

    async def go():
        await chat_of(h).start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Make a deck.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["slides"]))
        a1 = s.add_turn(child, "assistant", "text", "Which audience?")
        u1 = s.add_turn(child, "user", "text", "Retirees.")
        s.add_turn(child, "assistant", "tool_use", {"id": "x1", "input": {"title": "T"}}, "slides_create")
        s.add_turn(child, "user", "tool_result", {"tool_use_id": "x1", "content": "ok"}, "slides_create")
        a2 = s.add_turn(child, "assistant", "text", "Built it.")
        calls_before = len(h.provider.calls)
        await h.do_join(child, "done")
        # no model call for the join: the one call after it is the original's run after the merge
        assert [("fork" in str(c["tools"])) for c in h.provider.calls[calls_before:]] == [True]
        mine = [r for r in s.view("o") if r["owner_timeline"] == child]
        assert [r["row_id"] for r in mine][:3] == [a1, u1, a2]  # text rows, seed row skipped (already held)
        # then the harness's merge call (its turn ended without one) and the original's merge result
        assert [(r["kind"], r["tool_name"], r["harness"]) for r in mine[3:]] == [("tool_use", "merge", 1)]
        result = s.view("o")[[r["row_id"] for r in s.view("o")].index(mine[3]["row_id"]) + 1]
        assert result["owner_timeline"] == "o" and result["tool_name"] == "merge" and result["harness"] == 1
        assert status(store=s, tl="o") == "active" and status(store=s, tl=child) == "joined"
        assert not [r for r in s.view("o") if r["kind"] in ("tool_use", "tool_result")
                    and r["owner_timeline"] == child and r["tool_name"] != "merge"]
        await h.machines.stop()

    asyncio.run(go())


def test_default_join_failed_puts_the_merge_call_and_its_result_before_the_trigger(harness):
    h = harness

    async def go():
        await chat_of(h).start()
        s = h.store
        s.add_turn("o", "user", "text", "Make a site.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"]))
        sure = s.add_turn(child, "assistant", "text", "Sure.")
        trig = s.add_turn(child, "user", "text", "Actually, slides.")
        assert h.default_join_rows("o", child, trig) == [sure]  # the trigger goes last, on its own
        s.set_join(child, "failed", trigger=trig)
        assert h.default_join(src="o", child=child, reason="failed", trigger=trig, running=[])
        tail = s.view("o")[-4:]
        assert tail[0]["row_id"] == sure and tail[3]["row_id"] == trig  # trigger last
        assert (tail[1]["tool_name"], tail[1]["owner_timeline"]) == ("merge", child)  # its merge call
        result = __import__("json").loads(tail[2]["content"])
        assert result["content"].startswith(f"{child} merged after it failed")  # the original's merge result
        await h.machines.stop()

    asyncio.run(go())


def test_working_view_after_compaction_bounds_the_default_join(harness):
    h = harness

    async def go():
        await chat_of(h).start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Plan a long project.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"]))
        old_a = s.add_turn(child, "assistant", "text", "Early detail.")
        old_u = s.add_turn(child, "user", "text", "Early answer.")
        old_b = s.add_turn(child, "assistant", "text", "Long early reply.")
        fresh = s.add_turn(child, "user", "text", "Late request.")
        comp = s.add_compressed_turn(child, "assistant", "Early detail, condensed.", old_a, old_a)
        s.mark_compaction(child, [seed, comp, old_u, fresh])  # old_b dropped from the working view
        done = s.add_turn(child, "assistant", "text", "Done.")
        assert [r["row_id"] for r in s.working_view(child)] == [seed, comp, old_u, fresh, done]
        assert len(s.view(child)) == 7  # the full timeline is unchanged
        rows = h.default_join_rows("o", child, None)
        assert rows == [comp, old_u, fresh, done] and old_b not in rows and old_a not in rows
        await h.machines.stop()

    asyncio.run(go())


def test_clone_skill_has_only_load_skill_clone_and_cancel(harness):
    s = harness.store
    s.create_timeline(tl="o", source=None, skills=["fork"])
    assert [t["name"] for t in s.tools_for("o")] == ["load_skill", "fork", "stop"]


def test_clone_is_never_linked_to_a_clone(harness):
    s = harness.store
    s.create_timeline(tl="o", source=None, skills=["fork"])
    assert check_skill_link(s, "o", ["slides"]) == []
    assert any("cannot be held by a branch" in v for v in check_skill_link(store=s, source="o", skills=["fork", "slides"]))


def test_progressive_disclosure_system_prompt(harness):
    s = harness.store
    s.create_timeline(tl="o", source=None, skills=["fork"])
    rid = s.add_turn("o", "user", "text", "x")
    s.commit_clone(source="o", grafts=[Graft(child="c1", seed=[rid], skills=["website", "merge"], focus=True)])
    sys_o, sys_c = harness.system("o"), harness.system("c1")
    for name in ("fork", "merge", "website", "slides"):
        instr = s.con.execute("SELECT instructions FROM skills WHERE name=?", (name,)).fetchone()[0]
        assert instr not in sys_o and instr not in sys_c
    assert "website: Build and edit a small website" in sys_c  # name + description only
    assert "slides" not in sys_c  # clones do not see the catalog
    assert "slides: Create slide decks" in sys_o  # the original carries the catalog
    assert "load_skill" in {t["name"] for t in s.tools_for("o")}
    assert "load_skill" in {t["name"] for t in s.tools_for("c1")}


def test_load_skill_scoped_to_linked_skills(harness):
    h = harness
    s = h.store
    s.create_timeline(tl="o", source=None, skills=["fork"])
    rid = s.add_turn("o", "user", "text", "x")
    s.commit_clone(source="o", grafts=[Graft(child="c1", seed=[rid], skills=["website", "merge"], focus=True)])
    content, err = h.load("c1", "website")
    assert not err and content.startswith("Website skill")
    content, err = h.load("c1", "slides")
    assert err and "not linked" in content
    content, err = h.load("o", "website")  # the original holds only clone
    assert err and "not linked" in content
    assert not h.load(tl="o", name="fork")[1]


def test_load_skill_result_is_an_ordinary_row(tmp_path):
    store = Store(str(tmp_path / "p.db"))
    install_fake(store)
    h = conversation(store, pivot_policy, quote=True).h

    async def go():
        await chat_of(h).start()
        await run_script(chat_of(h).agent, ["I need a website.", "For dentists in Ohio."], echo=False)
        await h.machines.stop()

    asyncio.run(go())
    rows = [r for r in store.view("c1") if r["tool_name"] == "load_skill"]
    assert [r["kind"] for r in rows] == ["tool_use", "tool_result"]
    assert "Website skill" in rows[1]["content"] and rows[1]["owner_timeline"] == "c1"
    assert not [r for r in store.view("o") if r["tool_name"] == "load_skill"]


def _no_compact_policy(system, messages, tools):
    """Refuses every compaction (answers in text, never calls compact); otherwise replies."""
    return [{"type": "text", "text": "ok"}]


def test_oversized_join_falls_back_to_last_reply_and_survives_refused_compaction(tmp_path):
    from graft import compaction
    from graft.db import render
    from graft.harness import JOIN_WINDOW_FRACTION
    store = Store(str(tmp_path / "w.db"))
    install_fake(store)
    h = conversation(store, _no_compact_policy, compact_tokens=50, window_tokens=400, quote=True).h
    limit = int(400 * JOIN_WINDOW_FRACTION)

    async def go():
        await chat_of(h).start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Write a long report.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"]))
        big = []
        for i in range(3):
            big.append(s.add_turn(child, "assistant", "text", f"Section {i}: " + "x" * 1000))
            big.append(s.add_turn(child, "user", "text", f"More on {i}."))
        last = s.add_turn(child, "assistant", "text", "Report finished.")
        trig = s.add_turn(child, "user", "text", "Now make slides instead.")
        await h.do_join(child, "out_of_scope")
        view = [r["row_id"] for r in s.working_view("o")]
        # tier 1: the clone's last reply, its merge call and the original's merge result, then the trigger
        assert view[-5] == last and view[-2] == trig
        assert [s.turn(r)["tool_name"] for r in view[-4:-2]] == ["merge", "merge"]
        assert "Only your last reply there is above" in s.turn(view[-3])["content"]
        assert s.turn(view[-1])["content"] == "ok"  # the source then answered
        assert not set(big) & set(view)  # oversized rows stay only in the clone's frozen timeline
        assert all(r in [x["row_id"] for x in s.view(child)] for r in big)
        assert h.join_fallbacks == [{"clone": child, "tier": 1, "before": h.join_fallbacks[0]["before"],
                                     "after": h.join_fallbacks[0]["after"]}]
        assert h.join_fallbacks[0]["before"] > limit >= h.join_fallbacks[0]["after"]
        assert h.compaction_failures == ["o"]  # compaction refused twice ...
        source_call = h.provider.calls[-1]  # ... and the source's next call still fits the window
        assert compaction.estimate_tokens(source_call["messages"]) <= limit
        assert compaction.estimate_tokens(render(s.working_view("o"))) <= limit
        await h.machines.stop()

    asyncio.run(go())


def test_join_without_window_is_not_size_checked(harness):
    h = harness

    async def go():
        await chat_of(h).start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Write.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"]))
        a = s.add_turn(child, "assistant", "text", "y" * 50000)
        merge, implicit = h.merge_call(child=child, closing=False)
        rows, result = h.fit_join(src="o", child=child, reason="done", rows=[a], merge=merge, trigger=None, running=[])
        assert implicit and rows == [a] and "did not fit" not in result and h.join_fallbacks == []
        await h.machines.stop()

    asyncio.run(go())


def test_window_defaults_per_model(tmp_path, monkeypatch):
    from graft.providers import context_window, window_tokens_for
    assert context_window("claude-sonnet-5-5") == 1_000_000
    assert context_window("anthropic.claude-opus-5-5") == 1_000_000
    assert context_window("claude-haiku-4-5") == 200_000 and context_window("claude-haiku-5-5") == 1_000_000
    assert context_window("gpt-5-mini") is None and context_window(None) is None
    monkeypatch.delenv("GRAFT_WINDOW_TOKENS", raising=False)
    fp = FakeProvider(_no_compact_policy)
    assert window_tokens_for(fp) is None  # unknown model: off
    fp.model = "claude-sonnet-5-5"
    store = Store(str(tmp_path / "m.db"))
    assert conversation(store, provider=fp).h.window_tokens == 1_000_000  # on by default for a known model
    monkeypatch.setenv("GRAFT_WINDOW_TOKENS", "5000")
    assert conversation(store, provider=fp).h.window_tokens == 5000
    monkeypatch.setenv("GRAFT_WINDOW_TOKENS", "0")
    assert conversation(store, provider=fp).h.window_tokens is None
    assert conversation(store, provider=fp, window_tokens=123).h.window_tokens == 123  # explicit wins


def _always_fails_policy(system, messages, tools):
    """The original clones for every message unless told a clone already failed; the clone
    loops on a tool until its step budget runs out (join failed)."""
    names = {t["name"] for t in tools}
    last = messages[-1]["content"]
    if "fork" in names:
        if any(b["type"] == "tool_result" and "already failed" in str(b["content"]) for b in last):
            return [{"type": "text", "text": "Sorry, I could not build that."}]
        user = [b["text"] for m in messages if m["role"] == "user" for b in m["content"] if b["type"] == "text"][-1]
        return [{"type": "tool_use", "id": f"c{len(messages)}", "name": "fork",
                 "input": {"current": {"skills": ["website"]}}}]
    return [{"type": "tool_use", "id": f"r{len(messages)}", "name": "site_read", "input": {"page": "home"}}]

