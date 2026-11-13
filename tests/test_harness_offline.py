"""End-to-end harness run with the scripted FakeProvider (no API key needed)."""

import asyncio

import pytest

from _util import clone_in_turn, clones, holders, status
from graft.db import Graft, Store
from graft.fake import pivot_policy
from graft.harness import Harness, check_skill_link
from graft.providers import FakeProvider
from graft.sim import run_script
from graft.skills import install


def clone_use(skills, focus=True, id="t", **seed):
    """A clone tool call with one entry."""
    return {"id": id, "name": "clone",
            "input": {"clone": [{"purpose": "p", "skills": skills, "focus": focus, **seed}]}}


@pytest.fixture
def harness(tmp_path):
    store = Store(str(tmp_path / "h.db"))
    install(store)
    return Harness(store, FakeProvider(pivot_policy))


def test_pivot_flow_offline(harness):
    h = harness

    async def go():
        await h.start()
        await run_script(h, [
            "I need a website.", "It is for dentists in Ohio.", "Formal tone please.",
            "Our services: cleaning, implants, whitening.", "Instead let's build a slide show."], echo=False)
        assert len(holders(h.store)) == 1
        return h

    asyncio.run(go())
    s = h.store
    assert s.timeline("c1")["join_reason"] == "out_of_scope"
    assert s.timeline("c1")["status"] == "joined"
    tx = s.con.execute("SELECT status FROM transactions WHERE kind='join' AND target='c1'").fetchone()
    assert tx["status"] == "committed"
    ohio = s.con.execute("SELECT row_id FROM turns WHERE content='It is for dentists in Ohio.'").fetchone()[0]
    assert s.turn(ohio)["owner_timeline"] == "c1"
    assert ohio in s.pointed("o") and ohio in s.pointed("c2")  # same row, pointed, never copied
    assert s.con.execute("SELECT count(*) FROM turns WHERE content='It is for dentists in Ohio.'").fetchone()[0] == 1
    # the default join is deterministic: no model call ran for it
    assert set(t for t, u in h.provider.by_tag.items() if u["calls"]) <= {"original", "original:clone_tx",
                                                                           "clone:c1", "clone:c2"}
    tx = s.con.execute("SELECT committed_sql FROM transactions WHERE kind='join' AND target='c1'").fetchone()
    assert tx[0].startswith("DEFAULT JOIN c1 reason=out_of_scope")
    # one clone call per clone: every original call either routed (clone_tx) or answered
    assert h.provider.by_tag["original:clone_tx"]["calls"] == 2
    assert "ohio" in h.world.deck_text().lower()
    assert holders(s) == ["c2"]
    # the clone only had its linked tools
    c2_tools = {c["tools"][i]["name"] for c in h.provider.calls if "slides_create" in str(c["tools"])
                for i in range(len(c["tools"]))}
    assert "site_write" not in c2_tools and "clone" not in c2_tools


def test_one_shot_clone_refused_with_violations_then_retry(harness):
    h = harness

    async def go():
        await h.start()
        s = h.store
        s.add_turn("o", "user", "text", "hello")
        s.add_turn("o", "assistant", "text", "hi")
        use = {"id": "t1", "name": "clone", "input": {"clone": [{
            "purpose": "p", "skills": ["clone"], "focus": True, "seed_quotes": ["hello", "hi", "nope"]}]}}
        child, text = await clone_in_turn(h=h, use=use)
        assert child is None
        assert "cannot be linked to a clone" in text and "matches no message" in text
        assert "end on a user turn" in text and "Your messages (row_id role: text):" in text
        (child,), text = await clone_in_turn(h=h, use=dict(use, input={"clone": [{
            "purpose": "p", "skills": ["website"], "focus": True, "seed_quotes": ["hello"]}]}))
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
        await h.start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Make a deck.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["slides"], seed_row_ids=[seed]))
        a1 = s.add_turn(child, "assistant", "text", "Which audience?")
        u1 = s.add_turn(child, "user", "text", "Retirees.")
        s.add_turn(child, "assistant", "tool_use", {"id": "x1", "input": {"title": "T"}}, "slides_create")
        s.add_turn(child, "user", "tool_result", {"tool_use_id": "x1", "content": "ok"}, "slides_create")
        a2 = s.add_turn(child, "assistant", "text", "Built it.")
        calls_before = len(h.provider.calls)
        await h.do_join(child, "done")
        assert len(h.provider.calls) == calls_before  # no model call
        joined = [r["row_id"] for r in s.view("o")][-3:]
        assert joined == [a1, u1, a2]  # text rows only, seed row skipped (already held)
        assert status(store=s, tl="o") == "active" and status(store=s, tl=child) == "joined"
        assert not [r for r in s.view("o") if r["kind"] in ("tool_use", "tool_result")
                    and r["owner_timeline"] == child]
        await h.machines.stop()

    asyncio.run(go())


def test_default_join_out_of_scope_and_failed(harness):
    h = harness

    async def go():
        await h.start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Make a site.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"], seed_row_ids=[seed]))
        s.add_turn(child, "assistant", "text", "Sure.")
        trig = s.add_turn(child, "user", "text", "Actually, slides.")
        rows, err = h.default_join_rows("o", child, "out_of_scope", trig)
        assert rows[-1] == trig and err is None
        rows, err = h.default_join_rows("o", child, "failed", trig)
        assert rows[-1] == trig and "failed" in err
        s.set_join(child, "failed", trigger=trig)
        assert h.default_join(src="o", child=child, reason="failed", trigger=trig, running=[])
        tail = s.view("o")[-3:]
        assert "failed" in tail[1]["content"] and tail[2]["row_id"] == trig  # error record, then trigger last
        await h.machines.stop()

    asyncio.run(go())


def test_working_view_after_compaction_bounds_the_default_join(harness):
    h = harness

    async def go():
        await h.start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Plan a long project.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"], seed_row_ids=[seed]))
        old_a = s.add_turn(child, "assistant", "text", "Early detail.")
        old_u = s.add_turn(child, "user", "text", "Early answer.")
        old_b = s.add_turn(child, "assistant", "text", "Long early reply.")
        fresh = s.add_turn(child, "user", "text", "Late request.")
        comp = s.add_compressed_turn(child, "assistant", "Early detail, condensed.", old_a, old_a)
        s.mark_compaction(child, [seed, comp, old_u, fresh])  # old_b dropped from the working view
        done = s.add_turn(child, "assistant", "text", "Done.")
        assert [r["row_id"] for r in s.working_view(child)] == [seed, comp, old_u, fresh, done]
        assert len(s.view(child)) == 7  # the full timeline is unchanged
        rows, _ = h.default_join_rows("o", child, "done", None)
        assert rows == [comp, old_u, fresh, done] and old_b not in rows and old_a not in rows
        await h.machines.stop()

    asyncio.run(go())


def test_silent_done_is_refused_until_the_clone_replies(tmp_path):
    store = Store(str(tmp_path / "d.db"))
    install(store)
    script = iter([
        [{"type": "tool_use", "id": "j1", "name": "join", "input": {"reason": "done"}}],
        [{"type": "text", "text": "I built the deck."},
         {"type": "tool_use", "id": "j2", "name": "join", "input": {"reason": "done"}}],
    ])
    h = Harness(store, FakeProvider(lambda *a: next(script)))

    async def go():
        await h.start()
        seed = store.add_turn("o", "user", "text", "Make a deck.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["slides"], seed_row_ids=[seed]))
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": "cloned"}, "clone")
        await h.run(child)
        assert h.silent_done_refusals == [child]
        refusal = [r for r in store.view(child) if r["kind"] == "tool_result"][0]
        assert "refused" in refusal["content"] and '"is_error": true' in refusal["content"]
        assert status(store=store, tl=child) == "joined" and store.timeline(child)["join_reason"] == "done"
        assert store.view("o")[-1]["content"] == "I built the deck."  # the reply was joined
        await h.machines.stop()

    asyncio.run(go())


def test_clone_skill_has_only_load_skill_clone_and_cancel(harness):
    s = harness.store
    s.create_timeline("o", None, ["clone"])
    assert [t["name"] for t in s.tools_for("o")] == ["load_skill", "clone", "cancel"]


def test_clone_is_never_linked_to_a_clone(harness):
    s = harness.store
    s.create_timeline("o", None, ["clone"])
    assert check_skill_link(s, "o", ["slides"]) == []
    assert any("cannot be linked" in v for v in check_skill_link(s, "o", ["clone", "slides"]))


def test_progressive_disclosure_system_prompt(harness):
    s = harness.store
    s.create_timeline("o", None, ["clone"])
    rid = s.add_turn("o", "user", "text", "x")
    s.commit_clone("o", [Graft(child="c1", seed=[rid], skills=["website", "join"], focus=True)])
    sys_o, sys_c = harness.system("o"), harness.system("c1")
    for name in ("clone", "join", "website", "slides"):
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
    s.create_timeline("o", None, ["clone"])
    rid = s.add_turn("o", "user", "text", "x")
    s.commit_clone("o", [Graft(child="c1", seed=[rid], skills=["website", "join"], focus=True)])
    content, err = h.load("c1", "website")
    assert not err and content.startswith("Website skill")
    content, err = h.load("c1", "slides")
    assert err and "not linked" in content
    content, err = h.load("o", "website")  # the original holds only clone
    assert err and "not linked" in content
    assert not h.load("o", "clone")[1]


def test_load_skill_result_is_an_ordinary_row(tmp_path):
    store = Store(str(tmp_path / "p.db"))
    install(store)
    h = Harness(store, FakeProvider(pivot_policy))

    async def go():
        await h.start()
        await run_script(h, ["I need a website.", "For dentists in Ohio."], echo=False)
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
    install(store)
    h = Harness(store, FakeProvider(_no_compact_policy), compact_tokens=50, window_tokens=400)
    limit = int(400 * JOIN_WINDOW_FRACTION)

    async def go():
        await h.start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Write a long report.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"], seed_row_ids=[seed]))
        big = []
        for i in range(3):
            big.append(s.add_turn(child, "assistant", "text", f"Section {i}: " + "x" * 1000))
            big.append(s.add_turn(child, "user", "text", f"More on {i}."))
        last = s.add_turn(child, "assistant", "text", "Report finished.")
        trig = s.add_turn(child, "user", "text", "Now make slides instead.")
        await h.do_join(child, "out_of_scope")
        view = [r["row_id"] for r in s.working_view("o")]
        assert view[-3:-1] == [last, trig]  # tier 1: the clone's last reply, then the trigger
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
        await h.start()
        s = h.store
        seed = s.add_turn("o", "user", "text", "Write.")
        (child,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"], seed_row_ids=[seed]))
        a = s.add_turn(child, "assistant", "text", "y" * 50000)
        rows, record = h.fit_join(src="o", child=child, reason="done", rows=[a], error=None, trigger=None, running=[])
        assert rows == [a] and "follow" in record and h.join_fallbacks == []
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
    assert Harness(store, fp).window_tokens == 1_000_000  # on by default for a known model
    monkeypatch.setenv("GRAFT_WINDOW_TOKENS", "5000")
    assert Harness(store, fp).window_tokens == 5000
    monkeypatch.setenv("GRAFT_WINDOW_TOKENS", "0")
    assert Harness(store, fp).window_tokens is None
    assert Harness(store, fp, window_tokens=123).window_tokens == 123  # explicit wins


def _always_fails_policy(system, messages, tools):
    """The original clones for every message unless told a clone already failed; the clone
    loops on a tool until its step budget runs out (join failed)."""
    names = {t["name"] for t in tools}
    last = messages[-1]["content"]
    if "clone" in names:
        if any(b["type"] == "tool_result" and "already failed" in str(b["content"]) for b in last):
            return [{"type": "text", "text": "Sorry, I could not build that."}]
        user = [b["text"] for m in messages if m["role"] == "user" for b in m["content"] if b["type"] == "text"][-1]
        return [{"type": "tool_use", "id": f"c{len(messages)}", "name": "clone",
                 "input": {"clone": [{"purpose": "p", "skills": ["website"], "focus": True, "seed_quotes": [user]}]}}]
    return [{"type": "tool_use", "id": f"r{len(messages)}", "name": "site_read", "input": {"page": "home"}}]


def test_failed_clone_is_not_recloned_for_the_same_message(tmp_path):
    store = Store(str(tmp_path / "f.db"))
    install(store)
    h = Harness(store, FakeProvider(_always_fails_policy), seeder=None)

    async def go():
        await h.start()
        out = await h.user_says("Build my site.")
        assert out == ["Sorry, I could not build that."]
        assert clones(store) == ["c1"]  # cloned once, not again
        assert store.timeline("c1")["join_reason"] == "failed"
        assert holders(store) == ["o"]
        out = await h.user_says("Try a different page.")  # a new message may clone again
        assert "c2" in clones(store)
        await h.machines.stop()

    asyncio.run(go())


def test_a_request_that_bounces_back_twice_is_not_recloned(harness):
    h = harness

    async def go():
        await h.start()
        s = h.store
        trig = s.add_turn("o", "user", "text", "Make a site.")
        (c1,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"], seed_row_ids=[trig], id="a"))
        s.add_turn(c1, "assistant", "text", "Hm.")  # then the same request comes back, unhandled
        h.provider.policy = lambda *a: [{"type": "text", "text": "I could not do that."}]
        await h.do_join(child=c1, reason="out_of_scope")  # the first arrival: the source answers the request itself
        s.add_turn("o", "assistant", "tool_use", {"id": "b", "input": {}}, "clone")
        s.add_turn("o", "user", "tool_result", {"tool_use_id": "b", "content": "ok"}, "clone")
        # after one arrival the part may be cloned for again
        (c2,), _ = await clone_in_turn(h=h, use=clone_use(skills=["website"], seed_row_ids=[trig], id="b"))
        await h.do_join(c2, "out_of_scope")  # the same request comes back a second time
        again, text = await clone_in_turn(h=h, use=clone_use(skills=["website"], seed_row_ids=[trig], id="c"))
        assert again is None and "handed it back twice" in text  # now the part is blocked
        # another part of the same message is not blocked
        (c3,), _ = await clone_in_turn(h=h, use=clone_use(skills=["slides"], seed_row_ids=[trig], id="d"))
        await h.machines.stop()

    asyncio.run(go())
