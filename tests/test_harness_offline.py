"""End-to-end harness run with the scripted FakeProvider (no API key needed)."""

import asyncio

import pytest

from graft.db import Store
from graft.fake import pivot_policy
from graft.harness import Harness, check_skill_link
from graft.providers import FakeProvider
from graft.sim import run_script
from graft.skills import install


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
        h.machines.check()
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
    assert s.running_leaf() == "c2"
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
        use = {"id": "t1", "name": "clone", "input": {"purpose": "p", "skills": ["clone"],
                                                       "seed_quotes": ["hello", "hi", "nope"]}}
        child, text = await h.clone_once("o", use, retry_left=True)
        assert child is None
        assert "cannot be linked to a clone in v1" in text and "matches no message" in text
        assert "end on a user turn" in text and "Your messages (row_id role: text):" in text
        child, text = await h.clone_once("o", dict(use, input={"purpose": "p", "skills": ["website"],
                                                               "seed_quotes": ["hello"]}), retry_left=False)
        assert child == "c1" and h.machines.state("c1") == "active" and h.machines.state("o") == "cloned"
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
        child, _ = await h.clone_once("o", {"id": "t", "name": "clone", "input": {
            "purpose": "p", "skills": ["slides"], "seed_row_ids": [seed]}}, retry_left=False)
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
        assert h.machines.state("o") == "active" and h.machines.state(child) == "joined"
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
        child, _ = await h.clone_once("o", {"id": "t", "name": "clone", "input": {
            "purpose": "p", "skills": ["website"], "seed_row_ids": [seed]}}, retry_left=False)
        s.add_turn(child, "assistant", "text", "Sure.")
        trig = s.add_turn(child, "user", "text", "Actually, slides.")
        rows, err = h.default_join_rows("o", child, "out_of_scope", trig)
        assert rows[-1] == trig and err is None
        rows, err = h.default_join_rows("o", child, "failed", trig)
        assert rows[-1] == trig and "failed" in err
        s.set_join(child, "failed", trig)
        await h.machines.join(child)
        assert h.default_join("o", child, "failed", trig)
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
        child, _ = await h.clone_once("o", {"id": "t", "name": "clone", "input": {
            "purpose": "p", "skills": ["website"], "seed_row_ids": [seed]}}, retry_left=False)
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
        child, _ = await h.clone_once("o", {"id": "t", "name": "clone", "input": {
            "purpose": "p", "skills": ["slides"], "seed_row_ids": [seed]}}, retry_left=False)
        store.add_turn("o", "assistant", "tool_use", {"id": "t", "input": {}}, "clone")
        store.add_turn("o", "user", "tool_result", {"tool_use_id": "t", "content": "cloned"}, "clone")
        await h.run(child)
        assert h.silent_done_refusals == [child]
        refusal = [r for r in store.view(child) if r["kind"] == "tool_result"][0]
        assert "refused" in refusal["content"] and '"is_error": true' in refusal["content"]
        assert h.machines.state(child) == "joined" and store.timeline(child)["join_reason"] == "done"
        assert store.view("o")[-1]["content"] == "I built the deck."  # the reply was joined
        await h.machines.stop()

    asyncio.run(go())


def test_clone_skill_has_only_load_skill_and_clone(harness):
    s = harness.store
    s.create_timeline("o", None, ["clone"])
    assert [t["name"] for t in s.tools_for("o")] == ["load_skill", "clone"]


def test_skill_subset_rule_for_nested_sources(harness):
    s = harness.store
    s.create_timeline("o", None, ["clone"])
    rid = s.add_turn("o", "user", "text", "x")
    s.commit_clone("o", "c1", [rid], ["website", "join"])
    assert check_skill_link(s, "o", ["slides"]) == []
    assert any("subset" in v for v in check_skill_link(s, "c1", ["slides"]))


def test_progressive_disclosure_system_prompt(harness):
    s = harness.store
    s.create_timeline("o", None, ["clone"])
    rid = s.add_turn("o", "user", "text", "x")
    s.commit_clone("o", "c1", [rid], ["website", "join"])
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
    s.commit_clone("o", "c1", [rid], ["website", "join"])
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
