"""Offline coverage for the benchmark: every scenario x arm runs with the fake policies, the
scorers grade synthetic final states correctly, and compaction works the same in both arms."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import bench  # noqa: E402
from scenarios import SCENARIOS, RunContext, score_failure, score_long, score_multi, score_revisions  # noqa: E402

from graft import compaction  # noqa: E402
from graft.db import Store  # noqa: E402
from graft.fake import bench_graft_policy, bench_subagent_policy  # noqa: E402
from graft.harness import Harness  # noqa: E402
from graft.providers import FakeProvider  # noqa: E402
from graft.sim import run_script  # noqa: E402
from graft.skills import World, install  # noqa: E402
from graft.subagent import SubagentRunner  # noqa: E402


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("arm", ["graft", "subagent"])
def test_every_scenario_runs_offline(name, arm):
    sc = SCENARIOS[name]
    rec = asyncio.run(bench.run_one(sc, arm, 0, fake=True, compact=sc.compact_tokens, nonce="n"))
    assert rec["error"] is None, rec.get("traceback")
    assert set(rec) >= {"ok", "checks", "diag", "roles", "total", "latency_s", "handoffs", "context_out",
                        "compactions"}
    assert rec["total"]["calls"] > 0 and rec["handoffs"] >= 1
    assert isinstance(rec["ok"], bool) and rec["checks"]


def _ctx(world, texts=(), arm="subagent"):
    return RunContext(arm, world, [("assistant", t) for t in texts], agent=None)


def test_revisions_scorer():
    w = World()
    w.decks.append({"title": "Riverside Library Book Fair", "audience": "Families with young children",
                    "tone": "Warm and professional", "slides": [
                        {"title": "Welcome To The Book Fair", "bullets": ["a", "b", "c"]},
                        {"title": "Volunteers", "bullets": ["a", "b", "c"]},
                        {"title": "Schedule", "bullets": ["a", "b", "c"]},
                        {"title": "Contact Us", "bullets": ["Email fair@riversidelib.org", "b", "c"]}]})
    ok, checks, _ = score_revisions(_ctx(w))
    assert ok, checks
    w.decks[-1]["slides"][0]["title"] = "Welcome To Riversde"
    ok, checks, _ = score_revisions(_ctx(w))
    assert not ok and not checks["typo_fixed"] and not checks["welcome_retitled_first"]


def test_failure_scorer_honesty():
    w = World(failures={"calendar_book": "ERROR"})
    w.execute("calendar_book", {"date": "2026-11-10", "time": "14:00", "what": "x"})
    assert not w.calendar
    assert score_failure(_ctx(w, ["Sorry, the calendar service is unavailable, so I could not book it."]))[0]
    assert not score_failure(_ctx(w, ["Your cleaning is booked for 2026-11-10 at 14:00."]))[0]
    assert not score_failure(_ctx(World(), ["I could not book it."]))[0]  # never attempted


def test_long_and_multi_scorers():
    w = World()
    w.decks.append({"title": "Harbor Light Dental", "audience": "Families in Duluth", "tone": "formal",
                    "slides": [{"title": "Our Dentists", "bullets": ["Dr. Amara Okafor", "Dr. Lena Brandt",
                                                                     "Call 218-555-0143"]},
                               {"title": "Services", "bullets": ["Cleanings", "Crowns", "Invisalign"]}]})
    ok, checks, diag = score_long(_ctx(w))
    assert ok, checks
    assert diag["diag_facts"]["families"] and not diag["diag_facts"]["tagline"]
    w2 = World()
    w2.execute("calendar_book", {"date": "2026-11-12", "time": "09:30", "what": "cleaning"})
    w2.decks.append({"title": "Open House", "audience": "local seniors", "tone": "formal",
                     "slides": [{"title": "Cleanings and Implants", "bullets": ["a", "b", "c"]}]})
    assert score_multi(_ctx(w2))[0]
    assert not score_multi(_ctx(World()))[0]


def test_subagent_compaction_replaces_parent_messages():
    store = Store(":memory:")
    install(store)
    fp = FakeProvider(bench_subagent_policy)
    r = SubagentRunner(store, fp, compact_tokens=30)

    async def go():
        await run_script(r, ["Our practice is Harbor Light Dental in Duluth, Minnesota near Canal Park.",
                             "The dentists are Dr. Amara Okafor and Dr. Lena Brandt, founded in 1987."], echo=False)

    asyncio.run(go())
    assert r.compactions == ["parent"] or "parent" in r.compactions
    assert r.messages[0]["content"][0]["text"].startswith(compaction.SUMMARY_HEADER)
    assert any("Dr. Amara Okafor" in b.get("text", "") for b in r.messages[0]["content"][1:])
    assert fp.by_tag["parent:compaction"]["calls"] >= 1


def test_estimate_and_threshold():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x" * 400}]}]
    assert compaction.estimate_tokens(msgs) == 100
    assert compaction.needs_compaction(msgs, 50) and not compaction.needs_compaction(msgs, 200)
    assert not compaction.needs_compaction(msgs, None)
    tail_assistant = msgs + [{"role": "assistant", "content": [{"type": "text", "text": "y"}]}]
    assert not compaction.needs_compaction(tail_assistant, 10)  # only before answering a user message
    req = compaction.request(tail_assistant)
    assert req[-1]["role"] == "user" and req[-1]["content"][-1]["text"] == compaction.COMPACT_INSTRUCTION


def test_welch_matches_known_values():
    t, df, p = bench.welch([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
    assert abs(t - (-1.8974)) < 1e-3 and abs(df - 5.882) < 1e-2 and abs(p - 0.1075) < 2e-3


# ------------------------------------------------------------------ graft compaction (turns)
def _compacting_harness(tmp_path, policy=None, mode="turns", threshold=80):
    store = Store(str(tmp_path / "c.db"))
    install(store)
    fp = FakeProvider(policy or bench_graft_policy)
    return store, fp, Harness(store, fp, compact_tokens=threshold, compaction_mode=mode)


LONG_SCRIPT = ["I need a website for my dental practice.",
               "The practice is called Harbor Light Dental, in Duluth, Minnesota, near Canal Park.",
               "Our services are cleanings, crowns, and Invisalign; phone 218-555-0143."]


def test_turns_compaction_keeps_user_rows_and_compresses_windows(tmp_path):
    store, fp, h = _compacting_harness(tmp_path)

    async def go():
        await h.start()
        await run_script(h, LONG_SCRIPT, echo=False)
        await h.machines.stop()

    asyncio.run(go())
    assert h.compactions and fp.by_tag["clone:c1:compaction"]["calls"] >= 1
    e = h.compaction_log[0]
    wv = store.working_view("c1")
    # no synthesized user row: every user text row in the working view is an original user turn
    for r in wv:
        if r["role"] == "user" and r["kind"] == "text":
            assert store.compression_window(r["row_id"]) is None
    comp = [r for r in wv if store.compression_window(r["row_id"])]
    assert comp and all(r["role"] == "assistant" and r["owner_timeline"] == "c1" for r in comp)
    # rendered as a normal turn in its role: no visible marker in the text
    assert all(not r["content"].startswith("(") for r in comp)
    # the window covers a contiguous run of the prefix, placed at its position
    ids = [p[0] for p in e["prefix"]]
    w = e["windows"][0]
    assert w["covers"] == ids[ids.index(w["begin"]):ids.index(w["end"]) + 1]
    assert e["after_tokens"] < e["before_tokens"]
    fresh = [r for r in wv if r["content"] == e["fresh"]]
    assert fresh and fresh[0]["role"] == "user"  # the fresh message stays verbatim after the items


def test_partition_validation():
    prefix = [{"row_id": i, "role": r, "kind": "text", "owner_timeline": "c1", "tool_name": None,
               "content": f"r{i}"} for i, r in zip((10, 11, 12, 13), ("user", "assistant", "user", "assistant"))]
    rows, probs = compaction.partition(prefix, [{"keep": 10}, {"role": "assistant", "text": "x",
                                                                "begin_row_id": 11, "end_row_id": 13}])
    assert not probs and [r["row_id"] for r in rows] == [10, None] and rows[1]["window"] == (11, 13)
    _, probs = compaction.partition(prefix, [{"keep": 10}, {"keep": 12}, {"keep": 13}])
    assert probs and "next uncovered row is 11" in probs[0]  # gap
    _, probs = compaction.partition(prefix, [{"keep": 10}, {"role": "user", "text": "x", "begin_row_id": 11,
                                                             "end_row_id": 12}, {"keep": 12}, {"keep": 13}])
    assert probs and "next uncovered row is 13" in probs[0]  # overlap
    _, probs = compaction.partition(prefix, [{"keep": 10}, {"keep": 11}])
    assert probs and "not covered" in probs[0]  # coverage
    # a window item with unused fields filled in (keep: 0, as models do) is still a window
    rows, probs = compaction.partition(prefix, [{"keep": 10, "role": "", "text": "", "begin_row_id": None},
                                                {"keep": 0, "role": "assistant", "text": "x", "begin_row_id": 11,
                                                 "end_row_id": 13}])
    assert not probs and rows[1]["window"] == (11, 13)
    _, probs = compaction.partition(prefix, [{"keep": 10}, {"role": "assistant", "text": "x",
                                                             "begin_row_id": 13, "end_row_id": 11}])
    assert probs  # window out of order
    _, probs = compaction.partition(prefix, [{"keep": 10}, {"role": "assistant", "text": "",
                                                             "begin_row_id": 11, "end_row_id": 13}])
    assert probs and "must be" in probs[0]  # empty text


def test_invalid_compaction_gets_one_retry_then_gives_up(tmp_path):
    calls = {"n": 0}

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        if names == {"compact"}:
            calls["n"] += 1
            return [{"type": "tool_use", "id": f"k{calls['n']}", "name": "compact", "input": {"items": [{"keep": 1}]}}]
        return bench_graft_policy(system, messages, tools)

    store, fp, h = _compacting_harness(tmp_path, policy)

    async def go():
        await h.start()
        await run_script(h, LONG_SCRIPT[:2], echo=False)
        await h.machines.stop()

    asyncio.run(go())
    assert calls["n"] == 2 and h.compaction_failures and not h.compaction_log
    assert not store.con.execute("SELECT 1 FROM compactions").fetchone()


def test_turn_structure_validated_after_partition(tmp_path):
    """A partition that leaves a tool_use without its tool_result is refused by the validator."""
    store = Store(str(tmp_path / "v.db"))
    install(store)
    store.create_timeline("o", None, ["clone"])
    u = store.add_turn("o", "user", "text", "hi")
    tu = store.add_turn("o", "assistant", "tool_use", {"id": "x", "input": {}}, "load_skill")
    tr = store.add_turn("o", "user", "tool_result", {"tool_use_id": "x", "content": "ok"}, "load_skill")
    a = store.add_turn("o", "assistant", "text", "done")
    prefix = store.view("o")
    rows, probs = compaction.partition(prefix, [{"keep": u}, {"keep": tu}, {"role": "user", "text": "y",
                                                                           "begin_row_id": tr, "end_row_id": a}])
    assert not probs
    from graft.db import validate
    assert validate(rows + [{"row_id": 99, "owner_timeline": "o", "role": "user", "kind": "text",
                             "tool_name": None, "content": "fresh"}], "o")


def test_summary_mode_is_one_user_row_over_the_whole_prefix(tmp_path):
    store, fp, h = _compacting_harness(tmp_path, mode="summary")

    async def go():
        await h.start()
        await run_script(h, LONG_SCRIPT, echo=False)
        await h.machines.stop()

    asyncio.run(go())
    e = h.compaction_log[-1]
    assert len(e["windows"]) == 1 and e["kept_ids"] == []
    w = e["windows"][0]
    assert w["begin"] == e["prefix"][0][0] and w["end"] == e["prefix"][-1][0]
    wv = store.working_view(e["timeline"])
    assert wv[0]["role"] == "user" and store.compression_window(wv[0]["row_id"]) == (w["begin"], w["end"])


def test_working_view_follows_latest_compaction(tmp_path):
    s = Store(str(tmp_path / "w.db"))
    install(s)
    s.create_timeline("o", None, ["clone"])
    a = s.add_turn("o", "user", "text", "old 1")
    b = s.add_turn("o", "assistant", "text", "old 2")
    fresh = s.add_turn("o", "user", "text", "fresh request")
    s.mark_compaction("o", [a, fresh])
    assert [r["row_id"] for r in s.working_view("o")] == [a, fresh]
    reply = s.add_turn("o", "assistant", "text", "reply")
    nxt = s.add_turn("o", "user", "text", "next")
    c = s.add_compressed_turn("o", "user", "old 1 and fresh request", a, fresh)
    s.mark_compaction("o", [c, reply, nxt])
    assert [r["row_id"] for r in s.working_view("o")] == [c, reply, nxt]
    assert {a, b, fresh} <= {r["row_id"] for r in s.view("o")}  # full view intact


def test_fact_survival_classification():
    from scenarios import fact_survival

    log = [{"timeline": "c1", "prefix": [(1, "We are Harbor Light Dental."), (2, "Nice."), (3, "Call 218-555-0143."),
                                          (4, "We are in Duluth.")],
            "kept_ids": [1], "windows": [{"row_id": 9, "begin": 2, "end": 3, "covers": [2, 3],
                                          "text": "Phone 218-555-0143."},
                                         {"row_id": 10, "begin": 4, "end": 4, "covers": [4], "text": "A city."}]}]
    out = fact_survival(log, {"name": r"harbor light", "phone": r"218-555-0143", "city": r"duluth",
                              "x": r"okafor"})
    assert out == {"name": "verbatim", "phone": "paraphrased", "city": "lost", "x": "uncompacted"}
    sub = [{"timeline": "parent", "prefix_texts": ["Harbor Light Dental", "Duluth"], "summary": "Harbor Light",
            "fresh": ""}]
    assert fact_survival(sub, {"name": r"harbor light", "city": r"duluth"}) == {"name": "paraphrased",
                                                                                 "city": "lost"}


def test_compaction_over_budget_is_refused(tmp_path):
    """A partition that keeps everything does not free room: refused by the budget check."""
    def keep_all(system, messages, tools):
        if {t["name"] for t in tools} == {"compact"}:
            import re as _re
            ids = [int(x) for x in _re.findall(r"row_id=(\d+) ", messages[-1]["content"][-1]["text"])]
            return [{"type": "tool_use", "id": "k", "name": "compact", "input": {"items": [{"keep": i} for i in ids]}}]
        return bench_graft_policy(system, messages, tools)

    store, fp, h = _compacting_harness(tmp_path, keep_all)

    async def go():
        await h.start()
        await run_script(h, LONG_SCRIPT, echo=False)
        await h.machines.stop()

    asyncio.run(go())
    assert h.compaction_failures and not h.compaction_log
