"""Prompt-cache accounting and request shapes (offline: stub clients, no API key)."""

import asyncio
from types import SimpleNamespace as NS

from graft import cachestats
from graft.db import Store
from graft.fake import pivot_policy
from graft.harness import Harness
from graft.providers import (AnthropicProvider, FakeProvider, OpenAIProvider, _accepts, prefix_nonce,
                               prompt_cache_key)
from graft.sim import run_script
from graft.skills import install

TOOLS = [{"name": "b_tool", "description": "b", "input_schema": {"type": "object", "properties": {}}},
         {"name": "a_tool", "description": "a", "input_schema": {"type": "object", "properties": {}}}]
MSGS = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


class Recorder:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, []

    def create(self, **kw):
        self.kwargs.append(kw)
        return self.resp


def test_anthropic_request_enables_caching_and_reads_cache_usage():
    resp = NS(stop_reason="end_turn", content=[NS(type="text", text="hi")],
              usage=NS(input_tokens=40, output_tokens=7, cache_read_input_tokens=900,
                       cache_creation_input_tokens=120))
    p = object.__new__(AnthropicProvider)
    p.model = "claude-sonnet-5-5"
    p._init_usage()
    rec = Recorder(resp)
    p.client = NS(messages=rec)
    with p.usage_tag("original"):
        assert p.complete("PERSONA", MSGS, TOOLS) == [{"type": "text", "text": "hi"}]
    kw = rec.kwargs[0]
    # explicit breakpoint on the last (only) system block: caches tools + system
    assert kw["system"] == [{"type": "text", "text": "PERSONA", "cache_control": {"type": "ephemeral"}}]
    # top-level automatic caching for the growing conversation tail
    assert kw["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in kw["tools"]] == ["b_tool", "a_tool"]  # order preserved (prefix stability)
    assert kw["thinking"] == {"type": "between_tools"}
    assert p.usage == {"calls": 1, "input": 1060, "uncached": 40, "cached": 900, "cache_write": 120,
                       "output": 7}
    assert p.by_tag["original"]["cached"] == 900
    assert p.call_log[0]["tag"] == "original"


def _openai(api: str, resp) -> tuple[OpenAIProvider, Recorder]:
    p = object.__new__(OpenAIProvider)
    p.model, p.api, p.effort, p.use_cache_key = "gpt-6-luna", api, "low", True
    p._init_usage()
    rec = Recorder(resp)
    p.client = NS(responses=rec, chat=NS(completions=rec))
    return p, rec


def test_openai_responses_splits_cached_tokens_and_sets_cache_key():
    usage = NS(input_tokens=2000, output_tokens=50,
               input_tokens_details=NS(cached_tokens=1536, cache_write_tokens=0))
    p, rec = _openai("responses", NS(usage=usage, output=[]))
    p.complete("SYS", MSGS, TOOLS)
    p.complete("SYS", MSGS, list(reversed(TOOLS)))
    key = rec.kwargs[0]["prompt_cache_key"]
    assert key == prompt_cache_key("SYS", TOOLS) == rec.kwargs[1]["prompt_cache_key"]  # tool order-free
    assert key != prompt_cache_key("SYS", TOOLS[:1]) and key != prompt_cache_key("SYS2", TOOLS)
    assert p.usage["uncached"] == 2 * 464 and p.usage["cached"] == 2 * 1536 and p.usage["input"] == 4000
    assert p.call_log[0]["cache_key"] == key and p.call_log[0]["raw_input"] == 2000


def test_openai_chat_reads_prompt_tokens_details():
    usage = NS(prompt_tokens=1500, completion_tokens=9, prompt_tokens_details=NS(cached_tokens=1024))
    msg = NS(content="ok", tool_calls=None)
    p, rec = _openai("chat", NS(usage=usage, choices=[NS(message=msg)]))
    p.use_cache_key = False
    p.complete("SYS", MSGS, [])
    assert "prompt_cache_key" not in rec.kwargs[0]
    assert (p.usage["uncached"], p.usage["cached"], p.usage["cache_write"]) == (476, 1024, 0)


def test_installed_openai_sdk_accepts_prompt_cache_key():
    import openai

    c = openai.OpenAI(api_key="offline-test-not-a-key")
    assert _accepts(c.responses.create, "prompt_cache_key")
    assert _accepts(c.chat.completions.create, "prompt_cache_key")


def test_prefix_nonce_copies_and_prefixes_first_message_only():
    out = prefix_nonce(MSGS + [{"role": "assistant", "content": [{"type": "text", "text": "x"}]}], "abc")
    assert out[0]["content"][0]["text"] == "[conversation abc]\nhello"
    assert out[1]["content"][0]["text"] == "x"
    assert MSGS[0]["content"][0]["text"] == "hello"  # input untouched
    assert prefix_nonce(MSGS, None) is MSGS
    tr = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]}]
    assert prefix_nonce(tr, "n")[0]["content"][0] == {"type": "text", "text": "[conversation n]"}


def test_harness_applies_nonce_and_tags_roles(tmp_path):
    store = Store(str(tmp_path / "c.db"))
    install(store)
    fp = FakeProvider(pivot_policy)
    h = Harness(store, fp, nonce="run123")

    async def go():
        await h.start()
        await run_script(h, ["I need a website.", "It is for dentists in Ohio.",
                             "Instead let's build a slide show."], echo=False)

    asyncio.run(go())
    assert fp.calls and all(c["messages"][0]["content"][0]["text"].startswith("[conversation run123]")
                            for c in fp.calls)
    assert not store.con.execute("SELECT 1 FROM turns WHERE content LIKE '%run123%'").fetchone()
    assert {"original:clone_tx", "clone:c1", "clone:c2"} <= set(fp.by_tag)
    assert not any("join" in t for t in fp.by_tag)  # the default join makes no model call
    assert sum(u["calls"] for u in fp.by_tag.values()) == len(fp.calls)


def test_effective_input_cost_and_hit_rate():
    u = {"calls": 2, "uncached": 1000, "cached": 4000, "cache_write": 200, "output": 100}
    p = cachestats.Pricing()
    assert p.effective_input(u) == 1000 + 400 + 250
    assert p.cost(u) is None
    priced = cachestats.Pricing(price_in=2.0, price_out=10.0, price_cached=0.2)
    assert abs(priced.cost(u) - (1000 * 2 + 4000 * 0.2 + 200 * 2.5 + 100 * 10) / 1e6) < 1e-12
    assert abs(cachestats.hit_rate(u) - 4000 / 5200) < 1e-12
    lines = cachestats.table([("clone", u)], priced, per=2)
    assert len(lines) == 3 and "$" in lines[1]


def test_pricing_from_env(monkeypatch):
    monkeypatch.setenv("GRAFT_CACHED_READ_MULT", "0")
    monkeypatch.setenv("GRAFT_PRICE_IN_PER_M", "1")
    monkeypatch.setenv("GRAFT_PRICE_OUT_PER_M", "4")
    p = cachestats.Pricing.from_env()
    assert p.read_mult == 0 and p.write_mult == 1.25 and p.priced


def test_subagent_baseline_isolates_subagent_and_tags_roles():
    from graft.fake import subagent_policy
    from graft.subagent import DELEGATE_TOOL, SubagentRunner

    store = Store(":memory:")
    install(store)
    fp = FakeProvider(subagent_policy)
    r = SubagentRunner(store, fp, nonce="n1")

    async def go():
        await run_script(r, ["It is for dentists in Ohio.", "Formal tone.", "Let's build a slide show."],
                         echo=False)

    asyncio.run(go())
    assert [d.skill for d in r.delegations] == ["slides"] and r.world.decks
    d = r.delegations[0]
    sub_calls = [c for c in fp.calls if c["tools"] != [DELEGATE_TOOL]]
    # the subagent sees only the parent-written prompt (plus its own tool loop), never the user turns
    assert all(c["messages"][0]["content"][0]["text"] == f"[conversation n1]\n{d.task_prompt}" for c in sub_calls)
    assert all("Formal tone." != b.get("text") for c in sub_calls for m in c["messages"] for b in m["content"])
    assert {t: u["calls"] for t, u in fp.by_tag.items() if u["calls"]} == {
        "parent": 3, "handoff": 1, "subagent": 2, "summary": 1}
    assert d.summary == "Built a formal deck for the dental practice."
    assert d.first_call is not None and d.first_call["tag"] == "subagent"


def test_retag_last_moves_usage_between_tags():
    fp = FakeProvider(lambda *a: [])
    with fp.usage_tag("parent"):
        fp._record(10, 2, cached=5)
    fp.retag_last("handoff")
    assert fp.by_tag["parent"]["calls"] == 0 and fp.by_tag["handoff"]["cached"] == 5
    assert fp.call_log[-1]["tag"] == "handoff" and fp.usage["calls"] == 1
