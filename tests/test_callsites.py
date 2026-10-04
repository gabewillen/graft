"""Per-call-site reasoning effort and model: resolution, tagging, and per-model pricing."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from graft import cachestats, callsites
from graft.db import Store
from graft.fake import pivot_policy
from graft.harness import Harness
from graft.providers import FakeProvider, OpenAIProvider
from graft.sim import run_script
from graft.skills import install

ENV = [f"GRAFT_{w}_{s.upper()}" for w in ("EFFORT", "MODEL") for s in callsites.SITES] + \
      ["GRAFT_REASONING_EFFORT", "GRAFT_MODEL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def user(text):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def test_site_mapping():
    tr = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]}]
    assert callsites.site_for("original", user("hi")) == "clone_call"
    assert callsites.site_for("original", tr) == "original"
    assert callsites.site_for("original:clone_tx") == "original"
    assert callsites.site_for("clone:c3") == "clone"
    assert callsites.site_for("clone:c3:compaction") == "compaction"
    assert callsites.site_for("parent") == "subagent_parent"
    assert callsites.site_for("subagent") == "subagent"
    assert callsites.site_for("parent:compaction") == "compaction"


def test_resolution_order(clean_env):
    m = clean_env
    assert callsites.effort_for("clone_call") == "low"  # built-in default
    m.setenv("GRAFT_REASONING_EFFORT", "medium")
    assert callsites.effort_for("clone") == "medium" and callsites.effort_for("clone_call") == "medium"
    m.setenv("GRAFT_EFFORT_ORIGINAL", "minimal")
    assert callsites.effort_for("clone_call") == "minimal"  # clone_call falls back to original
    m.setenv("GRAFT_EFFORT_CLONE_CALL", "high")
    assert callsites.effort_for("clone_call") == "high" and callsites.effort_for("original") == "minimal"
    assert callsites.effort_for("clone") == "medium"
    assert callsites.model_for("clone_call") is None
    m.setenv("GRAFT_MODEL_ORIGINAL", "gpt-x")
    assert callsites.model_for("clone_call") == "gpt-x" and callsites.model_for("clone") is None
    m.setenv("GRAFT_MODEL_CLONE_CALL", "gpt-y")
    assert callsites.model_for("clone_call") == "gpt-y"


def test_harness_tags_each_call_with_site_effort_and_model(tmp_path, clean_env):
    clean_env.setenv("GRAFT_EFFORT_CLONE_CALL", "high")
    clean_env.setenv("GRAFT_MODEL_CLONE", "model-b")
    store = Store(str(tmp_path / "s.db"))
    install(store)
    fp = FakeProvider(pivot_policy)
    h = Harness(store, fp)

    async def go():
        await h.start()
        await run_script(h, ["I need a website.", "It is for dentists in Ohio."], echo=False)
        await h.machines.stop()

    asyncio.run(go())
    route = [c for c in fp.call_log if c["tag"] == "original:clone_tx"]
    assert route and all(c["site"] == "clone_call" and c["effort"] == "high" and c["model"] == "fake"
                         for c in route)
    work = [c for c in fp.call_log if c["tag"].startswith("clone:")]
    assert work and all(c["site"] == "clone" and c["effort"] == "low" and c["model"] == "model-b" for c in work)
    assert set(fp.by_model) == {"fake", "model-b"}


def test_openai_provider_sends_per_call_model_and_effort():
    calls = []
    resp = NS(usage=NS(input_tokens=10, output_tokens=2, input_tokens_details=NS(cached_tokens=0,
                                                                                    cache_write_tokens=0)),
              output=[])
    p = object.__new__(OpenAIProvider)
    p.model, p.api, p.effort, p.use_cache_key = "gpt-6-luna", "responses", "low", False
    p._init_usage()
    p.client = NS(responses=NS(create=lambda **kw: calls.append(kw) or resp))
    p.complete("S", user("hi"), [], model="gpt-6-other", effort="high", site="clone_call")
    p.complete("S", user("hi"), [])
    assert calls[0]["model"] == "gpt-6-other" and calls[0]["reasoning"] == {"effort": "high"}
    assert calls[1]["model"] == "gpt-6-luna" and calls[1]["reasoning"] == {"effort": "low"}
    assert p.call_log[0]["model"] == "gpt-6-other" and p.call_log[0]["site"] == "clone_call"
    assert set(p.by_model) == {"gpt-6-other", "gpt-6-luna"}


def test_mixed_model_pricing_unknown_is_na(clean_env):
    pr = cachestats.Pricing.from_env()
    u = {"calls": 1, "uncached": 1_000_000, "cached": 0, "cache_write": 0, "output": 1_000_000}
    assert abs(pr.cost_model("gpt-6-luna", u) - 0.60) < 1e-9  # official table
    assert pr.cost_model("gpt-unknown", u) is None
    assert pr.cost_by_model({"gpt-6-luna": u, "gpt-unknown": u}) is None  # any unknown model -> n/a
    assert pr.cost_by_model({"gpt-6-luna": u, "gpt-unknown": dict(u, calls=0)}) is not None
    clean_env.setenv("GRAFT_PRICE_IN_PER_M__GPT_UNKNOWN", "1")
    clean_env.setenv("GRAFT_PRICE_OUT_PER_M__GPT_UNKNOWN", "2")
    pr = cachestats.Pricing.from_env()
    assert abs(pr.cost_by_model({"gpt-6-luna": u, "gpt-unknown": u}) - (0.60 + 3.0)) < 1e-9


def test_global_prices_apply_only_to_the_default_model(clean_env):
    clean_env.setenv("GRAFT_MODEL", "my-model")
    clean_env.setenv("GRAFT_PRICE_IN_PER_M", "1")
    clean_env.setenv("GRAFT_PRICE_OUT_PER_M", "1")
    pr = cachestats.Pricing.from_env()
    assert pr.prices_for("my-model")["in"] == 1 and pr.prices_for("other-model") is None
