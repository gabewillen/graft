"""The Jev seed actor (graft/seeder.py) in the harness, with a fake decision client (offline)."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from graft import seeder as sd
from graft.db import Store
from graft.harness import Harness
from graft.providers import FakeProvider
from graft.skills import install


class FakeJev:
    """Async context manager; scores rows mentioning facts high. Records every request."""

    def __init__(self, fail=False):
        self.requests, self.fail = [], fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def system_one(self, state, questions, model=None):
        if self.fail:
            raise TimeoutError("jev timeout")
        self.requests.append((state, questions, model))
        nouls = {k: NS(noul=0.9 if any(w in q.instructions["message"] for w in ("Harbor", "218", "slides")) else 0.1)
                 for k, q in questions.items()}
        return NS(nouls=nouls, usage=NS(input_tokens=100 * len(questions)), model="jev-1.13.0")


def seeder(fake, budget=None):
    return sd.JevSeeder(client_factory=lambda: fake, budget=budget)


def setup(tmp_path, fake, monkeypatch, threshold="0.4"):
    monkeypatch.setenv("GRAFT_SEED_THRESHOLD", threshold)
    store = Store(str(tmp_path / "s.db"))
    install(store)
    h = Harness(store, FakeProvider(lambda *a: []), seeder=seeder(fake))
    return store, h


def rows(store):
    asst0 = store.add_turn("o", "assistant", "text", "Hello there.")  # assistant first: gets a lead user row
    u1 = store.add_turn("o", "user", "text", "I need a website.")
    a1 = store.add_turn("o", "assistant", "text", "Sure, which practice?")
    u2 = store.add_turn("o", "user", "text", "Harbor Light Dental, phone 218-555-0143.")
    note = store.add_turn("o", "user", "text", "(harness note: clone c1 joined, reason=out_of_scope; rows 1 follow)")
    a2 = store.add_turn("o", "assistant", "text", "Noted the details.")
    trig = store.add_turn("o", "user", "text", "Make slides instead.")
    return dict(asst0=asst0, u1=u1, a1=a1, u2=u2, note=note, a2=a2, trig=trig)


def test_jev_selects_by_threshold_with_trigger_and_audit(tmp_path, monkeypatch):
    fake = FakeJev()
    store, h = setup(tmp_path, fake, monkeypatch)

    async def go():
        await h.start()
        r = rows(store)
        # the original's own seed fields are ignored when the seeder is jev (schema unchanged)
        child, text = await h.clone_once("o", {"id": "t", "name": "clone", "input": {
            "purpose": "build slides", "skills": ["slides"], "seed_quotes": ["I need a website."]}}, retry_left=True)
        assert child == "c1", text
        seed = [x["row_id"] for x in store.view("c1")]
        assert seed == [r["u2"], r["trig"]]  # selected in view order + trigger; harness note never a candidate
        state, questions, model = fake.requests[0]
        assert model == sd.JEV_MODEL and set(state) == {"new_task", "latest_user_request"}  # reduced state
        assert f"row_{r['note']}" not in questions and len(questions) == 6
        q = questions[f"row_{r['u2']}"].instructions
        assert q["before"]["row_id"] == r["a1"] and q["after"]["row_id"] == r["a2"]  # neighbour rows
        tx = store.con.execute("SELECT scratch FROM transactions WHERE kind='clone' AND status='committed'").fetchone()
        audit = json.loads(tx[0])["seeder"]
        assert audit["threshold"] == 0.4 and audit["selected"] == seed and audit["fallback"] is None
        assert audit["scores"][str(r["u2"])] == 0.9 and audit["model"] == "jev-1.13.0" and audit["input_tokens"] == 600
        assert h.seeder_log[0]["latency"] >= 0
        await h.machines.stop()

    asyncio.run(go())


def test_leading_assistant_row_gets_nearest_earlier_user_row(tmp_path, monkeypatch):
    class OnlyAssistant(FakeJev):
        async def system_one(self, state, questions, model=None):
            nouls = {k: NS(noul=0.95 if "Noted" in q.instructions["message"] else 0.0) for k, q in questions.items()}
            return NS(nouls=nouls, usage=NS(input_tokens=10), model="jev-1.13.0")

    store, h = setup(tmp_path, OnlyAssistant(), monkeypatch)

    async def go():
        await h.start()
        r = rows(store)
        child, _ = await h.clone_once("o", {"id": "t", "name": "clone", "input": {"purpose": "p", "skills": ["slides"]}},
                                      retry_left=True)
        assert [x["row_id"] for x in store.view(child)] == [r["u2"], r["a2"], r["trig"]]
        assert h.seeder_log[0]["added_lead_user_row"] == r["u2"]
        await h.machines.stop()

    asyncio.run(go())


@pytest.mark.parametrize("with_model_seed", [True, False])
def test_seeder_failure_falls_back_and_is_recorded(tmp_path, monkeypatch, with_model_seed):
    store, h = setup(tmp_path, FakeJev(fail=True), monkeypatch)

    async def go():
        await h.start()
        r = rows(store)
        args = {"purpose": "p", "skills": ["slides"]}
        if with_model_seed:
            args["seed_row_ids"] = [r["u2"], r["trig"]]
        child, text = await h.clone_once("o", {"id": "t", "name": "clone", "input": args}, retry_left=True)
        audit = h.seeder_log[0]
        assert "TimeoutError" in audit["error"]
        seed = [x["row_id"] for x in store.view(child)] if child else audit["selected"]
        if with_model_seed:
            assert audit["fallback"] == "model_seed" and seed == [r["u2"], r["trig"]]
        else:
            assert audit["fallback"] == "all_candidates"
            assert seed == [r["asst0"], r["u1"], r["a1"], r["u2"], r["a2"], r["trig"]]
        await h.machines.stop()

    asyncio.run(go())


def test_chunking_respects_budget_and_single_batch_when_it_fits():
    cands = [{"row_id": i, "role": "user" if i % 2 else "assistant", "text": f"message {i} " * 30} for i in range(1, 21)]
    state, batches, truncated = sd.reduced_batches("build slides", cands, "make slides", budget=100000)
    assert len(batches) == 1 and len(batches[0]) == 20 and not truncated
    state, batches, truncated = sd.reduced_batches("build slides", cands, "make slides", budget=600)
    assert len(batches) > 1 and sum(len(b) for b in batches) == 20
    crit = sd.est_tokens({"true": sd.NOUL_TRUE, "false": sd.NOUL_FALSE})
    for b in batches:
        assert sd.est_tokens(state) + sum(sd.est_tokens(q) + crit for q in b.values()) <= 600
    big = cands[:1] + [{"row_id": 99, "role": "user", "text": "x" * 20000}]
    _, batches, truncated = sd.reduced_batches("p", big, "t", budget=600)
    assert truncated == [99] and all(len(q["message"]) < 1500 for b in batches for q in b.values())


def test_chunked_scores_merge_across_parallel_requests(monkeypatch):
    fake = FakeJev()
    cands = [{"row_id": i, "role": "user", "text": ("Harbor " if i == 3 else "chat ") * 60} for i in range(1, 9)]
    d = asyncio.run(seeder(fake, budget=500).score("p", cands, "t"))
    assert d.requests == len(fake.requests) > 1 and set(d.scores) == set(range(1, 9)) and d.scores[3] == 0.9
    assert d.input_tokens == sum(100 * len(q) for _, q, _ in fake.requests)


def test_seeder_env(monkeypatch):
    monkeypatch.delenv("GRAFT_SEEDER", raising=False)
    assert sd.seeder_name() == "model" and sd.make_seeder() is None
    monkeypatch.setenv("GRAFT_SEEDER", "jev")
    assert isinstance(sd.make_seeder(), sd.JevSeeder)
    monkeypatch.setenv("GRAFT_SEEDER", "bogus")
    with pytest.raises(ValueError):
        sd.seeder_name()
    monkeypatch.delenv("GRAFT_SEED_THRESHOLD", raising=False)
    assert sd.seed_threshold() == 0.4
