"""Seed-selection replay (experiments/seed_eval.py) with fake decision clients (offline)."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import seed_eval as se  # noqa: E402


def event():
    cands = [{"row_id": 1, "role": "user", "text": "I need a website.", "tokens": 4},
             {"row_id": 2, "role": "assistant", "text": "Sure, what details?", "tokens": 5},
             {"row_id": 3, "role": "user", "text": "We are Harbor Light Dental in Duluth.", "tokens": 9},
             {"row_id": 4, "role": "user", "text": "Phone 218-555-0143. Dr. Okafor.", "tokens": 8},
             {"row_id": 5, "role": "user", "text": "Make slides instead.", "tokens": 5}]
    return se.Event("s.db", "build slides", 5, cands,
                    {"practice name": {3}, "city": {3}, "phone": {4}, "Dr. Okafor": {4}}, {1, 5})


def test_metrics_and_selection():
    e = event()
    m = se.metrics([e], [e.model_seed])
    assert m["fact_recall"] == 0 and m["full_recall_events"] == 0 and m["seed_rows"] == 2
    m = se.metrics([e], [{3, 4}])  # the trigger is always added
    assert m["fact_recall"] == 1 and m["full_recall_events"] == 1 and m["seed_rows"] == 3 and m["extra_rows"] == 0
    scores = {1: 0.2, 2: 0.1, 3: 0.9, 4: 0.6, 5: 0.95}
    assert se.select_threshold(e, scores, 0.5) == {3, 4, 5}
    assert se.select_topk(e, scores, 2) == {5, 3}


class FakeJev:
    def __init__(self):
        self.requests = []

    async def system_one(self, state, questions, model=None):
        self.requests.append((state, questions, model))
        nouls = {k: NS(noul=0.9 if any(w in json.dumps(q.instructions) for w in ("Harbor", "218")) else 0.1)
                 for k, q in questions.items()}
        return NS(nouls=nouls, usage=NS(input_tokens=1000, output_tokens=5), model="jev-1.13.0")


def test_jev_scorer_one_noul_per_candidate_in_one_request():
    e = event()
    fj = FakeJev()
    scores, meta = asyncio.run(se.JevScorer(fj).score(e))
    state, questions, model = fj.requests[0]
    assert len(fj.requests) == 1 and model == se.JEV_MODEL
    assert set(questions) == {f"row_{c['row_id']}" for c in e.candidates}
    assert all(q.type == "noul" for q in questions.values())
    assert state["new_task"] == "build slides" and len(state["conversation"]) == 5
    assert scores[3] == 0.9 and scores[2] == 0.1
    assert abs(meta["cost"] - 1000 * 0.042 / 1e6) < 1e-12


class FakeResponses:
    async def create(self, **kw):
        assert kw["text"]["format"]["type"] == "json_schema" and kw["model"] == "gpt-6-luna"
        rows = [{"row_id": 3, "needed": True, "confidence": 0.8}, {"row_id": 2, "needed": False, "confidence": 0.9}]
        return NS(output_text=json.dumps({"rows": rows}),
                  usage=NS(input_tokens=500, output_tokens=50, input_tokens_details=NS(cached_tokens=100)))


def test_luna_scorer_maps_needed_confidence_and_fills_missing():
    e = event()
    scores, meta = asyncio.run(se.LunaScorer(NS(responses=FakeResponses())).score(e))
    assert scores[3] == 0.8 and abs(scores[2] - 0.1) < 1e-9 and scores[4] == 0.0  # omitted row -> 0
    assert abs(meta["cost"] - (400 * 0.10 + 100 * 0.01 + 50 * 0.50) / 1e6) < 1e-12


def test_run_scorer_caches_and_records_errors():
    class Boom:
        name = "boom"

        async def score(self, e):
            raise RuntimeError("nope")

    cache = {}
    out = asyncio.run(se.run_scorer(Boom(), [event()], 2, cache))
    assert out[0]["error"].startswith("RuntimeError") and "boom|s.db" in cache


def test_needle_prompt_truncates_deterministically():
    e = event()
    text, cut = se.needle_prompt(e, 2)
    assert not cut and "CURRENT message: (user) We are Harbor Light Dental" in text and "Previous message" in text
    e.candidates[2]["text"] = "x" * 5000
    text2, cut2 = se.needle_prompt(e, 2)
    assert cut2 and len(text2) <= se.NEEDLE_WINDOW_CHARS and text2 == se.needle_prompt(e, 2)[0]


def test_needle_retest_text_and_decision_schema():
    e = event()
    t1, _ = se.needle_row_text(e, 2, with_context=False)
    t2, _ = se.needle_row_text(e, 2, with_context=True)
    assert "This message: (user) We are Harbor Light Dental" in t1 and "New task" not in t1
    assert t2.startswith("New task: build slides") and "Latest user request: Make slides instead." in t2
    m = se._decision_model(default=True)
    assert m().decision == "skip" and m(decision="keep").decision == "keep"
    assert "decision" in se._decision_model(default=False).model_fields
    assert [t["name"] for t in se.NEEDLE_PAIR] == ["keep_message", "skip_message"]
