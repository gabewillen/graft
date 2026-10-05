"""Offline replay of the original's seed selection (the "cloner actor") over stored
long_compaction runs. Evaluation only: harness behaviour is unchanged.

Event: in each stored run, the slides clone that built the deck (seeded by the original after
the website clone joined). Candidates = text rows in the original's working view before that clone
was created. Ground truth: for each scored fact (diagnose_long.SCORED), the candidate rows that carry
it. A fact is covered when at least one of its carrier rows is seeded.

Strategies (each also always seeds the trigger, the user's latest request):
  model       what the run actually seeded (baseline)
  jev         TypeSafe Jev (system_one): one Noul per candidate row, all in one request per event;
              score = P(yes); threshold sweep
  luna_actor  gpt-6-luna (Responses, structured output): one call per event returning
              {row_id, needed, confidence} for every candidate; score = confidence if needed else
              1 - confidence; threshold sweep
  embeddings  OpenAI text-embedding-3-small cosine(query = purpose + trigger, row); top-k and threshold
  model+jev, model+luna_actor   union of the run's seed with the strategy's selection

Prices (verified official pages, 2026-10-04): Jev jev-1.13.0 input $0.042 / 1M tokens, output free
(docs.typesafe.ai/models); gpt-6-luna $0.10 in / $0.01 cached / $0.50 out per 1M;
text-embedding-3-small $0.02 per 1M (developers.openai.com/api/docs/pricing).

Usage:
  uv run python experiments/seed_eval.py --json a.json [--json b.json] [--stores list.txt] \\
      [--strategies jev,luna_actor,embeddings] [--cache scores.json] [--concurrency 8]
Keys: TYPESAFE_API_KEY and OPENAI_API_KEY from the environment (never printed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnose_long import SCORED  # noqa: E402

from graft.db import Store  # noqa: E402

PRICE = {"jev_in": 0.042, "luna_in": 0.10, "luna_cached": 0.01, "luna_out": 0.50, "emb": 0.02}  # $ / 1M tokens
JEV_MODEL = "jev-1.13.0"
EMB_MODEL = "text-embedding-3-small"
LUNA_MODEL = "gpt-6-luna"
THRESHOLDS = [round(x, 2) for x in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
TOPK = (3, 5, 8, 12)


@dataclass
class Event:
    store: str
    purpose: str
    trigger: int
    candidates: list[dict]  # {row_id, role, text, tokens}
    facts: dict[str, set[int]]  # fact -> candidate rows carrying it (only facts present)
    model_seed: set[int]
    failed_run: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def needed_rows(self) -> set[int]:
        return set().union(*self.facts.values()) if self.facts else set()


# ------------------------------------------------------------------ events
def load_event(path: str) -> Event | None:
    s = Store(path)
    clones = [r[0] for r in s.con.execute("SELECT timeline_id FROM timelines WHERE source_id IS NOT NULL "
                                          "ORDER BY created_at")]
    slides = [c for c in clones if "slides" in {k["name"] for k in s.skills_for(c)}]
    web = [c for c in clones if "website" in {k["name"] for k in s.skills_for(c)}]
    if not slides or not web:
        return None
    sl = slides[-1]
    own = [r["row_id"] for r in s.view(sl) if r["owner_timeline"] == sl]
    first = min(own) if own else 10 ** 9
    o_wv = [r for r in s.working_view("o") if r["row_id"] < first]
    cands = [{"row_id": r["row_id"], "role": r["role"], "text": r["content"], "tokens": max(1, len(r["content"]) // 4)}
             for r in o_wv if r["kind"] == "text"]
    if not cands:
        return None
    seed = {r["row_id"] for r in s.view(sl) if r["owner_timeline"] != sl}
    tx = s.con.execute("SELECT scratch FROM transactions WHERE kind='clone' AND target=? AND status='committed'",
                       (sl,)).fetchone()
    purpose = ""
    if tx and tx[0]:
        try:
            purpose = str(json.loads(tx[0]).get("args", {}).get("purpose", ""))
        except (ValueError, AttributeError):
            purpose = ""
    users = [c for c in cands if c["role"] == "user" and not c["text"].startswith("(harness note")]
    trigger = users[-1]["row_id"] if users else cands[-1]["row_id"]
    facts = {}
    for f, pat in SCORED.items():
        rows = {c["row_id"] for c in cands if not c["text"].startswith("(harness note")
                and re.search(pat, c["text"], re.I)}
        if rows:
            facts[f] = rows
    deck = " ".join(r["content"] for r in s.view(sl) if (r["tool_name"] or "").startswith("slides_"))
    failed = any(not re.search(SCORED[f], deck, re.I) for f in ("practice name", "city", "Dr. Okafor",
                                                                  "Dr. Brandt", "phone"))
    return Event(path, purpose or "build a slide show", trigger, cands, facts, seed & {c["row_id"] for c in cands},
                 failed_run=failed)


def load_events(paths: list[str]) -> list[Event]:
    out, seen = [], set()
    for p in paths:
        p = os.path.realpath(p)
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        e = load_event(p)
        if e:
            out.append(e)
    return out


# ------------------------------------------------------------------ metrics
def metrics(events: list[Event], selections: list[set[int]]) -> dict:
    fact_tot = fact_cov = row_tot = row_cov = full = 0
    sizes, toks, extra = [], [], []
    for e, sel in zip(events, selections):
        sel = set(sel) | {e.trigger}
        for rows in e.facts.values():
            fact_tot += 1
            fact_cov += bool(rows & sel)
        need = e.needed_rows
        row_tot += len(need)
        row_cov += len(need & sel)
        full += all(rows & sel for rows in e.facts.values())
        sizes.append(len(sel))
        toks.append(sum(c["tokens"] for c in e.candidates if c["row_id"] in sel))
        extra.append(len(sel - need - {e.trigger}))
    n = len(events)
    return {"events": n, "fact_recall": fact_cov / max(1, fact_tot), "row_recall": row_cov / max(1, row_tot),
            "full_recall_events": full, "full_recall_rate": full / max(1, n),
            "seed_rows": statistics.mean(sizes) if sizes else 0, "seed_tokens": statistics.mean(toks) if toks else 0,
            "extra_rows": statistics.mean(extra) if extra else 0}


def select_threshold(e: Event, scores: dict[int, float], t: float) -> set[int]:
    return {rid for rid, p in scores.items() if p >= t}


def select_topk(e: Event, scores: dict[int, float], k: int) -> set[int]:
    return {rid for rid, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]}


# ------------------------------------------------------------------ scorers
NOUL_TRUE = ("The clone needs this message: it carries facts, names, numbers, requirements or "
             "preferences the new task should use, or it is the request for the task.")
NOUL_FALSE = "The clone can do the new task well without this message."


def jev_request(e: Event) -> tuple[dict, dict]:
    """(state, questions) for one event: one Noul per candidate row."""
    import typesafe_sdk

    state = {"new_task": e.purpose, "latest_user_request": next(c["text"] for c in e.candidates
                                                                 if c["row_id"] == e.trigger),
             "conversation": [{"row_id": c["row_id"], "role": c["role"], "text": c["text"][:2000]}
                              for c in e.candidates]}
    questions = {f"row_{c['row_id']}": typesafe_sdk.Noul(
        instructions={"question": "Does a clone doing the new task need this message from the conversation?",
                      "row_id": c["row_id"], "message": c["text"][:600]},
        criteria={"true": NOUL_TRUE, "false": NOUL_FALSE}) for c in e.candidates}
    return state, questions


class JevScorer:
    name = "jev"

    def __init__(self, client=None):
        self.client = client  # an open typesafe_sdk.AsyncTypeSafeClient, or a fake with .system_one

    async def score(self, e: Event) -> tuple[dict[int, float], dict]:
        state, questions = jev_request(e)
        t0 = time.perf_counter()
        resp = await self.client.system_one(state=state, questions=questions, model=JEV_MODEL)
        lat = time.perf_counter() - t0
        scores = {int(k.split("_", 1)[1]): float(v.noul) for k, v in resp.nouls.items()}
        tin = getattr(resp.usage, "input_tokens", None)
        return scores, {"latency": lat, "calls": 1, "questions": len(questions), "input_tokens": tin,
                        "output_tokens": getattr(resp.usage, "output_tokens", None),
                        "cost": None if tin is None else tin * PRICE["jev_in"] / 1e6, "model": resp.model}


LUNA_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["rows"], "properties": {
    "rows": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                        "required": ["row_id", "needed", "confidence"],
                                        "properties": {"row_id": {"type": "integer"}, "needed": {"type": "boolean"},
                                                       "confidence": {"type": "number"}}}}}}
LUNA_SYSTEM = ("You choose which messages of a conversation a focused clone needs for its new task. The "
               "clone sees only the messages you choose. It needs every message that carries facts, names, "
               "numbers, requirements or preferences the task should use, and the request for the task. "
               "Return one entry per row_id with needed and your confidence (0 to 1) in that judgment.")


class LunaScorer:
    name = "luna_actor"

    def __init__(self, client=None, effort: str = "low"):
        self.client, self.effort = client, effort

    async def score(self, e: Event) -> tuple[dict[int, float], dict]:
        payload = {"new_task": e.purpose, "conversation": [{"row_id": c["row_id"], "role": c["role"],
                                                            "text": c["text"][:2000]} for c in e.candidates]}
        t0 = time.perf_counter()
        resp = await self.client.responses.create(
            model=LUNA_MODEL, instructions=LUNA_SYSTEM, input=json.dumps(payload), store=False,
            reasoning={"effort": self.effort}, max_output_tokens=8000,
            text={"format": {"type": "json_schema", "name": "seed_selection", "schema": LUNA_SCHEMA, "strict": True}})
        lat = time.perf_counter() - t0
        rows = json.loads(resp.output_text).get("rows", [])
        valid = {c["row_id"] for c in e.candidates}
        scores = {}
        for r in rows:
            rid = int(r["row_id"])
            if rid in valid:
                c = min(1.0, max(0.0, float(r["confidence"])))
                scores[rid] = c if r["needed"] else 1 - c
        for rid in valid - set(scores):
            scores[rid] = 0.0
        u = resp.usage
        cached = u.input_tokens_details.cached_tokens if u and u.input_tokens_details else 0
        cost = None if u is None else ((u.input_tokens - cached) * PRICE["luna_in"] + cached * PRICE["luna_cached"]
                                       + u.output_tokens * PRICE["luna_out"]) / 1e6
        return scores, {"latency": lat, "calls": 1, "input_tokens": u.input_tokens if u else None,
                        "output_tokens": u.output_tokens if u else None, "cost": cost, "model": LUNA_MODEL,
                        "missing_rows": len(valid) - len(set(scores) & {int(r['row_id']) for r in rows})}


class EmbeddingScorer:
    name = "embeddings"

    def __init__(self, client=None):
        self.client = client

    async def score(self, e: Event) -> tuple[dict[int, float], dict]:
        trig = next(c["text"] for c in e.candidates if c["row_id"] == e.trigger)
        texts = [f"{e.purpose}\n{trig}"] + [c["text"][:4000] for c in e.candidates]
        t0 = time.perf_counter()
        resp = await self.client.embeddings.create(model=EMB_MODEL, input=texts)
        lat = time.perf_counter() - t0
        vecs = [d.embedding for d in resp.data]
        q = vecs[0]

        def cos(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

        scores = {c["row_id"]: cos(q, v) for c, v in zip(e.candidates, vecs[1:])}
        tin = resp.usage.prompt_tokens if resp.usage else None
        return scores, {"latency": lat, "calls": 1, "input_tokens": tin,
                        "cost": None if tin is None else tin * PRICE["emb"] / 1e6, "model": EMB_MODEL}


async def run_scorer(scorer, events: list[Event], concurrency: int, cache: dict) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = [None] * len(events)  # type: ignore[list-item]

    async def one(i: int, e: Event) -> None:
        key = f"{scorer.name}|{e.store}"
        if key in cache:
            out[i] = cache[key]
            return
        async with sem:
            try:
                scores, meta = await scorer.score(e)
                out[i] = {"scores": {str(k): v for k, v in scores.items()}, "meta": meta, "error": None}
            except Exception as ex:  # record and continue
                out[i] = {"scores": {}, "meta": {}, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
            cache[key] = out[i]

    await asyncio.gather(*(one(i, e) for i, e in enumerate(events)))
    return out


# ------------------------------------------------------------------ report
def sweep(events: list[Event], results: list[dict], name: str, model_union: bool = False,
          topk: bool = False) -> list[tuple[str, dict]]:
    rows = []
    params = [("k", k) for k in TOPK] if topk else []
    params += [("t", t) for t in THRESHOLDS]
    if name == "embeddings":
        params = [("k", k) for k in TOPK] + [("t", t) for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)]
    for kind, v in params:
        sels = []
        for e, r in zip(events, results):
            sc = {int(k): x for k, x in r["scores"].items()}
            sel = select_topk(e, sc, v) if kind == "k" else select_threshold(e, sc, v)
            sels.append(sel | e.model_seed if model_union else sel)
        rows.append((f"{kind}={v}", metrics(events, sels)))
    return rows


def best(rows: list[tuple[str, dict]]) -> tuple[str, dict]:
    return max(rows, key=lambda x: (x[1]["full_recall_events"], -x[1]["seed_tokens"]))


def fmt(m: dict) -> str:
    return (f"fact recall {m['fact_recall']:.1%}; row recall {m['row_recall']:.1%}; full-recall events "
            f"{m['full_recall_events']}/{m['events']}; seed {m['seed_rows']:.1f} rows / {m['seed_tokens']:.0f} tok; "
            f"extra rows {m['extra_rows']:.1f}")


def cost_line(results: list[dict]) -> str:
    ok = [r for r in results if not r["error"]]
    if not ok:
        return "no successful calls"
    lat = [r["meta"]["latency"] for r in ok]
    costs = [r["meta"].get("cost") for r in ok]
    tin = [r["meta"].get("input_tokens") or 0 for r in ok]
    c = f"${statistics.mean(costs):.6f}/event" if None not in costs else "$ n/a"
    return (f"latency {statistics.mean(lat):.2f}s mean / {max(lat):.2f}s max per event; input tokens/event "
            f"{statistics.mean(tin):.0f}; {c}; errors {len(results) - len(ok)}")


async def amain(a) -> int:
    paths: list[str] = []
    for j in a.json:
        paths += [r["store"] for r in json.load(open(j)) if r.get("store") and r.get("scenario") == "long_compaction"]
    for lst in a.stores:
        paths += [ln.strip() for ln in open(lst) if ln.strip()]
    events = load_events(paths)
    print(f"events: {len(events)} (from {len(set(map(os.path.realpath, paths)))} stores); runs that failed the deck "
          f"check: {sum(e.failed_run for e in events)}; candidates/event "
          f"{statistics.mean(len(e.candidates) for e in events):.1f}; facts present/event "
          f"{statistics.mean(len(e.facts) for e in events):.1f} of {len(SCORED)}")
    cache = json.load(open(a.cache)) if a.cache and os.path.exists(a.cache) else {}
    strategies = a.strategies.split(",")
    results: dict[str, list[dict]] = {}
    if "jev" in strategies:
        import typesafe_sdk

        async with typesafe_sdk.AsyncTypeSafeClient(model=JEV_MODEL, timeout=60) as tc:
            results["jev"] = await run_scorer(JevScorer(tc), events, a.concurrency, cache)
    if {"luna_actor", "embeddings"} & set(strategies):
        import openai

        oc = openai.AsyncOpenAI()
        if "luna_actor" in strategies:
            results["luna_actor"] = await run_scorer(LunaScorer(oc, a.luna_effort), events, a.concurrency, cache)
        if "embeddings" in strategies:
            results["embeddings"] = await run_scorer(EmbeddingScorer(oc), events, a.concurrency, cache)
    if a.cache:
        json.dump(cache, open(a.cache, "w"))
    report(events, results)
    return 0


def report(events: list[Event], results: dict[str, list[dict]]) -> None:
    print("\n================ SEED SELECTION REPLAY ================")
    print(f"model (what the run seeded): {fmt(metrics(events, [e.model_seed for e in events]))}")
    fails = [e for e in events if e.failed_run]
    if fails:
        print(f"  on the {len(fails)} failed runs: {fmt(metrics(fails, [e.model_seed for e in fails]))}")
    for name, res in results.items():
        print(f"\n{name}: {cost_line(res)}")
        errs = [r["error"] for r in res if r["error"]]
        if errs:
            print(f"  errors e.g. {errs[0]}")
        ok_idx = [i for i, r in enumerate(res) if not r["error"]]
        ev = [events[i] for i in ok_idx]
        rs = [res[i] for i in ok_idx]
        rows = sweep(ev, rs, name, topk=name != "embeddings")
        for label, m in rows:
            print(f"  {label:<7} {fmt(m)}")
        b = best(rows)
        print(f"  BEST {b[0]}: {fmt(b[1])}")
        if name in ("jev", "luna_actor"):
            urows = sweep(ev, rs, name, model_union=True)
            ub = best(urows)
            print(f"  model+{name} BEST {ub[0]}: {fmt(ub[1])}")
            for label, m in urows:
                print(f"    union {label:<7} {fmt(m)}")
            fidx = [i for i, e in enumerate(ev) if e.failed_run]
            if fidx:
                t = float(b[0].split("=")[1]) if b[0].startswith("t=") else None
                if t is not None:
                    fsel = [select_threshold(ev[i], {int(k): x for k, x in rs[i]["scores"].items()}, t) for i in fidx]
                    print(f"  on failed runs at {b[0]}: {fmt(metrics([ev[i] for i in fidx], fsel))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", default=[])
    ap.add_argument("--stores", action="append", default=[], help="file with one store path per line")
    ap.add_argument("--strategies", default="jev,luna_actor,embeddings")
    ap.add_argument("--cache", default=None, help="JSON cache of per-event scores (re-runs skip API calls)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--luna-effort", default="low")
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
