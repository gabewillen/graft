"""Benchmark runner: scenarios (experiments/scenarios.py) x arms (graft, subagent).

Every run gets its own provider instance (so usage is per run), its own store and World, and a
unique conversation nonce (--cache-scope run). Metrics per run: pass, scored checks, usage by
role, effective total, $ cost (when prices are set), calls, wall-clock latency, clones or
delegations, output spent on context passing, compactions.

Context passing: graft = output of the original's calls that emitted `clone` (the default
join makes no call); subagent = output of the parent calls that wrote handoffs plus the
subagents' final summary calls. Compaction calls are reported separately in both arms.

Usage:
  uv run python experiments/bench.py --scenario all --arm both -n 10 --parallel 6 [--fake]
      [--compact-tokens N] [--out results.json]
Prices: GRAFT_PRICE_IN_PER_M / _CACHED_IN_PER_M / _CACHE_WRITE_PER_M / _OUT_PER_M (graft/cachestats.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isolation import fisher_two_sided, permutation_p  # noqa: E402
from scenarios import SCENARIOS, RunContext, Scenario, fact_survival  # noqa: E402

from graft import cachestats, callsites  # noqa: E402
from graft.db import Store  # noqa: E402
from graft.fake import bench_graft_policy, bench_subagent_policy  # noqa: E402
from graft.harness import Harness  # noqa: E402
from graft.providers import FakeProvider, add_usage, empty_usage, make_provider  # noqa: E402
from graft.sim import run_script  # noqa: E402
from graft.skills import install  # noqa: E402
from graft.subagent import SubagentRunner  # noqa: E402

ROLES = ("parent/original", "handoff/clone-tx", "subagent/clone", "summary", "compaction")


def role_of(arm: str, tag: str) -> str:
    if tag.endswith(":compaction"):
        return "compaction"
    if arm == "graft":
        if tag == "original":
            return "parent/original"
        if tag.endswith(":clone_tx"):
            return "handoff/clone-tx"
        if tag.startswith("clone:"):
            return "subagent/clone"
        return tag
    return {"parent": "parent/original", "handoff": "handoff/clone-tx", "subagent": "subagent/clone",
            "summary": "summary"}.get(tag, tag)


ARMS = ("graft", "subagent", "subagent_batched")


async def run_one(sc: Scenario, arm: str, i: int, fake: bool, compact: int | None, nonce: str | None,
                  compaction_mode: str = "turns") -> dict:
    """arm: graft | subagent | subagent_batched. The graft record's arm is labelled
    'graft' (turns compaction) or 'graft_summary' (summary ablation)."""
    provider = FakeProvider(bench_graft_policy if arm == "graft" else bench_subagent_policy) \
        if fake else make_provider()
    world = sc.world()
    label = arm if arm != "graft" or compaction_mode == "turns" else f"graft_{compaction_mode}"
    rec: dict = {"scenario": sc.name, "arm": label, "run": i}
    t0 = time.perf_counter()
    agent = None
    transcript: list = []
    try:
        if arm == "graft":
            store_path = os.path.join(tempfile.mkdtemp(prefix=f"bench-{sc.name}-"), "m.db")
            rec["store"] = store_path  # kept on disk for post-hoc diagnosis
            store = Store(store_path)
            install(store)
            agent = Harness(store, provider, world=world, nonce=nonce, compact_tokens=compact,
                            compaction_mode=compaction_mode)
            await agent.start()
            transcript = await run_script(agent, sc.script, echo=False)
        else:
            store = Store(":memory:")
            install(store)
            agent = SubagentRunner(store, provider, world=world, nonce=nonce, compact_tokens=compact,
                                   batched=arm == "subagent_batched")
            transcript = await run_script(agent, sc.script, echo=False)
        rec["error"] = None
    except Exception as e:  # a crashed run is a failed run; keep going
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["latency_s"] = time.perf_counter() - t0
    ctx = RunContext(arm, world, transcript, agent)
    try:
        ok, checks, diag = sc.score(ctx)
    except Exception as e:
        ok, checks, diag = False, {"scorer_error": False}, {"scorer_error": f"{type(e).__name__}: {e}"}
    rec.update(ok=bool(ok) and rec["error"] is None, checks=checks, diag=diag)
    roles: dict[str, dict] = {}
    for tag, u in provider.by_tag.items():
        if u["calls"]:
            add_usage(roles.setdefault(role_of(arm, tag), empty_usage()), u)
    total = empty_usage()
    for u in roles.values():
        add_usage(total, u)
    rec["roles"], rec["total"] = roles, total
    rec["by_model"] = {m: u for m, u in provider.by_model.items() if u["calls"]}
    # per-call-site settings actually used, and $ per role (n/a if a model has no known price)
    pricing = cachestats.Pricing.from_env()
    sites: Counter = Counter()
    role_cost: dict[str, float | None] = {}
    for c in provider.call_log:
        role = role_of(arm, c["tag"])
        sites[f"{role} | site={c.get('site')} model={c.get('model')} effort={c.get('effort')}"] += 1
        cc = pricing.cost_model(c.get("model"), c)
        role_cost[role] = None if cc is None or (role in role_cost and role_cost[role] is None) \
            else role_cost.get(role, 0.0) + cc
    rec["call_sites"], rec["role_cost"] = dict(sites), role_cost
    if arm == "graft" and agent is not None:
        rec["handoffs"] = agent.store.con.execute(
            "SELECT count(*) FROM timelines WHERE source_id IS NOT NULL").fetchone()[0]
        rec["joins"] = [tuple(r) for r in agent.store.con.execute(
            "SELECT timeline_id, join_reason FROM timelines WHERE source_id IS NOT NULL ORDER BY created_at")]
        rec["silent_done_refusals"] = len(agent.silent_done_refusals)
        rec["context_out"] = roles.get("handoff/clone-tx", {}).get("output", 0)
        await agent.machines.stop()
    elif agent is not None:
        rec["handoffs"] = len(agent.delegations)
        rec["delegations"] = [(d.skill, len(d.task_prompt), d.steps) for d in agent.delegations]
        rec["context_out"] = sum(roles.get(k, {}).get("output", 0) for k in ("handoff/clone-tx", "summary"))
    else:
        rec["handoffs"], rec["context_out"] = 0, 0
    rec["compactions"] = len(getattr(agent, "compactions", []) or [])
    log = list(getattr(agent, "compaction_log", []) or [])
    rec["compaction"] = {
        "ratios": [e["before_tokens"] / max(1, e["after_tokens"]) for e in log],
        "before": [e["before_tokens"] for e in log], "after": [e["after_tokens"] for e in log],
        "kept": [len(e.get("kept_ids", [])) for e in log],
        "compressed_msgs": [len(e.get("windows", [])) if "windows" in e else 1 for e in log],
        "failures": len(getattr(agent, "compaction_failures", []) or []),
        "output_tokens": [e.get("output_tokens", 0) for e in log],
        "calls": [e.get("calls", 0) for e in log],
        "kept_head": [e.get("kept_head") for e in log if "kept_head" in e],
        "prefix_rows": [len(e.get("prefix", [])) for e in log],
        "after_cached": [e["first_call_after"]["cached"] for e in log if e.get("first_call_after")],
        "after_input": [e["first_call_after"]["input"] for e in log if e.get("first_call_after")],
        "failed_output_tokens": sum(f["output_tokens"] for f in getattr(agent, "compaction_failure_calls", []) or [])}
    if sc.survival_facts:
        rec["fact_survival"] = fact_survival(log, sc.survival_facts)
    rec["transcript"] = transcript
    rec["decks"] = world.decks
    rec["compaction_log"] = [{k: v for k, v in e.items() if k not in ("prefix", "prefix_texts")} for e in log]
    return rec


def welch(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Welch's t-test: (t, df, two-sided p). p from the Student t CDF via the regularized
    incomplete beta function (no scipy)."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    m1, m2 = statistics.mean(x), statistics.mean(y)
    v1, v2 = statistics.variance(x), statistics.variance(y)
    se2 = v1 / n1 + v2 / n2
    if se2 == 0:
        return float("nan"), float("nan"), float("nan")
    t = (m1 - m2) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = _betainc(df / 2, 0.5, df / (df + t * t))
    return t, df, p


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
        de = d * c
        h *= de
        if abs(de - 1) < 3e-14:
            break
    return h


def msd(v: list[float], fmt: str = "{:.0f}") -> str:
    if not v:
        return "n/a"
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return f"{fmt.format(statistics.mean(v))} ± {fmt.format(sd)}"


def run_cost(pricing: cachestats.Pricing, r: dict) -> float | None:
    """$ for one run, priced per model (n/a if any model has no known price)."""
    return pricing.cost_by_model(r["by_model"]) if r.get("by_model") else pricing.cost(r["total"])


def report(recs: list[dict], pricing: cachestats.Pricing, scenarios: list[str], arms: list[str]) -> None:
    print("\n================ BENCH SUMMARY ================")
    print(pricing.assumptions())
    head = ("scenario", "arm", "pass", "eff. total", "$ / run", "latency s", "calls", "handoffs",
            "ctx-pass out", "compactions")
    rows = []
    for sc in scenarios:
        for arm in arms:
            rs = [r for r in recs if r["scenario"] == sc and r["arm"] == arm]
            if not rs:
                continue
            costs = [run_cost(pricing, r) for r in rs]
            rows.append((sc, arm, f"{sum(r['ok'] for r in rs)}/{len(rs)}",
                         msd([pricing.effective_total(r["total"]) for r in rs]),
                         msd(costs, "{:.5f}") if None not in costs else "n/a",
                         msd([r["latency_s"] for r in rs], "{:.1f}"),
                         msd([r["total"]["calls"] for r in rs], "{:.1f}"),
                         msd([r["handoffs"] for r in rs], "{:.1f}"),
                         msd([r["context_out"] for r in rs]),
                         msd([r["compactions"] for r in rs], "{:.1f}")))
    w = [max(len(str(x[i])) for x in [head] + rows) for i in range(len(head))]
    for r in [head] + rows:
        print("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))

    print("\nper-role means (calls / effective total) by scenario and arm:")
    for sc in scenarios:
        for arm in arms:
            rs = [r for r in recs if r["scenario"] == sc and r["arm"] == arm]
            if not rs:
                continue
            parts = []
            for role in ROLES:
                us = [r["roles"].get(role, empty_usage()) for r in rs]
                if any(u["calls"] for u in us):
                    parts.append(f"{role} {statistics.mean(u['calls'] for u in us):.1f}/"
                                 f"{statistics.mean(pricing.effective_total(u) for u in us):.0f}")
            hit = statistics.mean(cachestats.hit_rate(r["total"]) for r in rs)
            print(f"  {sc:<16} {arm:<9} hit rate {hit:.0%}; " + "; ".join(parts))
            costs = [run_cost(pricing, r) for r in rs]
            cc = [r.get("role_cost", {}).get("handoff/clone-tx") for r in rs]
            if None not in costs and cc and None not in cc and sum(costs):
                print(f"      clone-call / handoff $ share {sum(cc) / sum(costs):.1%}")
            sites: Counter = Counter()
            for r in rs:
                sites.update(r.get("call_sites", {}))
            for k, n in sorted(sites.items()):
                print(f"      calls: {k}: {n}")

    pairs = [(a, b) for a in ("graft", "graft_summary") for b in ("subagent", "subagent_batched")
             if a in arms and b in arms] + [("graft", b) for b in ("graft_summary",)
                                            if {"graft", b} <= set(arms)]
    print("\nsignificance (effective total and pass; two-sided):")
    for a_arm, b_arm in pairs:
        for sc in scenarios:
            m = [r for r in recs if r["scenario"] == sc and r["arm"] == a_arm]
            s = [r for r in recs if r["scenario"] == sc and r["arm"] == b_arm]
            if not m or not s:
                continue
            em = [pricing.effective_total(r["total"]) for r in m]
            es = [pricing.effective_total(r["total"]) for r in s]
            t, df, p = welch(em, es)
            pp = permutation_p(em, es)
            pm, ps = sum(r["ok"] for r in m), sum(r["ok"] for r in s)
            pf = fisher_two_sided(pm, len(m), ps, len(s))
            ratio = statistics.mean(em) / statistics.mean(es) if statistics.mean(es) else float("nan")
            print(f"  {a_arm} vs {b_arm:<16} {sc:<16} cost ratio {ratio:.2f}; Welch t={t:.2f} df={df:.1f} "
                  f"p={p:.4f}; permutation p={pp:.2g}; pass {pm}/{len(m)} vs {ps}/{len(s)} Fisher p={pf:.4f}")

    comp = [r for r in recs if r.get("compaction", {}).get("ratios")]
    if comp:
        print("\ncompaction (runs that compacted): compression ratio = working-view tokens before/after")
        for sc in scenarios:
            for arm in arms:
                rs = [r for r in recs if r["scenario"] == sc and r["arm"] == arm]
                cs = [r for r in rs if r["compaction"]["ratios"]]
                if not cs:
                    continue
                ratios = [x for r in cs for x in r["compaction"]["ratios"]]
                line = (f"  {sc:<16} {arm:<17} runs compacted {len(cs)}/{len(rs)}; events {len(ratios)}; "
                        f"ratio {statistics.mean(ratios):.2f} (min {min(ratios):.2f}, max {max(ratios):.2f}); "
                        f"kept rows/event {statistics.mean(k for r in cs for k in r['compaction']['kept']):.1f}; "
                        f"compressed msgs/event {statistics.mean(k for r in cs for k in r['compaction']['compressed_msgs']):.1f}; "
                        f"refused-twice {sum(r['compaction']['failures'] for r in rs)}")
                print(line)
                c = [r["compaction"] for r in cs]
                out = [x for k in c for x in k.get("output_tokens", [])]
                head = [x for k in c for x in k.get("kept_head", [])]
                pre = [x for k in c for x in k.get("prefix_rows", [])]
                ac, ai = [x for k in c for x in k.get("after_cached", [])], [x for k in c for x in k.get("after_input", [])]
                if out:
                    print(f"      compaction output tokens/event {statistics.mean(out):.0f}; kept head (rows before first "
                          f"window) {statistics.mean(head):.1f} of {statistics.mean(pre):.1f} prefix rows; first call "
                          f"after compaction: cached {sum(ac)}/{sum(ai)} input tokens ({sum(ac) / max(1, sum(ai)):.0%}); "
                          f"refused-twice output tokens {sum(k.get('failed_output_tokens', 0) for k in c)}")
                surv = Counter(v for r in rs for v in (r.get("fact_survival") or {}).values())
                if surv:
                    per = {}
                    for r in rs:
                        for f, v in (r.get("fact_survival") or {}).items():
                            per.setdefault(f, Counter())[v] += 1
                    print(f"      fact survival (fact x run): {dict(surv)}")
                    print("      per fact: " + "; ".join(f"{f}: " + ",".join(f"{k}={n}" for k, n in c.most_common())
                                                       for f, c in per.items()))

    print("\nfailure modes (failed checks and errors per scenario/arm):")
    for sc in scenarios:
        for arm in arms:
            rs = [r for r in recs if r["scenario"] == sc and r["arm"] == arm]
            if not rs:
                continue
            fails = Counter(k for r in rs for k, v in r["checks"].items() if not v)
            errs = Counter(r["error"].split(":")[0] for r in rs if r["error"])
            extra = ""
            if arm.startswith("graft"):
                joins = Counter(j[1] for r in rs for j in r.get("joins", []))
                extra = (f"; joins {dict(joins)}; silent-done refusals "
                         f"{sum(r.get('silent_done_refusals', 0) for r in rs)}")
            print(f"  {sc:<16} {arm:<9} failed checks {dict(fails) or '{}'}; errors {dict(errs) or '{}'}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", help=f"one of {list(SCENARIOS)} or 'all' (comma list ok)")
    ap.add_argument("--arm", default="both",
                    help="comma list of graft,subagent,subagent_batched; 'both' = graft,subagent; 'all' = all three")
    ap.add_argument("--compaction", choices=("turns", "summary"), default="turns",
                    help="graft compaction: turns (kept rows + compressed windows) or summary (ablation)")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--cache-scope", choices=("run", "global"), default="run")
    ap.add_argument("--compact-tokens", type=int, default=None,
                    help="override every scenario's compaction threshold (estimated tokens); 0 = off")
    ap.add_argument("--out", default=None, help="write per-run records as JSON")
    ap.add_argument("--effort", action="append", default=[], metavar="SITE=LEVEL",
                    help="per-call-site effort, e.g. clone_call=high (sets GRAFT_EFFORT_<SITE>; see graft/callsites.py)")
    ap.add_argument("--model", action="append", default=[], metavar="SITE=MODEL",
                    help="per-call-site model, e.g. clone_call=gpt-6-luna (sets GRAFT_MODEL_<SITE>)")
    a = ap.parse_args()
    for what, pairs in (("EFFORT", a.effort), ("MODEL", a.model)):
        for kv in pairs:
            site, _, val = kv.partition("=")
            if site not in callsites.SITES or not val:
                raise SystemExit(f"--{what.lower()} expects SITE=VALUE with SITE in {callsites.SITES}")
            os.environ[f"GRAFT_{what}_{site.upper()}"] = val
    names = list(SCENARIOS) if a.scenario == "all" else a.scenario.split(",")
    arms = {"both": ["graft", "subagent"], "all": list(ARMS)}.get(a.arm) or a.arm.split(",")
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; use {ARMS}")
    jobs = [(SCENARIOS[n], arm, i) for n in names for i in range(a.n) for arm in arms]

    def job(sc: Scenario, arm: str, i: int) -> dict:
        compact = sc.compact_tokens if a.compact_tokens is None else (a.compact_tokens or None)
        nonce = uuid.uuid4().hex[:16] if a.cache_scope == "run" else None
        return asyncio.run(run_one(sc, arm, i, a.fake, compact, nonce, a.compaction))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        recs = list(ex.map(lambda j: job(*j), jobs))
    pricing = cachestats.Pricing.from_env()
    print(f"bench: scenarios={names} arms={arms} n={a.n} parallel={a.parallel} fake={a.fake} "
          f"cache_scope={a.cache_scope} wall={time.perf_counter() - t0:.0f}s")
    for r in recs:
        if r["error"]:
            print(f"  ERROR {r['scenario']}/{r['arm']}#{r['run']}: {r['error'][:200]}")
    labels = [x if x != "graft" or a.compaction == "turns" else f"graft_{a.compaction}" for x in arms]
    report(recs, pricing, names, labels)
    spend = [run_cost(pricing, r) for r in recs]
    if None not in spend:
        print(f"\ntotal spend this invocation: ${sum(spend):.4f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(recs, f, indent=1, default=str)
        print(f"records: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
