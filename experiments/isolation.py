"""Isolation experiment (spec: Validation Experiment), progressive-disclosure edition.

The same slides task runs N times in each arm (default 20), with arms run concurrently.

  shared: a realistic single agent with progressive disclosure. Its system prompt lists
          the catalog only (name + description). Every capability tool is declared. Its
          history is unrelated earlier work: it called load_skill for website, calendar,
          newsletter, social and recipes, and did real tool work in each skill's house
          style (emoji, exclamation marks, ALL CAPS, #hashtags, slang, closing questions,
          lowercase titles). Then the slides task arrives, and it must load_skill('slides').
  clone:  graft. The clone is seeded only with the task's user row, linked to slides+join,
          and also has to load_skill('slides') itself.

  subagent: the delegation baseline. A parent with the same history (and the persona plus a
          delegate(skill, task_prompt) tool) writes a handoff of the task; a subagent with the
          persona, a slides listing and load_skill sees only that handoff and builds the deck
          (graft/subagent.py). Usage covers both the parent's handoff call and the subagent.

Scoring is deterministic. Compliance measures the slides task itself. Leakage counts
features of the deck that come from the *other* skills' style or format rules, or from
the earlier conversation. Each leakage type is reported separately. Significance:
Fisher's exact test (two-sided) on pass rates and on each leakage rate, and a permutation
test on mean scores.

Usage: uv run python experiments/isolation.py [-n 20] [--parallel 8] [--fake] [--cache-scope run|global]

Prompt caching: --cache-scope run (default) puts a unique nonce at the very start of the first
message of every run, so runs can share only tools + system prompt (as separate real
conversations would); requests inside one run still share their growing history prefix.
--cache-scope global omits the nonce (identical runs; an upper bound, not realistic).
Cache accounting and optional cost: see graft/cachestats.py (GRAFT_PRICE_IN_PER_M,
GRAFT_PRICE_CACHED_IN_PER_M, GRAFT_PRICE_CACHE_WRITE_PER_M, GRAFT_PRICE_OUT_PER_M,
GRAFT_CACHED_READ_MULT, GRAFT_CACHE_WRITE_MULT).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft import cachestats  # noqa: E402
from graft.db import Store, render  # noqa: E402
from graft.harness import Harness  # noqa: E402
from graft.providers import FakeProvider, add_usage, empty_usage, make_provider, prefix_nonce  # noqa: E402
from graft.skills import (BASE_SYSTEM, LOAD_SKILL_TOOL, World, install, load_skill,  # noqa: E402
                            skill_listing)
from graft.subagent import DELEGATE_TOOL, SubagentRunner  # noqa: E402

TASK = ("Make a short slide show for the Lakeview Dental open house. The audience is retirees in "
        "Columbus, Ohio. Cover our three services: cleanings, implants, and whitening, one slide "
        "per service.")
CAPABILITY = ["website", "slides", "calendar", "newsletter", "social", "recipes"]

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✨❤]")
LEAKS = {
    "emoji": lambda d: bool(EMOJI.search(d["text"])),
    "exclamation": lambda d: "!" in d["text"],
    "hashtag": lambda d: bool(re.search(r"(?<![\w&])#[A-Za-z]\w*", d["text"])),
    "all_caps_words": lambda d: bool([w for w in re.findall(r"\b[A-Z]{4,}\b", d["text"])
                                      if w not in {"FAQ", "LAKEVIEW", "OHIO", "DENTAL"}]),
    "slang_casual": lambda d: bool(re.search(
        r"\b(awesome|super|yay|woo+hoo|cheers|gonna|wanna|cool|amazing|y'all|hey|vibes?|"
        r"fun|treats?|yummy|sweet)\b", d["text"], re.I)),
    "question_to_reader": lambda d: any(s.strip().endswith("?") for s in d["strings"]),
    "lowercase_titles": lambda d: any(t == t.lower() and re.search("[a-z]", t) for t in d["titles"]),
    "history_bleed": lambda d: bool(re.search(r"sunny crumbs|bakery|croissant|sourdough|pumpkin|cinnamon",
                                              d["text"], re.I)),
    "foreign_tools": lambda d: bool(d["foreign_calls"]),
}


def _hist(store: Store, tl: str, role, kind, content, tool=None):
    return store.add_turn(tl, role, kind, content, tool)


def build_history(store: Store, tl: str) -> None:
    """Earlier work done in the same timeline: load_skill + real tool calls per skill."""
    linked = set(CAPABILITY)
    n = iter(range(1, 100))

    def task(user: str, skill: str, tool: str, args: dict, result: str, reply: str):
        _hist(store, tl, "user", "text", user)
        lid = f"call_hist_{next(n)}"
        _hist(store, tl, "assistant", "tool_use", {"id": lid, "input": {"name": skill}}, "load_skill")
        _hist(store, tl, "user", "tool_result",
              {"tool_use_id": lid, "content": load_skill(store, linked, skill)[0]}, "load_skill")
        tid = f"call_hist_{next(n)}"
        _hist(store, tl, "assistant", "tool_use", {"id": tid, "input": args}, tool)
        _hist(store, tl, "user", "tool_result", {"tool_use_id": tid, "content": result}, tool)
        _hist(store, tl, "assistant", "text", reply)

    task("Can you build a website for my bakery, Sunny Crumbs?", "website", "site_write",
         {"page": "home", "content": "# Welcome to Sunny Crumbs! 🥐✨\nFresh bread every morning, baked "
          "with love 💛\n## Our Treats 🍰\nCroissants, sourdough and cinnamon rolls! 🎉"},
         "wrote page home", "Done! 🎉 Your homepage is live with a fun, playful vibe 🥐")
    task("Book me a dentist checkup on 2026-11-03 at 9am.", "calendar", "calendar_book",
         {"date": "2026-11-03", "time": "09:00", "what": "dentist checkup"},
         "booked dentist checkup on 2026-11-03 at 09:00", "Booked your dentist checkup for 2026-11-03 at 09:00.")
    task("Write a newsletter announcing our pumpkin bread.", "newsletter", "newsletter_draft",
         {"subject": "PUMPKIN SEASON IS HERE!", "body": "Big news, friends! Our PUMPKIN BREAD is BACK "
          "for a LIMITED time!\nCome by early, it sells out FAST!\nCheers! 🎉"},
         "newsletter saved", "Your newsletter is drafted! It's got all the EXCITEMENT you wanted.")
    task("Now post about it on Instagram.", "social", "social_post",
         {"platform": "instagram", "text": "Pumpkin bread is back and it's super awesome 🎃 Grab a loaf "
          "before it's gone! #SunnyCrumbs #PumpkinSeason #BakeryLife What's your fall fave?"},
         "social saved", "Posted! Super awesome reach so far. Anything else? 😄")
    task("Save my grandma's cinnamon roll recipe: proof dough, roll with cinnamon sugar, bake 25 min.",
         "recipes", "recipe_save",
         {"title": "grandma's cinnamon rolls", "steps": ["🍞 proof the dough", "🍯 roll with cinnamon sugar",
                                                         "🔥 bake for 25 minutes"]},
         "recipes saved", "Saved grandma's cinnamon rolls 🍯")


def build(path: str, provider):
    store = Store(path)
    install(store)
    store.create_timeline("o", None, ["clone"])
    build_history(store, "o")
    task_row = store.add_turn("o", "user", "text", TASK)
    # clone arm: graft clone seeded only with the task-relevant user row
    store.commit_clone("o", "c1", [task_row], ["slides", "join"])
    # shared arm: one timeline with every capability skill linked and the same history
    store.create_timeline("shared", None, CAPABILITY)
    for r in store.view("o"):
        store.append("shared", r["row_id"])
    h = Harness(store, provider)
    arms = {
        "clone": {
            "rows": store.view("c1"), "system": h.system("c1"), "tools": store.tools_for("c1"),
            "linked": {"slides", "join"}},
        "shared": {
            "rows": store.view("shared"),
            "system": "\n\n".join([BASE_SYSTEM, skill_listing(
                store, CAPABILITY,
                "Your skills (call load_skill(name) to read a skill's full instructions before using it):")]),
            "tools": store.tools_for("shared") + [LOAD_SKILL_TOOL], "linked": set(CAPABILITY)},
        "subagent": {
            "rows": store.view("shared"), "system": SubagentRunner(store, provider).parent_system(),
            "tools": [DELEGATE_TOOL], "linked": set()},
    }
    return store, arms


def subagent_run(provider, arm: dict, world: World, nonce: str | None) -> dict:
    """Parent writes the handoff (one call on the shared history), then the subagent works."""
    with provider.usage_tag("parent"):
        blocks = provider.complete(arm["system"], prefix_nonce(render(arm["rows"]), nonce), arm["tools"])
    use = next((b for b in blocks if b["type"] == "tool_use" and b["name"] == "delegate"), None)
    if use is None:
        return {"texts": [b["text"] for b in blocks if b["type"] == "text"], "loaded": [], "handoff": None}
    provider.retag_last("handoff")
    store = Store(":memory:")  # SQLite is per-thread: a private catalog copy for this worker
    install(store)
    d = SubagentRunner(store, provider, world=world, nonce=nonce).run_subagent(
        str(use["input"].get("skill", "")), str(use["input"].get("task_prompt", "")))
    return {"texts": [d.summary], "loaded": d.loaded, "handoff": d.task_prompt, "summary": d.summary}


def agent_loop(provider, instructions: dict[str, str], arm: dict, world: World, max_steps: int = 8,
               nonce: str | None = None) -> dict:
    msgs = prefix_nonce(render(arm["rows"]), nonce)
    texts, loaded = [], []
    for _ in range(max_steps):
        blocks = provider.complete(arm["system"], msgs, arm["tools"])
        msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(empty)"}]})
        texts += [b["text"] for b in blocks if b["type"] == "text"]
        uses = [b for b in blocks if b["type"] == "tool_use"]
        if not uses:
            break
        results, stop = [], False
        for u in uses:
            if u["name"] == "join":
                content, err, stop = "joined", False, True
            elif u["name"] == "load_skill":
                loaded.append(str(u["input"].get("name")))
                want = str(u["input"].get("name", ""))
                content, err = ((instructions[want], False) if want in arm["linked"] and want in instructions
                                else (f"skill '{want}' is not linked to you; you can load: "
                                      f"{sorted(arm['linked'])}", True))
            else:
                content, err = world.execute(u["name"], u["input"])
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": content, "is_error": err})
        msgs.append({"role": "user", "content": results})
        if stop:
            break
    return {"texts": texts, "loaded": loaded}


def title_case(t: str) -> bool:
    small = {"a", "an", "and", "the", "of", "for", "to", "in", "on", "at", "with", "or", "by", "your", "our"}
    words = [w for w in re.findall(r"[A-Za-z][\w'-]*", t)]
    return bool(words) and all(w[0].isupper() for i, w in enumerate(words) if i == 0 or w.lower() not in small)


def score(world: World, run: dict) -> dict:
    calls = Counter(name for name, _ in world.log)
    deck = world.decks[-1] if world.decks else {"title": "", "slides": []}
    titles = [deck.get("title", "")] + [s["title"] for s in deck["slides"]]
    strings = titles + [b for s in deck["slides"] for b in s["bullets"]]
    d = {"text": "\n".join(strings), "strings": strings, "titles": titles,
         "foreign_calls": [n for n in calls if n.startswith(("site_", "calendar_", "newsletter_",
                                                              "social_", "recipe_"))]}
    slide_titles = " ".join(s["title"].lower() for s in deck["slides"])
    compliance = {
        "one_deck": calls["slides_create"] == 1,
        "slide_per_service": all(k in slide_titles for k in ("clean", "implant", "whiten")),
        "three_bullets_each": bool(deck["slides"]) and all(len(s["bullets"]) == 3 for s in deck["slides"]),
        "audience_facts": all(re.search(k, str(deck.get("audience", "")), re.I) for k in ("retire", "ohio|columbus")),
        "title_case": bool(deck["slides"]) and all(title_case(t) for t in titles),
    }
    leaks = {k: bool(world.decks) and f(d) for k, f in LEAKS.items()}
    leaks["foreign_tools"] = bool(d["foreign_calls"])
    total = len(compliance) + len(leaks)
    ok = sum(compliance.values()) + sum(not v for v in leaks.values())
    return {
        "compliance": compliance, "leaks": leaks, "score": ok / total,
        "correct": all(compliance.values()) and not any(leaks.values()),
        "signature": tuple(s["title"].strip().lower() for s in deck["slides"]),
        "loaded_slides": "slides" in run["loaded"], "loaded": run["loaded"],
        "old_formal_tone_check": "formal" in str(deck.get("tone", "")).lower(),
        "tone_field": deck.get("tone"), "deck": deck,
    }


# ------------------------------------------------------------------ statistics
def fisher_two_sided(a: int, n1: int, c: int, n2: int) -> float:
    """Fisher's exact test on [[a, n1-a], [c, n2-c]]; two-sided (sum of p <= p_obs)."""
    k = a + c
    n = n1 + n2

    def p(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    lo, hi = max(0, k - n2), min(k, n1)
    p_obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * (1 + 1e-9)))


def permutation_p(x: list[float], y: list[float], iters: int = 20000, seed: int = 7) -> float:
    rng = random.Random(seed)
    obs = abs(statistics.mean(x) - statistics.mean(y))
    pool, n = x + y, len(x)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(statistics.mean(pool[:n]) - statistics.mean(pool[n:])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1)


# ------------------------------------------------------------------ fake
def fake_policy(system, messages, tools):
    if any(t["name"] == "delegate" for t in tools):
        return [{"type": "tool_use", "id": "fd", "name": "delegate",
                 "input": {"skill": "slides", "task_prompt": TASK + " Use a formal tone."}}]
    last = messages[-1]["content"][-1]
    used = [b["name"] for m in messages if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    if last["type"] == "tool_result" and used and used[-1] != "load_skill":
        return [{"type": "text", "text": "Built the deck."}]
    if used.count("load_skill") <= (5 if "newsletter" in system else 0):
        return [{"type": "tool_use", "id": "fl", "name": "load_skill", "input": {"name": "slides"}}]
    out = [{"type": "tool_use", "id": "f0", "name": "slides_create",
            "input": {"title": "Lakeview Dental Open House", "audience": "Retirees in Columbus, Ohio",
                      "tone": "formal"}}]
    for i, s in enumerate(("Cleanings", "Implants", "Whitening")):
        out.append({"type": "tool_use", "id": f"f{i + 1}", "name": "slides_add_slide",
                    "input": {"title": s, "bullets": ["One.", "Two.", "Three."]}})
    return out


# ------------------------------------------------------------------ report
def arm_usage(runs: list[dict]) -> dict:
    u = empty_usage()
    for r in runs:
        add_usage(u, r["usage"])
    return u


def arm_summary(name: str, runs: list[dict], pricing: cachestats.Pricing) -> list[str]:
    scores = [r["score"] for r in runs]
    u = arm_usage(runs)
    tin, tout = u["input"], u["output"]
    cost = pricing.cost(u)
    sigs = Counter(r["signature"] for r in runs)
    lines = [
        f"[{name}] correct {sum(r['correct'] for r in runs)}/{len(runs)}  mean score {statistics.mean(scores):.3f}  "
        f"stdev {statistics.pstdev(scores):.3f}  distinct slide-title structures {len(sigs)}  "
        f"loaded slides {sum(r['loaded_slides'] for r in runs)}/{len(runs)}",
        f"    tokens in {tin} (uncached {u['uncached']}, cached {u['cached']}, writes {u['cache_write']}) "
        f"out {tout} (per run {tin / len(runs):.0f}/{tout / len(runs):.0f}), calls {u['calls']}, "
        f"effective in {pricing.effective_input(u):.0f}, cost {f'${cost:.4f}' if cost is not None else 'n/a'}",
        "    compliance failures: " + str(dict(Counter(k for r in runs for k, v in r["compliance"].items() if not v)) or "{}"),
        "    leakage (runs with >=1): " + ", ".join(f"{k}={sum(r['leaks'][k] for r in runs)}" for k in LEAKS),
        f"    old scorer 'formal' in tone field: {sum(r['old_formal_tone_check'] for r in runs)}/{len(runs)}; "
        f"tone values: {dict(Counter(str(r['tone_field']) for r in runs).most_common(4))}",
    ]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--cache-scope", choices=("run", "global"), default="run",
                    help="run: unique nonce per run (realistic); global: no nonce (upper bound)")
    a = ap.parse_args()
    provider = FakeProvider(fake_policy) if a.fake else make_provider()
    path = os.path.join(tempfile.mkdtemp(prefix="graft-iso-"), "iso.db")
    store, arms = build(path, provider)
    # SQLite connections are per-thread; snapshot the skills' instructions for worker threads
    instructions = {n: i for n, i in store.con.execute("SELECT name, instructions FROM skills")}
    prefix_counts = {}
    for name, arm in arms.items():
        print(f"--- arm {name}: {len(arm['rows'])} context rows; tools {[t['name'] for t in arm['tools']]}; "
              f"system prompt {len(arm['system'])} chars")
        try:
            prefix_counts[name] = provider.count_prefix_tokens(arm["system"], arm["tools"])
        except Exception as e:  # counting is informational; never fail the experiment on it
            print(f"    (prefix token count failed: {type(e).__name__}: {e})")
            prefix_counts[name] = None

    def job(arm_name: str, i: int) -> tuple[str, int, dict]:
        usage = provider.thread_usage_reset()
        world = World()
        nonce = uuid.uuid4().hex[:16] if a.cache_scope == "run" else None
        try:
            if arm_name == "subagent":
                run = subagent_run(provider, arms[arm_name], world, nonce)
            else:
                with provider.usage_tag(f"{arm_name}#{i}"):
                    run = agent_loop(provider, instructions, arms[arm_name], world, nonce=nonce)
        except Exception as e:  # count API failures as failed runs, keep going
            run = {"texts": [f"ERROR {type(e).__name__}: {e}"], "loaded": []}
        r = score(world, run)
        r["usage"] = dict(usage)
        r["handoff"], r["summary"] = run.get("handoff"), run.get("summary")
        r["error"] = run["texts"][0] if run["texts"] and run["texts"][0].startswith("ERROR") else None
        return arm_name, i, r

    jobs = [(arm, i) for i in range(a.n) for arm in arms]
    results: dict[str, list[dict]] = {k: [None] * a.n for k in arms}  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for arm_name, i, r in ex.map(lambda j: job(*j), jobs):
            results[arm_name][i] = r

    for arm_name in arms:
        print(f"\n--- runs: {arm_name}")
        for i, r in enumerate(results[arm_name]):
            bad = [k for k, v in r["compliance"].items() if not v] + [k for k, v in r["leaks"].items() if v]
            print(f"  {i + 1:>2}: {r['score']:.2f} {'OK ' if r['correct'] else 'BAD'} loaded={r['loaded']} "
                  f"slides={list(r['signature'])} issues={bad}" + (f" {r['error'][:120]}" if r["error"] else ""))

    pricing = cachestats.Pricing.from_env()
    print("\n================ ISOLATION REPORT ================")
    print(f"provider={provider.name}:{provider.model} N={a.n} per arm; cache scope={a.cache_scope}; "
          f"prompt_cache_key={'set' if getattr(provider, 'use_cache_key', False) else 'not set'}; "
          f"total usage={provider.usage}")
    for name in arms:
        print("\n".join(arm_summary(name, results[name], pricing)))
    print("\nprompt cache (per arm; totals then per-run means):")
    print("\n".join("  " + line for line in cachestats.table(
        [(name, arm_usage(results[name])) for name in arms], pricing, per=a.n)))
    print("  " + pricing.assumptions())
    print("where cache reads came from (with --cache-scope run a run's first call can reuse only "
          "tools+system from other runs; later calls can also reuse their own run's history):")
    for name in [n for n in arms if n != "subagent"]:
        first, later = empty_usage(), empty_usage()
        for i in range(a.n):
            calls = [c for c in provider.call_log if c["tag"] == f"{name}#{i}"]
            if calls:
                add_usage(first, calls[0])
                for call in calls[1:]:
                    add_usage(later, call)
        print(f"  {name}: first calls cached {first['cached']}/{first['input']} prompt tokens "
              f"({cachestats.hit_rate(first):.1%}); later calls cached {later['cached']}/{later['input']} "
              f"({cachestats.hit_rate(later):.1%})")
    if "subagent" in arms:
        print("subagent arm by role (all runs):")
        print("\n".join("    " + line for line in cachestats.table(
            [(t, provider.by_tag[t]) for t in ("parent", "handoff", "subagent", "summary")
             if t in provider.by_tag and provider.by_tag[t]["calls"]], pricing)))
        sub = results["subagent"]
        print(f"  handoffs written {sum(1 for r in sub if r['handoff'])}/{len(sub)}; mean handoff "
              f"{statistics.mean(len(r['handoff']) for r in sub if r['handoff']) if any(r['handoff'] for r in sub) else 0:.0f} chars; "
              f"example handoff: {next((r['handoff'] for r in sub if r['handoff']), None)!r}")
        print("  (the subagent itself starts from a fresh static prefix: slides listing + load_skill + slides tools)")
    print("cacheable static prefix (identical across runs of an arm):")
    for name in arms:
        print(cachestats.prefix_line(name, prefix_counts[name], cachestats.min_cacheable(provider.name)))
    print("\nsignificance (two-sided; alpha=0.05):")
    for other in [n for n in ("shared", "subagent") if n in results]:
        c, s = results["clone"], results[other]
        pc, ps = sum(r["correct"] for r in c), sum(r["correct"] for r in s)
        p_pass = fisher_two_sided(pc, len(c), ps, len(s))
        print(f"  pass rate clone {pc}/{len(c)} vs {other} {ps}/{len(s)}: Fisher p={p_pass:.4f}"
              f" -> {'SIGNIFICANT' if p_pass < 0.05 else 'not significant'}")
        p_score = permutation_p([r["score"] for r in c], [r["score"] for r in s])
        print(f"  mean score clone {statistics.mean(r['score'] for r in c):.3f} vs {other} "
              f"{statistics.mean(r['score'] for r in s):.3f}: permutation p={p_score:.4f}"
              f" -> {'SIGNIFICANT' if p_score < 0.05 else 'not significant'}")
        for k in LEAKS:
            lc, ls = sum(r["leaks"][k] for r in c), sum(r["leaks"][k] for r in s)
            if lc or ls:
                p = fisher_two_sided(lc, len(c), ls, len(s))
                print(f"  leak {k}: clone {lc}/{len(c)} vs {other} {ls}/{len(s)}: Fisher p={p:.4f}"
                      f" -> {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
        anyc, anys = sum(any(r["leaks"].values()) for r in c), sum(any(r["leaks"].values()) for r in s)
        p_any = fisher_two_sided(anyc, len(c), anys, len(s))
        print(f"  any leakage: clone {anyc}/{len(c)} vs {other} {anys}/{len(s)}: Fisher p={p_any:.4f}"
              f" -> {'SIGNIFICANT' if p_any < 0.05 else 'not significant'}")
    print(f"db: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
