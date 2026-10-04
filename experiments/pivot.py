"""Pivot experiment: website -> slide show, graft vs a subagent baseline.

The user describes a website (dentists in Ohio, formal tone, 3 services), then pivots:
"Instead of a website let's just build a slide show". Same scripted user in both arms.

  graft:  the original clones (one call) a website clone that hears the facts; the clone
            joins out_of_scope on the pivot; the deterministic default join (no model call)
            appends pointers to the clone's text rows; the original seeds a slides clone with them.
  subagent: the parent (same persona) talks to the user throughout and calls
            delegate(skill, task_prompt); the subagent sees only that parent-written prompt and
            returns a summary (graft/subagent.py).

Primary output: per-role prompt-cache and cost tables for both arms. Fact preservation (each
user fact in the handoff/seed and in the deck) and the verbatim handoff prompts and summaries
are printed as a DIAGNOSTIC only: this pivot is too small to show fact loss (no size-capped
summaries, compaction or nested delegation), so no verdict is drawn from it.

Graft checks:
  1. the website clone joined with reason=out_of_scope
  2. the original's (default) join for it committed
  3. the slides clone's seed points at the user's own audience and tone rows
     (same row_ids -> verbatim, no paraphrase)
  4. the deck uses those facts

Usage:
  uv run python experiments/pivot.py            # live (GRAFT_PROVIDER / keys from env)
  uv run python experiments/pivot.py --fake     # offline, scripted FakeProvider
  --arm graft|subagent|both (default both)
  --cache-scope run|global   run (default): a unique nonce starts the first message of every
                             request in this run, so separate invocations share only tools +
                             system; global: no nonce. Cache table per role: see graft/cachestats.py.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft import cachestats  # noqa: E402
from graft.db import Store  # noqa: E402
from graft.fake import pivot_policy, subagent_policy  # noqa: E402
from graft.harness import Harness  # noqa: E402
from graft.providers import FakeProvider, add_usage, empty_usage, make_provider  # noqa: E402
from graft.sim import run_script  # noqa: E402
from graft.skills import BASE_SYSTEM, LOAD_SKILL_TOOL, install  # noqa: E402
from graft.subagent import DELEGATE_TOOL, SKILL_TOOLS, SubagentRunner  # noqa: E402

FACT_AUDIENCE = "The site is for a group of dentists in Ohio; the audience is their patients."
FACT_TONE = "Please keep the tone formal."
FACT_SERVICES = "We offer 3 services: teeth cleaning, dental implants, and teeth whitening."
PIVOT = "Instead of a website let's just build a slide show."


def nudge(h) -> str | None:
    return None if h.world.decks and h.world.decks[-1]["slides"] else \
        "Please go ahead and build the slide show now with what you know."


SCRIPT = ["Hi! I need a website for my business.", FACT_AUDIENCE, FACT_TONE, FACT_SERVICES, PIVOT,
          nudge, nudge]


def clones_with(store: Store, skill: str) -> list[str]:
    return [r[0] for r in store.con.execute(
        "SELECT ts.timeline_id FROM timeline_skills ts JOIN skills s USING(skill_id) "
        "JOIN timelines t USING(timeline_id) WHERE s.name=? AND t.source_id IS NOT NULL "
        "ORDER BY t.created_at", (skill,))]


def user_row(store: Store, text: str) -> int | None:
    r = store.con.execute("SELECT row_id FROM turns WHERE role='user' AND kind='text' AND content=? "
                          "ORDER BY row_id LIMIT 1", (text,)).fetchone()
    return r[0] if r else None


def graft_checks(h: Harness, verbose: bool = True) -> bool:
    """Structural graft checks. verbose=False returns the verdict without printing (bench.py)."""
    checks = pivot_structural_checks(h)
    if not verbose:
        return all(c[1] for c in checks)
    s = h.store
    piv = user_row(s, PIVOT)
    print("\n================ PIVOT REPORT: graft ================")
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    print(f"info: provider={h.provider.name}:{h.provider.model} usage={getattr(h.provider, 'usage', {})}")
    ok = all(c[1] for c in checks)
    print(f"OVERALL (graft): {'PASS' if ok else 'FAIL'}")
    clones = [r for r in s.con.execute(
        "SELECT timeline_id, join_reason, join_trigger FROM timelines WHERE source_id IS NOT NULL "
        "ORDER BY created_at")]
    premature = [c["timeline_id"] for c in clones
                 if c["join_reason"] == "out_of_scope" and c["join_trigger"] != piv]
    print(f"info: clones={len(clones)} joins={[(c['timeline_id'], c['join_reason'], c['join_trigger']) for c in clones]} "
          f"premature out_of_scope (trigger is not the pivot)={premature} "
          f"silent-done refusals={h.silent_done_refusals}")
    return ok


def pivot_structural_checks(h: Harness) -> list[tuple[str, bool, str]]:
    s = h.store
    checks: list[tuple[str, bool, str]] = []
    web = clones_with(s, "website")
    slides = clones_with(s, "slides")
    aud, tone, svc = user_row(s, FACT_AUDIENCE), user_row(s, FACT_TONE), user_row(s, FACT_SERVICES)
    piv = user_row(s, PIVOT)
    fact_owners = {s.turn(r)["owner_timeline"] for r in (aud, tone, svc) if r}
    checks.append(("facts were told to a website clone (not the original)",
                   bool(fact_owners) and fact_owners <= set(web),
                   f"fact rows {aud},{tone},{svc} owned by {sorted(fact_owners)}; website clones={web}"))
    # the website clone that received the pivot message
    w = s.turn(piv)["owner_timeline"] if piv else None
    t = s.timeline(w) if w else None
    ok1 = bool(t and w in web and t["join_reason"] == "out_of_scope" and t["join_trigger"] == piv)
    checks.append(("website clone joined out_of_scope on the pivot message", ok1,
                   f"pivot row={piv} owner={w} join_reason={t and t['join_reason']} "
                   f"trigger={t and t['join_trigger']}"))
    tx = s.con.execute("SELECT tx_id, status FROM transactions WHERE kind='join' AND target=?",
                       (w,)).fetchone() if w else None
    held = s.pointed(s.root())
    remembered = [r for r in (aud, tone, svc) if r in held]
    checks.append(("original's default join committed and it remembers the facts",
                   bool(tx and tx["status"] == "committed" and aud in held and tone in held),
                   f"join tx={dict(tx) if tx else None}; fact rows now in original's context: {remembered}"))
    sl = slides[0] if slides else None
    # the seed: pointers the clone did not produce itself
    seed = [r["row_id"] for r in s.view(sl) if r["owner_timeline"] != sl] if sl else []
    owners = {r: s.turn(r)["owner_timeline"] for r in (aud, tone, svc) if r}
    checks.append(("slides seed holds audience+tone rows verbatim (same row_ids)",
                   bool(sl and aud in seed and tone in seed),
                   f"slides clone={sl} seed={seed} audience_row={aud} tone_row={tone} "
                   f"services_row={svc} (in seed: {svc in seed}) owners={owners}"))
    paraphrase = [r for r in seed if s.turn(r)["role"] == "assistant" and s.turn(r)["kind"] == "text"
                  and s.turn(r)["owner_timeline"] == s.root()]
    deck = h.world.deck_text().lower()
    used_ohio = "ohio" in deck
    used_dent = "dent" in deck
    services = sum(k in deck for k in ("clean", "implant", "whiten"))
    tone_field = str(h.world.decks[-1].get("tone", "")).lower() if h.world.decks else ""
    formal = bool(h.world.decks) and any(k in tone_field for k in ("formal", "professional")) \
        and not re.search("[\U0001F300-\U0001FAFF\u2600-\u27BF]", deck)
    checks.append(("deck uses the facts (Ohio, dentists, formal tone, >=2 services)",
                   used_ohio and used_dent and formal and services >= 2,
                   f"ohio={used_ohio} dentists={used_dent} formal_tone={formal} services_mentioned={services} "
                   f"decks={len(h.world.decks)}"))
    return checks


def deck_check(world) -> tuple[bool, str]:
    deck = world.deck_text().lower()
    services = sum(k in deck for k in ("clean", "implant", "whiten"))
    tone_field = str(world.decks[-1].get("tone", "")).lower() if world.decks else ""
    formal = bool(world.decks) and any(k in tone_field for k in ("formal", "professional")) \
        and not re.search("[\U0001F300-\U0001FAFF☀-➿]", deck)
    ok = "ohio" in deck and "dent" in deck and formal and services >= 2
    return ok, (f"ohio={'ohio' in deck} dentists={'dent' in deck} formal_tone={formal} "
                f"services_mentioned={services} decks={len(world.decks)}")


# ------------------------------------------------------------------ fact preservation
FACTS = {  # each user fact -> a pattern for its key term (presence only; read the logged text for paraphrase)
    "dentists": r"dentist",
    "Ohio": r"ohio",
    "audience = patients": r"patient",
    "formal tone": r"formal",
    "teeth cleaning": r"clean",
    "dental implants": r"implant",
    "teeth whitening": r"whiten",
}


def fact_hits(text: str) -> dict[str, bool]:
    return {k: bool(re.search(p, text, re.I)) for k, p in FACTS.items()}


def fact_table(handoff: str, deck: str) -> list[str]:
    h, d = fact_hits(handoff), fact_hits(deck)
    lines = [f"  {'fact':<22} {'in handoff/seed':>15} {'in deck':>8}"]
    lines += [f"  {k:<22} {('yes' if h[k] else 'NO'):>15} {('yes' if d[k] else 'NO'):>8}" for k in FACTS]
    lines.append(f"  preserved: handoff/seed {sum(h.values())}/{len(FACTS)}, deck {sum(d.values())}/{len(FACTS)}")
    return lines


# ------------------------------------------------------------------ cache tables
# Common role names across arms: parent/original, handoff/clone-tx, subagent/clone, summary/join-tx.
ROLE_ORDER = ("parent/original", "handoff/clone-tx", "subagent/clone", "summary/join-tx")


def graft_role(tag: str) -> str:
    if tag == "original":
        return "parent/original"
    if tag.endswith(":clone_tx"):
        return "handoff/clone-tx"
    if tag.startswith("clone:"):
        return "subagent/clone"
    return tag


def subagent_role(tag: str) -> str:
    return {"parent": "parent/original", "handoff": "handoff/clone-tx", "subagent": "subagent/clone",
            "summary": "summary/join-tx"}.get(tag, tag)


def cache_table(label: str, provider, role_of, extra: list[tuple[str, dict]] = ()) -> dict[str, dict]:
    roles: dict[str, dict] = {}
    for tag, u in getattr(provider, "by_tag", {}).items():
        if u["calls"]:
            add_usage(roles.setdefault(role_of(tag), empty_usage()), u)
    total = empty_usage()
    for u in roles.values():
        add_usage(total, u)
    rows = [(r, roles[r]) for r in ROLE_ORDER if r in roles] + \
           [(r, u) for r, u in roles.items() if r not in ROLE_ORDER] + list(extra) + [("TOTAL", total)]
    print(f"\nprompt cache by role [{label}] (prompt_cache_key="
          f"{'set' if getattr(provider, 'use_cache_key', False) else 'not set'}):")
    print("\n".join("  " + line for line in cachestats.table(rows, cachestats.Pricing.from_env())))
    roles["TOTAL"] = total
    return roles


def shared_prefix(provider, src_system: str, src_tools: list[dict], system: str, tools: list[dict]) -> str:
    """Where a child's static prefix (tools + system) stops matching its source's."""
    n_tools = 0
    for a, b in zip(src_tools, tools):
        if a != b:
            break
        n_tools += 1
    n_chars = 0
    for a, b in zip(src_system, system):
        if a != b:
            break
        n_chars += 1
    try:
        c = provider.count_prefix_tokens(system[:n_chars], tools[:n_tools])
    except Exception:  # informational only
        c = None
    toks = f"{c['tools_and_system']} tokens{'' if c.get('exact') else ' (estimate)'}" if c else "n/a"
    return (f"first {n_tools}/{len(tools)} tool definitions identical {[t['name'] for t in tools[:n_tools]]}; "
            f"system identical for its first {n_chars} chars (persona = {len(BASE_SYSTEM)} chars); "
            f"common static part {toks}")


def first_call_line(label: str, call: dict | None) -> str:
    if not call:
        return f"  {label}: first call n/a"
    return (f"  {label}: first call prompt {call['input']} tokens, cached {call['cached']} "
            f"({cachestats.hit_rate(call):.0%}), cache writes {call['cache_write']}")


# ------------------------------------------------------------------ arms
async def run_graft(provider, nonce: str | None, quiet: bool, db_path: str | None) -> dict:
    path = db_path or os.path.join(tempfile.mkdtemp(prefix="graft-pivot-"), "pivot.db")
    store = Store(path)
    install(store)
    h = Harness(store, provider, trace=None if quiet else (lambda m: print("  · " + m)), nonce=nonce)
    print("\n######## ARM: graft ########")
    await h.start()
    await run_script(h, SCRIPT)
    ok = graft_checks(h)
    slides = clones_with(store, "slides")
    sl = slides[0] if slides else None
    seed_rows = [r for r in store.view(sl) if r["owner_timeline"] != sl] if sl else []
    seed_text = "\n".join(r["content"] for r in seed_rows if r["kind"] == "text")
    print("\nslides clone seed (pointers to existing rows; no text generated for the handoff):")
    for r in seed_rows:
        print(f"  row {r['row_id']} owner={r['owner_timeline']} {r['role']} {r['kind']}: {r['content'][:200]}")
    print("fact preservation (DIAGNOSTIC only, no verdict; handoff = first slides clone's seed):")
    print("\n".join(fact_table(seed_text, h.world.deck_text())))
    roles = cache_table("graft", provider, graft_role,
                        [(f"  {t}", u) for t, u in sorted(provider.by_tag.items()) if t.startswith("clone:")])
    print("first call of each clone (does a fresh clone get any cache hit?):")
    root = store.root()
    for tl in [r[0] for r in store.con.execute(
            "SELECT timeline_id FROM timelines WHERE source_id IS NOT NULL ORDER BY created_at")]:
        print(first_call_line(tl, next((c for c in provider.call_log if c["tag"] == f"clone:{tl}"), None)))
        print("    static prefix vs original: " + shared_prefix(
            provider, h.system(root), store.tools_for(root), h.system(tl), store.tools_for(tl)))
    print(f"db: {path}")
    if not quiet:
        print(store.dump())
    await h.machines.stop()
    return {"ok": ok, "roles": roles, "deck": h.world.deck_text(), "handoff": seed_text}


async def run_subagent(provider, nonce: str | None, quiet: bool) -> dict:
    store = Store(":memory:")
    install(store)
    r = SubagentRunner(store, provider, nonce=nonce, trace=None if quiet else (lambda m: print("  · " + m)))
    print("\n######## ARM: subagent ########")
    await run_script(r, SCRIPT)
    ok, detail = deck_check(r.world)
    slides = [d for d in r.delegations if d.skill == "slides"]
    print("\n================ PIVOT REPORT: subagent ================")
    print(f"[{'PASS' if slides else 'FAIL'}] slides delegated after the pivot ({len(slides)} slides delegation(s))")
    print(f"[{'PASS' if ok else 'FAIL'}] deck uses the facts (Ohio, dentists, formal tone, >=2 services)\n"
          f"        {detail}")
    overall = ok and bool(slides)
    print(f"OVERALL (subagent): {'PASS' if overall else 'FAIL'}")
    print("\ndelegations (handoff prompts and summaries, verbatim):")
    for i, d in enumerate(r.delegations, 1):
        print(f"  #{i} delegate({d.skill}) subagent steps={d.steps}")
        print(f"    HANDOFF ({len(d.task_prompt)} chars): {d.task_prompt}")
        print(f"    SUMMARY ({len(d.summary)} chars): {d.summary}")
        print("    facts in handoff: " + (", ".join(k for k, v in fact_hits(d.task_prompt).items() if v) or "none"))
    handoff = slides[-1].task_prompt if slides else ""
    print("fact preservation (DIAGNOSTIC only, no verdict; handoff = last slides delegation prompt):")
    print("\n".join(fact_table(handoff, r.world.deck_text())))
    roles = cache_table("subagent", provider, subagent_role)
    print("first call of each subagent (does a fresh subagent get any cache hit?):")
    for i, d in enumerate(r.delegations, 1):
        print(first_call_line(f"#{i} {d.skill}", d.first_call))
        print("    static prefix vs parent: " + shared_prefix(
            provider, r.parent_system(), [DELEGATE_TOOL], d.system, d.tools))
    return {"ok": overall, "roles": roles, "deck": r.world.deck_text(), "handoff": handoff}


def comparison(res: dict, provider) -> None:
    pricing = cachestats.Pricing.from_env()
    print("\n================ ARM COMPARISON ================")
    rows = [(f"{arm}: {role}", r["roles"][role]) for arm, r in res.items()
            for role in ROLE_ORDER + ("TOTAL",) if role in r["roles"]]
    print("\n".join("  " + line for line in cachestats.table(rows, pricing)))
    print("  " + pricing.assumptions())
    if "graft" in res:
        m = res["graft"]["roles"]
        tx = m.get("handoff/clone-tx", {}).get("output", 0)
        print(f"  graft: output tokens in calls that emitted one-shot clone(seed) "
              f"(the default join makes no call; incl. reasoning) = {tx}")
    if "subagent" in res:
        sa = res["subagent"]["roles"]
        ho, sm = (sa.get(k, {}).get("output", 0) for k in ("handoff/clone-tx", "summary/join-tx"))
        print(f"  subagent: output tokens in calls that wrote handoff prompts {ho} + summaries {sm} = {ho + sm} "
              f"(incl. reasoning)")
    for arm, r in res.items():
        h, d = fact_hits(r["handoff"]), fact_hits(r["deck"])
        print(f"  {arm}: checks {'PASS' if r['ok'] else 'FAIL'}; diagnostic facts in handoff/seed "
              f"{sum(h.values())}/{len(FACTS)}, in deck {sum(d.values())}/{len(FACTS)}; effective input "
              f"{pricing.effective_input(r['roles']['TOTAL']):.0f}, effective total "
              f"{pricing.effective_total(r['roles']['TOTAL']):.0f}")
    store = Store(":memory:")
    install(store)
    store.create_timeline("o", None, ["clone"])
    h, sub = Harness(store, provider), SubagentRunner(store, provider)
    m = cachestats.min_cacheable(provider.name)
    print("static prefix (tools + system) lengths:")
    try:
        print(cachestats.prefix_line("graft original", provider.count_prefix_tokens(
            h.system("o"), store.tools_for("o")), m))
        print(cachestats.prefix_line("subagent parent", provider.count_prefix_tokens(
            sub.parent_system(), [DELEGATE_TOOL]), m))
        print(cachestats.prefix_line("slides subagent", provider.count_prefix_tokens(
            sub.subagent_system("slides"), [LOAD_SKILL_TOOL] + SKILL_TOOLS["slides"]), m))
    except Exception as e:  # informational only
        print(f"  prefix token count failed: {type(e).__name__}: {e}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="offline scripted provider")
    ap.add_argument("--db", default=None)
    ap.add_argument("--quiet", action="store_true", help="hide the trace and the DB dump")
    ap.add_argument("--arm", choices=("graft", "subagent", "both"), default="both")
    ap.add_argument("--cache-scope", choices=("run", "global"), default="run",
                    help="run: unique nonce per arm run (realistic); global: no nonce (upper bound)")
    a = ap.parse_args()
    nonce = (lambda: uuid.uuid4().hex[:16]) if a.cache_scope == "run" else (lambda: None)
    res: dict[str, dict] = {}
    provider = None
    if a.arm in ("graft", "both"):
        provider = FakeProvider(pivot_policy) if a.fake else make_provider()
        res["graft"] = await run_graft(provider, nonce(), a.quiet, a.db)
    if a.arm in ("subagent", "both"):
        provider = FakeProvider(subagent_policy) if a.fake else make_provider()
        res["subagent"] = await run_subagent(provider, nonce(), a.quiet)
    print(f"\ncache scope={a.cache_scope}; provider={provider.name}:{provider.model}")
    comparison(res, provider)
    return 0 if all(r["ok"] for r in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
