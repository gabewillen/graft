"""Ultimate thesis benchmark: multi-turn negotiation + mid-stream revisions + pivot + probe.

Tests the exact scenario where Agent Grafts should structurally differ from subagents:
1. Multi-turn focus negotiation (`website`, casual/emoji style) with specific facts and 3 mid-stream
   revisions (date moved May 12 -> May 19, Dr. Priya Patel dropped, whitening promo $199 -> $149).
2. Concurrent detached work (`calendar` booking for 2026-05-18 14:30) during the negotiation.
3. Cross-skill pivot (`slides`, strict formal style, zero emoji, zero '!', zero #hashtags) that must
   use all finalized revised details and ZERO retracted facts.
4. Direct probe question verifying verbatim recall of the final negotiated numbers/names.

Three arms:
  graft     GraftAgent (focus branches for negotiation & pivot, detached branch for calendar)
  subagent  SubagentRunner (parent mediates and delegates via paraphrased task_prompt + summary)
  shared    Single shared agent holding all capability skills at once (tests style leakage + tokens)

Usage:
  uv run python experiments/negotiate_pivot.py --fake              # offline dry run
  uv run python experiments/negotiate_pivot.py --arm all -n 5      # live comparison
  uv run python experiments/negotiate_pivot.py --arm graft --trace # inspect graft internals
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft import cachestats  # noqa: E402
from graft.agent import GraftAgent, GraftConfig  # noqa: E402
from graft.db import Store  # noqa: E402
from graft.fake import FakeTools, _clean, _last_block, _row_ids, _text, _use, _user_texts, _wake_reply, woken  # noqa: E402
from graft.providers import FakeProvider, Provider, make_provider, prefix_nonce  # noqa: E402
from graft.sim import run_script  # noqa: E402
from graft.skills import BASE_SYSTEM, BUILTIN, CATALOG, LOAD_SKILL_TOOL, install, load_skill, skill_listing  # noqa: E402
from graft.subagent import SubagentRunner  # noqa: E402
from graft.tools import CallToolResult, by_name  # noqa: E402

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")

TURN_1 = "Hi! I want to build a website for Riverbend Dental in Columbus, Ohio. Our lead dentists are Dr. Elena Vance and Dr. Priya Patel."
TURN_2 = "Our grand opening is May 12, we're running a $199 teeth whitening promo, and our three services are cleanings, dental implants, and teeth whitening."
TURN_3 = "Quick revisions before you write anything: move the grand opening to May 19, and drop Dr. Patel — she is leaving the practice, so Dr. Elena Vance is our sole lead dentist."
TURN_4 = "Also lower the whitening promo to $149, and book our open-house walkthrough on the calendar for 2026-05-18 at 14:30."
TURN_5 = "Great, go ahead and write the home page for the website now — keep it casual and fun with an emoji on every heading."
TURN_6 = "Now build an investor pitch slide deck for Riverbend Dental using all the finalized details we settled on. Do not include any outdated dates, prices, or dentists."
TURN_8 = "Before we send the deck, confirm: what are our final grand opening date, whitening promo price, and lead dentist?"


def nudge_deck(world: FakeTools) -> str | None:
    return None if world.decks and world.decks[-1]["slides"] else (
        "Please go ahead and create the finalized investor slide deck now with what we agreed on."
    )


SCRIPT = [TURN_1, TURN_2, TURN_3, TURN_4, TURN_5, TURN_6, nudge_deck, TURN_8]


# ------------------------------------------------------------------ shared-context baseline arm
class SharedAgent:
    """Single agent holding all capability skills in one context (no delegation)."""

    def __init__(self, store: Store, provider: Provider, tools, persona: str = BASE_SYSTEM,
                 nonce: str | None = None, max_steps: int = 10, trace=None):
        self.store = store
        self.provider = provider
        self.tools_map = by_name(list(tools))
        self.persona = persona
        self.nonce = nonce
        self.max_steps = max_steps
        self.trace = trace or (lambda s: None)
        self.messages: list[dict] = []
        self.caps = [s.name for s in CATALOG if s.name not in BUILTIN]

    def system(self) -> str:
        return "\n\n".join([
            self.persona,
            skill_listing(self.store, self.caps,
                          "Your skills (call load_skill(name) to read a skill's full instructions before using it):"),
        ])

    def tool_defs(self) -> list[dict]:
        return [LOAD_SKILL_TOOL] + [t.definition() for t in self.tools_map.values()]

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send(self, request: dict):
        from graft import a2a

        text = request["message"]["parts"][0]["text"]
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        said: list[str] = []
        tools = self.tool_defs()
        allowed = {t["name"] for t in tools}
        for _ in range(self.max_steps):
            tagger = getattr(self.provider, "usage_tag", None)
            with tagger("shared") if tagger else contextlib.nullcontext():
                blocks = self.provider.complete(self.system(), prefix_nonce(self.messages, self.nonce), tools)
            uses = [b for b in blocks if b["type"] == "tool_use"]
            self.messages.append({"role": "assistant",
                                  "content": [b for b in blocks if b["type"] in ("text", "tool_use")]
                                  or [{"type": "text", "text": "(empty)"}]})
            said += [b["text"] for b in blocks if b["type"] == "text"]
            if not uses:
                break
            results = []
            for u in uses:
                if u["name"] not in allowed:
                    content, err = f"tool '{u['name']}' is not available", True
                elif u["name"] == "load_skill":
                    content, err = load_skill(self.store, set(self.caps), str(u["input"].get("name", "")))
                else:
                    outcome = self.tools_map[u["name"]].fn(u["input"])
                    res = CallToolResult.from_value(outcome)
                    content, err = (res.text(), res.is_error) if res else (str(outcome), False)
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": content, "is_error": err})
            self.messages.append({"role": "user", "content": results})
        tid = f"shared-{len(self.messages)}"
        user_msg = a2a.Message(message_id=request["message"]["messageId"], role=a2a.ROLE_USER,
                               parts=[a2a.Part(text=text)], context_id="shared", task_id=tid)
        history = [user_msg] + [
            a2a.Message(message_id=f"{tid}-r{i}", role=a2a.ROLE_AGENT, parts=[a2a.Part(text=t)],
                        context_id="shared", task_id=tid)
            for i, t in enumerate(said)
        ]
        task = a2a.Task(id=tid, context_id="shared",
                        status=a2a.TaskStatus(state=a2a.COMPLETED, message=history[-1] if history else None),
                        history=history)
        return a2a.SendMessageResponse(task=task)


# ------------------------------------------------------------------ offline fake policies
def _fake_policy(arm: str):
    """Scripted offline policy for smoke-testing all three arms without API keys."""
    def policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        texts = [_clean(t) for t in _user_texts(messages)]
        latest = texts[-1] if texts else ""
        joined = " ".join(texts)

        if arm == "graft" and "fork" in names:
            if woken(messages):
                return _wake_reply(messages)
            if last["type"] == "tool_result":
                return _text("Started on that.")
            if "confirm:" in latest.lower():
                return _text("Finalized: grand opening May 19, whitening promo $149, lead dentist Dr. Elena Vance.")
            if "slide deck" in latest.lower() or "investor" in latest.lower():
                return _use(tool="fork", current={"skills": ["slides"], "messages": _row_ids(messages),
                                                  "reason": "Build the formal investor pitch deck with finalized details."})
            if "calendar" in latest.lower():
                return _use(tool="fork",
                            current={"skills": ["website"], "messages": _row_ids(messages),
                                     "reason": "Note the $149 whitening promo for the website."},
                            detached=[{"skills": ["calendar"], "messages": _row_ids(messages),
                                       "reason": "Book the open-house walkthrough on 2026-05-18 at 14:30."}])
            return _use(tool="fork", current={"skills": ["website"], "messages": _row_ids(messages),
                                              "reason": "Work on the Riverbend Dental website."})

        if arm == "subagent" and "delegate" in names:
            if last["type"] == "tool_result":
                return _text("Done: " + str(last["content"])[:200])
            if "confirm:" in latest.lower():
                return _text("Finalized: grand opening May 19, whitening promo $149, lead dentist Dr. Elena Vance.")
            if "calendar" in latest.lower():
                return _use("delegate", skill="calendar",
                            task_prompt="Book open-house walkthrough on 2026-05-18 at 14:30.")
            if "write the home page" in latest.lower():
                return _use("delegate", skill="website",
                            task_prompt="Write Riverbend Dental home page (Columbus, Ohio, Dr. Elena Vance, May 19, $149 whitening, cleanings, implants, whitening, casual with emoji).")
            if "slide deck" in latest.lower() or "investor" in latest.lower():
                return _use("delegate", skill="slides",
                            task_prompt="Build formal investor deck for Riverbend Dental in Columbus, Ohio, Dr. Elena Vance, grand opening May 19, $149 whitening, cleanings, implants, whitening.")
            return _text("Got it, noted those details.")

        # Branch / subagent / shared execution
        prev_tool = next((b["name"] for m in reversed(messages) if m["role"] == "assistant"
                          for b in m["content"] if b["type"] == "tool_use"), None)
        if last["type"] == "tool_result" and prev_tool not in ("load_skill", "fork"):
            return _text("Completed your request.")
        if "confirm:" in latest.lower():
            return _text("Finalized: grand opening May 19, whitening promo $149, lead dentist Dr. Elena Vance.")
        if "calendar_book" in names and "2026-05-18" in joined and ("calendar" in latest.lower() or "walkthrough" in latest.lower()):
            return _use("calendar_book", date="2026-05-18", time="14:30", what="open-house walkthrough")
        if "slides_create" in names and ("slide" in latest.lower() or "investor" in latest.lower()):
            return [
                *_use("slides_create", title="Riverbend Dental Investor Overview",
                      audience="Investors — Riverbend Dental, Columbus, Ohio, Dr. Elena Vance", tone="formal"),
                *_use("slides_add_slide", title="Practice Overview",
                      bullets=["Located in Columbus, Ohio led by Dr. Elena Vance",
                               "Grand opening scheduled for May 19",
                               "Whitening promotion priced at $149"]),
                *_use("slides_add_slide", title="Core Services",
                      bullets=["Preventive dental cleanings",
                               "Full-service dental implants",
                               "Professional teeth whitening"]),
            ]
        if "site_write" in names and "home page" in latest.lower():
            return _use("site_write", page="home",
                        content="# Welcome to Riverbend Dental 🦷\nColumbus, Ohio — Dr. Elena Vance, opening May 19, $149 whitening!")
        return _text("Noted — ready for the next update.")

    return policy


# ------------------------------------------------------------------ scoring
def score_run(world: FakeTools, transcript: list[tuple[str, str]], store: Store | None = None) -> dict:
    deck_json = world.deck_text()
    deck_low = deck_json.lower()
    latest_deck = world.decks[-1] if world.decks else {}
    slides = latest_deck.get("slides", [])

    # 1. Finalized revised facts present in deck
    has_may19 = "may 19" in deck_low or "05-19" in deck_low or "5/19" in deck_low
    has_149 = "149" in deck_low
    has_vance = "vance" in deck_low
    has_location = "columbus" in deck_low or "ohio" in deck_low
    services = sum(k in deck_low for k in ("clean", "implant", "whiten"))
    revised_ok = has_may19 and has_149 and has_vance and has_location and services >= 2

    # 2. Retracted / stale facts absent from deck
    no_may12 = "may 12" not in deck_low and "05-12" not in deck_low
    no_199 = "199" not in deck_low
    no_patel = "patel" not in deck_low
    no_stale_ok = no_may12 and no_199 and no_patel

    # 3. Formal style compliance on slides (zero emoji, zero '!', zero hashtags, 3 bullets/slide)
    slide_text = json.dumps(slides)
    no_emoji = not EMOJI_RE.search(deck_json)
    no_bang = "!" not in slide_text
    no_hashtag = "#" not in slide_text
    bullets_ok = bool(slides) and all(len(s.get("bullets", [])) == 3 for s in slides)
    formal_ok = bool(world.decks) and no_emoji and no_bang and no_hashtag and bullets_ok

    # 4. Website written with casual emoji + calendar booked
    site_written = bool(world.site)
    site_emoji = bool(EMOJI_RE.search(" ".join(world.site.values())))
    cal_ok = any("2026-05-18" in str(c.get("date", "")) and "14:30" in str(c.get("time", "")) for c in world.calendar)

    # 5. Final probe answer accuracy
    probe_reply = transcript[-1][1].lower() if transcript and transcript[-1][0] == "assistant" else ""
    probe_ok = ("may 19" in probe_reply and "149" in probe_reply and "vance" in probe_reply
                and "may 12" not in probe_reply and "199" not in probe_reply and "patel" not in probe_reply)

    fork_refusals = 0
    if store is not None:
        fork_refusals = store.con.execute(
            "SELECT count(*) FROM transactions WHERE kind='branch' AND status='rolled_back'"
        ).fetchone()[0]

    checks = {
        "revised_facts_in_deck": revised_ok,
        "no_stale_facts_in_deck": no_stale_ok,
        "slides_formal_style": formal_ok,
        "website_written": site_written,
        "website_emoji": site_emoji,
        "calendar_booked": cal_ok,
        "probe_accurate": probe_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fork_refusals": fork_refusals,
        "detail": {
            "has_may19": has_may19, "has_149": has_149, "has_vance": has_vance,
            "no_may12": no_may12, "no_199": no_199, "no_patel": no_patel,
            "no_emoji": no_emoji, "no_bang": no_bang, "bullets_ok": bullets_ok,
        },
    }


async def run_arm(arm: str, *, fake: bool, trace: bool, cache_scope: str) -> dict:
    db_path = os.path.join(tempfile.mkdtemp(prefix=f"graft-neg-{arm}-"), f"{arm}.db")
    store = Store(db_path)
    world = FakeTools()
    provider = FakeProvider(_fake_policy(arm)) if fake else make_provider()
    nonce = uuid.uuid4().hex[:12] if cache_scope == "run" else None
    t0 = time.perf_counter()

    if arm == "graft":
        agent = GraftAgent(store, provider, tools=world.tools(),
                           config=GraftConfig(nonce=nonce, trace=print if trace else None))
        async with agent:
            transcript = await run_script(agent, SCRIPT, echo=trace, world=world)
    elif arm == "subagent":
        install(store, tools=world.definitions())
        agent = SubagentRunner(store=store, provider=provider, tools=world.tools(),
                               nonce=nonce, trace=print if trace else None)
        async with agent:
            transcript = await run_script(agent, SCRIPT, echo=trace, world=world)
    elif arm == "shared":
        install(store, tools=world.definitions())
        agent = SharedAgent(store=store, provider=provider, tools=world.tools(),
                            nonce=nonce, trace=print if trace else None)
        transcript = await run_script(agent, SCRIPT, echo=trace, world=world)
    else:
        raise ValueError(f"unknown arm: {arm}")

    elapsed = time.perf_counter() - t0
    result = score_run(world, transcript, store=store if arm == "graft" else None)
    result["arm"] = arm
    result["elapsed_s"] = round(elapsed, 2)
    result["usage"] = dict(getattr(provider, "usage", {}))
    result["by_tag"] = dict(getattr(provider, "by_tag", {}))
    result["provider"] = f"{provider.name}:{provider.model}"
    return result


def print_report(results: list[dict]) -> None:
    print("\n================ NEGOTIATE + REVISE + PIVOT REPORT ================")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        u = r["usage"]
        print(f"\n[{status}] arm={r['arm']} ({r['provider']})  {r['elapsed_s']}s  "
              f"calls={u.get('calls', 0)}  in={u.get('input', 0)} (cached={u.get('cached', 0)})  "
              f"out={u.get('output', 0)}  fork_refusals={r['fork_refusals']}")
        for k, ok in r["checks"].items():
            print(f"    [{'ok' if ok else 'FAIL'}] {k}")
        if not r["passed"]:
            print(f"    detail: {r['detail']}")
        if r["by_tag"]:
            for tag, b in sorted(r["by_tag"].items()):
                print(f"    role={tag:<18} calls={b['calls']:<3} in={b['input']:<7} "
                      f"cached={b['cached']:<7} out={b['output']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-turn negotiation + revision + pivot benchmark")
    p.add_argument("--arm", choices=("graft", "subagent", "shared", "all"), default="all")
    p.add_argument("--fake", action="store_true", help="offline dry run with scripted FakeProvider")
    p.add_argument("--trace", action="store_true", help="print conversation & internal trace")
    p.add_argument("--cache-scope", choices=("run", "global"), default="run")
    p.add_argument("-n", type=int, default=1, help="repetitions per arm")
    args = p.parse_args()

    arms = ("graft", "subagent", "shared") if args.arm == "all" else (args.arm,)

    async def go():
        results = []
        for _ in range(args.n):
            for arm in arms:
                results.append(await run_arm(arm, fake=args.fake, trace=args.trace, cache_scope=args.cache_scope))
        print_report(results)
        return all(r["passed"] for r in results)

    ok = asyncio.run(go())
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
