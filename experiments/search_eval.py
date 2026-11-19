"""Live evaluation of the shipped `history` skill (`search(query)`): over-use vs. recovery.

Tests both failure modes on a real model:
1. OVER-USE (Turns 1-4): turns where everything needed is already in context or comes from a single
   fresh tool call. Measures:
   - `unnecessary_search_calls`: how many times `search()` was called during Turns 1-4
   - `unnecessary_history_grants`: how many branches in Turns 1-4 were unnecessarily granted `history`
2. RECOVERY (Turns 5-6): turns that ask for specific details buried inside earlier merged branches'
   dropped `tool_result` dumps (never included in those branches' 1-sentence merged replies):
   - Turn 5: exact pod name (`checkout-7d9f4c-x2k9`) and retry count (`retries=3`) from Turn 1's
     18 KB log dump — recoverable via `search("ERR-8492")` on main without re-calling `logs_fetch`.
   - Turn 6: update `status` website page (`site_write`) including the buried `pod` name AND the
     buried `cpu_pct` (`39.2`) from Turn 3's 16 KB metrics dump — tests granting `history` to a
     `website` branch.

Usage:
  uv run python experiments/history_search_eval.py --fake
  uv run python experiments/history_search_eval.py -n 2 --trace
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft.agent import GraftAgent, GraftConfig  # noqa: E402
from graft.db import Store  # noqa: E402
from graft.fake import _last_block, _row_ids, _text, _use, _user_texts, _wake_reply, woken  # noqa: E402
from graft.providers import FakeProvider, make_provider  # noqa: E402
from graft.sim import say  # noqa: E402

# Reuse the realistic ops catalog & ~16-18 KB dump generators from long_horizon.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_horizon import OPS_SKILLS, make_ops_tools  # noqa: E402

SCRIPT = [
    # Turn 1 (control): fresh logs fetch -> short 1-sentence summary (drops pod & retries from main)
    "Fetch the `checkout` service logs with logs_fetch and reply in ONE short sentence with ONLY the error code and failing dependency.",
    # Turn 2 (control — already in context!): should NOT call search or fork
    "What error code and failing dependency did you just report?",
    # Turn 3 (control — fresh tool call): should fork `metrics` ONLY (NO history grant, NO search)
    "Pull the `checkout` metrics with metrics_dump and reply in ONE short sentence with ONLY the spiking region and its error rate.",
    # Turn 4 (control — already in context!): should fork `tickets` ONLY (NO history grant, NO search)
    "Create a P1 incident ticket with ticket_create for the ERR-8492 checkout spike in us-east-2 (no need to list tickets first).",
    # Turn 5 (NEEDS search!): pod name & retries were ONLY in Turn 1's dropped 18 KB log dump
    "Without re-fetching logs, what exact pod name and retry count appeared on that ERR-8492 error line earlier?",
    # Turn 6 (NEEDS history grant to website branch!): cpu_pct of the us-east-2 spike was ONLY in Turn 3's dropped 16 KB metrics dump
    "Update our `status` page using site_write with a brief note mentioning the exact failing pod name and the exact `cpu_pct` recorded during the us-east-2 spike.",
]


def _fake_policy():
    def policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        texts = _user_texts(messages)
        cur = texts[-1] if texts else ""
        if "fork" in names:
            if woken(messages):
                return _wake_reply(messages)
            ids = _row_ids(messages)
            if last["type"] == "tool_result":
                if "checkout-7d9f4c-x2k9" in str(last.get("content", "")):
                    return _text("The failing pod was checkout-7d9f4c-x2k9 with retries=3.")
                return _text("Done.")
            if "logs_fetch" in cur:
                return _use(tool="fork", current={"skills": ["logs"], "messages": ids[-1:], "reason": cur})
            if "just report" in cur:
                return _text("ERR-8492 from upstream-auth-v3.")
            if "metrics_dump" in cur:
                return _use(tool="fork", current={"skills": ["metrics"], "messages": ids[-1:], "reason": cur})
            if "ticket_create" in cur:
                return _use(tool="fork", current={"skills": ["tickets"], "messages": ids[-1:], "reason": cur})
            if "exact pod name" in cur:
                return _use("search", query="ERR-8492")
            if "site_write" in cur:
                return _use(tool="fork", current={"skills": ["website", "search"], "messages": ids[-2:], "reason": cur})
            return _text("ok")
        prev = next((b["name"] for m in reversed(messages) if m["role"] == "assistant"
                     for b in m["content"] if b["type"] == "tool_use"), None)
        if last["type"] == "tool_result" and prev != "fork":
            if prev == "logs_fetch":
                return _text("ERR-8492 from upstream-auth-v3.")
            if prev == "metrics_dump":
                return _text("us-east-2 spiked to 14.7%.")
            if prev == "search" and "site_write" in names:
                return _use("site_write", page="status",
                            content="Investigating checkout-7d9f4c-x2k9 in us-east-2 (cpu_pct 39.2).")
            return _text("Done.")
        if "logs_fetch" in names:
            return _use("logs_fetch", service="checkout")
        if "metrics_dump" in names:
            return _use("metrics_dump", service="checkout")
        if "ticket_create" in names:
            return _use("ticket_create", title="Checkout ERR-8492", priority="P1", body="ERR-8492 in us-east-2")
        if "site_write" in names and "search" in names:
            return _use("search", query="us-east-2 14.7")
        return _text("Done.")
    return policy


async def run_trial(*, fake: bool, trace: bool) -> dict:
    db_path = os.path.join(tempfile.mkdtemp(prefix="graft-hist-"), "hist.db")
    store = Store(db_path)
    tools, state = make_ops_tools()
    provider = FakeProvider(_fake_policy()) if fake else make_provider()
    nonce = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    replies: list[str] = []
    turn_snapshots: list[dict] = []

    agent = GraftAgent(
        store, provider, tools=tools,
        config=GraftConfig(nonce=nonce, seeder=None, skills=OPS_SKILLS, trace=print if trace else None),
    )
    async with agent:
        for i, prompt in enumerate(SCRIPT, 1):
            if trace:
                print(f"\nUSER: {prompt}")
            _, turn_replies = await say(agent, prompt)
            reply = " ".join(turn_replies)
            if trace and reply:
                print(f"ASSISTANT: {reply}")
            replies.append(reply)
            searches = store.con.execute(
                "SELECT owner_timeline, content FROM turns WHERE kind='tool_use' AND tool_name='search' ORDER BY row_id"
            ).fetchall()
            hist_branches = [
                r[0] for r in store.con.execute(
                    "SELECT tl.timeline_id FROM timelines tl "
                    "JOIN timeline_skills ts ON ts.timeline_id = tl.timeline_id "
                    "JOIN skills s ON s.skill_id = ts.skill_id "
                    "WHERE tl.source_id IS NOT NULL AND s.name = 'search'"
                )
            ]
            turn_snapshots.append({
                "turn": i,
                "total_searches": len(searches),
                "history_branches": list(hist_branches),
                "logs_fetch_calls": sum(1 for n, _ in state["log"] if n == "logs_fetch"),
                "metrics_dump_calls": sum(1 for n, _ in state["log"] if n == "metrics_dump"),
            })

    elapsed = round(time.perf_counter() - t0, 2)
    # Control turns = Turns 1-4
    ctrl_searches = turn_snapshots[3]["total_searches"]
    ctrl_hist_grants = len(turn_snapshots[3]["history_branches"])
    # Recovery turns = Turns 5-6
    rec_searches = turn_snapshots[5]["total_searches"] - ctrl_searches
    t5_reply = replies[4].lower()
    t6_site = json.dumps(state["writes"]).lower()

    # Exact cpu_pct for i=121 in long_horizon._make_metrics_dump: 38.2 + (121 % 12) = 39.2
    checks = {
        "zero_unnecessary_searches_t1_t4": ctrl_searches == 0,
        "zero_unnecessary_history_grants_t1_t4": ctrl_hist_grants == 0,
        "t5_recovered_pod_and_retries": "checkout-7d9f4c-x2k9" in t5_reply and "3" in t5_reply,
        "t5_did_not_refetch_logs": turn_snapshots[4]["logs_fetch_calls"] == 1,
        "t6_wrote_pod_and_cpu_to_status": "checkout-7d9f4c-x2k9" in t6_site and "39.2" in t6_site,
        "t6_did_not_refetch_metrics": turn_snapshots[5]["metrics_dump_calls"] == 1,
    }
    all_searches = [
        (r[0], json.loads(r[1])["input"])
        for r in store.con.execute(
            "SELECT owner_timeline, content FROM turns WHERE kind='tool_use' AND tool_name='search' ORDER BY row_id"
        )
    ]
    return {
        "provider": f"{provider.name}:{provider.model}",
        "elapsed_s": elapsed,
        "usage": dict(getattr(provider, "usage", {})),
        "ctrl_searches": ctrl_searches,
        "ctrl_hist_grants": ctrl_hist_grants,
        "rec_searches": rec_searches,
        "all_searches": all_searches,
        "history_branches": turn_snapshots[5]["history_branches"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "t5_reply": replies[4],
        "writes": state["writes"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Live eval of history search skill: over-use vs recovery")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("-n", type=int, default=1)
    args = p.parse_args()

    async def go():
        for i in range(1, args.n + 1):
            r = await run_trial(fake=args.fake, trace=args.trace)
            u = r["usage"]
            verdict = "PASS" if r["passed"] == r["total"] else f"PARTIAL ({r['passed']}/{r['total']})"
            print(f"\n================ HISTORY SEARCH EVAL (trial {i}) ================")
            print(f"[{verdict}] ({r['provider']}) {r['elapsed_s']}s calls={u.get('calls', 0)} "
                  f"in={u.get('input', 0):,} out={u.get('output', 0):,}")
            print(f"    Control turns 1-4: search_calls={r['ctrl_searches']}  history_grants={r['ctrl_hist_grants']}")
            print(f"    Recovery turns 5-6: search_calls={r['rec_searches']}  history_branches={r['history_branches']}")
            print(f"    All search calls: {r['all_searches']}")
            for k, ok in r["checks"].items():
                print(f"    [{'ok' if ok else 'MISS'}] {k}")
            if not r["checks"]["t5_recovered_pod_and_retries"]:
                print(f"    t5_reply: {r['t5_reply']}")
            if not r["checks"]["t6_wrote_pod_and_cpu_to_status"]:
                print(f"    writes: {r['writes']}")

    asyncio.run(go())


if __name__ == "__main__":
    main()
