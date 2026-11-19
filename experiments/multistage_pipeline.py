"""E2E live test of sequential multi-stage pipelines with `fork(current={..., wake?})`.

Verifies both halves of `current.wake` on a real model (`openai:gpt-6-luna`) in one session:
- Turn 1 (2-stage sequential pipeline in ONE user request):
  "Audit the `billing` service first, tell me what you find, and then patch the misconfigured flag
  with the recommended value."
  Stage 2 (`patch`) depends on Stage 1's (`audit`) output (`cache_ttl -> 3600`).
  Expected:
    1. `main` forks `audit` as `current` with `wake=True` (or omitted).
    2. `audit` calls `audit_scan`, replies plainly to the user (no explicit `merge()`), and merges.
    3. `main` WAKES automatically and forks `patch` as `current` with `wake=False`.
    4. `patch` calls `config_apply(flag="cache_ttl", value=3600)`, replies plainly, and merges
       quietly (no wasted wake).
- Turn 2 (single-stage control):
  "Now notify `#ops` using `team_notify` that `cache_ttl` was updated to 3600."
  Expected:
    5. `main` forks `notify` as `current` with `wake=False`; merges quietly (0 extra wakes).

Usage:
  uv run python experiments/multistage_pipeline.py --fake
  uv run python experiments/multistage_pipeline.py -n 3 --trace
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
from graft.db import Skill, Store  # noqa: E402
from graft.fake import _last_block, _row_ids, _text, _use, _user_texts, woken  # noqa: E402
from graft.providers import FakeProvider, make_provider  # noqa: E402
from graft.sim import say  # noqa: E402
from graft.skills import BASE_SYSTEM, BUILTIN, CATALOG  # noqa: E402
from graft.tools import Tool  # noqa: E402

S = {"type": "string"}
I = {"type": "integer"}


def _obj(props: dict, req: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": req}


PIPELINE_PERSONA = (
    BASE_SYSTEM
    + "\n\nWorkflow rule: when asked to audit and patch a service, run the two stages sequentially "
    "as separate current branches (do not combine `audit` and `patch` in one branch): first fork "
    "`audit` with `wake: true` so it reports what it found to the user and returns control to main, "
    "then after `audit` merges, fork `patch` with `wake: false` using the discovered flag and value. "
    "For single-stage requests (like `notify`), fork with `wake: false`."
)

TOOL_DEFS = [
    {"name": "audit_scan", "description": "Scan a service configuration for misconfigured flags.",
     "inputSchema": _obj({"service": S}, ["service"])},
    {"name": "config_apply", "description": "Apply a configuration value for a flag.",
     "inputSchema": _obj({"flag": S, "value": I}, ["flag", "value"])},
    {"name": "team_notify", "description": "Post a short update message to a team channel.",
     "inputSchema": _obj({"channel": S, "message": S}, ["channel", "message"])},
]

SKILLS = [
    *[s for s in CATALOG if s.name in BUILTIN],
    Skill("audit", "Scan a service for misconfigured flags (read-only).",
          "Audit skill: call audit_scan(service) and concisely report the misconfigured flag name, "
          "current value, and recommended value directly to the user.",
          ["audit_scan"]),
    Skill("patch", "Apply a configuration flag patch.",
          "Patch skill: call config_apply(flag, value) with the exact flag name and integer value, "
          "then confirm the applied change concisely to the user.",
          ["config_apply"]),
    Skill("notify", "Send a status notification to a team channel.",
          "Notify skill: call team_notify(channel, message) and confirm concisely to the user.",
          ["team_notify"]),
]

SCRIPT = [
    "Audit the `billing` service first, tell me what you find, and then patch the misconfigured flag with the recommended value.",
    "Great — now notify `#ops` with team_notify that `cache_ttl` was updated to 3600.",
]


def make_tools() -> tuple[list[Tool], dict]:
    state: dict = {"scans": [], "patches": [], "notifications": [], "order": []}

    def make_fn(name: str):
        def fn(args: dict):
            state["order"].append(name)
            if name == "audit_scan":
                state["scans"].append(dict(args))
                return json.dumps({
                    "service": args.get("service", "billing"),
                    "status": "misconfigured",
                    "flag": "cache_ttl",
                    "current_value": 0,
                    "recommended_value": 3600,
                    "owner": "platform-eng",
                })
            if name == "config_apply":
                state["patches"].append(dict(args))
                return f"applied {args.get('flag')}={args.get('value')} (status=ok)"
            if name == "team_notify":
                state["notifications"].append(dict(args))
                return f"posted to {args.get('channel')}"
            return f"ok:{name}"
        return fn

    return [Tool(name=d["name"], description=d["description"], input_schema=d["inputSchema"], fn=make_fn(d["name"]))
            for d in TOOL_DEFS], state


def _fake_policy():
    def policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        texts = _user_texts(messages)
        if "fork" in names:
            ids = _row_ids(messages)
            if woken(messages):
                if "audit" in str(last.get("content", "")) and "cache_ttl" in json.dumps(messages):
                    return _use(tool="fork", current={
                        "id": "patch-billing", "skills": ["patch"], "messages": ids[-2:],
                        "reason": "Apply cache_ttl=3600 from audit.", "wake": False,
                    })
                return []
            cur = texts[-1] if texts else ""
            if "Audit" in cur:
                return _use(tool="fork", current={
                    "id": "audit-billing", "skills": ["audit"], "messages": ids[-1:],
                    "reason": "Audit billing service.", "wake": True,
                })
            if "notify" in cur:
                return _use(tool="fork", current={
                    "id": "notify-ops", "skills": ["notify"], "messages": ids[-1:],
                    "reason": "Notify #ops.", "wake": False,
                })
            return _text("Done.")
        prev = next((b["name"] for m in reversed(messages) if m["role"] == "assistant"
                     for b in m["content"] if b["type"] == "tool_use"), None)
        if last["type"] == "tool_result" and prev != "fork":
            if prev == "audit_scan":
                return _text("Audit found `cache_ttl` is set to 0 (recommended: 3600).")
            if prev == "config_apply":
                return _text("Applied `cache_ttl=3600` cleanly.")
            return _text("Notified `#ops`.")
        if "audit_scan" in names:
            return _use("audit_scan", service="billing")
        if "config_apply" in names:
            return _use("config_apply", flag="cache_ttl", value=3600)
        if "team_notify" in names:
            return _use("team_notify", channel="#ops", message="Updated cache_ttl to 3600.")
        return _text("Done.")
    return policy


async def run_trial(*, fake: bool, trace: bool) -> dict:
    db_path = os.path.join(tempfile.mkdtemp(prefix="graft-stage-"), "stage.db")
    store = Store(db_path)
    tools, state = make_tools()
    provider = FakeProvider(_fake_policy()) if fake else make_provider()
    nonce = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    replies: list[list[str]] = []
    agent = GraftAgent(
        store, provider, tools=tools,
        config=GraftConfig(
            persona=PIPELINE_PERSONA, nonce=nonce, seeder=None, skills=SKILLS,
            trace=print if trace else None,
        ),
    )
    async with agent:
        for prompt in SCRIPT:
            if trace:
                print(f"\nUSER: {prompt}")
            _, turn_replies = await say(agent, prompt)
            if trace:
                for r in turn_replies:
                    print(f"ASSISTANT: {r}")
            replies.append(turn_replies)
        wakes = list(agent._host.wakes)

    elapsed = round(time.perf_counter() - t0, 2)
    # Inspect exact fork calls made by main (skills and wake flag per branch)
    forks = []
    for (content,) in store.con.execute(
        "SELECT content FROM turns WHERE owner_timeline='o' AND kind='tool_use' AND tool_name='fork' ORDER BY row_id"
    ):
        inp = json.loads(content)["input"]
        cur = inp.get("current") or {}
        forks.append({"id": cur.get("id"), "skills": cur.get("skills"), "wake": cur.get("wake", True)})

    # Check whether stage 1 (audit) ended in an implicit merge (plain reply, harness=1)
    audit_tl = next(
        (r[0] for r in store.con.execute(
            "SELECT tl.timeline_id FROM timelines tl "
            "JOIN timeline_skills ts ON ts.timeline_id = tl.timeline_id "
            "JOIN skills s ON s.skill_id = ts.skill_id WHERE s.name='audit' ORDER BY tl.rowid LIMIT 1"
        )),
        None,
    )
    audit_plain_reply = False
    if audit_tl:
        merge_row = store.con.execute(
            "SELECT harness FROM turns WHERE owner_timeline=? AND kind='tool_use' AND tool_name='merge' "
            "ORDER BY row_id DESC LIMIT 1",
            (audit_tl,),
        ).fetchone()
        audit_plain_reply = bool(merge_row and merge_row[0] == 1)

    patch_ok = any("cache_ttl" in str(p.get("flag", "")) and int(p.get("value", -1)) == 3600 for p in state["patches"])
    notify_ok = any("ops" in str(n.get("channel", "")).lower() and "3600" in str(n.get("message", ""))
                    for n in state["notifications"])

    checks = {
        "sequential_tool_order": state["order"] == ["audit_scan", "config_apply", "team_notify"],
        "separate_branches_for_stages": len(forks) == 3 and forks[0]["skills"] == ["audit"] and forks[1]["skills"] == ["patch"],
        "stage1_ended_with_plain_reply": audit_plain_reply,
        "stage1_woke_main_for_stage2": forks[0]["wake"] is True and len(wakes) == 1,
        "stage2_patched_discovered_flag": patch_ok,
        "terminal_stages_skipped_wake": len(forks) == 3 and forks[1]["wake"] is False and forks[2]["wake"] is False,
        "turn2_notified_ops": notify_ok,
    }
    return {
        "provider": f"{provider.name}:{provider.model}",
        "elapsed_s": elapsed,
        "usage": dict(getattr(provider, "usage", {})),
        "forks": forks,
        "wakes": wakes,
        "order": state["order"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="E2E sequential multi-stage pipeline test for current.wake")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("-n", type=int, default=1)
    args = p.parse_args()

    async def go():
        for i in range(1, args.n + 1):
            r = await run_trial(fake=args.fake, trace=args.trace)
            u = r["usage"]
            verdict = "PASS" if r["passed"] == r["total"] else f"FAIL ({r['passed']}/{r['total']})"
            print(f"\n================ MULTISTAGE PIPELINE (trial {i}) ================")
            print(f"[{verdict}] ({r['provider']}) {r['elapsed_s']}s calls={u.get('calls', 0)} "
                  f"in={u.get('input', 0):,} out={u.get('output', 0):,} wakes={len(r['wakes'])}")
            print(f"    forks: {r['forks']}")
            print(f"    tool_order: {r['order']}")
            for k, ok in r["checks"].items():
                print(f"    [{'ok' if ok else 'MISS'}] {k}")

    asyncio.run(go())


if __name__ == "__main__":
    main()
