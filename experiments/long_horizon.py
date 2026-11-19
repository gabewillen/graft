"""Long-horizon heavy-tool-output benchmark: `graft` vs `shared`.

Measures the real crossover point where a single shared context bloats with noisy tool outputs
(server logs, metrics dumps, source diffs, runbooks, ticket backlogs, billing ledgers, IAM logs,
HTML history, calendar feeds) across a 10-turn multi-domain operations session:
- In `shared`, all 24 tools live in one context and every ~15-18 KB raw tool dump accumulates in
  history for every subsequent turn (~35-45K tokens by Turn 10).
- In `graft`, each branch holds only the 1 skill it needs (~2 tools), reads the ~16 KB dump inside
  the branch, and merges ONLY its clean summary back to `main` — dropping the raw tool dump.
- Turn 10 tests cross-domain synthesis: drafting an internal postmortem email combining all 8
  findings from Turns 1-9.

To keep the comparison maximally fair to `shared`, `SharedAgent` also has all skill instructions
inlined in its system prompt (zero `load_skill` round-trip penalty on either arm).

Usage:
  uv run python experiments/long_horizon.py --fake          # offline smoke test
  uv run python experiments/long_horizon.py --arm all       # live comparison (gpt-6-luna)
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

from graft import a2a  # noqa: E402
from graft.agent import GraftAgent, GraftConfig  # noqa: E402
from graft.db import Skill, Store  # noqa: E402
from graft.fake import _last_block, _row_ids, _text, _use, _user_texts, _wake_reply, woken  # noqa: E402
from graft.providers import FakeProvider, Provider, make_provider, prefix_nonce  # noqa: E402
from graft.sim import run_script  # noqa: E402
from graft.skills import BASE_SYSTEM, BUILTIN, CATALOG, install, skill_instructions  # noqa: E402
from graft.tools import CallToolResult, Tool, by_name  # noqa: E402

S = {"type": "string"}
N = {"type": "number"}


def _obj(props: dict, req: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": req}


# ------------------------------------------------------------------ Ground-truth buried needles
NEEDLES = {
    "error_code": "ERR-8492",
    "dependency": "upstream-auth-v3",
    "latency": "1480ms",
    "pod": "checkout-7d9f4c-x2k9",
    "region": "us-east-2",
    "error_rate": "14.7%",
    "cpu_pct": "39.2",
    "commit": "9a4f2c1",
    "bad_timeout": "50",
    "good_timeout": "5000",
    "rollback_cmd": "deployctl rollback --rev 418",
    "channel": "#sev1-platform",
    "account": "acct_8821",
    "amount": "412.50",
    "stale_key": "key_legacy_09",
}


# ------------------------------------------------------------------ Realistic ~12-18 KB payloads
def _make_logs_dump() -> str:
    lines = []
    for i in range(180):
        ts = f"2026-11-19T02:{i // 60:02d}:{i % 60:02d}.{100 + (i * 7) % 899:03d}Z"
        if i == 113:
            lines.append(
                f"{ts} ERROR [checkout-api] code={NEEDLES['error_code']} dep={NEEDLES['dependency']} "
                f"msg=\"token verification deadline exceeded\" latency_p99={NEEDLES['latency']} "
                f"pod=checkout-7d9f4c-x2k9 retries=3 status=503"
            )
        else:
            lines.append(
                f"{ts} INFO  [checkout-api] req_id=req_{10000 + i} route=/v2/cart/price "
                f"pod=checkout-7d9f4c-m{i % 9} latency_ms={18 + (i * 3) % 45} status=200 cache=hit"
            )
    return "\n".join(lines)


def _make_metrics_dump() -> str:
    rows = []
    regions = ("us-east-1", "us-east-2", "us-west-2", "eu-west-1", "ap-south-1")
    for i in range(175):
        reg = regions[i % len(regions)]
        err = 14.7 if (reg == NEEDLES["region"] and i == 121) else round(0.08 + (i % 5) * 0.03, 2)
        p99 = 1480 if (reg == NEEDLES["region"] and i == 121) else 42 + (i % 19)
        rows.append(json.dumps({
            "ts": f"2026-11-19T02:{i // 60:02d}:{i % 60:02d}Z",
            "service": "checkout",
            "region": reg,
            "rps": 1420 + (i * 11) % 300,
            "error_rate_pct": err,
            "p50_ms": 14 + (i % 7),
            "p99_ms": p99,
            "cpu_pct": 38.2 + (i % 12),
        }))
    return "\n".join(rows)


def _make_code_dump() -> str:
    lines = ["# services/checkout/auth_client.py + recent git log", ""]
    for i in range(110):
        lines.append(
            f"def _helper_check_{i:03d}(token: str, region: str) -> dict: "
            f"return {{'ok': bool(token), 'region': region, 'seq': {i}}}"
        )
    lines.extend([
        "",
        "--- recent commits ---",
        "commit 3c1e8b0  chore(checkout): bump prometheus histogram buckets",
        f"commit {NEEDLES['commit']}  perf(checkout): tighten auth client timeout",
        "--- diff for 9a4f2c1 ---",
        "@@ -412,7 +412,7 @@ class AuthClient:",
        "     def __init__(self, endpoint: str):",
        "         self.endpoint = endpoint",
        f"-        self.timeout_ms = {NEEDLES['good_timeout']}  # 5s default for upstream-auth-v3",
        f"+        self.timeout_ms = {NEEDLES['bad_timeout']}  # tightened aggressively",
        "         self.max_retries = 3",
    ])
    for i in range(60):
        lines.append(f"# trailing compatibility shim note {i}: keep legacy header X-Checkout-Trace-{i}")
    return "\n".join(lines)


def _make_runbook_dump() -> str:
    sections = ["# Platform Operations Runbook (v4.12)", ""]
    for i in range(75):
        if i == 47:
            sections.append(
                f"## Section 47: Checkout Sev-1 Escalation & Rollback\n"
                f"When checkout 503 error rates exceed 5%, immediately notify {NEEDLES['channel']} "
                f"and execute the verified rollback command: `{NEEDLES['rollback_cmd']}`.\n"
                f"Verify error rate drops below 0.5% within 3 minutes before closing the bridge.\n"
            )
        else:
            sections.append(
                f"## Section {i}: Routine Maintenance Procedure {i}\n"
                f"Run standard health verification `opsctl check --suite tier{i % 4} --cluster prod-{i % 6}` "
                f"and archive diagnostic bundles to s3://ops-archive/2026/suite-{i:02d}/.\n"
            )
    return "\n".join(sections)


def _make_tickets_dump() -> str:
    items = []
    for i in range(95):
        items.append(json.dumps({
            "id": f"OPS-{2100 + i}",
            "priority": f"P{(i % 3) + 2}",
            "status": "open" if i % 4 else "triaged",
            "component": ["search", "catalog", "images", "analytics", "emails"][i % 5],
            "title": f"Routine cleanup and telemetry alert tuning task #{i} for shard-{i % 8}",
            "owner": f"eng-{i % 12}@company.internal",
        }))
    return "\n".join(items)


def _make_billing_dump() -> str:
    rows = ["invoice_id,account,date,status,amount_usd,memo"]
    for i in range(200):
        if i == 142:
            rows.append(
                f"inv_2026_03_{i:04d},{NEEDLES['account']},2026-03-14,DUPLICATE_OVERCHARGE,"
                f"{NEEDLES['amount']},Duplicate enterprise seat charge - refund required"
            )
        else:
            rows.append(
                f"inv_2026_03_{i:04d},acct_{7000 + i},2026-03-{(i % 28) + 1:02d},PAID,"
                f"{99 + (i * 17) % 300}.00,Monthly standard subscription tier {(i % 3) + 1}"
            )
    return "\n".join(rows)


def _make_iam_dump() -> str:
    rows = []
    for i in range(150):
        if i == 98:
            rows.append(json.dumps({
                "key_id": NEEDLES["stale_key"],
                "principal": "svc-legacy-importer",
                "created": "2023-05-02",
                "last_used": "2024-03-11",
                "status": "ACTIVE_STALE",
                "scopes": ["billing:read", "checkout:read"],
            }))
        else:
            rows.append(json.dumps({
                "key_id": f"key_prod_{1000 + i}",
                "principal": f"svc-worker-{i % 20}",
                "created": "2026-01-15",
                "last_used": "2026-11-18",
                "status": "ACTIVE",
                "scopes": ["metrics:write"],
            }))
    return "\n".join(rows)


def _make_status_page_dump() -> str:
    lines = ["<html><body><h1>Platform Status History</h1>"]
    for i in range(110):
        lines.append(
            f"<article data-id='inc-{900 + i}'><h2>2026-{(i % 10) + 1:02d}-{(i % 27) + 1:02d} "
            f"Scheduled Maintenance Window #{i}</h2>"
            f"<p>All systems operational across us-east-1, us-west-2, eu-west-1. No customer impact observed.</p></article>"
        )
    lines.append("</body></html>")
    return "\n".join(lines)


def _make_calendar_dump() -> str:
    rows = []
    for i in range(110):
        rows.append(json.dumps({
            "event_id": f"evt_{500 + i}",
            "date": f"2026-11-{(i % 18) + 1:02d}",
            "time": f"{9 + (i % 8):02d}:00",
            "title": f"Weekly Engineering Sync & Roadmap Review #{i}",
            "attendees": ["alice@co.internal", "bob@co.internal", "carol@co.internal"],
        }))
    return "\n".join(rows)


# ------------------------------------------------------------------ 12-skill Ops Catalog (24 tools)
OPS_TOOL_DEFS: list[dict] = [
    {"name": "logs_fetch", "description": "Fetch raw service logs for a service.", "inputSchema": _obj({"service": S}, ["service"])},
    {"name": "logs_search", "description": "Search service logs by substring.", "inputSchema": _obj({"service": S, "query": S}, ["service", "query"])},
    {"name": "metrics_dump", "description": "Dump recent Prometheus telemetry for a service.", "inputSchema": _obj({"service": S}, ["service"])},
    {"name": "metrics_alert", "description": "Create or silence a metric alert rule.", "inputSchema": _obj({"name": S, "threshold": N}, ["name", "threshold"])},
    {"name": "code_read", "description": "Read a source file and its recent git diff.", "inputSchema": _obj({"path": S}, ["path"])},
    {"name": "code_diff", "description": "Inspect a specific git commit diff.", "inputSchema": _obj({"commit": S}, ["commit"])},
    {"name": "docs_read", "description": "Read an operational runbook document.", "inputSchema": _obj({"doc": S}, ["doc"])},
    {"name": "docs_update", "description": "Update an operational runbook document.", "inputSchema": _obj({"doc": S, "content": S}, ["doc", "content"])},
    {"name": "ticket_list", "description": "List open incident/ops tickets.", "inputSchema": _obj({}, [])},
    {"name": "ticket_create", "description": "Create a new incident ticket.", "inputSchema": _obj({"title": S, "priority": S, "body": S}, ["title", "priority", "body"])},
    {"name": "invoice_export", "description": "Export a monthly billing ledger.", "inputSchema": _obj({"month": S}, ["month"])},
    {"name": "credit_issue", "description": "Issue a billing credit to a customer account.", "inputSchema": _obj({"account": S, "amount": N, "reason": S}, ["account", "amount", "reason"])},
    {"name": "audit_dump", "description": "Dump IAM API key audit records.", "inputSchema": _obj({}, [])},
    {"name": "key_rotate", "description": "Rotate and revoke an IAM API key.", "inputSchema": _obj({"key_id": S}, ["key_id"])},
    {"name": "site_read", "description": "Read a public website/status page.", "inputSchema": _obj({"page": S}, ["page"])},
    {"name": "site_write", "description": "Write a public website/status page.", "inputSchema": _obj({"page": S, "content": S}, ["page", "content"])},
    {"name": "calendar_list", "description": "List upcoming team calendar events.", "inputSchema": _obj({}, [])},
    {"name": "calendar_book", "description": "Book a team calendar meeting.", "inputSchema": _obj({"date": S, "time": S, "what": S}, ["date", "time", "what"])},
    {"name": "status_post", "description": "Publish a short external status update.", "inputSchema": _obj({"message": S}, ["message"])},
    {"name": "email_draft", "description": "Draft an internal engineering email.", "inputSchema": _obj({"to": S, "subject": S, "body": S}, ["to", "subject", "body"])},
    {"name": "release_notes_read", "description": "Read release notes for a service version.", "inputSchema": _obj({"version": S}, ["version"])},
    {"name": "release_tag", "description": "Tag a release candidate.", "inputSchema": _obj({"version": S, "sha": S}, ["version", "sha"])},
    {"name": "db_schema_dump", "description": "Dump production database schema DDL.", "inputSchema": _obj({"database": S}, ["database"])},
    {"name": "db_query_explain", "description": "Run EXPLAIN on a read-only SQL query.", "inputSchema": _obj({"sql": S}, ["sql"])},
]

OPS_SKILLS: list[Skill] = [
    *[s for s in CATALOG if s.name in BUILTIN],
    Skill("logs", "Fetch and search raw application logs.",
          "Logs skill: call logs_fetch(service) to inspect logs. Extract the exact error code, dependency, and p99 latency concisely.",
          ["logs_fetch", "logs_search"]),
    Skill("metrics", "Inspect service telemetry and alert rules.",
          "Metrics skill: call metrics_dump(service) to inspect telemetry. Report the exact anomalous region and error rate.",
          ["metrics_dump", "metrics_alert"]),
    Skill("codebase", "Inspect source code and git commit diffs.",
          "Codebase skill: call code_read(path) to inspect code and recent diffs. Report exact commit hashes and changed values.",
          ["code_read", "code_diff"]),
    Skill("docs", "Read and update operational runbooks.",
          "Docs skill: call docs_read(doc) to inspect runbooks. Quote exact rollback commands and Slack escalation channels.",
          ["docs_read", "docs_update"]),
    Skill("tickets", "List and create incident tracking tickets.",
          "Tickets skill: check ticket_list() first to avoid duplicates, then call ticket_create(title, priority, body).",
          ["ticket_list", "ticket_create"]),
    Skill("billing", "Inspect billing ledgers and issue customer credits.",
          "Billing skill: call invoice_export(month) to locate overcharges, then call credit_issue(account, amount, reason).",
          ["invoice_export", "credit_issue"]),
    Skill("security", "Audit IAM API keys and rotate stale credentials.",
          "Security skill: call audit_dump() to inspect keys, then call key_rotate(key_id) for any stale key.",
          ["audit_dump", "key_rotate"]),
    Skill("website", "Read and update the public status website.",
          "Website skill: call site_read(page) first, then update via site_write(page, content).",
          ["site_read", "site_write"]),
    Skill("calendar", "List team events and book meetings.",
          "Calendar skill: check calendar_list(), then book with calendar_book(date, time, what) using YYYY-MM-DD and HH:MM.",
          ["calendar_list", "calendar_book"]),
    Skill("comms", "Draft internal incident emails and external status posts.",
          "Comms skill: draft internal engineering postmortems with email_draft(to, subject, body) including every verified fact.",
          ["status_post", "email_draft"]),
    Skill("releases", "Read release notes and tag release candidates.",
          "Releases skill: inspect release_notes_read(version) and tag with release_tag(version, sha).",
          ["release_notes_read", "release_tag"]),
    Skill("databases", "Inspect database schemas and query plans.",
          "Databases skill: inspect schema with db_schema_dump(database) or analyze queries with db_query_explain(sql).",
          ["db_schema_dump", "db_query_explain"]),
]


def make_ops_tools() -> tuple[list[Tool], dict]:
    payloads = {
        "logs": _make_logs_dump(),
        "metrics": _make_metrics_dump(),
        "code": _make_code_dump(),
        "docs": _make_runbook_dump(),
        "tickets": _make_tickets_dump(),
        "billing": _make_billing_dump(),
        "iam": _make_iam_dump(),
        "status": _make_status_page_dump(),
        "calendar": _make_calendar_dump(),
    }
    state: dict = {
        "tickets": [], "credits": [], "rotated": [], "writes": [], "meetings": [], "emails": [], "log": [],
        "raw_bytes_returned": 0,
    }

    def make_fn(name: str):
        def fn(args: dict):
            state["log"].append((name, dict(args)))
            if name == "logs_fetch":
                out = payloads["logs"]
            elif name == "metrics_dump":
                out = payloads["metrics"]
            elif name == "code_read":
                out = payloads["code"]
            elif name == "docs_read":
                out = payloads["docs"]
            elif name == "ticket_list":
                out = payloads["tickets"]
            elif name == "ticket_create":
                state["tickets"].append(dict(args))
                out = f"created ticket OPS-3001 ({args.get('priority', '')})"
            elif name == "invoice_export":
                out = payloads["billing"]
            elif name == "credit_issue":
                state["credits"].append(dict(args))
                out = f"credited {args.get('account')} ${args.get('amount')}"
            elif name == "audit_dump":
                out = payloads["iam"]
            elif name == "key_rotate":
                state["rotated"].append(dict(args))
                out = f"rotated and revoked {args.get('key_id')}"
            elif name == "site_read":
                out = payloads["status"]
            elif name == "site_write":
                state["writes"].append(dict(args))
                out = f"updated page {args.get('page')} ({len(str(args.get('content', '')))} chars)"
            elif name == "calendar_list":
                out = payloads["calendar"]
            elif name == "calendar_book":
                state["meetings"].append(dict(args))
                out = f"booked {args.get('date')} {args.get('time')} - {args.get('what')}"
            elif name == "email_draft":
                state["emails"].append(dict(args))
                out = f"drafted email to {args.get('to')}: {args.get('subject')}"
            else:
                out = f"ok:{name}"
            state["raw_bytes_returned"] += len(out)
            return out
        return fn

    tools = [Tool(name=d["name"], description=d["description"], input_schema=d["inputSchema"], fn=make_fn(d["name"]))
             for d in OPS_TOOL_DEFS]
    return tools, state


SCRIPT = [
    "Fetch the `checkout` service logs using logs_fetch, find the root-cause error code, failing dependency, and p99 latency, and summarize them concisely.",
    "Pull the `checkout` telemetry using metrics_dump, identify which region is spiking and its exact error rate, and summarize concisely.",
    "Read `auth_client.py` and its recent git diff using code_read, find the exact commit hash and timeout_ms change that caused the regression, and summarize concisely.",
    "Read the incident runbook using docs_read to find the exact rollback command and Slack escalation channel, and tell me both.",
    "List open tickets with ticket_list to confirm there is no duplicate, then create a P1 incident ticket with ticket_create referencing the error code, regressed commit hash, and rollback command we found.",
    "Export the March billing ledger with invoice_export, find the duplicate overcharge account and exact dollar amount, and issue a credit with credit_issue for that account and amount.",
    "Dump the IAM audit log with audit_dump, find the stale legacy API key last used in 2024, and rotate it using key_rotate.",
    "Read our current `status` page with site_read, then update `status` with site_write noting the us-east-2 checkout latency incident and rollback in progress.",
    "Check the team calendar with calendar_list and book a meeting with calendar_book on 2026-11-20 at 14:00 titled `Checkout Incident Postmortem`.",
    "Draft an internal engineering postmortem email to `eng@company.internal` using email_draft summarizing ALL findings from this session: error code, failing dependency, exact failing pod name, affected region, error rate, exact `cpu_pct` recorded during the spike, regressed commit hash and timeout change, rollback command, escalation channel, rotated IAM key, and credited billing account. Use your existing conversation/history rather than re-running diagnostics if possible.",
]


class SharedAgent:
    """Single shared context with ALL 12 skill instructions pre-loaded (zero load_skill penalty).
    When `context_limit` is set, summarizes earlier turns before each call that exceeds the budget."""

    def __init__(self, store: Store, provider: Provider, tools: list[Tool], persona: str = BASE_SYSTEM,
                 nonce: str | None = None, max_steps: int = 8, context_limit: int | None = None):
        self.store = store
        self.provider = provider
        self.tools_map = by_name(list(tools))
        self.persona = persona
        self.nonce = nonce
        self.max_steps = max_steps
        self.context_limit = context_limit
        self.compactions = 0
        self.messages: list[dict] = []
        self.caps = [s.name for s in OPS_SKILLS if s.name not in BUILTIN]

    def system(self) -> str:
        return "\n\n".join([self.persona, skill_instructions(self.store, self.caps)])

    def tool_defs(self) -> list[dict]:
        return [t.definition() for t in self.tools_map.values()]

    def _est_tokens(self) -> int:
        return (len(self.system()) + sum(len(json.dumps(m)) for m in self.messages)) // 4

    def _maybe_compact(self) -> None:
        if not self.context_limit or self._est_tokens() <= self.context_limit:
            return
        # Find the start of the latest plain-text user turn so in-flight tool loops stay valid.
        cut = None
        for idx in range(len(self.messages) - 1, 0, -1):
            m = self.messages[idx]
            if m["role"] == "user" and all(b.get("type") == "text" for b in m["content"]):
                cut = idx
                break
        if cut is None or cut <= 1:
            return
        prefix = self.messages[:cut]
        compact_prompt = (
            "Summarize the conversation so far into a concise briefing preserving key decisions, "
            "findings, identifiers, and completed actions. Omit verbose raw logs/dumps."
        )
        req = prefix + [{"role": "user", "content": [{"type": "text", "text": compact_prompt}]}]
        tagger = getattr(self.provider, "usage_tag", None)
        with tagger("shared:compaction") if tagger else contextlib.nullcontext():
            blocks = self.provider.complete("You summarize prior conversation state concisely.", req, [])
        summary = "\n".join(b["text"] for b in blocks if b["type"] == "text").strip() or "(compacted)"
        self.messages = [
            {"role": "user", "content": [{"type": "text", "text": f"[Summary of earlier conversation]\n{summary}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Understood, continuing from summary."}]},
            *self.messages[cut:],
        ]
        self.compactions += 1

    async def send(self, request: dict):
        text = request["message"]["parts"][0]["text"]
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        said: list[str] = []
        tools = self.tool_defs()
        allowed = {t["name"] for t in tools}
        for _ in range(self.max_steps):
            self._maybe_compact()
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
        return a2a.SendMessageResponse(task=a2a.Task(
            id=tid, context_id="shared",
            status=a2a.TaskStatus(state=a2a.COMPLETED, message=history[-1] if said else None),
            history=history,
        ))


def _fake_policy(arm: str):
    """Deterministic offline policy to smoke-test both arms without API calls."""
    def policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        if not tools:
            return _text("Summary: " + " ".join(NEEDLES.values()))
        names = {t["name"] for t in tools}
        last = _last_block(messages)
        texts = _user_texts(messages)
        cur = texts[-1] if texts else ""
        if arm == "graft" and "fork" in names:
            if woken(messages):
                return _wake_reply(messages)
            ids = _row_ids(messages)
            for kw, sk in (("logs_fetch", "logs"), ("metrics_dump", "metrics"), ("code_read", "codebase"),
                           ("docs_read", "docs"), ("ticket_list", "tickets"), ("invoice_export", "billing"),
                           ("audit_dump", "security"), ("site_read", "website"), ("calendar_list", "calendar"),
                           ("email_draft", "comms")):
                if kw in cur:
                    seed = ids if sk in ("tickets", "comms") else ids[-1:]
                    return _use(tool="fork", current={"skills": [sk], "messages": seed, "reason": cur[:120]})
            return _text("Done.")
        prev = next((b["name"] for m in reversed(messages) if m["role"] == "assistant"
                     for b in m["content"] if b["type"] == "tool_use"), None)
        if last["type"] == "tool_result" and prev != "fork":
            if prev == "logs_fetch":
                return _text(f"Found {NEEDLES['error_code']} from {NEEDLES['dependency']} (p99 {NEEDLES['latency']}).")
            if prev == "metrics_dump":
                return _text(f"Spike in {NEEDLES['region']} at {NEEDLES['error_rate']} error rate.")
            if prev == "code_read":
                return _text(f"Commit {NEEDLES['commit']} changed timeout_ms from {NEEDLES['good_timeout']} to {NEEDLES['bad_timeout']}.")
            if prev == "docs_read":
                return _text(f"Run `{NEEDLES['rollback_cmd']}` and notify {NEEDLES['channel']}.")
            if prev == "ticket_list":
                return _use("ticket_create", title="Checkout 503 outage", priority="P1",
                            body=f"{NEEDLES['error_code']} commit {NEEDLES['commit']} rollback: {NEEDLES['rollback_cmd']}")
            if prev == "invoice_export":
                return _use("credit_issue", account=NEEDLES["account"], amount=412.50, reason="Duplicate overcharge")
            if prev == "audit_dump":
                return _use("key_rotate", key_id=NEEDLES["stale_key"])
            if prev == "site_read":
                return _use("site_write", page="status", content="Investigating us-east-2 checkout latency; rollback in progress.")
            if prev == "calendar_list":
                return _use("calendar_book", date="2026-11-20", time="14:00", what="Checkout Incident Postmortem")
            return _text(f"Completed {prev} ({NEEDLES['stale_key']}, {NEEDLES['account']}).")
        for kw, tool, args in (
            ("logs_fetch", "logs_fetch", {"service": "checkout"}),
            ("metrics_dump", "metrics_dump", {"service": "checkout"}),
            ("code_read", "code_read", {"path": "auth_client.py"}),
            ("docs_read", "docs_read", {"doc": "incident-runbook"}),
            ("ticket_list", "ticket_list", {}),
            ("invoice_export", "invoice_export", {"month": "2026-03"}),
            ("audit_dump", "audit_dump", {}),
            ("site_read", "site_read", {"page": "status"}),
            ("calendar_list", "calendar_list", {}),
            ("email_draft", "email_draft", {
                "to": "eng@company.internal",
                "subject": "Checkout Incident Postmortem",
                "body": " ".join(NEEDLES.values()),
            }),
        ):
            if kw in cur and tool in names:
                return _use(tool, **args)
        return _text("Done.")
    return policy


def score(state: dict) -> dict[str, bool]:
    ticket_blob = json.dumps(state["tickets"]).lower()
    credit_ok = any(
        c.get("account") == NEEDLES["account"] and abs(float(c.get("amount", 0)) - 412.50) < 0.01
        for c in state["credits"]
    )
    rotate_ok = any(r.get("key_id") == NEEDLES["stale_key"] for r in state["rotated"])
    site_blob = json.dumps(state["writes"]).lower()
    cal_ok = any(
        m.get("date") == "2026-11-20" and "14:00" in str(m.get("time", "")) and "postmortem" in str(m.get("what", "")).lower()
        for m in state["meetings"]
    )
    email_blob = json.dumps(state["emails"]).lower()
    return {
        "ticket_created_with_facts": (
            bool(state["tickets"])
            and NEEDLES["error_code"].lower() in ticket_blob
            and NEEDLES["commit"].lower() in ticket_blob
            and "rollback" in ticket_blob
        ),
        "billing_credit_accurate": credit_ok,
        "stale_key_rotated": rotate_ok,
        "status_page_updated": bool(state["writes"]) and "us-east-2" in site_blob and "rollback" in site_blob,
        "postmortem_meeting_booked": cal_ok,
        "email_has_error_and_dep": NEEDLES["error_code"].lower() in email_blob and NEEDLES["dependency"].lower() in email_blob,
        "email_has_buried_pod": NEEDLES["pod"].lower() in email_blob,
        "email_has_region_and_rate": NEEDLES["region"].lower() in email_blob and "14.7" in email_blob,
        "email_has_buried_cpu_pct": NEEDLES["cpu_pct"] in email_blob,
        "email_has_commit_and_timeout": NEEDLES["commit"].lower() in email_blob and "50" in email_blob and "5000" in email_blob,
        "email_has_rollback_and_channel": "418" in email_blob and "sev1-platform" in email_blob,
        "email_has_key_and_account": NEEDLES["stale_key"].lower() in email_blob and NEEDLES["account"].lower() in email_blob,
    }


def _est_cost(usage: dict) -> float:
    """Relative USD estimate using standard OpenAI cache pricing ($2/M in, $2/M cache_write, $0.20/M cached, $8/M out)."""
    return (
        usage.get("uncached", 0) * 2.0
        + usage.get("cache_write", 0) * 2.0
        + usage.get("cached", 0) * 0.20
        + usage.get("output", 0) * 8.0
    ) / 1_000_000


_DIAG_TOOLS = ("logs_fetch", "metrics_dump", "code_read", "docs_read", "invoice_export", "audit_dump")


async def run_arm(arm: str, *, fake: bool, trace: bool, context_limit: int | None = None) -> dict:
    db_path = os.path.join(tempfile.mkdtemp(prefix=f"graft-lh-{arm}-"), f"{arm}.db")
    store = Store(db_path)
    tools, state = make_ops_tools()
    provider = FakeProvider(_fake_policy(arm)) if fake else make_provider()
    nonce = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    if arm == "graft":
        agent = GraftAgent(
            store, provider, tools=tools,
            config=GraftConfig(
                nonce=nonce, seeder=None, skills=OPS_SKILLS,
                compact_tokens=context_limit, compaction_mode="summary",
                trace=print if trace else None,
            ),
        )
        async with agent:
            await run_script(agent, SCRIPT, echo=trace)
        compactions = store.con.execute("SELECT COUNT(*) FROM compactions").fetchone()[0]
        searches = [
            (r[0], json.loads(r[1])["input"])
            for r in store.con.execute(
                "SELECT owner_timeline, content FROM turns WHERE kind='tool_use' AND tool_name='search' ORDER BY row_id"
            )
        ]
    else:
        install(store, tools={t.name: t.definition() for t in tools}, skills=OPS_SKILLS)
        agent = SharedAgent(store, provider, tools=tools, nonce=nonce, context_limit=context_limit)
        await run_script(agent, SCRIPT, echo=trace)
        compactions = agent.compactions
        searches = []

    elapsed = round(time.perf_counter() - t0, 2)
    usage = dict(getattr(provider, "usage", {}))
    call_log = list(getattr(provider, "call_log", []))
    peak_in = max((c.get("input", 0) for c in call_log), default=0)
    by_tag = {k: dict(v) for k, v in getattr(provider, "by_tag", {}).items()}
    checks = score(state)
    diag_counts = {n: sum(1 for t, _ in state["log"] if t == n) for n in _DIAG_TOOLS}
    refetches = sum(max(0, c - 1) for c in diag_counts.values())
    return {
        "arm": arm,
        "provider": f"{provider.name}:{provider.model}",
        "context_limit": context_limit,
        "elapsed_s": elapsed,
        "usage": usage,
        "peak_input": peak_in,
        "by_tag": by_tag,
        "est_cost": _est_cost(usage),
        "raw_bytes": state["raw_bytes_returned"],
        "compactions": compactions,
        "searches": searches,
        "refetches": refetches,
        "diag_counts": diag_counts,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Long-horizon heavy-tool-output benchmark: graft vs shared")
    p.add_argument("--arm", choices=("graft", "shared", "all"), default="all")
    p.add_argument("--context-limit", type=int, default=None,
                   help="Artificial context token limit forcing summary compaction on both arms (e.g. 16000)")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()

    arms = ("graft", "shared") if args.arm == "all" else (args.arm,)

    async def go():
        results = [
            await run_arm(arm, fake=args.fake, trace=args.trace, context_limit=args.context_limit)
            for arm in arms
        ]
        lim = f" (context_limit={args.context_limit:,} tok)" if args.context_limit else ""
        print(f"\n================ LONG-HORIZON HEAVY-TOOL BENCHMARK{lim} ================")
        for r in results:
            u = r["usage"]
            in_tok = u.get("input", 0)
            cached = u.get("cached", 0)
            cache_write = u.get("cache_write", 0)
            uncached = u.get("uncached", 0)
            out_tok = u.get("output", 0)
            pct = (100.0 * cached / in_tok) if in_tok else 0.0
            verdict = "PASS" if r["passed"] == r["total"] else f"FAIL ({r['passed']}/{r['total']})"
            print(
                f"\n[{verdict}] arm={r['arm']} ({r['provider']})  {r['elapsed_s']}s  "
                f"calls={u.get('calls', 0)}  compactions={r['compactions']}  "
                f"searches={len(r['searches'])}  refetches={r['refetches']}  raw_tool_bytes={r['raw_bytes']:,}"
            )
            print(
                f"    input={in_tok:,} (uncached={uncached:,}, cache_write={cache_write:,}, "
                f"cached={cached:,} [{pct:.1f}%])  peak_call_in={r['peak_input']:,}  "
                f"out={out_tok:,}  est_cost=${r['est_cost']:.4f}"
            )
            if r["searches"]:
                print(f"    search_calls: {r['searches']}")
            for k, ok in r["checks"].items():
                print(f"    [{'ok' if ok else 'MISS'}] {k}")
            if r["arm"] == "graft":
                orig_in = sum(v["input"] for t, v in r["by_tag"].items() if t.startswith("original"))
                br_in = sum(v["input"] for t, v in r["by_tag"].items() if t.startswith("branch:"))
                orig_calls = sum(v["calls"] for t, v in r["by_tag"].items() if t.startswith("original"))
                br_calls = sum(v["calls"] for t, v in r["by_tag"].items() if t.startswith("branch:"))
                print(f"    breakdown: main calls={orig_calls} in={orig_in:,} | branches calls={br_calls} in={br_in:,}")
        return results

    asyncio.run(go())


if __name__ == "__main__":
    main()
