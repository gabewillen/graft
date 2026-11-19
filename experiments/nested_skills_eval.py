"""Live E2E experiment: Hierarchical Nested Skills (`min_children_to_fork=2` vs `1` vs `bundled` vs `flat`).

Models a realistic production skill tree on disk (loaded via `GraftConfig(skills_dir=...)`):
  - `incident-triage` (owns 12 specialized subsystem runbook child skills, ~1.5 KB each = ~18 KB total)
    - `postgres-locks`, `postgres-vacuum`, `redis-evictions`, `kafka-lag`, `k8s-oom`,
      `dns-resolution`, `tls-certs`, `cdn-cache`, `load-balancer`, `auth-tokens`,
      `storage-iops`, `rate-limiter`
  - `status-page` (owns 1 child skill: `status-template`) — directly tests `min_children_to_fork=2` vs `1`!
  - `changelog` (leaf skill, 0 children)

Runs the same 4-turn incident-response script across four modes:
  1. `nested-min2` (default, `min_children_to_fork=2`):
     - `main` sees 3 top-level skills (+ `search`).
     - `incident-triage` (12 children >= 2) gains `fork` and forks only the 1-2 needed child skills.
     - `status-page` (1 child < 2) auto-links `status-template` directly (no extra fork hop).
  2. `nested-min1` (`min_children_to_fork=1`):
     - Same tree, except `status-page` (1 child) also gains `fork` and must fork `status-template`.
  3. `bundled` (`min_children_to_fork=999`):
     - `main` still sees 3 top-level skills, but `incident-triage` auto-links all 12 child skills (~18 KB + 13 tools).
  4. `flat` (all skills flattened to top-level on `main`):
     - Every child skill is exposed directly in `main`'s catalog.

Measures: input tokens, cached tokens, cache hit %, uncached tokens, output tokens, peak input,
estimated cost, LLM calls, branches by depth, latency, and ground-truth accuracy (6 checks).

Usage:
  uv run python experiments/nested_skills_eval.py --fake
  uv run python experiments/nested_skills_eval.py --provider openai --model gpt-6-luna
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft.agent import GraftAgent, GraftConfig  # noqa: E402
from graft.db import Skill, Store  # noqa: E402
from graft.fake import _last_block, _row_ids, _text, _use, _user_texts, woken  # noqa: E402
from graft.providers import FakeProvider, make_provider  # noqa: E402
from graft.sim import say  # noqa: E402
from graft.skills import BASE_SYSTEM, load_skills  # noqa: E402
from graft.tools import Tool, ToolContext  # noqa: E402

S = {"type": "string"}


def _obj(props: dict, req: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": req}


# Rich, realistic ~1.5 KB runbook body template for each subsystem child skill so token/caching
# differences reflect real production skill catalogs (e.g., 12-14 specialized skills under a parent).
def _runbook(subsystem: str, tool_name: str, focus_area: str, checklist: list[str], escalation: str) -> str:
    steps = "\n".join(f"{i}. {item}" for i, item in enumerate(checklist, 1))
    return f"""\
# {subsystem} Diagnostic Runbook

You are executing the specialized `{subsystem}` runbook for {focus_area}.

## Required Procedure
{steps}

## Tool Usage & Interpretation Rules
- Always invoke `{tool_name}(service)` first before drawing any conclusion.
- Quote exact identifiers from the tool output (PIDs, lock modes, key IDs, partition offsets,
  memory percentages, error counts, and metric thresholds) verbatim so downstream remediation and
  post-incident summaries remain auditable.
- Distinguish primary triggers (e.g., blocking locks, exhausted memory/evictions, expired signing
  keys, crashed pods, saturated disks) from secondary symptoms (e.g., upstream 503s or p99 latency
  spikes caused by queue buildup).
- Do not speculate about subsystems outside `{subsystem}` — report only verified telemetry returned
  by `{tool_name}` along with the concrete remediation guidance below.

## Severity & Remediation Matrix
- **Critical**: Immediate impact on live request path or data integrity. Recommend targeted
  mitigation (terminate blocking query, flush/scale cache, force JWKS refresh, or roll back bad
  config) and include exact identifiers from `{tool_name}`.
- **Degraded**: Elevated latency or retry pressure without total outage. Document trend and
  recommend capacity or tuning follow-up.
- **Healthy**: Metrics within normal operating envelope. State clearly that `{subsystem}` is not
  contributing to the incident.

## Escalation Notes
{escalation}
"""


SUBSYSTEM_SPECS: list[tuple[str, str, str, str, list[str], str]] = [
    ("postgres-locks", "pg_locks_inspect", "Inspect PostgreSQL lock contention and blocked queries.",
     "database lock queues and DDL contention",
     ["Query active blocking chains via `pg_locks_inspect(service)`.",
      "Capture the blocking PID, lock mode (e.g., ACCESS EXCLUSIVE), target relation, and waiter count.",
      "Report whether an unindexed migration or long transaction is holding the lock."],
     "Escalate to Database On-Call if blocking PID holds ACCESS EXCLUSIVE > 30s on checkout/orders tables."),
    ("postgres-vacuum", "pg_vacuum_stats", "Inspect PostgreSQL autovacuum lag, dead tuples, and XID wraparound.",
     "table bloat and autovacuum saturation",
     ["Fetch dead tuple ratios and oldest XID age via `pg_vacuum_stats(service)`.",
      "Verify whether autovacuum workers are throttled by `autovacuum_vacuum_cost_limit`.",
      "Report any table exceeding 20% dead tuples or >500M XID age."],
     "Escalate to Database On-Call if XID age exceeds 1B or bloat triggers sequential scan regressions."),
    ("redis-evictions", "redis_memory_stats", "Inspect Redis memory saturation, key evictions, and hit ratio.",
     "in-memory cache pressure and eviction storms",
     ["Call `redis_memory_stats(service)` to collect memory utilization, `evicted_keys`, and policy.",
      "Verify whether `maxmemory_policy` (e.g., `allkeys-lru`) is evicting hot session/cart keys.",
      "Report exact `evicted_keys` count, used memory %, and cache hit rate drop."],
     "Escalate to Cache Platform if memory >95% or eviction rate causes upstream database fallback spikes."),
    ("kafka-lag", "kafka_consumer_lag", "Inspect Kafka consumer group lag, partition skew, and rebalance loops.",
     "event pipeline backpressure and consumer stalls",
     ["Call `kafka_consumer_lag(service)` for per-partition offsets and consumer group state.",
      "Identify stalled partitions or repeated rebalance epochs.",
      "Report max partition lag and consumer throughput trend."],
     "Escalate to Streaming Infra if lag grows monotonically for >5 minutes."),
    ("k8s-oom", "k8s_pod_events", "Inspect Kubernetes pod OOMKills, CrashLoopBackOff, and CPU throttling.",
     "container restarts and memory limit breaches",
     ["Call `k8s_pod_events(service)` to list recent termination reasons and exit codes.",
      "Check for exit code 137 (OOMKilled) and readiness probe failures.",
      "Report affected pod names, restart counts, and configured memory limits."],
     "Escalate to Compute On-Call if >25% of replicas are restarting concurrently."),
    ("dns-resolution", "dns_trace", "Inspect CoreDNS latency, NXDOMAIN spikes, and upstream resolver timeouts.",
     "service discovery and DNS resolution failures",
     ["Call `dns_trace(service)` to inspect query latency and SERVFAIL/NXDOMAIN rates.",
      "Check ndots search path amplification and conntrack table exhaustion.",
      "Report resolver p99 latency and failing hostnames."],
     "Escalate to Network On-Call if CoreDNS SERVFAIL exceeds 1%."),
    ("tls-certs", "tls_cert_inspect", "Inspect TLS certificate expiry, chain validity, and handshake errors.",
     "TLS termination and mTLS certificate rotation",
     ["Call `tls_cert_inspect(service)` to verify NotAfter dates and intermediate chains.",
      "Check for mismatched SAN entries or revoked leaf certificates.",
      "Report exact expiry timestamps and handshake failure counts."],
     "Escalate to Security Infra immediately if any production leaf cert expires within 24h."),
    ("cdn-cache", "cdn_hit_ratio", "Inspect edge CDN cache hit ratio, origin shield load, and 5xx bursts.",
     "edge caching anomalies and origin overload",
     ["Call `cdn_hit_ratio(service)` to inspect edge HIT/MISS/BYPASS ratios.",
      "Check for cache-busting query strings or missing Cache-Control headers.",
      "Report origin request surge percentage and edge status codes."],
     "Escalate to Edge Engineering if origin QPS exceeds 3x baseline."),
    ("load-balancer", "lb_health_checks", "Inspect L7 load balancer target health, 502/504 rates, and draining.",
     "ingress routing and target group health checks",
     ["Call `lb_health_checks(service)` to list unhealthy targets and reset reasons.",
      "Distinguish upstream connection timeouts (504) from premature resets (502).",
      "Report healthy vs unhealthy target counts per availability zone."],
     "Escalate to Traffic Engineering if an entire AZ fails health checks."),
    ("auth-tokens", "jwt_key_status", "Inspect JWT signing keys (JWKS), key rotation state, and 401 causes.",
     "authentication token validation and JWKS rotation",
     ["Call `jwt_key_status(service)` to inspect active key IDs (`kid`), cache TTLs, and signature errors.",
      "Verify whether edge/gateway nodes are serving a stale JWKS cache missing the newly rotated `kid`.",
      "Report the exact missing/failing `kid`, JWKS cache TTL, and 401 rejection reason."],
     "Escalate to Identity Platform if a rotation dropped an active signing `kid` before TTL expiry."),
    ("storage-iops", "disk_iops_metrics", "Inspect EBS/NVMe IOPS throttling, queue depth, and fsync latency.",
     "block storage saturation and WAL write stalls",
     ["Call `disk_iops_metrics(service)` to read IOPS utilization and burst balance.",
      "Correlate high disk queue depth with WAL fsync latency spikes.",
      "Report volume ID, IOPS cap, and p99 write latency."],
     "Escalate to Storage Infra if burst balance drops below 10%."),
    ("rate-limiter", "ratelimit_buckets", "Inspect API gateway rate-limit buckets, 429 spikes, and quota rules.",
     "traffic shaping and unintended client throttling",
     ["Call `ratelimit_buckets(service)` to inspect top-throttled keys and rule versions.",
      "Check whether a misconfigured global quota is rejecting legitimate internal traffic.",
      "Report 429 rate, triggered rule ID, and top affected callers."],
     "Escalate to API Gateway Team if legitimate internal service-to-service traffic receives 429s."),
]

TRIAGE_INSTRUCTIONS = """\
You are the incident triage coordinator.
1. Always call `service_overview(service)` first to see which specific subsystems are firing alerts
   for the affected service.
2. Only inspect the specific subsystems flagged as degraded/critical in `service_overview` — never
   waste calls on healthy subsystems.
3. If you hold `fork` for your child skill catalog, fork ONLY the 1-2 flagged subsystem child skills
   (in parallel as `detached` branches when 2 subsystems are flagged, or as a `current` branch when
   1 subsystem is flagged). Wait for detached branches to merge, then report a concise root-cause
   summary with all exact IDs/metrics to the user.
4. If the subsystem tools are already directly available to you (no `fork`), call only the flagged
   subsystem tools directly and report the concise root-cause summary to the user.
"""

STATUS_PAGE_INSTRUCTIONS = """\
You manage customer-facing status page updates. Always format the update following the
`status-template` guidelines (including its exact required prefix) before calling `status_post`.
"""

STATUS_TEMPLATE_INSTRUCTIONS = """\
Status Page House Style Template:
- Every customer-facing status post MUST begin with the exact literal prefix `[INCIDENT UPDATE]`
  followed by one concise, calm sentence naming the affected customer experience (never internal
  PIDs or database table names) and stating that engineers are investigating.
- After formatting the text, call `status_post(component, message)` once and confirm to the user.
"""

CHANGELOG_INSTRUCTIONS = """\
Search recent production deployments with `changelog_search(service)` and report matching deploy
IDs and summaries concisely to the user.
"""

SCRIPT = [
    "Alert SEV-1 on checkout-api: elevated p99 latency and 503s. Run incident triage to find the root cause across affected subsystems.",
    "Post a brief customer update on the status page noting checkout delays are under investigation.",
    "Now we also got a separate alert on auth-gateway: 401 spikes after key rotation. Triage that too.",
    "Check recent deploys in the changelog for checkout-api and auth-gateway in the last 24h.",
]


def _write_skill_md(dir_path: Path, *, name: str, description: str, instructions: str,
                    tools: list[str], meta_parent: str | None = None) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    if tools:
        lines.append("tools:")
        for t in tools:
            lines.append(f"  - {t}")
    if meta_parent is not None:
        lines.append("metadata:")
        lines.append(f"  parent: {meta_parent}")
    lines.append("---")
    lines.append("")
    lines.append(instructions.strip())
    (dir_path / "SKILL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_skills_dir(root: Path, *, flat: bool = False) -> Path:
    """Materialize the full 15-skill catalog on disk as Agent Skills `SKILL.md` files."""
    _write_skill_md(
        root / "incident-triage",
        name="incident-triage",
        description="Coordinate SEV-1/SEV-2 incident triage for a service and diagnose affected subsystems.",
        instructions=TRIAGE_INSTRUCTIONS,
        tools=["service_overview"],
    )
    for name, tool_name, desc, area, checklist, esc in SUBSYSTEM_SPECS:
        body = _runbook(name, tool_name, area, checklist, esc)
        if flat:
            _write_skill_md(root / name, name=name, description=desc, instructions=body, tools=[tool_name])
        else:
            _write_skill_md(
                root / "incident-triage" / "skills" / name,
                name=name, description=desc, instructions=body, tools=[tool_name],
            )
    _write_skill_md(
        root / "status-page",
        name="status-page",
        description="Publish customer-facing incident updates to the public status page.",
        instructions=STATUS_PAGE_INSTRUCTIONS,
        tools=[],
    )
    if flat:
        _write_skill_md(
            root / "status-template",
            name="status-template",
            description="Format and publish status page posts with required house-style template.",
            instructions=STATUS_TEMPLATE_INSTRUCTIONS,
            tools=["status_post"],
        )
    else:
        _write_skill_md(
            root / "status-page" / "skills" / "status-template",
            name="status-template",
            description="Format and publish status page posts with required house-style template.",
            instructions=STATUS_TEMPLATE_INSTRUCTIONS,
            tools=["status_post"],
        )
    _write_skill_md(
        root / "changelog",
        name="changelog",
        description="Search recent production deployments and release changelogs for a service.",
        instructions=CHANGELOG_INSTRUCTIONS,
        tools=["changelog_search"],
    )
    return root


def make_tools() -> tuple[list[Tool], dict]:
    state: dict = {"calls": [], "status_posts": []}

    def record(name: str, args: dict, call: ToolContext) -> None:
        state["calls"].append({"tool": name, "args": dict(args), "branch": call.branch, "name": call.branch_name})

    def service_overview(args: dict, call: ToolContext) -> str:
        record("service_overview", args, call)
        svc = str(args.get("service", "")).lower()
        if "auth" in svc:
            return json.dumps({
                "service": "auth-gateway",
                "alert": "SEV-2 401 spike after key rotation",
                "flagged_subsystems": ["auth-tokens"],
                "healthy_subsystems": ["postgres-locks", "redis-evictions", "k8s-oom", "dns-resolution", "tls-certs"],
            })
        return json.dumps({
            "service": "checkout-api",
            "alert": "SEV-1 elevated p99 latency (4.8s) and 503s",
            "flagged_subsystems": ["postgres-locks", "redis-evictions"],
            "healthy_subsystems": ["postgres-vacuum", "kafka-lag", "k8s-oom", "dns-resolution", "tls-certs",
                                   "cdn-cache", "load-balancer", "auth-tokens", "storage-iops", "rate-limiter"],
        })

    def make_sub_fn(sub_name: str, tool_name: str):
        def fn(args: dict, call: ToolContext) -> str:
            record(tool_name, args, call)
            if tool_name == "pg_locks_inspect":
                return json.dumps({
                    "service": args.get("service", "checkout-api"),
                    "status": "CRITICAL",
                    "blocking_pid": 88412,
                    "lock_mode": "ACCESS EXCLUSIVE",
                    "relation": "orders",
                    "waiters": 47,
                    "query": "ALTER TABLE orders ADD COLUMN promo_tier TEXT;",
                })
            if tool_name == "redis_memory_stats":
                return json.dumps({
                    "service": args.get("service", "checkout-api"),
                    "status": "CRITICAL",
                    "used_memory_pct": 99.4,
                    "maxmemory_policy": "allkeys-lru",
                    "evicted_keys": 184290,
                    "cache_hit_ratio": 0.31,
                })
            if tool_name == "jwt_key_status":
                return json.dumps({
                    "service": args.get("service", "auth-gateway"),
                    "status": "CRITICAL",
                    "missing_kid": "key-2026-q4-b",
                    "jwks_cache_ttl_sec": 3600,
                    "error": "signature verification failed: unknown kid key-2026-q4-b on stale JWKS cache",
                })
            return json.dumps({"service": args.get("service", "unknown"), "status": "HEALTHY", "subsystem": sub_name})
        return fn

    def status_post(args: dict, call: ToolContext) -> str:
        record("status_post", args, call)
        state["status_posts"].append(dict(args))
        return json.dumps({"posted": True, "component": args.get("component"), "message": args.get("message")})

    def changelog_search(args: dict, call: ToolContext) -> str:
        record("changelog_search", args, call)
        return json.dumps({
            "deploys": [
                {"id": "deploy-4419", "service": "checkout-api", "time": "2h ago",
                 "summary": "Add promo_tier column migration on orders table & expand cart cache payload"},
                {"id": "deploy-4421", "service": "auth-gateway", "time": "45m ago",
                 "summary": "Rotate JWT signing key to key-2026-q4-b without lowering JWKS cache TTL"},
            ]
        })

    tools = [
        Tool("service_overview", _obj({"service": S}, ["service"]), service_overview,
             description="Fetch active alert summary and flagged subsystems for a service."),
        Tool("status_post", _obj({"component": S, "message": S}, ["component", "message"]), status_post,
             description="Publish a customer-facing incident update to the status page."),
        Tool("changelog_search", _obj({"service": S}, ["service"]), changelog_search,
             description="Search recent deployments for a service."),
    ]
    for sub_name, tool_name, desc, *_ in SUBSYSTEM_SPECS:
        tools.append(Tool(tool_name, _obj({"service": S}, ["service"]), make_sub_fn(sub_name, tool_name), description=desc))
    return tools, state


def _est_cost(u: dict) -> float:
    # Standard GPT-class ballpark pricing ($2.50/M uncached + cache_write, $0.25/M cached, $10/M output)
    # for apples-to-apples relative comparison across modes on the same model.
    full_in = u["uncached"] + u.get("cache_write", 0)
    return (full_in * 2.50 + u["cached"] * 0.25 + u["output"] * 10.0) / 1_000_000


def evaluate(replies: list[str], state: dict) -> dict[str, bool]:
    all_text = "\n".join(replies)
    t1 = replies[0] if len(replies) > 0 else ""
    t3 = replies[2] if len(replies) > 2 else ""
    t4 = replies[3] if len(replies) > 3 else ""
    posts = state["status_posts"]
    return {
        "pg_lock_pid": "88412" in t1 and "orders" in t1.lower(),
        "redis_evictions": "184290" in t1.replace(",", "") and "redis" in t1.lower(),
        "status_posted": bool(posts),
        "status_prefix": any("[INCIDENT UPDATE]" in str(p.get("message", "")) for p in posts),
        "auth_kid": "key-2026-q4-b" in t3 or "key-2026-q4-b" in all_text,
        "changelog_deploys": "deploy-4419" in t4 and "deploy-4421" in t4,
    }


def _fake_policy():
    """Offline smoke-test policy covering all 4 modes cleanly."""
    def policy(system, messages, tool_defs):
        names = {t["name"] for t in tool_defs}
        last = _last_block(messages)
        texts = _user_texts(messages)
        cur = texts[-1] if texts else ""
        if "You are on main" in system:
            if woken(messages):
                return []
            if "checkout-api" in cur and "triage" in cur.lower():
                return _use(tool="fork", current={"skills": ["incident-triage"], "messages": _row_ids(messages)[-1:], "wake": False})
            if "status page" in cur.lower():
                if "- status-template:" in system:
                    return _use(tool="fork", current={"skills": ["status-page", "status-template"], "messages": _row_ids(messages)[-1:], "wake": False})
                return _use(tool="fork", current={"skills": ["status-page"], "messages": _row_ids(messages)[-1:], "wake": False})
            if "auth-gateway" in cur and "triage" in cur.lower():
                return _use(tool="fork", current={"skills": ["incident-triage"], "messages": _row_ids(messages)[-1:], "wake": False})
            return _use(tool="fork", current={"skills": ["changelog"], "messages": _row_ids(messages)[-1:], "wake": False})
        # Non-leaf branch or leaf branch
        if "service_overview" in names:
            if woken(messages):
                if "auth-gateway" in cur:
                    return _text("Auth-gateway 401s caused by missing kid key-2026-q4-b in stale JWKS cache.")
                return _text("Checkout-api root causes: Postgres blocking PID 88412 on orders + Redis 184290 evicted keys.")
            if last["type"] == "tool_result" and str(last.get("content", "")).startswith("forked ->"):
                return []
            if last["type"] == "tool_result" and last.get("tool_use_id", "").startswith("ov-"):
                body = str(last.get("content", ""))
                if "fork" in names:
                    if "auth-gateway" in body:
                        return _use(tool="fork", current={"skills": ["auth-tokens"], "messages": _row_ids(messages)[-1:], "wake": False})
                    return _use(tool="fork", detached=[
                        {"id": "sub-pg", "skills": ["postgres-locks"], "messages": _row_ids(messages)[-1:], "reason": "check pg"},
                        {"id": "sub-redis", "skills": ["redis-evictions"], "messages": _row_ids(messages)[-1:], "reason": "check redis"},
                    ])
                # Bundled mode: call flagged tools directly
                if "auth-gateway" in body:
                    return [{"type": "tool_use", "id": "jwt-1", "name": "jwt_key_status", "input": {"service": "auth-gateway"}}]
                if "pg_locks_inspect" in names:
                    return [
                        {"type": "tool_use", "id": "pg-1", "name": "pg_locks_inspect", "input": {"service": "checkout-api"}},
                        {"type": "tool_use", "id": "rd-1", "name": "redis_memory_stats", "input": {"service": "checkout-api"}},
                    ]
                # Flat mode: incident-triage has no child tools, hand back via merge with reason
                return _use(tool="merge", reason="Need postgres-locks and redis-evictions.")
            if last["type"] == "tool_result" and last.get("tool_use_id") in ("pg-1", "rd-1", "jwt-1"):
                if last.get("tool_use_id") == "jwt-1":
                    return _text("Auth-gateway 401s caused by missing kid key-2026-q4-b.")
                return _text("Checkout-api root causes: Postgres blocking PID 88412 on orders + Redis 184290 evicted keys.")
            svc = "auth-gateway" if "auth-gateway" in cur else "checkout-api"
            return [{"type": "tool_use", "id": f"ov-{svc}", "name": "service_overview", "input": {"service": svc}}]
        if "pg_locks_inspect" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "pg-c":
                return _text("Postgres blocking PID 88412 holds ACCESS EXCLUSIVE on orders.")
            return [{"type": "tool_use", "id": "pg-c", "name": "pg_locks_inspect", "input": {"service": "checkout-api"}}]
        if "redis_memory_stats" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "rd-c":
                return _text("Redis at 99.4% RAM with 184290 evicted_keys.")
            return [{"type": "tool_use", "id": "rd-c", "name": "redis_memory_stats", "input": {"service": "checkout-api"}}]
        if "jwt_key_status" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "jwt-c":
                return _text("Auth-gateway missing kid key-2026-q4-b on stale JWKS cache.")
            return [{"type": "tool_use", "id": "jwt-c", "name": "jwt_key_status", "input": {"service": "auth-gateway"}}]
        if "status_post" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "sp-1":
                return _text("Published status update.")
            return [{"type": "tool_use", "id": "sp-1", "name": "status_post",
                     "input": {"component": "checkout", "message": "[INCIDENT UPDATE] Checkout delays are under investigation."}}]
        if "fork" in names and "- status-template:" in system:
            return _use(tool="fork", current={"skills": ["status-template"], "messages": _row_ids(messages)[-1:], "wake": False})
        if "changelog_search" in names:
            if last["type"] == "tool_result" and last.get("tool_use_id") == "ch-1":
                return _text("Recent deploys: deploy-4419 (checkout-api) and deploy-4421 (auth-gateway).")
            return [{"type": "tool_use", "id": "ch-1", "name": "changelog_search", "input": {"service": "checkout-api"}}]
        return _text("done")
    return policy


async def run_mode(mode: str, *, provider_name: str, model: str | None, fake: bool, trace: bool) -> dict:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        flat = mode == "flat"
        sdir = write_skills_dir(tmp / "skills", flat=flat)
        min_children = {"nested-min2": 2, "nested-min1": 1, "bundled": 999, "flat": 2}[mode]
        store = Store(str(tmp / f"{mode}.db"))
        tools, state = make_tools()
        if model and not fake:
            os.environ["GRAFT_MODEL"] = model
        provider = FakeProvider(_fake_policy()) if fake else make_provider(provider_name)
        nonce = f"run-{uuid.uuid4().hex[:8]}"
        agent = GraftAgent(
            store,
            provider,
            tools=tools,
            config=GraftConfig(
                persona=BASE_SYSTEM,
                skills_dir=sdir,
                min_children_to_fork=min_children,
                seeder=None,
                nonce=nonce,
                max_steps=12,
                trace=(lambda s: print(f"  [{mode}] {s}")) if trace else None,
            ),
        )
        t0 = time.monotonic()
        replies: list[str] = []
        async with agent:
            main_sys_chars = len(agent._host.system("o"))
            for turn in SCRIPT:
                _task, turn_replies = await say(agent, turn)
                replies.append("\n".join(turn_replies))
        elapsed = time.monotonic() - t0

        # Branch depth analysis from SQLite
        tl_rows = store.con.execute("SELECT timeline_id, source_id, name FROM timelines").fetchall()
        parent_map = {r["timeline_id"]: r["source_id"] for r in tl_rows}
        depths = {1: 0, 2: 0}
        for tid, src in parent_map.items():
            if src is None:
                continue
            d = 1 if parent_map.get(src) is None else 2
            depths[d] = depths.get(d, 0) + 1

        usage = getattr(provider, "usage", None)
        calls_log = list(getattr(provider, "call_log", []))
        if isinstance(usage, dict):
            u = {
                "input": usage.get("input", 0),
                "cached": usage.get("cached", 0),
                "cache_write": usage.get("cache_write", 0),
                "uncached": usage.get("uncached", 0),
                "output": usage.get("output", 0),
                "calls": usage.get("calls", len(calls_log)),
                "peak_input": max((c.get("input", 0) for c in calls_log), default=0),
            }
        else:
            u = {"input": 0, "cached": 0, "cache_write": 0, "uncached": 0, "output": 0,
                 "calls": len(calls_log), "peak_input": 0}
        cache_pct = (100.0 * u["cached"] / u["input"]) if u["input"] else 0.0
        checks = evaluate(replies, state)
        return {
            "mode": mode,
            "score": f"{sum(checks.values())}/{len(checks)}",
            "passed": all(checks.values()),
            "checks": checks,
            "elapsed_s": round(elapsed, 2),
            "main_sys_chars": main_sys_chars,
            "calls": u["calls"],
            "branches_d1": depths.get(1, 0),
            "branches_d2": depths.get(2, 0),
            "tool_calls": len(state["calls"]),
            "input": u["input"],
            "cached": u["cached"],
            "cache_write": u["cache_write"],
            "cache_pct": round(cache_pct, 1),
            "uncached": u["uncached"],
            "output": u["output"],
            "peak_input": u["peak_input"],
            "est_cost": round(_est_cost(u), 4),
            "wakes": len(agent._host.wakes),
        }


def main() -> None:
    p = argparse.ArgumentParser(description="E2E experiment comparing nested skills modes.")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--modes", nargs="+", default=["nested-min2", "nested-min1", "bundled", "flat"])
    p.add_argument("--fake", action="store_true")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()

    results = []
    for mode in args.modes:
        print(f"\n=== Running mode: {mode} ===", flush=True)
        res = asyncio.run(run_mode(mode, provider_name=args.provider, model=args.model,
                                   fake=args.fake, trace=args.trace))
        results.append(res)
        print(json.dumps(res, indent=2), flush=True)

    print("\n=== Summary ===")
    hdr = (f"{'Mode':<13} {'Score':>5} {'Time':>6} {'Calls':>5} {'B(d1/d2)':>8} "
           f"{'MainSys':>7} {'Input':>8} {'Cached':>8} {'Hit%':>5} {'PeakIn':>7} {'Out':>6} {'Cost$':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        bstr = f"{r['branches_d1']}/{r['branches_d2']}"
        print(f"{r['mode']:<13} {r['score']:>5} {r['elapsed_s']:>5.1f}s {r['calls']:>5} "
              f"{bstr:>8} {r['main_sys_chars']:>7} "
              f"{r['input']:>8} {r['cached']:>8} {r['cache_pct']:>4.1f}% {r['peak_input']:>7} "
              f"{r['output']:>6} ${r['est_cost']:>6.4f}")


if __name__ == "__main__":
    main()
