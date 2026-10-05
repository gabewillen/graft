"""Real-trace replay pilot (spec: Validation Experiments #4).

For each recorded Claude Code delegation event (read-only subagents):
  S  subagent baseline: gpt-6-luna re-runs the subagent with the recorded handoff prompt only and
     read-only code tools in a detached worktree at the spawn-time commit; its final text is the
     result returned to the parent.
  G  graft: the parent prefix becomes the original's timeline; the recorded handoff is appended as
     the subtask row (a replay adaptation: no live user asked for this subtask, so the recorded
     handoff stands in as the trigger, owned by the original); the Jev seeder picks the seed; the
     clone works autonomously with the same tools, then joins(done) and the default join applies.
  R  recorded: the real subagent's final report (reference only).
Evaluation: 8-12 probes generated from the real subagent transcript (facts it found), answered by
the continued parent (no tools) after each arm's delegation; graded by a model judge plus exact
match on a short key; key findings of R checked against each arm's returned text.

Usage: uv run python experiments/replay/run.py --events a6ff6e,a44ae7,... --repo PATH --out DIR
Keys: OPENAI_API_KEY (and TYPESAFE_API_KEY for Jev) from the environment; --env-file paths are read
only to build the exact-value secret filter (values are never printed or written).
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay import loader, sandbox, secret_filter  # noqa: E402

from graft import cachestats, callsites  # noqa: E402
from graft.db import Skill, Store, render  # noqa: E402
from graft.harness import Harness  # noqa: E402
from graft.providers import OpenAIProvider, add_usage, empty_usage  # noqa: E402
from graft.seeder import JevSeeder  # noqa: E402
from graft.skills import BASE_SYSTEM, World, install  # noqa: E402

MODEL = "gpt-6-luna"
PRICING = cachestats.Pricing.from_env()
SUBAGENT_NOTE = ("You are a subagent working on the task below in a read-only checkout. Use the tools to investigate; "
                 "you cannot edit files or ask questions. When done, reply with your final report for the agent "
                 "that delegated the task.")
CODE_SKILL = Skill("code_read", "Read and search the repository checkout (read-only).",
                   "Code reading skill: Read, Grep, Glob and a read-only Bash (ls, cat, rg, grep, find, wc, head, "
                   "tail, git log/show/diff/ls-files). You cannot write files. Work from the task you were given, "
                   "then reply with your final report and call join(reason='done').", sandbox.TOOLS)


def cost_of(usage_by_model: dict) -> float | None:
    return PRICING.cost_by_model(usage_by_model)


# ------------------------------------------------------------------ worktree
def worktree_for(repo: str, e: loader.Event, root: str) -> tuple[str, str, str]:
    """(path, commit, how): detached worktree at the last commit before the spawn on the recorded branch,
    else on any ref."""
    def git(*a):
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()

    commit, how = "", ""
    if e.branch and git("rev-parse", "--verify", "--quiet", e.branch):
        commit, how = git("rev-list", "-1", f"--before={e.timestamp}", e.branch), f"branch {e.branch} before spawn"
    if not commit:
        commit, how = git("rev-list", "-1", f"--before={e.timestamp}", "--all"), \
            f"branch {e.branch!r} not in repo; last commit on any ref before spawn"
    path = os.path.join(root, f"wt-{e.agent_id[:14]}")
    if not os.path.exists(path):
        subprocess.run(["git", "-C", repo, "worktree", "add", "--detach", path, commit], check=True,
                       capture_output=True, text=True)
    return path, commit, how


def aliases_for(e: loader.Event, repo: str) -> list[str]:
    found = set(re.findall(r"/Users/[^\s'\"`]*?/emel\.cpp(?:/\.claude/worktrees/[A-Za-z0-9._-]+)?", e.handoff))
    return sorted(found | {repo} | ({e.cwd} if e.cwd else set()), key=len, reverse=True)


# ------------------------------------------------------------------ arms
async def arm_s(e, sb: sandbox.Sandbox, max_steps: int) -> dict:
    p = OpenAIProvider(MODEL)
    msgs = [{"role": "user", "content": [{"type": "text", "text": e.handoff}]}]
    system = BASE_SYSTEM + "\n\n" + SUBAGENT_NOTE
    t0, final, steps = time.perf_counter(), "", 0
    with p.usage_tag("subagent"):
        for steps in range(1, max_steps + 1):
            blocks = await asyncio.to_thread(p.complete, system, msgs, sandbox.TOOLS, 8000,
                                             effort=callsites.effort_for("subagent"), site="subagent")
            msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(empty)"}]})
            texts = [b["text"] for b in blocks if b["type"] == "text"]
            uses = [b for b in blocks if b["type"] == "tool_use"]
            if texts:
                final = "\n".join(texts)
            if not uses:
                break
            res = []
            for u in uses:
                out, err = await asyncio.to_thread(sb.execute, u["name"], u["input"])
                res.append({"type": "tool_result", "tool_use_id": u["id"], "content": out, "is_error": err})
            msgs.append({"role": "user", "content": res})
    return {"returned": final, "latency": time.perf_counter() - t0, "steps": steps,
            "usage": dict(p.usage), "by_model": {m: u for m, u in p.by_model.items() if u["calls"]}}


class SandboxWorld(World):
    def __init__(self, sb: sandbox.Sandbox):
        super().__init__()
        self.sb = sb

    def execute(self, name, args):
        if name in {t["name"] for t in sandbox.TOOLS}:
            return self.sb.execute(name, args)
        return super().execute(name, args)


async def arm_g(e, sb: sandbox.Sandbox, max_steps: int, workdir: str) -> dict:
    p = OpenAIProvider(MODEL)
    store = Store(os.path.join(workdir, f"g-{e.agent_id[:14]}-{uuid.uuid4().hex[:6]}.db"))
    install(store)
    store.add_skill(CODE_SKILL)
    h = Harness(store, p, world=SandboxWorld(sb), max_steps=max_steps, nonce=uuid.uuid4().hex[:12],
                seeder=JevSeeder())
    await h.start()
    for t in e.parent_prefix:
        store.add_turn("o", t.role, "text", t.text)
    store.add_turn("o", "user", "text", subtask_text(e))
    t0 = time.perf_counter()
    child, text = await h.clone_once("o", {"id": "replay", "name": "clone", "input": {
        "purpose": f"{e.description}: {e.handoff[:1500]}", "skills": ["code_read"]}}, retry_left=False)
    forced = False
    if child:
        await h.run(child)
        if h.machines.state(child) == "active":  # no live user: end the task if it stopped without joining
            forced = True
            await h.do_join(child, "done")
    lat = time.perf_counter() - t0
    view = store.working_view("o")
    seed = [r for r in store.view(child) if r["owner_timeline"] != child] if child else []
    joined = [r for r in view if r["owner_timeline"] == child]
    by_model = {m: u for m, u in p.by_model.items() if u["calls"]}
    slog = h.seeder_log[0] if h.seeder_log else {}
    if slog.get("input_tokens"):
        su = empty_usage()
        su.update(calls=slog.get("requests") or 1, uncached=slog["input_tokens"], input=slog["input_tokens"])
        by_model[slog.get("model") or "jev-1.13.0"] = su
    await h.machines.stop()
    return {"returned": "\n\n".join(r["content"] for r in joined), "latency": lat, "clone": child, "clone_text": text,
            "forced_join": forced, "seed_rows": len(seed), "seed_tokens": sum(len(r["content"]) // 4 for r in seed),
            "candidates": len(slog.get("candidates") or []), "seeder": {k: slog.get(k) for k in
                                                                        ("latency", "input_tokens", "fallback", "error")},
            "joined_rows": len(joined), "usage": dict(p.usage), "by_model": by_model,
            "parent_messages": render(view)}


def subtask_text(e) -> str:
    return f"Subtask delegated by the agent ({e.description}):\n{e.handoff}"


def parent_messages(e, report: str) -> list[dict]:
    msgs = [{"role": t.role, "content": [{"type": "text", "text": t.text}]} for t in e.parent_prefix]
    msgs.append({"role": "user", "content": [{"type": "text", "text": subtask_text(e)}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": f"Subagent report:\n{report}"}]})
    return msgs


# ------------------------------------------------------------------ eval calls
def is_messages(payload) -> bool:
    return isinstance(payload, list) and bool(payload) and all(isinstance(x, dict) and "role" in x for x in payload)


class Eval:
    def __init__(self):
        import openai

        self.client = openai.AsyncOpenAI()
        self.usage = empty_usage()

    async def structured(self, instructions: str, payload, schema: dict, effort: str = "low", name: str = "out"):
        r = await self.client.responses.create(
            model=MODEL, instructions=instructions,
            input=payload if is_messages(payload) else json.dumps(payload),
            store=False, reasoning={"effort": effort}, max_output_tokens=12000,
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}})
        u = r.usage
        cached = u.input_tokens_details.cached_tokens if u.input_tokens_details else 0
        add_usage(self.usage, {"calls": 1, "uncached": u.input_tokens - cached, "cached": cached, "cache_write": 0,
                               "output": u.output_tokens, "input": u.input_tokens})
        return json.loads(r.output_text), {"input": u.input_tokens, "cached": cached, "output": u.output_tokens}


PROBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["probes", "key_findings"], "properties": {
    "probes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                          "required": ["question", "reference", "exact"],
                                          "properties": {"question": {"type": "string"}, "reference": {"type": "string"},
                                                         "exact": {"type": "string"}}}},
    "key_findings": {"type": "array", "items": {"type": "string"}}}}
ANSWER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"], "properties": {
    "answers": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "answer"],
                                          "properties": {"id": {"type": "integer"}, "answer": {"type": "string"}}}}}}
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"], "properties": {
    "verdicts": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "verdict"],
                                           "properties": {"id": {"type": "integer"},
                                                          "verdict": {"type": "string",
                                                                      "enum": ["correct", "partial", "incorrect"]}}}}}}
FIND_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["found"], "properties": {
    "found": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "found"],
                                        "properties": {"id": {"type": "integer"}, "found": {"type": "boolean"}}}}}}


def transcript_for_probes(e, cap: int = 120000) -> str:
    parts = [f"TASK GIVEN TO THE SUBAGENT:\n{e.handoff}"]
    for r in e.recorded_turns[1:]:
        t = r["text"] if r["kind"] != "tool_result" else r["text"][:1500]
        label = {"text": f"{r['role'].upper()} TEXT", "tool_use": f"TOOL CALL {r.get('name')}", "tool_result": "RESULT"}[r["kind"]]
        parts.append(f"{label}: {t}")
    s = "\n\n".join(parts)
    return s[:cap]


async def make_probes(ev: Eval, e, n: int = 10) -> dict:
    instr = (f"From this recorded investigation by a coding subagent, write {n} probe questions that test whether "
             "someone who received its results knows the concrete facts it established about the repository: file "
             "paths, function/class/type names, numbers, and specific findings. Only facts the subagent read or found "
             "(not files it created or edited). Each probe: the question, a reference answer (one or two sentences), "
             "and `exact`: a short distinctive string that a correct answer must contain (a path, identifier or "
             "number), or '' if none. Also list the 5-8 key findings of its final report as short statements.")
    out, _ = await ev.structured(instr, transcript_for_probes(e), PROBE_SCHEMA, effort="medium", name="probes")
    return out


async def answer(ev: Eval, msgs: list[dict], probes: list[dict]) -> tuple[list[str], dict]:
    q = "\n".join(f"{i}. {p['question']}" for i, p in enumerate(probes))
    items = []
    for m in msgs:
        text = "\n".join(b.get("text", "") for b in m["content"] if b.get("type") == "text")
        if text:
            items.append({"role": m["role"], "content": text})
    items.append({"role": "user", "content": "Answer each question using only what this conversation contains (no "
                                             "tools). Be specific (paths, names, numbers). If the conversation does "
                                             f"not contain the answer, answer 'unknown'.\n{q}"})
    out, u = await ev.structured(BASE_SYSTEM, items, ANSWER_SCHEMA, name="answers")
    ans = {a["id"]: a["answer"] for a in out["answers"]}
    return [ans.get(i, "") for i in range(len(probes))], u


async def judge(ev: Eval, probes: list[dict], answers: list[str]) -> list[str]:
    items = [{"id": i, "question": p["question"], "reference": p["reference"], "answer": a}
             for i, (p, a) in enumerate(zip(probes, answers))]
    out, _ = await ev.structured("Grade each answer against the reference: correct (states the reference fact), "
                                 "partial (some of it, or vaguer), incorrect (wrong, missing or 'unknown').",
                                 items, JUDGE_SCHEMA, name="verdicts")
    v = {x["id"]: x["verdict"] for x in out["verdicts"]}
    return [v.get(i, "incorrect") for i in range(len(probes))]


async def findings_covered(ev: Eval, findings: list[str], returned: str) -> list[bool]:
    if not findings:
        return []
    out, _ = await ev.structured("For each key finding, say whether the report states it (same substance).",
                                 {"report": returned[:60000], "findings": [{"id": i, "finding": f}
                                                                           for i, f in enumerate(findings)]},
                                 FIND_SCHEMA, name="found")
    v = {x["id"]: x["found"] for x in out["found"]}
    return [bool(v.get(i)) for i in range(len(findings))]


def score(probes: list[dict], answers: list[str], verdicts: list[str]) -> dict:
    pts = [1.0 if v == "correct" else 0.5 if v == "partial" else 0.0 for v in verdicts]
    exact = [bool(p["exact"]) and p["exact"].lower() in a.lower() for p, a in zip(probes, answers)]
    n_exact = sum(1 for p in probes if p["exact"])
    return {"judge": sum(pts) / max(1, len(pts)), "exact": sum(exact) / max(1, n_exact), "n": len(probes),
            "unknown": sum(1 for a in answers if a.strip().lower().startswith("unknown"))}


# ------------------------------------------------------------------ main
def find_subagent(projects: str, prefix: str) -> str:
    hits = glob.glob(f"{projects}/-Users-gabrielwillen-VSCode-stateforward-emel-emel-cpp*/*/subagents/agent-{prefix}*.jsonl")
    if len(hits) != 1:
        raise SystemExit(f"expected one transcript for {prefix}, found {len(hits)}")
    return hits[0]


async def run_event(e, a, ev: Eval, known: list[str]) -> dict:
    wt, commit, how = worktree_for(a.repo, e, a.out)
    sb_s = sandbox.Sandbox(wt, aliases_for(e, a.repo), known)
    sb_g = sandbox.Sandbox(wt, aliases_for(e, a.repo), known)
    probes = a.saved_probes.get(e.agent_id) or await make_probes(ev, e, a.probes)
    s, g = await asyncio.gather(arm_s(e, sb_s, a.max_steps), arm_g(e, sb_g, a.max_steps, a.out))
    rec = {"event": e.agent_id, "description": e.description, "type": e.subagent_type, "commit": commit, "commit_how": how,
           "prefix_turns": len(e.parent_prefix), "prefix_chars": sum(len(t.text) for t in e.parent_prefix),
           "recorded_tools": e.recorded_tools, "probes": probes}
    contexts = {"S": parent_messages(e, s["returned"]), "G": g.pop("parent_messages"),
                "R": parent_messages(e, e.final_report)}
    returned = {"S": s["returned"], "G": g["returned"], "R": e.final_report}
    for arm, msgs in contexts.items():
        ans, u = await answer(ev, msgs, probes["probes"])
        verdicts = await judge(ev, probes["probes"], ans)
        found = await findings_covered(ev, probes["key_findings"], returned[arm])
        rec[arm] = {"answers": ans, "verdicts": verdicts, "score": score(probes["probes"], ans, verdicts),
                    "findings_found": sum(found), "findings_total": len(found), "parent_answer_usage": u,
                    "returned_chars": len(returned[arm])}
    for arm, r in (("S", s), ("G", g)):
        rec[arm].update({k: v for k, v in r.items() if k not in ("returned",)})
        rec[arm]["delegation_cost"] = cost_of(r["by_model"])
        rec[arm]["parent_answer_cost"] = cost_of({MODEL: {"calls": 1, "uncached": rec[arm]["parent_answer_usage"]["input"]
                                                           - rec[arm]["parent_answer_usage"]["cached"],
                                                           "cached": rec[arm]["parent_answer_usage"]["cached"],
                                                           "cache_write": 0, "output": rec[arm]["parent_answer_usage"]["output"]}})
    rec["sandbox"] = {"S_refusals": sb_s.refusals, "G_refusals": sb_g.refusals,
                      "redactions": sorted(set(sb_s.redactions + sb_g.redactions))}
    return rec


def report(recs: list[dict]) -> None:
    print("\n================ REPLAY PILOT ================")
    head = ("event", "type", "probe acc S / G / R", "exact S / G / R", "findings S / G / R", "deleg $ S / G",
            "latency s S / G", "seed rows/tok (of cands)", "notes")
    print(" | ".join(head))
    for r in recs:
        f = lambda arm, k: f"{r[arm]['score'][k]:.2f}"
        notes = []
        if r["G"].get("forced_join"):
            notes.append("G forced join")
        if r["G"]["seeder"].get("fallback"):
            notes.append("seeder fallback")
        print(" | ".join([r["event"][:14], r["type"], f"{f('S','judge')} / {f('G','judge')} / {f('R','judge')}",
                          f"{f('S','exact')} / {f('G','exact')} / {f('R','exact')}",
                          f"{r['S']['findings_found']}/{r['G']['findings_found']}/{r['R']['findings_found']} of "
                          f"{r['R']['findings_total']}",
                          f"{r['S']['delegation_cost']:.4f} / {r['G']['delegation_cost']:.4f}",
                          f"{r['S']['latency']:.0f} / {r['G']['latency']:.0f}",
                          f"{r['G']['seed_rows']} / {r['G']['seed_tokens']} ({r['G']['candidates']})", ", ".join(notes)]))
    if recs:
        m = lambda arm, k: statistics.mean(r[arm]["score"][k] for r in recs)
        print(f"AGGREGATE (n={len(recs)}): probe accuracy S {m('S','judge'):.2f} / G {m('G','judge'):.2f} / R "
              f"{m('R','judge'):.2f}; exact S {m('S','exact'):.2f} / G {m('G','exact'):.2f} / R {m('R','exact'):.2f}; "
              f"delegation $ S {sum(r['S']['delegation_cost'] for r in recs):.4f} / G "
              f"{sum(r['G']['delegation_cost'] for r in recs):.4f}; latency S "
              f"{statistics.mean(r['S']['latency'] for r in recs):.0f}s / G {statistics.mean(r['G']['latency'] for r in recs):.0f}s")


async def amain(a) -> int:
    known = secret_filter.env_values(a.env_file)
    ev = Eval()
    recs, dropped = [], []
    os.makedirs(a.out, exist_ok=True)
    for prefix in a.events.split(","):
        e = loader.load_event(find_subagent(a.projects, prefix), a.max_prefix_chars)
        hits = secret_filter.scan(loader.event_texts(e), known)
        if hits:
            dropped.append({"event": e.agent_id, "rules": hits})
            print(f"DROPPED {e.agent_id}: secret filter matched {hits}")
            continue
        spent = (cost_of({MODEL: ev.usage}) or 0) + sum((r["S"]["delegation_cost"] or 0) + (r["G"]["delegation_cost"] or 0)
                                                        for r in recs)
        if spent > a.budget * 0.8:
            print(f"STOP: spend ${spent:.3f} near the ${a.budget} cap")
            break
        t0 = time.perf_counter()
        rec = await run_event(e, a, ev, known)
        print(f"event {e.agent_id[:14]} done in {time.perf_counter() - t0:.0f}s: S {rec['S']['score']} G {rec['G']['score']} "
              f"R {rec['R']['score']}")
        recs.append(rec)
        json.dump({"events": recs, "dropped": dropped, "eval_usage": ev.usage}, open(os.path.join(a.out, "replay.json"), "w"),
                  indent=1, default=str)
    report(recs)
    eval_cost = cost_of({MODEL: ev.usage}) or 0
    print(f"eval (probes, answers, judge) ${eval_cost:.4f}; total spend "
          f"${eval_cost + sum((r['S']['delegation_cost'] or 0) + (r['G']['delegation_cost'] or 0) for r in recs):.4f}")
    if dropped:
        print(f"dropped events: {dropped}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="comma list of agent id prefixes")
    ap.add_argument("--projects", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=tempfile.mkdtemp(prefix="replay-"))
    ap.add_argument("--env-file", action="append", default=[])
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-prefix-chars", type=int, default=40000)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--probes-from", default=None, help="reuse the probes of an earlier replay.json (same questions)")
    a = ap.parse_args()
    a.saved_probes = {}
    if a.probes_from:
        a.saved_probes = {r["event"]: r["probes"] for r in json.load(open(a.probes_from))["events"]}
    return asyncio.run(amain(a))


if __name__ == "__main__":
    raise SystemExit(main())
