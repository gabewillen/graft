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
import collections
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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay import loader, sandbox, secret_filter, worktree  # noqa: E402

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
EDIT_NOTE = ("You are a subagent working on the task below in a disposable checkout of the repository. Use the "
             "tools to investigate and make the changes (Edit/Write); you can run the test suite with "
             "`uv run pytest -q` (offline). You cannot ask questions. When done, reply with your final report for "
             "the agent that delegated the task: what you changed and the test result.")
EDIT_SKILL = Skill("code_edit", "Read, search and edit the repository checkout, and run its tests.",
                   "Code editing skill: Read, Grep, Glob, Edit, Write and a restricted Bash (ls, cat, rg, grep, find, "
                   "wc, head, tail, git log/show/diff/ls-files/status, and `uv run pytest`, offline). Make the change "
                   "the task asks for, run the tests, then reply with your final report (what you changed and the "
                   "test result) and call join(reason='done').", sandbox.TOOLS + sandbox.WRITE_TOOLS)
TEST_PREFIXES = [["uv", "run", "pytest"], ["uv", "run", "--frozen", "--offline", "pytest"]]
TEST_CMD = ["uv", "run", "--frozen", "--offline", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not live"]


def select_test_paths(paths: list[str], ref_files: list[str], root: str) -> tuple[list[str], list[str]]:
    """(paths, extra args): the recorded subagent's targeted test paths that exist at the base; if it only
    ran the whole suite, the test directories its reference diff touched; else the whole suite without
    tests/examples (its providers need packages that are not available offline), stopping at the first failure."""
    exist = [p for p in paths if os.path.exists(os.path.join(root, p))]
    specific = [p for p in exist if p != "tests"]
    specific = [p for p in specific if not any(p != q and p.startswith(q + "/") for q in specific)]
    if specific:
        return specific[:6], []
    dirs = sorted({os.path.dirname(f) for f in ref_files if f.startswith("tests/")})
    dirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
    if dirs:
        return dirs[:6], []
    if os.path.isdir(os.path.join(root, "tests")):
        return ["tests"], ["--ignore=tests/examples", "-x"]
    return [], []


def recorded_test_paths(e) -> list[str]:
    """Test paths the recorded subagent passed to pytest (e.g. tests/devices), most used first. The full
    suite does not finish offline within the time limit, so each event runs the tests its own subagent ran."""
    import collections as _c

    seen: _c.Counter = _c.Counter()
    for r in e.recorded_turns:
        if r["kind"] != "tool_use" or r.get("name") != "Bash":
            continue
        cmd = json.loads(r["text"]).get("command", "")
        for m in re.finditer(r"pytest([^\n;&|>]*)", cmd):
            for tok in m.group(1).split():
                tok = tok.strip("'\"")
                if tok.startswith("tests") and not tok.startswith("--"):
                    seen[tok.split("::")[0].rstrip("/")] += 1
    return [p for p, _ in seen.most_common()]
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
    system = BASE_SYSTEM + "\n\n" + (EDIT_NOTE if sb.writable else SUBAGENT_NOTE)
    t0, final, steps = time.perf_counter(), "", 0
    with p.usage_tag("subagent"):
        for steps in range(1, max_steps + 1):
            blocks = await asyncio.to_thread(p.complete, system, msgs, sb.tools(), 8000,
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
        if name in {t["name"] for t in self.sb.tools()}:
            return self.sb.execute(name, args)
        return super().execute(name, args)


async def arm_g(e, sb: sandbox.Sandbox, max_steps: int, workdir: str) -> dict:
    p = OpenAIProvider(MODEL)
    store = Store(os.path.join(workdir, f"g-{e.agent_id[:14]}-{uuid.uuid4().hex[:6]}.db"))
    install(store)
    skill = EDIT_SKILL if sb.writable else CODE_SKILL
    store.add_skill(skill)
    h = Harness(store, p, world=SandboxWorld(sb), max_steps=max_steps, nonce=uuid.uuid4().hex[:12],
                seeder=JevSeeder())
    await h.start()
    for t in e.parent_prefix:
        store.add_turn("o", t.role, "text", t.text)
    store.add_turn("o", "user", "text", subtask_text(e))
    t0 = time.perf_counter()
    child, text = await h.clone_once("o", {"id": "replay", "name": "clone", "input": {
        "purpose": f"{e.description}: {e.handoff[:1500]}", "skills": [skill.name]}}, retry_left=False)
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


async def make_probes(ev: Eval, e, n: int = 10, editing: bool = False) -> dict:
    scope = ("the concrete facts it established and the changes it made: files and functions it changed and how, "
             "behaviour it added or fixed, numbers, and test results" if editing else
             "the concrete facts it established about the repository: file paths, function/class/type names, numbers, "
             "and specific findings. Only facts the subagent read or found (not files it created or edited)")
    instr = (f"From this recorded work by a coding subagent, write {n} probe questions that test whether someone who "
             f"received its results knows {scope}. Each probe: the question, a reference answer (one or two sentences), "
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
def find_subagent(projects: str, prefix: str, project_glob: str = "-Users-gabrielwillen-VSCode-stateforward-emel-emel-cpp*") -> str:
    hits = glob.glob(f"{projects}/{project_glob}/*/subagents/agent-{prefix}*.jsonl")
    if len(hits) != 1:
        raise SystemExit(f"expected one transcript for {prefix}, found {len(hits)}")
    return hits[0]


def redact_event(e) -> dict:
    """Redact secret-looking NAME=value assignments in every text of the event that can be sent.
    Returns counts by rule name."""
    n = 0
    for t in e.parent_prefix:
        t.text, k = secret_filter.redact_assignments(t.text)
        n += k
    e.handoff, k = secret_filter.redact_assignments(e.handoff)
    n += k
    e.final_report, k = secret_filter.redact_assignments(e.final_report)
    n += k
    for r in e.recorded_turns:
        r["text"], k = secret_filter.redact_assignments(r["text"])
        n += k
    return {"secret_assignment": n} if n else {}


def is_editing(e) -> bool:
    return e.subagent_type != "Explore"


def reference_diff(e, root_alias: str) -> str:
    """A pseudo unified diff of the recorded subagent's Edit/Write calls ('' if it made none)."""
    files: dict[str, list[str]] = {}
    for r in e.recorded_turns:
        if r["kind"] != "tool_use" or r.get("name") not in ("Edit", "Write", "MultiEdit"):
            continue
        try:
            inp = json.loads(r["text"])
        except ValueError:
            continue
        path = str(inp.get("file_path", ""))
        rel = os.path.relpath(path, root_alias) if path.startswith(root_alias.rstrip("/") + "/") else path.lstrip("/")
        lines = files.setdefault(rel, [])
        if r["name"] == "Write":
            lines += ["+" + x for x in str(inp.get("content", "")).splitlines()]
        else:
            edits = inp.get("edits") or [inp]
            for ed in edits:
                lines += ["-" + x for x in str(ed.get("old_string", "")).splitlines()]
                lines += ["+" + x for x in str(ed.get("new_string", "")).splitlines()]
    return "".join(f"diff --git a/{f} b/{f}\n--- a/{f}\n+++ b/{f}\n" + "\n".join(ls) + "\n"
                   for f, ls in files.items())


def strip_subdir(diff: str, subdir: str) -> str:
    if not subdir:
        return diff
    pre = subdir.rstrip("/") + "/"
    return diff.replace(f" a/{pre}", " a/").replace(f" b/{pre}", " b/")


def test_summary(out: str) -> dict:
    rc = int(re.match(r"\(exit (-?\d+)\)", out).group(1)) if out.startswith("(exit") else None
    tail = [ln for ln in out.splitlines() if re.search(r"\b(passed|failed|error|errors|no tests ran)\b", ln)]
    return {"returncode": rc, "summary": tail[-1].strip()[:200] if tail else out.splitlines()[-1][:200] if out else ""}


def live_agent_branch(repo: str, e) -> str | None:
    """The recorded subagent's own worktree branch, only if it was committed to during or after the
    spawn (stale branches from earlier work are ignored)."""
    branch = f"worktree-agent-{e.agent_id}"
    if not worktree.git(repo, "rev-parse", "--verify", "--quiet", branch):
        return None
    tip = worktree.git(repo, "show", "-s", "--format=%cI", branch)
    spawn = datetime.fromisoformat(e.timestamp.replace("Z", "+00:00"))
    return branch if tip and datetime.fromisoformat(tip) >= spawn else None


def edit_aliases(e, superrepo: str, info: dict, orig_root: str) -> list[str]:
    """Recorded paths that mean "the checkout root": the parent's cwd, the original repo (or submodule)
    path, and the recorded subagent's own worktree (`.../.claude/worktrees/agent-<id>`, created under the
    superproject or under the submodule), with the subdir appended where the checkout is the superproject."""
    agent = f".claude/worktrees/agent-{e.agent_id}"
    cands = {e.cwd, orig_root, os.path.join(e.cwd or orig_root, agent)}
    if info["repo"] == superrepo and info["subdir"]:
        cands.add(os.path.join(superrepo, agent, info["subdir"]))
    else:
        cands.add(os.path.join(superrepo, agent))
        sub_rel = os.path.relpath(info["repo"], superrepo) if info["repo"].startswith(superrepo) else ""
        if sub_rel and sub_rel != ".":
            cands.add(os.path.join(superrepo, agent, sub_rel))
            cands.add(os.path.join(info["repo"], agent))
    return sorted({c for c in cands if c}, key=len, reverse=True)


def reference_diff_for(e, info: dict, aliases: list[str]) -> tuple[str, str]:
    """The recorded subagent's change: its agent branch diff when the branch exists (including the
    submodule's diff when the branch bumps a gitlink), else a pseudo diff of its Edit/Write calls that
    landed in the checkout (scripts written elsewhere are ignored)."""
    repo = info["repo"]
    branch = live_agent_branch(repo, e)
    if branch:
        tip = worktree.git(repo, "rev-parse", branch)
        d = worktree.git(repo, "diff", "--no-ext-diff", info["commit"], tip, "--", info["subdir"] or ".")
        for line in worktree.git(repo, "diff", "--raw", info["commit"], tip).splitlines():
            parts = line.split()
            if len(parts) >= 6 and parts[0].startswith(":160000") and parts[1] == "160000":
                sub = parts[5]
                sub_repo = os.path.join(repo, sub)
                if os.path.isdir(sub_repo):
                    sd = worktree.git(sub_repo, "diff", "--no-ext-diff", parts[2], parts[3])
                    d += "\n" + re.sub(r"(?m)^(diff --git a/)(\S+) b/", lambda m: f"{m.group(1)}{sub}/{m.group(2)} b/{sub}/", sd)
        d = strip_subdir(d, info["subdir"])
        return d, f"agent branch {branch} ({tip[:10]}) vs base"
    best = ""
    for al in aliases:
        cand = reference_diff(e, al)
        files = worktree.diff_files(cand)
        if files and all(not f.startswith(("/", "private/", "Users/", ".claude/")) for f in files):
            best = cand
            break
    if not best:
        full = reference_diff(e, aliases[0]) if aliases else ""
        keep = [blk for blk in re.split(r"(?=^diff --git )", full, flags=re.M)
                if blk and not re.match(r"diff --git a/(?:/|private/|Users/|\.claude/)", blk)]
        best = "".join(keep)
    return best, ("Edit/Write calls in the checkout" if best else "no reference (edits made through Bash scripts)")


def session_start(e) -> str:
    for d in loader.read_jsonl(e.parent_path):
        if d.get("timestamp"):
            return d["timestamp"]
    return ""


async def run_edit_event(e, a, ev: Eval, known: list[str]) -> dict:
    info = worktree.resolve_checkout(a.repo, e.cwd, e.branch, e.timestamp)
    rec = {"event": e.agent_id, "description": e.description, "type": e.subagent_type, "commit": info["commit"],
           "commit_how": info["how"], "checkout_repo": info["repo"], "subdir": info["subdir"],
           "prefix_turns": len(e.parent_prefix), "prefix_chars": sum(len(t.text) for t in e.parent_prefix),
           "recorded_tools": e.recorded_tools}
    if not info["commit"]:
        return dict(rec, skipped="recorded commit not resolvable")
    if info["subdir"] and not worktree.git(info["repo"], "ls-tree", info["commit"], "--", info["subdir"]):
        return dict(rec, skipped=f"subdir {info['subdir']} missing at the resolved commit")
    for repo_c in (info["repo"], a.repo):
        agent_branch = live_agent_branch(repo_c, e)
        if agent_branch and e.branch and worktree.git(repo_c, "rev-parse", "--verify", "--quiet", e.branch):
            fork = worktree.git(repo_c, "merge-base", agent_branch, e.branch)
            if fork:
                info = {"repo": repo_c, "subdir": worktree.split_recorded_cwd(a.repo, e.cwd) if repo_c == a.repo
                        else info["subdir"], "commit": fork,
                        "how": f"fork point of the recorded agent branch {agent_branch} from {e.branch}"}
                rec.update(commit=fork, commit_how=info["how"], checkout_repo=repo_c, subdir=info["subdir"])
                rec["agent_branch"] = agent_branch
                break
    when = worktree.git(info["repo"], "show", "-s", "--format=%cI", info["commit"])
    rec["base_commit_time"] = when
    rec["base"] = "faithful" if rec.get("agent_branch") or (when and session_start(e) and datetime.fromisoformat(when)
                                                             >= datetime.fromisoformat(session_start(e).replace("Z", "+00:00"))) \
        else "approximate"
    orig_root = os.path.join(info["repo"], info["subdir"]) if info["subdir"] else info["repo"]
    aliases = edit_aliases(e, a.repo, info, orig_root)
    paths = {arm: os.path.join(a.out, f"wt-{e.agent_id[:14]}-{arm}") for arm in ("S", "G")}
    wts = {arm: worktree.DisposableWorktree(info["repo"], info["commit"], p) for arm, p in paths.items()}
    try:
        entered = {arm: w.__enter__() for arm, w in wts.items()}
        roots = {arm: os.path.join(p, info["subdir"]) if info["subdir"] else p for arm, p in entered.items()}
        rec["tracked_env"] = wts["S"].tracked_env
        rec["submodules"] = wts["S"].submodules
        ref, ref_how = reference_diff_for(e, info, aliases)
        rec["reference"] = ref_how
        test_paths, extra = select_test_paths(recorded_test_paths(e), sorted(worktree.diff_files(ref)), roots["S"])
        has_tests = os.path.exists(os.path.join(roots["S"], "pyproject.toml")) and bool(test_paths)
        rec["test_paths"] = test_paths + extra
        test_cmd = TEST_CMD[:5] + test_paths + TEST_CMD[5:] + extra
        tests = TEST_PREFIXES if os.path.exists(os.path.join(roots["S"], "pyproject.toml")) else []
        sbs = {arm: sandbox.Sandbox(r, aliases, known, writable=True, test_commands=tests) for arm, r in roots.items()}
        probes = a.saved_probes.get(e.agent_id) or await make_probes(ev, e, a.probes, editing=True)
        rec["probes"] = probes
        if has_tests:
            before = await asyncio.to_thread(sbs["S"].run_tests, test_cmd)
            rec["tests_before"] = test_summary(before)
        s, g = await asyncio.gather(arm_s(e, sbs["S"], a.max_steps), arm_g(e, sbs["G"], a.max_steps, a.out))
        rec["reference_diff_files"] = sorted(worktree.diff_files(ref))
        for arm, r, sb in (("S", s, sbs["S"]), ("G", g, sbs["G"])):
            if has_tests:
                r["tests_after"] = test_summary(await asyncio.to_thread(sb.run_tests, test_cmd))
            d = strip_subdir(wts[arm].diff(), info["subdir"])
            r["diff_files"] = sorted(worktree.diff_files(d))
            r["diff_lines"] = sum(1 for ln in d.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---")))
            r["diff_similarity"] = ({k: v for k, v in worktree.diff_similarity(d, ref).items()
                                     if k in ("files_jaccard", "line_similarity")} if ref else None)
            r["writes"] = len(sb.writes)
            r["test_runs_by_arm"] = len(sb.test_runs) - (2 if arm == "S" and has_tests else 1 if has_tests else 0)
        rec.update(await evaluate(ev, e, probes, s, g, sbs["S"], sbs["G"]))
    finally:
        for w in wts.values():
            w.remove()
    rec["worktrees_removed"] = all(not os.path.exists(p) for p in paths.values())
    return rec


async def evaluate(ev: Eval, e, probes: dict, s: dict, g: dict, sb_s, sb_g) -> dict:
    rec: dict = {}
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
        pu = rec[arm]["parent_answer_usage"]
        rec[arm]["parent_answer_cost"] = cost_of({MODEL: {"calls": 1, "uncached": pu["input"] - pu["cached"],
                                                           "cached": pu["cached"], "cache_write": 0,
                                                           "output": pu["output"]}})
    rec["sandbox"] = {"S_refusals": sb_s.refusals, "G_refusals": sb_g.refusals,
                      "redactions": sorted(set(sb_s.redactions + sb_g.redactions)),
                      "redaction_hits": sb_s.redactions + sb_g.redactions,
                      "network_denied_tests": any(t.get("network_denied") for t in sb_s.test_runs + sb_g.test_runs)}
    return rec


async def run_readonly_disposable(e, a, ev: Eval, known: list[str]) -> dict:
    """Probes-only event (Explore) on a disposable read-only worktree at the resolved commit."""
    info = worktree.resolve_checkout(a.repo, e.cwd, e.branch, e.timestamp)
    rec = {"event": e.agent_id, "description": e.description, "type": e.subagent_type, "commit": info["commit"],
           "commit_how": info["how"], "prefix_turns": len(e.parent_prefix),
           "prefix_chars": sum(len(t.text) for t in e.parent_prefix), "recorded_tools": e.recorded_tools,
           "base": "readonly"}
    if not info["commit"]:
        return dict(rec, skipped="recorded commit not resolvable")
    orig_root = os.path.join(info["repo"], info["subdir"]) if info["subdir"] else info["repo"]
    aliases = sorted({x for x in (e.cwd, orig_root) if x}, key=len, reverse=True)
    path = os.path.join(a.out, f"wt-{e.agent_id[:14]}-RO")
    w = worktree.DisposableWorktree(info["repo"], info["commit"], path)
    try:
        base_path = w.__enter__()
        if not os.path.isdir(base_path):
            raise RuntimeError("worktree was not created")
        root = os.path.join(base_path, info["subdir"]) if info["subdir"] else base_path
        rec["submodules"] = w.submodules
        sb_s, sb_g = sandbox.Sandbox(root, aliases, known), sandbox.Sandbox(root, aliases, known)
        probes = a.saved_probes.get(e.agent_id) or await make_probes(ev, e, a.probes)
        rec["probes"] = probes
        s, g = await asyncio.gather(arm_s(e, sb_s, a.max_steps), arm_g(e, sb_g, a.max_steps, a.out))
        rec.update(await evaluate(ev, e, probes, s, g, sb_s, sb_g))
    finally:
        w.remove()
    rec["worktrees_removed"] = not os.path.exists(path)
    return rec


async def run_event(e, a, ev: Eval, known: list[str]) -> dict:
    if a.mode == "auto" and is_editing(e):
        return await run_edit_event(e, a, ev, known)
    if a.mode == "auto":
        return await run_readonly_disposable(e, a, ev, known)
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


def report_edit(recs: list[dict]) -> None:
    print("\n================ EDITING EVENTS ================")
    print("event | description | base | tests before -> after S / G | diff sim S / G (files, lines) | probe acc S / G / R"
          " | deleg $ S / G | latency s S / G | seed rows/tok (cands) | forced join G")
    for r in recs:
        if "tests_before" not in r and r.get("base") == "readonly":
            continue
        tb = r.get("tests_before", {}).get("returncode")
        ta = lambda arm: r[arm].get("tests_after", {}).get("returncode")
        ds = lambda arm: (f"{r[arm]['diff_similarity']['files_jaccard']:.2f}/{r[arm]['diff_similarity']['line_similarity']:.2f}"
                          if r[arm].get("diff_similarity") else "n/a")
        sc = lambda arm: f"{r[arm]['score']['judge']:.2f}"
        print(" | ".join([r["event"][:14], r["description"][:34], r.get("base", "?"),
                          f"{tb} -> {ta('S')} / {ta('G')}", f"{ds('S')} / {ds('G')}",
                          f"{sc('S')} / {sc('G')} / {sc('R')}",
                          f"{r['S']['delegation_cost']:.4f} / {r['G']['delegation_cost']:.4f}",
                          f"{r['S']['latency']:.0f} / {r['G']['latency']:.0f}",
                          f"{r['G']['seed_rows']} / {r['G']['seed_tokens']} ({r['G']['candidates']})",
                          str(r["G"].get("forced_join"))]))
    for base in ("faithful", "approximate"):
        rs = [r for r in recs if r.get("base") == base]
        if not rs:
            continue
        m = lambda arm: statistics.mean(r[arm]["score"]["judge"] for r in rs)
        passed = lambda arm: sum(1 for r in rs if r[arm].get("tests_after", {}).get("returncode") == 0)
        sims = lambda arm, k: [r[arm]["diff_similarity"][k] for r in rs if r[arm].get("diff_similarity")]
        msim = lambda arm, k: f"{statistics.mean(sims(arm, k)):.2f}" if sims(arm, k) else "n/a"
        print(f"AGGREGATE {base} (n={len(rs)}): probe acc S {m('S'):.2f} / G {m('G'):.2f} / R {m('R'):.2f}; tests pass "
              f"after S {passed('S')}/{len(rs)} G {passed('G')}/{len(rs)} (before {sum(1 for r in rs if r.get('tests_before', {}).get('returncode') == 0)}); "
              f"diff files-jaccard S {msim('S', 'files_jaccard')} / G {msim('G', 'files_jaccard')}, line sim S "
              f"{msim('S', 'line_similarity')} / G {msim('G', 'line_similarity')}; delegation $ S "
              f"{sum(r['S']['delegation_cost'] for r in rs):.4f} / G {sum(r['G']['delegation_cost'] for r in rs):.4f}")


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
    redaction_totals: collections.Counter = collections.Counter()
    skipped: list[dict] = []
    os.makedirs(a.out, exist_ok=True)
    for prefix in a.events.split(","):
        e = loader.load_event(find_subagent(a.projects, prefix, a.project_glob), a.max_prefix_chars)
        hits = secret_filter.scan(loader.event_texts(e), known)
        redacted = redact_event(e)
        redaction_totals.update(redacted)
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
        try:
            rec = await run_event(e, a, ev, known)
        except Exception as ex:  # mechanical failure: record and continue (or stop in smoke mode)
            import traceback

            rec = {"event": e.agent_id, "description": e.description, "error": f"{type(ex).__name__}: {ex}",
                   "traceback": traceback.format_exc()[-2000:]}
            print(f"FAILED {e.agent_id[:14]}: {rec['error']}")
            skipped.append(rec)
            if a.stop_on_error:
                break
            continue
        rec["redactions"] = redacted
        if rec.get("skipped"):
            print(f"SKIPPED {e.agent_id[:14]}: {rec['skipped']}")
            skipped.append(rec)
        else:
            print(f"event {e.agent_id[:14]} done in {time.perf_counter() - t0:.0f}s: S {rec['S']['score']} "
                  f"G {rec['G']['score']} R {rec['R']['score']}" + (f"; tests {rec.get('tests')}" if rec.get("tests") else ""))
            recs.append(rec)
        json.dump({"events": recs, "dropped": dropped, "skipped": skipped, "eval_usage": ev.usage,
                   "redactions": dict(redaction_totals)}, open(os.path.join(a.out, "replay.json"), "w"),
                  indent=1, default=str)
    report(recs)
    if any(r.get("base") in ("faithful", "approximate") for r in recs):
        report_edit(recs)
    eval_cost = cost_of({MODEL: ev.usage}) or 0
    print(f"eval (probes, answers, judge) ${eval_cost:.4f}; total spend "
          f"${eval_cost + sum((r['S']['delegation_cost'] or 0) + (r['G']['delegation_cost'] or 0) for r in recs):.4f}")
    if dropped:
        print(f"dropped events: {dropped}")
    if skipped:
        print("skipped/failed: " + "; ".join(f"{x['event'][:14]} {x.get('skipped') or x.get('error')}" for x in skipped))
    print(f"redactions by rule (event texts): {dict(redaction_totals)}; tool-output redactions: "
          f"{collections.Counter(h for r in recs for h in r.get('sandbox', {}).get('redaction_hits', []))}")
    return 0


def main() -> int:
    import signal

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
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
    ap.add_argument("--project-glob", default="-Users-gabrielwillen-VSCode-stateforward-emel-emel-cpp*")
    ap.add_argument("--mode", choices=("readonly", "auto"), default="readonly",
                    help="auto: Explore events read-only (probes only); other events writable (tests, diff, probes)")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()
    a.saved_probes = {}
    if a.probes_from:
        a.saved_probes = {r["event"]: r["probes"] for r in json.load(open(a.probes_from))["events"]}
    return asyncio.run(amain(a))


if __name__ == "__main__":
    raise SystemExit(main())
