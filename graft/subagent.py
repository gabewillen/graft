"""Subagent baseline: the usual delegation pattern, for comparison with graft.

The parent (same persona) talks to the user throughout and has one tool,
`delegate(skill, task_prompt)`. A subagent gets the persona plus a listing of that one skill
(progressive disclosure: it must call load_skill), sees ONLY the parent-written task_prompt,
does the work with the skill's tools, and returns its final text as the delegate tool_result.
The parent never sees the subagent's rows.

Where graft passes pointers (seed rows, join selection), this pattern generates text twice:
the parent writes the handoff prompt (output tokens, cold prefill in the subagent) and the
subagent writes a summary (output tokens, prefill in the parent).

Usage roles (provider tags): parent (parent calls), handoff (a parent call that emitted
delegate), subagent (subagent work calls), summary (the subagent's final, tool-free call).
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field

from . import compaction
from .db import Store
from .providers import Provider, prefix_nonce
from .skills import BASE_SYSTEM, CATALOG, LOAD_SKILL_TOOL, BUILTIN, World, load_skill, skill_listing
from .skills import obj

CAPABILITIES = [s.name for s in CATALOG if s.name not in BUILTIN]
SKILL_TOOLS = {s.name: s.tools for s in CATALOG}

DELEGATE_TOOL = {
    "name": "delegate",
    "description": ("Hand a task to a subagent that holds one skill. The subagent sees only "
                    "task_prompt, not this conversation, so include every fact it needs. It does the "
                    "work with that skill's tools and returns a summary of what it did."),
    "input_schema": obj({"skill": {"type": "string", "enum": CAPABILITIES},
                         "task_prompt": {"type": "string"}}, ["skill", "task_prompt"])}

SUBAGENT_NOTE = ("You are a subagent. You cannot talk to the user. Complete the task with your tools "
                 "using the facts in the task, then reply with a short summary of what you did for the "
                 "agent that delegated it.")


BATCHED_POLICY = (
    "Delegation policy: gather the requirements from the user yourself first, asking brief "
    "questions if needed. Delegate only when you have enough to produce the deliverable, not on "
    "every message. Each task_prompt must be self-contained: the subagent sees nothing else, so "
    "include every current requirement and fact the user gave, verbatim where possible. For a "
    "revision, delegate once per revision request with the full current requirements, including "
    "all earlier changes, plus the requested change.")


@dataclass
class Delegation:
    skill: str
    task_prompt: str
    summary: str = ""
    steps: int = 0
    loaded: list[str] = field(default_factory=list)
    first_call: dict | None = None  # usage of the subagent's first request (cache hit check)
    system: str = ""
    tools: list[dict] = field(default_factory=list)


@dataclass
class SubagentRunner:
    store: Store  # skill catalog only (descriptions/instructions); no timelines are used
    provider: Provider
    world: World = field(default_factory=World)
    persona: str = BASE_SYSTEM
    nonce: str | None = None
    max_steps: int = 10
    compact_tokens: int | None = None  # same compaction rule as the harness (graft/compaction.py)
    compactions: list[str] = field(default_factory=list)
    compaction_log: list[dict] = field(default_factory=list)
    batched: bool = False  # the `subagent_batched` baseline: gather first, delegate with full requirements
    trace: object = None
    messages: list[dict] = field(default_factory=list)
    delegations: list[Delegation] = field(default_factory=list)
    out: list[str] = field(default_factory=list)

    def parent_system(self) -> str:
        parts = [self.persona, skill_listing(
            self.store, CAPABILITIES, "Skills you can delegate (call delegate(skill, task_prompt)):")]
        if self.batched:
            parts.append(BATCHED_POLICY)
        return "\n\n".join(parts)

    def subagent_system(self, skill: str) -> str:
        return "\n\n".join([self.persona, skill_listing(
            self.store, [skill],
            "Your skills (call load_skill(name) to read a skill's full instructions before using it):"),
            SUBAGENT_NOTE])

    def _complete(self, tag: str, system: str, msgs: list[dict], tools: list[dict]) -> list[dict]:
        tagger = getattr(self.provider, "usage_tag", None)
        with tagger(tag) if tagger else contextlib.nullcontext():
            return self.provider.complete(system, prefix_nonce(msgs, self.nonce), tools)

    def _retag(self, tag: str) -> None:
        retag = getattr(self.provider, "retag_last", None)
        if retag:
            retag(tag)

    def _log(self, msg: str) -> None:
        if callable(self.trace):
            self.trace(msg)

    # ------------------------------------------------------------ parent
    async def user_says(self, text: str) -> list[str]:
        start = len(self.out)
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        self.messages = self._maybe_compact("parent", self.parent_system(), self.messages)
        for _ in range(self.max_steps):
            blocks = self._complete("parent", self.parent_system(), self.messages, [DELEGATE_TOOL])
            uses = [b for b in blocks if b["type"] == "tool_use"]
            if any(u["name"] == "delegate" for u in uses):
                self._retag("handoff")
            self.messages.append({"role": "assistant",
                                  "content": [b for b in blocks if b["type"] in ("text", "tool_use")]
                                  or [{"type": "text", "text": "(empty)"}]})
            self.out += [b["text"] for b in blocks if b["type"] == "text"]
            if not uses:
                return self.out[start:]
            results = []
            for u in uses:
                if u["name"] != "delegate":
                    results.append({"type": "tool_result", "tool_use_id": u["id"],
                                    "content": f"tool '{u['name']}' is not available to you", "is_error": True})
                    continue
                d = self.run_subagent(str(u["input"].get("skill", "")), str(u["input"].get("task_prompt", "")))
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": d.summary,
                                "is_error": False})
            self.messages.append({"role": "user", "content": results})
        return self.out[start:]

    def _maybe_compact(self, who: str, system: str, msgs: list[dict]) -> list[dict]:
        if len(msgs) < 2 or not compaction.needs_compaction(msgs, self.compact_tokens):
            return msgs
        # same rule as the harness: summarize everything before the fresh user message
        blocks = self._complete(f"{who}:compaction", system, compaction.request(msgs[:-1]), [])
        self.compactions.append(who)
        self._log(f"COMPACTED {who}")
        summary = compaction.summary_turn(blocks)
        new = [{"role": "user", "content": [{"type": "text", "text": summary}] + list(msgs[-1]["content"])}]
        self.compaction_log.append({
            "timeline": who, "before_tokens": compaction.estimate_tokens(msgs),
            "after_tokens": compaction.estimate_tokens(new), "rows_before": len(msgs),
            "prefix_texts": [str(b.get("text") or b.get("content") or json.dumps(b.get("input", "")))
                             for m in msgs[:-1] for b in m["content"]],
            "summary": summary, "fresh": " ".join(b.get("text", "") for b in msgs[-1]["content"])})
        return new

    # ------------------------------------------------------------ subagent
    def run_subagent(self, skill: str, task_prompt: str) -> Delegation:
        d = Delegation(skill, task_prompt)
        self.delegations.append(d)
        self._log(f"DELEGATE {skill}: {task_prompt}")
        if skill not in CAPABILITIES:
            d.summary = f"error: unknown skill '{skill}'"
            return d
        tools = [LOAD_SKILL_TOOL] + SKILL_TOOLS[skill]
        d.system, d.tools = self.subagent_system(skill), tools
        allowed = {t["name"] for t in tools}
        msgs = [{"role": "user", "content": [{"type": "text", "text": task_prompt}]}]
        texts: list[str] = []
        msgs = self._maybe_compact("subagent", d.system, msgs)
        for _ in range(self.max_steps):
            d.steps += 1
            blocks = self._complete("subagent", d.system, msgs, tools)
            if d.steps == 1 and hasattr(self.provider, "last_call"):
                d.first_call = self.provider.last_call()
            uses = [b for b in blocks if b["type"] == "tool_use"]
            texts = [b["text"] for b in blocks if b["type"] == "text"]
            msgs.append({"role": "assistant", "content": [b for b in blocks if b["type"] in ("text", "tool_use")]
                         or [{"type": "text", "text": "(empty)"}]})
            if not uses:
                self._retag("summary")
                break
            results = []
            for u in uses:
                if u["name"] not in allowed:
                    content, err = f"tool '{u['name']}' is not available to you", True
                elif u["name"] == "load_skill":
                    d.loaded.append(str(u["input"].get("name", "")))
                    content, err = load_skill(self.store, {skill}, str(u["input"].get("name", "")))
                else:
                    content, err = self.world.execute(u["name"], u["input"])
                    self._log(f"  [sub {skill}] {u['name']}({json.dumps(u['input'])[:120]}) -> {content[:80]}")
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": content, "is_error": err})
            msgs.append({"role": "user", "content": results})
        d.summary = "\n".join(texts) or "(subagent returned no summary)"
        self._log(f"SUMMARY {skill}: {d.summary}")
        return d
