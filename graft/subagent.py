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
import inspect
import json
from dataclasses import dataclass, field
from typing import Sequence

from . import callsites, compaction
from .db import Store
from .providers import Provider, prefix_nonce
from . import a2a
from .skills import BASE_SYSTEM, CATALOG, LOAD_SKILL_TOOL, BUILTIN, load_skill, skill_listing
from .tools import CallToolResult, Tool, by_name
from .skills import obj

CAPABILITIES = [s.name for s in CATALOG if s.name not in BUILTIN]
SKILL_TOOLS = {s.name: list(s.tools) for s in CATALOG}  # each capability skill's tool names

DELEGATE_TOOL = {
    "name": "delegate",
    "description": ("Hand a task to a subagent that holds one skill. The subagent sees only "
                    "task_prompt, not this conversation, so include every fact it needs. It does the "
                    "work with that skill's tools and returns a summary of what it did."),
    "inputSchema": obj({"skill": {"type": "string", "enum": CAPABILITIES},
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
    """The baseline arm, behind the same interface as `graft.agent.GraftAgent` (`send`, `get_task`,
    `start`, `close`), so an experiment drives both arms the same way. Each message is one task,
    answered within the call (no work outlives it), so `send` returns it completed."""
    store: Store  # skill catalog only (descriptions/instructions); no timelines are used
    provider: Provider
    tools: Sequence[Tool] = ()
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
    tasks: dict[str, a2a.Task] = field(default_factory=dict)
    context_id: str = "subagent"

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
        cs = callsites.settings(tag, msgs)  # per-call-site effort and model (graft/callsites.py)
        tagger = getattr(self.provider, "usage_tag", None)
        with tagger(tag) if tagger else contextlib.nullcontext():
            return self.provider.complete(system, prefix_nonce(msgs, self.nonce), tools, **cs)

    def _retag(self, tag: str) -> None:
        retag = getattr(self.provider, "retag_last", None)
        if retag:
            retag(tag)

    def _log(self, msg: str) -> None:
        if callable(self.trace):
            self.trace(msg)

    # ------------------------------------------------------------ the host interface
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> "SubagentRunner":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def send(self, request: a2a.SendMessageRequest | dict) -> a2a.SendMessageResponse:
        """SendMessage: the parent's turn on the message; its replies are the task's history."""
        req = request if isinstance(request, a2a.SendMessageRequest) else a2a.SendMessageRequest.from_dict(request)
        task_id = req.message.task_id or f"task-{len(self.tasks) + 1}"
        prior = self.tasks.get(task_id)
        if req.message.task_id is not None and prior is None:
            raise a2a.TaskNotFound(f"task {task_id[:64]!r} not found")
        if prior is not None:
            raise a2a.UnsupportedOperation(f"task {task_id[:64]!r} is in a terminal state")
        user = a2a.Message(message_id=req.message.message_id, role=a2a.ROLE_USER, parts=req.message.parts,
                           context_id=self.context_id, task_id=task_id)
        replies = await self._turn(req.message.text)
        history = [user] + [a2a.Message(message_id=f"{task_id}-r{i}", role=a2a.ROLE_AGENT, parts=[a2a.Part(text=t)],
                                        context_id=self.context_id, task_id=task_id) for i, t in enumerate(replies)]
        task = a2a.Task(id=task_id, context_id=self.context_id,
                        status=a2a.TaskStatus(state=a2a.COMPLETED, message=history[-1] if replies else None),
                        history=history)
        self.tasks[task_id] = task
        return a2a.SendMessageResponse(task=task)

    def get_task(self, request: a2a.GetTaskRequest | dict) -> a2a.Task:
        req = request if isinstance(request, a2a.GetTaskRequest) else a2a.GetTaskRequest.from_dict(request)
        if req.id not in self.tasks:
            raise a2a.TaskNotFound(f"task {req.id[:64]!r} not found")
        return self.tasks[req.id]

    # ------------------------------------------------------------ parent
    async def _turn(self, text: str) -> list[str]:
        said: list[str] = []
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
            said += [b["text"] for b in blocks if b["type"] == "text"]
            if not uses:
                return said
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
        return said

    def _call(self, name: str, args: dict) -> tuple[str, bool]:
        """A host tool, called synchronously (this baseline runs no tool concurrently); an async
        tool's awaitable is closed and refused."""
        try:
            outcome = by_name(list(self.tools))[name].fn(args)
        except Exception as e:  # noqa: BLE001 - it ran, with an error
            return f"raised {type(e).__name__}: {str(e)[:200]}; it may have acted", True
        if inspect.isawaitable(outcome):
            getattr(outcome, "close", lambda: None)()
            return "async tools are not supported by the subagent baseline", True
        result = CallToolResult.from_value(outcome)
        if result is None:
            return f"returned a {type(outcome).__name__}, not a tool result", True
        return result.text(), result.is_error

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
        tools = [LOAD_SKILL_TOOL] + [by_name(list(self.tools))[n].definition() for n in SKILL_TOOLS[skill]]
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
                    content, err = self._call(u["name"], u["input"])
                    self._log(f"  [sub {skill}] {u['name']}({json.dumps(u['input'])[:120]}) -> {content[:80]}")
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": content, "is_error": err})
            msgs.append({"role": "user", "content": results})
        d.summary = "\n".join(texts) or "(subagent returned no summary)"
        self._log(f"SUMMARY {skill}: {d.summary}")
        return d
