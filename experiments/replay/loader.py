"""Load a recorded Claude Code delegation event (parent transcript + subagent transcript).

Claude Code writes one JSONL per session (`<session>.jsonl`) and one per subagent
(`<session>/subagents/agent-<id>.jsonl`, optional `.meta.json` with the spawning `toolUseId`).
Lines carry `type` (user/assistant/attachment/system/...), `timestamp`, `cwd`, `gitBranch` and a
`message` with `role` and `content` (a string, or blocks: text / thinking / tool_use /
tool_result).

An Event holds:
  - parent_prefix: the parent conversation before the spawning Agent/Task tool_use, as text
    turns (user text and assistant text only; tool calls, tool results and thinking are dropped,
    and the prefix is capped to the last `max_prefix_chars` characters, cut at a turn boundary)
  - handoff: the recorded subagent prompt; description; subagent type
  - recorded: the subagent's own turns (for probe generation) and its final report text
  - spawn time, branch and cwd (to check out the matching commit)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # user | assistant
    text: str


@dataclass
class Event:
    agent_id: str
    subagent_path: str
    parent_path: str
    spawn_index: int
    spawn_tool_use_id: str
    timestamp: str
    branch: str | None
    cwd: str | None
    description: str
    subagent_type: str
    handoff: str
    parent_prefix: list[Turn]
    recorded_turns: list[dict] = field(default_factory=list)  # {role, kind, name?, text}
    final_report: str = ""
    recorded_tools: dict = field(default_factory=dict)


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _blocks(entry: dict) -> list[dict]:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def _result_text(b: dict) -> str:
    c = b.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return ""


def find_spawn(parent: list[dict], handoff_first: str, tool_use_id: str | None) -> tuple[int, dict, dict]:
    """(line index, entry, tool_use block) of the Agent/Task call that spawned the subagent: by the
    recorded toolUseId when known, else by the prompt text."""
    for i, d in enumerate(parent):
        if d.get("type") != "assistant":
            continue
        for b in _blocks(d):
            if b.get("type") != "tool_use" or b.get("name") not in ("Agent", "Task"):
                continue
            if tool_use_id and b.get("id") == tool_use_id:
                return i, d, b
            if not tool_use_id and (b.get("input") or {}).get("prompt", "").strip() == handoff_first.strip():
                return i, d, b
    raise LookupError("spawning Agent/Task tool_use not found in the parent transcript")


def text_turns(entries: list[dict]) -> list[Turn]:
    turns: list[Turn] = []
    for d in entries:
        if d.get("type") not in ("user", "assistant") or d.get("isSidechain"):
            continue
        texts = [b.get("text", "") for b in _blocks(d) if b.get("type") == "text" and b.get("text", "").strip()]
        if not texts:
            continue
        role = (d.get("message") or {}).get("role") or d["type"]
        text = "\n".join(texts)
        if turns and turns[-1].role == role:
            turns[-1].text += "\n\n" + text
        else:
            turns.append(Turn(role, text))
    return turns


def cap_prefix(turns: list[Turn], max_chars: int) -> list[Turn]:
    """Keep the most recent turns within max_chars, starting at a user turn."""
    out, used = [], 0
    for t in reversed(turns):
        if used + len(t.text) > max_chars and out:
            break
        out.append(t)
        used += len(t.text)
    out.reverse()
    while out and out[0].role != "user":
        out.pop(0)
    return out


def load_event(subagent_path: str, max_prefix_chars: int = 40000) -> Event:
    meta_path = subagent_path[:-len(".jsonl")] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    sub = read_jsonl(subagent_path)
    first_user = next(d for d in sub if d.get("type") == "user")
    handoff_first = "\n".join(b.get("text", "") for b in _blocks(first_user) if b.get("type") == "text")
    parent_path = os.path.dirname(os.path.dirname(subagent_path)) + ".jsonl"
    parent = read_jsonl(parent_path)
    idx, entry, use = find_spawn(parent, handoff_first, meta.get("toolUseId"))
    prefix = cap_prefix(text_turns(parent[:idx]), max_prefix_chars)
    recorded, tools, final = [], {}, ""
    for d in sub:
        if d.get("type") not in ("user", "assistant"):
            continue
        for b in _blocks(d):
            t = b.get("type")
            if t == "text" and b.get("text", "").strip():
                recorded.append({"role": d["type"], "kind": "text", "text": b["text"]})
                if d["type"] == "assistant":
                    final = b["text"]
            elif t == "tool_use":
                tools[b.get("name")] = tools.get(b.get("name"), 0) + 1
                recorded.append({"role": "assistant", "kind": "tool_use", "name": b.get("name"),
                                 "text": json.dumps(b.get("input"))})
            elif t == "tool_result":
                recorded.append({"role": "user", "kind": "tool_result", "text": _result_text(b)})
    inp = use.get("input") or {}
    aid = os.path.basename(subagent_path).removeprefix("agent-").removesuffix(".jsonl")
    return Event(aid, subagent_path, parent_path, idx, use.get("id", ""), entry.get("timestamp", ""),
                 entry.get("gitBranch"), entry.get("cwd") or first_user.get("cwd"), inp.get("description", ""),
                 inp.get("subagent_type", ""), inp.get("prompt", handoff_first), prefix, recorded, final, tools)


def event_texts(e: Event) -> list[str]:
    """Every string of the event that would be sent to a provider (for the secret filter)."""
    return [t.text for t in e.parent_prefix] + [e.handoff, e.description] + [r["text"] for r in e.recorded_turns]
