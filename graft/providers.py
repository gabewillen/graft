"""Model client interface with two adapters (anthropic, openai) plus a scripted fake.

Every provider takes the same neutral input -- system text, neutral messages
(`db.render` output, plus transaction scratch in the same shape) and neutral
tool specs -- and returns neutral content blocks:
  {"type":"text","text"} | {"type":"tool_use","id","name","input"} |
  {"type":"thinking","thinking","signature"} | {"type":"thinking","redacted"}
Each adapter owns the translation to and from its wire format.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import sys
import threading
from typing import Callable, Iterator, Protocol

ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5-5"
# Preference order for OpenAI; the first id that the Models API lists wins.
OPENAI_PREFERENCE = ("gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini")


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict],
                 max_tokens: int = 4096, *, model: str | None = None, effort: str | None = None,
                 site: str | None = None) -> list[dict]: ...


class MissingKey(SystemExit):
    pass


USAGE_KEYS = ("calls", "input", "uncached", "cached", "cache_write", "output")


def empty_usage() -> dict:
    """`input` = uncached + cached + cache_write (every prompt token, however billed)."""
    return {k: 0 for k in USAGE_KEYS}


def add_usage(into: dict, u: dict) -> dict:
    for k in USAGE_KEYS:
        into[k] = into.get(k, 0) + u.get(k, 0)
    return into


def prompt_cache_key(system: str, tools: list[dict]) -> str:
    """Stable routing key: hash of the system prompt (persona + skill listing) and tool set."""
    names = ",".join(sorted(t["name"] for t in tools))
    return "graft-" + hashlib.sha256(f"{system}\x00{names}".encode()).hexdigest()[:24]


def prefix_nonce(messages: list[dict], nonce: str | None) -> list[dict]:
    """Put a per-run nonce at the very start of the first message (after the system prompt).

    Requests in one run share the nonce, so their history prefix still caches; runs differ
    from the first message on, so only tools + system can be shared across runs. Returns a
    copy; the stored rows are untouched."""
    if not nonce or not messages:
        return messages
    first = dict(messages[0], content=list(messages[0]["content"]))
    tag = f"[conversation {nonce}]"
    if first["content"] and first["content"][0]["type"] == "text":
        first["content"][0] = dict(first["content"][0], text=f"{tag}\n{first['content'][0]['text']}")
    else:
        first["content"].insert(0, {"type": "text", "text": tag})
    return [first] + list(messages[1:])


class UsageMixin:
    """Thread-safe totals, per-thread counters (concurrent runs), per-tag buckets and a call log.

    Each call is recorded as disjoint buckets: uncached input (full price), cached input reads,
    cache writes, and output. Tags (see `usage_tag`) attribute calls to roles such as the
    original's main loop or a join transaction."""

    def _init_usage(self) -> None:
        self.usage = empty_usage()
        self.by_tag: dict[str, dict] = {}
        self.by_model: dict[str, dict] = {}
        self.call_log: list[dict] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    @contextlib.contextmanager
    def usage_tag(self, tag: str) -> Iterator[None]:
        prev = getattr(self._local, "tag", None)
        self._local.tag = tag
        try:
            yield
        finally:
            self._local.tag = prev

    def _record(self, uncached: int, output: int, cached: int = 0, cache_write: int = 0,
                **meta) -> dict:
        call = {"calls": 1, "uncached": max(0, uncached), "cached": cached, "cache_write": cache_write,
                "output": output}
        call["input"] = call["uncached"] + cached + cache_write
        tag = getattr(self._local, "tag", None) or "untagged"
        entry = dict(call, tag=tag, **meta)
        with self._lock:
            add_usage(self.usage, call)
            add_usage(self.by_tag.setdefault(tag, empty_usage()), call)
            add_usage(self.by_model.setdefault(str(meta.get("model") or getattr(self, "model", "?")),
                                               empty_usage()), call)
            self.call_log.append(entry)
        self._local.last_call = entry
        t = getattr(self._local, "usage", None)
        if t is not None:
            add_usage(t, call)
        return call

    def last_call(self) -> dict | None:
        """This thread's most recent call-log entry."""
        return getattr(self._local, "last_call", None)

    def retag_last(self, tag: str) -> None:
        """Move this thread's most recent call to another tag (a call's role is sometimes only
        known from its output, e.g. a parent call that turned out to write a handoff)."""
        entry = getattr(self._local, "last_call", None)
        if entry is None or entry["tag"] == tag:
            return
        u = {k: entry[k] for k in USAGE_KEYS}
        with self._lock:
            old = self.by_tag[entry["tag"]]
            for k in USAGE_KEYS:
                old[k] -= u[k]
            add_usage(self.by_tag.setdefault(tag, empty_usage()), u)
            entry["tag"] = tag

    def thread_usage_reset(self) -> dict:
        self._local.usage = empty_usage()
        return self._local.usage

    def count_prefix_tokens(self, system: str, tools: list[dict]) -> dict | None:
        """{'system': n, 'tools_and_system': n, 'exact': bool} or None if unavailable."""
        return None


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------ anthropic
class AnthropicProvider(UsageMixin):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise MissingKey("ANTHROPIC_API_KEY is not set (provider=anthropic).")
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("GRAFT_MODEL") or ANTHROPIC_DEFAULT_MODEL
        self._init_usage()

    @staticmethod
    def wire_messages(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            blocks = []
            for b in m["content"]:
                t = b["type"]
                if t == "text":
                    blocks.append({"type": "text", "text": b["text"]})
                elif t == "tool_use":
                    blocks.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]})
                elif t == "tool_result":
                    blocks.append({"type": "tool_result", "tool_use_id": b["tool_use_id"],
                                   "content": str(b["content"]), "is_error": bool(b.get("is_error"))})
                elif t == "thinking":
                    if "redacted" in b:
                        blocks.append({"type": "redacted_thinking", "data": b["redacted"]})
                    else:
                        blocks.append({"type": "thinking", "thinking": b["thinking"],
                                       "signature": b["signature"]})
            out.append({"role": m["role"], "content": blocks})
        return out

    @staticmethod
    def wire_tools(tools: list[dict]) -> list[dict]:
        return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                for t in tools]

    @classmethod
    def request_cache_kwargs(cls, system: str, tools: list[dict]) -> dict:
        """Prompt caching, the documented agent-loop combination: an explicit breakpoint on the
        (only) system block caches tools + system, the prefix shared by every run with the same
        persona and skills; top-level automatic caching moves a second breakpoint along the
        growing conversation tail. Both use the default 5-minute TTL."""
        kwargs: dict = {"system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                        "cache_control": {"type": "ephemeral"}}
        if tools:
            kwargs["tools"] = cls.wire_tools(tools)
        return kwargs

    def count_prefix_tokens(self, system, tools):
        probe = [{"role": "user", "content": "."}]
        base = self.client.messages.count_tokens(model=self.model, messages=probe).input_tokens
        sys_n = self.client.messages.count_tokens(model=self.model, system=system, messages=probe).input_tokens
        kw = {"tools": self.wire_tools(tools)} if tools else {}
        both = self.client.messages.count_tokens(model=self.model, system=system, messages=probe,
                                                 **kw).input_tokens
        return {"system": sys_n - base, "tools_and_system": both - base, "exact": True}

    def complete(self, system, messages, tools, max_tokens=4096, *, model=None, effort=None, site=None):
        model = model or self.model
        kwargs: dict = {}
        if model == "claude-sonnet-5-5":
            # Lowest thinking setting on Sonnet 5.5 (no extended thinking); effort <= high.
            kwargs["thinking"] = {"type": "between_tools"}
            effort = effort if effort in ("low", "medium", "high") else "medium"
        else:
            effort = effort or "low"
        kwargs["output_config"] = {"effort": effort}
        kwargs.update(self.request_cache_kwargs(system, tools))
        resp = self.client.messages.create(
            model=model, max_tokens=max_tokens, messages=self.wire_messages(messages), **kwargs)
        u = resp.usage
        self._record(u.input_tokens, u.output_tokens, cached=u.cache_read_input_tokens or 0,
                     cache_write=u.cache_creation_input_tokens or 0, cache_key=None,
                     model=model, effort=effort, site=site)
        if resp.stop_reason == "refusal":
            return [{"type": "text", "text": "(refused)"}]
        blocks = []
        for b in resp.content:
            if b.type == "text" and b.text:
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
            elif b.type == "thinking":
                blocks.append({"type": "thinking", "thinking": b.thinking, "signature": b.signature})
            elif b.type == "redacted_thinking":
                blocks.append({"type": "thinking", "redacted": b.data})
        return blocks


# ------------------------------------------------------------------ openai
def pick_openai_model(client) -> str:
    override = os.environ.get("GRAFT_MODEL")
    if override:
        return override
    available = {m.id for m in client.models.list()}
    for mid in OPENAI_PREFERENCE:
        if mid in available:
            return mid
    raise SystemExit(f"none of {OPENAI_PREFERENCE} listed by the OpenAI Models API; set GRAFT_MODEL")


class OpenAIProvider(UsageMixin):
    name = "openai"

    def __init__(self, model: str | None = None):
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingKey("OPENAI_API_KEY is not set (provider=openai).")
        import openai

        self.client = openai.OpenAI()
        self.model = model or pick_openai_model(self.client)
        # Responses API by default: it allows function tools together with reasoning.
        self.api = os.environ.get("GRAFT_OPENAI_API", "responses")
        if self.api not in ("responses", "chat"):
            raise SystemExit(f"GRAFT_OPENAI_API must be 'responses' or 'chat', got {self.api!r}")
        self.effort = os.environ.get("GRAFT_REASONING_EFFORT", "low")
        # prompt_cache_key: only if this installed SDK accepts it (checked, not assumed).
        create = self.client.responses.create if self.api == "responses" else self.client.chat.completions.create
        self.use_cache_key = (os.environ.get("GRAFT_PROMPT_CACHE_KEY", "1") != "0"
                              and _accepts(create, "prompt_cache_key"))
        self._init_usage()

    def _cache_kwargs(self, system: str, tools: list[dict]) -> dict:
        return {"prompt_cache_key": prompt_cache_key(system, tools)} if self.use_cache_key else {}

    @staticmethod
    def wire_tools_responses(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "name": t["name"], "description": t["description"],
                 "parameters": t["input_schema"]} for t in tools]

    def count_prefix_tokens(self, system, tools):
        """Exact counts via the Responses input-token counting endpoint (differences against a
        one-token probe message)."""
        probe = [{"role": "user", "content": "."}]
        count = self.client.responses.input_tokens.count
        base = count(model=self.model, input=probe).input_tokens
        sys_n = count(model=self.model, instructions=system, input=probe).input_tokens
        kw = {"tools": self.wire_tools_responses(tools)} if tools else {}
        both = count(model=self.model, instructions=system, input=probe, **kw).input_tokens
        return {"system": sys_n - base, "tools_and_system": both - base, "exact": True}

    @staticmethod
    def wire_input(messages: list[dict]) -> list[dict]:
        """Neutral messages -> Responses API input items (stateless, store=False)."""
        out: list[dict] = []
        for m in messages:
            for b in m["content"]:
                if b["type"] == "text":
                    out.append({"role": m["role"], "content": b["text"]})
                elif b["type"] == "tool_use":
                    out.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                                "arguments": json.dumps(b["input"])})
                elif b["type"] == "tool_result":
                    content = str(b["content"])
                    if b.get("is_error"):
                        content = "ERROR: " + content
                    out.append({"type": "function_call_output", "call_id": b["tool_use_id"],
                                "output": content})
                # thinking blocks are Anthropic-only and are not sent
        return out

    def complete_responses(self, system, messages, tools, max_tokens, model=None, effort=None, site=None):
        model, effort = model or self.model, effort or self.effort
        kwargs: dict = self._cache_kwargs(system, tools)
        if tools:
            kwargs["tools"] = self.wire_tools_responses(tools)
        if model.startswith(("gpt-5", "gpt-6", "o")):
            kwargs["reasoning"] = {"effort": effort}
        else:
            effort = None
        resp = self.client.responses.create(
            model=model, instructions=system, input=self.wire_input(messages),
            max_output_tokens=max_tokens, store=False, **kwargs)
        self._record_openai(resp.usage, "input_tokens", "output_tokens", "input_tokens_details",
                            kwargs.get("prompt_cache_key"), model=model, effort=effort, site=site)
        blocks: list[dict] = []
        for item in resp.output:
            if item.type == "message":
                text = "".join(c.text for c in item.content if c.type == "output_text")
                if text:
                    blocks.append({"type": "text", "text": text})
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_unparseable_arguments": item.arguments}
                blocks.append({"type": "tool_use", "id": item.call_id, "name": item.name, "input": args})
        return blocks

    def _record_openai(self, usage, in_f: str, out_f: str, details_f: str, key: str | None, **meta) -> None:
        """OpenAI counts cached reads (and cache writes) inside the input total; split them out."""
        if usage is None:
            self._record(0, 0, cache_key=key, **meta)
            return
        total = getattr(usage, in_f) or 0
        det = getattr(usage, details_f, None)
        cached = (getattr(det, "cached_tokens", 0) or 0) if det is not None else 0
        write = (getattr(det, "cache_write_tokens", 0) or 0) if det is not None else 0
        self._record(total - cached - write, getattr(usage, out_f) or 0, cached=cached, cache_write=write,
                     cache_key=key, raw_input=total, **meta)

    @staticmethod
    def wire_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                for b in m["content"]:
                    if b["type"] == "tool_result":
                        content = str(b["content"])
                        if b.get("is_error"):
                            content = "ERROR: " + content
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": content})
                texts = [b["text"] for b in m["content"] if b["type"] == "text"]
                if texts:
                    out.append({"role": "user", "content": "\n\n".join(texts)})
            else:
                texts = [b["text"] for b in m["content"] if b["type"] == "text"]
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                         for b in m["content"] if b["type"] == "tool_use"]
                msg: dict = {"role": "assistant", "content": "\n\n".join(texts) if texts else None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)  # thinking blocks have no OpenAI equivalent and are not sent
        return out

    def complete(self, system, messages, tools, max_tokens=4096, *, model=None, effort=None, site=None):
        if self.api == "responses":
            return self.complete_responses(system, messages, tools, max_tokens, model, effort, site)
        model = model or self.model
        kwargs: dict = self._cache_kwargs(system, tools)
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools]
        if model.startswith(("gpt-5.", "gpt-6")):
            kwargs["reasoning_effort"] = "none"  # no extended thinking; required for tools here
        resp = self.client.chat.completions.create(
            model=model, messages=self.wire_messages(system, messages),
            max_completion_tokens=max_tokens, **kwargs)
        self._record_openai(resp.usage, "prompt_tokens", "completion_tokens", "prompt_tokens_details",
                            kwargs.get("prompt_cache_key"), model=model,
                            effort="none" if model.startswith(("gpt-5.", "gpt-6")) else None, site=site)
        msg = resp.choices[0].message
        blocks: list[dict] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": tc.function.arguments}
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
        return blocks


# ------------------------------------------------------------------ fake
class FakeProvider(UsageMixin):
    """Deterministic provider driven by a policy function (offline tests / dry runs)."""

    name = "fake"
    model = "fake"

    def __init__(self, policy: Callable[[str, list[dict], list[dict]], list[dict]]):
        self.policy = policy
        self.calls: list[dict] = []
        self._init_usage()

    def complete(self, system, messages, tools, max_tokens=4096, *, model=None, effort=None, site=None):
        self.calls.append({"system": system, "messages": messages, "tools": tools, "model": model or self.model,
                           "effort": effort, "site": site})
        self._record(0, 0, model=model or self.model, effort=effort, site=site)
        return self.policy(system, messages, tools)

    def count_prefix_tokens(self, system, tools):
        """Estimate (chars / 4): the fake provider has no tokenizer."""
        tool_chars = len(json.dumps(tools)) if tools else 0
        return {"system": len(system) // 4, "tools_and_system": (len(system) + tool_chars) // 4,
                "exact": False}


def make_provider(name: str | None = None) -> Provider:
    name = name or os.environ.get("GRAFT_PROVIDER") or (
        "openai" if os.environ.get("OPENAI_API_KEY") else "anthropic")
    try:
        if name == "openai":
            return OpenAIProvider()
        if name == "anthropic":
            return AnthropicProvider()
    except MissingKey as e:
        print(f"error: {e} Export the key (never commit it) and retry.", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(f"unknown GRAFT_PROVIDER {name!r} (use openai|anthropic)")
