"""A deterministic scripted policy for FakeProvider (offline tests and dry runs).

It plays every role through the same neutral interface a real model sees, so the
whole branch -> converse -> merge -> re-branch path (graft's branch and join) runs without an API key.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import json
import re
from dataclasses import dataclass, field

from .machine import described
from .skills import STRS, S, obj
from .tools import CallToolResult, Tool

_ids = itertools.count(1)


def _tid() -> str:
    return f"toolu_fake_{next(_ids)}"


def _use(tool: str, **inp) -> list[dict]:
    return [{"type": "tool_use", "id": _tid(), "name": tool, "input": inp}]


def _text(t: str) -> list[dict]:
    return [{"type": "text", "text": t}]


# Main's result of a branch's merge call starts with the branch's id ("site merged. ...", "deck was
# stopped before it finished. ..."); a wake run's view ends on one.
_MERGED = re.compile(r"[a-z0-9][a-z0-9-]* (merged\b|was stopped before it finished)")


def woken(messages: list[dict]) -> str | None:
    """The merge result a wake run's view ends on, if it does (a branch merged)."""
    last = _last_block(messages)
    if last["type"] == "tool_result" and isinstance(last.get("content"), str) and _MERGED.match(last["content"]):
        return last["content"]
    return None


def _wake_reply(messages: list[dict]) -> list[dict]:
    """main after a merge: a current branch's replies already reached the user, so it ends its turn
    without saying anything; otherwise it reports what the branches wrote (the text of the merged
    assistant messages since its last user turn)."""
    results = [b for b in messages[-1]["content"] if b["type"] == "tool_result"]
    if all("already reached the user" in str(b["content"]) for b in results):
        return []
    replies: list[str] = []
    for m in reversed(messages[:-1]):
        if m["role"] == "user" and not all(b["type"] == "tool_result" for b in m["content"]):
            break
        if m["role"] == "assistant":
            replies = [b["text"] for b in m["content"] if b["type"] == "text"] + replies
    return _text("Background work finished. " + " ".join(replies)[:300])


def _last_block(messages: list[dict]) -> dict:
    return messages[-1]["content"][-1]


def _user_texts(messages: list[dict]) -> list[str]:
    """What the user said. Harness notes are user-role rows too; they carry a mark and are left out."""
    return [b["text"] for m in messages if m["role"] == "user" for b in m["content"]
            if b["type"] == "text" and not b.get("harness")]


def texts_of(messages: list[dict]) -> list[str]:
    """User text blocks (the fake original quotes them as its seed)."""
    return _user_texts(messages)


_ROW_TAG = re.compile(r"^(?:\[conversation [^\]]*\]\s*)?\[#(\d+)\]\s*")


def _row_ids(messages: list[dict]) -> list[int]:
    """The user's messages as integer row-id pointers (from main's `[#<id>]` prefixes)."""
    out: list[int] = []
    for t in _user_texts(messages):
        m = _ROW_TAG.match(t)
        if m:
            rid = int(m.group(1))
            if rid not in out:
                out.append(rid)
    return out


def _entry(messages: list[dict], skills: list[str]) -> dict:
    """A fork entry that points at the user's messages by integer row id."""
    return {"skills": skills, "messages": _row_ids(messages)}


def _ids_from_sql(out: str) -> list[int]:
    return [int(line.split(" | ")[0]) for line in out.splitlines()[1:] if re.match(r"^\d+", line)]


def pivot_policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    names = {t["name"] for t in tools}
    last = _last_block(messages)
    if "fork" in names:  # the original: route with one fork call
        if woken(messages):
            return _wake_reply(messages)
        if last["type"] == "tool_result":
            return _text("(waiting)")
        user = [re.sub(r"^\[conversation [^\]]*\]\s*", "", t) for t in texts_of(messages)]
        wants_slides = any("slide" in t.lower() for t in user)
        return _use(tool="fork", current=_entry(messages=messages, skills=["slides" if wants_slides else "website"]))
    # a branch: its linked skills are already auto-loaded in its system prompt
    prev_tool = next((b["name"] for m in reversed(messages) if m["role"] == "assistant"
                      for b in m["content"] if b["type"] == "tool_use"), None)
    texts = _user_texts(messages)
    if last["type"] == "tool_result" and prev_tool not in ("load_skill", "fork"):
        return _text("Done. Anything else?")
    msg = texts[-1] if texts else ""
    if "site_write" in names:
        # it merges with a reason (handing the request back) only for a slide request after it acted;
        # a seed that also mentions slides is another instance's part
        acted = any(b["type"] == "tool_use" and b["name"] not in ("load_skill", "fork")
                    for m in messages if m["role"] == "assistant" for b in m["content"])
        answered = any(m["role"] == "assistant" and any(b["type"] == "text" for b in m["content"])
                       for m in messages)
        if "slide" in msg.lower() and (acted or answered):
            return _use(tool="merge", reason="This needs a skill I do not hold here.")
        if "services" in msg.lower():
            return _use("site_write", page="home", content="\n".join(_user_texts(messages)))
        return _text("Noted. What else should the site include?")
    if "slides_create" in names:
        facts = " ".join(t for t in _user_texts(messages) if "Ohio" in t or "formal" in t)
        return [*_use("slides_create", title="Our Practice", audience=facts, tone="formal"),
                *_use("slides_add_slide", title="Services", bullets=["Cleanings", "Implants", "Whitening"])]
    return _text("ok")


def subagent_policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    """Scripted parent + subagent for the subagent baseline (offline)."""
    names = {t["name"] for t in tools}
    last = _last_block(messages)
    texts = _user_texts(messages)
    if "delegate" in names:  # the parent
        if last["type"] == "tool_result":
            return _text("Done: " + str(last["content"])[:200])
        msg = texts[-1].lower() if texts else ""
        if "slide" in msg:
            facts = " ".join(t for t in texts if "slide" not in t.lower())
            return _use("delegate", skill="slides", task_prompt=f"Build a slide show. Facts: {facts}")
        return _text("Noted. Tell me more about the website.")
    loaded = any(b["type"] == "tool_use" and b["name"] == "load_skill"
                 for m in messages if m["role"] == "assistant" for b in m["content"])
    if not loaded:
        return _use("load_skill", name="slides" if "slides_create" in names else "website")
    if last["type"] == "tool_result" and "created deck" in str(last["content"]) or \
            any(b["type"] == "tool_use" and b["name"] == "slides_add_slide"
                for m in messages if m["role"] == "assistant" for b in m["content"]):
        return _text("Built a formal deck for the dental practice.")
    task = texts[0] if texts else ""
    return [*_use("slides_create", title="Our Practice", audience=task, tone="formal"),
            *_use("slides_add_slide", title="Services", bullets=["Cleanings", "Implants", "Whitening"])]


# ------------------------------------------------------------------ bench (generic) policies
_SKILL_WORDS = (("slides", r"slide|deck|presentation"), ("calendar", r"\bbook\b|appointment|calendar"),
                ("website", r"website|\bsite\b"))
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})\D+(\d{1,2}:\d{2})")


def _clean(t: str) -> str:
    return re.sub(r"^(?:\[conversation [^\]]*\]\s*)?(?:\[#\d+\]\s*)?", "", t)


def _wants(text: str) -> list[str]:
    return [s for s, pat in _SKILL_WORDS if re.search(pat, text, re.I)]


def _summary(messages: list[dict]) -> list[dict]:
    return _text("summary: " + " | ".join(_clean(t) for t in _user_texts(messages))[-1500:])


def _act(names: set[str], messages: list[dict]) -> list[dict]:
    """A capability holder's action on the latest user request (shared by branches and subagents)."""
    texts = [_clean(t) for t in _user_texts(messages)]
    joined = " ".join(texts)
    out: list[dict] = []
    if "calendar_book" in names and (m := _DATE.search(joined)):
        out += _use("calendar_book", date=m.group(1), time=m.group(2).zfill(5), what="dental cleaning")
    if "slides_create" in names:
        out += _use("slides_create", title="Open House", audience=joined[:300], tone="formal")
        out += _use("slides_add_slide", title="Overview", bullets=[joined[:60], "Two.", "Three."])
    if "site_write" in names and "services" in joined.lower():
        out += _use("site_write", page="home", content=joined[:400])
    return out or _text("Noted. What else should I include?")


def _holder_policy(names: set[str], messages: list[dict]) -> list[dict]:
    last = _last_block(messages)
    used = [b for m in messages if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    caps = [s for s, prefix in (("slides", "slides_"), ("calendar", "calendar_"), ("website", "site_"))
            if any(n.startswith(prefix) for n in names)]
    if "load_skill" in names and not any(u["name"] == "load_skill" for u in used) and caps:
        return _use("load_skill", name=caps[0])
    if last["type"] == "tool_result":
        if used and used[-1]["name"] in ("load_skill", "fork"):
            return _act(names, messages)
        errs = [b for b in messages[-1]["content"] if b["type"] == "tool_result" and b.get("is_error")]
        return _text("Sorry, I could not complete that: " + str(errs[0]["content"])[:120]) if errs \
            else _text("Done: " + str(last["content"])[:120])
    msg = _clean(_user_texts(messages)[-1]) if _user_texts(messages) else ""
    wants = _wants(msg)
    if "merge" in names and wants and not set(wants) & set(caps):
        return _use(tool="merge", reason="This needs a skill I do not hold here.")
    return _act(names, messages)


def _fake_compact(messages: list[dict]) -> list[dict]:
    """Keep the user's text rows; compress each run of other rows into one assistant message
    (a lone row is kept)."""
    index = _user_texts(messages)[-1]
    rows = re.findall(r"row_id=(\d+) (user|assistant) (\w+)[^:\n]*?()(?=:)", index)
    items: list[dict] = []
    run: list[tuple[int, bool]] = []

    def flush():
        if len(run) == 1:
            items.append({"op": "keep", "row_id": run[0][0]})
        elif run:
            items.append({"op": "compress", "role": "assistant", "text": "I have been noting the details so far.",
                          "begin_row_id": run[0][0], "end_row_id": run[-1][0]})
        run.clear()

    for rid, role, kind, short in rows:
        if role == "user" and kind == "text":
            flush()
            items.append({"op": "keep", "row_id": int(rid)})
        else:
            run.append((int(rid), bool(short)))
    flush()
    return _use("compact", items=items)


def bench_graft_policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    names = {t["name"] for t in tools}
    if not tools:
        return _summary(messages)
    if names == {"compact"}:
        return _fake_compact(messages)
    if "fork" in names:  # the original
        if woken(messages):
            return _wake_reply(messages)
        if _last_block(messages)["type"] == "tool_result":
            return _text("(waiting)")
        texts = [_clean(t) for t in _user_texts(messages)]
        texts = [t for t in texts if not t.startswith("(compaction")]
        wants = _wants(texts[-1] if texts else "") or _wants(" ".join(texts)) or ["website"]
        return _use(tool="fork", current=_entry(messages=messages, skills=wants))
    return _holder_policy(names, messages)


def parallel_graft_policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    """Like bench_graft_policy, but a request that names several skills becomes one branch call
    with one background entry per skill; the original keeps the conversation and reports when
    the harness wakes it (experiments/parallel.py)."""
    names = {t["name"] for t in tools}
    if "fork" not in names:
        return _holder_policy(names, messages)
    last = _last_block(messages)
    if woken(messages):
        return _wake_reply(messages)
    if last["type"] == "tool_result":
        return _text("I have started on that and will report back when it is done.")
    texts = [_clean(t) for t in _user_texts(messages)]
    wants = _wants(texts[-1] if texts else "") or ["website"]
    if len(wants) == 1:
        return _use(tool="fork", current=_entry(messages=messages, skills=wants))
    return _use(tool="fork", detached=[_entry(messages=messages, skills=[w]) for w in wants])


def bench_subagent_policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    names = {t["name"] for t in tools}
    if not tools:
        return _summary(messages)
    if "delegate" in names:  # the parent
        last = _last_block(messages)
        if last["type"] == "tool_result":
            return _text("Done: " + str(last["content"])[:200])
        texts = [_clean(t) for t in _user_texts(messages)]
        wants = _wants(texts[-1]) if texts else []
        if not wants:
            return _text("Noted. Tell me more.")
        prompt = "Task and facts from the user: " + " ".join(texts)
        out: list[dict] = []
        for w in wants:
            out += _use("delegate", skill=w, task_prompt=prompt)
        return out
    if _last_block(messages)["type"] == "tool_result":
        used = [b for m in messages if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
        if used and used[-1]["name"] != "load_skill":
            return _text("Summary: did the task with " + used[-1]["name"])
    return _holder_policy(names, messages)


# ------------------------------------------------------------------ the simulated tools

FAKE_TOOLS: list[dict] = [
    {"name": "site_write", "description": "Create or replace a website page.",
     "inputSchema": obj({"page": S, "content": S}, ["page", "content"])},
    {"name": "site_read", "description": "Read a website page.",
     "inputSchema": obj({"page": S}, ["page"])},
    {"name": "slides_create", "description": "Create a new slide deck.",
     "inputSchema": obj({"title": S, "audience": S, "tone": S}, ["title", "audience", "tone"])},
    {"name": "slides_add_slide", "description": "Append a slide to the current deck.",
     "inputSchema": obj({"title": S, "bullets": STRS}, ["title", "bullets"])},
    {"name": "slides_read", "description": "Read the current deck (title, audience, tone, slides).",
     "inputSchema": obj({}, [])},
    {"name": "calendar_book", "description": "Book an appointment.",
     "inputSchema": obj({"date": S, "time": S, "what": S}, ["date", "time", "what"])},
    {"name": "calendar_list", "description": "List appointments.",
     "inputSchema": obj({}, [])},
    {"name": "newsletter_draft", "description": "Draft an email newsletter.",
     "inputSchema": obj({"subject": S, "body": S}, ["subject", "body"])},
    {"name": "social_post", "description": "Publish a social media post.",
     "inputSchema": obj({"platform": S, "text": S}, ["platform", "text"])},
    {"name": "recipe_save", "description": "Save a recipe.",
     "inputSchema": obj({"title": S, "steps": STRS}, ["title", "steps"])}
]


def _result(outcome):
    """A simulated tool's (content, is_error) as a CallToolResult; anything else as it is."""
    if type(outcome) is tuple and len(outcome) == 2 and isinstance(outcome[0], str) and isinstance(outcome[1], bool):
        return CallToolResult(content=[{"type": "text", "text": str.__str__(outcome[0])}], is_error=outcome[1])
    return outcome


async def _finish(awaitable):
    return _result(await awaitable)


@dataclass
class FakeTools:
    """The simulated world behind the simulated skills' tools (a fake website, slide decks, a
    calendar): `tools()` is the list of Tools a test, the simulator or an experiment gives graft.
    Each tool's function calls `execute(name, args)`, which returns (content, is_error) or an
    awaitable of it (a test subclass may define `async def execute`); (content, is_error) becomes a
    CallToolResult, and anything else is passed on as the tool's return value."""
    site: dict[str, str] = field(default_factory=dict)
    decks: list[dict] = field(default_factory=list)
    calendar: list[dict] = field(default_factory=list)
    other: list[tuple[str, dict]] = field(default_factory=list)
    log: list[tuple[str, dict]] = field(default_factory=list)
    # tool name -> error message returned on every call (deterministic failure injection)
    failures: dict[str, str] = field(default_factory=dict)

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a simulated tool. Missing or malformed arguments are a tool error, not a crash."""
        try:
            return self._execute(name, args)
        except (KeyError, TypeError, AttributeError) as e:
            return f"invalid arguments for {name}: {described(e)}", True

    def _execute(self, name: str, args: dict) -> tuple[str, bool]:
        self.log.append((name, args))
        if name in self.failures:
            return self.failures[name], True
        if name == "site_write":
            self.site[args["page"]] = args["content"]
            return f"wrote page {args['page']} ({len(args['content'])} chars)", False
        if name == "site_read":
            page = self.site.get(args["page"])
            return (page, False) if page is not None else (f"no page {args['page']}", True)
        if name == "slides_create":
            self.decks.append({**args, "slides": []})
            return f"created deck '{args['title']}' (deck #{len(self.decks)})", False
        if name == "slides_read":
            return (json.dumps(self.decks[-1]), False) if self.decks else ("no deck yet", False)
        if name == "slides_add_slide":
            if not self.decks:
                return "no deck; call slides_create first", True
            self.decks[-1]["slides"].append({"title": args["title"], "bullets": list(args["bullets"])})
            return f"added slide {len(self.decks[-1]['slides'])}: {args['title']}", False
        if name == "calendar_book":
            self.calendar.append(args)
            return f"booked {args['what']} on {args['date']} at {args['time']}", False
        if name == "calendar_list":
            return json.dumps(self.calendar), False
        if name in ("newsletter_draft", "social_post", "recipe_save"):
            self.other.append((name, args))
            return f"{name.split('_')[0]} saved", False
        return f"unknown tool {name}", True

    def deck_text(self) -> str:
        return json.dumps(self.decks)

    def call(self, name: str, args: dict):
        """A tool's function: `execute`, its outcome as a tool result (awaited if it is awaitable)."""
        outcome = self.execute(name, args)
        return _finish(outcome) if inspect.isawaitable(outcome) else _result(outcome)

    def tools(self) -> list[Tool]:
        """The simulated tools, each bound to this world."""
        return [Tool(name=d["name"], description=d["description"], input_schema=d["inputSchema"],
                     fn=functools.partial(self.call, d["name"])) for d in FAKE_TOOLS]

    def definitions(self) -> dict[str, dict]:
        """name -> MCP tool definition, for `install`."""
        return {t.name: t.definition() for t in self.tools()}
