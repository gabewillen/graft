"""A deterministic scripted policy for FakeProvider (offline tests and dry runs).

It plays every role through the same neutral interface a real model sees, so the
whole clone -> converse -> join -> re-clone path runs without an API key.
"""

from __future__ import annotations

import itertools
import re

_ids = itertools.count(1)


def _tid() -> str:
    return f"toolu_fake_{next(_ids)}"


def _use(tool: str, **inp) -> list[dict]:
    return [{"type": "tool_use", "id": _tid(), "name": tool, "input": inp}]


def _text(t: str) -> list[dict]:
    return [{"type": "text", "text": t}]


def _last_block(messages: list[dict]) -> dict:
    return messages[-1]["content"][-1]


def _user_texts(messages: list[dict]) -> list[str]:
    return [b["text"] for m in messages if m["role"] == "user" for b in m["content"] if b["type"] == "text"]


def texts_of(messages: list[dict]) -> list[str]:
    """User text blocks (the fake original quotes them as its seed)."""
    return _user_texts(messages)


def _ids_from_sql(out: str) -> list[int]:
    return [int(line.split(" | ")[0]) for line in out.splitlines()[1:] if re.match(r"^\d+", line)]


def pivot_policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    names = {t["name"] for t in tools}
    last = _last_block(messages)
    if "clone" in names:  # the original: route with one clone call
        if last["type"] == "tool_result":
            return _text("(waiting)")
        user = [re.sub(r"^\[conversation [^\]]*\]\s*", "", t) for t in texts_of(messages)]
        user = [t for t in user if not t.startswith("(harness note")]
        wants_slides = any("slide" in t.lower() for t in user)
        return _use("clone", purpose="handle the user's request",
                    skills=["slides" if wants_slides else "website"], seed_quotes=user)
    # a clone: load its capability skill first (progressive disclosure)
    skill = "website" if "site_write" in names else "slides" if "slides_create" in names else None
    loaded = any(b["type"] == "tool_use" and b["name"] == "load_skill"
                 for m in messages if m["role"] == "assistant" for b in m["content"])
    prev_tool = next((b["name"] for m in reversed(messages) if m["role"] == "assistant"
                      for b in m["content"] if b["type"] == "tool_use"), None)
    texts = _user_texts(messages)
    if skill and not loaded and last["type"] == "text":
        return _use("load_skill", name=skill)
    if last["type"] == "tool_result" and prev_tool != "load_skill":
        return _text("Done. Anything else?")
    msg = texts[-1] if texts else ""
    if "site_write" in names:
        if "slide" in msg.lower():
            return _use("join", reason="out_of_scope")
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
    return re.sub(r"^\[conversation [^\]]*\]\s*", "", t)


def _wants(text: str) -> list[str]:
    return [s for s, pat in _SKILL_WORDS if re.search(pat, text, re.I)]


def _summary(messages: list[dict]) -> list[dict]:
    return _text("summary: " + " | ".join(_clean(t) for t in _user_texts(messages))[-1500:])


def _act(names: set[str], messages: list[dict]) -> list[dict]:
    """A capability holder's action on the latest user request (shared by clones and subagents)."""
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
    if not any(u["name"] == "load_skill" for u in used) and caps:
        return _use("load_skill", name=caps[0])
    if last["type"] == "tool_result":
        if used and used[-1]["name"] == "load_skill":
            return _act(names, messages)
        errs = [b for b in messages[-1]["content"] if b["type"] == "tool_result" and b.get("is_error")]
        return _text("Sorry, I could not complete that: " + str(errs[0]["content"])[:120]) if errs \
            else _text("Done: " + str(last["content"])[:120])
    msg = _clean(_user_texts(messages)[-1]) if _user_texts(messages) else ""
    wants = _wants(msg)
    if "join" in names and wants and not set(wants) & set(caps):
        return _use("join", reason="out_of_scope")
    return _act(names, messages)


def _fake_compact(messages: list[dict]) -> list[dict]:
    """Keep the user's text rows; compress each run of other rows into one assistant message."""
    index = _user_texts(messages)[-1]
    rows = re.findall(r"row_id=(\d+) (user|assistant) (\w+)", index)
    items: list[dict] = []
    run: list[int] = []

    def flush():
        if run:
            items.append({"role": "assistant", "text": "I have been noting the details so far.",
                          "begin_row_id": run[0], "end_row_id": run[-1]})
            run.clear()

    for rid, role, kind in rows:
        if role == "user" and kind == "text":
            flush()
            items.append({"keep": int(rid)})
        else:
            run.append(int(rid))
    flush()
    return _use("compact", items=items)


def bench_graft_policy(system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    names = {t["name"] for t in tools}
    if not tools:
        return _summary(messages)
    if names == {"compact"}:
        return _fake_compact(messages)
    if "clone" in names:  # the original
        if _last_block(messages)["type"] == "tool_result":
            return _text("(waiting)")
        texts = [_clean(t) for t in _user_texts(messages)]
        texts = [t for t in texts if not t.startswith("(harness note") and not t.startswith("(compaction")]
        wants = _wants(texts[-1] if texts else "") or _wants(" ".join(texts)) or ["website"]
        uniq = [t for t in dict.fromkeys(texts) if texts.count(t) == 1]
        return _use("clone", purpose="handle the user's request", skills=wants, seed_quotes=uniq or texts[-1:])
    return _holder_policy(names, messages)


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
