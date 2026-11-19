"""Per-call-site reasoning effort and model.

Every model call has a usage tag; the tag (plus, for the original, the request shape) maps to a
call site. Each site's effort and model resolve most-specific first:

    site            env (effort / model)                              falls back to
    original        GRAFT_EFFORT_ORIGINAL / GRAFT_MODEL_ORIGINAL        global
    branch_call      GRAFT_EFFORT_BRANCH_CALL / GRAFT_MODEL_BRANCH_CALL    original, then global
    branch           GRAFT_EFFORT_BRANCH / GRAFT_MODEL_BRANCH              global
    compaction      GRAFT_EFFORT_COMPACTION / GRAFT_MODEL_COMPACTION    global
    subagent_parent GRAFT_EFFORT_SUBAGENT_PARENT / GRAFT_MODEL_SUBAGENT_PARENT  global
    subagent        GRAFT_EFFORT_SUBAGENT / GRAFT_MODEL_SUBAGENT        global

Global: effort GRAFT_REASONING_EFFORT (default "low"); model GRAFT_MODEL (provider default).

The branch_call rule (the effort must be chosen before the call, so it cannot depend on whether the
response routes): an `original` call whose request ends with a user text message, i.e. the
original is answering a fresh user message (or a join trigger) that it may route with `fork`.
The original's other calls (after a tool result, e.g. a refused branch retry) are `original`.
Compaction calls (tag `<role>:compaction`) are `compaction` in both arms; subagent-baseline tags
`parent`/`handoff` are `subagent_parent` and `subagent`/`summary` are `subagent`.
"""

from __future__ import annotations

import os

SITES = ("original", "branch_call", "branch", "compaction", "subagent_parent", "subagent")
FALLBACK = {"branch_call": ("branch_call", "original")}
DEFAULT_EFFORT = "low"


def site_for(tag: str, messages: list[dict] | None = None) -> str | None:
    if tag.endswith(":compaction"):
        return "compaction"
    if tag == "original":
        last = messages[-1] if messages else None
        fresh_user = bool(last) and last["role"] == "user" and bool(last["content"]) \
            and last["content"][-1]["type"] == "text"
        return "branch_call" if fresh_user else "original"
    if tag.startswith("original:"):
        return "original"
    if tag.startswith("branch:"):
        return "branch"
    if tag in ("parent", "handoff"):
        return "subagent_parent"
    if tag in ("subagent", "summary"):
        return "subagent"
    return None


def _lookup(site: str | None, what: str) -> str | None:
    for s in FALLBACK.get(site, (site,)) if site else ():
        v = os.environ.get(f"GRAFT_{what}_{s.upper()}")
        if v:
            return v
    return None


def effort_for(site: str | None) -> str:
    return _lookup(site, "EFFORT") or os.environ.get("GRAFT_REASONING_EFFORT") or DEFAULT_EFFORT


def model_for(site: str | None) -> str | None:
    """A per-site model override, or None for the provider's default model."""
    return _lookup(site, "MODEL")


def settings(tag: str, messages: list[dict] | None = None) -> dict:
    site = site_for(tag, messages)
    return {"site": site, "effort": effort_for(site), "model": model_for(site)}
