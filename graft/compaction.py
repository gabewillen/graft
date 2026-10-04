"""Compaction for both arms.

Trigger (identical in both arms): before a model call that answers a fresh user message, if the
conversation's working view is estimated above `threshold` tokens (characters / 4 over every
block's text; no tokenizer).

Graft (spec: Compaction, One primitive), mode `turns`: one `compact` tool call (usage tag
`<role>:compaction`) over the current working view before the fresh message returns an ordered
list of items, each {keep: row_id} or a compressed message {role, text, begin_row_id,
end_row_id}. Kept rows and windows must PARTITION that prefix: in order, contiguous, no gaps,
no overlap. A compressed message becomes a new text row owned by the timeline (provenance in
`compressions`: its window), placed at its window's position, rendered as a normal turn in its
role. New working view = the items in order + the fresh message, validated like a seed or a
join (turn structure); on violations the model gets them back for one retry; if that fails too,
no compaction happens. `Store.mark_compaction` records the pointer list.
Mode `summary` (ablation): one user-role summary row whose window is the whole prefix.

The subagent baseline keeps SUMMARY compaction (the usual practice): the model summarizes
everything before the fresh message into a user-role summary turn that replaces it.
"""

from __future__ import annotations

import json

# ------------------------------------------------------------------ shared
def estimate_tokens(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        for b in m["content"]:
            if b["type"] == "text":
                n += len(b["text"])
            elif b["type"] == "tool_use":
                n += len(b["name"]) + len(json.dumps(b["input"]))
            elif b["type"] == "tool_result":
                n += len(str(b["content"]))
    return n // 4


def needs_compaction(messages: list[dict], threshold: int | None) -> bool:
    if not threshold or not messages:
        return False
    last = messages[-1]
    fresh_user = last["role"] == "user" and any(b["type"] == "text" for b in last["content"])
    return fresh_user and estimate_tokens(messages) > threshold


def _with_user_text(prefix: list[dict], text: str) -> list[dict]:
    out = [dict(m, content=list(m["content"])) for m in prefix]
    if out and out[-1]["role"] == "user":
        out[-1]["content"].append({"type": "text", "text": text})
    else:
        out.append({"role": "user", "content": [{"type": "text", "text": text}]})
    return out


# ------------------------------------------------------------------ extractive (graft)
# The compacted view (items + fresh message) must fit this fraction of the trigger threshold, so a
# compaction actually frees room instead of re-triggering on the next turn.
BUDGET_FRACTION = 0.6

COMPACT_TOOL = {
    "name": "compact",
    "description": "Compact your conversation into an ordered list of items covering every row of "
                   "the index exactly once: keep a row, or compress a contiguous window of rows "
                   "into one message.",
    "input_schema": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"anyOf": [
            {"type": "object", "description": "Keep one row unchanged (a pointer).",
             "properties": {"op": {"type": "string", "enum": ["keep"]}, "row_id": {"type": "integer"}},
             "required": ["op", "row_id"], "additionalProperties": False},
            {"type": "object", "description": "Replace rows begin_row_id..end_row_id with one message.",
             "properties": {"op": {"type": "string", "enum": ["compress"]},
                            "role": {"type": "string", "enum": ["user", "assistant"]},
                            "text": {"type": "string"},
                            "begin_row_id": {"type": "integer"}, "end_row_id": {"type": "integer"}},
             "required": ["op", "role", "text", "begin_row_id", "end_row_id"], "additionalProperties": False}]}}},
        "required": ["items"]}}

EXTRACT_INSTRUCTION = (
    "Before you continue: your conversation is getting long. Compact it by calling the `compact` "
    "tool once with an ordered list of `items` that covers every row of the index below exactly "
    "once, in order:\n"
    "- {\"op\": \"keep\", \"row_id\": id} keeps that row unchanged (a pointer; nothing is rewritten).\n"
    "- {\"op\": \"compress\", \"role\": \"user\"|\"assistant\", \"text\": ..., \"begin_row_id\": a, "
    "\"end_row_id\": b} replaces the contiguous rows a..b (inclusive, in index order) with one condensed "
    "message. A user-role message must only restate what the user said in those rows.\n"
    "Keep short rows and exact details (names, numbers, dates, quotes) verbatim; compress long or "
    "redundant rows (tool noise, superseded drafts, chit-chat). Keep every fact, requirement and "
    "preference the user gave and the results of work you still need.\n"
    "Windows and kept rows may not overlap or skip rows. A kept tool_use row needs its kept "
    "tool_result row (or put both inside one window). The result, followed by the user's latest "
    "message (kept automatically), must start with a user turn.")


def index_text(rows: list[dict], width: int = 160) -> str:
    ids = ", ".join(str(r["row_id"]) for r in rows)
    lines = [f"Row ids in order (not consecutive numbers; use these exact ids): {ids}",
             "Index of your conversation before the latest message (row_id role kind: content):"]
    for r in rows:
        body = r["content"].replace("\n", " ")
        tool = f" {r['tool_name']}" if r.get("tool_name") else ""
        lines.append(f"row_id={r['row_id']} {r['role']} {r['kind']}{tool}: {body[:width]}")
    return "\n".join(lines)


def extract_request(prefix_msgs: list[dict], prefix_rows: list[dict], violations: list[str] | None,
                    budget: int | None = None, current: int | None = None) -> list[dict]:
    text = EXTRACT_INSTRUCTION
    if budget:
        text += (f"\nBudget: the conversation is about {current} tokens; the compacted result must be at "
                 f"most about {budget} tokens, so compress windows of older or verbose rows.")
    text += "\n\n" + index_text(prefix_rows)
    if violations:
        text += ("\n\nYour previous compaction was refused:\n- " + "\n- ".join(violations)
                 + "\nCall compact once more with a valid list of items.")
    return _with_user_text(prefix_msgs, text)


def _int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_items(blocks: list[dict]) -> list | None:
    """The compact call's raw item list, or None if no compact call was made."""
    use = next((b for b in blocks if b["type"] == "tool_use" and b["name"] == "compact"), None)
    if use is None:
        return None
    items = use["input"].get("items")
    return items if isinstance(items, list) else []


def partition(prefix: list[dict], items: list) -> tuple[list[dict], list[str]]:
    """Check that items partition `prefix` (in order, contiguous, no gaps or overlap) and build
    the new rows: kept row dicts, or virtual compressed rows (row_id None, with 'window').
    Returns (rows, problems)."""
    ids = [r["row_id"] for r in prefix]
    at = {rid: i for i, rid in enumerate(ids)}
    out: list[dict] = []
    problems: list[str] = []
    cur = 0
    for k, it in enumerate(items):
        if not isinstance(it, dict):
            problems.append(f"item {k} is not an object")
            continue
        if cur >= len(ids):
            problems.append(f"item {k} is past the end of the index (every row is already covered)")
            break
        want = ids[cur]
        # The op field decides; extra fields on a keep are ignored. Items without op fall back to
        # the old shape ({keep: id} or a window) for robustness.
        op = it.get("op")
        if op is None:
            op = "compress" if (str(it.get("text") or "").strip() and _int(it.get("begin_row_id")) is not None
                                and it.get("keep") in (None, 0)) else ("keep" if it.get("keep") is not None else None)
        if op not in ("keep", "compress"):
            problems.append(f"item {k} needs op 'keep' or 'compress'")
            break
        if op == "keep":
            rid = _int(it.get("row_id", it.get("keep")))
            if rid != want:
                problems.append(f"item {k} keeps row {it.get('row_id', it.get('keep'))}, but the next uncovered row is {want}: item {k} "
                                f"must be {{op: keep, row_id: {want}}} or a window with begin_row_id {want} (items must cover "
                                f"the row ids in order without gaps or overlap; ids are not consecutive)")
                break
            out.append(prefix[cur])
            cur += 1
            continue
        role, text = it.get("role"), str(it.get("text") or "").strip()
        b, e = _int(it.get("begin_row_id")), _int(it.get("end_row_id"))
        if role not in ("user", "assistant") or not text or b is None or e is None:
            problems.append(f"item {k} must be {{op: keep, row_id}} or {{op: compress, role, text, begin_row_id, "
                            f"end_row_id}}")
            break
        if b != want:
            problems.append(f"item {k} window begins at row {b}, but the next uncovered row is {want}: item {k} "
                            f"must be {{op: keep, row_id: {want}}} or a window with begin_row_id {want}")
            break
        if e not in at or at[e] < at[b]:
            problems.append(f"item {k} window end row {e} is not at or after row {b} in the index")
            break
        out.append({"row_id": None, "owner_timeline": None, "role": role, "kind": "text", "tool_name": None,
                    "content": text, "window": (b, e)})
        cur = at[e] + 1
    if not problems and cur < len(ids):
        problems.append(f"rows {ids[cur:]} are not covered (every row must be kept or inside one window)")
    return out, problems


# ------------------------------------------------------------------ summary (subagent baseline)
SUMMARY_HEADER = ("(compaction summary of the earlier conversation, written by you for yourself; "
                  "the conversation continues after it)")

COMPACT_INSTRUCTION = (
    "Before you continue: your conversation is getting long. Write a compaction summary of "
    "everything above for your own continued work. Keep every specific fact, name, number, date, "
    "requirement and preference the user gave, verbatim where possible, and what you have done so "
    "far (including any tool work and its results) and what was still open. Output only the summary.")


def request(prefix: list[dict]) -> list[dict]:
    """Summary compaction request: the conversation before the fresh user message, plus the
    instruction as a user turn (merged into a trailing user message, so roles alternate)."""
    return _with_user_text(prefix, COMPACT_INSTRUCTION)


def summary_turn(blocks: list[dict]) -> str:
    text = "\n".join(b["text"] for b in blocks if b["type"] == "text").strip() or "(empty summary)"
    return f"{SUMMARY_HEADER}\n{text}"
