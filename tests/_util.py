"""Shared test helpers. Tests observe the harness through what it stores and says: each
machine's state is copied to `timelines.status` in the step that enters it, so after any awaited
call the store says where every timeline is."""

import re


def conversation(store, src="o"):
    """The row ids of src's conversation a test branch starts from by default: its messages (text
    rows that are not harness notes or withdrawn) from the first user message to the last."""
    rows = [r for r in store.working_view(src)
            if r["kind"] == "text" and not r["harness"] and not r["withdrawn"]]
    users = [i for i, r in enumerate(rows) if r["role"] == "user"]
    return [r["row_id"] for r in rows[users[0]:users[-1] + 1]] if users else []


def pointing(store, use, src="o"):
    """`use` with every entry that names no messages pointed at src's conversation (by row id):
    a driven turn stores no fork call, so its branches start from their messages alone."""
    args = use.get("input")
    if not isinstance(args, dict):
        return use
    seeded = lambda e: dict(e, messages=conversation(store=store, src=src)) if isinstance(e, dict) and not e.get(
        "messages") and "reason" not in e else e
    out = dict(args)
    if "current" in out:
        out["current"] = seeded(out["current"])
    if isinstance(out.get("detached"), list):
        out["detached"] = [seeded(e) for e in out["detached"]]
    return dict(use, input=out)


async def clone_in_turn(h, use, src="o"):
    """Make a clone call the way the harness loop does, inside a turn of `src` (a turn its caller
    drives, so the call is not a row of src's view: an entry that names no messages is pointed at
    the conversation, `pointing`). A turn that still holds the conversation afterwards (background
    grafts only, or a refused call) is ended, so `src` is left at rest. Returns (new timeline ids or
    None, result text)."""
    use = pointing(store=h.store, use=use, src=src)
    await h.begin_turn(src)
    children, focus, text = await h.clone_call(src=src, use=use)
    if focus is None:
        await h.end_turn(src)
    return children, text


def status(store, tl):
    """tl's state, as its machine copied it to the store."""
    return store.timeline(tl)["status"]


def holders(store):
    """The timelines holding the conversation: one, whatever happened."""
    return [r[0] for r in store.con.execute(
        "SELECT timeline_id FROM timelines WHERE status IN ('active', 'joining') ORDER BY timeline_id")]


def clones(store):
    """The grafts made, in order (a clone call undone because its machines could not start is
    left out: its grafts are `abandoned` without ever running)."""
    return [r[0] for r in store.con.execute(
        "SELECT t.timeline_id FROM timelines t WHERE t.source_id IS NOT NULL AND NOT (t.status = 'abandoned' "
        "AND t.join_reason IS NULL) ORDER BY t.rowid")]


def holder(store):
    """The timeline holding the conversation, or None if the store says it is not exactly one."""
    run = holders(store)
    return run[0] if len(run) == 1 else None


def branch_input(*entries):
    """A fork call's input from test entries: each {"skills", "focus", "id"?, "messages"?,
    "reason"?} (no messages: an empty list); the one with focus=True is `current`, the others are
    `detached`, in order. A dict without `focus` is
    passed through as a detached entry (a malformed one, for refusal tests)."""
    current = [e for e in entries if isinstance(e, dict) and e.get("focus") is True]
    detached = [e for e in entries if not (isinstance(e, dict) and e.get("focus") is True)]
    strip = lambda e: {"messages": [], **{k: v for k, v in e.items() if k != "focus"}} if isinstance(e, dict) else e
    out = {}
    if current:
        out["current"] = strip(current[0])
    if detached or not current:
        out["detached"] = [strip(e) for e in detached]
    return out


def _quotes(messages):
    """The user's own messages in a rendered view, as quotes (no harness rows, no nonce). A text
    sent more than once is left out: a quote of it would match more than one message."""
    texts = []
    for m in messages:
        if m["role"] != "user":
            continue
        for b in m["content"]:
            if b["type"] == "text" and not b.get("harness"):
                q = re.sub(r"^\[conversation [^\]]*\]\s*", "", b["text"]).strip()
                if q:
                    texts.append(q)
    return [q for q in dict.fromkeys(texts) if texts.count(q) == 1]


def quoting(policy):
    """A scripted policy whose fork calls point every entry that names no messages and gives no
    reason at the user's messages in its view (as quotes), as a test default: the branch then
    starts from the conversation's requests, as a model that quotes them would make it."""
    def wrapped(system, messages, tools):
        blocks = policy(system, messages, tools)
        if not isinstance(blocks, list):
            return blocks
        out = []
        for b in blocks:
            args = b.get("input") if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "fork" else None
            if isinstance(args, dict):
                fill = lambda e: dict(e, messages=_quotes(messages)) if isinstance(e, dict) and not e.get(
                    "messages") and "reason" not in e else e
                args = dict(args)
                if "current" in args:
                    args["current"] = fill(args["current"])
                if isinstance(args.get("detached"), list):
                    args["detached"] = [fill(e) for e in args["detached"]]
                b = dict(b, input=args)
            out.append(b)
        return out
    return wrapped


def started(block):
    """Is this block a branch's own fork result, the last row of its first view (spec: Clone)?"""
    return (block.get("type") == "tool_result" and isinstance(block.get("content"), str)
            and block["content"].startswith("You are now on branch "))


def merged(block):
    """Is this block main's result of a branch's merge call (spec: Join), which ends main's view
    when it is woken?"""
    content = block.get("content")
    return (block.get("type") == "tool_result" and isinstance(content, str)
            and bool(re.match(r"[a-z0-9][a-z0-9-]* (merged\b|was stopped before it finished)", content)))


def ran(block):
    """Is this block the result of a tool call the timeline made itself (not a branch's fork
    result, which ends a branch's first view, nor main's result of a branch's merge call)?"""
    return block.get("type") == "tool_result" and not started(block) and not merged(block)


def woke(block):
    """Is this block what the harness wrote for main to read, a note or a merge result (what a
    wake run's view ends on)?"""
    if merged(block):
        block.setdefault("text", block["content"])  # read like a note's text
        return True
    return bool(block.get("harness"))
