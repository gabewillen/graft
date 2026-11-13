"""Shared test helpers. Tests observe the harness through what it stores and says: each
machine's state is copied to `timelines.status` in the step that enters it, so after any awaited
call the store says where every timeline is."""


async def clone_in_turn(h, use, src="o"):
    """Make a clone call the way the harness loop does, inside a turn of `src` (a turn of its
    own: the call is the turn's first, with its one retry left). A turn that still holds the
    conversation afterwards (background grafts only, or a refused call) is ended, so `src` is
    left at rest. Returns (new timeline ids or None, result text)."""
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
