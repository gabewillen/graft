"""Shared test helpers."""


async def clone_in_turn(h, use, retry_left=False, src="o"):
    """Make a clone call the way the harness loop does, inside a turn of `src`. A turn that
    still holds the conversation afterwards (background grafts only, or a refused call) is
    ended, so `src` is left at rest. Returns (new timeline ids or None, result text)."""
    await h.machines.begin_turn(src)
    children, focus, text = await h.clone_call(src, use, retry_left=retry_left)
    if focus is None:
        await h.machines.end_turn(src)
    return children, text
