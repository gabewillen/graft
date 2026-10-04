"""Scripted user simulator: reproducible conversations without a human."""

from __future__ import annotations

from typing import Callable, Union

from .harness import Harness

Step = Union[str, Callable[[Harness], "str | None"]]


async def run_script(h: Harness, steps: list[Step], echo: bool = True) -> list[tuple[str, str]]:
    """Each step is a user message, or a callable returning one (or None to skip)."""
    transcript: list[tuple[str, str]] = []
    for step in steps:
        text = step(h) if callable(step) else step
        if text is None:
            continue
        transcript.append(("user", text))
        if echo:
            print(f"\nUSER: {text}")
        for reply in await h.user_says(text):
            transcript.append(("assistant", reply))
            if echo:
                print(f"ASSISTANT: {reply}")
    return transcript
