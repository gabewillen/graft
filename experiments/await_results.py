"""A focus clone's host tool that awaits what three background clones file.

The host's shape: one request gets one focus clone (the website, which talks to the user) and
three background clones (a slide deck, a calendar booking, a newsletter). Publishing the home
page needs all three, since the page links to them. In this scenario the website's `site_write`
is an async host tool: it awaits, with asyncio (no polling tool, no time limit of graft's), until
the three background clones have filed their results (`slides_add_slide`, `calendar_book`,
`newsletter_draft`), and then writes the page. While it waits, the background clones run.

Recorded: model calls, input tokens (uncached + cached + cache writes), wall time, whether the
focus clone finished (replied while holding the conversation, or joined done) or failed, how long
its tool waited, and each background clone's join reason.

    uv run python experiments/await_results.py --live          # needs OPENAI_API_KEY (and a model)
    uv run python experiments/await_results.py                 # offline: the scripted fake (mechanics only)
    uv run python experiments/await_results.py --live --gate   # the filings wait until site_write waits
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft.db import Store, validate  # noqa: E402
from graft.harness import Harness  # noqa: E402
from graft.providers import FakeProvider, make_provider  # noqa: E402
from graft.skills import World, install  # noqa: E402

REQUEST = ("We are Okafor Dental in Columbus, Ohio. Our services: cleaning, implants, whitening. "
           "Our open house is on 2026-12-05 at 10:00. Please build our website home page with me; "
           "it must link to the open-house slide deck, the calendar booking and the newsletter. "
           "Meanwhile, in the background: make a slide deck for the open house, book the open house "
           "on my calendar, and draft a newsletter announcing it.")
FOLLOW_UP = "No other details: use placeholder links where you need them, and finish."
MAX_FOLLOW_UPS = 3
FILED = ("slides", "calendar", "newsletter")
WAIT_LIMIT = 900.0  # seconds the whole conversation may take before the run is reported as stuck


FILING = {"slides_add_slide": "slides", "calendar_book": "calendar", "newsletter_draft": "newsletter"}


class Awaiting(World):
    """site_write awaits until the slides, the booking and the newsletter are filed, then writes.
    Each filing sets its event when its tool runs; the wait is an asyncio wait, not a poll.
    `gate`: each filing tool first awaits `focus_waiting`, which site_write sets when it starts
    waiting, so the focus clone's tool is always reached before anything is filed and the wait
    is exercised. `events`: (seconds since the run began, what happened), for the timeline."""

    def __init__(self, gate: bool = False):
        super().__init__()
        self.filed = {name: asyncio.Event() for name in FILED}
        self.focus_waiting = asyncio.Event()
        self.gate = gate
        self.waits: list[dict] = []
        self.events: list[tuple[float, str]] = []
        self.t0 = time.perf_counter()
        self.provider = None  # set by the run: model calls made while site_write waits are counted

    def note(self, what: str) -> None:
        self.events.append((round(time.perf_counter() - self.t0, 3), what))

    def _file(self, name: str) -> None:
        if not self.filed[name].is_set():
            self.filed[name].set()
            print(f"  [world] {name} filed")

    async def execute(self, name, args):
        if name == "site_write":
            began = time.perf_counter()
            calls_before = len(self.provider.call_log) if self.provider else 0
            pending = [n for n, e in self.filed.items() if not e.is_set()]
            print(f"  [world] site_write waits for: {pending or 'nothing'}")
            self.note(f"site_write starts waiting for {pending}")
            self.focus_waiting.set()
            cancelled = False
            try:
                await asyncio.gather(*(e.wait() for e in self.filed.values()))
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                during = self.provider.call_log[calls_before:] if self.provider else []
                self.waits.append({"waited_s": round(time.perf_counter() - began, 3), "pending": pending,
                                   "cancelled": cancelled, "model_calls_while_waiting": len(during),
                                   "model_calls_while_waiting_by_tag": {
                                       t: sum(1 for c in during if c.get("tag") == t)
                                       for t in sorted({c.get("tag") for c in during})}})
                self.note("site_write stops waiting" + (" (cancelled)" if cancelled else ""))
        if name in FILING and self.gate and not self.focus_waiting.is_set():
            self.note(f"{name} waits until site_write is waiting")
            await self.focus_waiting.wait()
        result = World.execute(self, name, args)
        self.note(f"{name} ran" + (" (error)" if result[1] else ""))
        if not result[1] and name in FILING:
            self._file(FILING[name])
        return result


def scripted(system, messages, tools):
    """Offline stand-in: the original clones website (focus) and three background parts."""
    names = {t["name"] for t in tools}
    last = messages[-1]["content"][-1]
    use = lambda i, n, **a: {"type": "tool_use", "id": f"{i}{len(messages)}", "name": n, "input": a}  # noqa: E731
    if "clone" in names:
        if last["type"] == "tool_result" or last.get("harness"):
            return [{"type": "text", "text": "On it."}]
        entry = lambda s, f: {"purpose": s, "skills": [s], "focus": f, "seed_quotes": [REQUEST[:40]]}  # noqa: E731
        return [use("k", "clone", clone=[entry("website", True), entry("slides", False),
                                          entry("calendar", False), entry("newsletter", False)])]
    results = sum(1 for m in messages for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result")
    if "slides_create" in names and results < 2:
        if results == 1:
            return [use("s", "slides_add_slide", title="Open house", bullets=["2026-12-05 10:00"])]
        return [use("s", "slides_create", title="Okafor Dental open house")]
    if last["type"] == "tool_result":
        return [{"type": "text", "text": "Done."}]
    if "site_write" in names:
        return [use("w", "site_write", page="home", content="Okafor Dental: cleaning, implants, whitening")]
    if "calendar_book" in names:
        return [use("c", "calendar_book", what="open house", date="2026-12-05", time="10:00")]
    return [use("n", "newsletter_draft", subject="Open house", body="Join us on 2026-12-05.")]


async def run(live: bool, trace: bool, gate: bool = False) -> dict:
    provider = make_provider() if live else FakeProvider(scripted)
    path = os.path.join(tempfile.mkdtemp(prefix="graft-await-"), "m.db")
    store = Store(path)
    install(store)
    world = Awaiting(gate=gate)
    world.provider = provider
    h = Harness(store, provider, world=world, trace=(lambda m: print("  · " + m)) if trace else None,
                **({} if live else {"seeder": None}))
    await h.start()
    transcript, follow_ups, stuck = [], 0, False
    t0 = time.perf_counter()

    def heard(replies):
        for reply in replies:
            print(f"ASSISTANT: {reply}")
            transcript.append(("assistant", reply))
        return replies

    async def say(text):
        print(f"\nUSER: {text}")
        transcript.append(("user", text))
        return heard(await h.user_says(text))

    async def converse():
        # settle only at the end: with the gate, a background clone waits for the focus clone's
        # tool, so settling before the focus clone has reached it would wait for ever
        nonlocal follow_ups
        replies = await say(REQUEST)
        # the focus clone holds the conversation until it writes the page: while it has not (it asked
        # something, however it put it), it gets the one fixed follow-up, at most MAX_FOLLOW_UPS times
        while follow_ups < MAX_FOLLOW_UPS and not world.site:
            follow_ups += 1
            replies = await say(FOLLOW_UP)
        heard(await h.settle())

    try:
        await asyncio.wait_for(converse(), WAIT_LIMIT)
    except asyncio.TimeoutError:
        stuck = True
    wall = time.perf_counter() - t0
    rows = store.con.execute("SELECT timeline_id, status, join_reason FROM timelines "
                             "WHERE source_id IS NOT NULL ORDER BY rowid").fetchall()
    kinds = {}
    for r in store.con.execute("SELECT scratch FROM transactions WHERE kind='clone' AND status='committed'"):
        scratch = json.loads(r[0])
        for g in scratch.get("grafts", []) if isinstance(scratch, dict) else []:
            kinds[g.get("child")] = g.get("focus")
    clones = []
    for tl, status, reason in rows:
        skills = [k["name"] for k in store.skills_for(tl) if k["name"] not in ("join", "background")]
        replied = any(r["role"] == "assistant" and r["kind"] == "text" for r in store.view(tl)
                      if r["owner_timeline"] == tl)
        clones.append({"clone": tl, "skills": skills, "status": status, "join_reason": reason,
                       "replied": replied, "tools_run": h.tools_run(tl)})
    focus = [c for c in clones if "website" in c["skills"]]
    usage = provider.usage
    res = {
        "live": live, "model": getattr(provider, "model", "fake"), "stuck": stuck,
        "model_calls": usage["calls"], "input_tokens": usage["input"], "uncached_input": usage["uncached"],
        "cached_input": usage["cached"], "output_tokens": usage["output"], "wall_s": round(wall, 1),
        "calls_by_tag": {k: v["calls"] for k, v in provider.by_tag.items()},
        "follow_ups": follow_ups, "clones": clones,
        "focus_outcome": (None if not focus else
                          "failed" if focus[0]["join_reason"] == "failed" else
                          "joined " + focus[0]["join_reason"] if focus[0]["join_reason"] else
                          "replied (holds the conversation)" if focus[0]["replied"] else "no reply"),
        "gate": gate, "site_write_waits": world.waits, "timeline": world.events,
        "filed": {n: e.is_set() for n, e in world.filed.items()},
        "site_written": bool(world.site), "view_violations": validate(store.view("o"), "o", require_user_end=False),
        "replies": [t for who, t in transcript if who == "assistant"], "db": path,
    }
    await h.close()
    return res


def summary(r: dict) -> str:
    lines = [f"await-results: model={r['model']} live={r['live']} gate={r['gate']} stuck={r['stuck']}",
             f"model calls {r['model_calls']}; input tokens {r['input_tokens']} "
             f"(uncached {r['uncached_input']}, cached {r['cached_input']}); output tokens {r['output_tokens']}; "
             f"wall {r['wall_s']}s; follow-ups {r['follow_ups']}",
             f"calls by tag: {r['calls_by_tag']}",
             f"focus clone: {r['focus_outcome']}; site written {r['site_written']}; "
             f"site_write waits {r['site_write_waits']}; filed {r['filed']}"]
    for c in r["clones"]:
        lines.append(f"  {c['clone']} {c['skills']} status={c['status']} join_reason={c['join_reason']} "
                     f"replied={c['replied']} tools_run={c['tools_run']}")
    lines.append("timeline (s since the run began):")
    lines += [f"  {t:8.3f}  {what}" for t, what in r["timeline"]]
    lines.append(f"original's view violations: {r['view_violations']}")
    return "\n".join(lines)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="each filing tool waits until site_write is waiting, so the wait is exercised")
    ap.add_argument("--out", help="write <out>.json and <out>.txt")
    a = ap.parse_args()
    r = await run(live=a.live, trace=a.trace, gate=a.gate)
    text = summary(r)
    print("\n" + text)
    if a.out:
        with open(a.out + ".json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
    return 0 if not r["stuck"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
