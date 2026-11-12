"""Parallel grafts vs sequential clone-join on one multi-part request.

The same three deliverables (a website page, a slide deck, a calendar booking) are produced in
two ways:

  sequential: the user asks for one part per message. Each part gets a focus clone that takes the
              conversation; the next message is out of its scope, so it joins and the original
              clones again (clone, join, clone, join).
  parallel:   the user asks for all three in one message. The original makes one clone call with
              three background entries (focus=false), keeps the conversation, and is woken when
              they have joined.

Reported per arm: the checks on the simulated world, user turns, model calls, clones with their
join reasons, wakes, and wall-clock.

Offline by default: a scripted provider (graft/fake.py) plays every role, and `--latency` adds a
fixed delay to each model call so that overlap between background clones is visible in the
wall-clock. That shows the mechanics only. It says nothing about how a real model routes, seeds
or writes; use `--live` (needs an API key) for that.

    uv run python experiments/parallel.py
    uv run python experiments/parallel.py --latency 0.2 --trace
    uv run python experiments/parallel.py --live
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
from graft.fake import bench_graft_policy, parallel_graft_policy  # noqa: E402
from graft.harness import Harness  # noqa: E402
from graft.providers import FakeProvider, make_provider  # noqa: E402
from graft.sim import run_script  # noqa: E402
from graft.skills import World, install  # noqa: E402

FACTS = ("We are Okafor Dental in Columbus, Ohio. Our services: cleaning, implants, whitening. "
         "The open house is on 2026-12-05 at 10:00.")
PARTS = {
    "website": "build our website home page listing the services",
    "slides": "make a slide deck for the open house",
    "calendar": "book the open house on my calendar",
}
SEQUENTIAL = [f"{FACTS} Please {PARTS['website']}.",
              f"Thanks. Now {PARTS['slides']}.",
              f"Good. Finally, {PARTS['calendar']}: 2026-12-05 at 10:00."]
PARALLEL = [f"{FACTS} Please do three things: {PARTS['website']}; {PARTS['slides']}; and "
            f"{PARTS['calendar']}."]


def checks(world: World) -> dict[str, bool]:
    site = " ".join(world.site.values()).lower()
    deck = world.decks[-1] if world.decks else {}
    booked = [(b.get("date"), b.get("time")) for b in world.calendar]
    return {
        "site_lists_services": all(w in site for w in ("cleaning", "implants", "whitening")),
        "deck_has_slides": bool(deck.get("slides")),
        "deck_knows_the_practice": "okafor" in json.dumps(deck).lower(),
        "open_house_booked": ("2026-12-05", "10:00") in booked,
    }


def slowed(policy, latency: float):
    def call(system, messages, tools):
        time.sleep(latency)  # runs in the harness's worker thread: concurrent calls overlap
        return policy(system, messages, tools)
    return call


async def run_arm(arm: str, live: bool, latency: float, trace: bool) -> dict:
    policy = bench_graft_policy if arm == "sequential" else parallel_graft_policy
    provider = make_provider() if live else FakeProvider(slowed(policy, latency))
    path = os.path.join(tempfile.mkdtemp(prefix=f"graft-parallel-{arm}-"), "m.db")
    store = Store(path)
    install(store)
    world = World()
    # Offline means offline: no seeder call either, even with a TypeSafe key exported.
    seeder = {} if live else {"seeder": None}
    h = Harness(store, provider, world=world, trace=(lambda m: print("  · " + m)) if trace else None,
                **seeder)
    await h.start()
    script = SEQUENTIAL if arm == "sequential" else PARALLEL
    print(f"\n######## ARM: {arm} ########")
    t0 = time.perf_counter()
    transcript = await run_script(h, script)  # run_script settles background clones at the end
    wall = time.perf_counter() - t0
    h.machines.check()
    clones = [(r["timeline_id"], r["status"], r["join_reason"]) for r in store.con.execute(
        "SELECT timeline_id, status, join_reason FROM timelines WHERE source_id IS NOT NULL ORDER BY timeline_id")]
    modes = [json.loads(r[0]).get("focus") for r in store.con.execute(
        "SELECT scratch FROM transactions WHERE kind='clone' AND status='committed' ORDER BY tx_id")]
    res = {"arm": arm, "checks": checks(world), "user_turns": len(script),
           "model_calls": provider.usage["calls"], "wall_s": wall, "clones": clones,
           "background": modes.count(False), "focus": modes.count(True), "wakes": len(h.wakes),
           "clone_calls": provider.by_tag.get("original:clone_tx", {}).get("calls", 0),
           "view_violations": validate(store.view("o"), "o", require_user_end=False),
           "replies": [t for who, t in transcript if who == "assistant"], "db": path}
    await h.machines.stop()
    return res


def report(results: list[dict], live: bool, latency: float) -> bool:
    print("\n================ PARALLEL REPORT ================")
    print("provider: " + ("live model" if live else f"scripted fake, {latency:.2f}s per model call"))
    ok = True
    for r in results:
        passed = all(r["checks"].values()) and not r["view_violations"]
        ok &= passed
        print(f"\n{r['arm']}: {'PASS' if passed else 'FAIL'}")
        for name, good in r["checks"].items():
            print(f"  {'ok  ' if good else 'FAIL'} {name}")
        if r["view_violations"]:
            print(f"  FAIL original's view is invalid: {r['view_violations']}")
        print(f"  user turns {r['user_turns']}; clone calls {r['clone_calls']}; clones {len(r['clones'])} "
              f"({r['focus']} focus, {r['background']} background); wakes {r['wakes']}")
        print(f"  clones: {', '.join(f'{c} {s}/{j}' for c, s, j in r['clones'])}")
        print(f"  model calls {r['model_calls']}; wall-clock {r['wall_s']:.2f}s")
        print(f"  db: {r['db']}")
    seq, par = results
    print(f"\nwall-clock: sequential {seq['wall_s']:.2f}s, parallel {par['wall_s']:.2f}s "
          f"({seq['wall_s'] / par['wall_s']:.2f}x); model calls {seq['model_calls']} vs {par['model_calls']}")
    if not live:
        print("note: offline numbers show the mechanics only (the scripted provider plays every role and "
              "the delay is artificial). They are not evidence about model behaviour or real speed.")
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", action="store_true", help="use a real provider (needs an API key)")
    ap.add_argument("--latency", type=float, default=0.1, help="seconds added to each scripted model call")
    ap.add_argument("--trace", action="store_true", help="show clone/join internals")
    a = ap.parse_args()
    results = [await run_arm(arm, a.live, a.latency, a.trace) for arm in ("sequential", "parallel")]
    return 0 if report(results, a.live, a.latency) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
