"""Trace where each scored long_compaction fact disappears in a graft run (post-hoc, from the store).

For every scored fact missing from the final deck, the first stage where it is no longer present:
  (a) compaction  - the fact was told to the website clone but is not in its working view at join
                    (no kept or compressed row carries it); also if the original compacted it away
  (b) join        - in the website clone's working view, but not in the rows joined to the original
                    (the default join takes text rows only)
  (c) seeding     - in the original's working view before the slides clone, but not in its seed
  (d) clone       - in the slides clone's seed, but not in the deck
  (e) scorer      - a loose pattern finds it in the deck (scorer false negative)

Usage: uv run python experiments/diagnose_long.py STORE.db [...]   (or --json bench_out.json)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graft.db import Store  # noqa: E402

SCORED = {"practice name": r"harbor light", "city": r"duluth", "Dr. Okafor": r"okafor", "Dr. Brandt": r"brandt",
          "phone": r"218-555-0143", "cleanings": r"clean", "crowns": r"crown", "Invisalign": r"invisalign"}
LOOSE = {"practice name": r"harbor", "city": r"duluth|minnesota", "Dr. Okafor": r"okaf|amara", "Dr. Brandt": r"brandt|lena",
         "phone": r"218\D{0,3}555\D{0,3}0143|5550143", "cleanings": r"clean|hygien|prophyl", "crowns": r"crown",
         "invisalign": r"invisalign|aligner", "Invisalign": r"invisalign|aligner"}
STAGES = ("a compaction", "b join", "c seeding", "d clone", "e scorer")


def has(pat: str, rows) -> bool:
    return any(re.search(pat, r["content"], re.I) for r in rows)


def deck_rows(s: Store, slides: str) -> list[dict]:
    rows = [r for r in s.view(slides) if r["kind"] == "tool_use" and (r["tool_name"] or "").startswith("slides_")
            and r["tool_name"] != "slides_read"]
    last = max((i for i, r in enumerate(rows) if r["tool_name"] == "slides_create"), default=None)
    return rows[last:] if last is not None else []


def diagnose(path: str) -> dict:
    s = Store(path)
    con = s.con
    clones = [(r[0]) for r in con.execute("SELECT timeline_id FROM timelines WHERE source_id IS NOT NULL ORDER BY created_at")]
    skills = {c: [r["name"] for r in s.skills_for(c)] for c in clones}
    web = [c for c in clones if "website" in skills[c]]
    slides = [c for c in clones if "slides" in skills[c]]
    out = {"store": path, "web": web, "slides": slides, "missing": {}, "examples": {}}
    if not slides:
        out["missing"] = {f: "c seeding" for f in SCORED}  # no slides clone at all
        return out
    sl = slides[-1]
    deck = deck_rows(s, sl)
    seed = [r for r in s.view(sl) if r["owner_timeline"] != sl]
    first_sl_row = min((r["row_id"] for r in s.view(sl) if r["owner_timeline"] == sl), default=10 ** 9)
    # harness notes (join records index row snippets) are not seedable carriers of a fact
    o_wv = [r for r in s.working_view("o") if r["row_id"] < first_sl_row
            and not r["content"].startswith("(harness note")]
    joined = [r for r in s.view("o") if r["owner_timeline"] in web]
    web_wv = [r for c in web for r in s.working_view(c)]
    web_full = [r for c in web for r in s.view(c)]
    o_full = [r for r in s.view("o") if not r["content"].startswith("(harness note")]
    svc = sum(has(SCORED[k], deck) for k in ("cleanings", "crowns", "Invisalign"))
    core = [f for f in ("practice name", "city", "Dr. Okafor", "Dr. Brandt", "phone") if not has(SCORED[f], deck)]
    out["failed"] = bool(core) or svc < 2
    if not out["failed"]:
        return out
    check = core + ([k for k in ("cleanings", "crowns", "Invisalign") if not has(SCORED[k], deck)] if svc < 2 else [])
    for f in check:
        pat = SCORED[f]
        if has(LOOSE.get(f, pat), deck):
            stage, ex = "e scorer", "a loose pattern matches the deck"
        elif has(pat, seed):
            stage, ex = "d clone", f"seeded into {sl} (seed {[r['row_id'] for r in seed]}) but absent from its deck"
        elif has(pat, o_wv):
            src = [r for r in o_wv if re.search(pat, r["content"], re.I)][0]
            stage = "c seeding"
            ex = (f"o holds row {src['row_id']} ({src['role']} {src['kind']}, owner {src['owner_timeline']}): "
                  f"{src['content'][:90]!r}; {sl} seed = {[r['row_id'] for r in seed]}")
        elif has(pat, joined) or has(pat, o_full):
            stage, ex = "a compaction", "reached the original but its own compaction dropped it before seeding"
        elif has(pat, web_wv):
            src = [r for r in web_wv if re.search(pat, r["content"], re.I)][0]
            stage, ex = "b join", f"only in {src['owner_timeline']} {src['role']} {src['kind']} row {src['row_id']} (not a text row)"
        else:
            where = [r["row_id"] for r in web_full if re.search(pat, r["content"], re.I)][:3]
            stage, ex = "a compaction", f"in {web} full view (rows {where}) but in no kept/compressed row of its working view"
        out["missing"][f] = stage
        out["examples"][f] = ex
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", nargs="*")
    ap.add_argument("--json", action="append", default=[], help="bench --out file(s); uses each graft record's store")
    a = ap.parse_args()
    paths = list(a.stores)
    for j in a.json:
        paths += [r["store"] for r in json.load(open(j)) if r.get("store") and r["scenario"] == "long_compaction"]
    counts = {k: 0 for k in STAGES}
    examples: dict[str, str] = {}
    failing = 0
    for p in paths:
        d = diagnose(p)
        if not d["missing"]:
            continue
        failing += 1
        print(f"{os.path.basename(os.path.dirname(p))}: web={d['web']} slides={d['slides']} missing={d['missing']}")
        for f, st in d["missing"].items():
            counts[st] += 1
            examples.setdefault(st, f"{f}: {d['examples'].get(f, '')}")
    print(f"\nruns with a missing scored fact: {failing}/{len(paths)}")
    for st in STAGES:
        print(f"  {st:<13} {counts[st]:>3}   e.g. {examples.get(st, '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
