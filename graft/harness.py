"""The Graft harness loop.

1. Read the running timeline's context view (validated).
2. Resolve its tools through timeline_skills -> skill_tools.
3. Call the model.
4. Insert the new turns once and append their pointers; execute tools.

Clone is one call: `clone(purpose, skills, seed_quotes | seed_row_ids)`; the harness resolves
the seed to pointers, validates, and commits (on violations the source gets them back for one
retry). Join is deterministic (no model call): the harness appends pointers to the text rows of
the clone's working view that the source lacks. Both commits are recorded in `transactions`.
A clone may not join `done` before it has replied to the user's last message.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Callable

from . import compaction
from .db import NESTING_ENABLED, Store, render, validate
from .machine import Machines
from .providers import Provider, prefix_nonce
from .skills import BASE_SYSTEM, World, load_skill, skill_listing

CLONE_ATTEMPTS = 2  # the first clone call plus one retry after violations
_NONCE_TAG = re.compile(r"^\[conversation [^\]]*\]\s*")


def check_skill_link(store: Store, source: str, skills: list[str]) -> list[str]:
    """The one place skill linking is checked (nesting adds the subset rule here)."""
    v = []
    known = set(store.skill_names())
    if not skills:
        v.append("select at least one skill")
    for s in skills:
        if s not in known:
            v.append(f"unknown skill '{s}'")
        elif s == "join":
            v.append("'join' is linked to every clone automatically; do not select it")
        elif s == "clone" and not NESTING_ENABLED:
            v.append("'clone' cannot be linked to a clone in v1 (depth 1)")
    if store.timeline(source)["source_id"] is not None:  # a clone cloning: capability narrows
        own = {r["name"] for r in store.skills_for(source)}
        for s in skills:
            if s not in own:
                v.append(f"skill '{s}' is not held by {source} (child skills must be a subset)")
    return v


class Harness:
    def __init__(self, store: Store, provider: Provider, world: World | None = None,
                 trace: Callable[[str], None] | None = None, max_steps: int = 10,
                 persona: str = BASE_SYSTEM, nonce: str | None = None,
                 compact_tokens: int | None = None, compaction_mode: str = "turns"):
        self.store = store
        self.persona = persona  # user-provided; identical, verbatim, on every timeline
        self.provider = provider
        self.world = world or World()
        self.machines = Machines()
        self.trace = trace or (lambda s: None)
        self.max_steps = max_steps
        self.silent_done_refusals: list[str] = []  # clones whose join(done) was refused
        self.out: list[str] = []
        # Per-run conversation nonce, prefixed to the first message of every request at render
        # time (never stored). None = no nonce (cache experiments: --cache-scope global).
        self.nonce = nonce
        # Compaction threshold (estimated tokens of a working view); None = never compact.
        self.compact_tokens = compact_tokens
        if compaction_mode not in ("turns", "summary"):
            raise ValueError(f"compaction_mode must be 'turns' or 'summary', got {compaction_mode!r}")
        # turns: kept rows + compressed windows (default); summary: one summary row (ablation)
        self.compaction_mode = compaction_mode
        self.compactions: list[str] = []  # timelines that compacted, in order
        self.compaction_log: list[dict] = []  # per compaction: tokens before/after, kept, notes
        self.compaction_failures: list[str] = []  # timelines whose selection was refused twice

    def complete(self, tag: str, system: str, messages: list[dict], tools: list[dict], **kw) -> list[dict]:
        """Every model call goes through here: applies the run nonce and the usage-role tag."""
        tagger = getattr(self.provider, "usage_tag", None)
        with tagger(tag) if tagger else contextlib.nullcontext():
            return self.provider.complete(system, prefix_nonce(messages, self.nonce), tools, **kw)

    def role(self, tl: str) -> str:
        return "original" if self.store.timeline(tl)["source_id"] is None else f"clone:{tl}"

    # ------------------------------------------------------------ setup
    async def start(self, root: str = "o") -> None:
        self.store.create_timeline(root, None, ["clone"])
        await self.machines.start_root(root)
        self._mirror()

    def _mirror(self) -> None:
        for t in self.machines.timelines:
            self.store.set_status(t, self.machines.state(t))

    def system(self, tl: str) -> str:
        """Persona (same on every timeline) + skill listing (the only per-timeline part)."""
        names = [s["name"] for s in self.store.skills_for(tl)]
        parts = [self.persona, skill_listing(
            self.store, names,
            "Your skills (call load_skill(name) to read a skill's full instructions before using it):")]
        if "clone" in names:
            catalog = [r[0] for r in self.store.con.execute(
                "SELECT name FROM skills WHERE name NOT IN ('clone','join') ORDER BY skill_id")]
            parts.append(skill_listing(self.store, catalog, "Skill catalog (available to clones):"))
        return "\n\n".join(parts)

    def load(self, tl: str, name: str) -> tuple[str, bool]:
        return load_skill(self.store, {s["name"] for s in self.store.skills_for(tl)}, name)

    def say(self, tl: str, text: str) -> None:
        self.out.append(text)
        self.trace(f"[{tl}] assistant: {text}")

    # ------------------------------------------------------------ user entry
    async def user_says(self, text: str) -> list[str]:
        leaf = self.store.running_leaf()
        if self.machines.state(leaf) != "active":
            raise RuntimeError(f"leaf {leaf} is {self.machines.state(leaf)}, not active")
        start = len(self.out)
        rid = self.store.add_turn(leaf, "user", "text", text)
        self.trace(f"[{leaf}] user (row {rid}): {text}")
        await self.run(leaf)
        return self.out[start:]

    # ------------------------------------------------------------ main loop
    def _store_results(self, tl: str, results: list[tuple[dict, str, bool]]) -> None:
        for use, content, err in results:
            self.store.add_turn(tl, "user", "tool_result",
                                {"tool_use_id": use["id"], "content": content, "is_error": err},
                                tool_name=use["name"])

    def _store_blocks(self, tl: str, blocks: list[dict]) -> None:
        for b in blocks:
            if b["type"] == "text":
                self.store.add_turn(tl, "assistant", "text", b["text"])
            elif b["type"] == "tool_use":
                self.store.add_turn(tl, "assistant", "tool_use", {"id": b["id"], "input": b["input"]},
                                    tool_name=b["name"])
            elif b["type"] == "thinking":
                payload = {k: v for k, v in b.items() if k != "type"}
                self.store.add_turn(tl, "assistant", "thinking", payload)

    def maybe_compact(self, tl: str) -> None:
        """Compaction (spec: Compaction; graft/compaction.py). When the working view, ending in
        a fresh user message, is over the threshold, one `compact` call partitions the prefix into
        kept rows and compressed windows; the new view (items in order + the fresh message) is
        validated (partition, then turn structure like a seed or join), with one retry on
        violations, then recorded with Store.mark_compaction. The full timeline is unchanged."""
        rows = self.store.working_view(tl)
        if len(rows) < 2 or not compaction.needs_compaction(render(rows), self.compact_tokens):
            return
        prefix, fresh = rows[:-1], rows[-1]
        if self.compaction_mode == "summary":
            self._summary_compact(tl, rows)
            return
        violations: list[str] | None = None
        for attempt in range(2):
            budget = int(self.compact_tokens * compaction.BUDGET_FRACTION)
            blocks = self.complete(f"{self.role(tl)}:compaction", self.system(tl),
                                   compaction.extract_request(render(prefix), prefix, violations, budget,
                                                              compaction.estimate_tokens(render(rows))),
                                   [compaction.COMPACT_TOOL], max_tokens=4000)
            items = compaction.parse_items(blocks)
            if items is None:
                violations = ["call the compact tool (no tool call was made)"]
                continue
            new_rows, problems = compaction.partition(prefix, items)
            for it in new_rows:
                if it["row_id"] is None:
                    it["owner_timeline"] = tl
            violations = problems or validate(new_rows + [fresh], tl)
            if not violations:
                size = compaction.estimate_tokens(render(new_rows + [fresh]))
                if size > budget:
                    violations = [f"the compacted conversation is about {size} tokens, over the budget of "
                                  f"{budget}; compress more rows into windows"]
            if violations:
                self.trace(f"[{tl}] compaction refused (attempt {attempt + 1}): {violations}")
                continue
            view_ids = [it["row_id"] if it["row_id"] is not None else
                        self.store.add_compressed_turn(tl, it["role"], it["content"], *it["window"])
                        for it in new_rows] + [fresh["row_id"]]
            self.store.mark_compaction(tl, view_ids)
            self._log_compaction(tl, rows, view_ids, attempt + 1)
            self.trace(f"[{tl}] COMPACTED: {len(new_rows)} items for {len(prefix)} rows")
            return
        self.compaction_failures.append(tl)
        self.trace(f"[{tl}] compaction abandoned after 2 attempts: {violations}")

    def _summary_compact(self, tl: str, rows: list[dict]) -> None:
        """Ablation (--compaction summary): one summary of everything before the fresh message,
        stored as a user-role compressed row whose window is the whole prefix; view = [summary,
        fresh message]."""
        prefix, fresh = rows[:-1], rows[-1]
        blocks = self.complete(f"{self.role(tl)}:compaction", self.system(tl),
                               compaction.request(render(prefix)), [], max_tokens=2000)
        sid = self.store.add_compressed_turn(tl, "user", compaction.summary_turn(blocks),
                                             prefix[0]["row_id"], prefix[-1]["row_id"])
        view_ids = [sid, fresh["row_id"]]
        self.store.mark_compaction(tl, view_ids)
        self._log_compaction(tl, rows, view_ids, 1)
        self.trace(f"[{tl}] COMPACTED (summary) -> row {sid}")

    def _log_compaction(self, tl: str, rows: list[dict], view_ids: list[int], attempts: int) -> None:
        """Record a committed compaction for metrics: tokens before/after, kept rows, and each
        compressed message with its window (the prefix rows it condenses)."""
        prefix = rows[:-1]
        ids = [r["row_id"] for r in prefix]
        windows = []
        for rid in view_ids:
            w = self.store.compression_window(rid)
            if w:
                covered = ids[ids.index(w[0]):ids.index(w[1]) + 1]
                windows.append({"row_id": rid, "begin": w[0], "end": w[1], "covers": covered,
                                "text": self.store.turn(rid)["content"]})
        self.compactions.append(tl)
        self.compaction_log.append({
            "timeline": tl, "before_tokens": compaction.estimate_tokens(render(rows)),
            "after_tokens": compaction.estimate_tokens(render(self.store.working_view(tl))),
            "rows_before": len(rows), "kept_ids": [i for i in view_ids[:-1] if i in set(ids)],
            "windows": windows, "prefix": [(r["row_id"], r["content"]) for r in prefix],
            "fresh": rows[-1]["content"], "attempts": attempts})

    async def run(self, tl: str) -> None:
        clone_attempts = 0
        self.maybe_compact(tl)
        for _ in range(self.max_steps):
            rows = self.store.working_view(tl)
            violations = validate(rows, tl)
            if violations:
                raise RuntimeError(f"{tl} view invalid before model call: {violations}")
            tools = self.store.tools_for(tl)
            blocks = self.complete(self.role(tl), self.system(tl), render(rows), tools)
            if not any(b["type"] in ("text", "tool_use") for b in blocks):
                self.trace(f"[{tl}] (empty response)")
                return
            self._store_blocks(tl, blocks)
            for b in blocks:
                if b["type"] == "text":
                    self.say(tl, b["text"])
            uses = [b for b in blocks if b["type"] == "tool_use"]
            if not uses:
                return
            allowed = {t["name"] for t in tools}
            join_use = next((u for u in uses if u["name"] == "join" and "join" in allowed), None)
            clone_use = next((u for u in uses if u["name"] == "clone" and "clone" in allowed), None)
            refused_join = None
            if join_use and str(join_use["input"].get("reason")) == "done" and not self.replied(tl):
                refused_join, join_use = join_use, None
                self.silent_done_refusals.append(tl)
                self.trace(f"[{tl}] join(done) refused: no reply to the user since their last message")
            if join_use:
                results = [(u, "not executed: this task is being joined", True) for u in uses if u is not join_use]
                results.append((join_use, "joined", False))
                self._store_results(tl, results)
                await self.do_join(tl, str(join_use["input"].get("reason", "failed")))
                return
            if clone_use:
                # this call's role was routing into a clone: account it as the clone transaction
                retag = getattr(self.provider, "retag_last", None)
                if retag:
                    retag(f"{self.role(tl)}:clone_tx")
                others = [(u, "not executed: call clone on its own", True) for u in uses if u is not clone_use]
                clone_attempts += 1
                child, text = await self.clone_once(tl, clone_use, retry_left=clone_attempts < CLONE_ATTEMPTS)
                self._store_results(tl, others + [(clone_use, text, child is None)])
                if child:
                    await self.run(child)
                    return
                continue
            results = []
            for u in uses:
                if u is refused_join:
                    results.append((u, "join(reason='done') refused: you have not replied to the user since "
                                       "their last message. First tell the user, in one or two sentences, what "
                                       "you did; then call join(reason='done').", True))
                elif u["name"] not in allowed:
                    results.append((u, f"tool '{u['name']}' is not available to you", True))
                elif u["name"] == "load_skill":
                    content, err = self.load(tl, str(u["input"].get("name", "")))
                    self.trace(f"[{tl}] load_skill({u['input'].get('name')}) -> {'error' if err else 'ok'}")
                    results.append((u, content, err))
                else:
                    content, err = self.world.execute(u["name"], u["input"])
                    self.trace(f"[{tl}] tool {u['name']}({json.dumps(u['input'])[:120]}) -> {content[:80]}")
                    results.append((u, content, err))
            self._store_results(tl, results)
        if self.store.timeline(tl)["source_id"] is not None:
            self.trace(f"[{tl}] step budget exhausted -> join failed")
            await self.do_join(tl, "failed")

    # ------------------------------------------------------------ join event
    def replied(self, tl: str) -> bool:
        """Has tl produced user-facing assistant text since the last user text message (in its
        working view)?"""
        answered = False
        for r in self.store.working_view(tl):
            if r["role"] == "user" and r["kind"] == "text":
                answered = False
            elif r["role"] == "assistant" and r["kind"] == "text":
                answered = True
        return answered

    def compute_trigger(self, child: str, reason: str) -> int | None:
        rows = self.store.working_view(child)
        last_user = None
        answered = False
        for r in rows:
            if r["role"] == "user" and r["kind"] == "text":
                last_user, answered = r["row_id"], False
            elif r["role"] == "assistant" and r["kind"] == "text":
                answered = True
        if last_user is None:
            return None
        if reason == "out_of_scope" or not answered:
            return last_user
        return None

    async def do_join(self, child: str, reason: str) -> None:
        if reason not in ("done", "out_of_scope", "failed"):
            reason = "failed"
        trigger = self.compute_trigger(child, reason)
        self.store.set_join(child, reason, trigger)
        src = await self.machines.join(child)  # deterministic, no model call
        self._mirror()
        self.trace(f"JOIN EVENT {{clone: {child}, reason: {reason}, trigger: {trigger}}} -> {src} joining")
        committed = self.default_join(src, child, reason, trigger)
        await self.machines.finish_join(src, committed)
        self._mirror()
        if trigger is not None:
            await self.run(src)

    # ------------------------------------------------------------ default join (deterministic)
    def default_join_rows(self, src: str, child: str, reason: str, trigger: int | None
                          ) -> tuple[list[int], str | None]:
        """Spec 'Default join': text rows (user + assistant) of the clone's working view that the
        source lacks; out_of_scope ends with the trigger; failed adds a harness error record.
        Returns (row ids to append, error record text or None)."""
        have = self.store.pointed(src)
        rows = [r["row_id"] for r in self.store.working_view(child)
                if r["kind"] == "text" and r["row_id"] not in have and r["row_id"] != trigger]
        error = (f"(harness note: clone {child} failed; its timeline is frozen)"
                 if reason == "failed" else None)
        if trigger is not None and trigger not in have:
            rows.append(trigger)
        return rows, error

    def default_join(self, src: str, child: str, reason: str, trigger: int | None) -> bool:
        rows, error = self.default_join_rows(src, child, reason, trigger)
        record = (f"(harness note: clone {child} joined, reason={reason}; remembered rows "
                  f"{', '.join(map(str, rows)) or 'none'} follow)")
        pending = self.store.view(src) + [self._virtual(src, "user", "text", record)]
        pending += [self.store.turn(r) for r in rows if trigger is None or r != trigger]
        if error:
            pending.append(self._virtual(src, "user", "text", error))
        if trigger is not None and trigger in rows:
            pending.append(self.store.turn(trigger))
        trig = trigger if trigger is not None and trigger not in self.store.pointed(src) else None
        v = validate(pending, src, trigger=trig, require_user_end=trig is not None)
        tx = self.store.begin_tx(src, "join", target=child)
        if v:
            # Cannot happen for well-formed text rows; keep the source continuable anyway.
            self.trace(f"[{src}] default join invalid: {v}; appending record + trigger only")
            self.store.commit_join(src, record.replace("follow", "skipped (invalid view)"),
                                   [trigger] if trig is not None else [])
            self.store.end_tx(tx, "rolled_back", None, {"default": True, "violations": v})
            return False
        self.store.commit_join(src, record, [r for r in rows if r != trigger])
        if error:
            self.store.add_turn(src, "user", "text", error)
        if trig is not None:
            self.store.append(src, trig)
        sql = f"DEFAULT JOIN {child} reason={reason} rows={rows}"
        self.store.end_tx(tx, "committed", sql, {"default": True})
        self.trace(f"[{src}] {sql}")
        return True

    @staticmethod
    def _virtual(owner: str, role: str, kind: str, content: str) -> dict:
        return {"row_id": None, "owner_timeline": owner, "role": role, "kind": kind,
                "tool_name": None, "content": content}

    # ------------------------------------------------------------ one-shot clone
    def resolve_seed(self, src: str, quotes: list, row_ids: list) -> tuple[list[int], list[str]]:
        """Seed items -> row ids (pointers). A quote must match exactly one text row in the
        source's view (exact text first, then substring). Order: as given, ids then quotes,
        re-sorted by position in the source's view so the seed reads in conversation order."""
        view = self.store.view(src)
        texts = [r for r in view if r["kind"] == "text"]
        out: list[int] = []
        problems: list[str] = []
        for x in row_ids:
            try:
                rid = int(x)
            except (TypeError, ValueError):
                problems.append(f"seed_row_ids item {x!r} is not an integer")
                continue
            if not self.store.visible_to(src, rid):
                problems.append(f"row {rid} is not visible to {src}")
            else:
                out.append(rid)
        for q in quotes:
            qs = _NONCE_TAG.sub("", str(q)).strip().strip('"').strip()
            if not qs:
                continue
            exact = [r for r in texts if r["content"].strip() == qs]
            hits = exact or [r for r in texts if qs in r["content"]]
            if len(hits) == 1:
                out.append(hits[0]["row_id"])
            elif not hits:
                problems.append(f"seed quote {qs[:60]!r} matches no message")
            else:
                problems.append(f"seed quote {qs[:60]!r} matches {len(hits)} messages "
                                f"(rows {[r['row_id'] for r in hits]}); quote more or use seed_row_ids")
        pos = {r["row_id"]: r["pos"] for r in view}
        seen: set[int] = set()
        uniq = [r for r in out if not (r in seen or seen.add(r))]
        return sorted(uniq, key=lambda r: pos.get(r, 1 << 30)), problems

    def seed_index(self, src: str) -> str:
        lines = [f"Your messages (row_id role: text):"]
        for r in self.store.view(src):
            if r["kind"] == "text":
                lines.append(f"  {r['row_id']} {r['role']}: {r['content'].replace(chr(10), ' ')[:100]}")
        return "\n".join(lines)

    async def clone_once(self, src: str, use: dict, retry_left: bool) -> tuple[str | None, str]:
        """One-shot clone: resolve, validate, commit. No model call of its own."""
        args = use["input"]
        child = self.store.next_clone_id()
        tx = self.store.begin_tx(src, "clone", target=child)
        skills = [str(x) for x in (args.get("skills") or [])]
        seed, problems = self.resolve_seed(src, list(args.get("seed_quotes") or []),
                                           list(args.get("seed_row_ids") or []))
        v = problems + check_skill_link(self.store, src, skills)
        rows = [t for t in (self.store.turn(r) for r in seed) if t]
        v += validate(rows, child) if rows else ["the seed is empty: seed at least the user message "
                                                 "that asks for the task"]
        scratch = {"args": args, "seed": seed, "violations": v}
        if v:
            self.store.end_tx(tx, "rolled_back", None, scratch)
            self.trace(f"[{src}] clone refused: {v}")
            tail = ("Fix the seed or skills and call clone once more.\n" + self.seed_index(src)) if retry_left \
                else "No retries left for this message; answer the user instead."
            return None, "clone refused:\n- " + "\n- ".join(v) + "\n" + tail
        self.store.commit_clone(src, child, seed, skills + ["join"])
        await self.machines.clone(src, child)
        self._mirror()
        sql = f"CLONE {child} SEED {seed} SKILLS {skills + ['join']}"
        self.store.end_tx(tx, "committed", sql, scratch)
        self.trace(f"[{src}] COMMIT clone -> {child} seed={seed} skills={skills}")
        return child, f"cloned -> {child}, active (seed rows {seed}; skills {skills + ['join']})"
