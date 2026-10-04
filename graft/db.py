"""SQLite store for Agent Grafts (spec: "Everything is SQLite").

Tables follow the spec, with two additions to `transactions` (`target`, `scratch`): it is
the audit record of clone and default-join commits (the call's arguments and the resolved
pointers), and a `compactions` table recording each extractive compaction's pointer selection.

Row content encoding (provider-neutral):
  text         content = the text
  tool_use     content = {"id", "input"}            tool_name = tool
  tool_result  content = {"tool_use_id", "content", "is_error"}  tool_name = tool
  thinking     content = {"thinking", "signature"} or {"redacted": data}

Rendering rule: consecutive rows with the same role form one message. That is the
only mapping from rows to messages; there is no other normalization at render time.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills(
  skill_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
  description TEXT NOT NULL, instructions TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tools(
  tool_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, input_schema TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS skill_tools(
  skill_id INTEGER NOT NULL REFERENCES skills, tool_id INTEGER NOT NULL REFERENCES tools,
  PRIMARY KEY(skill_id, tool_id));

CREATE TABLE IF NOT EXISTS timelines(
  timeline_id TEXT PRIMARY KEY, source_id TEXT REFERENCES timelines,
  status TEXT NOT NULL, created_at REAL NOT NULL,
  join_reason TEXT, join_trigger INTEGER);
CREATE TABLE IF NOT EXISTS timeline_skills(
  timeline_id TEXT NOT NULL REFERENCES timelines, skill_id INTEGER NOT NULL REFERENCES skills,
  PRIMARY KEY(timeline_id, skill_id));

CREATE TABLE IF NOT EXISTS turns(
  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  owner_timeline TEXT NOT NULL REFERENCES timelines,
  role TEXT NOT NULL CHECK(role IN ('user','assistant')),
  kind TEXT NOT NULL CHECK(kind IN ('text','tool_use','tool_result','thinking')),
  tool_name TEXT, content TEXT NOT NULL, tokens INTEGER, ts REAL NOT NULL);

CREATE TABLE IF NOT EXISTS context(
  timeline_id TEXT NOT NULL REFERENCES timelines, pos INTEGER NOT NULL,
  row_id INTEGER NOT NULL REFERENCES turns,
  PRIMARY KEY(timeline_id, pos), UNIQUE(timeline_id, row_id));

CREATE TABLE IF NOT EXISTS transactions(
  tx_id INTEGER PRIMARY KEY AUTOINCREMENT, timeline_id TEXT NOT NULL REFERENCES timelines,
  kind TEXT NOT NULL CHECK(kind IN ('clone','join')),
  status TEXT NOT NULL, committed_sql TEXT,
  target TEXT, scratch TEXT);

-- Extractive compactions (spec: Compaction; see Store.working_view). Each row stores the new
-- working view as an ordered pointer list (kept rows + note rows + the fresh message) and
-- after_pos, the context position at commit; later turns (pos > after_pos) follow the list.
-- Compressed messages written by a compaction: row_id condenses the contiguous window
-- begin_row_id..end_row_id (inclusive, in the timeline's working-view order at compaction time).
CREATE TABLE IF NOT EXISTS compressions(
  row_id INTEGER PRIMARY KEY REFERENCES turns, timeline_id TEXT NOT NULL REFERENCES timelines,
  begin_row_id INTEGER NOT NULL REFERENCES turns, end_row_id INTEGER NOT NULL REFERENCES turns);
CREATE TABLE IF NOT EXISTS compactions(
  compaction_id INTEGER PRIMARY KEY AUTOINCREMENT, timeline_id TEXT NOT NULL REFERENCES timelines,
  view_row_ids TEXT NOT NULL, after_pos INTEGER NOT NULL, ts REAL NOT NULL);

CREATE TRIGGER IF NOT EXISTS turns_no_update BEFORE UPDATE ON turns
  BEGIN SELECT RAISE(ABORT, 'turns is append-only'); END;
CREATE TRIGGER IF NOT EXISTS turns_no_delete BEFORE DELETE ON turns
  BEGIN SELECT RAISE(ABORT, 'turns is append-only'); END;
CREATE TRIGGER IF NOT EXISTS context_no_update BEFORE UPDATE ON context
  BEGIN SELECT RAISE(ABORT, 'context is append-only'); END;
CREATE TRIGGER IF NOT EXISTS context_no_delete BEFORE DELETE ON context
  BEGIN SELECT RAISE(ABORT, 'context is append-only'); END;
"""

TURN_COLS = "row_id, owner_timeline, role, kind, tool_name, content, tokens, ts"
NESTING_ENABLED = False  # v1 is depth 1 (spec: Depth)
BUILTIN_SKILLS = ("clone", "join")


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    tools: list[dict]  # {"name", "description", "input_schema"}


class Store:
    def __init__(self, path: str):
        self.path = path
        self.con = sqlite3.connect(path, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(SCHEMA)

    # ---------- skills ----------
    def add_skill(self, skill: Skill) -> int:
        cur = self.con.execute(
            "INSERT INTO skills(name, description, instructions) VALUES (?,?,?)",
            (skill.name, skill.description, skill.instructions),
        )
        sid = cur.lastrowid
        for t in skill.tools:
            schema = dict(t["input_schema"])
            schema["description"] = t["description"]
            existing = self.con.execute("SELECT tool_id FROM tools WHERE name=?", (t["name"],)).fetchone()
            tid = existing[0] if existing else self.con.execute(
                "INSERT INTO tools(name, input_schema) VALUES (?,?)",
                (t["name"], json.dumps(schema)),
            ).lastrowid
            self.con.execute("INSERT INTO skill_tools VALUES (?,?)", (sid, tid))
        return sid

    def skill_names(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT name FROM skills ORDER BY skill_id")]

    def skills_for(self, tl: str) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT s.* FROM skills s JOIN timeline_skills ts USING(skill_id) "
            "WHERE ts.timeline_id=? ORDER BY s.skill_id", (tl,)).fetchall()

    def tools_for(self, tl: str) -> list[dict]:
        """Capabilities are exactly timeline_skills -> skill_tools (spec)."""
        out = []
        for r in self.con.execute(
            "SELECT DISTINCT t.tool_id, t.name, t.input_schema FROM tools t JOIN skill_tools st "
            "USING(tool_id) JOIN timeline_skills ts USING(skill_id) WHERE ts.timeline_id=? "
            "ORDER BY t.tool_id",
            (tl,),
        ):
            schema = json.loads(r["input_schema"])
            desc = schema.pop("description", "")
            out.append({"name": r["name"], "description": desc, "input_schema": schema})
        return out

    # ---------- timelines ----------
    def create_timeline(self, tl: str, source: str | None, skills: Iterable[str]) -> None:
        self.con.execute(
            "INSERT INTO timelines(timeline_id, source_id, status, created_at) VALUES (?,?,?,?)",
            (tl, source, "active", time.time()),
        )
        self._link(tl, skills)

    def _link(self, tl: str, skills: Iterable[str]) -> None:
        for name in skills:
            row = self.con.execute("SELECT skill_id FROM skills WHERE name=?", (name,)).fetchone()
            if row is None:
                raise KeyError(f"unknown skill {name}")
            self.con.execute("INSERT INTO timeline_skills VALUES (?,?)", (tl, row[0]))

    def timeline(self, tl: str) -> sqlite3.Row:
        return self.con.execute("SELECT * FROM timelines WHERE timeline_id=?", (tl,)).fetchone()

    def set_status(self, tl: str, status: str) -> None:
        self.con.execute("UPDATE timelines SET status=? WHERE timeline_id=?", (status, tl))

    def set_join(self, tl: str, reason: str, trigger: int | None) -> None:
        self.con.execute(
            "UPDATE timelines SET join_reason=?, join_trigger=? WHERE timeline_id=?",
            (reason, trigger, tl))

    def next_clone_id(self) -> str:
        n = self.con.execute("SELECT count(*) FROM timelines").fetchone()[0]
        return f"c{n}"

    def children(self, tl: str) -> list[str]:
        return [r[0] for r in self.con.execute(
            "SELECT timeline_id FROM timelines WHERE source_id=? ORDER BY created_at", (tl,))]

    def descendants(self, tl: str) -> list[str]:
        """Read scope: descendants of tl (recursive over source_id; spec Depth)."""
        return [r[0] for r in self.con.execute(
            "WITH RECURSIVE d(id) AS (SELECT timeline_id FROM timelines WHERE source_id=? "
            "UNION SELECT t.timeline_id FROM timelines t JOIN d ON t.source_id=d.id) "
            "SELECT id FROM d", (tl,))]

    def root(self) -> str:
        return self.con.execute(
            "SELECT timeline_id FROM timelines WHERE source_id IS NULL ORDER BY created_at"
        ).fetchone()[0]

    def running_leaf(self) -> str:
        """Deepest non-joined leaf, walking down from the root (never hard-coded)."""
        cur = self.root()
        while True:
            nxt = [c for c in self.children(cur) if self.timeline(c)["status"] != "joined"]
            if not nxt:
                return cur
            cur = nxt[-1]

    # ---------- turns & context ----------
    def insert_turn(self, owner: str, role: str, kind: str, content: Any,
                    tool_name: str | None = None) -> int:
        text = content if isinstance(content, str) else json.dumps(content)
        return self.con.execute(
            "INSERT INTO turns(owner_timeline, role, kind, tool_name, content, tokens, ts) "
            "VALUES (?,?,?,?,?,?,?)",
            (owner, role, kind, tool_name, text, max(1, len(text) // 4), time.time()),
        ).lastrowid

    def append(self, tl: str, row_id: int) -> int | None:
        """Append a pointer. Rows already pointed to are skipped (dedup by row_id)."""
        if self.con.execute("SELECT 1 FROM context WHERE timeline_id=? AND row_id=?",
                            (tl, row_id)).fetchone():
            return None
        pos = self.con.execute("SELECT coalesce(max(pos),-1)+1 FROM context WHERE timeline_id=?",
                               (tl,)).fetchone()[0]
        self.con.execute("INSERT INTO context VALUES (?,?,?)", (tl, pos, row_id))
        return pos

    def add_turn(self, tl: str, role: str, kind: str, content: Any,
                 tool_name: str | None = None) -> int:
        rid = self.insert_turn(tl, role, kind, content, tool_name)
        self.append(tl, rid)
        return rid

    def turn(self, row_id: int) -> dict | None:
        r = self.con.execute(f"SELECT {TURN_COLS} FROM turns WHERE row_id=?", (row_id,)).fetchone()
        return dict(r) if r else None

    def view(self, tl: str) -> list[dict]:
        """The context view: SELECT t.* FROM context JOIN turns ... ORDER BY pos."""
        return [dict(r) for r in self.con.execute(
            f"SELECT c.pos, t.{TURN_COLS.replace(', ', ', t.')} FROM context c "
            "JOIN turns t USING(row_id) WHERE c.timeline_id=? ORDER BY c.pos", (tl,))]

    def mark_compaction(self, tl: str, view_row_ids: list[int]) -> int:
        """Record an extractive compaction of tl: its working view becomes exactly view_row_ids
        (ordered pointers to rows already in tl's context), followed by turns appended later.
        The full view and the timeline are unchanged."""
        have = self.pointed(tl)
        missing = [r for r in view_row_ids if r not in have]
        if missing:
            raise ValueError(f"rows {missing} are not in {tl}'s context")
        after = self.con.execute("SELECT coalesce(max(pos), -1) FROM context WHERE timeline_id=?",
                                 (tl,)).fetchone()[0]
        return self.con.execute(
            "INSERT INTO compactions(timeline_id, view_row_ids, after_pos, ts) VALUES (?,?,?,?)",
            (tl, json.dumps(list(view_row_ids)), after, time.time())).lastrowid

    def add_compressed_turn(self, tl: str, role: str, text: str, begin_row_id: int, end_row_id: int) -> int:
        """A compressed message: a new text row owned by tl (rendered as a normal turn in its role)
        recording the window begin..end of tl's working view that it condenses."""
        rid = self.add_turn(tl, role, "text", text)
        self.con.execute("INSERT INTO compressions VALUES (?,?,?,?)", (rid, tl, begin_row_id, end_row_id))
        return rid

    def compression_window(self, row_id: int) -> tuple[int, int] | None:
        r = self.con.execute("SELECT begin_row_id, end_row_id FROM compressions WHERE row_id=?",
                             (row_id,)).fetchone()
        return (r[0], r[1]) if r else None

    def working_view(self, tl: str) -> list[dict]:
        """The context as it would be sent to the model (spec: Default join, Compaction): the full
        view, or the latest compaction's pointer selection followed by the turns appended after
        it. Without a compaction this is the full view."""
        rows = self.view(tl)
        mark = self.con.execute("SELECT view_row_ids, after_pos FROM compactions WHERE timeline_id=? "
                                "ORDER BY compaction_id DESC LIMIT 1", (tl,)).fetchone()
        if mark is None:
            return rows
        by_id = {r["row_id"]: r for r in rows}
        return [by_id[r] for r in json.loads(mark[0])] + [r for r in rows if r["pos"] > mark[1]]

    def pointed(self, tl: str) -> set[int]:
        return {r[0] for r in self.con.execute(
            "SELECT row_id FROM context WHERE timeline_id=?", (tl,))}

    def visible_to(self, viewer: str, row_id: int) -> bool:
        """Downward-only: own rows, rows in own context, or rows owned by descendants."""
        t = self.turn(row_id)
        if t is None:
            return False
        return (t["owner_timeline"] == viewer or row_id in self.pointed(viewer)
                or t["owner_timeline"] in self.descendants(viewer))

    # ---------- transactions ----------
    def begin_tx(self, tl: str, kind: str, target: str | None = None) -> int:
        return self.con.execute(
            "INSERT INTO transactions(timeline_id, kind, status, target) VALUES (?,?,?,?)",
            (tl, kind, "open", target)).lastrowid

    def end_tx(self, tx_id: int, status: str, committed_sql: str | None, scratch: Any) -> None:
        self.con.execute(
            "UPDATE transactions SET status=?, committed_sql=?, scratch=? WHERE tx_id=?",
            (status, committed_sql, json.dumps(scratch, default=str), tx_id))

    def commit_clone(self, source: str, child: str, seed: list[int], skills: list[str]) -> None:
        """One SQL transaction: timelines row, seed pointers, timeline_skills links."""
        self.con.execute("BEGIN")
        try:
            self.create_timeline(child, source, skills)
            for rid in seed:
                self.append(child, rid)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def commit_join(self, source: str, record: str, rows: list[int]) -> int:
        self.con.execute("BEGIN")
        try:
            rec = self.add_turn(source, "user", "text", record)
            for rid in rows:
                self.append(source, rid)
            self.con.execute("COMMIT")
            return rec
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def dump(self) -> str:
        out = []
        for t in self.con.execute("SELECT * FROM timelines ORDER BY created_at"):
            skills = ",".join(s["name"] for s in self.skills_for(t["timeline_id"]))
            out.append(f"== timeline {t['timeline_id']} source={t['source_id']} status={t['status']} "
                       f"skills=[{skills}] join_reason={t['join_reason']} trigger={t['join_trigger']}")
            for r in self.view(t["timeline_id"]):
                body = r["content"].replace("\n", " ")
                out.append(f"  pos={r['pos']:<3} row={r['row_id']:<4} owner={r['owner_timeline']:<4} "
                           f"{r['role']:<9} {r['kind']:<11} {r['tool_name'] or '':<16} {body[:110]}")
        for x in self.con.execute("SELECT tx_id, timeline_id, kind, status, target, committed_sql "
                                  "FROM transactions"):
            out.append(f"-- tx {x['tx_id']} {x['kind']} on {x['timeline_id']} target={x['target']} "
                       f"status={x['status']} committed={x['committed_sql']}")
        return "\n".join(out)


# ---------------------------------------------------------------- rendering
def row_block(r: dict) -> dict:
    kind = r["kind"]
    if kind == "text":
        return {"type": "text", "text": r["content"]}
    data = json.loads(r["content"])
    if kind == "tool_use":
        return {"type": "tool_use", "id": data["id"], "name": r["tool_name"], "input": data["input"]}
    if kind == "tool_result":
        return {"type": "tool_result", "tool_use_id": data["tool_use_id"],
                "content": data["content"], "is_error": bool(data.get("is_error"))}
    return {"type": "thinking", **data}


def render(rows: list[dict]) -> list[dict]:
    """Neutral messages [{role, content:[blocks]}]: consecutive same-role rows merge."""
    msgs: list[dict] = []
    for r in rows:
        if msgs and msgs[-1]["role"] == r["role"]:
            msgs[-1]["content"].append(row_block(r))
        else:
            msgs.append({"role": r["role"], "content": [row_block(r)]})
    return msgs


# ---------------------------------------------------------------- validation
def validate(rows: list[dict], viewer: str, trigger: int | None = None,
             require_user_end: bool = True) -> list[str]:
    """Validate a (pending) context view. Returns violations; empty means valid."""
    v: list[str] = []
    if not rows:
        return ["view is empty"]
    seen: set = set()
    for r in rows:
        rid = r.get("row_id")
        if rid is not None:
            if rid in seen:
                v.append(f"row {rid} appears twice")
            seen.add(rid)
        expect = {"tool_use": "assistant", "thinking": "assistant", "tool_result": "user"}.get(r["kind"])
        if expect and r["role"] != expect:
            v.append(f"row {rid}: {r['kind']} must have role {expect}")
        if r["kind"] == "thinking" and r["owner_timeline"] != viewer:
            v.append(f"row {rid}: thinking block owned by {r['owner_timeline']} is bound to its "
                     f"own conversation; exclude it from {viewer}'s view")
    if rows[0]["role"] != "user":
        v.append(f"view must start with a user turn (row {rows[0].get('row_id')} is assistant); "
                 "user and assistant turns must alternate starting with user")
    # group into messages (consecutive same role)
    groups: list[list[dict]] = []
    for r in rows:
        if groups and groups[-1][0]["role"] == r["role"]:
            groups[-1].append(r)
        else:
            groups.append([r])
    for i, g in enumerate(groups):
        if g[0]["role"] == "assistant":
            if not any(r["kind"] in ("text", "tool_use") for r in g):
                v.append(f"assistant message at rows {[r.get('row_id') for r in g]} has no text or tool_use")
            uses = {json.loads(r["content"])["id"]: r.get("row_id") for r in g if r["kind"] == "tool_use"}
            nxt = groups[i + 1] if i + 1 < len(groups) else []
            results = {json.loads(r["content"])["tool_use_id"] for r in nxt if r["kind"] == "tool_result"}
            for uid, rid in uses.items():
                if uid not in results:
                    v.append(f"tool_use row {rid} ({uid}) has no tool_result in the next user turn")
        else:
            prev = groups[i - 1] if i > 0 else []
            prev_uses = {json.loads(r["content"])["id"] for r in prev if r["kind"] == "tool_use"}
            text_seen = False
            for r in g:
                if r["kind"] == "text":
                    text_seen = True
                elif r["kind"] == "tool_result":
                    uid = json.loads(r["content"])["tool_use_id"]
                    if uid not in prev_uses:
                        v.append(f"tool_result row {r.get('row_id')} ({uid}) has no tool_use in the "
                                 "preceding assistant turn")
                    if text_seen:
                        v.append(f"tool_result row {r.get('row_id')} must come before text in its user turn")
    last = rows[-1]
    if require_user_end and last["role"] != "user":
        v.append(f"view must end on a user turn to be continuable (last row {last.get('row_id')} "
                 "is assistant)")
    if trigger is not None:
        user_text = [r for r in rows if r["role"] == "user" and r["kind"] == "text"]
        if not user_text or user_text[-1].get("row_id") != trigger or last.get("row_id") != trigger:
            v.append(f"trigger row {trigger} must be the last row (the last user turn)")
    return v
