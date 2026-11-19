"""SQLite store for Agent Grafts (spec: "Everything is SQLite").

Tables follow the spec, with two additions to `transactions` (`target`, `scratch`): it is
the audit record of branch and default-join commits (the call's arguments and the resolved
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
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

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
  -- the id the model knows a graft by (`fork`'s `id`, or one drawn): unique, never reused
  name TEXT UNIQUE,
  join_reason TEXT, join_trigger INTEGER);
CREATE TABLE IF NOT EXISTS timeline_skills(
  timeline_id TEXT NOT NULL REFERENCES timelines, skill_id INTEGER NOT NULL REFERENCES skills,
  PRIMARY KEY(timeline_id, skill_id));

CREATE TABLE IF NOT EXISTS turns(
  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  owner_timeline TEXT NOT NULL REFERENCES timelines,
  role TEXT NOT NULL CHECK(role IN ('user','assistant')),
  kind TEXT NOT NULL CHECK(kind IN ('text','tool_use','tool_result','thinking')),
  tool_name TEXT, content TEXT NOT NULL, tokens INTEGER, ts REAL NOT NULL,
  -- 1: written by the harness (merge results, a branch's fork result, an implicit merge call and
  -- its result, a skipped join's note). Set only by the harness; a harness row is recognised by
  -- this mark, never by what its text says.
  harness INTEGER NOT NULL DEFAULT 0 CHECK(harness IN (0, 1)));

CREATE TABLE IF NOT EXISTS context(
  timeline_id TEXT NOT NULL REFERENCES timelines, pos INTEGER NOT NULL,
  row_id INTEGER NOT NULL REFERENCES turns,
  PRIMARY KEY(timeline_id, pos), UNIQUE(timeline_id, row_id));

CREATE TABLE IF NOT EXISTS transactions(
  tx_id INTEGER PRIMARY KEY AUTOINCREMENT, timeline_id TEXT NOT NULL REFERENCES timelines,
  kind TEXT NOT NULL CHECK(kind IN ('branch','join')),
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

-- User messages whose callers withdrew them after the row was written (spec: State machine), per
-- timeline that holds the conversation they were asked in. The row itself is never edited: in
-- that timeline's view the mark shows it as a withdrawn message, never a request, trigger or seed
-- candidate, and a marked row is never seeded. Another timeline points at the row only if it was
-- seeded before the withdrawal (a background graft made for that request, which stops); it sees
-- the row as it was.
CREATE TABLE IF NOT EXISTS withdrawals(
  timeline_id TEXT NOT NULL REFERENCES timelines, row_id INTEGER NOT NULL REFERENCES turns,
  ts REAL NOT NULL, PRIMARY KEY(timeline_id, row_id));

-- The host's event log (spec: Host Interface, Delivery): one A2A StreamResponse per row, at the
-- position the session gave it; append-only. `task_id` is the task it is about (NULL: context-level).
CREATE TABLE IF NOT EXISTS a2a_events(
  seq INTEGER PRIMARY KEY, task_id TEXT, body TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS a2a_events_no_update BEFORE UPDATE ON a2a_events
BEGIN SELECT RAISE(ABORT, 'a2a_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS a2a_events_no_delete BEFORE DELETE ON a2a_events
BEGIN SELECT RAISE(ABORT, 'a2a_events is append-only'); END;

CREATE TRIGGER IF NOT EXISTS turns_no_update BEFORE UPDATE ON turns
  BEGIN SELECT RAISE(ABORT, 'turns is append-only'); END;
CREATE TRIGGER IF NOT EXISTS turns_no_delete BEFORE DELETE ON turns
  BEGIN SELECT RAISE(ABORT, 'turns is append-only'); END;
CREATE TRIGGER IF NOT EXISTS context_no_update BEFORE UPDATE ON context
  BEGIN SELECT RAISE(ABORT, 'context is append-only'); END;
CREATE TRIGGER IF NOT EXISTS context_no_delete BEFORE DELETE ON context
  BEGIN SELECT RAISE(ABORT, 'context is append-only'); END;
"""

TURN_COLS = "row_id, owner_timeline, role, kind, tool_name, content, tokens, ts, harness"
# The withdrawal mark, read with every row of a timeline's view (1: withdrawn in that timeline).
WITHDRAWN_COL = ("EXISTS(SELECT 1 FROM withdrawals w WHERE w.timeline_id = c.timeline_id "
                 "AND w.row_id = t.row_id) AS withdrawn")
# How a withdrawn message is shown to the model: marked, so it is never read as a live request.
WITHDRAWN_NOTE = "(harness note: the user withdrew the following message; do not answer it or act on it) "


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    tools: list  # graft's own tool definitions {"name", "description", "inputSchema"}, or host tool names


@dataclass
class Graft:
    """One entry of a committed branch call: the new timeline, its seed pointers and skills.
    A focus graft takes the conversation; the others run in the background."""
    child: str
    seed: list[int]
    skills: list[str]
    focus: bool = True
    name: str | None = None  # the id the model knows it by
    # The source's branch turn as the graft sees it: pointers to the assistant rows that hold the
    # branch call and to the source's results of that turn's other calls, then the graft's own
    # fork result (a tool_result block, written as a row the graft owns). Empty and None for a call
    # that is not a row in the source's view (a turn its caller drives).
    turn: list[int] = field(default_factory=list)
    result: dict | None = None


class Store:
    def __init__(self, path: str):
        self.path = path
        self.con = sqlite3.connect(path, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA foreign_keys=ON")
        columns = [r[1] for r in self.con.execute("PRAGMA table_info(turns)")]
        if columns and "harness" not in columns:
            raise RuntimeError(f"{path} was written by an older graft: its turns carry no harness mark, so "
                               "harness notes cannot be told from user messages. Re-run to produce a new store.")
        self.con.executescript(SCHEMA)

    # ---------- skills ----------
    def add_skill(self, skill: Skill) -> int:
        cur = self.con.execute(
            "INSERT INTO skills(name, description, instructions) VALUES (?,?,?)",
            (skill.name, skill.description, skill.instructions),
        )
        sid = cur.lastrowid
        for t in skill.tools:
            schema = dict(t["inputSchema"])
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
            out.append({"name": r["name"], "description": desc, "inputSchema": schema})
        return out

    # ---------- timelines ----------
    def create_timeline(self, tl: str, source: str | None, skills: Iterable[str],
                        status: str = "active", name: str | None = None) -> None:
        self.con.execute(
            "INSERT INTO timelines(timeline_id, source_id, status, created_at, name) VALUES (?,?,?,?,?)",
            (tl, source, status, time.time(), name),
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

    def name_of(self, tl: str) -> str | None:
        row = self.con.execute("SELECT name FROM timelines WHERE timeline_id=?", (tl,)).fetchone()
        return row[0] if row else None

    def timeline_named(self, name: str) -> str | None:
        row = self.con.execute("SELECT timeline_id FROM timelines WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def names(self) -> set[str]:
        """Every graft name ever given in this store: an id is never reused."""
        return {r[0] for r in self.con.execute("SELECT name FROM timelines WHERE name IS NOT NULL")}

    def set_status(self, tl: str, status: str) -> None:
        self.con.execute("UPDATE timelines SET status=? WHERE timeline_id=?", (status, tl))

    def set_join(self, tl: str, reason: str, trigger: int | None) -> None:
        self.con.execute(
            "UPDATE timelines SET join_reason=?, join_trigger=? WHERE timeline_id=?",
            (reason, trigger, tl))

    def next_branch_ids(self, n: int) -> list[str]:
        """Distinct ids for the n grafts of one branch call (none is committed yet)."""
        first = self.con.execute("SELECT count(*) FROM timelines").fetchone()[0]
        return [f"c{first + i}" for i in range(n)]

    def root(self) -> str:
        return self.con.execute(
            "SELECT timeline_id FROM timelines WHERE source_id IS NULL ORDER BY created_at"
        ).fetchone()[0]

    # ---------- turns & context ----------
    def insert_turn(self, owner: str, role: str, kind: str, content: Any,
                    tool_name: str | None = None, harness: bool = False) -> int:
        text = content if isinstance(content, str) else json.dumps(content)
        return self.con.execute(
            "INSERT INTO turns(owner_timeline, role, kind, tool_name, content, tokens, ts, harness) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (owner, role, kind, tool_name, text, max(1, len(text) // 4), time.time(), int(harness)),
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
                 tool_name: str | None = None, harness: bool = False) -> int:
        """`harness=True` marks a row the harness wrote (a note); only the harness passes it."""
        rid = self.insert_turn(tl, role, kind, content, tool_name, harness=harness)
        self.append(tl, rid)
        return rid

    def turn(self, row_id: int) -> dict | None:
        r = self.con.execute(f"SELECT {TURN_COLS} FROM turns WHERE row_id=?", (row_id,)).fetchone()
        return dict(r) if r else None

    def view(self, tl: str) -> list[dict]:
        """The context view: SELECT t.* FROM context JOIN turns ... ORDER BY pos."""
        return [dict(r) for r in self.con.execute(
            f"SELECT c.pos, t.{TURN_COLS.replace(', ', ', t.')}, {WITHDRAWN_COL} FROM context c "
            "JOIN turns t USING(row_id) WHERE c.timeline_id=? ORDER BY c.pos", (tl,))]

    # ---------- the host's event log ----------
    def append_events(self, events: Sequence[tuple[int, dict]]) -> int:
        """Append events at their positions, in one transaction: a position already written is
        left as it is (a batch written twice is written once). Returns the last position written."""
        self.con.execute("BEGIN")
        try:
            for seq, event in events:
                body = event.get("task") or event.get("statusUpdate") or {}
                task = body.get("id") if "task" in event else body.get("taskId")
                self.con.execute("INSERT OR IGNORE INTO a2a_events(seq, task_id, body) VALUES (?,?,?)",
                                 (seq, task, json.dumps(event)))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return self.event_position()

    def event_position(self) -> int:
        """The last position written (0: none)."""
        return self.con.execute("SELECT COALESCE(MAX(seq), 0) FROM a2a_events").fetchone()[0]

    def events_after(self, cursor: int, task: str | None = None, limit: int = 1000) -> list[tuple[int, dict]]:
        """The events after `cursor`, in order (of one task, if given)."""
        if task is None:
            rows = self.con.execute("SELECT seq, body FROM a2a_events WHERE seq > ? ORDER BY seq LIMIT ?",
                                    (cursor, limit))
        else:
            rows = self.con.execute("SELECT seq, body FROM a2a_events WHERE seq > ? AND task_id = ? ORDER BY seq "
                                    "LIMIT ?", (cursor, task, limit))
        return [(r[0], json.loads(r[1])) for r in rows]

    def task_events(self, task: str, upto: int | None = None) -> list[tuple[int, dict]]:
        """Every event of one task, in order (up to a position, if given)."""
        rows = self.con.execute("SELECT seq, body FROM a2a_events WHERE task_id = ? AND seq <= ? ORDER BY seq",
                                (task, upto if upto is not None else 2 ** 62))
        return [(r[0], json.loads(r[1])) for r in rows]

    def task_index(self) -> list[tuple[str, int, str]]:
        """Every task in the log, most recently updated first: (task id, its last position, its
        state there)."""
        rows = self.con.execute(
            "SELECT e.task_id, e.seq, e.body FROM a2a_events e JOIN (SELECT task_id, MAX(seq) AS last FROM a2a_events "
            "WHERE task_id IS NOT NULL GROUP BY task_id) m ON e.task_id = m.task_id AND e.seq = m.last "
            "ORDER BY e.seq DESC")
        out = []
        for task, seq, body in rows:
            event = json.loads(body)
            out.append((task, seq, (event.get("task") or event.get("statusUpdate"))["status"]["state"]))
        return out

    def context_id(self) -> str | None:
        """The conversation this store's log is of (None: no event yet)."""
        row = self.con.execute("SELECT body FROM a2a_events ORDER BY seq LIMIT 1").fetchone()
        if row is None:
            return None
        event = json.loads(row[0])
        body = event.get("task") or event.get("statusUpdate") or event.get("message")
        return body["contextId"]

    def task_ids(self) -> list[str]:
        """Every task in the log, in the order they were made."""
        return [r[0] for r in self.con.execute(
            "SELECT task_id FROM a2a_events WHERE task_id IS NOT NULL GROUP BY task_id ORDER BY MIN(seq)")]

    def mark_withdrawn(self, timelines: list[str], rows: list[int]) -> None:
        """Mark user messages whose callers withdrew them, in the views of `timelines` (append-only,
        one transaction; marking twice is a no-op)."""
        self.con.execute("BEGIN")
        try:
            self.con.executemany("INSERT OR IGNORE INTO withdrawals VALUES (?,?,?)",
                                 [(tl, r, time.time()) for tl in timelines for r in rows])
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

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

    # ---------- transactions ----------
    def begin_tx(self, tl: str, kind: str, target: str | None = None) -> int:
        return self.con.execute(
            "INSERT INTO transactions(timeline_id, kind, status, target) VALUES (?,?,?,?)",
            (tl, kind, "open", target)).lastrowid

    def open_txs(self, tl: str, kind: str) -> list[int]:
        """The transactions of tl of this kind that are still open, oldest first."""
        return [r[0] for r in self.con.execute(
            "SELECT tx_id FROM transactions WHERE timeline_id=? AND kind=? AND status='open' ORDER BY tx_id",
            (tl, kind))]

    def tx_status(self, tx_id: int) -> str:
        return self.con.execute("SELECT status FROM transactions WHERE tx_id=?", (tx_id,)).fetchone()[0]

    def join_outcome(self, child: str) -> str | None:
        """'committed' or 'rolled_back' if child's join has been written to its source (the
        outcome of its audit record), else None. A join attempt that failed and wrote nothing is
        recorded as 'failed' and does not count."""
        row = self.con.execute(
            "SELECT status FROM transactions WHERE kind='join' AND target=? AND status IN "
            "('committed','rolled_back') ORDER BY tx_id DESC LIMIT 1", (child,)).fetchone()
        return row[0] if row else None

    def end_tx(self, tx_id: int, status: str, committed_sql: str | None, scratch: Any) -> None:
        self.con.execute(
            "UPDATE transactions SET status=?, committed_sql=?, scratch=? WHERE tx_id=?",
            (status, committed_sql, json.dumps(scratch, default=str), tx_id))

    def commit_branch(self, source: str, grafts: Sequence[Graft],
                     audits: Sequence[tuple[int, str, Any]] = ()) -> None:
        """One SQL transaction for the whole branch call: every graft's timelines row, seed and
        branch-turn pointers, fork result row and timeline_skills links, and the `committed` outcome of its audit records
        (`audits`: (tx_id, committed_sql, scratch)), or none of it."""
        self.con.execute("BEGIN")
        try:
            for g in grafts:
                self.create_timeline(g.child, source, g.skills,
                                     status="active" if g.focus else "background", name=g.name)
                for rid in [*g.seed, *g.turn]:
                    self.append(tl=g.child, row_id=rid)
                if g.result is not None:
                    self.add_turn(tl=g.child, role="user", kind="tool_result", content=g.result, tool_name="fork",
                                  harness=True)
            for tx, sql, scratch in audits:
                self.end_tx(tx, status="committed", committed_sql=sql, scratch=scratch)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def commit_join(self, source: str, child: str, rows: list[int], *, merge: int | None,
                    implicit: dict | None, result: dict | None, note: str | None, trigger: int | None,
                    tx: int, status: str, committed_sql: str | None, scratch: Any,
                    closing: Sequence[tuple[str, str, Any]] = ()) -> int | None:
        """One join, one SQL transaction: the branch's close-out it could not write as it froze
        (`closing`: (timeline, tool name, tool_result block) rows, written first, into its own
        timeline), the harness's merge call and its result at the end of the branch's timeline
        (`implicit`: {"use", "result"}, for a turn that ended without one), the remembered
        pointers, a pointer to the merge call (`merge`, the branch's own, or the one just written),
        the source's merge result row (`result`, a tool_result block), or, for a join whose rows
        were skipped, a note instead of both (`note`), the trigger pointer, and the outcome of the
        join's audit record `tx`. All of it or none of it, so a join that failed halfway leaves
        nothing and can be retried. Returns the merge result's (or the note's) row id."""
        if (note is None) == (result is None) or (result is not None and merge is None and implicit is None):
            raise ValueError("a join appends a note, or a merge result with a merge call (its own or the harness's)")
        self.con.execute("BEGIN")
        try:
            for tl, tool_name, block in closing:
                self.add_turn(tl=tl, role="user", kind="tool_result", content=block, tool_name=tool_name)
            if implicit is not None:
                written = self.add_turn(tl=child, role="assistant", kind="tool_use", content=implicit["use"],
                                        tool_name="merge", harness=True)
                self.add_turn(tl=child, role="user", kind="tool_result", content=implicit["result"], tool_name="merge",
                              harness=True)
                merge = written if merge is None else merge
            if note is not None:
                rec = self.add_turn(tl=source, role="user", kind="text", content=note, harness=True)
            else:
                for rid in [*rows, merge]:
                    self.append(tl=source, row_id=rid)
                rec = self.add_turn(tl=source, role="user", kind="tool_result", content=result, tool_name="merge",
                                    harness=True)
            if trigger is not None:
                self.append(tl=source, row_id=trigger)
            self.end_tx(tx, status=status, committed_sql=committed_sql, scratch=scratch)
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
def row_block(r: dict, tag_ids: bool = False) -> dict:
    kind = r["kind"]
    if kind == "text":
        # The mark travels with the block so a caller can tell a harness note from a user message
        # without reading its text. Provider adapters build their wire blocks from type and text.
        if r.get("withdrawn"):  # never a live request: shown as withdrawn, marked like a note
            return {"type": "text", "text": WITHDRAWN_NOTE + r["content"], "harness": True}
        if r.get("harness"):
            return {"type": "text", "text": r["content"], "harness": True}
        text = f"[#{r['row_id']}] {r['content']}" if tag_ids and r.get("row_id") is not None else r["content"]
        return {"type": "text", "text": text}
    data = json.loads(r["content"])
    if kind == "tool_use":
        return {"type": "tool_use", "id": data["id"], "name": r["tool_name"], "input": data["input"]}
    if kind == "tool_result":
        return {"type": "tool_result", "tool_use_id": data["tool_use_id"],
                "content": data["content"], "is_error": bool(data.get("is_error"))}
    return {"type": "thinking", **data}


def render(rows: list[dict], tag_ids: bool = False) -> list[dict]:
    """Neutral messages [{role, content:[blocks]}]: consecutive same-role rows merge.
    `tag_ids=True` prefixes each non-harness text row with its immutable `[#<row_id>]` pointer
    (used on main so `fork` can pass integer row ids on the first try without breaking prefix cache)."""
    msgs: list[dict] = []
    for r in rows:
        block = row_block(r, tag_ids=tag_ids)
        if msgs and msgs[-1]["role"] == r["role"]:
            msgs[-1]["content"].append(block)
        else:
            msgs.append({"role": r["role"], "content": [block]})
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
    # A view may start with either role: a graft pointed at no messages starts on its source's
    # branch turn (spec: Validated views).
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
