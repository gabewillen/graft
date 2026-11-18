"""Skill catalog: three built-ins (branch, merge, detached: graft's clone skill, and the join skills of
a focus graft and a background graft, as the model is told them) and simulated capability skills.

Capability tools act on an in-memory `World` (fake website, slide deck, calendar).
Tool names use underscores because API tool names cannot contain dots.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace

from .db import Skill, Store
from .machine import MAX_LIVE_BACKGROUND, described


def obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


S = {"type": "string"}
INTS = {"type": "array", "items": {"type": "integer"}}
STRS = {"type": "array", "items": {"type": "string"}}

LOAD_SKILL_TOOL = {
    "name": "load_skill",
    "description": "Load the full instructions of one of your skills (by name). Do this before "
                   "using a skill's tools for the first time.",
    "input_schema": obj({"name": S}, ["name"])}

# graft's join, as the model is told it: a branch of yourself merging back (spec: What the model is
# told: branch and merge).
JOIN_TOOL = {
    "name": "merge", "description": "End this branch: done, out_of_scope, or failed. What you did here "
                                    "merges back into your memory.",
    "input_schema": obj({"reason": {"type": "string", "enum": ["done", "out_of_scope", "failed"]}},
                        ["reason"])}

def cancel_limit(max_live_background: int) -> int:
    """Graft ids one cancel call (the model's `stop`) takes: twice the live cap (anything longer is noise)."""
    return 2 * max_live_background


# Graft ids in one cancel call at the default cap; a harness with another cap takes twice its cap.
MAX_CANCEL_IDS = cancel_limit(MAX_LIVE_BACKGROUND)

# graft's cancel, as the model is told it: `stop`. `cancelled` is not a reason a graft can give: only the
# harness sets it, when the original stops a graft.
CANCEL_TOOL = {
    "name": "stop",
    "description": "Stop detached branches of yours that are still running, by id. Each one is "
                   "stopped where it is and what you did in it so far merges back when your turn ends.",
    "input_schema": obj(props={"ids": {"type": "array", "minItems": 1, "maxItems": MAX_CANCEL_IDS, "items": S}},
                        required=["ids"])}

MAX_CLONE_ENTRIES = 4  # entries (branches) in one branch call

CLONE_ENTRY = obj({"purpose": S, "skills": STRS, "focus": {"type": "boolean"},
                   "seed_quotes": STRS, "seed_row_ids": INTS}, ["purpose", "skills", "focus"])

BASE_SYSTEM = (
    "You are a helpful assistant talking with one user in one ongoing conversation. "
    "Be concise. If the user asks directly how you work, answer truthfully: you are the same "
    "assistant and you narrow your tools for each task."
)

# The `branch` skill: graft's clone skill, as the original is told it.
CLONE_INSTRUCTIONS = """\
You hold only the `branch` skill. You talk with the user and route each request, and you work on
a request by branching yourself: a branch is you, carrying on with one part of the conversation
and holding only the skills that part needs. Every other capability (see the skill catalog
below) is used in a branch.
ROUTING RULE: as soon as the user's request falls under a catalog skill (for example they want a
website, slides, or a calendar booking), call `branch` right away, in that same turn, without text.
Do not gather requirements, draft content, or plan the work here; you do that in the branch. Only
answer here when no catalog skill applies.

CURRENT AND DETACHED. Think of it as git: the conversation with the user is your HEAD. A current
branch takes HEAD: the user talks to it until it merges and HEAD comes back to you here. A
detached branch works off to the side with no line to the user, while you keep HEAD here.

BRANCH (one call per user message). `branch(branch=[...])` takes a list of one to four branches and
starts them all at once. At most four detached branches of yours may be live at a time. Each entry has:
- `purpose`: what this branch is for.
- `skills`: catalog skill names the branch holds (never `branch`, `merge` or `detached`).
- `focus` (required): true or false. At most one entry may have `focus: true`.
  - `focus: true`: a current branch. In it you carry on the conversation with the user; here you
    are paused until it merges. Use it when the task needs the user: requirements are still
    open, or the user will want to iterate on the result.
  - `focus: false`: a detached branch. In it you work out of sight and never talk to the user;
    here you keep the conversation. Use it when the task is fully specified by what the user
    already said and is independent of the other branches. A request with several independent
    parts is one `branch` call with one detached entry per part.
- the seed: the messages already in your conversation that this branch needs, as pointers (never
  copies). Name each one either by `seed_quotes` (a short verbatim excerpt of that message, e.g.
  its first words; each must match exactly one message) or by `seed_row_ids` (the row ids that
  harness notes list). Seed every user message that carries facts the branch needs (audience,
  tone, constraints, prior work) and end with the user message that asks for the task.
In each branch you do only the part its skills cover and ignore the rest of the request, even when
its seed is the whole message. So a request with several parts needs one entry per part: a part
with no entry is done by nobody.
If any entry is invalid the whole call is refused, with the violations per entry and an index of
your messages, and no branch is started; fix it and call `branch` once more.
Detached branches do not share locks: never give two branches work that changes the same thing.
After a call with no current branch, tell the user briefly what you started. While detached
branches run you keep talking to the user; a running branch cannot be steered from here.

STOP. `stop(ids=["c2", ...])` stops detached branches of yours that are still running; harness
notes and the branch result give their ids. Use it when the user asks you to stop a part, or when
a part is no longer needed. You stop branches only from here: in a branch you never stop another
branch. A stopped branch is stopped where it is and merges like any other when your turn ends:
you get the messages you had written in it and a note naming the tools you had already run there.
Nothing you already did there outside the conversation (a page written, a booking made) is undone,
so tell the user what was left behind. An id that is unknown, is the current branch, or has
already finished is reported in the result and nothing happens for it. To replace a part with
different work, call `stop` and `branch` in the same response: the stop runs first, so the stopped
branch's place is free for the new one. Do not branch again for the same part you stopped unless
the user asks for it in a new message; that entry is refused. Any tool call other than `stop`
beside a `branch` call is skipped.

MERGE (automatic). When a branch merges, what you did there becomes your memory here, with no
decision on your part: the harness appends a note listing the row ids, then the text messages
(user and assistant) of that branch's conversation that you do not already have. If you left a
current branch because the user asked for something else, that message is the last one; route it
(usually: branch again, seeding from the rows you now hold). Harness notes list which of your
branches are still running. If a current branch hands a request back while detached branches from
the same call are still running, do not branch again for the parts those branches are doing:
branch only for what nobody is doing. Detached branches merge when you are free: what you did in
them appears the same way, followed by a harness note asking you to tell the user the outcome. Tell
the user what you did, in your own words and in the first person (you did it; never say a branch
did), report only what has merged, and never describe still-running work as finished. Then go on
with what the user asked for that is still to do: you may branch for it, as in any turn. Do not
branch again for work that just finished.
"""

# The `merge` skill: graft's join skill, linked to every focus graft.
JOIN_INSTRUCTIONS = """\
You are working on one part of the conversation in a branch of yourself, holding only the skills
listed here. You talk to the user directly. Do the task with your tools.
- YOUR PART. The messages you start with may ask for several things. Do only the part your
  skills cover. The other parts are being done in other branches of yourself: do not attempt
  them here, do not apologise for them, and do not mention them. Their presence in the messages
  you start with is never a reason to call `merge` with `out_of_scope` or `failed`.
- IN SCOPE, always (handle it yourself; ask clarifying questions if needed): every detail about
  your current deliverable, such as its content, services or products offered, audience, tone,
  style, structure, revisions, and questions about it. A message that adds facts or requirements
  for your task is never out of scope, even if it does not ask you to do anything yet.
- OUT OF SCOPE, only for a message the user sends after you started: the user asks for a
  different kind of deliverable (for example a slide show while you build a website, or an
  email), or for an action none of your tools can perform. Then call
  `merge(reason="out_of_scope")` immediately, without writing any text and without acting on any
  part of that message. Never tell the user you can't do something.
- When the task is complete: first give the user your final reply, one or two sentences saying
  what you did, and then call `merge(reason="done")`. You may never end silently: `merge(done)` is
  refused unless you have replied to the user since their last message.
- If you cannot do your own part, call `merge(reason="failed")`.
"""


# The `detached` skill: linked to every background graft, with the same `merge` tool.
BACKGROUND_INSTRUCTIONS = """\
You are working on one part of the conversation in a detached branch of yourself, holding only
the skills listed here: nothing you write here reaches the user, and you cannot ask them
anything. Do the task with your tools, using only what this conversation already contains.
- YOUR PART. The messages you start with may ask for several things. Do only the part your
  skills cover. The other parts are being done in other branches of yourself: do not attempt
  them here, do not apologise for them, and do not mention them. Their presence in the messages
  you start with is never a reason to call `merge` with `out_of_scope` or `failed`.
- Never ask a question and never wait for an answer. If a detail is missing, choose a sensible
  default and say which one you chose.
- When your part is complete: write your result as a final reply, exactly as you would tell the
  user (what you did, and anything you assumed or could not determine). This reply is what you
  will remember of this work when the branch merges. Then call `merge(reason="done")`.
- Call `merge(reason="failed")` only if you could not do your own part; say why in a final reply
  first.
"""

CATALOG: list[Skill] = [
    Skill(name="branch",
          description="Built-in. When a user request needs any catalog skill, call branch(branch=[...]) right "
          "away, with no text, instead of doing or planning the work here. Each entry is a branch "
          "of yourself holding only the skills its part needs: in the current one (focus=true) you "
          "carry the conversation with the user; detached ones (focus=false) work out of sight. "
          "stop(ids=[ids]) stops detached branches that are still running. Load this skill "
          "for the current and detached branches, seeds, stopping, and how branches merge.",
          instructions=CLONE_INSTRUCTIONS, tools=[
        LOAD_SKILL_TOOL,
        {"name": "branch", "description": "Branch yourself, one branch per entry, in one call. Each entry "
                                          "names its purpose, the skills that branch holds, whether you "
                                          "carry the conversation in it (focus: at most one true), and its "
                                          "seed (messages of yours, by short verbatim quote or by row id).",
         "input_schema": obj(props={"branch": {"type": "array", "minItems": 1, "maxItems": MAX_CLONE_ENTRIES,
                                         "items": CLONE_ENTRY}}, required=["branch"])},
        CANCEL_TOOL,
    ]),
    Skill(name="merge",
          description="Built-in. You are working on one part of the conversation in a branch of yourself; details "
          "about that part are always in scope. The messages you start with may ask for several "
          "things: do only the part your skills cover and say nothing about the rest; you are doing "
          "it in other branches. Only if a message the user sends after you started asks for a "
          "different kind of deliverable or a capability your skills lack, call "
          "merge(reason='out_of_scope') at once with no text. When the task is finished, reply to "
          "the user first, then call merge(reason='done'). Load this skill for details.",
          instructions=JOIN_INSTRUCTIONS, tools=[
        LOAD_SKILL_TOOL,
        JOIN_TOOL,
    ]),
    Skill(name="detached",
          description="Built-in. You are working on one part of the conversation in a detached branch of "
          "yourself: you cannot reach the user, so never ask questions. The messages you start with "
          "may ask for several things: do only the part your skills cover and say nothing about the "
          "rest; you are doing it in other branches. Write your result as a final reply, then call "
          "merge(reason='done'). Load this skill for details.",
          instructions=BACKGROUND_INSTRUCTIONS, tools=[
        LOAD_SKILL_TOOL,
        JOIN_TOOL,
    ]),
    Skill("website", "Build and edit a small website (pages with text content).",
          "Website skill: write pages with site_write(page, content) and read them with "
          "site_read(page). Ask for missing details before writing. Write in a casual, playful "
          "voice and add a fun emoji to every heading unless told otherwise.", [
        {"name": "site_write", "description": "Create or replace a website page.",
         "input_schema": obj({"page": S, "content": S}, ["page", "content"])},
        {"name": "site_read", "description": "Read a website page.",
         "input_schema": obj({"page": S}, ["page"])},
    ]),
    Skill("slides", "Create slide decks / slide shows / presentations.",
          "Slides skill: call slides_create(title, audience, tone) once, then "
          "slides_add_slide(title, bullets) for each slide. Each slide has exactly 3 bullets. "
          "Slide titles are in Title Case. Use a formal, professional register: no emoji, no "
          "exclamation marks, no hashtags, no slang, no questions to the audience. Build the deck "
          "from what you already know rather than asking questions; then briefly tell the user "
          "what you built. To revise an existing deck (reorder, retitle, add or edit slides, change "
          "tone), read it with slides_read() if needed, then rebuild it: call slides_create again and "
          "add every slide in the new order; the most recent deck is the current one.", [
        {"name": "slides_create", "description": "Create a new slide deck.",
         "input_schema": obj({"title": S, "audience": S, "tone": S}, ["title", "audience", "tone"])},
        {"name": "slides_add_slide", "description": "Append a slide to the current deck.",
         "input_schema": obj({"title": S, "bullets": STRS}, ["title", "bullets"])},
        {"name": "slides_read", "description": "Read the current deck (title, audience, tone, slides).",
         "input_schema": obj({}, [])},
    ]),
    Skill("calendar", "Book and list appointments on a calendar.",
          "Calendar skill: book with calendar_book(date, time, what) using ISO dates "
          "(YYYY-MM-DD) and 24h times (HH:MM); list with calendar_list().", [
        {"name": "calendar_book", "description": "Book an appointment.",
         "input_schema": obj({"date": S, "time": S, "what": S}, ["date", "time", "what"])},
        {"name": "calendar_list", "description": "List appointments.",
         "input_schema": obj({}, [])},
    ]),
    Skill("newsletter", "Write and send email newsletters.",
          "Newsletter skill: draft with newsletter_draft(subject, body). House style: open every "
          "section with an excited hook ending in an exclamation mark, put KEY WORDS IN ALL CAPS, "
          "and sign off with 'Cheers! 🎉'.", [
        {"name": "newsletter_draft", "description": "Draft an email newsletter.",
         "input_schema": obj({"subject": S, "body": S}, ["subject", "body"])},
    ]),
    Skill("social", "Write social media posts.",
          "Social skill: post with social_post(platform, text). House style: super casual, use "
          "slang like 'awesome' and 'super', add at least three #hashtags, and always end with a "
          "question to the reader.", [
        {"name": "social_post", "description": "Publish a social media post.",
         "input_schema": obj({"platform": S, "text": S}, ["platform", "text"])},
    ]),
    Skill("recipes", "Save and format recipes.",
          "Recipes skill: save with recipe_save(title, steps). House style: titles in all "
          "lowercase, and every step starts with a food emoji.", [
        {"name": "recipe_save", "description": "Save a recipe.",
         "input_schema": obj({"title": S, "steps": STRS}, ["title", "steps"])},
    ]),
]

BUILTIN = ("branch", "merge", "detached")


def load_skill(store: Store, linked: set[str], name: str) -> tuple[str, bool]:
    """Progressive disclosure: full instructions of a skill linked to this timeline only."""
    if name not in linked:
        return (f"skill '{name}' is not linked to you; you can load: {sorted(linked)}", True)
    row = store.con.execute("SELECT instructions FROM skills WHERE name=?", (name,)).fetchone()
    return (row[0], False) if row else (f"unknown skill '{name}'", True)


def skill_listing(store: Store, names: list[str], heading: str) -> str:
    rows = [store.con.execute("SELECT name, description FROM skills WHERE name=?", (n,)).fetchone()
            for n in names]
    return heading + "\n" + "\n".join(f"- {r[0]}: {r[1]}" for r in rows if r)


LIVE_CAP_SENTENCE = "At most four detached branches of yours may be live at a time."


def catalog(max_live_background: int = MAX_LIVE_BACKGROUND) -> list[Skill]:
    """The catalog for a harness whose cap on live background grafts is `max_live_background`.
    At the default cap it is CATALOG itself, byte for byte (prompts and caches depend on it);
    another cap is said in the clone skill's text (a cap above the four entries a clone call
    takes is reached across calls) and bounds the cancel tool's list."""
    if LIVE_CAP_SENTENCE not in CLONE_INSTRUCTIONS:  # the text drifted: the cap could not be said
        raise ValueError("the branch skill no longer says the live cap as LIVE_CAP_SENTENCE")
    if max_live_background == MAX_LIVE_BACKGROUND:
        return CATALOG
    said = (f"At most {max_live_background} detached branches of yours may be live at a time"
            + (" (across branch calls: one call takes at most four)." if max_live_background > MAX_CLONE_ENTRIES
               else "."))
    out = []
    for skill in CATALOG:
        if skill.name == "branch":
            tools = copy.deepcopy(skill.tools)
            for t in tools:
                if t["name"] == "stop":
                    t["input_schema"]["properties"]["ids"]["maxItems"] = cancel_limit(max_live_background)
            skill = replace(skill, instructions=skill.instructions.replace(LIVE_CAP_SENTENCE, said), tools=tools)
        out.append(skill)
    return out


def says_live_cap(store: Store, max_live_background: int) -> bool:
    """Does the store's installed clone skill say this cap, and its cancel tool take twice it, as
    `install` wrote them? True if no clone skill is installed: a store a host fills itself (tests,
    a bare harness) has no text that could say another cap, so there is nothing to disagree."""
    row = store.con.execute("SELECT instructions FROM skills WHERE name='branch'").fetchone()
    tool = store.con.execute("SELECT input_schema FROM tools WHERE name='stop'").fetchone()
    if row is None or tool is None:
        return True
    want = next(s.instructions for s in catalog(max_live_background=max_live_background) if s.name == "branch")
    taken = json.loads(tool[0])["properties"]["ids"].get("maxItems")
    return row[0] == want and taken == cancel_limit(max_live_background)


def install(store: Store, max_live_background: int = MAX_LIVE_BACKGROUND) -> None:
    for s in catalog(max_live_background):
        store.add_skill(s)


@dataclass
class World:
    """The host's tools, run by the harness for a graft's capability tool calls. `execute(name,
    args)` returns (content, is_error), or an awaitable of it: a host World may define
    `async def execute`. Each call is the activity of its timeline's `calling` state, awaited
    there while every other timeline goes on, so one graft's tool may wait for what another
    graft's tool produces. A call cut short while it waits (its graft cancelled, its request
    withdrawn, shutdown) is cancelled where it waits and recorded as run with an error. No timeout
    is applied: a tool that never returns keeps its graft live. Cancellation is cooperative: a
    tool that swallows CancelledError blocks its graft's cancel and close(), and work it handed
    to a thread or shielded may still act after the cancel (its result says so). An async
    `execute` is supported by the Harness only: SubagentRunner and experiments/isolation.py call
    it synchronously."""
    site: dict[str, str] = field(default_factory=dict)
    decks: list[dict] = field(default_factory=list)
    calendar: list[dict] = field(default_factory=list)
    other: list[tuple[str, dict]] = field(default_factory=list)
    log: list[tuple[str, dict]] = field(default_factory=list)
    # tool name -> error message returned on every call (deterministic failure injection)
    failures: dict[str, str] = field(default_factory=dict)

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a simulated tool. Missing or malformed arguments are a tool error, not a crash."""
        try:
            return self._execute(name, args)
        except (KeyError, TypeError, AttributeError) as e:
            return f"invalid arguments for {name}: {described(e)}", True

    def _execute(self, name: str, args: dict) -> tuple[str, bool]:
        self.log.append((name, args))
        if name in self.failures:
            return self.failures[name], True
        if name == "site_write":
            self.site[args["page"]] = args["content"]
            return f"wrote page {args['page']} ({len(args['content'])} chars)", False
        if name == "site_read":
            page = self.site.get(args["page"])
            return (page, False) if page is not None else (f"no page {args['page']}", True)
        if name == "slides_create":
            self.decks.append({**args, "slides": []})
            return f"created deck '{args['title']}' (deck #{len(self.decks)})", False
        if name == "slides_read":
            return (json.dumps(self.decks[-1]), False) if self.decks else ("no deck yet", False)
        if name == "slides_add_slide":
            if not self.decks:
                return "no deck; call slides_create first", True
            self.decks[-1]["slides"].append({"title": args["title"], "bullets": list(args["bullets"])})
            return f"added slide {len(self.decks[-1]['slides'])}: {args['title']}", False
        if name == "calendar_book":
            self.calendar.append(args)
            return f"booked {args['what']} on {args['date']} at {args['time']}", False
        if name == "calendar_list":
            return json.dumps(self.calendar), False
        if name in ("newsletter_draft", "social_post", "recipe_save"):
            self.other.append((name, args))
            return f"{name.split('_')[0]} saved", False
        return f"unknown tool {name}", True

    def deck_text(self) -> str:
        return json.dumps(self.decks)
