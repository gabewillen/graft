"""Skill catalog: three built-ins (clone, join, background) and simulated capability skills.

Capability tools act on an in-memory `World` (fake website, slide deck, calendar).
Tool names use underscores because API tool names cannot contain dots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .db import Skill, Store


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

JOIN_TOOL = {
    "name": "join", "description": "End this task: done, out_of_scope, or failed.",
    "input_schema": obj({"reason": {"type": "string", "enum": ["done", "out_of_scope", "failed"]}},
                        ["reason"])}

MAX_CLONE_ENTRIES = 4  # entries (clones) in one clone call

CLONE_ENTRY = obj({"purpose": S, "skills": STRS, "focus": {"type": "boolean"},
                   "seed_quotes": STRS, "seed_row_ids": INTS}, ["purpose", "skills", "focus"])

BASE_SYSTEM = (
    "You are a helpful assistant talking with one user in one ongoing conversation. "
    "Be concise. If the user asks directly how you work, answer truthfully: you are the same "
    "assistant and you narrow your tools for each task."
)

CLONE_INSTRUCTIONS = """\
You hold only the `clone` skill. You converse, route, clone and absorb joins; every other
capability (see the skill catalog below) requires a clone.
ROUTING RULE: as soon as the user's request falls under a catalog skill (for example they want a
website, slides, or a calendar booking), call `clone` right away, in that same turn, without text.
Do not gather requirements, draft content, or plan the work yourself; a clone does that. Only
answer yourself when no catalog skill applies.

CLONE (one call per user message). `clone(clone=[...])` takes a list of one to four clones and
creates them all at once. At most four background clones of yours may be live at a time. Each entry has:
- `purpose`: what this clone is for.
- `skills`: catalog skill names to link (never `clone`, `join` or `background`).
- `focus` (required): true or false. At most one entry may have `focus: true`.
  - `focus: true`: this clone takes over the conversation and talks to the user directly; you
    are suspended until it joins. Use it when the task needs the user: requirements are still
    open, or the user will want to iterate on the result.
  - `focus: false`: this clone works in the background and never talks to the user; you keep
    the conversation. Use it when the task is fully specified by what the user already said and
    is independent of the other clones. A request with several independent parts is one `clone`
    call with one background entry per part.
- the seed: the messages already in your conversation that this clone needs, as pointers (never
  copies). Name each one either by `seed_quotes` (a short verbatim excerpt of that message, e.g.
  its first words; each must match exactly one message) or by `seed_row_ids` (the row ids that
  harness notes list). Seed every user message that carries facts the clone needs (audience,
  tone, constraints, prior work) and end with the user message that asks for the task.
Each clone does only the part its skills cover and ignores the rest of the request, even when
its seed is the whole message. So a request with several parts needs one entry per part: a part
with no entry is done by nobody.
If any entry is invalid the whole call is refused, with the violations per entry and an index of
your messages, and no clone is created; fix it and call `clone` once more.
Background clones do not share locks: never give two clones work that changes the same thing.
After a call with no focus clone, tell the user briefly what you started. While background
clones run you keep talking to the user; you cannot steer a running clone.

JOIN (automatic). When a clone joins, the harness remembers for you, with no decision on your
part: it appends a note listing the row ids, then the text messages (user and assistant) of the
clone's conversation that you do not already have. If a focus clone left because the user asked
for something else, that message is the last one; route it (usually: clone again, seeding from
the rows you now hold). Harness notes list which of your clones are still running. If a focus
clone hands a request back while background clones from the same call are still running, do not
clone again for the parts those clones are doing: clone only for what nobody is doing. Background clones join when you are free: their results appear the same
way, followed by a harness note asking you to tell the user the outcome. Do that in your own
words, report only what has joined, never describe still-running work as finished, and do not
clone again for work that just finished.
"""

JOIN_INSTRUCTIONS = """\
You are a focused instance of the assistant for one task, holding only the skills listed here.
You talk to the user directly. Do the task with your tools.
- YOUR PART. The messages you start with may ask for several things. Do only the part your
  skills cover. The other parts are handled by other instances of the assistant: do not attempt
  them, do not apologise for them, and do not mention them. Their presence in the messages you
  start with is never a reason to join `out_of_scope` or `failed`.
- IN SCOPE, always (handle it yourself; ask clarifying questions if needed): every detail about
  your current deliverable, such as its content, services or products offered, audience, tone,
  style, structure, revisions, and questions about it. A message that adds facts or requirements
  for your task is never out of scope, even if it does not ask you to do anything yet.
- OUT OF SCOPE, only for a message the user sends after you started: the user asks for a
  different kind of deliverable (for example a slide show while you build a website, or an
  email), or for an action none of your tools can perform. Then call
  `join(reason="out_of_scope")` immediately, without writing any text and without acting on any
  part of that message. Never tell the user you can't do something.
- When the task is complete: first give the user your final reply, one or two sentences saying
  what you did, and then call `join(reason="done")`. You may never end silently: `join(done)` is
  refused unless you have replied to the user since their last message.
- If you cannot do your own part, call `join(reason="failed")`.
"""


BACKGROUND_INSTRUCTIONS = """\
You are a focused instance of the assistant for one task, holding only the skills listed here,
and you run in the background: nothing you write reaches the user, and you cannot ask them
anything. Do the task with your tools, using only what your conversation already contains.
- YOUR PART. The messages you start with may ask for several things. Do only the part your
  skills cover. The other parts are handled by other instances of the assistant: do not attempt
  them, do not apologise for them, and do not mention them. Their presence in the messages you
  start with is never a reason to join `out_of_scope` or `failed`.
- Never ask a question and never wait for an answer. If a detail is missing, choose a sensible
  default and say which one you chose.
- When your part is complete: write your result as a final reply, exactly as you would tell the
  user (what you did, and anything you assumed or could not determine). This reply is what the
  assistant will remember of your work. Then call `join(reason="done")`.
- Call `join(reason="failed")` only if you could not do your own part; say why in a final reply
  first.
"""

CATALOG: list[Skill] = [
    Skill("clone",
          "Built-in. When a user request needs any catalog skill, call clone(clone=[...]) right "
          "away, with no text, instead of doing or planning the work yourself. Each entry is one "
          "clone: one with focus=true talks to the user; entries with focus=false work in the "
          "background. Load this skill for the focus and seed rules and the join protocol.",
          CLONE_INSTRUCTIONS, [
        LOAD_SKILL_TOOL,
        {"name": "clone", "description": "Create one or more clones in one call. Each entry names its "
                                         "purpose, the skills to link, whether it takes the conversation "
                                         "(focus: at most one true), and its seed (messages of yours, by "
                                         "short verbatim quote or by row id).",
         "input_schema": obj({"clone": {"type": "array", "minItems": 1, "maxItems": MAX_CLONE_ENTRIES,
                                        "items": CLONE_ENTRY}}, ["clone"])},
    ]),
    Skill("join",
          "Built-in. You are a focused instance for one task; details about that task are always in "
          "scope. The messages you start with may ask for several things: do only the part your "
          "skills cover and say nothing about the rest; other instances handle it. Only if a "
          "message the user sends after you started asks for a different kind of deliverable or a "
          "capability your skills lack, call join(reason='out_of_scope') at once with no text. When "
          "the task is finished, reply to the user first, then call join(reason='done'). Load this "
          "skill for details.",
          JOIN_INSTRUCTIONS, [
        LOAD_SKILL_TOOL,
        JOIN_TOOL,
    ]),
    Skill("background",
          "Built-in. You are a focused instance for one task, working in the background: you cannot "
          "reach the user, so never ask questions. The messages you start with may ask for several "
          "things: do only the part your skills cover and say nothing about the rest; other "
          "instances handle it. Write your result as a final reply, then call join(reason='done'). "
          "Load this skill for details.",
          BACKGROUND_INSTRUCTIONS, [
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

BUILTIN = ("clone", "join", "background")


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


def install(store: Store) -> None:
    for s in CATALOG:
        store.add_skill(s)


@dataclass
class World:
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
            return f"invalid arguments for {name}: {type(e).__name__}: {e}", True

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
