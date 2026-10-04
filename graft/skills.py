"""Skill catalog: two built-ins (clone, join) and simulated capability skills.

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
Do not gather requirements, draft content, or plan the work yourself; the clone does that and
talks to the user directly. Only answer yourself when no catalog skill applies.

CLONE (one call). `clone(purpose, skills, seed_quotes, seed_row_ids)` creates the clone at once:
- `skills`: catalog skill names to link (never `clone` or `join`).
- the seed: the messages already in your conversation that the clone needs, as pointers (never
  copies). Name each one either by `seed_quotes` (a short verbatim excerpt of that message, e.g.
  its first words; each must match exactly one message) or by `seed_row_ids` (the row ids that
  harness notes list). Seed every user message that carries facts the clone needs (audience,
  tone, constraints, prior work) and end with the user message that asks for the task.
If the seed is invalid the call is refused with the specific violations and an index of your
messages; fix it and call `clone` once more.
After the clone is created it talks to the user; you are suspended until it joins.

JOIN (automatic). When your clone joins, the harness remembers for you, with no decision on
your part: it appends a note listing the row ids, then the text messages (user and assistant) of
the clone's conversation that you do not already have. If the clone left because the user asked
for something else, that message is the last one; route it (usually: clone again, seeding from
the rows you now hold).
"""

JOIN_INSTRUCTIONS = """\
You are a focused instance of the assistant for one task, holding only the skills listed here.
You talk to the user directly. Do the task with your tools.
- IN SCOPE, always (handle it yourself; ask clarifying questions if needed): every detail about
  your current deliverable, such as its content, services or products offered, audience, tone,
  style, structure, revisions, and questions about it. A message that adds facts or requirements
  for your task is never out of scope, even if it does not ask you to do anything yet.
- OUT OF SCOPE, only: the user asks for a different kind of deliverable (for example a slide show
  while you build a website, or an email), or for an action none of your tools can perform. Then
  call `join(reason="out_of_scope")` immediately, without writing any text and without acting on
  any part of that message. Never tell the user you can't do something.
- When the task is complete: first give the user your final reply, one or two sentences saying
  what you did, and then call `join(reason="done")`. You may never end silently: `join(done)` is
  refused unless you have replied to the user since their last message.
- If you cannot proceed, call `join(reason="failed")`.
"""

CATALOG: list[Skill] = [
    Skill("clone",
          "Built-in. When a user request needs any catalog skill, call clone(purpose, skills, seed) "
          "right away, with no text, instead of doing or planning the work yourself; the clone talks "
          "to the user. Load this skill for the seed rules and the join protocol.",
          CLONE_INSTRUCTIONS, [
        LOAD_SKILL_TOOL,
        {"name": "clone", "description": "Create a clone in one call: the skills to link and the seed "
                                         "(messages of yours, by short verbatim quote or by row id).",
         "input_schema": obj({"purpose": S, "skills": STRS, "seed_quotes": STRS, "seed_row_ids": INTS},
                             ["purpose", "skills"])},
    ]),
    Skill("join",
          "Built-in. You are a focused instance for one task; details about that task are always in "
          "scope. Only if the user asks for a different kind of deliverable or a capability your "
          "skills lack, call join(reason='out_of_scope') at once with no text. When the task is "
          "finished, reply to the user first, then call join(reason='done'). Load this skill for details.",
          JOIN_INSTRUCTIONS, [
        LOAD_SKILL_TOOL,
        {"name": "join", "description": "End this task: done, out_of_scope, or failed.",
         "input_schema": obj({"reason": {"type": "string", "enum": ["done", "out_of_scope", "failed"]}},
                             ["reason"])},
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

BUILTIN = ("clone", "join")


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
