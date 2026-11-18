"""Skill catalog: three built-ins (fork, merge, detached: graft's clone skill, and the join skills of
a focus graft and a background graft, as the model is told them) and simulated capability skills.

Capability tools act on an in-memory `World` (fake website, slide deck, calendar).
Tool names use underscores because API tool names cannot contain dots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .db import Skill, Store
from .machine import described


def obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


S = {"type": "string"}
INTS = {"type": "array", "items": {"type": "integer"}}
STRS = {"type": "array", "items": {"type": "string"}}

REASON_CHARS = 2000  # the longest `reason` a fork entry or a merge call may give
MAX_MESSAGES = 50  # the most items a fork entry's `messages` may name

LOAD_SKILL_TOOL = {
    "name": "load_skill",
    "description": "Load the full instructions of one of your skills (by name). Do this before "
                   "using a skill's tools for the first time.",
    "input_schema": obj({"name": S}, ["name"])}

# graft's join, as the model is told it: a branch of yourself merging back (spec: What the model is
# told: fork, merge and stop).
JOIN_TOOL = {
    "name": "merge", "description": "End this turn and merge this branch back into main: what you did here "
                                    "becomes your memory there. The end of your turn merges it anyway.",
    "input_schema": obj({"reason": {"type": "string", "maxLength": REASON_CHARS, "description": "Optional: your own words to yourself "
                                                                      "on main, for example what is still to do."}},
                        [])}

# graft's cancel, as the model is told it: `stop`. `cancelled` is not a reason a graft can give: only the
# harness sets it, when the original stops a graft.
CANCEL_TOOL = {
    "name": "stop",
    "description": "Stop detached branches of yours that are still running, by id, only when the user "
                   "no longer wants their parts. Each one is stopped where it is and what you did in it so "
                   "far merges back when your turn ends. It does not end your turn: to end it, say nothing.",
    "input_schema": obj(props={"ids": {"type": "array", "minItems": 1, "items": S}},
                        required=["ids"])}

# What a branch id looks like: chosen by the model, or drawn by the harness (6 hex characters).
BRANCH_ID_PATTERN = "^[a-z0-9][a-z0-9-]{0,31}$"

# One entry of the fork call: the skills the branch holds, the messages it starts from (pointers:
# short exact quotes or row ids), optionally main's own words for it, and optionally its id.
MESSAGES = {"type": "array", "maxItems": MAX_MESSAGES, "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "description": "The messages of this conversation the branch starts from: each a short exact "
                           "quote of one message (or its row id). It may be empty if you give a reason."}
REASON = {"type": "string", "maxLength": REASON_CHARS, "description": "Your own words for this branch: the part it does, and anything "
                                           "it needs to know that the messages do not say."}
CLONE_ENTRY = obj({"id": {"type": "string", "pattern": BRANCH_ID_PATTERN}, "skills": STRS,
                   "messages": MESSAGES, "reason": REASON}, ["skills", "messages"])

BASE_SYSTEM = (
    "You are a helpful assistant talking with one user in one ongoing conversation. "
    "Be concise. If the user asks directly how you work, answer truthfully: you are the same "
    "assistant and you narrow your tools for each task."
)

# The `fork` skill: graft's clone skill, as the original is told it.
CLONE_INSTRUCTIONS = """\
You are on main, and you stay on main. You talk with the user, route each request, and work on it
by forking yourself: each branch you fork is you, carrying on with one part of the request and
holding only the skills that part needs. Every other capability (see the skill catalog below) is
used in a branch; on main you do no part of the work yourself.
ROUTING RULE: as soon as the user's request falls under a catalog skill (for example they want a
website, slides, or a calendar booking), call `fork` right away, in that same turn, without text.
Do not gather requirements, draft content, or plan the work here; you do that in the branch. Only
answer here when no catalog skill applies.

MAIN, CURRENT AND DETACHED. Think of it as git: the line you hold is main, and the conversation
with the user is HEAD. Forking never moves you off main: `current` and `detached` name new
branches, never main itself. A current branch takes HEAD for one turn: in it you talk with the
user (you answer, or ask what you need), and when that turn ends the branch merges and HEAD comes
back to you on main. A detached branch works off to the side with no line to the user, while HEAD
stays on main.

FORK. `fork(current={...}, detached=[{...}, ...])` starts every branch it names at once. Give
`current`, `detached`, or both. Each branch is `{id?, skills, messages, reason?}`:
- `skills`: catalog skill names the branch holds (never `fork`, `merge` or `detached`).
- `messages`: the messages of this conversation the branch starts from, each a short exact quote
  of one message (or its row id). The branch sees only these, then this fork call. You choose
  what each branch sees, as a git branch is cut from the commits you pick: point it at what its
  part needs. It may be empty when `reason` says everything.
- `reason`: your own words for that branch: which part of the request is its own, and anything it
  needs that the messages do not say. The branch reads it in this fork call. Give every branch a
  reason when the request has several parts, so each one knows its part.
- `id` (optional): a short name for the branch, lowercase letters, digits and `-` (for example
  "site"; never "main", the line you hold), unique in this conversation and never reused. Without
  one, the branch gets a random id. Results and `stop` use it. Two branches of one call with the
  same skills need ids.
- `current`: at most one branch, the one that talks with the user this turn. Use it when the part
  needs the user: details are still open, or the user will want to see the result and answer.
- `detached`: branches that work out of sight. Use them for parts fully specified by what the user
  already said and independent of the other branches.
A request with several parts needs one branch per part, in one `fork` call: a part with no branch
is done by nobody. If any branch is invalid the whole call is refused, with the violations per
branch and your messages with their row ids, and no branch is started; fix it and call `fork`
once more. Detached branches do not share locks: never give two branches work that changes the
same thing. After a call with no current branch, tell the user briefly what you started.

STOP. `stop(ids=["site", ...])` stops detached branches of yours that are still running; the fork
result and merge results give their ids. Use it only when the user asks you to stop a part, or
when the user's own words make a part no longer wanted. A stopped branch is stopped where it is and merges like any other when
your turn ends: you get the messages you had written in it and the tools you had already run
there. Nothing you already did there outside the conversation (a page written, a booking made) is
undone, so tell the user what was left behind. An id that is unknown, names a current branch, or
has already finished is reported in the result and nothing happens for it. To replace a part with
different work, call `stop` and `fork` in the same response: the stop runs first. Do not fork again
for a part you stopped unless the user asks for it in a new message. Any tool call other than `stop`
beside a `fork` call is skipped.

MERGE. Every branch merges when its turn ends, and you are back on main. What you did there
becomes your memory here, with no decision on your part: its messages you do not already have,
its `merge` call (with the reason it gave, if any), and the merge's result, which says which branch
merged and which branches of yours are still running. Then you run once:
- A current branch's replies already reached the user: never repeat them. If it asked the user
  something, end your turn without saying anything and wait: the answer comes to you on main, and
  then you fork again for it, pointing the new branch at the question and the answer.
- A detached branch goes on with its own part whatever another branch did: never stop it or fork
  its part again because a current branch merged or asked the user something. Ending your turn
  needs no tool: say nothing.
- What you did in a detached branch the user has not seen: tell the user, in your own words and in
  the first person (you did it; never say a branch did), and never describe still-running work as
  finished.
- Then go on with what the user asked for that is still to do (fork for it), unless a running
  branch is already doing it. If nothing is left to do or say, end your turn without saying anything.
"""

# The `merge` skill: graft's join skill, linked to every focus graft.
JOIN_INSTRUCTIONS = """\
You are on a branch of yourself for one turn of the conversation, holding only the skills listed
here, and you talk to the user directly. You forked this branch from main: your fork call above
says, in the reason you wrote for this branch, which part of the request is this branch's.
- YOUR PART. Do that part (without a reason: the part your skills cover). The messages you start
  with may ask for more: the other parts are yours on other branches or on main. Do not attempt
  them, apologise for them or mention them, and never hand them back.
- Do your part with your tools now. If you need details from the user, ask for them in your reply;
  their answer comes to main, which forks again for it.
- End your turn with a reply to the user: one or two sentences saying what you did, or your
  question. The end of your turn merges this branch back into main. You may call `merge` to end the
  turn at once, with an optional `reason`: your own words to yourself on main (for example what is
  still to do); the user never sees it.
- Only if your part needs a tool none of your skills has: call `merge` at once with no text, and
  say why in its reason.
"""


# The `detached` skill: linked to every background graft, with the same `merge` tool.
BACKGROUND_INSTRUCTIONS = """\
You are on a detached branch of yourself, holding only the skills listed here: nothing you write
here reaches the user, and you cannot ask them anything. You forked this branch from main: your
fork call above says, in the reason you wrote for this branch, which part of the request is this
branch's.
- YOUR PART. Do that part (without a reason: the part your skills cover). The messages you start
  with may ask for more: the other parts are yours on other branches or on main. Do not attempt
  them, apologise for them or mention them, and never hand them back.
- Never ask a question and never wait for an answer. If a detail is missing, choose a sensible
  default and say which one you chose.
- When your part is complete: write your result as a final reply, exactly as you would tell the
  user (what you did, and anything you assumed or could not determine). This reply is what you
  will remember of this work when the branch merges. The end of your turn merges the branch; you
  may also call `merge`, with an optional `reason` (your own words to yourself on main).
- If you could not do your part, say why in your final reply.
"""

CATALOG: list[Skill] = [
    Skill(name="fork",
          description="Built-in. You stay on main. When a user request needs any catalog skill, call "
          "fork(current={...}, detached=[...]) right away, with no text, instead of doing or planning "
          "the work here. Each is a new branch of yourself holding only the skills its part needs, "
          "starting from the messages you point it at and the reason you write for it: the current one "
          "talks with the user for one turn; detached ones work out of sight. Every branch merges back "
          "when its turn ends. stop(ids=[ids]) stops detached branches that are still running. Load this "
          "skill for main, current and detached branches, messages and reasons, stopping, and merging.",
          instructions=CLONE_INSTRUCTIONS, tools=[
        LOAD_SKILL_TOOL,
        {"name": "fork", "description": "Fork yourself in one call; you stay on main. `current`: a new "
                                          "branch that talks with the user for one turn; `detached`: new "
                                          "branches that work out of sight; give either or both. Each "
                                          "branch names the skills it holds, the messages it starts from, "
                                          "your reason (its part, in your words) and, optionally, its id.",
         "input_schema": obj(props={"current": CLONE_ENTRY,
                                    "detached": {"type": "array", "items": CLONE_ENTRY}}, required=[])},
        CANCEL_TOOL,
    ]),
    Skill(name="merge",
          description="Built-in. You are on a branch of yourself for one turn, talking with the user. "
          "Do the part the reason you wrote for this branch gives it (without one: the part your "
          "skills cover) and say nothing about the rest; you are doing it on other branches or on "
          "main. End your turn with a reply to the user (what you did, or your question): the end "
          "of your turn merges this branch back into main. Load this skill for details.",
          instructions=JOIN_INSTRUCTIONS, tools=[
        LOAD_SKILL_TOOL,
        JOIN_TOOL,
    ]),
    Skill(name="detached",
          description="Built-in. You are on a detached branch of yourself: you cannot reach the user, so "
          "never ask questions. Do the part the reason you wrote for this branch gives it (without "
          "one: the part your skills cover) and say nothing about the rest. Write your result as a "
          "final reply; the end of your turn merges this branch back into main. Load this skill for "
          "details.",
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

BUILTIN = ("fork", "merge", "detached")


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
