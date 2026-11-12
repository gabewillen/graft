# RFC: Agent Grafts

Status: draft. v1 (depth 1, with parallel background grafts) is implemented in this repository; decisions below are fixed for v1 unless listed in [Open Questions](#open-questions).

## Summary

Agent Grafts is an architecture for isolating specialized work without giving up the flexibility of a single general-purpose agent, and without the information loss of agent-to-agent communication.

> **An agent is a timeline. Cloning suspends it; joining resumes it with the clone's timeline available as its own memory. Context is isolated on the way down and lossless on the way up.**

The name comes from horticulture: a cutting is taken from a plant and later grafted back so both grow as one organism. Here an agent clones itself into a single-use timeline, and that timeline is grafted back onto its source as memory.

The core principle is:

> **Replicate the agent, not its context.**

The user talks to one agent. When specialized work is needed, the agent **clones** itself into a new timeline that holds only the context and skills it chose for that clone. The clone talks to the user directly. When the clone is done or out of scope, it **joins**: the source resumes with the conversation the user had with the clone appended to its own memory, by pointer, with no model call.

Several clones can exist at once, but **at most one branch holds the conversation**. One clone call creates one or more clones: at most one is the **focus** graft, which takes the conversation as described above, and the rest are **background** grafts, which work concurrently, never talk to the user, and join back when they are done while the source keeps the conversation (see [Parallel grafts](#parallel-grafts)).

The problem this solves is *lost in translation*: no worker ever summarizes its own work for someone else. What comes back is the record itself: the conversation the user actually had with the clone, joined by pointer.

## Terminology

| Term | Meaning |
|---|---|
| **original** | The user-facing agent at the top of the chain. It has no source. |
| **clone** | A single-use agent created by a source. It shares the source's identity and stable configuration and has its own timeline. |
| **source** | The agent that created a given clone (the original or another clone). |
| **timeline** | An agent's full, durable context: every turn, in order. An agent *is* its timeline. |
| **clone** (the call) | The source's single `clone` call. It takes a list of one or more entries; each names a purpose, the skills, the seed rows, and whether the clone takes the conversation (`focus`). The harness validates and commits the whole list or none of it. |
| **graft** | A clone seen as one entry of a clone call. A **focus graft** takes the conversation and its source is suspended until it joins. A **background graft** runs concurrently, never talks to the user, and its source keeps the conversation. |
| **join** | The clone's final act: it raises a join event and ceases to exist. The harness then applies the deterministic default join to the source. |

"Journal" and "daughter" from earlier drafts are retired. There is no separate log; the timeline is the record.

## Motivation

A single agent equipped with many skills can solve complex, cross-domain tasks. But as capabilities accumulate, unrelated instructions and histories share one reasoning context and influence each other. Individual skills can pass their evaluations while their composition misbehaves.

The usual fix, delegating to sub-agents, introduces a second problem: the sub-agent reports back in its own words. Every hop is a game of telephone.

Agent Grafts addresses both:

- **Isolation down:** a clone sees only the rows its source explicitly gave it, plus its linked skills.
- **No telephone up:** the clone never writes an account of its work. The source reads what actually happened and chooses what to keep.

## Model

### Everything is SQLite

All state lives in one SQLite database. A clone is a new timeline; cloning and joining are atomic, validated inserts of pointers.

```sql
skills(skill_id, name, description, instructions)
tools(tool_id, name, input_schema)
skill_tools(skill_id, tool_id)

timelines(timeline_id, source_id, status, created_at,
          join_reason, join_trigger)            -- set by the join event
timeline_skills(timeline_id, skill_id)

turns(row_id, owner_timeline, role,             -- role: user | assistant
      kind,                                     -- text | tool_use | tool_result | thinking
      tool_name, content, tokens, ts)           -- stored once, never edited

context(timeline_id, pos, row_id)               -- pointer table: an agent's context

transactions(tx_id, timeline_id, kind,          -- clone | join
             status, committed_sql)             -- audit record of every clone and join
```

### One primitive

Seeding a clone, joining it back, and compacting a timeline are the same operation: **a validated selection of pointers to existing rows**, plus new rows only where an agent writes them (replies, and compaction's compressed turns, which always record the window they condense). Stored rows are never edited, and every paraphrase links back to its originals.

### Context is a pointer table

Every turn is stored exactly once in `turns`, owned by the timeline that produced it. An agent's context is not a copy of rows; it is an **ordered, append-only list of pointers** in `context`, read through a view:

```sql
CREATE VIEW ctx_c7 AS
  SELECT t.* FROM context c JOIN turns t USING (row_id)
  WHERE c.timeline_id = 'c7' ORDER BY c.pos;
```

- **Cloning** inserts pointers for the new timeline to the rows the source selected.
- **Joining** appends pointers to the source's context, pointing at rows the clone owns.
- **New turns** are inserted into `turns` once and a pointer is appended.
- Nothing is ever copied, moved, edited or deleted. `turns` and `context` are both append-only.

Consequences:

- **No duplicates by construction.** `row_id` is the permanent identity; a join simply skips rows already pointed to.
- **Provenance is free.** `owner_timeline` records where every memory came from. The model sees joined turns natively, while the harness and evals can always trace their origin.
- **Rollback is always safe**, and earlier positions never change, so the prompt cache stays valid.
- **No table modifications** for clone or join beyond inserting pointers.

The harness is a small loop:

1. Read the active timeline's context view.
2. Resolve its tools through `timeline_skills → skill_tools`.
3. Call the model.
4. Insert the new turns and append their pointers.

### Validated views

The view is sent to the model exactly as stored. There is no hidden normalization at render time.

Instead, validity is a **commit condition** of every operation that changes a context (a clone's seed, a join). The harness validates the resulting view before committing:

- user and assistant turns alternate
- every `tool_use` has its `tool_result`, and vice versa
- the view ends in a state the model can continue from (for a join with a trigger, the trigger is the last user turn)
- thinking blocks are either valid for the model or excluded

If validation fails, the commit is refused with the specific violations. For a `clone` call, the source gets one retry to fix its selection. The deterministic default join is built to always validate; a failure there is a harness bug.

### Compaction

A timeline stores the full, uncompacted history. An agent's working window may be a compacted view over it, but the full record is never lost.

**Compaction keeps and rewrites, with provenance.** When a timeline's working view exceeds its budget, the model produces the new working view from two kinds of rows:

- **Kept rows:** pointers to original rows, preferred where rows are short or exact wording matters (names, numbers, quotes).
- **Compressed turns:** new user or assistant rows that condense several turns. Compaction is paraphrase by nature, for both roles. Each compressed row records the **window** it condenses, as pointers to the window's first and last rows:

  ```sql
  compressions(row_id, timeline_id, begin_row_id, end_row_id)
  ```

  A window is a contiguous, inclusive range in the timeline's context order. Windows never overlap each other or a kept row, so kept rows and windows partition the compacted history: every original row is either kept as itself or covered by exactly one compressed turn, which sits at its window's position.

Compressed turns render as ordinary user and assistant turns, so the model sees a natural conversation; only the store distinguishes verbatim from paraphrase. The result is validated like every other context change. Provenance means a seed can point at a compressed row while the audit trail still shows it is a paraphrase, and expanding it back to its originals is just reading the rows between `begin` and `end` (for example, a future `absorb`).

## Skills and Tools

- **Tools come only through skills.** A timeline's capabilities are exactly `timeline_skills → skill_tools`. There is no direct tool-to-timeline link.
- **Skills are the unit of scope.** A clone does only what its linked skills allow.
- Skills are rows. Creating a clone links skills to its timeline through the `timeline_skills` junction table (skills are reused across many clones).
- Skills declare no required input contract. The seed is whatever the source's `clone` call selected.
- **Progressive disclosure.** A timeline's system prompt lists only the `name` and `description` of its linked skills. Full `instructions` enter context only when the agent loads a skill with the built-in `load_skill(name)` tool, as an ordinary tool_use/tool_result pair in its timeline. Linked skills' tools are declared from the start, so the tool list stays stable. A loaded skill's instructions remain in that timeline's history, which is exactly why clones matter: they start with none of the source's loaded skills. The original's system prompt carries the catalog (name and description of every skill) so it can decide when to clone.

Three built-in skills define the architecture itself:

| Skill | Linked to | Provides |
|---|---|---|
| **clone** | the original only | Routing rules and the `clone` call (choose skills, seed rows and `focus` for each entry). |
| **join** | every focus graft, never the original | Instructions for when to end: done, out of scope, or failed. Using it raises the join event. |
| **background** | every background graft, in place of `join` | The same `join` tool, with the rules for working unseen: never ask the user anything, write the result as a final reply, then join. |

A graft learns whether it is in the background only from this skill listing; the persona and the seed are the same in both cases.

The original has only `clone`. It converses, routes and clones; every other capability requires a clone.

### Depth

**Grafts are one level deep:** only the original clones. Clones are linked to `join` or `background` but never `clone`, so the chain is at most `original → c7`. Several grafts side by side under the original are still one level deep.

**Nesting was tried and removed; it does not fit grafting.** Letting clones clone was implemented (strict skill narrowing, per-timeline context budgets, and rollup of out-of-scope joins past levels that lacked the skills), evaluated, and removed. A graft hands one continuous conversation to a clone that holds the whole task; splitting that task into sub-clones breaks the continuity grafting exists to keep. In the evals, models left to choose never nested. When pushed to (`clone` linked automatically to multi-skill clones), quality fell: `nested_project` 5/20 against 19/20 without nesting, `multi_part` 2/10 against 10/10, at 22–78% more cost. Part clones lost context, handed back parts they held, and sources waited on the user between parts. Records: `experiments/results/nesting-*`.

**Re-cloning is bounded.** A request that comes back to the original from a failed clone, or comes back out of scope a second time, is not cloned for again; the original answers the user. Within one turn, the original gets one retry after a refused clone call, and one committed clone call. Blocks are tracked per request, so an unrelated request reaching the source neither lifts another request's block nor inherits it; a new user message lets the source clone again. A block is recorded under the request the source is answering when the join is applied (its latest user message), never under whatever the clone was seeded with, so it always matches the check. A source woken only to report finished or failed background grafts does not clone in that run. A wake whose batch includes an `out_of_scope` join may route it, but a source commits at most one clone call in wake runs between two user messages; after that, wakes only report.

## Lifecycle

### Clone

1. The source makes **one `clone` call** carrying a list of one or more entries:

   ```json
   {"clone": [
     {"purpose": "build the site", "skills": ["website"], "focus": true},
     {"purpose": "draft the deck", "skills": ["slides"], "focus": false, "seed_row_ids": [12]}
   ]}
   ```

   Each entry names the clone's `purpose`, the skills to link (from the catalog in the source's system prompt), and `focus`. `focus` is required and has no default, so the source makes the routing decision every time. **At most one entry has `focus: true`.** With no focus entry the source keeps the conversation and every graft runs in the background.
2. A **seeder actor** picks the seed rows of each entry: for each text row in the source's working view it answers one yes/no question ("does a clone doing *purpose* need this message?") with a confidence score, using a small decision model (v1: TypeSafe Jev). Rows at or above a threshold are seeded in view order, and the triggering user message is always seeded. Each question carries only its row and its immediate neighbours plus the purpose and trigger, so the decision model's context limit never binds, and large views are split into parallel batches. The seeder runs once per entry, and the entries are seeded concurrently. The full decision (model, threshold, per-row scores, selection) is recorded with the clone, so seeds stay replayable. If the seeder fails, the source's own seed (row ids or exact quotes, optional in each entry) is the fallback. Either way the seed resolves to pointers, never copies or paraphrases.
3. The harness validates the list and every entry's resulting view (see [Validated views](#validated-views)). The input is model-written, so nothing about its shape is assumed: each entry must be an object with a purpose, a boolean `focus`, `skills` as a list of known skill names without repeats, and seeds as lists of strings and integers; at most one entry has focus. **Limits:** a clone call takes at most 4 entries, and a source has at most 4 live background grafts (running, or finished with their join still queued). **The call is one atomic batch:** any violation refuses the whole call, with the violations listed per entry (by its index in the list), and creates nothing; the source gets one retry. An error raised while handling the call (the seeder, the commit) is a refusal as well: the source always gets a tool result, every transaction of the call is recorded `rolled_back`, and no clone exists.
4. On success the harness inserts, in one transaction, every graft's `timelines` row, its seed pointers into `context`, and its `timeline_skills` links. The `clone` call's single result names every graft ("cloned → c7 (focus, active), c8 (background, running)"), so the source knows what it did.
5. With a focus graft, the source enters the **cloned** state: suspended, its timeline does not grow. The focus graft becomes the active leaf and talks to the user.
6. Without one, the source stays **active**, tells the user what it started, and keeps the conversation.
7. Background grafts start in the **background** state and run concurrently.

**A graft does only the part its skills cover.** A graft is never shown its `purpose`; it has its seed and its skills. When the request has several parts, every graft is typically seeded with the same whole message, so the rule is in the `join` and `background` skills and in their one-line descriptions: do your own part, leave the rest to the other instances, and never treat the rest as a reason to join. The source therefore needs one entry per part. A background graft joins `failed` only when it could not do its own part.

A clone inherits the source's stable configuration (persona, model, global policies). It does **not** inherit conversation history except the rows explicitly seeded.

**The system prompt has exactly two parts:**

1. **Persona:** the system prompt the user provides (identity, tone, policies). It is identical and verbatim on the original and every clone, and comes first so all timelines share one cached prefix. A clone *is* the agent, so it sounds like the agent.
2. **Skill listing:** the progressive-disclosure list (name and description) of the timeline's linked skills; the original also gets the catalog. **This is the only part that differs between timelines.**

Architecture rules (routing, cloning, joining) are not part of the persona; they live in the `clone` and `join` skills and load through progressive disclosure like any other skill.

### Join

1. The clone decides it is done, out of scope, or failed, and uses its **join** skill (a background graft: the `join` tool of its **background** skill).
2. The harness raises a join event. Everything after this is a deterministic state transition, with no model call:
   - focus graft: `active → joined` (timeline frozen permanently); source: `cloned → joining`
   - background graft: `background → queued` (frozen permanently, its join waiting on the source's queue), then `queued → joined` when the join is applied to the source at rest (see [Parallel grafts](#parallel-grafts))
3. The join event carries only harness-generated fields, never clone-written text:

   ```text
   {clone: c7, reason: done | out_of_scope | failed, trigger: <row_id> | null}
   ```

   `trigger` is the user message the clone did not handle, if any. A background graft never receives a user message, so its trigger is always null.
4. Mechanical failures (errors, timeouts, exhausted budget) raise `failed` without the model.
5. **A background graft always joins.** Nobody can answer it, so a turn that ends on a text reply without a join call joins it `done` (the reply is its result), and an empty response joins it `failed`.

### Default join (deterministic)

When a clone joins, the harness applies a fixed default with no model call, then validates it like any other commit:

| Reason | Default join |
|---|---|
| `done` | pointers to the text rows (user and assistant) of the clone's **working view**, skipping rows the source already has |
| `out_of_scope` | the same, plus the trigger message |
| `failed` | the same, plus a harness error record |

- The **working view** is the clone's context as it would be sent to the model: the full conversation if it never compacted, or its kept and compressed rows plus the turns after the last compaction if it did. The join is therefore bounded by one context window, and it is what a single agent would have remembered had it held the conversation itself.
- Compaction is the agent compressing its own memory, with every compressed row pointing at the window it condenses, not a worker reporting to someone else, so it does not reintroduce the telephone problem.
- Tool calls, tool results and thinking stay in the clone's frozen timeline.
- The source then returns to `active`. On `done` it waits for the user; on `out_of_scope` or `failed` its only remaining decision is routing.
- Before joining `done`, a clone must have given its final reply (a focus graft to the user; a background graft as its written result); a clone never ends silently.
- A background graft's join is the same default: the text rows of its working view that the source lacks. The seeded request is a row the source already holds, so it is skipped.

The only exit from `cloned` is the focus graft's join event; the default join runs during `joining` and returns the source to `active`.



### State machine

A timeline that can hold the conversation (the original, and a focus graft):

```text
active                       holds the conversation
  idle                       at rest: no model call in flight
  busy
    turn                     answering a request (a user message, or one a clone handed back); may clone
    reporting                woken only to tell the user what background grafts did; a clone is refused
cloned                       suspended while its focus graft holds the conversation
joining                      applying joins (a focus graft's, and queued background joins)
joined                       final: frozen

idle ──turn begins──▶ turn ──turn ends──▶ idle        idle ──report begins──▶ reporting ──turn ends──▶ idle
turn ──clone commit (focus graft)──▶ cloned ──focus graft joins──▶ joining
turn ──clone commit (background only)──▶ turn
idle ──drain begins [joins queued, holds the conversation]──▶ joining
joining ──join commit | join rollback──▶ active (idle)
active ──join skill [has a source]──▶ joined
```

A background graft:

```text
background ──finishes──▶ queued ──join commit | join rollback──▶ joined (final)
```

`queued` means the graft is frozen and its join is waiting; `joined` means the join was applied to the source. Both are recorded in `timelines.status`, so a join that has not landed is never mistaken for one that has. The queue belongs to the source's state machine, in finish order, and a graft leaves it only when its join has been applied.

**The state machine is the authority.** It refuses a drain when the source is in a turn, suspended, not holding the conversation, or has nothing queued; it refuses a clone outside `turn`; it refuses a second join on a joined graft. The harness asks and is refused; it does not decide from states it reads. A drain always ends through `join commit` or `join rollback`: rollback if any queued join was rolled back or its commit raised. A join whose commit raises stays `queued`, the user is told at once which clones are affected and that the commit failed, the source returns to rest, and the join is retried at the next commit point.

**Invariant: at most one branch holds the conversation.** Exactly one timeline is in `active` or `joining`, and it holds the single global conversation slot. Every other timeline is `cloned` (suspended), `background` or `queued` (outside the slot) or `joined` (frozen). A focus graft's join hands the slot from the clone straight to its source. Creating a background graft is not a handoff, and its join never moves the slot. The harness enforces this as a single global state, not per source.

**Exactly one timeline talks to the user:** the original, or its live focus graft (`original → c7`). A source has at most one focus graft at a time, because it is suspended while that graft lives.

### Parallel grafts

A request with independent parts ("build the site, draft the deck, book the room") is one clone call with one background entry per part. The source keeps the conversation while they run, and can also hand the conversation to one focus graft while others run in the background.

**Background grafts never speak.** Only the slot holder's text reaches the user. A background graft's text is stored in its own timeline and reaches the source's memory through its join; the source tells the user the outcome.

**Joins are queued, and commit only while the source is at rest.** A finishing background graft freezes at once (`queued`) and its join event goes on its source's queue, in finish order. The join itself is unchanged: deterministic, pointer-only, no model call, size-checked against the source's view at commit time. A queued join commits only when the state machine accepts a drain: its source holds the conversation and is `idle`, so the join never lands between a `tool_use` and its result or under a model call in flight. The commit points are:

- **before a new user message** is appended to the source: the queued joins commit first, and the source answers the message with them in view;
- **at the end of the source's turn**, and **when a graft finishes while the source is idle**: the queued joins commit and the source is woken (below);
- **when a focus graft joins back** to a suspended source: the background joins queued meanwhile commit *before* the focus join, so the focus graft's trigger stays the last row.

Because contexts are append-only, a late join never moves earlier rows, and the source's prompt cache stays valid.

**Notes name the grafts still running.** Every join record and every wake note appended to a source says, in harness-generated text, which of that source's grafts are still running, with their skills ("Still running: c9 (skills: slides)."), or that none are. A focus graft that hands a request back while background grafts are live makes this visible the same way, and the `clone` skill tells the source not to clone again for the parts those grafts are doing.

**Wake.** When queued joins commit and no user message is waiting, the harness appends one closing note as a user row ("background clones c8, c9 joined; tell the user the outcome") and runs the source once, for everything committed together. The note tells the source to report only what has joined and not to describe still-running work as finished. The run is a `reporting` run, in which the machine refuses a clone, unless the batch includes an `out_of_scope` join and the source still has its wake clone call: then it is a `turn` and the source may route the request, under the re-cloning bounds in [Depth](#depth). If the size check kept none of a graft's rows, the note says that its result did not fit and stays in its frozen timeline, instead of pointing at rows that are not there. If the wake run itself fails, the joins stay committed and the harness tells the user at once that reporting failed; this holds wherever the wake happens (after a user turn, after a focus graft's `done` join, when a graft finishes while the source is idle, or when the harness is asked to settle). This is the one place a `done` join leads to a model call: the user did not see the graft's work, so the source reports it.

**Shared resources: no locking.** Background grafts run concurrently, so there can be concurrent writers to external state, and v1 has no ownership or locking model. The source is responsible for handing out work that does not collide; the `clone` skill says so. Continuity across clones (c7 books an appointment, c8 later reschedules it) comes only through the seed: the source points c8 at the joined rows that record c7's actions, or c8 inspects the external state itself.

**Shutdown.** Closing the harness cancels the background grafts still running; each joins `failed`, and joins that can be applied without running anything are applied. No model call is started on shutdown.

**A running graft cannot be steered or cancelled.** Clones are single-use and timelines never message each other. A follow-up on work that is still out waits for the join and then means a new clone. Cancelling a running graft (a harness-raised `failed` join) is deferred.

**What this costs the "no telephone" claim.** A focus graft joins the conversation the user actually had. A background graft joins what it wrote with nobody reading: still its own rows, by pointer, with no model call and nothing rewritten, but the user saw none of it, and what the user hears is the source's account of those rows.

### Single use

Clones are single-use. A clone joins once and never runs again. Agents never message each other: the only cross-timeline events are the **seed** going down and the **join event** going up. Follow-up work always means a new clone, seeded from the source's timeline (which now holds whatever it joined).

## User Experience

- **Clones are invisible to the user.** There is one agent and one conversation; the user never sees the seams.
- **A clone never says "I can't do that."** Anything out of scope joins back silently and the source handles it.
- **Background work is reported by the agent, not by the clone.** While background grafts run the user keeps talking to the same agent; when they finish, the agent says so in its own words.
- **A clone does only its own part of what it starts with.** The messages a clone is seeded with may ask for several things. It does the part its skills cover. The other parts belong to other clones: it does not attempt them, apologise for them or mention them, and they are never a reason to join `out_of_scope` or `failed`.
- **A clone joins with the whole of a later message.** If a message the user sends after the clone started is partly in scope ("make the header darker and email the draft to Bob"), the clone joins before acting on any of it. Only the source splits a request. In the example, the source joins c7, sends the email itself, then creates c8 for the header.
- **Honesty guard.** Invisibility is an implementation detail, like process scheduling, not a persona. Asked directly, the agent answers truthfully: "I'm the same assistant; I narrowed my tools for this task."

## Isolation Rules

1. A clone sees only its seed pointers and its linked skills. No implicit history.
2. Only the text rows of the clone's working view join back (for a focus graft, its user-visible conversation); tool calls, tool results and thinking stay in the clone's frozen timeline.
3. Information crosses timelines only through the `clone` call (down) and the default join (up).
4. **Reads go downward only.** A source may query its descendants' timelines; a clone may never read its source or its siblings, including siblings running at the same time. Enforced by the harness (SQLite authorizer or per-agent views), not by prompt.
5. The only writes are new turns and appended pointers from a committed clone or join.
6. **Least privilege:** a clone holds only its linked skills' tools.
7. **Joined rows keep their role and kind.** Clone content is never rewritten or promoted into a different role; the only new row a join creates is the harness's join record (and error record on `failed`). Tool output is never joined by default, so a focus graft's join adds no prompt-injection exposure beyond the conversation the user already saw. A background graft's text was not seen by the user: it is model-written text that may restate tool output, and it enters the source's context in the assistant role.

> **Isolation by default, permeability on demand.**

## Testing

Grafts give natural evaluation boundaries:

- **Skill evals** run as `skill + seed rows + tools`. Every clone's seed is a recorded pointer set, so any run is exactly replayable, and a skill's eval corpus is a set of seeds.
- **Gathering evals:** given this timeline, did the source seed the right rows and pick the right skills?
- **Join evals:** given this frozen clone timeline and trigger, did the source remember the right things?
- **Integration:** routing, scope detection (did the clone join when it should have), and end-to-end conversations.

Honest limit: clones inherit system instructions and the model, so changing those changes every clone. What grafts guarantee is that adding a skill does not change *other skills'* execution environments.

## Prior Art

| Existing | What it does | What differs here |
|---|---|---|
| OpenAI Agents SDK handoffs | Control passes to another agent that talks to the user; history filtered by `input_filter`. | Targets are separate agents, not clones; history passes by default; nothing joins back. |
| Context-Folding (Sun et al., 2025) | `branch`/`return` inside one agent's reasoning. | The branch returns a self-written summary (telephone); no user takeover. |
| Claude Code / Agent SDK subagents | Fresh context, task, returned summary. | Worker reports on itself; never talks to the user. |
| MemGPT / Letta | Paged memory tiers. | Memory tiers, not execution contexts. |
| Conversation branching (edit-branches, Loom) | Split timelines. | No join. |

What appears new as a whole: single-use clones that take over the user conversation, plus a **deterministic, pointer-based join** of the conversation itself, in which the worker never summarizes its own work.

## Future Ideas

- **`absorb(clone)`:** an on-demand recall where the source pulls detail the default join left out (a tool result, a pre-compaction turn) from a frozen descendant, via a read-only SQL transaction with validated commit. Dropped from v1 because no run needed it; add it back only if an experiment shows the source needs detail it cannot get otherwise.
- **Cancelling and steering a running graft:** a harness-raised `failed` join for a background graft the user no longer wants, and a way to pass a follow-up to work that is still out.
- **Ownership of external state:** a locking or ownership model for concurrent background grafts, replacing the rule that the source hands out non-colliding work.
- **Other seeder models:** v1 uses Jev. Alternatives (other decision APIs, a fine-tuned local model such as Needle 3) are worth revisiting once real-trace replays provide varied training and evaluation data. Needle 3 zero-shot did not qualify: its confidence did not separate needed rows from the rest.

## Non-goals

- Making specialized agents more capable than a single frontier model.
- Agent-to-agent messaging or negotiation.
- Clones that clone (see [Depth](#depth)).
- A new model or training method; this is a context-management and orchestration layer.

## Open Questions

1. **API verification.** Can spliced `thinking` blocks keep valid signatures? Do historical `tool_use` blocks require those tools to be declared on the source? If either fails, joins drop thinking blocks or declare the tools disabled.
2. **Join size** (implemented as a backstop). When the harness knows the model window (`window_tokens`), a join must leave the source's working view under 85% of it (the estimate is characters ÷ 4, and the reply needs room). The check runs at join time, not after compaction: compaction's own request carries the whole prefix, so it cannot rescue a view that is already over the window. The first of these that fits is committed: (0) every remembered row; (1) the clone's last assistant reply, then the trigger; (2) the trigger only. The join record says which rows were left out. They stay in the clone's frozen timeline, reachable by pointer. The window defaults per model (`providers.CONTEXT_WINDOWS`, by model-id prefix); `GRAFT_WINDOW_TOKENS` or bench `--window-tokens` overrides it (0 = off), and models with no known window are not checked.

## Validation Experiments

The baseline is **subagents**, not a single shared agent.

1. **Isolation (done, MVP):** the same skill as a clone vs inside a shared context with conflicting skills loaded. On gpt-6-luna both arms scored 20/20 with zero leakage; clones used about 3x fewer input tokens. Isolation alone does not distinguish the architecture on frontier models.
2. **Cost and caching vs subagents (in progress):** subagents diverge from the parent's cache and pay output tokens for the handoff prompt and the summary, plus prefill of each. Grafts replace both with pointer selection. Measured per role: uncached/cached input, cache writes, output.
3. **Fact loss vs selection recall (planned):** subagents lose facts by paraphrase (summary caps, compaction, nested delegation); grafts lose them only by selection (the source failing to seed or join a relevant row). Compare on tasks with many specific facts, a sweep of summary caps, and delegation depth 1-3.
4. **Replay on real traces (pilot run, `experiments/replay/`):** take existing Claude Code / Codex transcripts with subagents. At each point where a subagent finished, emulate a join: replace the summary tool_result with the default join over the subagent's actual transcript, then compare how the parent continues (next action, answers to probe questions about the subagent's work, tokens) against the recorded trace. This tests the join on real work without building a full runtime first.

   Also force the handoff: at the point where the parent invoked the subagent, cut the parent and make a `clone` call instead of letting it write a handoff prompt, then run a fresh clone from that seed and join it back. Compare that full counterfactual branch (seed → clone run → join → parent continuation) against the recorded branch (handoff prompt → subagent → summary → parent continuation). Clone runs need the environment as it was at that point, so start with read-only subagents (code search and research) replayed against the repository checked out at the trace's commit; subagents with side effects come later, if at all.

   Pilot results are in the README under "Results so far": with both arms on the same delegate model, neither the clone nor the subagent was clearly better, and the delegate model dominated the outcome.
5. **Parallel grafts vs sequential clone-join (mechanics only, `experiments/parallel.py`):** the same multi-part request as one clone call with background grafts, and as one focus clone per part. Offline, with a scripted provider, both arms pass the same checks on the simulated world (the two worlds are not compared with each other, and the arms are not the same script: one message per part against one message for all parts). Whether a real model routes parts to background grafts sensibly, whether quality holds against the sequential baseline, and the real wall-clock gain are not measured yet.

## Principle

Agent Grafts assumes **context boundaries are useful even when the model is capable of doing everything**, and that **the agent which delegated should be the one that decides what it remembers**.

> **One agent identity, single-use cloned timelines, everything queryable, and a lossless join chosen by the source.**
