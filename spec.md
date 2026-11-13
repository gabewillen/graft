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
      tool_name, content, tokens, ts,           -- stored once, never edited
      harness)                                  -- 1 = written by the harness (a note), else 0

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
| **clone** | the original only | Routing rules and the `clone` call (choose skills, seed rows and `focus` for each entry), and the `cancel` call that stops running background grafts. |
| **join** | every focus graft, never the original | Instructions for when to end: done, out of scope, or failed. Using it raises the join event. |
| **background** | every background graft, in place of `join` | The same `join` tool, with the rules for working unseen: never ask the user anything, write the result as a final reply, then join. |

A graft learns whether it is in the background only from this skill listing; the persona and the seed are the same in both cases.

The original has only `clone`. It converses, routes and clones; every other capability requires a clone.

### Depth

**Grafts are one level deep:** only the original clones. Clones are linked to `join` or `background` but never `clone`, so the chain is at most `original → c7`. Several grafts side by side under the original are still one level deep.

**Nesting was tried and removed; it does not fit grafting.** Letting clones clone was implemented (strict skill narrowing, per-timeline context budgets, and rollup of out-of-scope joins past levels that lacked the skills), evaluated, and removed. A graft hands one continuous conversation to a clone that holds the whole task; splitting that task into sub-clones breaks the continuity grafting exists to keep. In the evals, models left to choose never nested. When pushed to (`clone` linked automatically to multi-skill clones), quality fell: `nested_project` 5/20 against 19/20 without nesting, `multi_part` 2/10 against 10/10, at 22–78% more cost. Part clones lost context, handed back parts they held, and sources waited on the user between parts. Records: `experiments/results/nesting-*`.

**Re-cloning is bounded, per part.** A request with several parts is given to several clones, and each clone's *part* is identified by the request it was working on and its own linked catalog skills, both known to the harness. The request a clone was working on is harness-owned: a user message the clone received after it started and hands back, or otherwise the source's current request at the moment it was cloned, recorded then for every graft, focus or background. It is never a row the model chose to seed, and never a harness note: a clone that hands back straight away is counted under its source's request, whatever its seed ended on. A part that comes back from a failed clone, or comes back out of scope a second time, is not cloned for again: a clone call with an entry whose skills overlap a blocked part of the source's current request is refused, and the refusal names that entry. Other parts of the same request are not affected, so when one sibling fails and another hands its part back, the handed-back part can still be routed once; and a clone that fails after the user has moved on to a new message blocks nothing for that new message. A part the original cancelled is blocked the same way, from the moment of the cancel until the next user message, without being counted as a hand-back. Further bounds hold whatever the parts are. **At most 4 clone calls are committed for one user message**, counted across every hand-back and wake that message sets off and reset only by the next user message; this cap reads no key, so no mistake in attributing a hand-back can make cloning loop. Within one run the original makes at most two clone calls: one, and one retry after a refusal; a committed call is the run's last. The machine counts them, and a further call is answered "clone not executed" without anything being seeded or audited; a request that has been handed back 8 times in all is not cloned for again; a source woken only to report finished or failed background grafts does not clone in that run; and a source commits at most one clone call in wake runs between two user messages, after which wakes only report. Every hand-back is counted, including one for a source that had no user message when it cloned. A new user message lets the source clone again.

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
2. A **seeder actor** picks the seed rows of each entry: for each text row in the source's working view it answers one yes/no question ("does a clone doing *purpose* need this message?") with a confidence score, using a small decision model (v1: TypeSafe Jev). Rows at or above a threshold are seeded in view order, and the triggering user message is always seeded. Each question carries only its row and its immediate neighbours plus the purpose and trigger, so the decision model's context limit never binds, and large views are split into parallel batches. The seeder runs once per entry, and the entries are seeded concurrently; if one seeder call raises, the others are cancelled and awaited before the call is refused. The full decision (model, threshold, per-row scores, selection) is recorded with the clone, so seeds stay replayable. If the seeder fails, the source's own seed (row ids or exact quotes, optional in each entry) is the fallback. Either way the seed resolves to pointers, never copies or paraphrases, and only to rows the source holds in its own context.
3. The harness validates the list and every entry's resulting view (see [Validated views](#validated-views)). The input is model-written, so nothing about its shape is assumed: each entry must be an object with a purpose, a boolean `focus`, `skills` as a list of known skill names without repeats, and seeds as lists of strings and integers; at most one entry has focus. **Limits:** a clone call takes at most 4 entries (a longer list is refused whole, before any entry is looked at, with one audit record), and a source has at most 4 live background grafts (running, or finished with their join still queued); the cap is a guard of the clone transition. **The call is one atomic batch:** any violation refuses the whole call, with the violations listed per entry (by its index in the list), and creates nothing; the source gets one retry. An error raised while handling the call (the seeder, a refusal by the state machine, the commit) is a refusal as well: the source always gets a tool result, every transaction of the call is recorded `rolled_back`, and no clone exists. The state machine's step and the store's commit are one unit: the machine decides the prepared call first and may refuse it, and the commit is the activity of its `committing` state, so if the single SQL transaction fails the call is refused (`committing ──error──▶ refusing`) and no graft machine is made; the machine and the store never disagree about which grafts exist.
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
   - focus graft: `active → queued` (frozen permanently, its join last on the source's queue); source: `cloned → joining`. When the join has been written to the source: `queued → joined`.
   - background graft: `background → queued` (frozen permanently, its join waiting on the source's queue), then `queued → joined` when the join is applied to the source at rest (see [Parallel grafts](#parallel-grafts))

   The state machine is asked first. If it refuses the join event (the timeline is the original, or has already raised its join event), nothing is written: a second join never changes a graft's recorded reason. The reason and trigger travel with the queue entry and are written to `timelines` once the event is accepted.
3. The join event carries only harness-generated fields, never clone-written text:

   ```text
   {clone: c7, reason: done | out_of_scope | failed | cancelled, trigger: <row_id> | null}
   ```

   `cancelled` is a reason only the harness sets, when the original cancels a running background graft (see [Parallel grafts](#parallel-grafts)). It is not among the reasons the `join` tool offers; a clone that sends it, like any reason outside `done`, `out_of_scope` and `failed`, is recorded as `failed`.

   `trigger` is the user message the clone did not handle, if any: the last user message in its view when it hands it back or never answered it. A harness note is not a user message and is never a trigger. A background graft never receives a user message, so its trigger is always null.
4. Mechanical failures (errors, timeouts, exhausted budget) raise `failed` without the model. A model call has a time limit (300 seconds by default, configurable, and it can be switched off); a call that does not return in time is such a failure, so a hung call cannot keep a graft live for ever. This holds for a focus graft as for a background graft: if its run raises, it joins `failed`, its source takes the conversation back and answers the user, and the error does not escape as an exception from the user's turn.
5. **A background graft always joins.** Nobody can answer it, so a turn that ends on a text reply without a join call joins it `done` (the reply is its result), and an empty response joins it `failed`.

### Default join (deterministic)

When a clone joins, the harness applies a fixed default with no model call, then validates it like any other commit:

| Reason | Default join |
|---|---|
| `done` | pointers to the text rows (user and assistant) of the clone's **working view**, skipping rows the source already has |
| `out_of_scope` | the same, plus the trigger message |
| `failed` | the same, plus a harness error record |
| `cancelled` | the same (whatever the graft had written before it was stopped), plus a harness record that the original cancelled it, naming the capability tools it had already executed, including any that returned an error (marked as such) |

- The **working view** is the clone's context as it would be sent to the model: the full conversation if it never compacted, or its kept and compressed rows plus the turns after the last compaction if it did. The join is therefore bounded by one context window, and it is what a single agent would have remembered had it held the conversation itself.
- Compaction is the agent compressing its own memory, with every compressed row pointing at the window it condenses, not a worker reporting to someone else, so it does not reintroduce the telephone problem.
- Tool calls, tool results and thinking stay in the clone's frozen timeline.
- The source then returns to `active`. After a focus graft's `done` join it waits for the user. After `out_of_scope` or `failed` it always runs once and answers the user, whether or not the clone had already spoken: in a turn on the handed-back message if that message is now its last row, otherwise woken with a closing harness note. Its only remaining decision is routing.
- **One join is one SQL transaction:** the join record, the pointers, a failed clone's error record, the trigger pointer and the outcome of the join's audit record are written together or not at all. A join that fails halfway leaves nothing in the source, its audit record says `failed`, and it can be retried without duplicating anything. The write and the state machine's step are one unit: the graft becomes `joined` only after the write, and if that step fails after the write, the retry finds the join in the store, writes nothing, and completes the step.
- Before joining `done`, a clone must have given its final reply (a focus graft to the user; a background graft as its written result); a clone never ends silently.
- A background graft's join is the same default: the text rows of its working view that the source lacks. The seeded request is a row the source already holds, so it is skipped.

A suspended source leaves `cloned` through its focus graft's join event (a clone call that could not be committed never reaches `cloned`: it is refused while the source is still in its turn). Joins are applied during `joining`, which always returns the source to `active`.



### State machine

The state machines (stateforward `hsm`) own all of the coordination: what may happen, who waits, and in what order. The harness holds no lock, no task of its own and no future except a reply carried in an event's data; it does only I/O for the machines (model calls, store reads and writes, tool execution, emitting text, building prompts). A run of the model loop is the activity of the machine state it belongs to: leaving that state ends the run, and the library awaits its end inside the same step, before the transition's effect. Machines talk only by events: an effect forwards an event to another machine. A caller outside the machines (a user message, `settle`, `close`, a clone or cancel call made by a model loop) gets its answer either as a response event or from a future carried in the event's data, which an effect resolves; the future only carries the answer, and the graph decides who waits. Decisions are guards and choices; an effect only changes its own machine's data, answers a reply carried in the event, and dispatches at most one event. A step that has more to send goes through states: a chain of choices with one send per transition (giving up a join: the repeat of the commit failure, the abandonment line, the graft's `join abandon`), or a state that ticks once on entry and a choice that sends the next item (the failures of a background graft's `recording`, the cancel result rows that could not be written), so the reply a caller waits on is answered last. Everything that can fail is the activity of a state: a clone call's preparation (`preparing`: seeding and validation), every store write (a user message's row in `requesting`, a clone call's commit in `committing` and its result row in `recording`, a refused call's audit in `refusing`, the cancel results in `writing`, a run's open tool calls in `closing_calls`, each join in `applying`, a wake note in `noting`, a background graft's join reason in its own `recording`), starting graft machines (`spawning`, which awaits each start), and the lines and trace the user's callbacks receive (the scribe machine's `writing`). The activity ends by dispatching its outcome as an event; if it raises, the library dispatches `hsm/error`, and every such state has a transition for it. No write sits between a machine step and its closing event: the write is the activity of the state that the closing event leaves, so the machine cannot move on before the write's outcome is known, and cannot stay in that state after it, whether the write succeeded or failed. A clone call's grafts are started (`spawning`) before anything else learns of them; if one cannot start, the machines made are stopped and forgotten, the committed grafts are marked abandoned, and the call's result is a refusal.

A timeline that can hold the conversation (the original; a focus graft, which starts on its request at once):

```text
live
  idle                       at rest: holds the conversation, nothing in flight, nothing queued
  engaged
    busy                     a run of the model loop (its activity): a turn (a user message, or one a
                             clone handed back; may clone), a routing wake (may clone once), a report
                             (clone refused), or an external turn (a caller drives it: tests, replay)
      running
      clone_gate             (choice) may the clone call be tried at all? (before anything is seeded)
      preparing              the harness seeds and validates the call (activity; an interrupt ends it)
      steps                  one clone or cancel request, step by step (shutdown waits):
        cloning              rejecting | refusing | committing, spawning, recording (activities)
        cancels              the cancel calls of one response: `checking` decides each id (a choice
                             per id), `waiting` until the grafts it cancelled queue their joins,
                             `writing` the result rows (activity), `reporting` each row that failed
    ran                      (choice) how the run ended: a focus graft that joined, or spent its
                             step budget, -> handing_back; else -> rest
    handing_back             (a focus graft) its join event is made (activity) and goes to its source
    joining                  applying the joins on its queue, in order, one per step: a choice picks
                             the next join or leaves for `after`
    rest                     (choice) after a run: joins queued -> joining; a message waiting -> busy;
                             else -> idle
    after                    (choice) after joins were applied: shut down -> closed; a wake -> busy;
                             a message waiting -> busy; else -> idle
  cloned                     suspended while its focus graft holds the conversation
queued                       (a focus graft) frozen, its join waiting on its source; not final
joined                       final: its join was applied
abandoned                    final: its join could not be applied and was given up
closed                       final (the original): shut down

idle ──user message──▶ busy (turn)                  joins queued: ──▶ joining first
idle ──graft queued──▶ joining                      a background graft finished while it was idle
busy ──run done──▶ ran ──▶ rest | handing_back ──▶ queued
busy ──interrupt [its caller's] | the run raised──▶ rest (its open calls closed first)
busy ──user message──▶ busy                         kept on the machine, answered after the run
running ──clone request──▶ clone_gate ▶ preparing ▶ decided: a focus graft ▶ cloned; otherwise ▶ running
running ──clone request [with cancels]──▶ cancels, then clone_gate (unless an interrupt came)
running ──cancel request──▶ cancels: grafts to stop ▶ waiting; otherwise ▶ writing ▶ running
waiting ──graft queued [the last graft it cancelled]──▶ writing     (interrupt and shutdown wait in cancels)
cloned ──user message──▶ forwarded to the focus graft
cloned ──focus graft joins──▶ joining               the focus join goes last on the queue
live ──join [a focus graft]──▶ queued               also when its run raises (reason failed)
queued ──join commit | join rollback──▶ joined, ──join abandon──▶ abandoned
```

A background graft:

```text
background                   its run (the activity); `held` instead while a caller drives the turn that made it
background ──finish (choice on how the run ended: joined with a reason, replied: done, else failed)
           | cancel | shutdown | join (forced) | the run raised──▶ recording ──▶ queued
queued ──join commit | join rollback──▶ joined (final), ──join abandon──▶ abandoned (final)
```

Leaving `background` ends the run first, whatever the event, so a graft is never `queued` while its run still goes, and there is one closer: `recording` writes the join reason (its activity), tells the user of each write that failed, one per step, and then forwards the join to the source. A graft that was about to finish when it was cancelled cannot race the cancel: the graft's machine takes one of the two events, and the other finds it `queued` and changes nothing.

The session machine (one per harness) knows whether anything is in flight: the original engaged (running or applying joins), or a background graft whose join has not reached its source. `settle` is a request to it, answered when nothing is; `close` is `shutdown`, which it forwards first to the original (from then on nothing is woken), then to every background graft still running, and, once all their joins have reached the original, to the original again, which applies what it can and closes. A clone call that was already past the original's gate when closing began may still make grafts: the session shuts them down the same way when it learns of them, and the original, shut down while a graft of its own still runs, waits at rest for that graft's join before it closes. A `settle` whose caller goes away is withdrawn from the session, so the errors it would have raised stay kept for the next one. It also keeps the errors `settle` raises: a failure that happens while a user's call is waiting is shown in that call's output and is that call's; any other is raised by the next `settle`, once.

`queued` means the graft is frozen and its join event, with its reason and trigger, is waiting on its source; `joined` means that join was applied to the source. A focus graft and a background graft go through the same two steps. Both states are recorded in `timelines.status`, so a join that has not landed is never mistaken for one that has. The queue belongs to the original's machine, in finish order; it changes only in effects, and a graft leaves it only when its join has been applied or given up.

**The state machine is the authority.** The graph refuses a clone outside a turn or routing wake, from a graft (grafts are one level deep), with more than one focus entry, or beyond the cap on live background grafts; a join from the original, or a second time; a cancel from a graft, from the original when it is not running, or of anything but a background graft of its own whose join has not been raised. Joins are applied only from rest: when a graft's join arrives at an idle original, at the end of a run, before a user message that waits, and when a focus graft joins back. The harness never reads a machine: what it needs for a run or a note (the timeline's kind, the source's grafts still running, whether the clones named are background grafts) comes in the host call the machine's activity makes, and tests observe only replies, the calls the machines make to the host, and what is stored. The clone rules are the machine's, in two steps around the seeding. Before anything is seeded, the machine's gate on the whole call refuses it while the harness is closing, from a graft, once the run's two clone calls are used, in a report run, past the per-message cap, or for a request handed back too often. Then the harness prepares the call (the activity of the machine's `preparing` state): it checks the call's shape, including at most one focus entry, and seeds and validates every entry. Then the machine decides the prepared call: new ids, at most one focus graft, the cap on live background grafts, and the re-clone blocks, which are the original machine's own data. A join that cannot be applied, a focus graft's as much as a background graft's, stays `queued` with its reason and trigger; the user is told at once which clones are affected and that the commit failed; the original returns to rest holding the conversation; and the join is retried at the next commit point, not at once. A handed-back user message is still answered once its join lands. The retries are bounded: after 3 failed attempts the join is given up, the graft ends `abandoned`, the user is told once that its work is not part of the conversation's memory, and the joins behind it are applied.

**An interrupted turn leaves a valid view.** If a run ends (an interrupt, an error, shutdown) after a tool call was stored and before its result was, every tool call of that last response that has no result gets one: a result saying the call was not executed, or, for a cancel call that did run and whose result alone could not be written, its real result, which the machine kept. A clone call's open transactions are closed as `rolled_back`. The next model call on that timeline is well formed.

**Invariant: at most one branch holds the conversation.** Exactly one timeline is `active` or `joining`: the original, or, while it is `cloned`, its focus graft. Every other timeline is `cloned` (suspended), `background` or `queued`, or `joined` or `abandoned` (final). A user message always goes to the original's machine, which answers it, keeps it until the run in progress is done, or forwards it to its focus graft; a focus graft that joins hands any message it never got to back to the original, in order. Creating a background graft does not move the conversation, and neither does its join.

**Exactly one timeline talks to the user:** the original, or its live focus graft (`original → c7`). A source has at most one focus graft at a time, because it is suspended while that graft lives.

### Parallel grafts

A request with independent parts ("build the site, draft the deck, book the room") is one clone call with one background entry per part. The source keeps the conversation while they run, and can also hand the conversation to one focus graft while others run in the background.

**Background grafts never speak.** Only the conversation holder's text reaches the user. A background graft's text is stored in its own timeline and reaches the source's memory through its join; the source tells the user the outcome.

**Joins are queued, and commit only while the source is at rest.** A finishing background graft freezes at once (`queued`) and its join event goes on its source's queue, in finish order. The join itself is unchanged: deterministic, pointer-only, no model call, size-checked against the source's view at commit time. A queued join commits only from rest, in the original's machine (`idle`, or right after a run), so the join never lands between a `tool_use` and its result or under a model call in flight. Nothing waits outside the machine: a user message that arrives while the original runs or applies joins is kept on its machine and answered after; a join that arrives while it runs waits on its queue. The commit points are:

- **before a new user message** is appended to the source: the queued joins commit first, and the source answers the message with them in view, with no wake;
- **at the end of the source's turn**, and **when a graft finishes while the source is idle**: the queued joins commit and the source is woken (below);
- **when a focus graft joins back** to a suspended source: the background joins queued meanwhile commit *before* the focus join, so the focus graft's trigger stays the last row.

Because contexts are append-only, a late join never moves earlier rows, and the source's prompt cache stays valid.

**Harness notes are marked, not recognised.** Join records, error records and wake notes are user-role rows the harness writes, and each carries `harness = 1` in `turns`. Everything that must tell a note from a user message (what the source's current request is, which rows are seed candidates, the re-cloning bounds) reads that mark, never the text, so a user message that looks like a note is still a user message.

**Notes name the grafts still running.** Every join record and every wake note appended to a source says, in harness-generated text, which of that source's grafts are still running, with their skills ("Still running: c9 (skills: slides)."), or that none are. A focus graft that hands a request back while background grafts are live makes this visible the same way, and the `clone` skill tells the source not to clone again for the parts those grafts are doing.

**Wake.** When joins commit that the source must act on (background grafts' work, or a focus graft that failed or handed back without leaving a user message as the last row) and no user message is waiting, the harness appends one closing note as a user row ("background clones c8, c9 joined; tell the user the outcome") and runs the source once, for everything committed together. The note tells the source to report only what has joined and not to describe still-running work as finished. The run is a `reporting` run, in which the machine refuses a clone, unless a focus graft failed or handed back, or the batch includes a background `out_of_scope` join and the source still has its wake clone call: then it is a `turn` and the source may route the request, under the re-cloning bounds in [Depth](#depth). If the size check kept none of a graft's rows, the note says that its result did not fit and stays in its frozen timeline, instead of pointing at rows that are not there. If the wake run itself fails, the joins stay committed and the harness tells the user at once that reporting failed; this holds wherever the wake happens (after a user turn, after a focus graft's `done` join, when a graft finishes while the source is idle, or when the harness is asked to settle). This is the one place a `done` join leads to a model call: the user did not see the graft's work, so the source reports it.

**Shared resources: no locking.** Background grafts run concurrently, so there can be concurrent writers to external state, and v1 has no ownership or locking model. The source is responsible for handing out work that does not collide; the `clone` skill says so. Continuity across clones (c7 books an appointment, c8 later reschedules it) comes only through the seed: the source points c8 at the joined rows that record c7's actions, or c8 inspects the external state itself.

**Shutdown.** Closing the harness is a request to the session machine. The original is told first, so nothing is woken from then on; every background graft still running is shut down (its run ends, and it joins `failed`), including one that never ran; a graft whose cancel is in flight finishes it and joins `cancelled`, since its machine takes one event at a time. A failed write of a join reason is reported and changes nothing else. Once every graft's join has reached the original, the original applies the joins it can without running anything and closes; one suspended under a focus graft closes with its joins still queued. No model call is started on shutdown.

**Cancel: only the original, and always a forced join.** The original's `clone` skill carries a second tool, `cancel(cancel=[graft ids])`, for a part the user no longer wants or that is no longer needed. No graft can cancel anything: not a sibling, not itself, and not while it holds the conversation as the focus graft (a user who tells a focus graft to stop another part is asking for something outside its part; it joins back and the original cancels). The tool belongs to the `clone` skill, which is never linked to a graft, and the rule does not rest on that: a cancel call is a request to the caller's own machine, and the graph refuses it from any timeline but the original while it runs, whatever tools the caller was shown.

The cancel calls of a response are one request to the original's machine, made after the response's other tool calls have their results stored. In one step it cancels each graft the graph allows (the graft leaves its live grafts at once, its part is blocked, and `cancel` is forwarded to it) and refuses every other id; then it waits (`cancels`) until each cancelled graft's join is queued, and only then stores each call's real result. The graft's machine ends its run before anything else (the run is the activity of `background`): the model call it was waiting on is abandoned, a result that arrives later is discarded, never acted on, and a tool call left without a result gets one saying it was not executed, so the frozen timeline is valid. The graft's join is queued with the reason `cancelled` and is applied at the next commit point like any other join. An interrupt or shutdown of the original's run waits on its machine while it is in `cancels`, so a cancel that ran is always recorded with its real result.

**Cancel and clone in one response: the cancels run first.** A response of the original may carry a `clone` call together with one or more `cancel` calls. The response's cancel calls travel in the clone request itself, and the original's machine runs them first, wherever the model put them in the response; only then is the clone call gated, seeded and decided, so a cancel runs even when the clone call is then refused in its preparation. An interrupt that arrives meanwhile ends the run after the cancels, and the clone call is not run. So "stop this part and start that one" is one response, and it works at the cap on live background grafts, because a cancelled graft's place is free before the clone call is checked. The re-clone block set by the cancel applies to that clone call: an entry for the same part (the same request, with any of the cancelled graft's skills) is refused, and an entry for a different part is allowed. If the clone call is refused, the cancels stand. The clone call's result and the "still running" lists do not name the cancelled graft. If the clone call has a focus entry, the source is suspended with the cancelled graft's join on its queue, and that join commits before the focus graft's join, like any background join queued meanwhile. Any tool call other than `cancel` beside a `clone` call is answered "not executed: call clone on its own" and is not run. The original is in the middle of its turn when it cancels, so the tool result says only that the graft was stopped and that its work so far joins when the turn ends. The join brings the text the graft had written, by pointer, and a harness record naming the capability tools it had already executed. A tool that returned an error is listed too, marked as such, because it may have acted before it failed; only a call the harness itself did not run is left out. Whether a call was run is not read from the wording of its result: the harness marks each result it writes for a call it did not run (interrupted, skipped beside a clone call or a join, or a tool the timeline does not hold) in the stored row, and the record reads that mark. The mark is not sent to the model, and it is a field of the stored result, so the schema is unchanged. There is no way to cancel a graft and discard its work: what a graft did to the outside world is not undone when it stops, so the original must see it, to tell the user what was left behind. The wake note that follows says which grafts the original cancelled and tells it to report that and not to clone for that part again.

Everything else is a refusal in the tool result, never an error, and touches nothing: input that is not a list of one to eight ids, an item that is not shaped like a graft id (a letter followed by at most 15 letters, digits, `_` or `-`; the harness's own ids are `c<number>`), an unknown id, the original's own id, a focus graft, and a graft that has already raised its join event (finished, already cancelled, or abandoned). A graft that has already finished keeps its queued join and its reason. Ids are taken one at a time: in a call that names a running graft and a bad id, the graft is cancelled and the bad id is reported. A malformed item is shown back as a short, cut-off repr, never in full, so model-written text cannot put a long or multi-line string into the tool result.

A cancel is the original's own act, not a hand-back: it does not count toward the bounds on hand-backs, and it is never routed to a new clone. The part the graft was doing is blocked from re-cloning until the next user message: the block is the original machine's own data, added when the graft's join lands as `cancelled` (a graft that finished first blocks nothing). A cancelled graft stops counting against the cap on live background grafts when the cancel is accepted, and is left out of the "still running" lists from then on. If the user's call is cancelled while a graft is being stopped, the call is answered at once, and the machine still completes the stop, queues the join and stores the tool result. After the run has ended the join event is raised whatever the store does; a store error while it is recorded (closing the graft's timeline, writing the reason) is shown to the user at once as a harness error line and does not change the tool result, the turn or shutdown; the reason is written again when the join is applied. If the write of the cancel call's own result row fails, the user is told that this write failed, and that a graft was stopped only if the call stopped one; the run ends with that error, and the row then written holds the real result, which the machine kept.

**A running graft cannot be steered.** Clones are single-use and timelines never message each other. A follow-up on work that is still out waits for the join, or for a cancel, and then means a new clone.

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
3. Information crosses timelines only through the `clone` call (down) and the default join (up). A cancel carries no content: it stops a graft and forces its join.
4. **A timeline reads only its own context.** No timeline has a tool that queries the store: each sees exactly the rows its context points at, rendered by the harness. A clone never reads its source or its siblings, including siblings running at the same time, and a source reaches a clone's rows only through that clone's join. A seed may point only at rows the source itself holds in its context, so a clone cannot be seeded with a running sibling's work or with tool output that never joined; the harness refuses any other row id. This is enforced in the harness (seed resolution and rendering), not by prompt; there is no SQLite authorizer or per-agent view.
5. The only writes are new turns and appended pointers from a committed clone or join.
6. **Least privilege:** a clone holds only its linked skills' tools.
7. **Joined rows keep their role and kind.** Clone content is never rewritten or promoted into a different role; the only new rows a join creates are the harness's join record and, on `failed` or `cancelled`, a harness error or cancel record. The cancel record names the capability tools the graft executed and whether each returned an error, never their input or output. Tool output is never joined by default, so a focus graft's join adds no prompt-injection exposure beyond the conversation the user already saw. A background graft's text was not seen by the user: it is model-written text that may restate tool output, and it enters the source's context in the assistant role.

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
- **Steering a running graft:** a way to pass a follow-up to work that is still out. (Cancelling one is in v1: see [Parallel grafts](#parallel-grafts).)
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
5. **Parallel grafts vs sequential clone-join (mechanics only, `experiments/parallel.py`):** the same multi-part request as one clone call with background grafts, and as one focus clone per part. Offline, with a scripted provider, both arms pass the same checks on the simulated world (the two worlds are not compared with each other, and the arms are not the same script: one message per part against one message for all parts). In both arms a clone that ends its turn on a question while the checks are unmet gets one fixed user reply, at most three times per arm, so that a live clone asking for details is not scored as a harness failure; the number used is reported. Whether a real model routes parts to background grafts sensibly, whether quality holds against the sequential baseline, and the real wall-clock gain are not measured yet.

## Principle

Agent Grafts assumes **context boundaries are useful even when the model is capable of doing everything**, and that **the agent which delegated should be the one that decides what it remembers**.

> **One agent identity, single-use cloned timelines, everything queryable, and a lossless join chosen by the source.**
