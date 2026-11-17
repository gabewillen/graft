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

withdrawals(timeline_id, row_id, ts)            -- a user message withdrawn after its row was written,
                                                -- unanswered, in that conversation's view
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
- **Rollback is always safe**, and earlier positions never change, so the prompt cache stays valid. The one exception is a withdrawal mark: it changes how its row is shown, once, in the views it binds, when it is written (at the end of the run that did not answer the row, when a focus graft starts for a request withdrawn before it could act, or when the graft holding it joins back), so the cached prefix is kept up to that row.
- **No table modifications** for clone or join beyond inserting pointers.

The harness is a small loop:

1. Read the active timeline's context view.
2. Resolve its tools through `timeline_skills → skill_tools`.
3. Call the model.
4. Insert the new turns and append their pointers.

### Validated views

The view is sent to the model exactly as stored. There is no hidden normalization at render time: the one stored mark that changes how a row is shown is a withdrawal (`withdrawals`, see [State machine](#state-machine)), which shows a user message its caller withdrew as withdrawn.

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

  A window is a contiguous, inclusive range in the timeline's context order. Windows never overlap each other or a kept row, so kept rows and windows partition the compacted history: every original row is either kept as itself or covered by exactly one compressed turn, which sits at its window's position. A message the user withdrew is always kept, with its mark (see [State machine](#state-machine)): a compressed turn would restate it unmarked. The summary ablation (`--compaction summary`, one user-role summary of the whole prefix) leaves such a message out of what the summary call reads, so the summary cannot restate it either.

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
3. The harness validates the list and every entry's resulting view (see [Validated views](#validated-views)). The input is model-written, so nothing about its shape is assumed: each entry must be an object with a purpose, a boolean `focus`, `skills` as a list of known skill names without repeats, and seeds as lists of strings and integers; at most one entry has focus. **Limits:** a clone call takes at most 4 entries (a longer list is refused whole, before any entry is looked at, with one audit record), and a source has at most 4 live background grafts (running, or finished with their join still queued; 4 by default: the harness's `max_live_background`, an integer of at least 1, sets another cap, which the clone skill's text then says, written by `install(store, max_live_background=...)`, and the harness refuses to start with a store whose clone skill says another cap; the default leaves every prompt byte as it was. The 4-entry limit of one clone call stays, so a cap above 4 is reached across clone calls, at later messages or wakes, never in one call); the cap is a guard of the clone transition. **The call is one atomic batch:** any violation refuses the whole call, with the violations listed per entry (by its index in the list), and creates nothing; the source gets one retry. An error raised while handling the call (the seeder, a refusal by the state machine, the commit) is a refusal as well: the source always gets a tool result, every transaction of the call is recorded `rolled_back`, and no clone exists. The state machine's step and the store's commit are one unit: the machine decides the prepared call first and may refuse it, and the commit is the activity of its `committing` state, so if the single SQL transaction fails the call is refused (`committing ──error──▶ refusing`) and no graft machine is made; the machine and the store never disagree about which grafts exist.
4. On success the harness inserts, in one transaction, every graft's `timelines` row, its seed pointers into `context`, and its `timeline_skills` links. The `clone` call's single result names every graft ("cloned → c7 (focus, active), c8 (background, running)"), so the source knows what it did.
5. With a focus graft, the source enters the **cloned** state: suspended, its timeline does not grow. The focus graft becomes the active leaf and talks to the user.
6. Without one, the source stays **active**, tells the user what it started, and keeps the conversation.
7. Background grafts start in the **background** state and run concurrently (one made for a user's call that was withdrawn meanwhile never runs: it joins `cancelled` at once; see [State machine](#state-machine)).

**A graft does only the part its skills cover.** A graft is never shown its `purpose`; it has its seed and its skills. When the request has several parts, every graft is typically seeded with the same whole message, so the rule is in the `join` and `background` skills and in their one-line descriptions: do your own part, leave the rest to the other instances, and never treat the rest as a reason to join. The source therefore needs one entry per part. A background graft joins `failed` only when it could not do its own part.

A clone inherits the source's stable configuration (persona, model, global policies). It does **not** inherit conversation history except the rows explicitly seeded.

**The system prompt has exactly two parts:**

1. **Persona:** the system prompt the user provides (identity, tone, policies). It is identical and verbatim on the original and every clone, and comes first so all timelines share one cached prefix. A clone *is* the agent, so it sounds like the agent.
2. **Skill listing:** the progressive-disclosure list (name and description) of the timeline's linked skills; the original also gets the catalog. **This is the only part that differs between timelines.**

Architecture rules (routing, cloning, joining) are not part of the persona; they live in the `clone` and `join` skills and load through progressive disclosure like any other skill.

### Join

1. The clone decides it is done, out of scope, or failed, and uses its **join** skill (a background graft: the `join` tool of its **background** skill). A focus graft does not join because it has answered: it keeps the conversation until one of those holds, so a join means the clone has ended, not that its result is ready (its replies are already rows in the store; see [User Experience](#user-experience)).
2. The harness raises a join event. Everything after this is a deterministic state transition, with no model call:
   - focus graft: `active → queued` (frozen permanently, its join last on the source's queue); source: `cloned → joining`. When the join has been written to the source: `queued → joined`.
   - background graft: `background → queued` (frozen permanently, its join waiting on the source's queue), then `queued → joined` when the join is applied to the source at rest (see [Parallel grafts](#parallel-grafts))

   The state machine is asked first. If it refuses the join event (the timeline is the original, or has already raised its join event), nothing is written: a second join never changes a graft's recorded reason. The reason and trigger travel with the queue entry and are written to `timelines` once the event is accepted.
3. The join event carries only harness-generated fields, never clone-written text:

   ```text
   {clone: c7, reason: done | out_of_scope | failed | cancelled, trigger: <row_id> | null,
    closing: {<tool_use_id>: (<result>, <is_error>)} | null}
   ```

   `closing` is the graft's close-out it could not write as it froze: when a graft's run ends early (cancelled, shut down, or failed) its open tool calls get results as it records its join (a background graft's `recording`, a focus graft's `failing`); if that write fails, the graft's machine keeps those results (the real result of a call that ran, "cancelled while running" for a tool cut short where it waited) and carries them with its join instead of dropping them, since a frozen graft never writes to its view again. The source writes them into the graft's own timeline in the join's single SQL transaction, before the join's record and pointers (`applying`), so the graft's view is valid once its join lands and the cancel record lists every tool it ran. Only results its last turn still lacks are written, so a retry after a failed join write, which left nothing in the store, writes each once. A call it does not name is closed as not executed. A join that is given up (`abandoned`) never writes it, so the graft, as it is abandoned, has the scribe write its carried close-out into its own timeline (a write that fails is told, once, and raised by `settle`).

   `cancelled` is a reason only the harness sets: when the original cancels a running background graft (see [Parallel grafts](#parallel-grafts)), and when the user's call a background graft was made for is withdrawn (see [State machine](#state-machine)). It is not among the reasons the `join` tool offers; a clone that sends it, like any reason outside `done`, `out_of_scope` and `failed`, is recorded as `failed`.

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
| `cancelled` | the same (whatever the graft had written before it was stopped), plus a harness record that it was cancelled before it finished (by the original, or because the user's call it was made for was withdrawn), naming the capability tools it had already executed, including any that returned an error (marked as such) |

- The **working view** is the clone's context as it would be sent to the model: the full conversation if it never compacted, or its kept and compressed rows plus the turns after the last compaction if it did. The join is therefore bounded by one context window, and it is what a single agent would have remembered had it held the conversation itself.
- Compaction is the agent compressing its own memory, with every compressed row pointing at the window it condenses, not a worker reporting to someone else, so it does not reintroduce the telephone problem.
- Tool calls, tool results and thinking stay in the clone's frozen timeline.
- The source then returns to `active`. After a focus graft's `done` join it waits for the user. After `out_of_scope` or `failed` it always runs once and answers the user (unless the user's call it would answer was withdrawn: see [State machine](#state-machine)), whether or not the clone had already spoken: in a turn on the handed-back message if that message is now its last row, otherwise woken with a closing harness note. Its only remaining decision is routing.
- **One join is one SQL transaction:** the join record, the pointers, a failed clone's error record, the trigger pointer and the outcome of the join's audit record are written together or not at all. A join that fails halfway leaves nothing in the source, its audit record says `failed`, and it can be retried without duplicating anything. The write and the state machine's step are one unit: the graft becomes `joined` only after the write, and if that step fails after the write, the retry finds the join in the store, writes nothing, and completes the step.
- Before joining `done`, a clone must have given its final reply (a focus graft to the user; a background graft as its written result); a clone never ends silently.
- A background graft's join is the same default: the text rows of its working view that the source lacks. The seeded request is a row the source already holds, so it is skipped.

A suspended source leaves `cloned` through its focus graft's join event (a clone call that could not be committed never reaches `cloned`: it is refused while the source is still in its turn). Joins are applied during `joining`, which always returns the source to `active`.



### State machine

The state machines (stateforward `hsm`) own all of the coordination: what may happen, who waits, and in what order. The harness holds no lock, no task of its own and no future except a reply carried in an event's data; it does only I/O for the machines (model calls, store reads and writes, tool execution, emitting text, building prompts). A run of the model loop is the activity of the machine state it belongs to: leaving that state ends the run, and the library awaits its end inside the same step, before the transition's effect. Machines talk only by events: an effect forwards an event to another machine. A caller outside the machines (a user message, `settle`, `close`, a clone or cancel call made by a model loop) gets its answer either as a response event or from a future carried in the event's data, which an effect resolves; the future only carries the answer, and the graph decides who waits. Decisions are guards and choices, and the guarded branches of a choice (or of one event's transitions from one state) are disjoint, so no outcome depends on the order they are declared in: one unguarded default takes the rest. An effect only changes its own machine's data, answers a reply carried in the event, and dispatches at most one event. A step that has more to send goes through states: a chain of choices with one send per transition (giving up a join: the repeat of the commit failure, the abandonment line, the graft's `join abandon`), or a state that ticks once on entry and a choice that sends the next item (the failures of a background graft's `recording`, the cancel result rows that could not be written, the stops of the background grafts made for a withdrawn request), so the reply a caller waits on is answered last. Everything that can fail is the activity of a state: a clone call's preparation (`preparing`: seeding and validation) and its rollback when the run is left while it is prepared (`unpreparing`), every store write (a user message's row in `requesting`, a clone call's commit in `committing` and its result row in `recording`, a refused call's audit in `refusing`, the cancel results in `writing`, a run's open tool calls in `closing_calls`, a withdrawn request's mark in `dropping` and `withdrawing`, each join in `applying`, a wake note in `noting`, a background graft's join reason in its own `recording`), starting graft machines (`spawning`, which awaits each start), a host tool call (`calling`), its result row (`storing`), and the lines and trace the user's callbacks receive (the scribe machine's `writing`). The activity ends by dispatching its outcome as an event; if it raises, the library dispatches `hsm/error`, and every such state has a transition for it. No activity ends without an outcome, and one rule tells a cancel apart in every activity: a `CancelledError` means its state is being left only when the activity's context is done (the library marks it before it cancels an activity, which it does only when the state is left), and is then never swallowed; any other cancel (a host's or a provider's own, surfacing in a call the activity awaited) is the activity's failure, like any other error, and the task's cancel requests are withdrawn. The library tells only an `Exception`, so whatever else an activity raises (but a cancel, or `KeyboardInterrupt` and `SystemExit`, which asyncio lets out of the loop) is raised as one (`Aborted`) and taken by the same transition, or is the activity's own named outcome where it has one (a run's `run_failed`, a tool's result, a row write's `tool_unstored`, a graft start's `spawn_failed`). A run of the model loop is the one activity whose failure is its own event (`run_failed`, whatever it raised but a cancel or the interpreter's own exits): states inside the run have activities of their own (a clone call's preparation, a step's writes), whose `hsm/error` they handle as their own failure, so the run's failure is never taken for theirs. A clone call being prepared when the run fails is cut short and rolled back (`unpreparing`) and the run's failure is then handled as anywhere in `busy`; a step finishes first. A guard, effect, entry or exit is not meant to fail; one that raises is a fault of its machine, recovered from in the graph (see [Internal faults](#internal-faults)). No write sits between a machine step and its closing event: the write is the activity of the state that the closing event leaves, so the machine cannot move on before the write's outcome is known, and cannot stay in that state after it, whether the write succeeded or failed. A clone call's grafts are started (`spawning`) before anything else learns of them, the focus graft last (it starts on its request at once, so it is made only once every background graft of the call has started); if one cannot start, the machines made are stopped and forgotten, the committed grafts are marked abandoned, the session forgets them (`grafts_undone`), so nothing an undone graft told it stays, and the call's result is a refusal.

**A user's call is a request the original's machine owns.** `user_says` makes one request (an id, the text, and the reply future) and puts it on the original's machine without waiting, so its only await is the reply (a step its message sets off that fails answers that reply with the error, which `user_says` raises). The machine holds the request in its own data from arrival until it is answered or withdrawn, wherever it is: waiting (the inbox), its row being written (`requesting`), the request a run answers (its turn, the joins and the wake that follow it, a run that answers it after a focus graft hands it back), a caller waiting at rest, or handed to a focus graft with the conversation. A request is live while it is not withdrawn and its reply is not done. If the call is cancelled, at any point, its reply is cancelled at once (unless the machine has answered it already: see the end of this paragraph) and the caller withdraws the request by id (`withdraw`). The withdrawal is handled in every state of the original's and a focus graft's machine, never ignored: it is recorded while the machine holds the request (the record keeps no id of a request that has left the machine, so it stays bounded), a waiting message is dropped and answered at once and never taken, a suspended original forwards it to its focus graft (also while the clone call that makes the graft is in flight, when the original still holds the request and records it too; while the grafts' machines are being started, `spawning`, it waits for them, and every graft is made with the withdrawals recorded before), and a frozen focus graft, which has handed every request back, holds nothing to record (a request it handed back reaches its source with its reply, which a cancelled call has already cancelled). A run answers at most one request (its turn's message, the one its source answered when it was made, or one handed back), and one rule decides whether a withdrawal ends it: that request is withdrawn. A run whose request is withdrawn ends (its open calls closed); one in a clone or cancel step ends once the step is done. Every point that would start a run or let a graft act for a request is guarded on it being live: the row written (`requesting` → `busy`, else to rest: the row stays, a user turn the next message joins), the next waiting message (`rest`, `after`: withdrawn ones are dropped and answered), the wake, hand-back and routing runs (`after`, and `noting` → `busy`), the clone gate, a focus graft's start (a focus graft whose request was withdrawn before it could act never runs: it marks the request's row, `withdrawing`, and comes to rest), and a background graft's start and each of its steps. A background graft is made for the request the run that made it answers: if that request is withdrawn before it is answered (in the clone step that makes the graft, or later while the user's call still waits), the graft never starts, or is stopped by the original once it stands after recording the withdrawal (before the answer, or after it while the call is not done: the cancel wins): at once if it is at rest (`idle`, or `cloned` under a focus graft), and otherwise as soon as the run in progress ends (`rest` stops them before it takes anything else, a waiting message included). The original stops them as its own steps, one `cancel` per step (a state that ticks once on entry and a choice that sends the next stop, `stop_one`), as a `cancel` call does, so no graft's answer is needed for the next stop and a stop whose step fails is made again, at most 3 times for one graft: a stop that keeps failing is then given up and told once, the source goes on to the next graft, and that graft stops at its next step instead; a host tool the graft awaits in `calling` is cancelled where it waits and recorded as run with an error. A late withdrawal that comes while an unrelated run is in progress therefore stops its grafts when that run ends. As a backstop the graft also stops at its next step (it asks its own machine, which reads the reply its caller cancels at once, and then its source, which stops it, `cancel`, once the request is withdrawn by its record). Either way it joins `cancelled` without a model call or a tool call after the cancel; its join is applied like any other, and wakes nothing. A graft made for a request that was answered, and whose call took that answer, goes on and is reported when it joins. A graft that joins at once in the clone step that makes it is queued after that step, once its source knows it made it; if that step is undone because a graft could not be started, its join is never applied (the source applies only joins of grafts it holds). Whether a run may go on is the machine's decision as well: before every step that could act (each model call, compaction calls included, when such a call returns, after each text it says, before each tool call and before the response's cancel calls) the run asks its machine (`may_continue`, a reply in the event's data), and the machine answers while the request the run answers, if any, is live (not withdrawn, reply not done: a cancelled call cancels its reply at once, before its withdrawal arrives). Once it is withdrawn, the machine leaves `busy` the way a withdrawal leaves it, which ends the run where it waits: a response that came back is not stored, said or acted on, nothing more of a response is said or run, and the calls it stored are closed as not run. A run starts inside a machine step, so its first question is answered in that step and its answer reaches the run only after the step, by which time a caller cancelled in that same step has withdrawn its request and the run has been ended; every later question, the one before the first model call included, is answered in the step the run acts in, so a cancel at any earlier moment is seen. Every other point is a machine step, where the withdrawal is handled. So a withdrawn request never causes a model call, a tool call, text or a stored row after the cancel. A withdrawn request whose row was already written and that no run answered is closed out in the store: the row gets the withdrawal mark (`withdrawals`; the row itself is never edited), written by the state that learns of it (`dropping` when it was withdrawn while its row was written; `withdrawing` when the run answering it is cut short, when a focus graft starts for it after it was withdrawn, or when a focus graft handed it back and no run will answer it). A mark that cannot be written is told to the user once and stays to be written, in the machine's data: whatever next reads that view writes it first (a run of that timeline, its compaction, seeds and seeder included: if the write fails again, the run fails before it reads anything; a clone call's preparation, in a turn its caller drives too; a join's copy into the view), a focus graft hands it back to its source with its join, and a recovery (see [Internal faults](#internal-faults)) writes it. The mark binds the views of the conversation it was asked in: the timeline that holds it and, for a focus graft, its source, which gets the graft's rows back by its join. There the row is shown to the model as a withdrawn message, marked like a harness note, and is never a request, a trigger or a seed candidate, so no later turn or hand-back reads it as live user text. It is never seeded either: a clone entry that names it, by row id or by quote, is refused (a quote matches only the other messages, so it names a message the user sent again after withdrawing it), the seed index a refused call is shown and the seeder's candidates leave it out, and every entry's final seed is checked for it before the call commits. Compaction keeps it with its mark: the compaction call's index marks it, and a window that would compress it is refused; a summary compaction leaves it out of what the summary call reads. The mark does not bind the view of a background graft seeded with the same row before it was withdrawn, which is not that conversation: only the request a run answers can be withdrawn after its row is written, so that graft was made for it, and it stops (see above). A request its turn answered before the cancel (the joins, wake note and wake that follow the turn) is not marked. **A cancel after the answer wins.** A cancel that lands after the machine answered the call but before its caller took the answer (asyncio raises it in the caller's await of a reply that is already done) is a withdrawal like any other: `user_says` raises `CancelledError` and withdraws the request, and the machine honours the withdrawal while the call is not done. It keeps each request it answered at rest until the harness says the call is done (`call_done`, which `user_says` posts to the original as well as to the session; a suspended original passes it to its focus graft, and a focus graft hands back with its join the requests it answered), and a background graft's request until the graft's join is applied. The withdrawal is recorded by the machine that knows the request (the original, if it made background grafts for it, else the timeline that answered it): the background grafts made for it stop before their next step and join `cancelled`, their joins wake nothing, nothing more is said or done for it, and its row gets the withdrawal mark in that timeline's view (and, for a focus graft, its source's), a write of the scribe's (one that fails is told and kept for `settle`). A wake that reports only background grafts made for withdrawn requests does not run, or ends. A request whose reply a recovery failed is kept the same way, as answered. An error the request was answered with, which its caller never took, is told once after that mark: a line, shown in the next call's output (or raised by `settle` if no call is in flight). Lines said for it before the cancel stay in the output the next call returns. A withdrawal of an unknown id, or of one whose call is done, changes nothing.

A timeline that can hold the conversation (the original; a focus graft, which starts on its request at once unless that request was withdrawn before it could act):

```text
live
  idle                       at rest: holds the conversation, nothing in flight, nothing queued;
                             `gate` (choice), `stopping` (ticks once on entry) and `stop_next` (choice:
                             one stop per step) stop the background grafts made for a withdrawn
                             request, then `still`
  engaged
    busy                     a run of the model loop (its activity): a turn (a user message, or one a
                             clone handed back; may clone), a routing wake (may clone once), a report
                             (clone refused), or an external turn (a caller drives it: tests, replay);
                             the run asks `may_continue` before each step, answered while the
                             request it answers, if any, is live
      running
      calling                a host tool call (activity): the tool, awaited if it returns an awaitable;
                             leaving the run cancels it where it waits, and the call is closed as run
                             with an error (it may have acted)
      storing                the result row of a tool that returned (activity)
      clone_gate             (choice) may the clone call be tried at all? (before anything is seeded)
      preparing              the harness seeds and validates the call (activity; leaving the run ends it,
                             and `unpreparing` rolls the call's audit back)
      steps                  one clone or cancel request, step by step (shutdown, a driven end and the
                             run's own failure wait):
        cloning              rejecting | refusing | committing, spawning, recording (activities);
                             flushing (activity): a hand-over first writes the run's open calls
                             and any row the machine kept, then leaves for `cloned`
        cancels              the cancel calls of one response: `checking` decides each id (a choice
                             per id), `waiting` until the grafts it cancelled queue their joins,
                             `writing` the result rows (activity), `reporting` each row that failed
    ran                      (choice) how the run ended: a focus graft that joined, or spent its
                             step budget, -> handing_back; else -> rest
    handing_back             (a focus graft) its join event is made (activity) and goes to its source
    joining                  applying the joins on its queue, in order, one per step: a choice picks
                             the next join or leaves for `after`
    rest                     (choice) after a run: background grafts made for a withdrawn request ->
                             `stops`/`stops_next` (one stop per step) first; then `resting` (choice):
                             joins queued -> joining; a live message waiting -> requesting; else -> idle
    after                    (choice) after joins were applied: shut down -> closed; the request it
                             answers withdrawn -> withdrawing; a live message waiting -> requesting;
                             a wake -> busy; else -> idle
    requesting               a message's row is written (activity); live -> busy, withdrawn -> dropping
    dropping                 the row of a message withdrawn while it was written is marked (activity)
    closing_calls            a run that ended early: its open tool calls get results (activity)
    withdrawing              the row of the withdrawn request it answers, unanswered, is marked
                             (activity) -> rest; also where a focus graft starts if that request was
                             withdrawn before it could act
    unpreparing              a clone call cut short in its preparation is rolled back (activity)
    closing_out              closing with a close-out a recovery could not write: written first (activity)
    recovering               a step of it failed: what the step left open is closed out (activity),
                             then `resumed` (choice): a focus graft it made holds the conversation ->
                             the hand-over finishes; else -> rest
  cloned                     suspended while its focus graft holds the conversation (stops grafts made
                             for a withdrawn request through the same substates as `idle`)
queued                       (a focus graft) frozen, its join waiting on its source; not final
joined                       final: its join was applied
abandoned                    final: its join could not be applied and was given up
closed                       final (the original): shut down

idle ──user message [live]──▶ requesting ──▶ busy (turn)   joins queued: ──▶ joining first
idle ──graft queued──▶ joining                      a background graft finished while it was idle
busy ──run done──▶ ran ──▶ rest | handing_back ──▶ queued
busy ──withdraw [the request it answers] | the run raised──▶ rest (its open calls closed first;
                                                    a clone call being prepared is rolled back first;
                                                    a withdrawn request's row is marked)
busy ──may_continue [it answers no request, or a live one]──▶ busy (answered)
running ──tool request──▶ calling ──tool returned──▶ storing ──tool stored──▶ running   (its row failed: the
                                                    run ends with the error; leaving busy cancels the tool: run, error)
busy ──may_continue [the request it answers withdrawn]──▶ closing_calls ▶ withdrawing ▶ rest
busy ──user message──▶ busy                         kept on the machine, answered after the run
any ──withdraw──▶ (recorded; a waiting message is answered and never taken; never ignored)
running ──clone request──▶ clone_gate ▶ preparing ▶ decided: a focus graft ▶ cloned; otherwise ▶ running
running ──clone request [with cancels]──▶ cancels, then clone_gate (unless its request was withdrawn)
running ──cancel request──▶ cancels: grafts to stop ▶ waiting; otherwise ▶ writing ▶ running
waiting ──graft queued [the last graft it cancelled]──▶ writing     (shutdown waits; a withdrawal is recorded)
cloning ──withdraw──▶ a waiting message's: dropped and answered; else recorded, forwarded to the focus
                                                    graft once it is made; the run ends after the step
spawning ──withdraw──▶ deferred until the grafts' machines are started (each is made with the
                                                    withdrawals recorded before), then handled as above
start [a focus graft, its request withdrawn before it could act]──▶ withdrawing ──▶ rest
cloning ──graft queued──▶ deferred until the step is done
cloning ... ──hand-over──▶ cloned                   the waiting messages go to the focus graft first, in order
cloned ──user message | withdraw──▶ forwarded to the focus graft
cloned ──focus graft joins──▶ joining               the focus join goes last on the queue
live ──join [a focus graft]──▶ queued               also when its run raises (reason failed)
queued ──join commit | join rollback──▶ joined, ──join abandon──▶ abandoned
queued ──user message──▶ passed back to its source, once (`message_returned`: the source keeps it and
                                                    never sends it on again, so no message goes to a focus
                                                    graft and back more than once; a withdrawal is not
                                                    sent back: it holds no request, and its source
                                                    forwarded it)
live ──fault──▶ recovering ──▶ resumed ──▶ rest | passing    a step failed (see Internal faults)
cloned | queued | closed ──fault──▶ the same state           its record re-stated, its join re-sent
```

A background graft:

```text
background                   its run (the activity), in `stepping`, `calling` a host tool or `storing` its row (activities, as for
                             a timeline that holds the conversation); `held` instead while a caller drives
                             the turn that made it
start [the request it was made for withdrawn]──▶ recording (cancelled; it never runs)
background ──fault──▶ recording (failed)            a step failed: it stops; the error is told
queued ──fault──▶ queued                            its join re-sent if the failed step did not send it
background ──finish (choice on how the run ended: joined with a reason, replied: done, else failed)
           | cancel | shutdown | join (forced) | the run raised
           | may_continue [the request it was made for withdrawn: cancelled]──▶ recording ──▶ queued
queued ──join commit | join rollback──▶ joined (final), ──join abandon──▶ abandoned (final)
```

Leaving `background` ends the run first, whatever the event, so a graft is never `queued` while its run still goes, and there is one closer: `recording` writes the join reason (its activity), tells the user of each write that failed, one per step, and then forwards the join to the source. A graft that was about to finish when it was cancelled cannot race the cancel: the graft's machine takes one of the two events, and the other finds it `queued` and changes nothing.

The session machine (one per harness) knows whether anything is in flight: the original engaged (running or applying joins), or a background graft whose join has not reached its source. `settle` is a request to it, answered when nothing is; `close` is `shutdown`, which it forwards first to the original (from then on nothing is woken), then to every background graft still running, and, once all their joins have reached the original, to the original again, which applies what it can and closes. A clone call that was already past the original's gate when closing began may still make grafts: the session shuts them down the same way when it learns of them, and the original, shut down while a graft of its own still runs, waits at rest for that graft's join before it closes. A `settle` whose caller goes away is withdrawn from the session, so the errors it would have raised stay kept for the next one. It also keeps the errors `settle` raises: a failure whose line goes out while a user's call is in flight is shown in that call's output (or, if the call raised, the next call's) and is that call's; any other is raised by the next `settle`, once. Which calls are in flight the harness tells it, as I/O: each `user_says` says when it begins and when it is done (`call_began`, `call_done`, by request id). Its record of each timeline (engaged, and for the original its background grafts whose joins have not reached it) is kept by the events the timelines send it, and re-stated whole by a timeline that recovers from a fault (`resync`); a fault of the session's own steps has the original re-state its record, and, while the original is suspended, its focus graft with it.

`queued` means the graft is frozen and its join event, with its reason and trigger, is waiting on its source; `joined` means that join was applied to the source. A focus graft and a background graft go through the same two steps. Both states are recorded in `timelines.status`, so a join that has not landed is never mistaken for one that has. The queue belongs to the original's machine, in finish order; it changes only in effects, and a graft leaves it only when its join has been applied or given up.

**The state machine is the authority.** The graph refuses a clone outside a turn or routing wake, from a graft (grafts are one level deep), with more than one focus entry, or beyond the cap on live background grafts; a join from the original, or a second time; a cancel from a graft, from the original when it is not running, or of anything but a background graft of its own whose join has not been raised. Joins are applied only from rest: when a graft's join arrives at an idle original, at the end of a run, before a user message that waits, and when a focus graft joins back. The harness never reads a machine: what it needs for a run or a note (the timeline's kind, the source's grafts still running, whether the clones named are background grafts) comes in the host call the machine's activity makes, and tests observe only replies, the calls the machines make to the host, and what is stored (a machine's record of withdrawals included: it is observed by what it decides, a message whose request reuses a withdrawn id). The clone rules are the machine's, in two steps around the seeding. Before anything is seeded, the machine's gate on the whole call refuses it while the harness is closing, from a graft, once the run's two clone calls are used, in a report run, past the per-message cap, or for a request handed back too often. Then the harness prepares the call (the activity of the machine's `preparing` state): it checks the call's shape, including at most one focus entry, and seeds and validates every entry. Then the machine decides the prepared call: new ids, at most one focus graft, the cap on live background grafts, and the re-clone blocks, which are the original machine's own data. A join that cannot be applied, a focus graft's as much as a background graft's, stays `queued` with its reason and trigger; the user is told at once which clones are affected and that the commit failed; the original returns to rest holding the conversation; and the join is retried at the next commit point, not at once. A handed-back user message is still answered once its join lands. The retries are bounded: after 3 failed attempts the join is given up, the graft ends `abandoned`, the user is told once that its work is not part of the conversation's memory, and the joins behind it are applied.

**An interrupted turn leaves a valid view.** If a run ends (a withdrawal, an error, shutdown) after a tool call was stored and before its result was, every tool call of that last response that has no result gets one: a result saying the call was not executed, or, for a call that did run (a tool, or a cancel call) and whose result alone could not be written, its real result, which the machine kept (the run hands it to its machine before the write's error ends the run, so a tool that ran is never recorded as not run). A clone call's open transactions are closed as `rolled_back` (by `unpreparing` when the run is left while the call is being prepared: if that rollback fails, the user is told and `settle` raises it). The next model call on that timeline is well formed: a clone call whose result row, or refusal row, cannot be written ends its run with that error (the user is told, and the call's caller, the run, gets it), so no model call reads a view without that row, and the run's closing writes the row it kept. Each tool call's result is stored as soon as that call returns, so a later call of the same response that raises never leaves a call that ran recorded as not run.

**A host tool call is the activity of a state.** The run does not execute a capability tool itself: it asks its machine (`tool_request`, a reply in the event data), and the machine's `calling` state (inside the run: `busy`'s run, or a background graft's `running`) makes the call as its activity. A host's tool may return its result or an awaitable of it (a World may define `async def execute`): an awaitable is awaited in that activity, so while one timeline's tool waits, every other timeline's model calls, tools and joins go on, and one graft's tool may wait for what another graft's tool produces. The activity only invokes the tool; its outcome (`tool_returned`) carries the result, which the step that takes it keeps with the call, and `storing` (a state of its own, inside the run) writes the row as its activity before the run is answered (its outcome is `tool_stored`, or `tool_unstored` with the write's error). The run asks for one call at a time, after its `may_continue` for that call, so a response's results are stored in call order, and the rules for calls beside `clone`, `cancel` and `join` are unchanged. Leaving the run (its request withdrawn, a cancel, shutdown, a fault's recovery) leaves `calling` and so cancels the tool where it waits (the tool sees the cancellation; graft never swallows it). Cancellation is cooperative: a tool that swallows `CancelledError` and keeps waiting blocks its graft's cancel and `close()` until it returns, and work a tool handed to a thread (`asyncio.to_thread`) or shielded may still act after the cancel, which is why the call's result says it may still act; a withdrawn request therefore gets no row after its withdrawal from the call. A background graft whose request is withdrawn is stopped by the original when it records the withdrawal (see the withdrawal paragraph above), so a tool it awaits is cancelled then too. A call cut short that way may already have acted, so it is never closed as not executed: before the tool is called, the activity keeps the call with the machine's other known results as run with an error ("cancelled while running: it may have acted, or may still act"), the run's closing writes it, and a cancelled graft's record lists it among the tools it executed, marked as returning an error. A tool that is cancelled while its state is still active (the host cancelled a future or task the tool awaited, a sibling graft's cancel included, or the task the tool runs in: only the state's own context being done, which the library marks before it cancels an activity, means the state was left) has not been cancelled by its graft: its call gets a result marked run, with an error ("cancelled by the tool: it may have acted"), and its run goes on; the cancellation that leaving the state makes is never taken for this. Once a tool is invoked it has run, whatever follows: a tool that raises (after acting, or at its own deadline; any exception, also one that is not an `Exception`, but for the interpreter's own exits, `KeyboardInterrupt` and `SystemExit`, the only exceptions asyncio lets out of the loop; a `GeneratorExit` is the tool's outcome like any other) or returns anything but (text, is_error) gets a result marked run, with an error naming what it raised or returned ("raised TimeoutError: ...; it may have acted"), the cancel record lists it, and its run goes on, so the model never redoes a side effect it was told did not happen; the call's trace line is written by the scribe, so a trace callback that fails never touches the result. A failure of the harness's own call after the tool was invoked (`calling`'s error) keeps the call as run too, with an error ("failed after it was invoked: it may have acted"), its row is written (`storing`) and its run goes on. Only a step of the machine that failed before the tool was invoked (its activity then holds back) closes the call as not executed. Once the tool's outcome is taken, its real result replaces that kept one before the row is written, so a result row that cannot be written, or a run left while it is written, closes the call with its real result (a write's failure that comes after the run was left is still told, once); a row that cannot be written ends the run. graft sets no time limit on a host tool call: a tool that waits is the point (it may wait, for example, until another clone files its result), and the host owns its deadlines (an `asyncio.wait_for` inside its own tool, say). A tool that hangs is still stoppable: the original's `cancel`, a withdrawal of the request, or `close()` ends it where it waits, and the call is recorded as run with an error.

**Invariant: at most one branch holds the conversation.** Exactly one timeline is `active` or `joining`: the original, or, while it is `cloned`, its focus graft. Every other timeline is `cloned` (suspended), `background` or `queued`, or `joined` or `abandoned` (final). A user message always goes to the original's machine, which answers it, keeps it until the run in progress is done, or forwards it to its focus graft. Messages are answered in arrival order: when the original hands the conversation to a focus graft, the messages waiting on it go to the graft first, ahead of any forwarded later; a focus graft that joins hands any message it never got to back to the original, in order. Creating a background graft does not move the conversation, and neither does its join.

**Exactly one timeline talks to the user:** the original, or its live focus graft (`original → c7`). A source has at most one focus graft at a time, because it is suspended while that graft lives.

### Internal faults

A guard, effect, entry or exit of the machines that raises is a bug in graft's own code (whatever it raises, an `Exception` or not, but `KeyboardInterrupt` and `SystemExit`: no step escapes the graph): sends between machines never raise (a send to a stopped, unstarted or forgotten machine is lost, not raised), and every I/O is an activity with its own error transition. The library (stateforward `hsm` 1.3.11) turns only an activity's failure into `hsm/error`, so each guard, effect, entry and exit is run so that nothing it gated acts and its failure is recovered from in the graph. The contract:

* **Guards are pure.** A guard reads the machine's data, the event's payload and the state of the reply futures it carries (read when the decision is made: a caller cancels its reply at any moment), and writes nothing. The clone rules, which work out reasons from the run's record, are worked out as data by the step before each gate (`judge_call` before `clone_gate`; the prepared call's verdict as it is taken), with any error they raised, and the gates' guards read that outcome. A guard that raises does not hold; it records, sends, reports and answers nothing. Every branch that acts (commits, starts a graft, starts or continues a run, takes a message, applies a join) is guarded positively, and every decision has an explicit unguarded branch, the safe one: refuse (the clone gate refuses the call, the prepared gate rolls the call back), stop (a run's `may_continue` ends the run, a background graft's start or next step stops it, joining `failed`), keep (a join no guard placed is kept for the recovery), or rest. That branch's effect checks that none of the decision's guarded branches held (it reads them again, as the decision read them, in a step that handles a fault too): if one did, a guard raised and changed the decision, and the effect reports it as a `DecisionFault` naming the guard (a precomputed error, from the clone rules, is raised by the branch that declines the call as itself). Each decision has its safe branch at its own level, so a guard that raises never lets the event fall through to an ancestor's transitions (a withdrawal while a clone call is prepared, for one: its safe branch records it, and a guard that raised there is a fault of the timeline, whose recovery ends the run and rolls the call back). A guard that raised where its branch was not the one to take changes nothing, and nothing is reported: nothing it gated acted, and a guard can record nothing.
* **One event per behavior.** An effect, entry or exit sends at most one event, and its sends are made only when it returns: one that raises, or sends a second, sends none of them, and its one event is its fault. The sends a behavior makes belong to its step alone (they are held per thread and per task, never in shared data), so harnesses on separate threads, each with its own loop, never send through each other's steps. That fault is the machine's data (written by the behavior that raised), and it is sent to the machine as one event of the error kind (`fault`), so it is the next event the machine takes: nothing else is taken between the failed step and its recovery, except that a fault that comes while a clone call's graft machines are being started (`spawning`) waits for the start's outcome, so no machine is left half made. Once a step has failed, no guard of it holds (its decisions take their safe branch on purpose), every activity it enters does nothing (no model call, tool, row, commit or graft start), and no event of the step reaches the session (it would see part of the step). Whether the step in progress has failed is read only by the plumbing that runs the graphs' guards, behaviors and activities, and by guards: no effect, entry, exit or activity reads it. A decision whose safe branch would act otherwise has a branch for a step that failed: such a step comes to rest without answering its callers (they wait for the recovery), and a clone call it committed is undone instead of starting its grafts (`spawn_gate`). One failure is one fault, however many behaviors of the step raised it.
* **A call's step fails alone.** A step that only took a call from outside (an internal transition: a message stashed behind a run, a withdrawal recorded, a message forwarded to a focus graft) touched nothing else, so its fault (`call_fault`) is recovered from where the timeline is (the session's request for the timeline's record whose step failed is answered again, with that step's errors): that call's caller gets the error (its message leaves the inbox), the session gets the record as it stands, and the run in progress, if any, goes on. The caller is every reply the call's event carries, read from its payload (a user's message, a `settle`, a clone call and the cancel calls it brought, a run's `may_continue`), whatever the event: a clone or cancel request from a run whose step failed is answered with the error, so that run ends with it (`run_failed`, handled as any failure of the run). A join that arrived in that step is queued.
* **A timeline recovers to rest.** From anywhere in `live`, any other fault leaves for `recovering` (leaving a run ends the run, and any clone or cancel request the run held is answered). A clone call whose step failed before its grafts were started is undone first, as when a graft cannot start; grafts already started stay the original's (the recovery records them from the step's event if the step did not: a focus graft among them holds the conversation, and the user's calls it was made with stay its own). What else the failed step's event brought is kept too, once (a step that failed after it kept it changes nothing): a graft's join goes on the queue, a focus graft's join brings the conversation back as when it is taken, and a shutdown still closes the timeline once it is at rest. Grafts a cancel stopped whose joins had not come when the step failed stay its own: they no longer count as live, and their joins are applied when they come. `recovering`'s activity closes out what the step left open (if that write fails too, what it could not write is kept, and written first by whatever next writes to or reads that view at rest: a message's row, a run's start, a join's copy, a wake note, a mark, the next recovery, and the timeline's close; so no row lands after a tool call left without its result; a run's open calls that could not be closed, a cut-short call's rollback that failed and a refusal row that could not be written are kept the same way): the clone call's open audit transactions are rolled back, the run's open tool calls get results (a tool that ran, its real result; a clone call that committed, its result; a cancel call whose step was cut short after it stopped a graft, what it did: the graft is stopped and its join is applied when it comes), and the rows of withdrawn requests still to be marked are marked. The timeline re-states its whole record to the session (`resync`: engaged, the user calls it holds, for the original its background grafts still running and the focus graft holding the conversation) first, then the callers waiting on the failed work get the error, once: the caller of the event whose step failed, the message whose row was being written, and the user call whose request no run has answered yet (its run was ended, or the step that would start it failed), unless a focus graft holds the conversation and that call. A reply answered with an error is not a withdrawal: no row is marked withdrawn for it. Every other caller waits on work the failed step did not touch: a waiting message is taken at rest, a call whose request a run answered is answered at rest. It goes on as after a run: to rest, or, if a focus graft it made holds the conversation, through the hand-over to `cloned`. A timeline that is suspended, frozen or closed re-states its record where it is (a frozen focus graft first sends its join if the failed step did not: the join it kept as the step's first act, or, if the step failed before it kept it, one made again from that step's event and what it still holds; a closed original's record says it closed, so `close` goes on: a step that ends in a final state other than `joined` or `abandoned` and fails has closed the timeline, so its callers are answered and its record says so). Recoveries are bounded, so a bug that raises every time never spins: a step that handles a fault is never a fault of its own. The fault is taken before anything of that step runs, and a behavior handling it that raises (in its `fault` step, or in the step that ends its recovery, `recovered`) is part of it: the fault is handled again, from where the machine now is, with that error added, and every handler is safe to run again (a caller answered is not answered twice; what was kept is kept once; callers leave the machine's data only once answered), so the first fault's callers get its error once and nothing waits for ever; past `MAX_FAULT_RETRIES` handlings (a handler that raises every time) the raise is only reported, and only the callers the failed event carries get the error. Past `MAX_RECOVERIES` recoveries in a row the timeline gives up: the session gets its record, every caller waiting on its work gets the error, and it goes on from rest without closing anything out (the count starts again with the next user message or run).
* **A background graft stops.** A fault while it runs stops it (`recording`, reason `failed`); a fault while it records its join is added to what it reports; a frozen graft re-sends its join if the failed step did not send it.
* **The session and the scribe.** A fault of the session's own steps fails its event's caller and every `settle` waiting (the answer they wait on is the session's own, which the failed step may have got wrong), keeps what its event said of a user's call in flight, and has the original re-state its record (suspended, the original passes the question on to its focus graft with its own record, and the graft re-states both in one event); a state that ticks on entry is entered again. A settle left waiting in the session's data by a step that failed is answered by the next record that leaves the session with nothing in flight. A fault of the scribe loses the note it was taking, and the session keeps the error, with the error that note was to keep, for `settle` whatever call is in flight (no line of them went out).
* **Each failure is reported once.** A caller given the error raises it, and nothing else reports it. A fault no caller was given is told to the user as one harness error line and kept for `settle` once the line is out, which raises it once, unless a user's call was in flight when the line went out: that call's output has the line (or, if it raised, the next call's), so the error is that call's and `settle` does not raise it. A behavior that raises while its machine handles a fault is reported the same way, as its one event: told with the fault a timeline is recovering from, carried on the record it was re-stating, or else a line kept for `settle` (the scribe's own goes to the session to keep). The one case it is not reported is a report that cannot be sent either (the scribe or session itself failing as it is told), and a guard raise that changed no decision (above).
* **After a fault the harness is usable**: the next `user_says` is answered, `settle` returns (or raises the kept error, once) and `close` returns (or raises what it could not stop).

### Parallel grafts

A request with independent parts ("build the site, draft the deck, book the room") is one clone call with one background entry per part. The source keeps the conversation while they run, and can also hand the conversation to one focus graft while others run in the background.

**Background grafts never speak.** Only the conversation holder's text reaches the user. A background graft's text is stored in its own timeline and reaches the source's memory through its join; the source tells the user the outcome.

**Joins are queued, and commit only while the source is at rest.** A finishing background graft freezes at once (`queued`) and its join event goes on its source's queue, in finish order. The join itself is unchanged: deterministic, pointer-only, no model call, size-checked against the source's view at commit time. A queued join commits only from rest, in the original's machine (`idle`, or right after a run), so the join never lands between a `tool_use` and its result or under a model call in flight. Nothing waits outside the machine: a user message that arrives while the original runs or applies joins is kept on its machine and answered after (unless its caller withdraws it); a join that arrives while it runs waits on its queue. The commit points are:

- **before a new user message** is appended to the source: the queued joins commit first, and the source answers the message with them in view, with no wake;
- **at the end of the source's turn**, and **when a graft finishes while the source is idle**: the queued joins commit and the source is woken (below);
- **when a focus graft joins back** to a suspended source: the background joins queued meanwhile commit *before* the focus join, so the focus graft's trigger stays the last row.

Because contexts are append-only, a late join never moves earlier rows, and the source's prompt cache stays valid.

**Harness notes are marked, not recognised.** Join records, error records and wake notes are user-role rows the harness writes, and each carries `harness = 1` in `turns`. Everything that must tell a note from a user message (what the source's current request is, which rows are seed candidates, the re-cloning bounds) reads that mark, never the text, so a user message that looks like a note is still a user message. A user message withdrawn after its row was written is marked the same way, in `withdrawals` (see [State machine](#state-machine)).

**Notes name the grafts still running.** Every join record and every wake note appended to a source says, in harness-generated text, which of that source's grafts are still running, with their skills ("Still running: c9 (skills: slides)."), or that none are. A focus graft that hands a request back while background grafts are live makes this visible the same way, and the `clone` skill tells the source not to clone again for the parts those grafts are doing.

**Wake.** When joins commit that the source must act on (background grafts' work, or a focus graft that failed or handed back without leaving a user message as the last row) and no user message is waiting, the harness appends one closing note as a user row ("background clones c8, c9 joined; tell the user the outcome") and runs the source once, for everything committed together. The one exception is a wake that would answer a user's call that was withdrawn (the joins follow a turn, or a hand-back, whose every request was withdrawn): then no note is written (one already being written when the withdrawal comes stays as a row) and no run starts; the joins stay committed in the source's memory, unreported, and the source reports them only if a later turn asks. Likewise the join of a background graft made for a request that was withdrawn is never news: it is applied, and wakes nothing. The note tells the source to report only what has joined and not to describe still-running work as finished. The run is a `reporting` run, in which the machine refuses a clone, unless a focus graft failed or handed back, or the batch includes a background `out_of_scope` join and the source still has its wake clone call: then it is a `turn` and the source may route the request, under the re-cloning bounds in [Depth](#depth). If the size check kept none of a graft's rows, the note says that its result did not fit and stays in its frozen timeline, instead of pointing at rows that are not there. If the wake run itself fails, the joins stay committed and the harness tells the user at once that reporting failed; this holds wherever the wake happens (after a user turn, after a focus graft's `done` join, when a graft finishes while the source is idle, or when the harness is asked to settle). This is the one place a `done` join leads to a model call: the user did not see the graft's work, so the source reports it.

**Shared resources: no locking.** Background grafts run concurrently, so there can be concurrent writers to external state, and v1 has no ownership or locking model. The source is responsible for handing out work that does not collide; the `clone` skill says so. A host tool that waits for another graft's tool (an async `World.execute` awaiting what a sibling files) coordinates through the host's own state, never through graft's context: no graft reads another's rows, and what the waiting tool returns is its own result. Continuity across clones (c7 books an appointment, c8 later reschedules it) comes only through the seed: the source points c8 at the joined rows that record c7's actions, or c8 inspects the external state itself.

**Shutdown.** Closing the harness is a request to the session machine. The original is told first, so nothing is woken from then on; every background graft still running is shut down (its run ends, and it joins `failed`), including one that never ran; a graft whose cancel is in flight finishes it and joins `cancelled`, since its machine takes one event at a time. A failed write of a join reason is reported and changes nothing else. Once every graft's join has reached the original, the original applies the joins it can without running anything and closes; one suspended under a focus graft closes with its joins still queued. No model call is started on shutdown. Once the original has closed (it tells the session, or, if the step that closed it failed, its re-stated record says so), the session stops every machine (the original is told to close by the entry of the session's `finishing`, entered again if that step fails, so it is always told; a fault of the step that stops the machines or tells their failures gives `close` the error), the scribe included, so nothing stays live; a later request is refused. A machine that cannot be stopped does not keep the others live: the rest are still stopped, each failure is shown to the user as a harness error line (each whatever showing another one raised), and `close` raises it (the one error, or an ExceptionGroup), together with anything the user's callbacks raised while the failures were shown. Leaving a run, however it is left (a withdrawal, shutdown, an error), answers any clone or cancel request it had not answered yet with a refusal, so no caller is left waiting.

**Cancel: only the original, and always a forced join.** The original's `clone` skill carries a second tool, `cancel(cancel=[graft ids])`, for a part the user no longer wants or that is no longer needed. No graft can cancel anything: not a sibling, not itself, and not while it holds the conversation as the focus graft (a user who tells a focus graft to stop another part is asking for something outside its part; it joins back and the original cancels). The tool belongs to the `clone` skill, which is never linked to a graft, and the rule does not rest on that: a cancel call is a request to the caller's own machine, and the graph refuses it from any timeline but the original while it runs, whatever tools the caller was shown.

The cancel calls of a response are one request to the original's machine, made after the response's other tool calls have their results stored. In one step it cancels each graft the graph allows (the graft leaves its live grafts at once, its part is blocked, and `cancel` is forwarded to it) and refuses every other id; then it waits (`cancels`) until each cancelled graft's join is queued, and only then stores each call's real result. The graft's machine ends its run before anything else (the run is the activity of `background`): the model call it was waiting on is abandoned, a result that arrives later is discarded, never acted on, and a tool call left without a result gets one saying it was not executed (a host tool that was running when the graft was stopped is cancelled where it waits and recorded as run with an error: it may have acted), so the frozen timeline is valid. The graft's join is queued with the reason `cancelled` and is applied at the next commit point like any other join. A shutdown of the original's run waits on its machine while it is in `cancels`, and a withdrawal is recorded there and ends the run once the cancels are done, so a cancel that ran is always recorded with its real result.

**Cancel and clone in one response: the cancels run first.** A response of the original may carry a `clone` call together with one or more `cancel` calls. The response's cancel calls travel in the clone request itself, and the original's machine runs them first, wherever the model put them in the response; only then is the clone call gated, seeded and decided, so a cancel runs even when the clone call is then refused in its preparation. A withdrawal of the run's requests that arrives meanwhile ends the run after the cancels, and the clone call is not run. So "stop this part and start that one" is one response, and it works at the cap on live background grafts, because a cancelled graft's place is free before the clone call is checked. The re-clone block set by the cancel applies to that clone call: an entry for the same part (the same request, with any of the cancelled graft's skills) is refused, and an entry for a different part is allowed. If the clone call is refused, the cancels stand. The clone call's result and the "still running" lists do not name the cancelled graft. If the clone call has a focus entry, the source is suspended with the cancelled graft's join on its queue, and that join commits before the focus graft's join, like any background join queued meanwhile. Any tool call other than `cancel` beside a `clone` call is answered "not executed: call clone on its own" and is not run. The original is in the middle of its turn when it cancels, so the tool result says only that the graft was stopped and that its work so far joins when the turn ends. The join brings the text the graft had written, by pointer, and a harness record naming the capability tools it had already executed. A tool that returned an error is listed too, marked as such, because it may have acted before it failed; only a call the harness itself did not run is left out. Whether a call was run is not read from the wording of its result: the harness marks each result it writes for a call it did not run (interrupted, skipped beside a clone call or a join, or a tool the timeline does not hold) in the stored row, and the record reads that mark. The mark is not sent to the model, and it is a field of the stored result, so the schema is unchanged. There is no way to cancel a graft and discard its work: what a graft did to the outside world is not undone when it stops, so the original must see it, to tell the user what was left behind. The wake note that follows says which grafts the original cancelled and tells it to report that and not to clone for that part again.

Everything else is a refusal in the tool result, never an error, and touches nothing: input that is not a list of one to eight ids (twice the cap on live background grafts: another cap takes twice it), an item that is not shaped like a graft id (a letter followed by at most 15 letters, digits, `_` or `-`; the harness's own ids are `c<number>`), an unknown id, the original's own id, a focus graft, and a graft that has already raised its join event (finished, already cancelled, or abandoned). A graft that has already finished keeps its queued join and its reason. Ids are taken one at a time: in a call that names a running graft and a bad id, the graft is cancelled and the bad id is reported. A malformed item is shown back as a short, cut-off repr, never in full, so model-written text cannot put a long or multi-line string into the tool result.

A cancel is the original's own act, not a hand-back: it does not count toward the bounds on hand-backs, and it is never routed to a new clone. The part the graft was doing is blocked from re-cloning until the next user message: the block is the original machine's own data, added when the graft's join lands as `cancelled` (a graft that finished first blocks nothing). A cancelled graft stops counting against the cap on live background grafts when the cancel is accepted, and is left out of the "still running" lists from then on. If the user's call is cancelled while a graft is being stopped, the call is answered at once, and the machine still completes the stop, queues the join and stores the tool result. After the run has ended the join event is raised whatever the store does; a store error while it is recorded (closing the graft's timeline, writing the reason) is shown to the user at once as a harness error line and does not change the tool result, the turn or shutdown; the reason is written again when the join is applied. If the write of the cancel call's own result row fails, the user is told that this write failed, and that a graft was stopped only if the call stopped one; the run ends with that error, and the row then written holds the real result, which the machine kept. A clone call that came in the same response is then not run (its tool call is answered as not executed when the run's open calls are closed).

**A running graft cannot be steered.** Clones are single-use and timelines never message each other. A follow-up on work that is still out waits for the join, or for a cancel, and then means a new clone.

**What this costs the "no telephone" claim.** A focus graft joins the conversation the user actually had. A background graft joins what it wrote with nobody reading: still its own rows, by pointer, with no model call and nothing rewritten, but the user saw none of it, and what the user hears is the source's account of those rows.

### Single use

Clones are single-use. A clone joins once and never runs again. Agents never message each other: the only cross-timeline events are the **seed** going down and the **join event** going up. Follow-up work always means a new clone, seeded from the source's timeline (which now holds whatever it joined).

## User Experience

- **Clones are invisible to the user.** There is one agent and one conversation; the user never sees the seams.
- **A clone never says "I can't do that."** Anything out of scope joins back silently and the source handles it.
- **A focus graft joins when it has ended, not when it has answered.** A focus graft does not join after it answers: it holds the conversation and joins only when it is done, hands a message back out of scope, or fails. There is no implicit join and no separate record of a result. A host that shows a clone's progress shows its replies as they are written (they are rows in the store), and reads a join as "this clone has ended", not "its result is ready".
- **Background work is reported by the agent, not by the clone.** While background grafts run the user keeps talking to the same agent; when they finish, the agent says so in its own words.
- **A clone does only its own part of what it starts with.** The messages a clone is seeded with may ask for several things. It does the part its skills cover. The other parts belong to other clones: it does not attempt them, apologise for them or mention them, and they are never a reason to join `out_of_scope` or `failed`.
- **A clone joins with the whole of a later message.** If a message the user sends after the clone started is partly in scope ("make the header darker and email the draft to Bob"), the clone joins before acting on any of it. Only the source splits a request. In the example, the source joins c7, sends the email itself, then creates c8 for the header.
- **Honesty guard.** Invisibility is an implementation detail, like process scheduling, not a persona. Asked directly, the agent answers truthfully: "I'm the same assistant; I narrowed my tools for this task."

## Isolation Rules

1. A clone sees only its seed pointers and its linked skills. No implicit history.
2. Only the text rows of the clone's working view join back (for a focus graft, its user-visible conversation); tool calls, tool results and thinking stay in the clone's frozen timeline.
3. Information crosses timelines only through the `clone` call (down) and the default join (up). A cancel carries no content: it stops a graft and forces its join. Neither does a focus graft's withdrawal mark on its source's view: it names a row the conversation already holds or gets back by the join.
4. **A timeline reads only its own context** (a host tool that waits on another graft's tool goes through the host's state, not graft's context: see Shared resources). No timeline has a tool that queries the store: each sees exactly the rows its context points at, rendered by the harness. A clone never reads its source or its siblings, including siblings running at the same time, and a source reaches a clone's rows only through that clone's join. A seed may point only at rows the source itself holds in its context, so a clone cannot be seeded with a running sibling's work or with tool output that never joined; the harness refuses any other row id. This is enforced in the harness (seed resolution and rendering), not by prompt; there is no SQLite authorizer or per-agent view.
5. The only writes are new turns and appended pointers from a committed clone or join, and withdrawal marks (which carry no content: they say a row was withdrawn, in the views of the conversation it was asked in).
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
