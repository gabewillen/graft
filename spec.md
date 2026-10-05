# RFC: Agent Grafts

Status: draft, under review. Decisions below are locked unless listed in [Open Questions](#open-questions).

## Summary

Agent Grafts is an architecture for isolating specialized work without giving up the flexibility of a single general-purpose agent, and without the information loss of agent-to-agent communication.

> **An agent is a timeline. Cloning suspends it; joining resumes it with the clone's timeline available as its own memory. Context is isolated on the way down and lossless on the way up.**

The name comes from horticulture: a cutting is taken from a plant and later grafted back so both grow as one organism. Here an agent clones itself into a single-use timeline, and that timeline is grafted back onto its source as memory. (Earlier drafts called this "Agent Mitosis"; cells that divide never rejoin, so the name no longer fit.)

The core principle is:

> **Replicate the agent, not its context.**

The user talks to one agent. When specialized work is needed, the agent **clones** itself into a new timeline that holds only the context and skills it chose for that clone. The clone talks to the user directly. When the clone is done or out of scope, it **joins**: the source resumes with the conversation the user had with the clone appended to its own memory, by pointer, with no model call.

The problem this solves is *lost in translation*: no worker ever summarizes its own work for someone else. What comes back is the record itself: the conversation the user actually had with the clone, joined by pointer.

## Terminology

| Term | Meaning |
|---|---|
| **original** | The user-facing agent at the top of the chain. It has no source. |
| **clone** | A single-use agent created by a source. It shares the source's identity and stable configuration and has its own timeline. |
| **source** | The agent that created a given clone (the original or another clone). |
| **timeline** | An agent's full, durable context: every turn, in order. An agent *is* its timeline. |
| **clone** (the call) | The source's single `clone` call: it chooses the skills and the seed rows; the harness validates and commits it. |
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

Two built-in skills define the architecture itself:

| Skill | Linked to | Provides |
|---|---|---|
| **clone** | original (and clones, once nesting is enabled) | Routing rules and the `clone` call (choose skills and seed rows). |
| **join** | every clone, never the original | Instructions for when to end: done, out of scope, or failed. Using it raises the join event. |

The original has only `clone`. It converses, routes and clones; every other capability requires a clone.

### Depth

**v1 is depth 1:** only the original clones. Clones are linked to `join` but not `clone`, so the chain is at most `original → c7`.

**Nesting is a planned extension, and v1 must not preclude it.** The intended rule: a clone linked to `clone` may clone itself, and its child's skills must be a subset of its own (capability only narrows down the chain). Enabling it should require only linking `clone` to clones and adding the subset check, not a rewrite. Implementation constraints for v1:

- `timelines.source_id` is a real parent pointer, and nothing assumes the source is the original.
- Cloning and joining are written against "this timeline and its source", never against "the original" by name or special case.
- The per-timeline state machine (`active → cloned → joining → active`, `active → joined`) is the same for every timeline; the original is just a timeline with no source and no `join` skill.
- Read access is defined as "descendants of this timeline" (a recursive query over `source_id`), even though v1 descendants are one level deep.
- "Which timeline talks to the user" is computed as the deepest active leaf, not hard-coded to the clone.
- Skill linking goes through one check function, so the subset rule is added in one place.

## Lifecycle

### Clone

1. The source makes **one `clone` call** naming the clone's `purpose` and the skills to link (from the catalog in its system prompt).
2. A **seeder actor** picks the seed rows: for each text row in the source's working view it answers one yes/no question ("does a clone doing *purpose* need this message?") with a confidence score, using a small decision model (v1: TypeSafe Jev). Rows at or above a threshold are seeded in view order, and the triggering user message is always seeded. Each question carries only its row and its immediate neighbours plus the purpose and trigger, so the decision model's context limit never binds, and large views are split into parallel batches. The full decision (model, threshold, per-row scores, selection) is recorded with the clone, so seeds stay replayable. If the seeder fails, the source's own seed (row ids or exact quotes, optional in the `clone` call) is the fallback. Either way the seed resolves to pointers, never copies or paraphrases.
3. The harness validates the resulting view (see [Validated views](#validated-views)). On violations it returns them to the source for one retry.
4. On success the harness inserts the `timelines` row, the seed pointers into `context`, and the `timeline_skills` links. The `clone` call's result confirms it ("cloned → c7, active"), so the source knows what it did.
5. The source enters the **cloned** state: suspended, its timeline does not grow.
6. The clone becomes the active leaf and talks to the user.

A clone inherits the source's stable configuration (persona, model, global policies). It does **not** inherit conversation history except the rows explicitly seeded.

**The system prompt has exactly two parts:**

1. **Persona:** the system prompt the user provides (identity, tone, policies). It is identical and verbatim on the original and every clone, and comes first so all timelines share one cached prefix. A clone *is* the agent, so it sounds like the agent.
2. **Skill listing:** the progressive-disclosure list (name and description) of the timeline's linked skills; the original also gets the catalog. **This is the only part that differs between timelines.**

Architecture rules (routing, cloning, joining) are not part of the persona; they live in the `clone` and `join` skills and load through progressive disclosure like any other skill.

### Join

1. The clone decides it is done, out of scope, or failed, and uses its **join** skill.
2. The harness raises a join event. Everything after this is a deterministic state transition, with no model call:
   - clone: `active → joined` (timeline frozen permanently)
   - source: `cloned → joining`
3. The join event carries only harness-generated fields, never clone-written text:

   ```text
   {clone: c7, reason: done | out_of_scope | failed, trigger: <row_id> | null}
   ```

   `trigger` is the user message the clone did not handle, if any.
4. Mechanical failures (errors, timeouts, exhausted budget) raise `failed` without the model.

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
- Before joining `done`, a clone must have given the user its final reply; a clone never ends silently.

The only exit from `cloned` is the child's join event; the default join runs during `joining` and returns the source to `active`.



### State machine

```text
source:  active ──clone commit──▶ cloned ──child joins──▶ joining ──default join──▶ active
clone:   active ──join skill──▶ joined (frozen)
```

**Invariant: exactly one timeline is running at any moment, at any depth**, meaning in `active` or `joining`. Every other timeline is `cloned` (suspended) or `joined` (frozen). A join hands the single running slot from the clone straight to its source. The harness enforces this as a single global state, not per source.

**Shared resources follow from this.** With one active timeline, there are never concurrent writers to external state, so v1 has no ownership or locking model. Continuity across clones (c7 books an appointment, c8 later reschedules it) comes only through the seed: the source points c8 at the joined rows that record c7's actions, or c8 inspects the external state itself.

**A source has at most one clone at a time.** Multi-part requests ("book the dentist and pay the invoice") run sequentially: clone, join, clone, join. Because each source has a single child, the system is always a single path with exactly one leaf talking to the user: `original → c7` in v1, `original → c7 → c12` once nesting is enabled. Parallel work is a non-goal for now; it could later return as *background clones* that never talk to the user.

### Single use

Clones are single-use. A clone joins once and never runs again. Agents never message each other: the only cross-timeline events are the **seed** going down and the **join event** going up. Follow-up work always means a new clone, seeded from the source's timeline (which now holds whatever it joined).

## User Experience

- **Clones are invisible to the user.** There is one agent and one conversation; the user never sees the seams.
- **A clone never says "I can't do that."** Anything out of scope joins back silently and the source handles it.
- **A clone joins with the whole message.** If a user message is partly in scope ("make the header darker and email the draft to Bob"), the clone joins before acting on any of it. Only the source splits a request. In the example, the source joins c7, sends the email itself, then creates c8 for the header.
- **Honesty guard.** Invisibility is an implementation detail, like process scheduling, not a persona. Asked directly, the agent answers truthfully: "I'm the same assistant; I narrowed my tools for this task."

## Isolation Rules

1. A clone sees only its seed pointers and its linked skills. No implicit history.
2. Only the clone's user-visible conversation (the text rows of its working view) joins back; tool calls, tool results and thinking stay in the clone's frozen timeline.
3. Information crosses timelines only through the `clone` call (down) and the default join (up).
4. **Reads go downward only.** A source may query its descendants' timelines; a clone may never read its source or its siblings. Enforced by the harness (SQLite authorizer or per-agent views), not by prompt.
5. The only writes are new turns and appended pointers from a committed clone or join.
6. **Least privilege:** a clone holds only its linked skills' tools.
7. **Joined rows keep their role and kind.** Clone content is never rewritten or promoted into a different role; the only new row a join creates is the harness's join record (and error record on `failed`). Tool output is never joined by default, so a join adds no prompt-injection exposure beyond the conversation the user already saw.

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
- **Other seeder models:** v1 uses Jev. Alternatives (other decision APIs, a fine-tuned local model such as Needle 3) are worth revisiting once real-trace replays provide varied training and evaluation data. Needle 3 zero-shot did not qualify: its confidence did not separate needed rows from the rest.

## Non-goals

- Making specialized agents more capable than a single frontier model.
- Agent-to-agent messaging or negotiation.
- Parallel clones under one source.
- A new model or training method; this is a context-management and orchestration layer.

## Open Questions

1. **API verification.** Can spliced `thinking` blocks keep valid signatures? Do historical `tool_use` blocks require those tools to be declared on the source? If either fails, joins drop thinking blocks or declare the tools disabled.
2. **Nesting limits** (future). When nesting is enabled: strict-subset skills (depth bounded by skill count, no clean-context re-clones) or subset-or-equal plus an explicit depth/token budget.

## Validation Experiments

The baseline is **subagents**, not a single shared agent.

1. **Isolation (done, MVP):** the same skill as a clone vs inside a shared context with conflicting skills loaded. On gpt-6-luna both arms scored 20/20 with zero leakage; clones used about 3x fewer input tokens. Isolation alone does not distinguish the architecture on frontier models.
2. **Cost and caching vs subagents (in progress):** subagents diverge from the parent's cache and pay output tokens for the handoff prompt and the summary, plus prefill of each. Grafts replace both with pointer selection. Measured per role: uncached/cached input, cache writes, output.
3. **Fact loss vs selection recall (planned):** subagents lose facts by paraphrase (summary caps, compaction, nested delegation); grafts lose them only by selection (the source failing to seed or join a relevant row). Compare on tasks with many specific facts, a sweep of summary caps, and delegation depth 1-3.
4. **Replay on real traces (planned):** take existing Claude Code / Codex transcripts with subagents. At each point where a subagent finished, emulate a join: replace the summary tool_result with the default join over the subagent's actual transcript, then compare how the parent continues (next action, answers to probe questions about the subagent's work, tokens) against the recorded trace. This tests the join on real work without building a full runtime first.

   Also force the handoff: at the point where the parent invoked the subagent, cut the parent and make a `clone` call instead of letting it write a handoff prompt, then run a fresh clone from that seed and join it back. Compare that full counterfactual branch (seed → clone run → join → parent continuation) against the recorded branch (handoff prompt → subagent → summary → parent continuation). Clone runs need the environment as it was at that point, so start with read-only subagents (code search and research) replayed against the repository checked out at the trace's commit; subagents with side effects come later, if at all.

## Principle

Agent Grafts assumes **context boundaries are useful even when the model is capable of doing everything**, and that **the agent which delegated should be the one that decides what it remembers**.

> **One agent identity, single-use cloned timelines, everything queryable, and a lossless join chosen by the source.**
