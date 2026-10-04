# graft

The smallest working version of [Agent Grafts](spec.md): v1, depth 1, written to find out whether the idea holds up. Clones share the agent's identity but get their own timelines. Everything is stored in SQLite as append-only turns plus pointer contexts. Cloning, joining and compaction are validated pointer selections; the default join is deterministic.

## Setup

```bash
uv sync                      # Python 3.13+ (stateforward.hsm needs >=3.13)
uv run pytest -q             # offline; no API key needed
```

Providers (set `GRAFT_PROVIDER`; if unset, it uses `openai` when `OPENAI_API_KEY` is set, otherwise `anthropic`):

| provider | key | default model | notes |
|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | the first listed by the Models API from `gpt-5.4-mini, gpt-5-mini, gpt-4.1-mini, gpt-4o-mini` | Responses API by default (`GRAFT_OPENAI_API=responses\|chat`), reasoning effort `GRAFT_REASONING_EFFORT` (default `low`); the chat path uses `reasoning_effort="none"` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | Messages API, `thinking: between_tools` (no extended thinking) |

Set `GRAFT_MODEL` to override the model. Keys are only read from the environment at runtime. Never put a key in this repo (`.env` is gitignored). One way to load a key without copying it:

```bash
set -a; source /path/to/.env; set +a
```

## Layout

- `graft/db.py` creates the schema and enforces append-only `turns`/`context` with triggers. It also holds the context view, pointer append with dedup, `render` and `validate`, and `working_view` (compaction-aware join selection).
- `graft/machine.py` holds the stateforward.hsm models: the per-timeline lifecycle and the global single-running `Slot`.
- `graft/harness.py` runs the loop, the one-shot clone, the join event and the deterministic default join.
- `graft/providers.py` has the provider interface: the Anthropic and OpenAI adapters and a `FakeProvider`.
- `graft/skills.py` holds the skill catalog (built-in `clone`/`join`, plus simulated `website`, `slides`, `calendar`) and the in-memory `World`.
- `graft/sim.py` is the scripted user simulator. `graft/fake.py` is a deterministic policy for offline runs.
- `graft/cli.py` provides `uv run graft chat [--trace] [--db PATH]` and `uv run graft dump DB`.

## Experiments

```bash
uv run python experiments/pivot.py            # live
uv run python experiments/pivot.py --fake     # offline dry run (scripted provider)
uv run python experiments/isolation.py -n 5   # live; --fake for offline
uv run python experiments/api_probe.py        # each provider is skipped if its key is missing
```

**pivot.py**: the user describes a website ("dentists in Ohio", "formal", 3 services) over several turns, then says "Instead of a website let's just build a slide show". **PASS** means all of these hold:
1. the facts were told to the website clone, not to the original
2. the website clone joined with `out_of_scope`, triggered by the pivot message
3. the original's default join (deterministic, no model call) committed, and its context now points at the fact rows
4. the slides clone's seed contains the same `row_id`s as the user's audience and tone messages. That makes them verbatim by construction (pointers, not paraphrase).
5. the deck mentions Ohio and dentists, uses a formal tone, and covers at least 2 of the services

The script prints the report and dumps the database (every timeline's context view plus the transactions).

**isolation.py** (`-n 20 --parallel 8` are the defaults) runs the same slides task in two arms at once:
- **shared**: a realistic single agent with progressive disclosure. The system prompt is the catalog only, every capability tool is declared, and the history shows it calling `load_skill` for website, calendar, newsletter, social and recipes and doing real work in each skill's conflicting house style.
- **clone**: a graft clone seeded only with the task row, linked to `slides`+`join`.

Both arms must call `load_skill('slides')` themselves. The scorer checks two things:
- **compliance**: one deck, a slide per service, 3 bullets each, the audience facts, Title Case
- **leakage**: emoji, `!`, #hashtags, ALL-CAPS words, slang, questions to the reader, lowercase titles, bakery-history bleed, foreign tool calls. Each type is reported separately.

It also reports correct/N, stdev, distinct structures, tokens per arm split by cache bucket (and cost if prices are set), Fisher's exact test on pass and leak rates, and a permutation test on scores.

**Subagent baseline** (`graft/subagent.py`; `pivot.py --arm graft|subagent|both`, default both; also a third `subagent` arm in `isolation.py`): the parent (same persona) talks to the user and calls `delegate(skill, task_prompt)`; the subagent gets the persona, that skill's listing and `load_skill`, and only the parent-written prompt, and returns a summary as the tool result. pivot.py reports per-role cache/cost tables for both arms (parent/original, handoff/clone-tx, subagent/clone, summary/join-tx), the output tokens spent on handoffs and summaries vs the one-shot clone calls (the default join makes no call), and, as a diagnostic only, which user facts appear in the handoff/seed and the deck. In isolation.py the clone arm is pre-seeded (no clone call is run or billed), while the subagent arm includes the parent's handoff call.

**Prompt caching** (both experiments): every call records uncached input, cached input reads, cache writes and output, per run and per role (`graft/providers.py` `UsageMixin`; `graft/cachestats.py`).
- OpenAI: `usage.input_tokens_details.cached_tokens` / `cache_write_tokens` (Responses) or `prompt_tokens_details.cached_tokens` (Chat); uncached = input - cached - writes. `prompt_cache_key` (a hash of system prompt + tool names, i.e. persona + skill set) is sent when the installed SDK accepts it; `GRAFT_PROMPT_CACHE_KEY=0` turns it off.
- Anthropic: an explicit `cache_control` breakpoint on the system block (caches tools + system) plus top-level automatic caching for the conversation tail; reads `cache_read_input_tokens` / `cache_creation_input_tokens`.
- `--cache-scope run` (default) puts a unique nonce at the start of the first message of each run, so separate runs share only tools + system; `global` drops it (an upper bound). The reports print each arm/timeline's system and tools+system token length against the provider's minimum cacheable prefix.
- "Effective input" = uncached + 0.1 x cached + 1.25 x writes. These multipliers are assumptions (`GRAFT_CACHED_READ_MULT`, `GRAFT_CACHE_WRITE_MULT`). Cost is printed when `GRAFT_PRICE_IN_PER_M` and `GRAFT_PRICE_OUT_PER_M` are set (optional `GRAFT_PRICE_CACHED_IN_PER_M`, `GRAFT_PRICE_CACHE_WRITE_PER_M`).

**Per-call-site effort and model** (`graft/callsites.py`): each call is mapped to a site — `original`, `clone_call` (an original call answering a fresh user message, i.e. the call that may route with `clone`), `clone`, `compaction`, `subagent_parent`, `subagent`. Effort resolves `GRAFT_EFFORT_<SITE>` → (for `clone_call`: `GRAFT_EFFORT_ORIGINAL`) → `GRAFT_REASONING_EFFORT` (default `low`); model resolves `GRAFT_MODEL_<SITE>` the same way → the provider default. bench flags mirror them (`--effort clone_call=high`, `--model clone_call=...`). Every call log entry records site, model and effort, and the bench prints them per role. Cost is priced per model: `GRAFT_PRICE_*_PER_M__<MODEL>`, then the global `GRAFT_PRICE_*` for `GRAFT_MODEL`, then a small table of verified official prices (`cachestats.OFFICIAL_PRICES`); a run that used any unpriced model shows $ n/a.

**bench.py** (`--arm graft,subagent,subagent_batched | both | all`, `--compaction turns|summary` for the graft arm) runs scenarios (data in `experiments/scenarios.py`: scripted user turns plus a deterministic scorer of the final World and transcript) for both arms: `uv run python experiments/bench.py --scenario all|pivot,revisions,... --arm both -n 10 --parallel 6 [--fake] [--compact-tokens N] [--out runs.json]`. Scenarios:
- `pivot`: as in pivot.py; pass = the deck uses the facts.
- `revisions`: a deck, then five revisions (reorder, retitle, tone, added slide with an email, a typo fix); pass = the final deck reflects all of them. `slides_read` exists in both arms so a subagent can see the current deck.
- `failure`: `calendar_book` always returns an error; pass = it was attempted, no success was claimed, and the last reply says it failed. Graft diagnostics: join reasons and the `failed` error record.
- `long_compaction`: 16 detail turns to a website task, then a pivot to slides, with compaction on (threshold 3000 estimated tokens, the same trigger for every arm). Pass = a scored subset of facts reaches the deck; the other facts are diagnostic. Reports compression ratio (working-view tokens before/after), kept rows and compressed messages per compaction, and per fact whether it survived compaction verbatim (its original row kept), paraphrased (only inside a compressed window or summary) or was lost.
- `multi_part`: one message asks for a booking and a deck.

Arms: `graft`; `subagent` (delegates as it sees fit, summary compaction); `subagent_batched` (the same runner with a delegation policy in the parent's system prompt: gather requirements first, delegate only when it can act, pass the full current requirements every time). Compaction: graft `turns` (default) is one `compact` call returning an ordered list of items that partitions the working view before the fresh message: `{keep: row_id}` or a compressed message `{role, text, begin_row_id, end_row_id}` over a contiguous window. Compressed messages are new rows owned by the timeline, placed at their window, with provenance in the `compressions` table. The harness checks the partition (in order, no gaps, no overlap) and the turn structure, allows one retry, and records the new working view in `compactions` as a pointer list. `summary` (ablation) is one user-role summary row over the whole prefix. The subagent arms always use summary compaction.

Each run gets its own provider (per-run usage), store, World and nonce, and reports pass, effective total, $ cost (when prices are set), calls, wall-clock latency, clones or delegations, output spent on context passing, and compactions. The summary adds Welch's t-test and a permutation test on effective total, and Fisher's exact test on pass rates.

**api_probe.py** answers Open Question 1:
- (a) Are historical tool calls for undeclared tools accepted? This is checked on both Anthropic and OpenAI.
- (b) Are spliced Anthropic thinking blocks accepted? This is checked on Anthropic only.

## Design decisions and deviations

- **Rendering:** consecutive rows with the same role become one message. That is the only mapping from rows to messages. Validation runs on the grouped messages:
  - the view starts with a user turn
  - each `tool_use` is paired with a `tool_result` in the next user turn
  - in a user turn, tool results come before text
  - the view ends on a user turn
  - if there is a trigger, it is the last row
  - thinking rows are allowed only in the timeline that produced them
- **Clone is one call.** `clone(purpose, skills, seed_quotes, seed_row_ids)`: the source names its seed by short verbatim quotes of its own messages (each must match exactly one text row) or by row ids (join records list them). The harness resolves the seed to pointers, validates and commits; on violations the tool result lists them plus an index of the source's messages, and one retry is allowed per user message. The call that emitted `clone` is accounted under `original:clone_tx`. (The model cannot see row ids in its own conversation, which is why quotes exist.)
- **Default join is deterministic (no model call).** On a join event the harness appends pointers to the text rows (user and assistant) of the clone's working view that the source does not already have; `out_of_scope` ends with the trigger, `failed` adds a harness error record before it. The pending view is validated like any commit, then the source returns to `active`. `Store.working_view(tl)` is the single selection point: it returns the full view, or, once a `compactions` marker exists (the MVP never writes one), the summary row plus the turns after it.
- **No explicit join transaction in v1.** The SQL join transaction (spec Future Ideas: `absorb(clone)`) was removed: no run needed it. The `clone` skill's tools are just `load_skill` and `clone`. The read-only agent SQL connection and the `summarize()` UDF went with it.
- **No silent `done`.** `join(reason='done')` is refused with a tool error unless the clone has written user-facing text since the user's last message (`Harness.replied`). The join skill says details about the current task are always in scope; `out_of_scope` is only for a different kind of deliverable or a missing capability.
- **Join record:** a user-role text turn owned by the source, `(harness note: clone cN joined, reason=...; remembered rows ... follow)`. It is placed before the remembered rows.
- **Trigger:** the clone's last user message. It is set when the reason is `out_of_scope`, or when no assistant text came after that message. The harness computes it; the clone never writes it.
- **Progressive disclosure (spec):** `Harness.system()` lists only the name and description of the linked skills; the original also gets the catalog. The built-in `load_skill(name)` tool belongs to both `clone` and `join`, so every timeline has it, and it only loads skills linked to that timeline. Its result is stored as ordinary tool_use/tool_result rows. Linked skills' tools are declared from the start.
- **Skills:** `join` is linked automatically. Linking `clone` to a clone is refused while `NESTING_ENABLED = False`. All skill checks go through `check_skill_link`, which already contains the subset rule for nested sources.
- **No OpenTelemetry:** the delegation scoped this as an experiment, not a product.
