# graft

graft is an experimental alternative to subagents for delegating work inside an LLM agent. Instead of starting a fresh subagent from a handoff prompt the parent writes, the agent **clones** itself: the clone's context is seeded with rows picked from the parent's real conversation, the clone does the task, and when it finishes the conversation it had is **joined** back into the parent's memory by pointer. No model call is needed for the join. All state lives in SQLite as append-only turns plus per-agent pointer lists.

The design is written up in [spec.md](spec.md) ("Agent Grafts"). This repository is the smallest working version of it (v1, depth 1), plus the experiments used to compare it against a subagent baseline.

**Status:** experimental, pre-1.0. The APIs, schema and CLI may change. The experiments so far do not show graft to be better or worse than subagents (see [Results so far](#results-so-far)).

## How it differs from subagents

```text
subagent                                    graft
--------                                    -----
parent ── writes handoff prompt ──▶ worker  parent ── clone([purpose, skills]) ──▶ clone
          (paraphrase of context)                     seed = pointers to the parent's own rows,
                                                      picked by a seeder
worker works in a fresh context             clone works in its own timeline
worker ── writes summary ──▶ parent         clone ── join ──▶ parent
          (paraphrase of its work)                    pointers to the clone's conversation rows,
                                                      appended deterministically
```

- **Down:** a subagent sees only what the parent wrote for it. A clone sees verbatim rows from the parent's timeline, chosen by a seeder (a small decision model, or the parent model itself).
- **Up:** a subagent returns a summary in its own words. A clone returns nothing; the harness appends pointers to the text rows of its conversation to the parent's context.
- **Same:** both isolate the delegated work from the parent's loaded skills and tool history, and both cost extra calls.

## Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# as a dependency
uv add git+https://github.com/gabewillen/graft

# or from a clone
git clone https://github.com/gabewillen/graft
cd graft
uv sync
uv run pytest -q          # offline; no API keys needed
```

The optional `jev` extra installs the TypeSafe SDK used by the Jev seeder: `uv add "graft[jev] @ git+https://github.com/gabewillen/graft"` or `uv sync --extra jev`.

## Quickstart

The CLI has two commands:

```bash
export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY
uv run graft chat --trace            # interactive chat; --trace shows clone/join internals
uv run graft chat --db chat.db       # keep the SQLite store (default: a temp file)
uv run graft dump chat.db            # print every timeline's context and the transactions
```

Inside `chat`, `/dump` prints the store and `/quit` exits. Replies go to stdout; status and error lines (waiting for background clones at quit, an error raised while settling) go to stderr. Ctrl-C exits at once and cancels running background clones. Without a key, `chat` exits with an error naming the missing variable.

The chat agent uses the simulated skills in `graft/skills.py` (`website`, `slides`, `calendar`), so you can watch it clone for a task, hand the conversation to the clone, and join back when you change topic. Ask for several independent things at once and it can start them as background clones: it keeps talking to you, and tells you the outcome when they join.

To see a full clone/join cycle offline, run the scripted pivot experiment and inspect its store:

```bash
uv run python experiments/pivot.py --fake --quiet --db pivot.db
uv run graft dump pivot.db
```

## Configuration

Everything is read from environment variables at runtime. Keys are never read from files by the library; keep them out of the repository (`.env` is gitignored).

| Variable | Effect |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Provider keys. |
| `GRAFT_PROVIDER` | `openai` or `anthropic`. Default: `openai` if `OPENAI_API_KEY` is set, else `anthropic`. |
| `GRAFT_MODEL` | Model override. Defaults: OpenAI uses the first available of `gpt-5.4-mini`, `gpt-5-mini`, `gpt-4.1-mini`, `gpt-4o-mini`; Anthropic uses `claude-sonnet-5-5`. |
| `GRAFT_OPENAI_API` | `responses` (default) or `chat`. |
| `GRAFT_REASONING_EFFORT` | Global reasoning effort (default `low`). |
| `GRAFT_EFFORT_<SITE>`, `GRAFT_MODEL_<SITE>` | Per-call-site effort and model. Sites: `ORIGINAL`, `CLONE_CALL`, `CLONE`, `COMPACTION`, `SUBAGENT_PARENT`, `SUBAGENT` (see `graft/callsites.py`). |
| `GRAFT_PROMPT_CACHE_KEY` | `0` disables the OpenAI `prompt_cache_key`. |
| `GRAFT_SEEDER` | `jev` or `model`. Default: `jev` if `TYPESAFE_API_KEY` is set, else `model` (the parent names its own seed rows). |
| `TYPESAFE_API_KEY` | Key for the Jev seeder (needs the `jev` extra). |
| `GRAFT_SEED_THRESHOLD`, `GRAFT_SEED_TOKEN_BUDGET` | Jev seeder score threshold (default 0.4) and per-request token budget (default 24000). |
| `GRAFT_PRICE_IN_PER_M`, `GRAFT_PRICE_OUT_PER_M`, `GRAFT_PRICE_CACHED_IN_PER_M`, `GRAFT_PRICE_CACHE_WRITE_PER_M` | USD per 1M tokens, for cost reports. Per-model overrides use the suffix `__<MODEL>` (upper-cased, non-alphanumerics as `_`). |
| `GRAFT_CACHED_READ_MULT`, `GRAFT_CACHE_WRITE_MULT`, `GRAFT_OUTPUT_MULT` | Assumed multipliers for the "effective tokens" metric (defaults 0.1, 1.25, 4.0). |

## Layout

- `graft/db.py`: SQLite schema with append-only `turns` and `context` (enforced by triggers), context views, pointer append with dedup, rendering and validation.
- `graft/machine.py`: the [stateforward-hsm](https://pypi.org/project/stateforward-hsm/) state machines that own all coordination: each timeline's lifecycle (a run of the model loop is the activity of its state; turn, report, suspended, joining; background grafts and the join queue), and a session machine for `settle` and `close`. The harness (`graft/harness.py`) is their host and does only I/O.
- `graft/harness.py`: the machines' host: the agent loop (run as a machine state's activity), the clone call's seeding and validation (one or more grafts), the store writes and default join the machines ask for, and the user-facing entry points (`user_says`, `settle`, `close`).
- `graft/seeder.py`: seed selection (`model` or `jev`).
- `graft/compaction.py`: turn-based compaction with window provenance, and a summary compaction used by the subagent baseline.
- `graft/subagent.py`: the subagent baseline (parent writes a handoff, subagent returns a summary).
- `graft/providers.py`: OpenAI and Anthropic adapters with per-call cache accounting, and a scripted `FakeProvider`.
- `graft/skills.py`, `graft/sim.py`, `graft/fake.py`: simulated skills and world, a scripted user, and a deterministic offline policy.
- `graft/cli.py`: the `graft` command.
- `experiments/`: benchmarks and evaluations (not part of the installed package).

## Experiments

All experiments accept `--fake` (scripted provider, offline) unless noted; without it they call the configured provider. `parallel.py` is the reverse: offline unless `--live` is given.

| Script | What it measures |
|---|---|
| `experiments/pivot.py` | A user describes a website over several turns, then pivots to a slide show. Checks that the facts reach the slides clone verbatim and the join is deterministic. `--arm graft\|subagent\|both`. |
| `experiments/isolation.py` | The same slides task as a clone, inside a shared agent with conflicting skills loaded, and as a subagent. Scores compliance and style leakage. `-n 20 --parallel 8` by default. |
| `experiments/parallel.py` | One multi-part request as sequential clone-join (one focus clone per part) and as one clone call with background grafts. Reports world-state checks, follow-ups used (a bounded scripted reply when a clone asks a question), model calls and wall-clock. Offline by default (scripted provider with an artificial per-call delay, which shows the mechanics only); `--live` uses the configured provider. |
| `experiments/bench.py` | Five scripted scenarios (`pivot`, `revisions`, `failure`, `long_compaction`, `multi_part`) for the graft and subagent arms, with pass rate, tokens by cache bucket, cost, latency and significance tests. Example: `uv run python experiments/bench.py --scenario all --arm both -n 10 --fake`. |
| `experiments/seed_eval.py` | Offline seed-selection evaluation over stored bench runs (Jev, a model actor, embeddings, Needle 3). Needs stored runs and the relevant keys. |
| `experiments/api_probe.py` | Whether providers accept historical tool calls for undeclared tools, and spliced Anthropic thinking blocks. Skips providers without keys. |
| `experiments/replay/run.py` | Real-trace replay: re-runs recorded Claude Code subagent delegations as a subagent (S) and as a graft clone (G) against the repository checked out at the spawn-time commit, and compares both with the recorded subagent (R). Needs local transcripts, a checkout of the traced repository, [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`), `OPENAI_API_KEY` and `TYPESAFE_API_KEY`; not runnable offline. |

Every run records uncached input, cached reads, cache writes and output per call and per role. Costs are printed only when prices are set.

## Results so far

These are small samples. They are reported as measured, not as evidence that either approach wins.

- **Isolation** (`isolation.py`, gpt-6-luna): both the clone and the shared-context arm scored 20/20 with zero leakage; the clone arm used about 3x fewer input tokens. Isolation alone does not separate the approaches on a frontier model.
- **Seeder threshold** (`seed_eval.py`, 140 stored `long_compaction` events): the Jev seeder at threshold 0.4 kept full fact recall on 140/140 events (139 at 0.5, 136 at 0.7). Needle 3 zero-shot did not qualify: its confidence did not separate needed rows from the rest.
- **Real-trace replay** (`replay/run.py`, 14 delegation events from a public Claude Code session on the [mosfet](https://github.com/stateforward/mosfet) project; S and G both run on gpt-6-luna; R is the recorded Claude subagent):
  - probe accuracy (the parent answering questions about the delegated work): subagent 0.17, graft 0.13, recorded Claude subagent 0.60
  - on the Explore (read-only search) event: S 0.85, G 0.75, R 1.00
  - delegation cost: graft $0.64 vs subagent $0.26 in total; average latency: graft 295 s vs subagent 427 s
  - reading: the weaker delegate model dominates the outcome; there is no evidence here that graft is better or worse than a subagent. An earlier 5-event pilot on another project showed the same pattern.

## Limitations

- One level deep: only the original clones. Nesting (clones that clone) was implemented, evaluated and removed because it does not fit grafting; see spec: Depth.
- Parallel grafts are implemented but only their mechanics are tested, offline. There is no measurement yet of how a real model uses background clones, or of quality and speed against sequential clone-join.
- Background clones share external state without locking, and a running clone cannot be steered. The original can cancel a running background clone (`cancel`); a clone never can. A cancelled clone is stopped and joins what it had done, and nothing it did outside the conversation is undone; see spec: Parallel grafts.
- The chat skills are simulated; there is no plugin interface for real tools yet.
- The Jev seeder depends on a hosted TypeSafe API; without it the parent model picks its own seed.
- The benchmarks are scripted scenarios and a small number of replayed traces. Results may not transfer to other tasks or models.
- The replay counterfactual is approximate: the recorded handoff stands in for the user message that would have triggered the clone.

## Design notes

Implementation choices that go beyond, or differ from, the spec:

- **Rendering:** consecutive rows with the same role become one message. Validation runs on those messages: the view starts with a user turn, every `tool_use` has its `tool_result` in the next user turn, tool results come before text in a user turn, the view ends on a user turn, a trigger is the last row, and thinking rows stay in the timeline that produced them.
- **Clone is one call:** `clone(clone=[{purpose, skills, focus, seed_quotes, seed_row_ids}, ...])`, one entry per clone, at most one with `focus: true`. The harness resolves every seed to pointers, validates, and commits all entries or none; on violations the parent gets them back per entry. A run makes at most two clone calls (one, and one retry after a refusal); a further call is not executed and nothing is seeded for it. With the Jev seeder, the seed fields are only a fallback.
- **Coordination is the state machines':** a run of the model loop is the activity of its machine state (leaving the state ends it); a user message, a clone or cancel call, `settle` and `close` are events, answered by a reply future carried in the event or by a response event; a user message that arrives mid-run waits on the original's machine. The harness holds no lock, task or event of its own. Each model call runs in a daemon worker thread and hands its result back to the event loop; all store and world access is on the loop.
- **Background grafts:** entries with `focus: false` run concurrently, each as the activity of its machine's `background` state. Only the conversation holder's text is emitted. Every join is a queue entry on the original's machine, carrying its reason and trigger: a graft is `queued` when it raises its join event and `joined` only once the join is written, for focus and background grafts alike, so a failed commit is reported, loses nothing and is retried at the next commit point. Joins are applied only from rest, and the original is then woken once with a closing harness note. The machines own the lifecycle, the queue and what is allowed, including depth 1 and the cap on live background grafts; the harness keeps no copy.
- **Limits and failure handling:** at most 4 entries per clone call and 4 live background grafts per source. Any error inside clone handling is a refused tool result with the call's transactions rolled back; the state machine step and the store commit of a clone call are one unit, and a join is one SQL transaction. Re-clone blocks are per part (a harness-owned request and the graft's catalog skills), so a failed graft does not block a sibling's handed-back part; under them sits a cap of 4 committed clone calls per user message that reads no key. After a focus graft fails or hands back, its source always runs once and answers the user. A model call has a time limit (`Harness(call_timeout=...)`, 300 s by default, `None` to disable); a timed-out call fails the run it belongs to, so a graft joins `failed`. A join that cannot be applied is retried at up to 3 commit points and then abandoned. A cancelled turn leaves a valid view (interrupted tool calls get a "not executed" result). A failed wake run or join commit is shown to the user at once as a harness error line and raised by `Harness.settle()` (one error, or an `ExceptionGroup`). `Harness.settle()` asks the session machine to answer when nothing is in flight and returns what was said meanwhile; it raises the failures no user's call was there to report. `Harness.close()` shuts every running background graft down (it joins `failed`; one whose cancel is in flight joins `cancelled`), lets the original apply what it can without running anything (a graft made by a clone call that was already in flight is shut down the same way), and then stops every machine, the scribe included; a later request is refused, and `settle()` and a second `close()` return at once. A graft whose run raises joins `failed` without the model, focus or background. `cancel(cancel=[ids])` is the original's alone (the state machine refuses it from a graft): each named background graft that is still running is stopped (its run has ended before its join is queued) and joins with reason `cancelled`, a bad or finished id is a line in the tool result, a store error while the cancel is recorded is shown as a harness error line and never changes what the tool result says, the cancelled part is not re-cloned before the next user message, and its place under the cap of 4 is free at once. A response may carry `cancel` and `clone` together: the cancels run first, then the clone, so a part can be replaced in one response even at the cap; any other tool call beside a `clone` call is not executed.
- **A background graft always joins:** a text-only turn joins it `done`; an empty response, an error or an exhausted step budget joins it `failed`.
- **Default join is deterministic:** pointers to the text rows of the clone's working view that the parent does not already have; `out_of_scope` ends with the trigger, `failed` adds a harness error record, and `cancelled` (a graft the original stopped) adds a harness record naming the tools the graft had already run.
- **Harness rows and seeds:** rows the harness writes (join records, wake notes) carry `turns.harness = 1` and are told from user messages by that mark only. A seed may point only at rows the source holds in its own context. A store written before the mark existed is refused when opened.
- **No silent `done`:** `join(reason='done')` is refused unless the clone has replied since the last user message (a background graft: unless it has written its result).
- **Progressive disclosure:** a timeline's system prompt lists only the names and descriptions of its linked skills; `load_skill(name)` loads the instructions as ordinary tool rows.

## License

MIT. See [LICENSE](LICENSE).
