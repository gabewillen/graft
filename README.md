# graft

graft is an experimental alternative to subagents for delegating work inside an LLM agent. Instead of starting a fresh subagent from a handoff prompt the parent writes, the agent **clones** itself: the clone starts from the messages of the parent's real conversation the parent points at (by pointer, like a git branch; or the rows a decisions client picks) and the parent's own clone call, with its words for the clone, the clone does its part for one turn, and when that turn ends the conversation it had is **joined** back into the parent's memory by pointer. No model call is needed for the join. All state lives in SQLite as append-only turns plus per-agent pointer lists.

The design is written up in [spec.md](spec.md) ("Agent Grafts"). This repository is the smallest working version of it (v1, depth 1), plus the experiments used to compare it against a subagent baseline.

**Status:** experimental, pre-1.0. The APIs, schema and CLI may change. The experiments so far do not show graft to be better or worse than subagents (see [Results so far](#results-so-far)).

## How it differs from subagents

```text
subagent                                    graft
--------                                    -----
parent ── writes handoff prompt ──▶ worker  parent ── fork(current, detached) ──▶ clone
          (paraphrase of context)                     seed = pointers to the parent's own rows
                                                      it names (or a seeder's pick), then its
                                                      fork call with its reason for the clone
worker works in a fresh context             clone works in its own timeline
worker ── writes summary ──▶ parent         clone ── merge ──▶ parent (at its turn's end)
          (paraphrase of its work)                    pointers to the clone's conversation rows
                                                      and its merge call, appended
                                                      deterministically
```

- **Down:** a subagent sees only what the parent wrote for it. A clone sees verbatim rows from the parent's timeline, the ones the parent points at (or a seeder, a small decision model, picks), then the parent's own fork call, whose `reason` says the clone's part in the parent's words.
- **Up:** a subagent returns a summary in its own words. A clone returns nothing of its own making for someone else to read: the harness appends pointers to the text rows of its conversation and to its merge call (with its optional `reason`) to the parent's context, then the parent's result of that call.
- **Same:** both isolate the delegated work from the parent's loaded skills and tool history, and both cost extra calls.

**What the model is told.** "Clone" and "join" are graft's own words, for its harness and store. The model never sees them: its built-in tools are `fork(current={id?, skills, messages, reason?}, detached=[{...}, ...])` (the clone call: `messages` are short exact quotes or row ids of the messages the branch starts from, `reason` the agent's own words for the branch; a branch's `id` is optional, unique and never reused, and drawn at random when left out), `merge(reason?)` (the join; its optional reason is the branch's own words to main, not a join reason) and `stop(ids=[...])` (graft's cancel); the agent stays on **main**; a focus graft is a **current** branch, which takes the conversation (its HEAD, in git's terms) for one turn, and a background graft is a **detached** branch, which works off to the side; every branch merges when its turn ends; every prompt and tool result it reads says, in the first person, that a branch is itself carrying on with one part of the conversation with fewer skills, and that when a branch merges, what it did there is its own memory ("site merged. Its replies above already reached the user: ..."). Nothing it reads calls a branch a clone, an instance, a subagent or a helper, or says focus, background or cancel, so the user hears one "I" (spec: What the model is told: fork, merge and stop).

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

Inside `chat`, `/dump` prints the store and `/quit` exits. Each message is a task streamed through the host interface: replies go to stdout; status and harness error lines (waiting for background branches at quit, a failed report) go to stderr. Ctrl-C exits at once and cancels running background clones. Without a key, `chat` exits with an error naming the missing variable.

The chat agent uses the simulated skills in `graft/skills.py` (`website`, `slides`, `calendar`), so you can watch it clone for a task, hand the conversation to the clone for one turn, and join back when that turn ends. Ask for several independent things at once and it can start them as background clones: it keeps talking to you, and tells you the outcome when they join.

To see a full clone/join cycle offline, run the scripted pivot experiment and inspect its store:

```bash
uv run python experiments/pivot.py --fake --quiet --db pivot.db
uv run graft dump pivot.db
```

## Host interface (A2A)

graft speaks the [Agent2Agent protocol](https://a2a-protocol.org) (v1.0.1) to its host, in-process: `GraftAgent` has one method per A2A operation, taking and returning the A2A objects in their JSON forms (spec: Host Interface). A task is one user request and all the work it sets off, in every branch; branches never appear on the wire.

```python
import asyncio
from graft.agent import GraftAgent, GraftConfig
from graft.db import Store
from graft.fake import FakeWorld          # the simulated tools; a host passes its own Tools
from graft.providers import make_provider

async def main():
    world = FakeWorld()
    async with GraftAgent(Store("chat.db"), make_provider(), tools=world.tools(), config=GraftConfig()) as agent:
        task = (await agent.send({"message": {"messageId": "m1", "role": "ROLE_USER",
                                              "parts": [{"text": "Build the practice a website and a deck."}]}})).task
        print(task.status.state, task.status.message.text)   # INPUT_REQUIRED while branches work
        async for event in agent.subscribe({"id": task.id}):   # their reports, until COMPLETED
            print(event.to_dict())

asyncio.run(main())
```

`send` (SendMessage) returns when main's turn ends: `TASK_STATE_INPUT_REQUIRED` while background branches made for the task still work, `TASK_STATE_COMPLETED` (or `FAILED`) once none is out. `stream` (SendStreamingMessage) and `subscribe` (SubscribeToTask) yield the task's events until it is terminal; `cancel_task` (CancelTask) stops the request: its runs end, its branches stop and merge, and a current branch holding the conversation for it merges too. `get_task`, `list_tasks` and `card()` read the event log. Host tools are `Tool(name, input_schema, fn, description=...)`, an MCP tool definition and the function graft calls with the arguments.

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
| `GRAFT_SEEDER` | `jev` or `model`. Default: `jev` if `TYPESAFE_API_KEY` is set, else `model` (no seeder: a clone starts from the messages its fork entry points at). |
| `TYPESAFE_API_KEY` | Key for the Jev seeder (needs the `jev` extra). |
| `GRAFT_SEED_THRESHOLD`, `GRAFT_SEED_TOKEN_BUDGET` | Jev seeder score threshold (default 0.4) and per-request token budget (default 24000). |
| `GRAFT_PRICE_IN_PER_M`, `GRAFT_PRICE_OUT_PER_M`, `GRAFT_PRICE_CACHED_IN_PER_M`, `GRAFT_PRICE_CACHE_WRITE_PER_M` | USD per 1M tokens, for cost reports. Per-model overrides use the suffix `__<MODEL>` (upper-cased, non-alphanumerics as `_`). |
| `GRAFT_CACHED_READ_MULT`, `GRAFT_CACHE_WRITE_MULT`, `GRAFT_OUTPUT_MULT` | Assumed multipliers for the "effective tokens" metric (defaults 0.1, 1.25, 4.0). |

## Layout

- `graft/db.py`: SQLite schema with append-only `turns` and `context` (enforced by triggers), context views, pointer append with dedup, rendering and validation.
- `graft/machine.py`: the [stateforward-hsm](https://pypi.org/project/stateforward-hsm/) state machines that own all coordination: each timeline's lifecycle (a run of the model loop is the activity of its state; turn, report, suspended, joining; background grafts and the join queue), and a session machine that owns the host's tasks and the event log, and `close`. The host (`graft/harness.py`) does only I/O for them.
- `graft/agent.py`: `GraftAgent`, the host interface: the A2A operations in-process (see Host interface below).
- `graft/a2a.py`: the A2A objects (A2A v1.0.1 JSON forms) and errors.
- `graft/tools.py`: `Tool`, a host tool in MCP's shape, and `CallToolResult`.
- `graft/harness.py`: the machines' host (internal): the agent loop (run as a machine state's activity), the clone call's seeding and validation (one or more grafts), and the store writes, event-log writes and default join the machines ask for.
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
- Background clones share external state without locking, and a running clone cannot be steered. The original can cancel a running background clone (the model's `stop`); a clone never can. A cancelled clone is stopped and joins what it had done, and nothing it did outside the conversation is undone; see spec: Parallel grafts.
- The chat skills are simulated; there is no plugin interface for real tools yet.
- The Jev seeder depends on a hosted TypeSafe API; without it a clone starts from the messages the model points it at.
- The benchmarks are scripted scenarios and a small number of replayed traces. Results may not transfer to other tasks or models.
- The replay counterfactual is approximate: the recorded handoff stands in for the user message that would have triggered the clone.

## Design notes

Implementation choices that go beyond, or differ from, the spec:

- **Rendering:** consecutive rows with the same role become one message. Validation runs on those messages (a view may start with an assistant turn: a branch pointed at no messages starts on its parent's fork turn): every `tool_use` has its `tool_result` in the next user turn, tool results come before text in a user turn, the view ends on a user turn, a trigger is the last row, and thinking rows stay in the timeline that produced them.
- **Clone is one call:** `fork(current={id?, skills, messages, reason?}, detached=[{...}, ...])`: at most one focus clone (`current`) and any background clones (`detached`), at least one in all. An entry names its skills, the messages it starts from (quotes or row ids; empty only with a reason), optionally the parent's reason for it, and, optionally, its id (`^[a-z0-9][a-z0-9-]{0,31}$`, unique in the conversation, never reused; drawn as 6 random hex characters when left out, from `GraftConfig(names=...)`); the clone is told its id, and notes, results and `stop` use it, never the timeline id. Two entries with the same skills need ids. The harness resolves every entry's messages (each must name exactly one message the parent holds; a refusal lists the messages with their row ids), seeds it (those messages, or the Jev seeder's pick, with them as its fallback), puts the parent's fork turn and the clone's own fork result after the seed, validates, and commits all entries or none; on violations the parent gets them back per entry. A run may make any number of clone calls; a refused call returns its violations.
- **Coordination is the state machines':** a run of the model loop is the activity of its machine state (leaving the state ends it); a host's message, cancel and wait for events, a clone or cancel call, and `close` are events, answered by a reply future carried in the event or by a response event; a user message that arrives mid-run waits on the original's machine. The harness holds no lock, task or event of its own. Each model call runs in a daemon worker thread and hands its result back to the event loop; all store and world access is on the loop.
- **Background grafts:** `detached` entries run concurrently, each as the activity of its machine's `background` state. Only the conversation holder's text is told to the host. Every join is a queue entry on the original's machine, carrying its reason and trigger: a graft is `queued` when it raises its join event and `joined` only once the join is written, for focus and background grafts alike, so a failed commit is reported, loses nothing and is retried at the next commit point. Joins are applied only from rest, and the original is then woken once (every merge is news), on a view that ends on the merge results. The machines own the lifecycle, the queue and what is allowed, including depth 1; the harness keeps no copy.
- **No usage limits; failure handling:** graft does not bound how much an agent forks (entries per call, live detached branches, clone calls per run or per message, hand-backs, re-forks of a part that failed or was stopped): that is the host's to enforce (today with `close()`). Any error inside clone handling is a refused tool result with the call's transactions rolled back; the state machine step and the store commit of a clone call are one unit, and a join is one SQL transaction. A focus graft holds the conversation for one turn; after it merges, its source always runs once (it may end silently). A model call has a time limit (`GraftConfig(call_timeout=...)`, 300 s by default, `None` to disable); a timed-out call fails the run it belongs to, so a graft joins `failed`. A join that cannot be applied is retried at up to 3 commit points and then abandoned. A cancelled turn leaves a valid view (a tool call that never ran gets a "not executed" result; a host tool that was invoked and then cut short, raised or returned something malformed is recorded as run, with an error, so the model never redoes a side effect). A host tool's function may be `async def`: it is awaited as the activity of the graft's `calling` state while every other timeline goes on, with no time limit of graft's (the host owns its deadlines); `SubagentRunner` and `experiments/isolation.py` call tools synchronously. A failed wake run or join commit is a harness error event, told on every open task (and raised by the call that delivers it). `GraftAgent.close()` shuts every running background graft down (it joins `failed`; one whose cancel is in flight joins `cancelled`), lets the original apply what it can without running anything (a graft made by a clone call that was already in flight is shut down the same way), and then stops every machine, the scribe included; every task still open fails, a later request is refused, and a second `close()` returns at once. A graft whose run raises joins `failed` without the model, focus or background. `stop(ids=[ids])` (graft's cancel) is the original's alone (the state machine refuses it from a graft): each named background graft that is still running is stopped (its run has ended before its join is queued) and joins with reason `cancelled`, a bad or finished id is a line in the tool result, a store error while the cancel is recorded is shown as a harness error line and never changes what the tool result says. A response may carry `stop` and `fork` together: the cancels run first, then the clone, so a part can be replaced in one response; any other tool call beside a `fork` call is not executed.
- **Every graft's turn ends in its join:** a `merge` call ends it; a turn that ends without one merges implicitly (the harness writes an argument-less `merge()` call into the clone's timeline in the join's transaction). The join reason is graft's own and never shown: `done` (a merge or a reply; a focus graft that did not reply hands the request back, `out_of_scope`), `failed` (an error, an exhausted step budget, a background graft's empty response), `cancelled`.
- **Default join is deterministic:** pointers to the text rows of the clone's working view that the parent does not already have, then to its merge call, then the parent's merge result (which branch merged, whether the user already saw its replies, which branches are still running, for a stopped graft the tools it had run, and to go on or end the turn), then a trigger the parent lacks.
- **Harness rows and seeds:** rows the harness writes (merge results, fork results, implicit merge calls) carry `turns.harness = 1` and are told from user messages by that mark only. A seed holds only text rows the source holds in its own context: never its tool calls, tool results or harness notes. A store written before the mark existed is refused when opened.
- **Progressive disclosure:** a timeline's system prompt lists only the names and descriptions of its linked skills; `load_skill(name)` loads the instructions as ordinary tool rows.

## License

MIT. See [LICENSE](LICENSE).
