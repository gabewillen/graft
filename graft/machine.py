"""State machines (stateforward.hsm) for timelines, the session and the scribe: the one place where
graft decides what may happen, who waits, and in what order.

Three rules hold everywhere in this module.

* Decisions are guards and choices, and the guarded branches of a choice (or of one event's
  transitions from one state) are disjoint: no outcome depends on their order, and one unguarded
  default takes the rest. Effects never decide what happens next: an effect only changes its own
  instance's data, answers a reply future carried in the event that caused it, and dispatches at
  most one event (to another machine, the scribe, or its own). A step that must send more goes
  through states: a chain of choices, one send per transition, or a state whose entry ticks once
  and whose choice sends the next one (`reporting`, below).
* Anything that can fail is a state's activity: a run of the model loop, every store write, the
  report lines and trace the user's callbacks receive, starting a graft's machine. The activity
  ends by dispatching its outcome as an event; if it raises, the library dispatches `hsm/error`,
  and a transition of that state handles it. A run of the model loop types its own failure
  (`run_failed`), since states inside it (`preparing`, a step's writes) have activities and
  `hsm/error` transitions of their own: a clone call being prepared when the run fails is cut
  short and rolled back, and a step finishes first. So a failed write can never leave a machine stuck:
  the state that does the write is left by its outcome event or by its error event, and both are
  transitions. A guard, effect, entry or exit is not meant to fail, but hsm turns only an
  activity's failure into `hsm/error`: so each one is declared through `Guard`, `Effect`, `Call`,
  `Entry` and `Exit` below. Guards are pure reads, and the clone rules a gate decides on are
  worked out as data by the step before it; a guard that raises does not hold and writes nothing,
  and every decision's unguarded branch is the safe one (refuse, stop, keep, rest), whose effect
  checks that no guarded branch held (`checked`: a broken decision is a DecisionFault). An
  effect, entry or exit sends at most one event, made when it returns (its sends are its step's
  own, held per thread and task); one that raises sends none, and its failure is a fault of the
  machine (`_step`), its one event: once a step has failed no guard of it holds, no send of it
  reaches the session, the activities it enters do nothing (all three are the plumbing's: no
  effect, entry, exit or activity reads whether its step failed; a decision whose safe branch
  would act has a branch of its own for a failed step), and the machine's `fault` (or
  `call_fault`) transitions recover: what the step's event brought is kept (`_salvage`), what
  the step left open is closed out (if that write fails, it is kept and written first by the
  next writer of the view, `_close_out`), the callers waiting on the failed work (every reply the
  event carries, and the work's own) get the error once, the session's record is re-stated, and
  the machine rests. A behavior handling a fault that raises is part of that fault, which is
  handled again with its error added (`_handling_failed`; every handler is safe to run again)
  (spec: Internal faults). Fire-and-forget writes (the store copy of each state, report
  lines, trace) go to the scribe machine, which does them one at a time in its `writing` state.
* Machines talk only by events. A caller outside the machines (`user_says`, `settle`, `close`, a
  clone or cancel call made by a model loop) gets its answer from a future carried in the event's
  data, which an effect resolves.
* A user's call is a Request (id, text, reply) the original's machine owns from arrival until it
  is answered or withdrawn, wherever it is (inbox, `requesting`, the request a run answers, the
  waiters, handed to a focus graft with the conversation). `withdraw` (by id) is handled in every
  state: recorded while the request is held, or answered and its call not done (a cancel after
  the answer wins: the harness says when a call is done, `call_done`; the record is pruned as
  requests leave and calls end), a waiting
  message dropped and answered (one shared response, which the session hears of in the event
  the step sends anyway), forwarded to a focus graft (also one made in `cloning`, once its machine is started), and a
  run whose request is withdrawn ends (`_run_withdrawn`, the one rule). Every point that would
  start a run or let a graft act is guarded on its request being live (not withdrawn, reply not
  done): `requesting` -> busy, the next inbox message, the wake / hand-back / routing runs,
  `noting` -> busy, the clone gate, a focus graft's start (one whose request was withdrawn before
  it could act marks its row in `withdrawing` and rests), and a background graft's start and
  each of its steps (a background graft made for a request withdrawn never runs, or stops at its
  next step, and joins `cancelled`; it asks its own machine and then its source, `graft_step`,
  which knows a withdrawal after the answer too; that join wakes nothing). Whether a run
  may go on is decided here too: the run asks (`may_continue`, a reply in the event) before every
  step that could act, and is answered while the request it answers, if any, is live; once it is
  withdrawn, `busy` is left the way a withdrawal leaves it, which ends the run where it waits. A
  withdrawn request whose row was written and that no run answered gets the withdrawal mark on
  that row (`dropping`, `withdrawing`) in the views of the conversation it was asked in (the
  timeline and, for a focus graft, its source), so no later turn, hand-back or seed there reads
  it as a request. Nothing outside reads a machine's data: what the host needs to know (a
  timeline's kind, the grafts still running, whether clones are background grafts) is passed, by
  keyword, on the host call an activity makes.

A timeline that can hold the conversation (the original; a focus graft, which starts in `busy`
unless the request it answers was withdrawn before it could act: then in `withdrawing`, which
marks that request's row, and then at rest):

    live
      idle            at rest: holds the conversation, nothing in flight, nothing queued
      engaged
        requesting    writing a user message's row (activity); then busy if it is still live, else
                      `dropping` (activity: the row gets the withdrawal mark), then rest
        busy          running, in one of four modes (turn, route, report, external); `may_continue`
                      is answered while the request the run answers, if any, is live, else busy is
                      left for `closing_calls`
          run         a run of the model loop: its activity
          driven      a turn its caller drives (tests, replay): no activity
            running
            clone_gate  (choice) may the clone call be tried at all; decided before seeding
            preparing   the harness seeds and validates the call (activity; leaving the run ends it)
            prepared_gate, clone_choice, handed, cancels_done, resume   choices (cancels whose rows
                        failed end the run with that error: a clone call with them is not run; a
                        run whose request was withdrawn during a step ends after it)
            leaving the run (exit) answers any clone or cancel request it still holds
            steps       one clone or cancel request (shutdown and a driven end wait):
              cloning: rejecting, or committing / spawning / recording, or refusing (unspawning
                        first if a machine could not start)   a clone call, step by step (a
                        withdrawal waits while `spawning` starts the grafts' machines); a call
                        that hands over writes the run's open calls and kept rows (`flushing`,
                        then the choice `handed`) before the original is suspended; a graft's join
                        that comes meanwhile waits for the step
              cancels: checking / waiting / writing / reporting   cancel calls, id by id; then
                        their result rows, and each row that could not be written told
        ran           (choice) how the run ended (a RunOutcome): a focus graft that joined or spent
                      its step budget hands back (`handing_back`, activity: its join event)
        unpreparing   the run was left while a clone call was prepared: the call is rolled back
                      (activity; then `unwound`, a choice: a driven turn rests, others close calls)
        recovering    a step failed: its open audit rolled back, the run's open calls closed, rows to
                      mark marked (activity); its callers get the error, the session its record;
                      then `resumed` (choice): a focus graft holds the conversation -> the hand-over
                      finishes (`passing`), else `rest`
        closing_out   closing with a close-out a recovery could not write: it is written first
                      (activity), then the timeline closes
        closing_calls the run ended early: its open tool calls get results (activity); then
                      `withdrawn_choice`: if that run answered a request that was withdrawn,
        withdrawing   the row of the withdrawn request it answers gets the withdrawal mark in its
                      view and, for a focus graft, its source's (activity); also entered at a focus
                      graft's start when that request was withdrawn before it could act, and after
                      the joins (`after`) and the wake note (`noting`) when a focus graft handed back
                      its request unanswered and it was withdrawn
        failing       (a focus graft) its run raised: it joins `failed` (activity; `unrecorded`,
                      a choice, if that failed)
        joining       applying the joins on its queue, one per step
          next          (choice) a join can be applied -> applying; else -> after
          applying      one join (activity); a failure is kept for the next commit point, or,
                        on its last attempt, `giving_up` -> `abandoning` (choices) tell the user
                        and abandon it
        noting        writing a wake note (activity); then busy, or `withdrawing` if the request the
                      wake would answer was withdrawn (`unnoted`, a choice, if it failed)
        rest, after   choices: what follows a run, and what follows the joins
      receiving, shutting   choices: a graft's join that wakes it; closing the original
      passing         (choice) a hand-over: the waiting messages go to the focus graft first
      cloned          suspended while its focus graft holds the conversation (a message its frozen
                      focus graft passes back, `message_returned`, is kept: never forwarded again)
      releasing       closed while suspended: its focus graft is shut down first, and a close-out
                      a recovery could not write is written (activity)
    queued, joined, abandoned, closed   (a fault here: its record re-stated, its join sent if it was
                      not; queued passes a user message back to its source once; a fault that
                      ends a step in a final state other than joined or abandoned has closed)

A background graft: `background` (its run, or `held` when a caller drives the turn that made it;
never entered if the request it was made for was withdrawn before it was made)
--finish (`ending`, a choice on how the run ended) | cancel | shutdown | join (forced) | error
| a fault (reason failed) | `may_continue` once the request it was made for is withdrawn (reason
cancelled)-->
`recording` (`writing`, an activity: its open
calls and its join reason; then each failure told, one per step, in `reporting`) --> `queued`
--join commit | rollback--> `joined`, --join abandon--> `abandoned`.

The session machine knows whether anything is in flight (the original or a running focus graft
engaged, a background graft whose join has not reached it) and answers `settle` when nothing is;
`close` is `shutdown`. It keeps the errors `settle` raises, each once, unless a user's call was in
flight when the line telling it went out (the harness says when each call begins and is done,
`call_began` and `call_done`): that call's output shows it. A timeline that recovers from a fault
re-states its whole record (`resync`, a Record); a fault of the session's own steps fails the
settles waiting and asks the original to (`restate`): suspended, the original passes the question
on to its focus graft with its own record, and the graft re-states both in one event. At the end of `close` it stops every machine (`stopping`); one that cannot be
stopped is reported (`telling`) and raised by `close`, and the rest are still stopped.

Rules the graph enforces. Only the original clones, only in a turn or routing wake, never while
closing, at most CLONE_TRIES clone calls per run (one, and one retry; a committed call is the
run's last; a call past them is not prepared), never more than one focus graft per call, never beyond MAX_LIVE_BACKGROUND live background grafts, never
more than MAX_CLONE_CALLS_PER_MESSAGE committed clone calls per user message, never for a request
handed back MAX_REQUEST_HANDBACKS times, and never for a part of the current request that failed,
came back twice, or was cancelled (the re-clone blocks are the original's own data). A clone call
that arrives while cancel calls of the same response are being carried out waits for them (it is
deferred). Only the original cancels, only while it runs, and only a background graft of its own
whose join has not been raised. Joins are applied only from rest, one per step.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Protocol

import hsm

MAX_LIVE_BACKGROUND = 4  # live (running or queued, not cancelled) background grafts per source
CANCELLED = "cancelled"  # the join reason of a cancelled graft: set by the harness only
# Times a join's commit is tried, at successive commit points, before it is given up.
MAX_JOIN_ATTEMPTS = 3
# Clone calls a source may commit in wake runs between two user messages.
MAX_WAKE_CLONE_CALLS = 1
# Joins that may hand one request back to a timeline before it stops cloning for that request.
MAX_REQUEST_HANDBACKS = 8
# Clone calls committed for one user message, across every hand-back and wake it sets off.
MAX_CLONE_CALLS_PER_MESSAGE = 4
# Clone calls a run may make: the first, and one retry after a refusal. A committed call uses
# them all (one committed clone call per run).
CLONE_TRIES = 2


HEAD = 0  # the first item of a list: queues and work lists are taken from their head
NONE_YET = 0  # a count not started: no attempt, no hand-back, no arrival so far
STATE_HEAD = 2  # a state path split at its first two separators: '', the model, and the rest


class TransitionRefused(Exception):
    pass


class Unwritten(str):
    """A clone call's refusal by a timeline that is not running (its audit and result row are the
    caller's to write: no step of this machine wrote them)."""


class QueuedJoin(NamedTuple):
    """A join event waiting on its source: who, why, and the user message it hands back."""
    child: str
    reason: str
    trigger: int | None


class Mode(NamedTuple):
    """How a run of the model loop goes. `wake`: started for joins (counts against the wake clone
    calls); `report`: it may only report; `user`: a user's call waits on it, so its error is that
    call's error; `about`: the grafts whose joins a wake run is for."""
    name: str
    wake: bool = False
    report: bool = False
    user: bool = False
    about: tuple[str, ...] = ()


TURN = Mode(name="turn", user=True)


class RunOutcome(NamedTuple):
    """How a run of the model loop ended, as its activity reports it; each graph routes it.
    `kind`: replied (a reply with no tool call), empty (a response with nothing in it), join (the
    model called join, with `reason`), exhausted (the step budget ran out). A run whose request
    is withdrawn does not report an outcome: its machine leaves the run's state, which ends it."""
    kind: str
    reason: str | None = None


class Prepared(NamedTuple):
    """A clone call the harness has seeded and validated: [(graft id, focus)], each entry's (index
    in the model's list, catalog skills), and the harness's record of the batch."""
    pairs: list[tuple[str, bool]]
    parts: list[tuple[int, frozenset[str]]]
    data: Any


@dataclass(eq=False)
class Request:
    """One user's call (`user_says`): its id, its text, and the future that carries its answer.
    The original's machine holds it from arrival until it is answered or withdrawn, wherever it
    is (the inbox, the row being written, the run answering it, the waiters, a focus graft it was
    handed to), and, once answered, until its call is done (`call_done`: a cancel that lands
    before the caller takes the answer withdraws it too). The caller withdraws it by id
    (`withdraw`); a request is live while it is not withdrawn and its reply is not done. `row`: its user row once written (a withdrawn request
    whose row was written gets the withdrawal mark on it, so no later turn reads it as a live
    request)."""
    id: int
    text: str
    reply: asyncio.Future
    row: int | None = None


class GraftPlan(NamedTuple):
    """A graft of a committed clone call, to be made in `spawning`: a focus graft is made with
    the user's calls waiting on the run and the request it answers (`answering`); a background
    graft with the request the run that made it answers (`made_for`), which it stops for if that
    request is withdrawn."""
    child: str
    focus: bool
    waiters: list["Waiter"]
    answering: "Request | None"
    made_for: "Request | None"


@dataclass
class Waiter:
    """A caller waiting for a timeline to come to rest: a user's call (it follows the
    conversation; `request` is that call) or a run started by `run` (answered with how the run
    ended)."""
    reply: asyncio.Future
    user: bool
    error: BaseException | None = None
    request: Request | None = None


@dataclass
class Clone:
    """A clone call put to the original's machine as the model made it. The machine runs the
    cancels carried with it first, then decides whether the call may be tried at all, has the
    harness prepare it (seed and validate: an activity), and decides and commits it. `use`: the
    model's tool call; `store`: the machine writes the call's result row; `request`: the
    original's current request row (a store read the turn's run made). `pairs`, `parts` and `data`
    are filled when the call is prepared (see Prepared)."""
    use: dict
    reply: asyncio.Future
    store: bool = False
    request: int | None = None
    cancels: "Cancel | None" = None  # cancel calls of the same response: the machine runs them first
    pairs: list[tuple[str, bool]] = field(default_factory=list)
    parts: list[tuple[int, frozenset[str]]] = field(default_factory=list)
    data: Any = None


@dataclass
class CancelCall:
    use: dict
    ids: list[str]
    problems: list[str]


@dataclass
class Cancel:
    calls: list[CancelCall]
    reply: asyncio.Future
    request: int | None = None


class Host(Protocol):
    """What the machines need from the harness: I/O only, always called from an activity (or the
    scribe's), always by keyword. Whatever the host needs to know about a timeline (its kind, the
    grafts still running) comes in the call: the host never reads a machine."""
    def run_busy(self, tl: str, mode: Mode, kind: str) -> Any: ...
    def run_background(self, tl: str) -> Any: ...
    def mirror(self, tl: str, status: str) -> None: ...
    def report(self, kind: str, src: str | None, clones: list[str], error: BaseException, quiet: bool,
               background: bool) -> None: ...
    def trace(self, text: str) -> None: ...
    def graft_failed(self, tl: str, error: BaseException) -> None: ...
    def begin_request(self, tl: str, text: str) -> int: ...
    def mark_withdrawn(self, timelines: list[str], rows: list[int]) -> None: ...
    async def prepare_clone(self, src: str, clone: Clone, retry_left: bool) -> "Prepared | str": ...
    def roll_back_clone(self, src: str) -> None: ...
    def commit_clone(self, src: str, clone: Clone) -> str: ...
    def refuse_clone(self, src: str, clone: Clone, why: str, error: BaseException | None) -> str: ...
    def store_result(self, tl: str, use: dict, content: str, is_error: bool) -> None: ...
    def close_tool_uses(self, tl: str, known: dict[str, tuple[str, bool]]) -> None: ...
    def record_join(self, child: str, reason: str) -> None: ...
    def apply_join(self, src: str, join: QueuedJoin, cloned_for: int | None, running: list[str]
                   ) -> tuple[bool, int | None, bool]: ...
    def wake_note(self, src: str, joined: list[tuple[str, str]], routing: bool, running: list[str],
                  background: bool) -> None: ...
    def cancel_text(self, call: CancelCall, outcomes: dict[str, str]) -> tuple[str, bool]: ...
    def join_of(self, tl: str, reason: str) -> QueuedJoin: ...
    def abandon_clone(self, src: str, grafts: list[str]) -> None: ...
    def recover(self, tl: str, known: dict[str, tuple[str, bool]], close_calls: bool, viewers: list[str],
                rows: list[int], cancels: list[tuple[CancelCall, dict[str, str]]]) -> None: ...


# ------------------------------------------------------------------ instances
class TimelineSM(hsm.Instance):
    def __init__(self, machines: "Machines", timeline_id: str, source_id: str | None, kind: str,
                 held: bool = False, waiters: list[Waiter] | None = None,
                 answering: Request | None = None, made_for: Request | None = None,
                 withdrawn: set[int] | None = None):
        super().__init__()
        self.machines = machines
        self.host = machines.host
        self.timeline_id = timeline_id
        self.source_id = source_id
        self.kind = kind  # original | focus | background: fixed at creation
        self.held = held  # made in a turn its caller drives: it does not run on its own
        self.queue: list[QueuedJoin] = []  # joins waiting on this timeline, in finish order
        self.grafts: dict[str, bool] = {}  # its grafts whose join has not been applied: id -> focus
        self.parts: dict[str, frozenset[str]] = {}  # graft -> its catalog skills (its part)
        self.made: dict[str, bool] = {}  # every graft it made -> focus (kept after its join)
        self.cloned_for: dict[str, int | None] = {}  # graft -> the request it was made for
        self.mode = TURN  # the mode of its current or next run
        self.inbox: list[Request] = []  # user messages not yet begun, in arrival order
        self.request: Request | None = None  # the message whose row is being written
        self.waiters: list[Waiter] = list(waiters or [])
        self.watchers: list[asyncio.Future] = []  # background: callers waiting for its join event
        # The user request its current or next run answers (a focus graft: its source's), kept
        # through the joins and the wake that follow a turn, until it is at rest. A run answers at
        # most one: its turn's message, the one its source answered when it was made, or the one
        # a focus graft handed back.
        self.answering: Request | None = answering
        self.answered = False  # a run has answered `answering` (its turn, or the focus graft's)
        # Ids of withdrawn requests it holds (and, for the original, of requests its background
        # grafts were made for, until their joins are applied): pruned as requests leave it.
        self.withdrawn: set[int] = set(withdrawn or ())
        # User requests it answered (at rest) whose calls are not done yet (`call_done`): a cancel
        # that lands after the answer and before its caller took it is a withdrawal (cancel wins).
        self.returned: list[Request] = []
        # The requests the background grafts whose joins the next wake reports were made for: a
        # wake for none but withdrawn ones does not run, or ends (a cancel after the answer wins).
        self.woken_for: list[Request] = []
        # A background graft: the request the run that made it answers. If that request is
        # withdrawn (by its caller: not answered), the graft stops at its next step and joins
        # `cancelled`; it never starts if it was withdrawn before the graft was made.
        self.made_for: Request | None = made_for
        # The original: each background graft made for a request -> that request, until the
        # graft's join is applied. A join of a graft made for a withdrawn request wakes nothing.
        self.graft_requests: dict[str, Request] = {}
        self.focus: str | None = None
        self.closing = False  # nothing is woken from now on
        self.final = False  # shut down: once nothing is left to apply, close
        self.hold = False  # a join commit just failed: retried at the next commit point
        self.attempts: dict[str, int] = {}  # graft -> failed attempts to commit its join here
        self.news: list[tuple[str, str]] = []  # joins applied in this drain that the source must act on
        self.ends_on_request = False  # the last join applied left a handed-back message last
        self.clone: Clone | None = None  # the clone call being carried out
        # The clone rules' outcome for that call, worked out by the step before each gate (its
        # guards read it): why the whole call is declined, and why the prepared call is refused,
        # each with the error, if working it out raised.
        self.declined: tuple[str | None, BaseException | None] = (None, None)
        self.refused: tuple[str | None, BaseException | None] = (None, None)
        self.clone_text = ""
        self.spawn_plan: list[GraftPlan] = []
        self.rejection = ""  # the result text of a clone call refused before or in its preparation
        self.handing: Clone | None = None  # a clone call that hands the conversation over, being finished
        self.ended = ""  # the final state it reached: joined | abandoned | active (closed)
        self.cancel: Cancel | None = None  # the cancel calls being carried out
        self.to_check: list[str] = []  # ids of those calls not yet decided
        self.pending: set[str] = set()  # grafts they cancelled whose joins are not queued yet
        # Grafts a cancel stopped whose joins had not come when a recovery ended the cancel step:
        # no longer live (they do not count as running), but their joins are still its own.
        self.stopping: set[str] = set()
        self.outcomes: dict[str, str] = {}
        self.claims: dict[str, int] = {}  # graft -> the call of the response that names it first
        self.unstored: dict[str, tuple[str, bool]] = {}  # tool_use id -> a result not yet written
        # Cancel calls of a step a recovery ended after it had stopped grafts: each call with what
        # became of its ids, for the recovery to write its real result (it ran: never "not run").
        self.cut_cancels: list[tuple[CancelCall, dict[str, str]]] = []
        self.reported: BaseException | None = None  # an error already shown, that ends the run
        self.blocks: set[tuple[int | None, frozenset[str]]] = set()  # re-clone blocks (per part)
        self.arrivals: dict[tuple[int | None, frozenset[str]], int] = {}
        self.handbacks: dict[int | None, int] = {}
        self.clone_calls = 0  # committed since the last user message
        self.wake_clones = 0  # clone calls committed in wake runs since the last user message
        self.reason = ""  # background: its join reason, while it is recorded
        self.close_first = False  # background: its run ended early (its open calls need results)
        self.forced: asyncio.Future | None = None
        self.claims_next: list[tuple[int, str]] = []
        self.clone_tries = 0  # clone calls tried in the current run (see CLONE_TRIES)
        self.spawned_ids: list[str] = []  # grafts of a clone call whose machines must be undone
        self.spawn_error: BaseException | None = None
        # failures to tell the user, one per step: (kind, clones, error, the source it is told from)
        self.unreported: list[tuple[str, list[str], BaseException, str | None]] = []
        self.held_grafts: set[str] = set()  # the original: background grafts made in a driven turn
        # Rows of withdrawn requests still to be marked (a mark write that failed is retried by
        # recovery, and before the next row or wake note is written).
        self.unmarked: list[int] = []
        self.handed_back: tuple | None = None  # a focus graft: its join event, until it is sent
        self.hand_back_sent = False  # a focus graft: its join event reached its source
        self.join_sent = False  # a background graft: its join event reached its source
        self.strays: list[QueuedJoin] = []  # joins that came in a step that failed, for the recovery
        self.closed = False  # closed: its last record is `root_closed`
        self.fault: Fault | None = None  # a step that failed, until its recovery begins
        self.fault_sent = False
        self.held_back: str | None = None  # the activity a failed step held back
        self.recovery: list[BaseException] = []  # the errors of the fault being recovered from
        self.recoveries = 0  # recoveries since the last user message or run began
        self.faulted: hsm.Event | None = None  # the event whose step failed
        self.handling: Fault | None = None  # the fault its recovery (`recovering`) handles
        # A close-out a recovery could not write (`host.recover`'s arguments): written before
        # anything next writes to or reads its view (`_close_out`).
        self.close_out: list[dict[str, Any]] = []


class SessionSM(hsm.Instance):
    def __init__(self, machines: "Machines"):
        super().__init__()
        self.machines = machines
        self.engaged: set[str] = set()
        self.grafts: set[str] = set()
        self.held: set[str] = set()
        self.settles: list[asyncio.Future] = []
        # Errors kept for `settle`, each with the user calls in flight when its line was out: a
        # call that then returns its lines has shown it (it is that call's), so it is no longer kept.
        self.errors: list[tuple[BaseException, frozenset[int]]] = []
        self.calls: set[int] = set()  # user calls in flight (`call_began` .. `call_done`), by request id
        self.closers: list[asyncio.Future] = []
        self.root: str | None = None
        self.to_shut: list[str] = []
        self.stop_errors: list[BaseException] = []  # machines that could not be stopped at close
        self.closing = False
        self.fault: Fault | None = None
        self.fault_sent = False
        self.held_back: str | None = None


class ScribeSM(hsm.Instance):
    def __init__(self, machines: "Machines"):
        super().__init__()
        self.host = machines.host
        self.machines = machines
        self.note: tuple = ((), None)
        self.fault: Fault | None = None
        self.fault_sent = False
        self.held_back: str | None = None


# ------------------------------------------------------------------ plumbing
def _resolve(reply: asyncio.Future | None, value: Any = None) -> None:
    if reply is not None and not reply.done():
        reply.set_result(value)


def _fail(reply: asyncio.Future | None, error: BaseException) -> None:
    if reply is not None and not reply.done():
        reply.set_exception(error)


# The sends of the behavior being run (`_step`), made when it returns (none if it raises). Each
# step sets its own list here and resets it when it returns, so the list belongs to that step:
# a context variable is per thread and per task, and a list is never shared as a default, so a
# harness on another thread (or a step of another machine) never sends through it.
_SENDS: contextvars.ContextVar[list[tuple[hsm.Instance, hsm.Event]] | None] = contextvars.ContextVar(
    "graft_step_sends", default=None)


def _send(target: hsm.Instance, event: hsm.Event) -> None:
    sends = _SENDS.get()
    if sends is not None:
        sends.append((target, event))
    else:  # an activity's send
        hsm.Dispatch(None, target, event)


def _forward(target: hsm.Instance | None, name: str, data: Any = None) -> None:
    if target is not None:
        _send(target=target, event=hsm.Event(name=name, data=data))


def _self(inst: hsm.Instance, name: str, data: Any = None) -> None:
    _send(target=inst, event=hsm.Event(name=name, data=data))


def _session(inst: TimelineSM, name: str, data: Any = None) -> None:
    """An event that keeps the session's record of this timeline. The plumbing does not send it in
    a step that failed (`_step`): the session would see part of the step, and the recovery
    re-states the whole record."""
    _forward(target=inst.machines.session, name=name, data=data)


def _note(inst, method: str, keep: BaseException | None = None, **args: Any) -> None:
    """A fire-and-forget write (store copy of a state, a report line, trace): the scribe calls the
    host's `method` with `args`, by name. `keep`: an error the session keeps for `settle` once the
    line is out (or failed to go out)."""
    _forward(target=inst.machines.scribe, name="note", data=((method, args), keep))


# ------------------------------------------------------------------ internal faults
FAULT = "fault"  # a step of this machine failed (`Fault`): handled in every state of every graph
# A step that only took a call from outside (an internal transition: no state left, nothing
# decided) failed: recovered from by answering that call and re-stating the record, nothing more.
CALL_FAULT = "call_fault"
FAULTS = (FAULT, CALL_FAULT)


class MultipleSends(RuntimeError):
    """A behavior tried to send more than one event (each sends at most one)."""


class DecisionFault(RuntimeError):
    """A decision's unguarded branch was taken while one of its guarded branches held: a guard
    raised (a guard writes nothing, so its branch's safe default reports it)."""


@dataclass(eq=False)
class Fault:
    """A step that failed: an effect, entry or exit raised (a bug in graft's own code; sends
    between machines never raise). `event`: the event whose step failed; `errors`: what it
    raised, the first one first (one failure is one fault, however many behaviors of the step
    raised it); `call`: the step only took a call (see CALL_FAULT)."""
    event: hsm.Event
    errors: list[BaseException]
    call: bool = False
    retries: int = NONE_YET  # times it was handled again because a behavior handling it raised
    told: bool = False  # its first error was given to a caller (that caller reports it)

    @property
    def error(self) -> BaseException:
        return self.errors[HEAD]


def _keep_fault(inst: hsm.Instance, event: hsm.Event, error: BaseException, call: bool) -> None:
    """The machine's own record of a failed step (written by the effect, entry or exit that raised,
    never by a guard): one fault per step, every distinct error in it."""
    if inst.fault is None:
        inst.fault = Fault(event=event, errors=[error], call=call)
    elif all(error is not e for e in inst.fault.errors):
        inst.fault.errors.append(error)
        inst.fault.call = inst.fault.call and call


# Times one fault is handled again because a behavior handling it raised, before that raise is
# only reported (a bug in the handling itself that raises every time must not spin).
MAX_FAULT_RETRIES = 2


def _handling_failed(inst: hsm.Instance, event: hsm.Event, error: BaseException,
                     sends: list[tuple[hsm.Instance, hsm.Event]]) -> None:
    """A behavior of a step that handles a fault raised (a `fault` or `call_fault` step, or a step
    that ends a timeline's recovery). That is never a fault of its own: it is part of the fault
    being handled, which is handled again, from where the machine now is, with this error added
    (a new Fault with the same event, so the same callers, and every error so far; the step's
    other behaviors see a step that failed). Every handler is safe to run again: a caller already
    answered is not answered twice, and what was kept is kept once, so the first fault's callers
    get its error once. Past MAX_FAULT_RETRIES (a handler that raises every time) the raise is
    reported instead, and the fault's own callers get its error: a behavior that was re-stating
    its record sends that record, the error added to what the session keeps and shows; otherwise
    it is reported, its one event, and kept for `settle` (the scribe's own goes to the session).
    Lost only if that report cannot be sent either."""
    fault = inst.fault or (event.data if event.name in FAULTS else inst.handling) or Fault(event=event, errors=[])
    errors = list(fault.errors) + ([] if any(error is e for e in fault.errors) else [error])
    if fault.retries < MAX_FAULT_RETRIES:
        inst.fault = Fault(event=fault.event, errors=errors, call=fault.call, retries=fault.retries + 1,
                           told=fault.told)
        inst.fault_sent = False
        return
    _fail_all(replies=_caller_of(event=fault.event), error=fault.error)
    records = [(t, e) for t, e in sends if e.name == "resync"]
    if records:
        target, event = records[HEAD]
        record = event.data._replace(errors=(*event.data.errors, error))
        hsm.Dispatch(None, target, hsm.Event(name="resync", data=record))
    elif inst is inst.machines.scribe:
        hsm.Dispatch(None, inst.machines.session, hsm.Event(name="failure", data=((error, False),)))
    else:
        hsm.Dispatch(None, inst.machines.scribe, hsm.Event(name="note", data=(("report", dict(
            kind="machine", src=None, clones=[], error=error, quiet=False, background=False)), error)))


def _signal(inst: hsm.Instance) -> None:
    """The step's fault goes to its own machine as one event (`fault`, or `call_fault`, sent once;
    the error kind, so it is the next event the machine takes: no other event is taken between the
    failed step and its recovery, but `spawning` defers it until the grafts' machines have
    started, and that outcome's step, taken while the fault is pending, does nothing more)."""
    if inst.fault is not None and not inst.fault_sent:
        inst.fault_sent = True
        name = CALL_FAULT if inst.fault.call else FAULT
        hsm.Dispatch(None, inst, hsm.Event(name=name, data=inst.fault, kind=hsm.ErrorEventKind))


def _failed(inst: hsm.Instance, event: hsm.Event | None) -> bool:
    """The step in progress has failed: a fault is pending that is not the one the step handles (a
    `fault` step's own fault is pending while its transition is chosen). The one definition of a
    failed step, read by the plumbing (no guard holds, the re-check of a decision's safe branch
    reads them so, no send of the step reaches the session, every activity it enters holds back)
    and by guards (`step_intact`), never by an effect, entry, exit or activity."""
    return inst.fault is not None and inst.fault is not getattr(event, "data", None)


def _held_back(inst: hsm.Instance, event: hsm.Event | None, activity: str) -> bool:
    """A state entered by a step that failed does nothing (its activity returns at once, before any
    I/O): the machine's recovery, the next event it takes, decides what is done instead. The
    machine records which activity held back (a recovery that resumes a write runs it again)."""
    if not _failed(inst=inst, event=event):
        return False
    inst.held_back = activity
    _signal(inst)
    return True


def held_back_in(activity: str) -> Callable:
    """A guard of a `fault` transition: the failed step entered this state and its activity held
    back, so the recovery enters the state again (the activity then runs)."""
    def guard(ctx, inst, event) -> bool:
        return inst.held_back == activity
    guard.__name__ = f"held_back_in_{activity}"
    return guard


def _take_fault(inst: hsm.Instance, event: hsm.Event) -> Fault:
    """The fault a `fault` step carries is the machine's to handle now (a later step that fails is
    a fault of its own). Taken by the plumbing before any behavior of the step runs, so a handler
    that raises at once leaves nothing pending."""
    if inst.fault is event.data:
        inst.fault, inst.fault_sent, inst.held_back = None, False, None
    return event.data


def _pending(inst: hsm.Instance) -> BaseException | None:
    """The error of a step that failed, for the text of the safe branch it took (None if none)."""
    return inst.fault.error if inst.fault is not None else None


def _rule(behavior: Callable) -> Callable:
    """A guard or behavior defined at module level is read through the module when it is run, so
    each is one definition wherever the graphs use it (and a test can stand in for it)."""
    return globals().get(behavior.__name__, behavior)


def _guard(expression: Callable) -> Callable:
    """A guard of these graphs: a pure predicate over the machine's data, the event's payload and
    the state of the reply futures it carries (read at decision time: a caller cancels its reply
    at any moment). It reads, and writes nothing. Once a step has failed (an effect, entry or exit
    raised) no guard of it holds, so every decision the step still makes takes its unguarded
    branch, the safe one. A guard that raises does not hold: it records nothing, and the
    decision's unguarded branch, which checks that no guarded branch held (`checked`), reports a
    decision the raise changed."""
    @functools.wraps(expression)
    def guard(ctx, inst, event):
        if _failed(inst=inst, event=event):
            return False
        try:
            return bool(_rule(expression)(ctx=ctx, inst=inst, event=event))
        except Exception:  # noqa: BLE001 - fails closed; the unguarded branch reports it
            return False
    return guard


def _holds(guard: Callable, ctx, inst: hsm.Instance, event: hsm.Event) -> bool:
    """A decision's guard read again by its safe branch (`checked`): as the decision read it (in a
    step that failed none holds), but a raise is not swallowed: it is the step's fault."""
    return not _failed(inst=inst, event=event) and bool(_rule(guard)(ctx=ctx, inst=inst, event=event))


def _step(behavior: Callable, call: bool = False, handles: bool = False) -> Callable:
    """An effect, entry or exit of these graphs. hsm 1.3.11 turns only an activity's failure into
    `hsm/error`; one of these that raised would leave its machine half way through a step. So each
    is run here: its sends are made when it returns (at most one: a second is a fault), and if it
    raises, none is made and its failure becomes a fault of the machine (`_keep_fault`), sent to the
    machine as its one event (`_signal`); the step completes. Once a step has failed, no send of it
    reaches the session (it would see part of the step), and every activity it enters holds back;
    the machine's `fault` transitions recover: what the step left open is closed out, the callers
    waiting on it get the error once, the session's record is re-stated, and the machine rests.
    `call`: the behavior only takes a call from outside (an internal transition), so its failure is
    a `call_fault`. A step that handles a fault (a `fault` or `call_fault` step: its fault is taken
    before anything of it runs; `handles`: a behavior that ends a recovery) is never a fault of its
    own (`_handling_failed`)."""
    @functools.wraps(behavior)
    def step(ctx, inst, event):
        if event.name in FAULTS:
            _take_fault(inst=inst, event=event)
        made: list[tuple[hsm.Instance, hsm.Event]] = []
        token = _SENDS.set(made)
        try:
            result = _rule(behavior)(ctx=ctx, inst=inst, event=event)
            sends, dropped = made, []
        except Exception as error:  # noqa: BLE001 - the machine's fault, recovered from in the graph
            result, sends, dropped, failure = False, None, made, error
        finally:
            _SENDS.reset(token)
        if sends is not None:
            failure = None
            if len(sends) > 1:
                failure, sends, dropped = MultipleSends(f"{behavior.__name__} sent {len(sends)} events"), None, sends
        if sends and _failed(inst=inst, event=event) and (handles or event.name in FAULTS):
            sends = []  # a handling that failed is done again: what it sends then is sent once
        elif sends and _failed(inst=inst, event=event) and isinstance(inst, TimelineSM):
            sends = [(target, sent) for target, sent in sends if target is not inst.machines.session]
        if sends:
            target, sent = sends[HEAD]
            try:
                hsm.Dispatch(None, target, sent)
            except Exception as error:  # noqa: BLE001 - its one send failed: that is its fault
                failure = error
        if failure is not None:
            if handles or event.name in FAULTS:
                _handling_failed(inst=inst, event=event, error=failure, sends=dropped)
            else:
                _keep_fault(inst=inst, event=event, error=failure, call=call)
        _signal(inst)
        return result
    return step


def Effect(*behaviors: Callable) -> hsm.RedefinableBehaviors:
    return hsm.Effect(*(_step(behavior=b) for b in behaviors))


def Call(*behaviors: Callable) -> hsm.RedefinableBehaviors:
    """The effect of an internal transition that takes a call from outside (see CALL_FAULT)."""
    return hsm.Effect(*(_step(behavior=b, call=True) for b in behaviors))


def Recovers(*behaviors: Callable) -> hsm.RedefinableBehaviors:
    """The effect that ends a timeline's recovery from a fault: part of handling that fault (a raise
    in it is folded into the fault, `_handling_failed`)."""
    return hsm.Effect(*(_step(behavior=b, handles=True) for b in behaviors))


def Entry(*behaviors: Callable) -> hsm.RedefinableBehaviors:
    return hsm.Entry(*(_step(behavior=b) for b in behaviors))


def Exit(*behaviors: Callable) -> hsm.RedefinableBehaviors:
    return hsm.Exit(*(_step(behavior=b) for b in behaviors))


def Guard(expression: Callable) -> hsm.RedefinableConstraint:
    return hsm.Guard(_guard(expression))


def Activity(work: Callable, done: str) -> hsm.RedefinableBehaviors:
    """A state's activity that runs on its own (a run of the model loop, a clone call's preparation,
    stopping the machines): entered by a step that failed, it does nothing (`_held_back`, recorded
    as `done`, the outcome it would have dispatched)."""
    async def activity(ctx, inst, event) -> None:
        if _held_back(inst=inst, event=event, activity=done):
            return
        await work(ctx=ctx, inst=inst, event=event)
    activity.__name__ = work.__name__
    return hsm.Activity(activity)


def checked(*guarded: Callable, effects: tuple[Callable, ...] = (), first: bool = False) -> Callable:
    """The effect of a decision's unguarded branch: it checks that none of the decision's guarded
    branches held, read as the decision read them (`_holds`: in a step that failed none holds, and
    its safe branch is taken on purpose); a guard that raised did not hold, and wrote nothing, so
    this is where the decision it changed is reported, a DecisionFault. Then it runs `effects`
    (`first`: the effects run first, for a branch that keeps what it was given whatever the check
    finds)."""
    def effect(ctx, inst, event) -> None:
        if first:
            for e in effects:
                _rule(e)(ctx=ctx, inst=inst, event=event)
        held = [g.__name__ for g in guarded if _holds(guard=g, ctx=ctx, inst=inst, event=event)]
        if held:
            raise DecisionFault(f"{', '.join(held)} held, but its branch was not taken")
        if not first:
            for e in effects:
                _rule(e)(ctx=ctx, inst=inst, event=event)
    effect.__name__ = "checked_" + "_".join(e.__name__ for e in effects)
    return effect


def _background(inst: TimelineSM, clones: list[str]) -> bool:
    """Are all these grafts background grafts? From this machine's own record: the grafts it made,
    or itself."""
    return all(inst.made.get(c) is False or (c == inst.timeline_id and inst.kind == "background")
               for c in clones)


def _report(inst: TimelineSM, kind: str, clones: list[str], error: BaseException, quiet: bool = False,
            keep: bool = True, src: str | None = None) -> None:
    _note(inst=inst, method="report", keep=error if keep else None, kind=kind, src=src or inst.timeline_id,
          clones=clones, error=error, quiet=quiet, background=_background(inst=inst, clones=clones))


def mirrored(status: str) -> hsm.RedefinableBehaviors:
    def mirror(ctx, inst, event) -> None:
        _note(inst=inst, method="mirror", tl=inst.timeline_id, status=status)
    mirror.__name__ = f"mirror_{status}"
    return Entry(mirror)


def into(status: str) -> hsm.RedefinableBehaviors:
    """The effect of a transition into a final state: the state reached is recorded (`ended`) and
    copied to the store."""
    def mirror(ctx, inst, event) -> None:
        inst.ended = status
        _note(inst=inst, method="mirror", tl=inst.timeline_id, status=status)
    mirror.__name__ = f"mirror_into_{status}"
    return Effect(mirror)


def io(done: str, work: Callable) -> hsm.RedefinableBehaviors:
    """A state's activity that does I/O: `work(inst, event)` runs, and its result comes back as the
    event `done`. If it raises, the library dispatches `hsm/error`, handled by the state. Entered
    by a step that failed, it does nothing (`_held_back`)."""
    async def activity(ctx, inst, event) -> None:
        if _held_back(inst=inst, event=event, activity=done):
            return
        result = _rule(work)(inst=inst, event=event)
        inst.dispatch(inst.context(), hsm.Event(name=done, data=result))
    activity.__name__ = f"io_{done}"
    return hsm.Activity(activity)


# ------------------------------------------------------------------ rule words (read by guards)
def clone_refusal(inst: TimelineSM, grafts: list[tuple[str, bool]]) -> str | None:
    """The focus rule and the cap on live background grafts (a graft whose join is queued but not
    yet applied is still live)."""
    if not grafts:
        return "a clone call needs at least one graft"
    focus = [g for g, f in grafts if f]
    if len(focus) > 1:
        return f"at most one graft may take the conversation, got {focus}"
    live = [g for g, is_focus in inst.grafts.items() if not is_focus]
    if len(live) + sum(1 for _, f in grafts if not f) > MAX_LIVE_BACKGROUND:
        return (f"at most {MAX_LIVE_BACKGROUND} background clones may be live at once "
                f"({', '.join(live) or 'none'} still {'is' if len(live) == 1 else 'are'}); use fewer "
                "background entries or wait for them to join")
    return None


def blocked_parts(inst: TimelineSM, clone: Clone) -> list[tuple[int, frozenset[str]]]:
    return [(i, part) for i, part in clone.parts
            if any(r == clone.request and blocked & part for r, blocked in inst.blocks)]


def decline_why(inst: TimelineSM, clone: Clone) -> str | None:
    """Why the whole call is not tried at all, or None: decided before anything is seeded."""
    if inst.closing:
        return "the harness is closing; answer the user instead"
    if _run_withdrawn(inst):
        return "the user's call this turn answers was withdrawn"
    if inst.source_id is not None:
        return "only the original clones (grafts are one level deep)"
    if inst.clone_tries >= CLONE_TRIES:
        return "no clone calls left for this message; answer the user instead"
    if inst.mode.report:
        return "you were woken to report background clones that just finished; do not clone now"
    if inst.clone_calls >= MAX_CLONE_CALLS_PER_MESSAGE:
        return (f"the {MAX_CLONE_CALLS_PER_MESSAGE} clone calls for this user message are used up. Do not "
                "clone again: answer the user yourself, briefly saying what could not be done")
    if inst.handbacks.get(clone.request, NONE_YET) >= MAX_REQUEST_HANDBACKS:
        return ("clones for this message have failed or handed it back too many times. Do not clone "
                "for it again: answer the user yourself, briefly saying what could not be done")
    return None


def clone_why(inst: TimelineSM, clone: Clone) -> str | None:
    """Why the graph refuses this prepared clone call, or None."""
    if any(g in inst.made for g, _ in clone.pairs):
        return f"graft ids must be new: {[g for g, _ in clone.pairs]}"
    rule = clone_refusal(inst=inst, grafts=clone.pairs)
    if rule is not None:
        return f"clone is not possible now: {rule}"
    blocked = blocked_parts(inst=inst, clone=clone)
    if blocked:
        return "\n- ".join(
            f"clone[{i}]: a clone with skills {sorted(part)} for this message already failed, was cancelled, "
            "or handed it back twice. Do not clone for that part again: leave this entry out and tell "
            "the user, briefly, what could not be done" for i, part in blocked)
    return None


def may_cancel(inst: TimelineSM, child: str) -> bool:
    return (inst.source_id is None and inst.grafts.get(child) is False
            and all(q.child != child for q in inst.queue))


def cancel_refusal(inst: TimelineSM, child: str) -> str:
    """Why this id is not cancelled, from the original's own record of the grafts it made."""
    if inst.source_id is not None:
        return "only the original cancels; a clone can never cancel another clone"
    if child == inst.timeline_id:
        return "the original cannot be cancelled"
    if child not in inst.made:
        return "no such clone"
    if inst.made[child]:
        return "it is a focus clone; only a background clone can be cancelled"
    return "it has already finished or was already cancelled; nothing to cancel"


def _live(inst: TimelineSM, request: Request) -> bool:
    """Not withdrawn (by this machine's record) and not answered or given up by its caller: a
    caller that went away cancels its reply before its withdrawal reaches the machine."""
    return request.id not in inst.withdrawn and not request.reply.done()


def _withdrawn(inst: TimelineSM, request: Request) -> bool:
    """Withdrawn by its caller: by this machine's record, or its reply cancelled (a caller that
    goes away cancels it at once). Unlike `not _live`, a request that was answered is not
    withdrawn, unless its caller was cancelled before it took the answer (recorded: the cancel
    wins)."""
    return request.id in inst.withdrawn or request.reply.cancelled()


def _run_withdrawn(inst: TimelineSM, withdrawing: int | None = None) -> bool:
    """The one rule for ending a run on a withdrawal: the run answers a user request, and that
    request is not live (or is the one being withdrawn now, `withdrawing`); or, a wake that
    answers none, every background graft it reports was made for a request withdrawn (after its
    answer: the cancel wins)."""
    request = inst.answering
    if request is not None:
        return request.id == withdrawing or not _live(inst=inst, request=request)
    woken = inst.woken_for
    return bool(woken) and all(r.id == withdrawing or _withdrawn(inst=inst, request=r) for r in woken)


def _held(inst: TimelineSM) -> list[Request]:
    """Every request this machine holds: waiting, being written, answered by its run, waiting at rest."""
    held = list(inst.inbox) + [w.request for w in inst.waiters if w.request is not None]
    return held + [r for r in (inst.answering, inst.request) if r is not None]


def _answered(inst: TimelineSM) -> list[Request]:
    """Requests it answered whose calls may still be withdrawn: those answered at rest whose calls
    are not done, those its background grafts were made for whose joins are not applied, and
    those of the grafts the next wake reports."""
    answered: list[Request] = []
    for r in [*inst.returned, *inst.graft_requests.values(), *inst.woken_for]:
        answered += [] if any(r is a for a in answered) else [r]
    return answered


def _known(inst: TimelineSM) -> list[Request]:
    """Every request whose withdrawal it records: those it holds and those it answered."""
    return _held(inst) + _answered(inst)


def _prune(inst: TimelineSM) -> None:
    """The withdrawal record keeps only ids of requests it still holds, of requests it answered
    whose calls are not done, and of requests its background grafts were made for whose joins are
    not applied yet: it is bounded by them."""
    inst.withdrawn &= {r.id for r in _known(inst)}


def _unwanted(inst: TimelineSM, child: str) -> bool:
    """A background graft made for a request that was withdrawn: its join wakes nothing."""
    request = inst.graft_requests.get(child)
    return request is not None and _withdrawn(inst=inst, request=request)


def _made_for_withdrawn(inst: TimelineSM) -> bool:
    """A background graft whose request was withdrawn: it stops (or never starts)."""
    return inst.made_for is not None and _withdrawn(inst=inst, request=inst.made_for)


# ------------------------------------------------------------------ guards
def has_queue(ctx, inst, event) -> bool:
    return bool(inst.queue)


def no_queue(ctx, inst, event) -> bool:
    return not inst.queue


def is_external(ctx, inst, event) -> bool:
    return event.data[0].name == "external"


def not_external(ctx, inst, event) -> bool:
    return not is_external(ctx=ctx, inst=inst, event=event)


def ends_run(ctx, inst, event) -> bool:
    """The withdrawal leaves the request this run answers withdrawn: the run ends."""
    return inst.answering is not None and _run_withdrawn(inst=inst, withdrawing=event.data)


def ends_wake(ctx, inst, event) -> bool:
    """A late withdrawal (`withdraws_late`) leaves every graft this wake reports made for a
    withdrawn request: the wake ends."""
    return (inst.answering is None and withdraws_late(ctx=ctx, inst=inst, event=event)
            and _run_withdrawn(inst=inst, withdrawing=event.data))


def withdraws_late_only(ctx, inst, event) -> bool:
    """A late withdrawal that ends nothing running here."""
    return withdraws_late(ctx=ctx, inst=inst, event=event) and not ends_wake(ctx=ctx, inst=inst, event=event)


def arrives_withdrawn(ctx, inst, event) -> bool:
    return not _live(inst=inst, request=event.data)


def arrives_behind_joins(ctx, inst, event) -> bool:
    """A live message arrives while joins are queued: they are applied first."""
    return _live(inst=inst, request=event.data) and bool(inst.queue)


def arrives_to_answer(ctx, inst, event) -> bool:
    return _live(inst=inst, request=event.data) and not inst.queue


def request_live(ctx, inst, event) -> bool:
    return _live(inst=inst, request=inst.request)


def request_withdrawn_meanwhile(ctx, inst, event) -> bool:
    """The message whose row was just written was withdrawn meanwhile."""
    return not _live(inst=inst, request=inst.request)


def answering_live(ctx, inst, event) -> bool:
    """A run may start: it answers no user request (a wake for grafts that finished while it
    was idle), or at least one it answers is live."""
    return not _run_withdrawn(inst)


def _withdrawn_rows(inst: TimelineSM) -> list[int]:
    """The row of the request its run answers, if its caller withdrew it after its row was written
    (a reply answered with an error, by a recovery, is not a withdrawal: its row is not marked)."""
    request = inst.answering
    withdrawn = request is not None and _withdrawn(inst=inst, request=request)
    return [request.row] if withdrawn and request.row is not None else []


def _answers_requests(inst: TimelineSM) -> bool:
    """Its current run is the one that answers its request: a turn on it (a focus graft's run
    included), or a routing run on a request a focus graft handed back. A report or a wake for
    background grafts follows a turn that already answered it."""
    return inst.mode.user or (inst.mode.name == "route" and not inst.mode.wake)


def _handed_back(inst: TimelineSM) -> bool:
    """The joins just applied include a focus graft that came back without answering (not done):
    the request it answered is this timeline's to answer now."""
    return any(r != "done" and inst.made.get(c) for c, r in inst.news)


def cut_short(ctx, inst, event) -> bool:
    """The run that ended early was answering a request that was withdrawn after its row was
    written: that row is marked (the run never answered it)."""
    return _answers_requests(inst) and bool(_withdrawn_rows(inst))


def nothing_to_mark(ctx, inst, event) -> bool:
    return not cut_short(ctx=ctx, inst=inst, event=event)


def _marks_after(inst: TimelineSM) -> bool:
    """After its joins: a focus graft handed back its request unanswered, and it was withdrawn
    after its row was written (no run will answer it)."""
    return _handed_back(inst) and bool(_withdrawn_rows(inst))


def withdrawn_after(ctx, inst, event) -> bool:
    """The handed-back request's row is marked before anything else is taken."""
    return not inst.final and _marks_after(inst)


def noted_withdrawn(ctx, inst, event) -> bool:
    """The wake note is written, but the request the wake would answer was withdrawn meanwhile;
    a routing wake for a request a focus graft handed back leaves it unanswered."""
    return _marks_after(inst)


def handed_live(ctx, inst, event) -> bool:
    inbox, withdrawn = event.data
    return any(r.id not in withdrawn and _live(inst=inst, request=r) for r in inbox)


def handed_withdraws_run(ctx, inst, event) -> bool:
    """The withdrawals handed over with the conversation include the request this run answers
    (one that reached the original while this graft was being started)."""
    _, withdrawn = event.data
    request = inst.answering
    return request is not None and (request.id in withdrawn or not _live(inst=inst, request=request))


def handed_live_with_queue(ctx, inst, event) -> bool:
    return bool(inst.queue) and handed_live(ctx=ctx, inst=inst, event=event)


def handed_live_to_answer(ctx, inst, event) -> bool:
    return not inst.queue and handed_live(ctx=ctx, inst=inst, event=event)


def starts_busy(ctx, inst, event) -> bool:
    """A focus graft starts on its request at once, unless a caller drives the turn that made it
    or the request was withdrawn before it could act."""
    return inst.kind == "focus" and not inst.held and not _run_withdrawn(inst)


def starts_withdrawn(ctx, inst, event) -> bool:
    return inst.kind == "focus" and not inst.held and _run_withdrawn(inst)


def is_held(ctx, inst, event) -> bool:
    return inst.held


def made_for_live(ctx, inst, event) -> bool:
    """A background graft may take its next step, by its own record: the request it was made for is
    not withdrawn (its source decides then, `graft_step`, knowing also a withdrawal that came after
    that request was answered)."""
    return not _made_for_withdrawn(inst)


def made_for_withdrawn(ctx, inst, event) -> bool:
    return _made_for_withdrawn(inst)


def starts_cancelled(ctx, inst, event) -> bool:
    """A background graft made for a request withdrawn before it was made: it never runs."""
    return not inst.held and _made_for_withdrawn(inst)


def starts_running(ctx, inst, event) -> bool:
    """A background graft that runs on its own, made for a request not withdrawn before it was
    made (the other start, a graft that never runs, is the unguarded default)."""
    return not inst.held and not _made_for_withdrawn(inst)


def _own(inst: TimelineSM, child: str) -> bool:
    """A graft whose join is this timeline's to apply: one it holds, or one a cancel stopped."""
    return child in inst.grafts or child in inst.stopping


def own_graft(ctx, inst, event) -> bool:
    """A join from a graft it holds, not one a cancel step it is in waits for: one of a clone call
    undone because a machine could not start (its join held for the step) is never applied."""
    return _own(inst=inst, child=event.data.child) and event.data.child not in inst.pending


def foreign_graft(ctx, inst, event) -> bool:
    return not _own(inst=inst, child=event.data.child) and event.data.child not in inst.pending


def other_pending(ctx, inst, event) -> bool:
    """A graft the cancel step stopped, not the last one it waits for."""
    return event.data.child in inst.pending and inst.pending != {event.data.child}


def _join_arrivals(T, On, G, besides: tuple[Callable, ...] = ()) -> list:
    """How a graft's join is taken where nothing else is made of it: queued if it is one of its
    own grafts, ignored if not (a graft of a call undone at its start); neither (a guard raised:
    the decision is reported) keeps it for the recovery, which queues it if it is its own.
    `besides`: the guards of the same event declared before these (checked too)."""
    return [T(On("graft_queued"), G(own_graft), Call(enqueue)),
            T(On("graft_queued"), G(foreign_graft), Call(ignore)),
            T(On("graft_queued"), Effect(checked(*besides, own_graft, foreign_graft, effects=(keep_stray,),
                                                 first=True)))]


def is_focus(ctx, inst, event) -> bool:
    return inst.kind == "focus"


def carries_cancels(ctx, inst, event) -> bool:
    return event.data.cancels is not None


def no_cancels(ctx, inst, event) -> bool:
    return event.data.cancels is None


def step_intact(ctx, inst, event) -> bool:
    """The step in progress has not failed (in one that has, this decision takes its safe branch)."""
    return not _failed(inst=inst, event=event)


def comes_to_rest(ctx, inst, event) -> bool:
    """After a run: nothing to drain, close or take, and the step has not failed: at rest, its
    callers are answered."""
    return (step_intact(ctx=ctx, inst=inst, event=event) and not drains_now(ctx=ctx, inst=inst, event=event)
            and not closes_now(ctx=ctx, inst=inst, event=event) and not answers_inbox(ctx=ctx, inst=inst, event=event))


def declines(ctx, inst, event) -> bool:
    """The rules' outcome, worked out before the gate (`judge_call`), declines the whole call (or
    working it out raised: its error is raised by `decline`)."""
    return inst.declined != (None, None)


def may_try(ctx, inst, event) -> bool:
    """The whole call may be tried (the refusal, `rejecting`, is the gate's unguarded default): the
    rules' outcome, worked out before the gate (`judge_call`), declines nothing."""
    return inst.declined == (None, None)


def rejected_in_preparation(ctx, inst, event) -> bool:
    return isinstance(event.data, str)


def graph_allows(ctx, inst, event) -> bool:
    """A prepared call (not one the harness refused in its preparation) the graph allows: it is
    committed. A prepared call the graph refuses is the gate's unguarded default (`refusing`,
    which rolls the call back)."""
    return not isinstance(event.data, str) and inst.refused == (None, None)


def run_goes_on(ctx, inst, event) -> bool:
    """The run's request, if any, is still live: it goes on (its end is the unguarded default)."""
    return not _run_withdrawn(inst)


def cancels_only(ctx, inst, event) -> bool:
    """Cancel calls with no clone call carried, and the run's request still live: the run goes on."""
    return inst.clone is None and not _run_withdrawn(inst)


def clone_waits(ctx, inst, event) -> bool:
    """A clone call carried with the cancels, the run's request was not withdrawn meanwhile,
    and their rows were written."""
    return inst.clone is not None and not _run_withdrawn(inst) and inst.reported is None


def clone_dropped(ctx, inst, event) -> bool:
    """A clone call carried with cancels whose rows could not be written: the run ends with that
    error, so the clone call is not run."""
    return inst.clone is not None and not _run_withdrawn(inst) and inst.reported is not None


def hands_over(ctx, inst, event) -> bool:
    return inst.focus is not None


def has_joined(ctx, inst, event) -> bool:
    return inst.ended == "joined"


def joins_back(ctx, inst, event) -> bool:
    """A focus graft's run that ended by joining, or with its step budget spent (it joins failed)."""
    return inst.kind == "focus" and event.data.kind in ("join", "exhausted")


def run_joined(ctx, inst, event) -> bool:
    return event.data.kind == "join"


def run_replied(ctx, inst, event) -> bool:
    return event.data.kind == "replied"


def head_cancellable(ctx, inst, event) -> bool:
    return bool(inst.to_check) and inst.to_check[0] not in inst.claims and may_cancel(inst=inst, child=inst.to_check[0])


def head_refused(ctx, inst, event) -> bool:
    return bool(inst.to_check) and not head_cancellable(ctx=ctx, inst=inst, event=event)


def checked_with_pending(ctx, inst, event) -> bool:
    return not inst.to_check and bool(inst.pending)


def last_pending(ctx, inst, event) -> bool:
    return inst.pending == {event.data.child}


def one_of_pending(ctx, inst, event) -> bool:
    return event.data.child in inst.pending


def rows_failed(ctx, inst, event) -> bool:
    return bool(event.data[1])


def rows_written(ctx, inst, event) -> bool:
    return not event.data[1]


def error_to_user(ctx, inst, event) -> bool:
    """The run raised and a caller waits on it: the error is that caller's."""
    return inst.kind != "focus" and inst.mode.user and any(w.error is None for w in inst.waiters)


def error_already_shown(ctx, inst, event) -> bool:
    return (inst.kind != "focus" and inst.reported is event.data
            and not error_to_user(ctx=ctx, inst=inst, event=event))


def drains_now(ctx, inst, event) -> bool:
    return bool(inst.queue) and not inst.hold


def closes_now(ctx, inst, event) -> bool:
    """Shut down, nothing left to apply, and no graft of its own still running (whose join it
    waits for, at rest)."""
    return inst.final and not drains_now(ctx=ctx, inst=inst, event=event) and not _still_running(inst)


def closes_clean(ctx, inst, event) -> bool:
    """It closes now, and no close-out a recovery could not write is kept."""
    return closes_now(ctx=ctx, inst=inst, event=event) and not inst.close_out


def closes_out_first(ctx, inst, event) -> bool:
    """It closes now, and a close-out a recovery could not write is written first (`closing_out`)."""
    return closes_now(ctx=ctx, inst=inst, event=event) and bool(inst.close_out)


def _inbox_live(inst: TimelineSM) -> bool:
    return any(_live(inst=inst, request=r) for r in inst.inbox)


def answers_inbox(ctx, inst, event) -> bool:
    return _inbox_live(inst) and not inst.final and not drains_now(ctx=ctx, inst=inst, event=event)


def applies_next(ctx, inst, event) -> bool:
    return bool(inst.queue) and not inst.hold


def retry_left(ctx, inst, event) -> bool:
    return inst.attempts.get(inst.queue[0].child, NONE_YET) + 1 < MAX_JOIN_ATTEMPTS


def last_attempt(ctx, inst, event) -> bool:
    return not retry_left(ctx=ctx, inst=inst, event=event)


def is_final(ctx, inst, event) -> bool:
    return inst.final and not _still_running(inst)


def final_clean(ctx, inst, event) -> bool:
    return is_final(ctx=ctx, inst=inst, event=event) and not inst.close_out


def final_out_first(ctx, inst, event) -> bool:
    return is_final(ctx=ctx, inst=inst, event=event) and bool(inst.close_out)


def closes_out_at_rest(ctx, inst, event) -> bool:
    """Shut down at rest with nothing queued, and a close-out a recovery could not write kept."""
    return not inst.queue and bool(inst.close_out)


def final_waits(ctx, inst, event) -> bool:
    """Shut down while a graft of its own still runs: at rest, its join is applied when it comes."""
    return inst.final and bool(_still_running(inst))


# The `after` choice's guards are disjoint: each outcome is one case of the one before it left
# open (not shut down; no handed-back withdrawn row to mark; no live message waiting; something
# to answer), and the report wake is the remainder, its one unguarded default.
def _after_open(inst: TimelineSM) -> bool:
    """Not shut down, and no handed-back withdrawn row to mark first."""
    return not inst.final and not _marks_after(inst)


def inbox_after(ctx, inst, event) -> bool:
    return _after_open(inst) and _inbox_live(inst)


def _nothing_waits(inst: TimelineSM) -> bool:
    return _after_open(inst) and not _inbox_live(inst)


def quiet_after(ctx, inst, event) -> bool:
    """Nothing to answer: no news (or closing), or every request a run would answer withdrawn."""
    return _nothing_waits(inst) and (not inst.news or inst.closing or _run_withdrawn(inst))


def _wakes(inst: TimelineSM) -> bool:
    """Joins brought news to act on, and no request a run would answer is withdrawn."""
    return _nothing_waits(inst) and bool(inst.news) and not inst.closing and not _run_withdrawn(inst)


def answers_handback(ctx, inst, event) -> bool:
    return _wakes(inst) and inst.ends_on_request


def _routing(inst: TimelineSM) -> bool:
    focus_back = any(r != "done" and inst.made.get(c) for c, r in inst.news)
    return focus_back or (any(r == "out_of_scope" for _, r in inst.news)
                          and inst.wake_clones < MAX_WAKE_CLONE_CALLS)


def wakes_to_route(ctx, inst, event) -> bool:
    return _wakes(inst) and not inst.ends_on_request and _routing(inst)


def wakes_to_report(ctx, inst, event) -> bool:
    """The rest of the wakes: a report run (coming to rest is the unguarded default)."""
    return _wakes(inst) and not inst.ends_on_request and not _routing(inst)


def is_driven(ctx, inst, event) -> bool:
    """A turn its caller drives: the caller writes its own rows, so its open calls are its own."""
    return inst.mode.name == "external"


# ------------------------------------------------------------------ activities
def _still_running(inst: TimelineSM) -> list[str]:
    """Its background grafts still running: live, and whose join is not on its queue."""
    queued = {q.child for q in inst.queue}
    return [g for g, focus in inst.grafts.items() if not focus and g not in queued]


async def busy_activity(ctx, inst: TimelineSM, event) -> None:
    """A run of the model loop. It ends by raising `turn_done` on its own machine with how the run
    ended (a RunOutcome); the graph decides what follows (leaving the state ends this activity).
    Before each step that could act (a model call, a tool, the next text) the run asks its
    machine (`may_continue`); a run whose request was withdrawn is answered by leaving
    this state, so it is ended there and then. A run that raises ends with its own outcome,
    `run_failed`, never `hsm/error`: a state inside the run that does I/O of its own (a clone
    call being prepared, a step) handles `hsm/error` as its own failure, so the run's failure is
    typed to be told apart from it. Like the library's own `hsm/error`, it is sent only while the
    run's state is still active: a run that raises as it is ended (its state left) is stale."""
    try:
        _close_out(inst)
        _mark_pending(inst)
        outcome = await inst.host.run_busy(tl=inst.timeline_id, mode=inst.mode, kind=inst.kind)
    except Exception as error:  # noqa: BLE001 - the outcome event carries it
        if not ctx.is_done():
            inst.dispatch(inst.context(), hsm.Event(name="run_failed", data=error))
        return
    await inst.dispatch(inst.context(), hsm.Event(name="turn_done", data=outcome))


async def background_activity(ctx, inst: TimelineSM, event) -> None:
    outcome = await inst.host.run_background(tl=inst.timeline_id)
    await inst.dispatch(inst.context(), hsm.Event(name="finish", data=outcome))


async def prepare_activity(ctx, inst: TimelineSM, event) -> None:
    """The harness seeds and validates the clone call (its seeder may take a while: a withdrawal
    or shutdown ends this activity, and `unpreparing` rolls the call's audit back). The outcome is
    `prepared`: a Prepared call, or the text of a refusal the harness already audited."""
    _mark_pending(inst)  # the call's seeds are resolved from the view (a driven turn has no run start)
    prepared = await inst.host.prepare_clone(src=inst.timeline_id, clone=inst.clone,
                                             retry_left=inst.clone_tries < CLONE_TRIES)
    inst.dispatch(inst.context(), hsm.Event(name="prepared", data=prepared))


async def spawn_activity(ctx, inst: TimelineSM, event) -> None:
    """Make and start the machines of the grafts the clone call committed, awaiting each start.
    Its inputs are taken when it starts, before its first await: the grafts to make, whether a
    caller drives the turn, and the withdrawals recorded so far, which every graft is made with
    (one whose request was withdrawn before it could act never acts on it). It reads nothing of
    its machine after that: a withdrawal that comes while it runs waits for its outcome
    (`spawning` defers it) and is then handled like any later one (forwarded to the focus graft by
    event; a background graft reads it from its request's reply, which a cancelled call cancels
    before it withdraws). The focus graft, if any, is started last: it starts on its request at
    once (it engages, and runs), so it is made only once every background graft of the call has
    started, and a call undone because a graft could not start never had a focus graft act. The
    outcome is an event: `spawned`, or `spawn_failed` with the machines made so far (they are
    undone in `unspawning`). A step that failed never enters it (`spawn_gate` undoes the call
    instead)."""
    machines, source, clone = inst.machines, inst.timeline_id, inst.clone
    plans = sorted(inst.spawn_plan, key=lambda plan: plan.focus)  # the focus graft last
    held, withdrawn = inst.mode.name == "external", set(inst.withdrawn)
    made: list[TimelineSM] = []
    try:
        for plan in plans:
            sm = TimelineSM(machines=machines, timeline_id=plan.child, source_id=source,
                            kind="focus" if plan.focus else "background", held=held,
                            waiters=plan.waiters, answering=plan.answering, made_for=plan.made_for,
                            withdrawn=withdrawn)
            machines.timelines[plan.child] = sm
            made.append(sm)
            await hsm.Started(None, sm, TIMELINE if plan.focus else BACKGROUND, hsm.Config(ID=plan.child))
    except Exception as e:  # noqa: BLE001 - the outcome event carries it
        inst.dispatch(inst.context(), hsm.Event(name="spawn_failed", data=([sm.timeline_id for sm in made], e)))
        return
    inst.dispatch(inst.context(), hsm.Event(name="spawned", data=clone))


async def unspawn_activity(ctx, inst: TimelineSM, event) -> None:
    """Undo a clone call whose machines could not all start: the machines made are stopped and
    forgotten, and the committed grafts are marked abandoned in the store. The refusal follows.
    Its inputs are taken before its first await (it reads nothing of its machine after that)."""
    made, grafts, error = list(inst.spawned_ids), [g for g, _ in inst.clone.pairs], inst.spawn_error
    machines, host, source = inst.machines, inst.host, inst.timeline_id
    for child in made:
        sm = machines.timelines.pop(child, None)
        if sm is not None and sm.state():
            await hsm.Stop(sm)
    host.abandon_clone(src=source, grafts=grafts)
    inst.dispatch(inst.context(), hsm.Event(name="unspawned", data=error))


def _mark_pending(inst: TimelineSM) -> None:
    """Rows of withdrawn requests whose mark could not be written are marked before anything reads
    the view they are in: a run (its compaction, seeds and seeder included), a clone call's
    preparation (a driven turn's too) and a join's copy; a focus graft hands its own back with its
    join. So no later turn, seed or join reads them as a request."""
    if inst.unmarked:
        inst.host.mark_withdrawn(timelines=_viewers(inst), rows=list(inst.unmarked))


def _close_out(inst: TimelineSM) -> None:
    """A close-out a recovery could not write (the run's open calls, the clone call's rollback, the
    rows to mark) is written first by every activity that next writes to or reads the view at
    rest: a message's row, a run's start, a join's copy, a wake note, a mark, a recovery. So no
    row lands after a tool call left without its result. Writing it again changes nothing."""
    for args in inst.close_out:
        inst.host.recover(**args)


def _close_out_of(inst: TimelineSM) -> dict[str, Any]:
    """What a recovery closes out (`host.recover`'s arguments), as it stands now."""
    return dict(tl=inst.timeline_id, known=dict(inst.unstored), close_calls=inst.mode.name != "external",
                viewers=_viewers(inst), rows=list(inst.unmarked), cancels=list(inst.cut_cancels))


def _closing_out(inst, event):
    """The activity of `closing_out` and `releasing`: the kept close-out, if any, is written before
    the timeline closes."""
    _close_out(inst)


def close_out_failed(ctx, inst, event) -> None:
    """The kept close-out could not be written as the timeline closes: the user is told (settle
    raises it), and it closes."""
    _report(inst=inst, kind="task", clones=[], error=event.data)


def closed_out(inst: TimelineSM) -> None:
    """The activity that wrote the kept close-out first has succeeded: nothing is left to write."""
    inst.close_out = []


def _write_request(inst, event):
    _close_out(inst)
    return inst.host.begin_request(tl=inst.timeline_id, text=inst.request.text)


def _viewers(inst: TimelineSM) -> list[str]:
    """The timelines whose views a withdrawal mark binds: the one the request was asked of and, for
    a focus graft, its source, which holds the same conversation and gets the graft's rows back."""
    return [inst.timeline_id] + ([inst.source_id] if inst.kind == "focus" else [])


def _mark_rows(inst, event):
    """The rows of withdrawn requests that no run answered get the withdrawal mark (the activity of
    `dropping` and `withdrawing`): the one the state was entered for, and any whose mark could
    not be written before."""
    _close_out(inst)
    inst.host.mark_withdrawn(timelines=_viewers(inst), rows=list(inst.unmarked))


def _unprepare(inst, event):
    """A clone call whose preparation was cut short: its open audit transactions are closed as
    rolled back (the activity of `unpreparing`)."""
    inst.host.roll_back_clone(src=inst.timeline_id)


def _commit(inst, event):
    return inst.host.commit_clone(src=inst.timeline_id, clone=inst.clone)


def _clone_row(inst, event):
    if inst.clone.store:
        inst.host.store_result(tl=inst.timeline_id, use=inst.clone.use, content=inst.clone_text, is_error=False)


def _refuse(inst, event):
    """The refused clone call's audit is closed and its result stored."""
    error = event.data if isinstance(event.data, BaseException) else None  # a failed commit or start
    why, _ = inst.refused  # the graph's reason, worked out before the prepared gate
    text = inst.host.refuse_clone(src=inst.timeline_id, clone=inst.clone, why=why or "internal error",
                                  error=error)
    if inst.clone.store:
        inst.host.store_result(tl=inst.timeline_id, use=inst.clone.use, content=text, is_error=True)
    return text


def _reject(inst, event):
    """A clone call refused before or in its preparation: its result row (no clone was made, and
    the harness audited what it prepared)."""
    if inst.clone.store:
        inst.host.store_result(tl=inst.timeline_id, use=inst.clone.use, content=inst.rejection, is_error=True)
    return inst.rejection


def _join_back(inst, event):
    """A focus graft's run ended by joining (or with its step budget spent: it joins failed): its
    join event, with the user message it leaves unanswered."""
    return inst.host.join_of(tl=inst.timeline_id, reason=event.data.reason or "failed")


def _cancel_rows(inst, event):
    """Each cancel call's real result row. A row that cannot be written is kept for the run's
    closing (it then holds the real result); the run ends with that error."""
    results, failures = [], []
    again = "it has already finished or was already cancelled; nothing to cancel"
    for index, call in enumerate(inst.cancel.calls):
        outcomes = {c: inst.outcomes[c] if inst.claims.get(c) == index else again for c in call.ids}
        content, is_error = inst.host.cancel_text(call=call, outcomes=outcomes)
        try:
            inst.host.store_result(tl=inst.timeline_id, use=call.use, content=content, is_error=is_error)
        except Exception as e:  # noqa: BLE001 - the outcome event carries it
            stopped = [c for c in call.ids if outcomes.get(c) == CANCELLED]
            failures.append((call.use["id"], content, is_error, stopped, e))
        results.append((content, is_error))
    return results, failures


def _close_calls(inst, event):
    inst.host.close_tool_uses(tl=inst.timeline_id, known=dict(inst.unstored))


def _focus_failure(inst, event):
    """A focus graft whose run raised: its open calls get results, the failure is recorded, and its
    join (reason failed, with the request it leaves unanswered) is computed."""
    error = event.data
    inst.host.close_tool_uses(tl=inst.timeline_id, known=dict(inst.unstored))
    inst.host.graft_failed(tl=inst.timeline_id, error=error)
    return inst.host.join_of(tl=inst.timeline_id, reason="failed")


def _apply(inst, event):
    _close_out(inst)
    _mark_pending(inst)  # the join copies rows into the view: rows still to be marked are marked first
    join = inst.queue[0]
    return inst.host.apply_join(src=inst.timeline_id, join=join, cloned_for=inst.cloned_for.get(join.child),
                                running=_still_running(inst))


def _wake_note(inst, event):
    _close_out(inst)
    inst.host.wake_note(src=inst.timeline_id, joined=list(inst.news), routing=inst.mode.name == "route",
                        running=_still_running(inst),
                        background=_background(inst=inst, clones=[c for c, _ in inst.news]))


def _record(inst, event):
    """A background graft's join reason is written; its open calls first get results if its run
    ended early. Each failure is collected: the join is queued whatever the store does."""
    errors = []
    if inst.close_first:
        try:
            inst.host.close_tool_uses(tl=inst.timeline_id, known=dict(inst.unstored))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    try:
        inst.host.record_join(child=inst.timeline_id, reason=inst.reason)
    except Exception as e:  # noqa: BLE001
        errors.append(e)
    return errors


def _recover(inst, event):
    """What a failed step left open is closed out (the activity of `recovering`): the clone call's
    open audit transactions are rolled back, the run's open tool calls get results (a tool that
    ran, its real result; a cancel call that stopped grafts, what it did; a driven turn's calls are
    its caller's), and the rows of withdrawn requests still to be marked are marked."""
    _close_out(inst)
    inst.host.recover(**_close_out_of(inst))


def _scribe_write(inst, event):
    (method, args), _keep = inst.note
    getattr(inst.host, method)(**args)


# ------------------------------------------------------------------ effects: user requests
def _answer_dead(requests: list[Request]) -> None:
    """The one response to a request dropped because it is no longer live: it is answered (nothing
    is run for it)."""
    for r in requests:
        _resolve(reply=r.reply, value="ended")


def _drop_dead(inst: TimelineSM) -> None:
    """The waiting messages whose callers have gone leave the inbox, each answered."""
    dead = [r for r in inst.inbox if not _live(inst=inst, request=r)]
    inst.inbox = [r for r in inst.inbox if all(r is not d for d in dead)]
    _answer_dead(dead)


def take_message(ctx, inst, event) -> None:
    inst.request, inst.recoveries = event.data, 0
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def drop_message(ctx, inst, event) -> None:
    """A message whose caller had gone before it arrived: nothing is written or run for it."""
    _answer_dead([event.data])


def take_inbox(ctx, inst, event) -> None:
    """The first live waiting message is taken (a guard saw one); the dead ones are dropped."""
    _drop_dead(inst)
    inst.request = inst.inbox.pop(HEAD)


def request_written(ctx, inst, event) -> None:
    """The message's row is written: it is the run's request; the bounds of the last one are
    cleared (a new request may clone again)."""
    closed_out(inst)
    request, inst.request = inst.request, None
    request.row = event.data
    inst.answering, inst.answered = request, False
    inst.mode, inst.clone_tries = TURN, 0
    inst.blocks, inst.clone_calls, inst.wake_clones = set(), 0, 0
    inst.waiters.append(Waiter(reply=request.reply, user=True, request=request))


def request_dropped(ctx, inst, event) -> None:
    """The message's row is written, but its caller withdrew it meanwhile: no run starts for it.
    The row stays (rows are never removed) and is marked withdrawn next (`dropping`), so no later
    turn reads it as a live request. A new message was written, so the bounds of the last one
    are cleared."""
    closed_out(inst)
    request, inst.request = inst.request, None
    inst.blocks, inst.clone_calls, inst.wake_clones = set(), 0, 0
    inst.unmarked.append(event.data)
    _prune(inst)
    _answer_dead([request])


def request_held(ctx, inst, event) -> None:
    """The step that would decide what follows the message's written row failed: its row is to
    be marked (no run answered it), and the request stays held, for the recovery to answer with
    the step's error."""
    closed_out(inst)
    inst.request.row = event.data
    inst.unmarked.append(event.data)


def request_failed(ctx, inst, event) -> None:
    """The message's row could not be written: its caller gets the error, the user is told, and no
    run starts for it."""
    request, inst.request = inst.request, None
    inst.waiters.append(Waiter(reply=request.reply, user=True, error=event.data, request=request))
    _report(inst=inst, kind="request", clones=[], error=event.data, keep=False)


def stash_message(ctx, inst, event) -> None:
    inst.inbox.append(event.data)


def stash_and_engage(ctx, inst, event) -> None:
    inst.inbox.append(event.data)
    inst.recoveries = 0
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def note_withdrawn(ctx, inst, event) -> None:
    """A caller withdrew its request (by id): recorded if this machine holds it, wherever it is;
    a waiting message is dropped and answered at once. An unknown id, or one whose call is done,
    changes nothing."""
    gone = [r for r in _held(inst) if r.id == event.data]
    inst.withdrawn |= {event.data} if gone else set()
    _drop_dead(inst)
    _prune(inst)


def _late(inst: TimelineSM, rid: int) -> list[Request]:
    """The request it answered that this withdrawal is of (its caller was cancelled after the
    answer and before it took it), unless it holds that request still or recorded it already."""
    held = {r.id for r in _held(inst)}
    return [r for r in _answered(inst) if r.id == rid and rid not in held and rid not in inst.withdrawn
            and not r.reply.cancelled()]  # (one whose caller went away first was withdrawn already)


def withdraws_late(ctx, inst, event) -> bool:
    """The withdrawal is of a request it answered whose call is not done: the cancel wins."""
    return bool(_late(inst=inst, rid=event.data))


def _reply_error(reply: asyncio.Future) -> BaseException | None:
    """The error a reply was answered with, if any (read: its caller never took it)."""
    return reply.exception() if reply.done() and not reply.cancelled() else None


def note_late_withdrawal(ctx, inst, event) -> None:
    """A request it answered is withdrawn: its caller was cancelled before it took the answer, and
    the cancel wins. Recorded, so the background grafts made for it stop at their next step (each
    asks this machine, `graft_step`) and join `cancelled`, and their joins wake nothing. Its row
    gets the withdrawal mark (a write of the scribe's) in the views it was asked in; an error it
    was answered with, which its caller never took, is told after that write (one line, which the
    next call's output shows, or else `settle` raises)."""
    request, = _late(inst=inst, rid=event.data)
    inst.withdrawn.add(request.id)
    _note(inst=inst, method="mark_withdrawn", keep=_reply_error(request.reply), timelines=_viewers(inst),
          rows=[request.row] if request.row is not None else [])


def call_finished(ctx, inst, event) -> None:
    """A user's call is done (its request id): a request it answered can no longer be withdrawn."""
    inst.returned = [r for r in inst.returned if r.id != event.data]
    _prune(inst)


def call_finished_and_forward(ctx, inst, event) -> None:
    """Suspended: the call may be one its focus graft answered, so the focus graft hears it too."""
    call_finished(ctx=ctx, inst=inst, event=event)
    _forward(target=inst.machines.timelines.get(inst.focus), name="call_done", data=event.data)


def graft_wanted(ctx, inst, event) -> bool:
    """A background graft of its own asks whether it may take its next step: the request it was made
    for, if any, is not withdrawn (by this machine's record, or its reply cancelled)."""
    child, _ = event.data
    return not _unwanted(inst=inst, child=child)


def let_graft_go(ctx, inst, event) -> None:
    _, reply = event.data
    _resolve(reply=reply, value=True)


def stop_graft(ctx, inst, event) -> None:
    """The request a background graft was made for is withdrawn: the graft is stopped (`cancel`: it
    joins `cancelled`); the step it asked about never runs."""
    child, _ = event.data
    _forward(target=inst.machines.timelines.get(child), name="cancel")


def ask_source(ctx, inst, event) -> None:
    """A background graft's run asks before each step: its source decides (it knows whether the
    request the graft was made for is withdrawn, also after that request was answered)."""
    _forward(target=inst.machines.timelines.get(inst.source_id), name="graft_step",
             data=(inst.timeline_id, event.data))


def withdraws_waiting(ctx, inst, event) -> bool:
    """The withdrawal is of a message waiting on this machine."""
    return any(r.id == event.data for r in inst.inbox)


def withdraw_and_forward(ctx, inst, event) -> None:
    """Recorded here if this machine holds the request (while the clone call that makes the focus
    graft is in flight it still does), and forwarded to the focus graft that holds the
    conversation once it is made (while its machine is started, `spawning` defers the withdrawal),
    which records it if it holds the request. A suspended original holds
    none, so its record stays bounded however long the focus graft holds the conversation; a
    request the graft hands back carries its reply, which a cancelled call has already cancelled."""
    inst.withdrawn |= {event.data} if any(r.id == event.data for r in _held(inst)) else set()
    _forward(target=inst.machines.timelines.get(inst.focus) if inst.focus else None, name="withdraw",
             data=event.data)


def take_handed(ctx, inst, event) -> None:
    """Messages handed over with the conversation: they arrived before any this timeline holds,
    so they go first, in order."""
    inbox, withdrawn = event.data
    inst.inbox = list(inbox) + inst.inbox
    inst.withdrawn |= withdrawn
    _prune(inst)


def take_handed_and_begin(ctx, inst, event) -> None:
    """Handed messages reach it at rest: the dead ones are dropped and the first live one is
    taken now (one session event says both)."""
    take_handed(ctx=ctx, inst=inst, event=event)
    _drop_dead(inst)
    inst.request = inst.inbox.pop(HEAD)
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def take_handed_and_engage(ctx, inst, event) -> None:
    """Handed messages reach it at rest with joins queued: the joins are applied first."""
    take_handed(ctx=ctx, inst=inst, event=event)
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def begin_run(ctx, inst, event) -> None:
    mode, reply = event.data
    inst.mode, inst.answering, inst.clone_tries, inst.recoveries = mode, None, 0, 0
    inst.waiters.append(Waiter(reply=reply, user=False))
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def begin_driven(ctx, inst, event) -> None:
    mode, reply = event.data
    inst.mode, inst.answering, inst.clone_tries, inst.recoveries = mode, None, 0, 0
    _session(inst=inst, name="engaged", data=inst.timeline_id)
    _resolve(reply=reply, value="started")


def add_run_waiter(ctx, inst, event) -> None:
    inst.waiters.append(Waiter(reply=event.data, user=False))


def answer_ended(ctx, inst, event) -> None:
    _resolve(reply=event.data, value="ended")


def answer_handed_off(ctx, inst, event) -> None:
    _resolve(reply=event.data, value="handed_off")


def answer_joined(ctx, inst, event) -> None:
    _resolve(reply=event.data, value="joined")


def answer_true(ctx, inst, event) -> None:
    _resolve(reply=event.data, value=True)


def ignore(ctx, inst, event) -> None:
    return None


def _reply_of(data: Any) -> asyncio.Future | None:
    if isinstance(data, asyncio.Future):
        return data
    if isinstance(data, tuple):
        return next((x for x in reversed(data) if isinstance(x, asyncio.Future)), None)
    return getattr(data, "reply", None)


def refuse_request(ctx, inst, event) -> None:
    """A request this state does not take is refused."""
    where = inst.state().split("/", STATE_HEAD)[-1] or "stopped"
    _fail(reply=_reply_of(event.data), error=TransitionRefused(
        f"{getattr(inst, 'timeline_id', 'session')}: '{event.name}' refused in state {where}"))


def refuse_clone_call(ctx, inst, event) -> None:
    """A clone call this timeline may not make at all (it is not running): refused, and so are
    the cancel calls carried with it. The caller stores the results."""
    clone: Clone = event.data
    why = Unwritten(f"clone refused; no clone was created: {inst.timeline_id} is not running a turn that may clone")
    _resolve(reply=clone.reply, value=(None, None, why))
    if clone.cancels is not None:
        _resolve(reply=clone.cancels.reply, value="the clone call they came with was not run")


def refuse_cancel_call(ctx, inst, event) -> None:
    """A cancel this timeline may not make at all (it is not the original, or not running): every
    id is refused. The caller stores the results."""
    cancel: Cancel = event.data
    why = ("only the original cancels; a clone can never cancel another clone" if inst.source_id is not None
           else "you can cancel only during your own turn, while you hold the conversation")
    _resolve(reply=cancel.reply, value=why)  # a str: every id is refused with it


# ------------------------------------------------------------------ effects: the end of a run
def end_run(ctx, inst, event) -> None:
    """The run ended on its own: the rows it marked as it started are marked, and the request it
    answers, if it is the run that answers it, is answered."""
    closed_out(inst)
    inst.reported, inst.unmarked = None, []
    inst.answered = inst.answered or _answers_requests(inst)


def error_for_user(ctx, inst, event) -> None:
    [w for w in inst.waiters if w.error is None][-1].error = event.data
    inst.reported = None


def end_driven(ctx, inst, event) -> None:
    _resolve(reply=event.data, value="ended")


def unprepare_failed(ctx, inst, event) -> None:
    """Rolling the cut-short call back failed: the user is told, and settle raises it. The rollback
    is kept, and written before anything next writes to the view (`_close_out`)."""
    inst.close_out = [*inst.close_out, _close_out_of(inst)]
    _report(inst=inst, kind="task", clones=[], error=event.data)


def error_shown(ctx, inst, event) -> None:
    inst.reported = None


def error_reported(ctx, inst, event) -> None:
    _report(inst=inst, kind="wake", clones=list(inst.mode.about), error=event.data)
    inst.reported = None


def set_closing(ctx, inst, event) -> None:
    inst.closing = True


def set_final(ctx, inst, event) -> None:
    inst.closing = inst.final = True


def run_failed_after_it_ended(ctx, inst, event) -> None:
    """The run raised while a step waited, and the step ended the run: its failure is reported."""
    _report(inst=inst, kind="task", clones=[], error=event.data)


def interrupted_closing(ctx, inst, event) -> None:
    inst.closing = inst.final = True


def rows_marked(ctx, inst, event) -> None:
    inst.unmarked = []
    closed_out(inst)


def to_mark(ctx, inst, event) -> None:  # entry of `withdrawing`
    """The row of the withdrawn request its run answered is to be marked."""
    inst.unmarked += [r for r in _withdrawn_rows(inst) if r not in inst.unmarked]


def mark_failed(ctx, inst, event) -> None:
    """The mark could not be written: the user is told, and the rows stay to be marked before the
    next run reads the view (`_mark_pending`), or by a recovery."""
    _report(inst=inst, kind="task", clones=[], error=event.data)


def calls_closed(ctx, inst, event) -> None:
    inst.unstored = {}


def keep_unstored(ctx, inst, event) -> None:
    """A tool its run executed whose result row could not be written: its real result is kept,
    and written when the run's open calls are closed (it ran: it is never closed as not run)."""
    use, content, is_error = event.data
    inst.unstored[use["id"]] = (content, is_error)


def calls_not_closed(ctx, inst, event) -> None:
    """The run's open calls could not be closed: the user is told, and the close is kept, written
    before anything next writes to the view (`_close_out`), so no row lands after them."""
    inst.close_out = [*inst.close_out, _close_out_of(inst)]
    _report(inst=inst, kind="task", clones=[], error=event.data)


def _answer_all(inst: TimelineSM) -> None:
    """Every caller waiting on this timeline is answered."""
    for w in inst.waiters:  # answered before they leave the data
        if w.error is not None:
            _fail(reply=w.reply, error=w.error)
        else:
            _resolve(reply=w.reply, value="ended")
    for r in inst.inbox:
        _fail(reply=r.reply, error=TransitionRefused("the harness was closed"))
    inst.waiters, inst.inbox, inst.answering = [], [], None
    _prune(inst)


def come_to_rest(ctx, inst, event) -> None:
    """At rest: every caller waiting is answered, waiting messages whose callers have gone are
    dropped, and the session told in one event. (A step that failed comes to rest without it: the
    callers wait for the recovery, since the work they wait on may not be done.)"""
    waiters = list(inst.waiters)
    for w in waiters:  # answered before they leave the data (a step that fails here leaves the rest)
        if w.error is not None:
            _fail(reply=w.reply, error=w.error)
        else:
            _resolve(reply=w.reply, value="ended")
    inst.waiters = [w for w in inst.waiters if all(w is not a for a in waiters)]
    _returned(inst=inst, requests=[w.request for w in waiters
                                   if w.request is not None and not _withdrawn(inst=inst, request=w.request)])
    _drop_dead(inst)
    inst.answering, inst.woken_for = None, []
    _prune(inst)
    _session(inst=inst, name="rested", data=inst.timeline_id)


def close_root(ctx, inst, event) -> None:
    inst.closed = True
    _answer_all(inst)
    _session(inst=inst, name="root_closed", data=inst.timeline_id)


def close_at_rest(ctx, inst, event) -> None:
    inst.closed = True
    _answer_all(inst)
    _session(inst=inst, name="root_closed", data=inst.timeline_id)


def close_and_release(ctx, inst, event) -> None:
    """Closed while suspended: the focus graft is shut down (it answers whoever waits on it and
    closes); the joins on the queue stay queued."""
    _forward(target=inst.machines.timelines.get(inst.focus), name="shutdown")


def forward_to_focus(ctx, inst, event) -> None:
    _forward(target=inst.machines.timelines.get(inst.focus), name=event.name, data=event.data)


def forward_to_source(ctx, inst, event) -> None:
    _forward(target=inst.machines.timelines.get(inst.source_id), name=event.name, data=event.data)


def return_to_source(ctx, inst, event) -> None:
    """A frozen focus graft passes a user message back to its source, which keeps it (`RETURNED`):
    the source never sends it on again, so a message goes to a focus graft and back at most once."""
    _forward(target=inst.machines.timelines.get(inst.source_id), name=RETURNED, data=event.data)


# ------------------------------------------------------------------ effects: clone
def take_clone(ctx, inst, event) -> None:
    inst.clone = event.data
    inst.cancel = event.data.cancels
    inst.declined = inst.refused = (None, None)


def clone_refused_written(ctx, inst, event) -> None:
    clone, inst.clone = inst.clone, None
    _resolve(reply=clone.reply, value=(None, None, event.data))


def clone_refusal_not_written(ctx, inst, event) -> None:
    """The refusal's result row could not be written: kept, and written with the run's open calls.
    The user is told, and the call's caller, the run, gets that error, so it ends (its closing
    writes the row) instead of calling the model on a view without it."""
    clone, inst.clone = inst.clone, None
    text = f"clone refused; no clone was created ({event.data})"
    if clone.store:
        inst.unstored[clone.use["id"]] = (text, True)
    inst.close_out = [*inst.close_out, _close_out_of(inst)]  # its audit, left open, is rolled back first
    inst.reported = event.data
    _report(inst=inst, kind="task", clones=[], error=event.data)
    _fail(reply=clone.reply, error=event.data)


def attempt_clone(ctx, inst, event) -> None:
    """The call may be tried: it uses one of the run's clone tries, and is prepared next."""
    inst.clone_tries += 1


def _outcome(rule: Callable, inst: TimelineSM) -> tuple[str | None, BaseException | None]:
    """A clone rule worked out as data (why it refuses, or None), with the error if it raised."""
    try:
        return rule(inst=inst, clone=inst.clone), None
    except Exception as error:  # noqa: BLE001 - kept for the gate's safe branch, which raises it
        return None, error


def judge_call(ctx, inst, event) -> None:
    """Before `clone_gate`: whether the whole call is declined, and why."""
    inst.declined = _outcome(rule=decline_why, inst=inst)


def decline(ctx, inst, event) -> None:
    """The whole call is not tried (nothing is seeded): its result says why. If working out the
    rules raised, that error is this step's fault (raised here, in the branch it took)."""
    why, error = inst.declined
    if error is not None:
        raise error
    inst.rejection = f"clone not executed: {why}"


def decline_failed_step(ctx, inst, event) -> None:
    """The step that brought the call failed: the call is not tried, and its result names the error
    (its rules were not worked out on what the step left)."""
    error = _pending(inst)
    inst.rejection = f"clone not executed: internal error ({type(error).__name__}: {error})"


def refuse_prepared(ctx, inst, event) -> None:
    """The prepared gate's safe branch (`refusing` rolls the call back): if working out the rules
    raised, that error is this step's fault."""
    _, error = inst.refused
    if error is not None:
        raise error


def take_prepared(ctx, inst, event) -> None:
    """The harness prepared the call: its grafts, parts and batch, or the text of its refusal."""
    if isinstance(event.data, str):
        inst.rejection = event.data
    else:
        inst.clone.pairs, inst.clone.parts, inst.clone.data = event.data
        inst.refused = _outcome(rule=clone_why, inst=inst)  # read by the prepared gate


def preparation_crashed(ctx, inst, event) -> None:
    error = event.data
    inst.rejection = f"clone refused; no clone was created: internal error ({type(error).__name__}: {error})"


def committed(ctx, inst, event) -> None:
    """The clone call is in the store: its grafts' machines are made next (`spawning`). A focus
    graft is made with the user's calls waiting on this run, and answers the same request."""
    inst.clone_text = event.data
    if inst.clone.store:  # its result until the row is written (`recording`): it ran
        inst.unstored[inst.clone.use["id"]] = (inst.clone_text, False)
    users = [w for w in inst.waiters if w.user]
    inst.spawn_plan = [GraftPlan(child=child, focus=f, waiters=users if f else [],
                                 answering=inst.answering if f else None,
                                 made_for=None if f else inst.answering)
                       for child, f in inst.clone.pairs]


def row_recorded(ctx, inst, event) -> None:
    inst.unstored.pop(inst.clone.use["id"], None)


def row_not_written(ctx, inst, event) -> None:
    """The clone call's result row could not be written: kept, and written with the run's open
    calls, by the run's closing (`closing_calls`) or, before a hand-over, by `flushing`. The user
    is told; a run that goes on ends with that error (its view would lack the row)."""
    inst.unstored[inst.clone.use["id"]] = (inst.clone_text, False)
    inst.reported = event.data
    _report(inst=inst, kind="task", clones=[], error=event.data)


def row_lost(ctx, inst, event) -> bool:
    """The clone call's result row could not be written, and the run would go on (no hand-over)."""
    return inst.focus is None and inst.clone is not None and inst.clone.use.get("id") in inst.unstored


def clone_row_lost(ctx, inst, event) -> None:
    """The clone call ran, but its result row is not in the view: its caller, the run, gets that
    error, so the run ends (its closing writes the row it kept) instead of calling the model on a
    view without it."""
    clone, inst.clone = inst.clone, None
    _fail(reply=clone.reply, error=inst.reported)


def spawned(ctx, inst, event) -> None:
    """Every graft's machine started (the event carries the clone call they were made for): only
    now are they the original's, and the session told."""
    clone = event.data
    _register_grafts(inst=inst, clone=clone)
    background = [g for g, f in clone.pairs if not f]
    _session(inst=inst, name="grafts_made", data=(background, inst.mode.name == "external"))


def _register_grafts(inst: TimelineSM, clone: Clone) -> None:
    """The grafts of a clone call whose machines started become the original's (also done by a
    recovery, for a step that failed after they started: grafts already started stay its own)."""
    for (child, focus), (_, part) in zip(clone.pairs, clone.parts or [(0, frozenset())] * len(clone.pairs)):
        inst.grafts[child] = focus
        inst.parts[child] = part
        inst.cloned_for[child] = clone.request
        inst.made[child] = focus
        if not focus and inst.answering is not None:
            inst.graft_requests[child] = inst.answering
    inst.clone_calls += 1
    inst.clone_tries = CLONE_TRIES  # one committed clone call per run
    inst.wake_clones += int(inst.mode.wake)
    inst.focus = next((g for g, f in clone.pairs if f), None)
    inst.spawn_plan = []


def spawn_not_started(ctx, inst, event) -> None:
    """The step that committed the call failed: no graft is started, and the call is undone
    (`unspawning`) as when a graft cannot start, with the step's error."""
    inst.spawned_ids, inst.spawn_error = [], _pending(inst)
    inst.spawn_plan = []
    inst.unstored.pop(inst.clone.use["id"], None)  # the call is undone: its result is the refusal


def spawn_failed(ctx, inst, event) -> None:
    """A graft's machine could not be started: the ones made are undone (`unspawning`)."""
    inst.spawned_ids, inst.spawn_error = event.data
    inst.spawn_plan = []
    inst.unstored.pop(inst.clone.use["id"], None)  # the call is undone: its result is the refusal


def unspawn_failed(ctx, inst, event) -> None:
    """The grafts of a call that could not all start could not be undone (stopped, or marked
    abandoned in the store): the user is told, and the call is refused next."""
    _report(inst=inst, kind="task", clones=[g for g, _ in inst.clone.pairs], error=event.data)


def undone(ctx, inst, event) -> None:
    """The grafts of a clone call that could not all start are stopped and forgotten: the session
    forgets them too, so nothing a graft told it before it was undone stays."""
    _session(inst=inst, name="grafts_undone", data=[g for g, _ in inst.clone.pairs])


def take_hand_over(ctx, inst, event) -> None:
    """The clone call hands the conversation over: it is finished below (the run's open calls,
    and any row kept, are written first), not answered as a call left pending."""
    inst.handing, inst.clone = inst.clone, None


def drop_clone(ctx, inst, event) -> None:
    clone, inst.clone = inst.clone, None
    _resolve(reply=clone.reply, value=(None, None, "clone not executed: the cancel calls beside it could not be recorded"))


def answer_pending(ctx, inst, event) -> None:  # exit of a run: a request it never answered is refused
    clone, inst.clone = inst.clone, None
    cancel, inst.cancel = inst.cancel, None
    if cancel is not None and inst.claims:  # left mid-way (a fault): the calls that stopped grafts ran
        inst.cut_cancels = [(call, {c: _cut_outcome(inst=inst, child=c, index=index) for c in call.ids})
                            for index, call in enumerate(cancel.calls)]
    if clone is not None:
        _resolve(reply=clone.reply, value=(None, None, "clone not executed: the turn ended before it ran"))
        if clone.cancels is not None and clone.cancels is not cancel:
            _resolve(reply=clone.cancels.reply, value="the turn ended before the cancel ran")
    if cancel is not None:
        _resolve(reply=cancel.reply, value="the turn ended before the cancel ran")


def _cut_outcome(inst: TimelineSM, child: str, index: int) -> str:
    """What became of an id of a cancel call whose step was cut short: decided (cancelled, its join
    landed, or refused), stopped with its join still to come, or never reached."""
    if inst.claims.get(child) != index:
        return "the turn ended before the cancel ran" if child not in inst.claims else (
            "it has already finished or was already cancelled; nothing to cancel")
    return inst.outcomes.get(child, CANCELLED if child in inst.pending else "the turn ended before the cancel ran")


def pass_inbox(ctx, inst, event) -> None:
    """The messages waiting on the original go to the focus graft with the conversation, in
    order and with the withdrawals recorded, ahead of any message forwarded to it later."""
    _forward(target=inst.machines.timelines.get(inst.focus), name="inbox_handed",
             data=(list(inst.inbox), set(inst.withdrawn)))
    inst.inbox = []


def hand_over(ctx, inst, event) -> None:
    """Suspended: the user's calls now wait on the focus graft, which was made with them and
    with the request it answers; the session no longer counts the original as engaged (the
    focus graft counts itself while it runs). After a recovery the clone call's caller, the run,
    has already been ended: no call is left to answer."""
    clone, inst.handing = inst.handing, None
    for w in inst.waiters:
        if not w.user:
            _resolve(reply=w.reply, value="handed_off")
    inst.waiters, inst.answering = [], None
    _prune(inst)
    if clone is not None:
        _resolve(reply=clone.reply, value=([g for g, _ in clone.pairs], inst.focus, inst.clone_text))
    _session(inst=inst, name="suspended", data=inst.timeline_id)


def clone_done(ctx, inst, event) -> None:
    clone, inst.clone = inst.clone, None
    _resolve(reply=clone.reply, value=([g for g, _ in clone.pairs], None, inst.clone_text))


# ------------------------------------------------------------------ effects: cancel
def take_cancel(ctx, inst, event) -> None:
    inst.cancel = event.data
    inst.clone = None


def start_checking(ctx, inst, event) -> None:  # entry of `checking`: the next id
    _self(inst=inst, name="check")


def checks_from(ctx, inst, event) -> None:  # entry of `cancels`: the ids to decide, in order
    cancel = inst.cancel
    inst.pending, inst.outcomes, inst.claims = set(), {}, {}
    inst.to_check = [c for call in cancel.calls for c in call.ids]
    inst.claims_next = [(i, c) for i, call in enumerate(cancel.calls) for c in call.ids]


def cancel_head(ctx, inst, event) -> None:
    """The graph allows this id: the graft leaves the live grafts at once, and `cancel` is
    forwarded to it. Its part is blocked only if its join lands as cancelled."""
    child = inst.to_check.pop(HEAD)
    index, _ = inst.claims_next.pop(HEAD)
    inst.claims[child] = index
    inst.grafts.pop(child, None)
    inst.pending.add(child)
    _forward(target=inst.machines.timelines.get(child), name="cancel")


def refuse_head(ctx, inst, event) -> None:
    child = inst.to_check.pop(HEAD)
    index, _ = inst.claims_next.pop(HEAD)
    inst.outcomes.setdefault(child, cancel_refusal(inst=inst, child=child))
    inst.claims.setdefault(child, index)


def cancel_landed(ctx, inst, event) -> None:
    """A graft this call cancelled has queued its join: its outcome is the join's reason, and its
    part is blocked for the current request only if it was cancelled (a graft that finished
    first is not)."""
    join: QueuedJoin = event.data
    inst.queue.append(join)
    inst.pending.discard(join.child)
    inst.outcomes[join.child] = join.reason
    inst.blocks |= {(inst.cancel.request, inst.parts.get(join.child, frozenset()))} if join.reason == CANCELLED else set()
    _session(inst=inst, name="graft_received", data=join.child)


def cancels_written(ctx, inst, event) -> None:
    cancel, inst.cancel = inst.cancel, None
    inst.outcomes, inst.claims = {}, {}
    _resolve(reply=cancel.reply, value=event.data[0])


def take_row_failures(ctx, inst, event) -> None:
    """Rows that could not be written are kept for the run's closing (with the real result); each
    failure is reported next, one per step, and the run ends with the first one."""
    _, failures = event.data
    for use_id, content, is_error, _stopped, _error in failures:
        inst.unstored[use_id] = (content, is_error)
    inst.unreported = [("cancel_result", stopped, error, None) for _, _, _, stopped, error in failures]
    inst.reported = failures[0][4]


def cancel_rows_failed(ctx, inst, event) -> None:
    """Every failure is reported: the cancel calls' caller gets the first one."""
    cancel, inst.cancel = inst.cancel, None
    inst.outcomes, inst.claims = {}, {}
    _fail(reply=cancel.reply, error=inst.reported)


def cancel_rows_crashed(ctx, inst, event) -> None:
    cancel, inst.cancel = inst.cancel, None
    _report(inst=inst, kind="machine", clones=[], error=event.data)
    inst.reported = event.data
    _fail(reply=cancel.reply, error=event.data)


# ------------------------------------------------------------------ effects: joins
def enqueue(ctx, inst, event) -> None:
    inst.queue.append(event.data)
    _session(inst=inst, name="graft_received", data=event.data.child)


def engage(ctx, inst, event) -> None:
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def engage_with_waiter(ctx, inst, event) -> None:
    add_run_waiter(ctx=ctx, inst=inst, event=event)
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def close_and_engage(ctx, inst, event) -> None:
    inst.closing = inst.final = True
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def focus_back(ctx, inst, event) -> None:
    """Its focus graft joined: the conversation is back, with the calls waiting on the graft,
    the messages it never got to (in order, ahead of any that came back since), the request
    it was answering, the withdrawals it recorded, the rows it could not mark (marked here
    before the next run reads the conversation) and the requests it answered whose calls are not
    done (a late cancel of one is withdrawn here)."""
    focus = inst.focus
    _take_focus_join(inst=inst, handed=event.data)
    _session(inst=inst, name="handed", data=(focus, inst.timeline_id))


def _take_focus_join(inst: TimelineSM, handed: tuple) -> None:
    """The conversation comes back with its focus graft's join (also done by a recovery, for a
    step that failed taking it: the join it carried is not lost)."""
    join, waiters, inbox, answering, withdrawn, unmarked, returned = handed
    inst.focus = None
    inst.queue.append(join)
    inst.waiters.extend(waiters)
    inst.inbox = list(inbox) + inst.inbox
    inst.answering, inst.answered = answering, join.reason == "done"
    inst.withdrawn |= withdrawn
    inst.unmarked += [r for r in unmarked if r not in inst.unmarked]
    inst.returned += [r for r in returned if all(r is not k for k in inst.returned)]


def start_joining(ctx, inst, event) -> None:  # entry of `joining`: a commit point
    inst.hold, inst.news, inst.ends_on_request, inst.woken_for = False, [], False, []


def applied(ctx, inst, event) -> None:
    """One join applied: it leaves the queue, the graft is told, and a hand-back is counted under
    the request the graft was working on (blocking that part if it failed or came back twice)."""
    committed_, request, ends = event.data
    closed_out(inst)
    join = inst.queue.pop(HEAD)
    inst.grafts.pop(join.child, None)
    inst.stopping.discard(join.child)
    news = (not inst.made.get(join.child) or join.reason != "done") and not _unwanted(inst=inst, child=join.child)
    inst.news += [(join.child, join.reason)] if news else []
    # the wake its news sets off is for the request the graft was made for, if any
    inst.woken_for += [r for r in [inst.graft_requests.pop(join.child, None)] if news and r is not None]
    inst.ends_on_request = ends
    if join.reason not in ("done", CANCELLED):
        part = inst.parts.get(join.child, frozenset())
        inst.handbacks[request] = inst.handbacks.get(request, NONE_YET) + 1
        n = inst.arrivals[(request, part)] = inst.arrivals.get((request, part), NONE_YET) + 1
        inst.blocks |= {(request, part)} if join.reason == "failed" or n >= 2 else set()
    _forward(target=inst.machines.timelines.get(join.child), name="join_commit" if committed_ else "join_rollback")


def apply_failed(ctx, inst, event) -> None:
    """A join could not be applied: it and those behind it wait for the next commit point."""
    child = inst.queue[0].child
    attempts = inst.attempts[child] = inst.attempts.get(child, NONE_YET) + 1
    inst.hold = True
    _report(inst=inst, kind="commit", clones=[j.child for j in inst.queue], error=event.data, quiet=attempts > 1)


def last_commit_failed(ctx, inst, event) -> None:
    """Its last attempt failed (a repeat for the user): the join is given up in the next steps."""
    _report(inst=inst, kind="commit", clones=[inst.queue[0].child], error=event.data, quiet=True, keep=False)


def report_abandoned(ctx, inst, event) -> None:
    _report(inst=inst, kind="abandon", clones=[inst.queue[0].child], error=event.data, keep=False)


def abandon_head(ctx, inst, event) -> None:
    """The given-up join leaves the queue, and the graft is told; the queue moves on."""
    join = inst.queue.pop(HEAD)
    inst.grafts.pop(join.child, None)
    inst.stopping.discard(join.child)
    inst.graft_requests.pop(join.child, None)
    _forward(target=inst.machines.timelines.get(join.child), name="join_abandon")


def answer_handback(ctx, inst, event) -> None:
    inst.clone_tries = 0
    inst.mode = Mode(name="turn", user=True, about=tuple(c for c, _ in inst.news))


def wake_route(ctx, inst, event) -> None:
    focus_back = any(r != "done" and inst.made.get(c) for c, r in inst.news)
    inst.clone_tries = 0
    inst.mode = Mode(name="route", wake=not focus_back, about=tuple(c for c, _ in inst.news))


def wake_report(ctx, inst, event) -> None:
    inst.clone_tries = 0
    inst.mode = Mode(name="report", report=True, about=tuple(c for c, _ in inst.news))


def note_written(ctx, inst, event) -> None:
    closed_out(inst)


def note_failed(ctx, inst, event) -> None:
    """The wake note could not be written: the user is told; the original comes to rest next."""
    _report(inst=inst, kind="wake", clones=[c for c, _ in inst.news], error=event.data)


# ------------------------------------------------------------------ effects: a focus graft's join
def _hand_back_of(event: hsm.Event | None) -> tuple[QueuedJoin | None, list[Waiter]]:
    """The join a focus graft's hand-back step carries, and the caller waiting on it: the join
    computed by the state just left, or one raised from outside with its caller (`join`); else
    (its failure could not be recorded) it joins `failed`."""
    data = event.data if event is not None else None
    if isinstance(data, QueuedJoin):
        return data, []
    if isinstance(data, tuple) and data and isinstance(data[HEAD], QueuedJoin):
        join, reply = data
        return join, [Waiter(reply=reply, user=False)] if reply is not None else []
    return None, []


def _hand_back(inst: TimelineSM, event: hsm.Event | None) -> None:
    """Its join event, with the calls and messages waiting on it, is kept in its data first (so a
    fault after this keeps it, and the recovery sends it), then goes to its source."""
    join, also = _hand_back_of(event=event)
    join = join or QueuedJoin(child=inst.timeline_id, reason="failed", trigger=None)
    users = [w for w in inst.waiters if w.user] + also
    others = [w for w in inst.waiters if not w.user]
    inst.handed_back = (join, users, list(inst.inbox), inst.answering, set(inst.withdrawn), list(inst.unmarked),
                        list(inst.returned))
    inst.waiters, inst.inbox, inst.answering = [], [], None
    for w in others:
        _resolve(reply=w.reply, value="joined")
    _send_hand_back(inst)


def _send_hand_back(inst: TimelineSM) -> None:
    """Its join event goes to its source; once sent it is not kept (a fault in the step that sent
    it, before the send, leaves it kept, and the recovery sends it)."""
    _forward(target=inst.machines.timelines.get(inst.source_id), name="focus_joined", data=inst.handed_back)
    inst.handed_back, inst.hand_back_sent = None, True


def focus_join(ctx, inst, event) -> None:
    """A focus graft raises its join event: frozen, its join goes on its source's queue, with the
    calls and messages waiting on it; the caller of the join is answered at the source's rest."""
    _hand_back(inst=inst, event=event)


def hand_back_join(ctx, inst, event) -> None:
    """Its join event (computed by the state just left) goes to its source."""
    _hand_back(inst=inst, event=event)


def focus_failure_not_recorded(ctx, inst, event) -> None:
    """Its failure could not be recorded: the user is told; it joins `failed` next."""
    _report(inst=inst, kind="task", clones=[inst.timeline_id], error=event.data, src=inst.source_id)


def focus_failed_unrecorded(ctx, inst, event) -> None:
    _hand_back(inst=inst, event=event)  # the event is its failure: it joins `failed`


# ------------------------------------------------------------------ effects: a background graft
def ended_as(reason: str, close_first: bool):
    def effect(ctx, inst, event) -> None:
        inst.reason, inst.close_first = reason, close_first
    effect.__name__ = f"ended_{reason}{'_early' if close_first else ''}"
    return effect


def ended_by_join(ctx, inst, event) -> None:
    inst.reason, inst.close_first = event.data.reason, False


def forced(ctx, inst, event) -> None:
    """Its join is raised from outside its run (tests, replay): the join's reason is its reason."""
    join, inst.forced = event.data
    inst.reason, inst.close_first = join.reason, True


def run_raised(ctx, inst, event) -> None:
    inst.reason, inst.close_first = "failed", True
    _note(inst=inst, method="graft_failed", tl=inst.timeline_id, error=event.data)


def _unreported(inst: TimelineSM, errors: list[BaseException]) -> list[tuple]:
    kind = "cancel" if inst.reason == CANCELLED else "task"
    return [(kind, [inst.timeline_id], e, inst.source_id) for e in errors]


def recorded(ctx, inst, event) -> None:
    inst.unreported += _unreported(inst=inst, errors=event.data)


def record_crashed(ctx, inst, event) -> None:
    inst.unreported += _unreported(inst=inst, errors=[event.data])


def join_raised(ctx, inst, event) -> None:
    """Its join event goes to its source; whoever waited for it is answered."""
    for w in inst.watchers:  # answered before they leave the data
        _resolve(reply=w, value=True)
    _resolve(reply=inst.forced, value=True)
    inst.watchers = []
    _send_join(inst)


def _send_join(inst: TimelineSM) -> None:
    """A background graft's join event goes to its source (recorded once sent: a fault in the step
    that sends it, before the send, leaves it unsent, and the recovery sends it)."""
    _forward(target=inst.machines.timelines.get(inst.source_id), name="graft_queued",
             data=QueuedJoin(child=inst.timeline_id, reason=inst.reason, trigger=None))
    inst.join_sent = True


# ------------------------------------------------------------------ effects: failures told one per step
def report_next(ctx, inst, event) -> None:  # entry of a reporting state: the next failure
    _self(inst=inst, name="report_next")


def has_unreported(ctx, inst, event) -> bool:
    return bool(inst.unreported)


def all_reported(ctx, inst, event) -> bool:
    """Every failure was told, and the step has not failed: the caller gets the first one."""
    return step_intact(ctx=ctx, inst=inst, event=event) and not inst.unreported


def report_one(ctx, inst, event) -> None:
    kind, clones, error, src = inst.unreported.pop(HEAD)
    _report(inst=inst, kind=kind, clones=clones, error=error, src=src)


def add_watcher(ctx, inst, event) -> None:
    inst.watchers.append(event.data)


# ------------------------------------------------------------------ effects: recovering from a fault
# A user message a frozen focus graft passed back to its source: the source keeps it (it is taken
# like a user message, and never forwarded again), so no message bounces between the two.
RETURNED = "message_returned"
MESSAGES = ("user_message", RETURNED)
REQUESTS = ("turn_request", "drain_request", "join", "user_message", "settle_request", "driven_done")
# Recoveries a timeline makes in a row before it gives a fault up (a fault that recovery itself
# would hit again, e.g. a send that always fails): a new user message or run starts the count again.
MAX_RECOVERIES = 2


class Record(NamedTuple):
    """A timeline's record as the session keeps it, re-stated by the timeline after a fault
    (`resync`), whatever the failed step told the session or did not: whether it is engaged; for
    the original, its background grafts whose joins have not reached it and those of them made in
    a turn its caller drives (`held`); whether it has closed; and the fault's errors no caller was
    given, for the session to keep and show. `focus` (the original): the focus graft that holds
    the conversation, if any (no other timeline does). `also`: the records of the timelines that
    passed the session's `restate` on to this one (the suspended original's, re-stated by its
    focus graft with its own, in one event)."""
    tl: str
    engaged: bool | None  # None: as the session has it (a call's step failed: nothing else changed)
    grafts: tuple[str, ...] | None
    held: tuple[str, ...]
    closed: bool
    errors: tuple[BaseException, ...]
    focus: str | None = None
    also: tuple["Record", ...] = ()


class Restate(NamedTuple):
    """The session's request for a record (its own step failed): the errors no caller was given,
    to come back with the record, and the records of the timelines that passed it on."""
    errors: tuple[BaseException, ...]
    also: tuple[Record, ...] = ()


def _awaited(inst: TimelineSM) -> list[str]:
    """Its background grafts whose joins have not reached it: those still running, and those a
    cancel stopped whose joins have not come yet."""
    queued = {q.child for q in inst.queue}
    return _still_running(inst) + sorted(g for g in inst.stopping if g not in queued)


def _record_of(inst: TimelineSM, engaged: bool | None, errors: list[BaseException],
               also: tuple[Record, ...] = ()) -> Record:
    running = _awaited(inst) if inst.kind == "original" else None
    return Record(tl=inst.timeline_id, engaged=engaged,
                  grafts=None if running is None else tuple(g for g in running if g not in inst.held_grafts),
                  held=tuple(g for g in running or [] if g in inst.held_grafts), closed=inst.closed,
                  errors=tuple(errors), focus=inst.focus, also=also)


def recover_call(ctx, inst, event) -> None:
    """A step that only took a call from outside failed (an internal transition: nothing else of
    the timeline's work was touched): that call's caller gets the error (its message, if it was
    kept, leaves the inbox once answered), and the session gets the record as it stands."""
    fault = _take_fault(inst=inst, event=event)
    replies = _caller_of(event=fault.event)
    _salvage(inst=inst, event=fault.event)
    _requeue_strays(inst)
    told = _told(fault=fault, replies=replies)
    _session(inst=inst, name="resync", data=_record_of(
        inst=inst, engaged=None, errors=_errors_for_session(errors=fault.errors, told=told)))
    fault.told = _fail_all(replies=replies, error=fault.error) or fault.told
    _returned(inst=inst, requests=[fault.event.data])
    inst.inbox = [r for r in inst.inbox if r is not fault.event.data]


def _caller_of(event: hsm.Event | None) -> list[asyncio.Future]:
    """The replies the event whose step failed carries: a caller outside the machines waits on
    each (a user's call, a settle, a run's clone or cancel request or its `may_continue`), so
    each is that caller's to be given the error. Read from the payload, whatever the event: a
    clone request carries its own reply and that of the cancel calls it brought."""
    if event is None:
        return []
    data = event.data
    if isinstance(data, asyncio.Future):
        return [data]
    if isinstance(data, tuple):
        return [x for x in data if isinstance(x, asyncio.Future)]
    carried = [getattr(data, "reply", None), getattr(getattr(data, "cancels", None), "reply", None)]
    return [r for r in carried if isinstance(r, asyncio.Future)]


def _salvage(inst: TimelineSM, event: hsm.Event | None) -> None:
    """What the event whose step failed brought that the step did not keep: a graft's join (kept
    for the queue), a focus graft's join (the conversation comes back with it), the grafts of a
    clone call whose machines had started (they stay its own), a shutdown (it still closes). Each
    is kept once: a step that failed after it kept them changed nothing here."""
    name = event.name if event is not None else None
    if name == "shutdown":  # the timeline was shut down: it closes once it is at rest, as without the fault
        inst.closing = inst.final = True
    elif name == "graft_queued" and isinstance(event.data, QueuedJoin):
        join = event.data
        if all(j.child != join.child for j in inst.queue + inst.strays):
            inst.strays.append(join)
    elif name == "focus_joined" and inst.focus is not None and event.data[HEAD].child == inst.focus:
        _take_focus_join(inst=inst, handed=event.data)
    elif name == "spawned" and isinstance(event.data, Clone) and any(g not in inst.made for g, _ in event.data.pairs):
        _register_grafts(inst=inst, clone=event.data)


def _fail_all(replies: list[asyncio.Future | None], error: BaseException) -> bool:
    """Each caller still waiting gets the error, once; were any given it?"""
    told = False
    for reply in replies:
        if reply is not None and not reply.done():
            reply.set_exception(error)
            told = True
    return told


def _told(fault: Fault, replies: list[asyncio.Future]) -> bool:
    """Is the fault's first error a caller's to report: given to one already (by a handling of it
    that then raised), or about to be?"""
    return fault.told or any(not r.done() for r in replies)


def _unanswered(inst: TimelineSM) -> list[Waiter]:
    """The waiters of the user call whose request no run has answered yet (its run was ended by
    the recovery, or the step that would start it failed), unless a focus graft holds the
    conversation and that call."""
    request = inst.answering if inst.focus is None and not inst.answered else None
    return [w for w in inst.waiters if request is not None and w.request is request and w.error is None]


def _callers_to_fail(inst: TimelineSM) -> list[asyncio.Future]:
    """The callers waiting on the work a failed step left open: the caller of the event whose step
    failed, the message whose row was being written, and the user call whose request no run has
    answered yet (`_unanswered`). Read, not taken: they leave the machine's data only once answered
    (`_drop_failed`), so a handling that raises and runs again still finds the ones it did not
    answer. Every other caller waits on work the failed step did not touch: a waiting message is
    taken once the timeline is at rest, and a call whose request a run answered is answered at rest."""
    replies = _caller_of(event=inst.faulted)
    replies += [inst.request.reply] if inst.request is not None else []
    return replies + [w.reply for w in _unanswered(inst)]


def _drop_failed(inst: TimelineSM) -> None:
    """The callers given the failed work's error leave the machine's data. The user requests
    among them are kept as answered (`returned`) until their calls are done: a cancel that lands
    before the caller took the error wins, and the error is told then."""
    faulted = inst.faulted.data if inst.faulted is not None else None
    users = _unanswered(inst)
    _returned(inst=inst, requests=[faulted, inst.request, *[w.request for w in users]])
    inst.inbox = [r for r in inst.inbox if r is not faulted]
    inst.waiters = [w for w in inst.waiters if all(w is not u for u in users)]
    inst.answering = None if inst.focus is None and not inst.answered else inst.answering
    inst.request = None


def _returned(inst: TimelineSM, requests: list) -> None:
    """User requests answered (a reply set, not cancelled) are kept until their calls are done."""
    inst.returned += [r for r in requests if isinstance(r, Request) and r.reply.done() and not r.reply.cancelled()
                      and all(r is not k for k in inst.returned)]


def _restate_and_fail(inst: TimelineSM, fault: Fault, errors: list[BaseException], engaged: bool,
                      also: list[asyncio.Future] | None = None) -> None:
    """The end of a recovery: the callers waiting on the failed work get its first error, once,
    and the session gets the timeline's whole record, with the errors no caller was given (it
    keeps and shows them). The record is sent when the step returns, before any caller answered
    here runs; the callers leave the data once answered."""
    replies = _callers_to_fail(inst) + list(also or [])
    _session(inst=inst, name="resync", data=_record_of(
        inst=inst, engaged=engaged, errors=_errors_for_session(errors=errors, told=_told(fault=fault, replies=replies))))
    fault.told = _fail_all(replies=replies, error=errors[HEAD]) or fault.told
    _drop_failed(inst)
    _prune(inst)


def _errors_for_session(errors: list[BaseException], told: bool) -> list[BaseException]:
    """A fault's first error given to a caller is that caller's to report: the rest are the session's."""
    return errors[1:] if told else errors


def may_recover(ctx, inst, event) -> bool:
    return inst.recoveries < MAX_RECOVERIES


def begin_recovery(ctx, inst, event) -> None:
    """A step of this timeline failed (the run, if one was going, has been ended by leaving its
    state, and any clone or cancel request it held answered): what the step left open is closed
    out next (`recovering`). What the failed step's event brought is kept (`_salvage`). Grafts a
    cancel stopped whose joins had not come stay its own (`stopping`): no longer live, and their
    joins are applied when they come. Safe to run again for the same fault (a handling of it that
    raised): what it keeps is kept once."""
    fault = _take_fault(inst=inst, event=event)
    inst.handling, inst.faulted = fault, fault.event
    inst.recoveries += 1
    inst.recovery = list(fault.errors) + [e for e in inst.recovery if all(e is not x for x in fault.errors)]
    _salvage(inst=inst, event=fault.event)
    inst.unmarked += [r for r in _withdrawn_rows(inst) if r not in inst.unmarked]
    inst.stopping |= inst.pending  # cancelled: their joins are still its own (one may be kept above)
    inst.pending, inst.to_check, inst.claims_next, inst.outcomes, inst.claims = set(), [], [], {}, {}
    inst.spawn_plan, inst.reported = [], None
    _session(inst=inst, name="resync", data=_record_of(inst=inst, engaged=True, errors=[]))


def fold_fault(ctx, inst, event) -> None:
    """A fault of another step while recovering is part of the recovery: its errors are told with
    the rest, and its callers get its error."""
    fault = _take_fault(inst=inst, event=event)
    told = _fail_all(replies=_caller_of(event=fault.event), error=fault.error) or fault.told
    inst.recovery += [e for e in _errors_for_session(errors=fault.errors, told=told)
                      if all(e is not x for x in inst.recovery)]


def give_up_fault(ctx, inst, event) -> None:
    """A fault past MAX_RECOVERIES in a row: the session gets the record as it stands (with the
    errors), every caller waiting on the timeline's work gets the error, and the timeline goes on
    from rest without closing anything out (the waiting messages are still taken, one at a time)."""
    fault = _take_fault(inst=inst, event=event)
    inst.faulted = fault.event
    _salvage(inst=inst, event=fault.event)
    inst.stopping, inst.pending = inst.stopping | inst.pending, set()
    _requeue_strays(inst)
    _restate_and_fail(inst=inst, fault=fault, errors=fault.errors, engaged=False)
    inst.faulted = None


def _end_recovery(inst: TimelineSM, errors: list[BaseException]) -> None:
    """The callers waiting on the failed work get its first error, once, and the session gets the
    timeline's whole record, with the errors no caller was given (it keeps and shows them). What
    follows is decided as after any run (`resumed`). Part of handling the fault (`Recovers`): if
    it raises, the fault is handled again."""
    _requeue_strays(inst)
    _restate_and_fail(inst=inst, fault=inst.handling, errors=errors, engaged=True)
    inst.recovery, inst.unstored, inst.unmarked, inst.cut_cancels = [], {}, [], []
    inst.faulted = inst.handling = None


def recovered(ctx, inst, event) -> None:
    """Closed out (a close-out kept from before as well): the recovery ends (`_end_recovery`)."""
    closed_out(inst)
    _end_recovery(inst=inst, errors=list(inst.recovery))


def recovery_failed(ctx, inst, event) -> None:
    """Closing out failed as well: its error is told with the fault's, and what it could not write
    (the run's open calls, the clone call's rollback, the rows to mark) is kept as one close-out
    (`close_out`), written before anything next writes to or reads the view (`_close_out`)."""
    kept = [*inst.close_out, _close_out_of(inst)]
    _end_recovery(inst=inst, errors=[*inst.recovery, event.data])
    inst.close_out = kept


def fault_suspended(ctx, inst, event) -> None:
    """A step of the suspended original failed (its focus graft holds the conversation, and the
    calls waiting on it): the messages it still holds (one its step did not forward, waiting
    messages not handed over) get the error, and the session its record."""
    fault = _take_fault(inst=inst, event=event)
    inst.faulted = fault.event
    held = list(inst.inbox)
    _salvage(inst=inst, event=fault.event)
    _requeue_strays(inst)
    _restate_and_fail(inst=inst, fault=fault, errors=fault.errors, engaged=False, also=[r.reply for r in held])
    _returned(inst=inst, requests=held)
    inst.inbox = [r for r in inst.inbox if all(r is not h for h in held)]
    inst.faulted = None


def fault_releasing(ctx, inst, event) -> None:
    """A step of the original being closed while suspended failed: it closes now (its focus graft,
    if it was not told, is stopped with every machine)."""
    fault = _take_fault(inst=inst, event=event)
    inst.closed = True
    replies = _caller_of(event=fault.event)
    _session(inst=inst, name="resync", data=_record_of(inst=inst, engaged=False, errors=_errors_for_session(
        errors=fault.errors, told=_told(fault=fault, replies=replies))))
    fault.told = _fail_all(replies=replies, error=fault.error) or fault.told
    _answer_all(inst)


def take_end_fault(ctx, inst, event) -> None:
    """A step failed after the timeline's run (joined, abandoned, closed): its join event is sent if
    the step did not send it, and the fault is told next."""
    fault = _take_fault(inst=inst, event=event)
    inst.faulted = fault.event
    replies = _caller_of(event=fault.event)
    told = _told(fault=fault, replies=replies)
    fault.told = _fail_all(replies=replies, error=fault.error) or fault.told
    inst.recovery = _errors_for_session(errors=fault.errors, told=told)


def hand_back_unsent(ctx, inst, event) -> bool:
    """A frozen focus graft whose join event never reached its source."""
    return inst.kind == "focus" and not inst.hand_back_sent


def resend_hand_back(ctx, inst, event) -> None:
    """Its join event goes to its source: the one it kept, or, if the step failed before it kept
    it, one made again from the event of that step and what it still holds."""
    faulted = inst.faulted
    if inst.handed_back is None:
        _hand_back(inst=inst, event=faulted)
    else:
        _send_hand_back(inst)


def tell_end_fault(ctx, inst, event) -> None:
    """The session gets its record (it is not engaged; it holds no user call to re-state)."""
    _session(inst=inst, name="resync", data=_record_of(inst=inst, engaged=False, errors=inst.recovery))
    inst.recovery, inst.faulted = [], None


def fault_at_end(ctx, inst, event) -> None:  # in a final state: one event, the record
    """A step that ended in a final state failed. A timeline that neither joined nor was abandoned
    has closed (the original only ends that way), whatever the step did: every caller still waiting
    on it is answered, and its record says it closed, so `close` goes on."""
    closed = inst.ended not in ("joined", "abandoned")
    inst.closed = inst.closed or closed
    take_end_fault(ctx=ctx, inst=inst, event=event)
    tell_end_fault(ctx=ctx, inst=inst, event=event)  # the record goes even if answering below raises
    if closed:
        _answer_all(inst)


def restating(engaged: bool):
    """The session asks for the timeline's record (its own step failed): sent as it stands, with
    the errors the session is to keep and show and the records passed on with the question."""
    def effect(ctx, inst, event) -> None:
        restate: Restate = event.data
        _session(inst=inst, name="resync", data=_record_of(inst=inst, engaged=engaged, errors=list(restate.errors),
                                                            also=restate.also))
    effect.__name__ = f"restate_{'engaged' if engaged else 'at_rest'}"
    return effect


def restate_failed(ctx, inst, event) -> bool:
    """The call whose step failed was the session's request for this timeline's record."""
    return event.data.event.name == "restate"


def restate_failed_through_focus(ctx, inst, event) -> bool:
    """Suspended under a focus graft whose machine is there to ask, and the step that passed the
    session's request on to it failed."""
    return restate_failed(ctx=ctx, inst=inst, event=event) and focus_restates(ctx=ctx, inst=inst, event=event)


def restate_failed_alone(ctx, inst, event) -> bool:
    return restate_failed(ctx=ctx, inst=inst, event=event) and not focus_restates(ctx=ctx, inst=inst, event=event)


def restating_again(engaged: bool):
    """The step that answered the session's request for the record failed: the record is sent as
    it stands, with the errors the session was to keep and show and those of the failed step (no
    caller waits on them), and the records passed on with the request."""
    def effect(ctx, inst, event) -> None:
        fault = _take_fault(inst=inst, event=event)
        restate: Restate = fault.event.data
        _session(inst=inst, name="resync", data=_record_of(
            inst=inst, engaged=engaged, errors=[*restate.errors, *fault.errors], also=restate.also))
    effect.__name__ = f"restate_again_{'engaged' if engaged else 'at_rest'}"
    return effect


def restate_through_focus_again(ctx, inst, event) -> None:
    """Suspended, the step that passed the session's request on to the focus graft failed: it is
    passed on again, with the errors of the failed step added (the graft re-states both records)."""
    fault = _take_fault(inst=inst, event=event)
    restate: Restate = fault.event.data
    own = _record_of(inst=inst, engaged=False, errors=[])
    _forward(target=inst.machines.timelines.get(inst.focus), name="restate",
             data=Restate(errors=(*restate.errors, *fault.errors), also=(*restate.also, own)))


def focus_restates(ctx, inst, event) -> bool:
    """The original is suspended under a focus graft whose machine is there to ask."""
    return inst.focus in inst.machines.timelines


def restate_through_focus(ctx, inst, event) -> None:
    """Suspended, the original passes the session's question on to its focus graft, with its own
    record: the graft re-states both in one event (its record may be stale too)."""
    restate: Restate = event.data
    own = _record_of(inst=inst, engaged=False, errors=[])
    _forward(target=inst.machines.timelines.get(inst.focus), name="restate",
             data=Restate(errors=restate.errors, also=(*restate.also, own)))


def keep_stray(ctx, inst, event) -> None:
    """A join whose step failed before it was decided: kept, not lost (the recovery queues it if the
    graft is one of its own)."""
    inst.strays.append(event.data)


def _requeue_strays(inst: TimelineSM) -> None:
    """Joins kept from a failed step go on the queue if they are its own grafts' (and not queued)."""
    for join in inst.strays:
        if _own(inst=inst, child=join.child) and all(q.child != join.child for q in inst.queue):
            inst.queue.append(join)
    inst.strays = []


# ------------------------------------------------------------------ effects: a background graft's fault
def _bg_errors(inst: TimelineSM, event: hsm.Event) -> None:
    """A background graft's fault: the caller of its step, if any, gets the error, and every error
    is told, one per step, before its join is raised (`reporting`). Each error is kept once (a
    handling of the fault that raised runs again)."""
    fault = _take_fault(inst=inst, event=event)
    fault.told = _fail_all(replies=_caller_of(event=fault.event), error=fault.error) or fault.told
    kept = [e for _, _, e, _ in inst.unreported]
    inst.unreported += [("task", [inst.timeline_id], e, inst.source_id) for e in fault.errors
                        if all(e is not k for k in kept)]


def bg_fault_stops(ctx, inst, event) -> None:
    """A step of a running (or held) background graft failed: it stops, and joins `failed`."""
    inst.reason, inst.close_first = "failed", True
    _bg_errors(inst=inst, event=event)


def bg_fault_rewrites(ctx, inst, event) -> None:
    """The step that entered `writing` failed, so its write held back: it is written now, with its
    open calls closed first (its run may have ended early), and the fault told with the rest."""
    inst.reason, inst.close_first = inst.reason or "failed", True
    _bg_errors(inst=inst, event=event)


def bg_fault_noted(ctx, inst, event) -> None:
    """A step failed while it records its join: the reason stands (`failed` if none was set yet)."""
    _bg_errors(inst=inst, event=event)
    inst.reason = inst.reason or "failed"


def join_unsent(ctx, inst, event) -> bool:
    return not inst.join_sent


def resend_join(ctx, inst, event) -> None:
    _send_join(inst)


def bg_take_end_fault(ctx, inst, event) -> None:
    _bg_errors(inst=inst, event=event)


def bg_tell_end_fault(ctx, inst, event) -> None:
    """Its errors are told (one line, kept for settle)."""
    errors, inst.unreported = [e for _, _, e, _ in inst.unreported], []
    if errors:
        error = errors[HEAD] if len(errors) == 1 else ExceptionGroup("a step failed", errors)
        _report(inst=inst, kind="task", clones=[inst.timeline_id], error=error, src=inst.source_id)


def bg_fault_at_end(ctx, inst, event) -> None:  # in a final state: one event, the line
    bg_take_end_fault(ctx=ctx, inst=inst, event=event)
    bg_tell_end_fault(ctx=ctx, inst=inst, event=event)


# ------------------------------------------------------------------ the graphs


def _work(name: str, *activity: hsm.RedefinableBehaviors) -> hsm.RedefinableState:
    """What a running timeline does: run its model loop (`run`, an activity) or wait for the
    caller that drives the turn (`driven`). Both clone and cancel, step by step. A clone call is
    gated whole first (`clone_gate`: nothing is seeded for a call that may not be tried), then
    prepared (`preparing`, the harness's activity: leaving the run ends it), then decided and carried
    out in `steps/cloning`; the cancel calls of one response are `steps/cancels`, and a clone call
    carried with them goes to the gate after them. While a step runs, shutdown and a driven end
    wait (`steps`)."""
    T, S, On, G, E = hsm.Transition, hsm.Target, hsm.On, Guard, hsm.ErrorEvent
    return hsm.State(
        name,
        *activity,
        Exit(answer_pending),  # however the run is left, a clone or cancel request it holds is answered
        hsm.Initial(S("running")),
        hsm.State(
            "running",
            T(On("clone_request"), S("../cancel_gate"), G(carries_cancels), Effect(take_clone)),
            T(On("clone_request"), S("../clone_gate"), G(no_cancels), Effect(take_clone, judge_call)),
            T(On("clone_request"), Call(checked(carries_cancels, no_cancels, effects=(refuse_clone_call,)))),
            T(On("cancel_request"), S("../cancel_gate"), Effect(take_cancel)),
        ),
        hsm.Choice(
            "cancel_gate",  # a step that failed taking the cancels does not carry them out: the
            # recovery that follows ends the run, which answers them
            T(S("steps/cancels"), G(step_intact)),
            T(S("running"), Effect(checked(step_intact))),
        ),
        hsm.Choice(
            "clone_gate",  # the whole call: may it be tried at all? Decided before anything is seeded
            T(S("preparing"), G(may_try), Effect(attempt_clone)),
            T(S("steps/cloning/rejecting"), G(declines), Effect(decline)),
            # the step failed (its rules were not worked out on what it brought): refused with its error
            T(S("steps/cloning/rejecting"), Effect(checked(may_try, declines, effects=(decline_failed_step,)))),
        ),
        hsm.State(
            "preparing", Activity(work=prepare_activity, done="prepared"),
            T(On("prepared"), S("../prepared_gate"), Effect(take_prepared)),
            T(On(E), S("../steps/cloning/rejecting"), Effect(preparation_crashed)),
            # left early (the run's request withdrawn, the run raised, shutdown, a driven turn
            # ended): the call it was preparing is rolled back in a state of its own before the
            # run's closing. A run that raised is told the way `busy` tells it (only the original
            # prepares a clone call, so the focus graft's branch does not arise here)
            T(On("withdraw"), S("../../../unpreparing"), G(ends_run), Effect(note_withdrawn)),
            T(On("withdraw"), G(withdraws_late), Call(note_late_withdrawal)),
            # its own safe branch: recorded; a guard that raised is a fault of the timeline (not of
            # the call alone), so its recovery ends the run and rolls the call back
            T(On("withdraw"), Effect(checked(ends_run, withdraws_late, effects=(note_withdrawn,), first=True))),
            T(On("run_failed"), S("../../../unpreparing"), G(error_to_user), Effect(error_for_user)),
            T(On("run_failed"), S("../../../unpreparing"), G(error_already_shown), Effect(error_shown)),
            T(On("run_failed"), S("../../../unpreparing"),
              Effect(checked(error_to_user, error_already_shown, effects=(error_reported,)))),
            T(On("shutdown"), S("../../../unpreparing"), Effect(interrupted_closing)),
            T(On("driven_done"), S("../../../unpreparing"), Effect(end_driven)),
        ),
        hsm.Choice(
            "prepared_gate",  # a prepared call the graph does not allow is rolled back (`refusing`)
            T(S("steps/cloning/rejecting"), G(rejected_in_preparation)),
            T(S("steps/cloning/committing"), G(graph_allows)),
            T(S("steps/cloning/refusing"), Effect(checked(rejected_in_preparation, graph_allows,
                                                           effects=(refuse_prepared,)))),
        ),
        hsm.State(
            "steps",  # one clone or cancel request, step by step: shutdown, a driven end and the
            # failure of the run (a clone or cancel request from outside it) wait
            hsm.Defer("shutdown", "driven_done", "run_failed"),
            hsm.State(
                "cloning",  # one clone call: committed, its machines started, its result written
                # a graft made here that joins at once (its request withdrawn) is queued after
                # the step, once the original knows it made it
                hsm.Defer("focus_joined", "graft_queued"),
                # a withdrawal of a waiting message drops it here; any other is recorded and reaches
                # the focus graft once it is made; the run ends once the step is done (`resume`), and
                # no graft made for it acts on it
                T(On("withdraw"), G(withdraws_waiting), Call(note_withdrawn)),
                T(On("withdraw"), G(withdraws_late), Call(note_late_withdrawal)),
                T(On("withdraw"), Call(checked(withdraws_waiting, withdraws_late, effects=(withdraw_and_forward,)))),
                hsm.State(
                    "rejecting", io(done="rejected", work=_reject),  # refused before or in its preparation
                    T(On("rejected"), S("../../../resume"), Effect(clone_refused_written)),
                    T(On(E), S("../../../resume"), Effect(clone_refusal_not_written)),
                ),
                hsm.State(
                    "refusing", io(done="refused", work=_refuse),
                    T(On("refused"), S("../../../resume"), Effect(clone_refused_written)),
                    T(On(E), S("../../../resume"), Effect(clone_refusal_not_written)),
                ),
                hsm.State(
                    "committing", io(done="committed", work=_commit),
                    T(On("committed"), S("../spawn_gate"), Effect(committed)),
                    T(On(E), S("../refusing")),
                ),
                hsm.Choice(
                    "spawn_gate",  # a step that failed starts no graft: the committed call is undone
                    T(S("spawning"), G(step_intact)),
                    T(S("unspawning"), Effect(checked(step_intact, effects=(spawn_not_started,), first=True))),
                ),
                hsm.State(
                    "spawning", hsm.Activity(spawn_activity),
                    # a withdrawal waits for the grafts' machines (the activity reads no data of
                    # this one after it starts), then is handled by `cloning` like any later one
                    # a fault waits too: the call is then undone (`unspawning`), and recovered from
                    hsm.Defer("withdraw", "graft_step", FAULT),
                    T(On("spawned"), S("../recording"), Effect(spawned)),
                    T(On("spawn_failed"), S("../unspawning"), Effect(spawn_failed)),
                ),
                hsm.State(
                    "unspawning", hsm.Activity(unspawn_activity),
                    # a graft's question waits too: while a fault is pending no guard holds, and it would be stopped
                    hsm.Defer("graft_step", FAULT),
                    T(On("unspawned"), S("../refusing"), Effect(undone)),
                    T(On(E), S("../refusing"), Effect(unspawn_failed)),
                ),
                hsm.State(
                    "recording", io(done="row_written", work=_clone_row),
                    T(On("row_written"), S("../../../clone_choice"), Effect(row_recorded)),
                    T(On(E), S("../../../clone_choice"), Effect(row_not_written)),
                ),
                hsm.State(
                    "flushing", io(done="flushed", work=_close_calls),  # before a hand-over: the run's rows
                    T(On("flushed"), S("../../../handed"), Effect(calls_closed)),
                    T(On(E), S("../../../handed"), Effect(calls_not_closed)),
                ),
            ),
            hsm.State(
                "cancels",  # the cancel calls of one response, id by id; a clone call carried with them waits
                Entry(checks_from),
                hsm.Defer("clone_request"),
                T(On("withdraw"), G(withdraws_late), Call(note_late_withdrawal)),
                T(On("withdraw"), Call(checked(withdraws_late, effects=(note_withdrawn,)))),  # the run ends once the cancels are done
                T(On("graft_queued"), G(one_of_pending), Effect(cancel_landed)),
                *_join_arrivals(T=T, On=On, G=G, besides=(one_of_pending,)),
                hsm.Initial(S("checking")),
                hsm.State(
                    "checking", Entry(start_checking),
                    T(On("check"), S("../check_choice")),
                ),
                hsm.Choice(
                    "check_choice",
                    T(S("checking"), G(head_cancellable), Effect(cancel_head)),
                    T(S("checking"), G(head_refused), Effect(refuse_head)),
                    T(S("waiting"), G(checked_with_pending)),
                    T(S("writing"), Effect(checked(head_cancellable, head_refused, checked_with_pending))),
                ),
                hsm.State(
                    "waiting",  # until each cancelled graft's join is queued
                    T(On("graft_queued"), S("../writing"), G(last_pending), Effect(cancel_landed)),
                    T(On("graft_queued"), G(other_pending), Effect(cancel_landed)),
                    *_join_arrivals(T=T, On=On, G=G, besides=(last_pending, other_pending)),
                ),
                hsm.State(
                    "writing", io(done="cancel_rows", work=_cancel_rows),  # each call's real result
                    T(On("cancel_rows"), S("../../../cancels_done"), G(rows_written), Effect(cancels_written)),
                    T(On("cancel_rows"), S("../row_choice"), G(rows_failed), Effect(take_row_failures)),
                    T(On("cancel_rows"), Effect(checked(rows_written, rows_failed))),
                    T(On(E), S("../../../cancels_done"), Effect(cancel_rows_crashed)),
                ),
                hsm.State(
                    "reporting",  # a row that could not be written: each failure is told, one per step
                    Entry(report_next),
                    T(On("report_next"), S("../row_choice")),
                ),
                hsm.Choice(
                    "row_choice",
                    T(S("reporting"), G(has_unreported), Effect(report_one)),
                    T(S("../../cancels_done"), G(all_reported), Effect(cancel_rows_failed)),
                    # a step that failed keeping the failures: the run's end answers the cancels
                    T(S("../../cancels_done"), Effect(checked(has_unreported, all_reported))),
                ),
            ),
        ),
        hsm.Choice(
            "clone_choice",  # a hand-over leaves the run: its open calls and kept rows are written first
            T(S("steps/cloning/flushing"), G(hands_over), Effect(take_hand_over)),
            T(S("resume"), G(row_lost), Effect(clone_row_lost)),
            T(S("resume"), Effect(checked(hands_over, row_lost, effects=(clone_done,)))),
        ),
        hsm.Choice(
            "resume",  # after a clone step: a run whose request was withdrawn meanwhile ends
            T(S("running"), G(run_goes_on)),
            T(S("../../closing_calls"), Effect(checked(run_goes_on))),
        ),
        hsm.Choice("handed", T(S("../../../passing"), Effect(pass_inbox))),
        hsm.Choice(
            "cancels_done",  # a withdrawal that came meanwhile ends the run: what waited is not run
            T(S("clone_gate"), G(clone_waits), Effect(judge_call)),
            T(S("running"), G(clone_dropped), Effect(drop_clone)),  # the run then ends with the error
            T(S("running"), G(cancels_only)),
            T(S("../../closing_calls"), Effect(checked(clone_waits, clone_dropped, cancels_only))),
        ),
    )


def _timeline(name: str) -> hsm.Model:
    T, S, On, G, E = hsm.Transition, hsm.Target, hsm.On, Guard, hsm.ErrorEvent
    return hsm.Define(
        name,
        hsm.Initial(S("start")),
        hsm.Choice(
            "start",  # a focus graft starts on its request, unless a caller drives the turn that made it
            T(S("live/engaged/busy/run"), G(starts_busy), Effect(engage)),
            # its request was withdrawn before it could act: it never acts on it, and the request's
            # written row is marked (in its view and its source's) before it is at rest
            T(S("live/engaged/withdrawing"), G(starts_withdrawn), Effect(engage)),
            T(S("live/idle"), Effect(checked(starts_busy, starts_withdrawn))),
        ),
        T(On(*REQUESTS, RETURNED, "when_queued", "may_continue"), Call(refuse_request)),
        # in every state: recorded, never ignored (one of a request it answered: the cancel wins)
        T(On("withdraw"), G(withdraws_late), Call(note_late_withdrawal)),
        T(On("withdraw"), Call(checked(withdraws_late, effects=(note_withdrawn,)))),
        T(On("call_done"), Call(call_finished)),
        # a background graft of its own asks before each step
        T(On("graft_step"), G(graft_wanted), Call(let_graft_go)),
        T(On("graft_step"), Call(checked(graft_wanted, effects=(stop_graft,)))),
        T(On("result_unstored"), Call(keep_unstored)),
        T(On("inbox_handed"), Call(take_handed)),
        T(On("rest_request"), G(has_joined), Call(answer_joined)),
        T(On("rest_request"), Call(checked(has_joined, effects=(refuse_request,)))),  # closed, or abandoned
        *_join_arrivals(T=T, On=On, G=G),
        T(On("clone_request"), Call(refuse_clone_call)),
        T(On("cancel_request"), Call(refuse_cancel_call)),
        T(On("poke", "shutdown", "check"), Call(ignore)),
        # a run that raised during a step that then ended it (a hand-over, its request withdrawn)
        T(On("run_failed"), Call(run_failed_after_it_ended)),
        T(On("closing"), Call(set_closing)),
        T(On(FAULT), Effect(fault_at_end)),  # joined, abandoned, closed: the session gets its record
        # in any state: nothing but that call was touched (a request for the record: answered again)
        T(On(CALL_FAULT), G(restate_failed), Effect(restating_again(engaged=False))),
        T(On(CALL_FAULT), Effect(checked(restate_failed, effects=(recover_call,), first=True))),
        T(On("restate"), Call(restating(engaged=False))),
        hsm.State(
            "live",
            T(On("join"), S("../queued"), G(is_focus), Effect(focus_join)),
            T(On("join"), Call(checked(is_focus, effects=(refuse_request,)))),
            T(On("shutdown"), S("../closed"), Effect(close_at_rest)),
            # a step failed: what it left open is closed out and the timeline rests (`recovering`)
            T(On(FAULT), S("engaged/recovering"), G(may_recover), Effect(begin_recovery)),
            T(On(FAULT), S("engaged/rest"), Effect(checked(may_recover, effects=(give_up_fault,), first=True))),
            T(On("restate"), Call(restating(engaged=False))),  # at rest, or suspended
            T(On(CALL_FAULT), G(restate_failed), Effect(restating_again(engaged=False))),
            T(On(CALL_FAULT), Effect(checked(restate_failed, effects=(recover_call,), first=True))),
            hsm.State(
                "idle",
                mirrored("active"),
                T(On(*MESSAGES), G(arrives_withdrawn), Call(drop_message)),
                T(On(*MESSAGES), S("../engaged/joining"), G(arrives_behind_joins), Effect(stash_and_engage)),
                T(On(*MESSAGES), S("../engaged/requesting"), G(arrives_to_answer), Effect(take_message)),
                T(On(*MESSAGES), Call(checked(arrives_withdrawn, arrives_behind_joins, arrives_to_answer,
                                                   effects=(refuse_request,)))),
                T(On("turn_request"), S("../engaged/busy/driven"), G(is_external), Effect(begin_driven)),
                T(On("turn_request"), S("../engaged/busy/run"), G(not_external), Effect(begin_run)),
                T(On("turn_request"), Call(checked(is_external, not_external, effects=(refuse_request,)))),
                T(On("graft_queued"), S("../receiving"), G(own_graft), Effect(engage)),
                T(On("graft_queued"), G(foreign_graft), Call(ignore)),
                T(On("graft_queued"), Effect(checked(own_graft, foreign_graft, effects=(keep_stray,), first=True))),
                T(On("drain_request"), S("../engaged/joining"), G(has_queue), Effect(engage_with_waiter)),
                T(On("drain_request"), G(no_queue), Call(answer_ended)),
                T(On("drain_request"), Call(checked(has_queue, no_queue, effects=(refuse_request,)))),
                T(On("rest_request"), Call(answer_ended)),
                T(On("poke"), S("../engaged/joining"), G(has_queue), Effect(engage)),
                T(On("poke"), Call(checked(has_queue, effects=(ignore,)))),
                T(On("shutdown"), S("../engaged/joining"), G(has_queue), Effect(close_and_engage)),
                T(On("shutdown"), S("../engaged/closing_out"), G(closes_out_at_rest), Effect(close_and_engage)),
                T(On("shutdown"), S("../../closed"), Effect(checked(has_queue, closes_out_at_rest, effects=(close_at_rest,)))),
                # messages handed over with the conversation (or back, past a focus graft that had
                # already joined) reach it at rest
                T(On("inbox_handed"), S("../engaged/joining"), G(handed_live_with_queue),
                  Effect(take_handed_and_engage)),
                T(On("inbox_handed"), S("../engaged/requesting"), G(handed_live_to_answer),
                  Effect(take_handed_and_begin)),
                T(On("inbox_handed"), Call(checked(handed_live_with_queue, handed_live_to_answer,
                                                   effects=(take_handed,)))),
            ),
            hsm.State(
                "engaged",
                hsm.Defer("shutdown"),  # taken by `busy` and `joining`; every other step finishes first
                T(On(*MESSAGES), Call(stash_message)),
                T(On("restate"), Call(restating(engaged=True))),
                T(On(CALL_FAULT), G(restate_failed), Effect(restating_again(engaged=True))),
                T(On(CALL_FAULT), Effect(checked(restate_failed, effects=(recover_call,), first=True))),
                T(On("rest_request", "drain_request"), Call(add_run_waiter)),
                hsm.State(
                    "requesting", mirrored("active"), io(done="request_written", work=_write_request),
                    T(On("request_written"), S("../busy/run"), G(request_live), Effect(request_written)),
                    T(On("request_written"), S("../dropping"), G(request_withdrawn_meanwhile), Effect(request_dropped)),
                    T(On("request_written"), S("../dropping"),  # neither: its step failed
                      Effect(checked(request_live, request_withdrawn_meanwhile, effects=(request_held,)))),
                    T(On(E), S("../rest"), Effect(request_failed)),
                ),
                hsm.State(
                    "dropping", io(done="dropped", work=_mark_rows),  # its row gets the withdrawal mark
                    T(On("dropped"), S("../rest"), Effect(rows_marked)),
                    T(On(E), S("../rest"), Effect(mark_failed)),
                ),
                hsm.State(
                    "busy",
                    mirrored("active"),
                    T(On("run_failed", E), S("../failing"), G(is_focus)),
                    T(On("run_failed", E), S("../closing_calls"), G(error_to_user), Effect(error_for_user)),
                    T(On("run_failed", E), S("../closing_calls"), G(error_already_shown), Effect(error_shown)),
                    T(On("run_failed", E), S("../closing_calls"),
                      Effect(checked(is_focus, error_to_user, error_already_shown, effects=(error_reported,)))),
                    T(On("withdraw"), S("../closing_calls"), G(ends_run), Effect(note_withdrawn)),
                    T(On("withdraw"), S("../closing_calls"), G(ends_wake), Effect(note_late_withdrawal)),
                    T(On("withdraw"), G(withdraws_late_only), Call(note_late_withdrawal)),
                    T(On("withdraw"), Call(checked(ends_run, ends_wake, withdraws_late_only, effects=(note_withdrawn,)))),
                    # the run asks before each step that could act: it goes on while the request it
                    # answers, if any, is live; once it is withdrawn the run is ended the way a
                    # withdrawal ends it (leaving `busy` ends the run that asked: nothing is left to answer)
                    T(On("may_continue"), G(run_goes_on), Effect(answer_true)),
                    T(On("may_continue"), S("../closing_calls"), Effect(checked(run_goes_on))),
                    T(On("inbox_handed"), S("../closing_calls"), G(handed_withdraws_run), Effect(take_handed)),
                    T(On("inbox_handed"), Call(checked(handed_withdraws_run, effects=(take_handed,)))),
                    T(On("shutdown"), S("../closing_calls"), Effect(interrupted_closing)),
                    _work("run", Activity(work=busy_activity, done="run"), T(On("turn_done"), S("../../ran"), Effect(end_run))),
                    _work("driven", T(On("driven_done"), S("../../rest"), Effect(end_driven))),
                ),
                hsm.State(
                    "unpreparing", io(done="unprepared", work=_unprepare),  # a clone call cut short is rolled back
                    T(On("unprepared"), S("../unwound")),
                    T(On(E), S("../unwound"), Effect(unprepare_failed)),
                ),
                hsm.Choice(
                    "unwound",
                    T(S("rest"), G(is_driven)),  # a driven turn: its caller closes its own calls
                    T(S("closing_calls"), Effect(checked(is_driven))),
                ),
                hsm.Choice(
                    "ran",  # how the run ended: a focus graft that joins hands back; anything else rests
                    T(S("handing_back"), G(joins_back)),
                    T(S("rest"), Effect(checked(joins_back))),
                ),
                hsm.State(
                    "handing_back", io(done="join_computed", work=_join_back),  # a focus graft's join event
                    T(On("join_computed"), S("../../../queued"), Effect(hand_back_join)),
                    T(On(E), S("../unrecorded"), Effect(focus_failure_not_recorded)),
                ),
                hsm.State(
                    "closing_calls", io(done="calls_closed", work=_close_calls),
                    T(On("calls_closed"), S("../withdrawn_choice"), Effect(calls_closed)),
                    T(On(E), S("../withdrawn_choice"), Effect(calls_not_closed)),
                ),
                hsm.Choice(
                    "withdrawn_choice",  # the run it cut short answered a request that was withdrawn
                    T(S("rest"), G(nothing_to_mark)),
                    T(S("withdrawing"), Effect(checked(nothing_to_mark))),
                ),
                hsm.State(
                    # the withdrawn request it answered gets the withdrawal mark on its row
                    "withdrawing", Entry(to_mark), io(done="rows_marked", work=_mark_rows),
                    T(On("rows_marked"), S("../rest"), Effect(rows_marked)),
                    T(On(E), S("../rest"), Effect(mark_failed)),
                ),
                hsm.State(
                    "failing", io(done="failure_recorded", work=_focus_failure),
                    T(On("failure_recorded"), S("../../../queued"), Effect(hand_back_join)),
                    T(On(E), S("../unrecorded"), Effect(focus_failure_not_recorded)),
                ),
                hsm.Choice("unrecorded", T(S("../../queued"), Effect(focus_failed_unrecorded))),
                hsm.State(
                    "joining",
                    mirrored("joining"),
                    Entry(start_joining),
                    T(On("shutdown"), Call(set_final)),
                    hsm.Initial(S("next")),
                    hsm.Choice(
                        "next",  # one join per step, while one can be applied
                        T(S("applying"), G(applies_next)),
                        T(S("../after"), Effect(checked(applies_next))),
                    ),
                    hsm.State(
                        "applying", io(done="applied", work=_apply),
                        T(On("applied"), S("../next"), Effect(applied)),
                        T(On(E), S("../next"), G(retry_left), Effect(apply_failed)),
                        T(On(E), S("../giving_up"), G(last_attempt), Effect(last_commit_failed)),
                        T(On(E), S("../next"), Effect(checked(retry_left, last_attempt))),
                    ),
                    hsm.Choice("giving_up", T(S("abandoning"), Effect(report_abandoned))),
                    hsm.Choice("abandoning", T(S("next"), Effect(abandon_head))),
                ),
                hsm.State(
                    "noting", mirrored("active"), io(done="noted", work=_wake_note),
                    T(On("noted"), S("../busy/run"), G(answering_live), Effect(note_written)),
                    T(On("noted"), S("../withdrawing"), G(noted_withdrawn), Effect(note_written)),  # its caller withdrew meanwhile
                    T(On("noted"), S("../rest"), Effect(checked(answering_live, noted_withdrawn, effects=(note_written,)))),
                    T(On(E), S("../unnoted"), Effect(note_failed)),
                ),
                hsm.Choice(
                    "unnoted",
                    T(S("../idle"), G(step_intact), Effect(come_to_rest)),
                    T(S("../idle"), Effect(checked(step_intact))),  # a step that failed: its callers wait for its recovery
                ),
                hsm.Choice(
                    "rest",
                    T(S("joining"), G(drains_now)),
                    T(S("../shutting"), G(closes_clean), into("active")),
                    T(S("closing_out"), G(closes_out_first)),
                    T(S("requesting"), G(answers_inbox), Effect(take_inbox)),
                    T(S("../idle"), G(comes_to_rest), Effect(come_to_rest)),
                    # a step that failed: at rest, its callers wait for its recovery
                    T(S("../idle"), Effect(checked(drains_now, closes_clean, closes_out_first, answers_inbox,
                                                   comes_to_rest))),
                ),
                hsm.Choice(
                    "after",
                    T(S("../shutting"), G(final_clean), into("active")),
                    T(S("closing_out"), G(final_out_first)),
                    T(S("../idle"), G(final_waits), Effect(come_to_rest)),
                    T(S("withdrawing"), G(withdrawn_after)),  # closed out before anything else is taken
                    T(S("requesting"), G(inbox_after), Effect(take_inbox)),
                    T(S("../idle"), G(quiet_after), Effect(come_to_rest)),
                    T(S("busy/run"), G(answers_handback), Effect(answer_handback)),
                    T(S("noting"), G(wakes_to_route), Effect(wake_route)),
                    T(S("noting"), G(wakes_to_report), Effect(wake_report)),
                    # the guards above cover every case: this is a step that failed (at rest, its
                    # callers wait for its recovery) or a guard that raised (reported)
                    T(S("../idle"), Effect(checked(final_clean, final_out_first, final_waits, withdrawn_after, inbox_after,
                                                   quiet_after, answers_handback, wakes_to_route, wakes_to_report))),
                ),
                hsm.State(
                    # a step failed: the clone call's open audit is rolled back, the run's open calls
                    # get results and rows to mark are marked (activity); the callers waiting on the
                    # failed work get the error, the session the timeline's record; then as after a run
                    "recovering", mirrored("active"), io(done="recovered", work=_recover),
                    T(On("recovered"), S("../resumed"), Recovers(recovered)),
                    T(On(E), S("../resumed"), Recovers(recovery_failed)),
                    # the step that began it failed, so its close-out held back: it begins again
                    T(On(FAULT), S("../recovering"), G(held_back_in(activity="recovered")), Effect(begin_recovery)),
                    # a step of another event failed meanwhile: told with the rest
                    T(On(FAULT), Effect(checked(held_back_in(activity="recovered"), effects=(fold_fault,), first=True))),
                ),
                hsm.State(
                    # closing with a close-out a recovery could not write: it is written first
                    "closing_out", io(done="closed_out", work=_closing_out),
                    T(On("closed_out"), S("../../shutting"), Effect(note_written), into("active")),
                    T(On(E), S("../../shutting"), Effect(close_out_failed), into("active")),
                ),
                hsm.Choice(
                    "resumed",  # a focus graft it made holds the conversation: the hand-over finishes
                    T(S("../passing"), G(hands_over), Effect(pass_inbox)),
                    T(S("rest"), Effect(checked(hands_over))),
                ),
            ),
            hsm.Choice("shutting", T(S("../closed"), Effect(close_root))),
            hsm.Choice("passing", T(S("cloned"), Effect(hand_over))),
            hsm.Choice("receiving", T(S("engaged/joining"), Effect(enqueue))),
            hsm.State(
                "cloned",
                mirrored("cloned"),
                T(On(FAULT), Effect(fault_suspended)),
                T(On("user_message"), Call(forward_to_focus)),
                # the session asks for records: the focus graft holding the conversation re-states its own too
                T(On("restate"), G(focus_restates), Call(restate_through_focus)),
                T(On("restate"), Call(checked(focus_restates, effects=(restating(engaged=False),)))),
                T(On(CALL_FAULT), G(restate_failed_through_focus), Effect(restate_through_focus_again)),
                T(On(CALL_FAULT), G(restate_failed_alone), Effect(restating_again(engaged=False))),
                T(On(CALL_FAULT), Effect(checked(restate_failed_through_focus, restate_failed_alone,
                                                 effects=(recover_call,), first=True))),
                # a message its focus graft passed back (it had joined): kept here, never sent on again
                T(On(RETURNED), Call(stash_message)),
                T(On("withdraw"), G(withdraws_late), Call(note_late_withdrawal)),
                T(On("withdraw"), Call(checked(withdraws_late, effects=(withdraw_and_forward,)))),
                T(On("call_done"), Call(call_finished_and_forward)),
                T(On("focus_joined"), S("../engaged/joining"), Effect(focus_back)),
                T(On("rest_request"), Call(answer_handed_off)),
                T(On("shutdown"), S("../releasing"), Effect(close_and_release)),
            ),
            hsm.State(
                # closed while suspended: a close-out a recovery could not write is written first
                "releasing", io(done="released", work=_closing_out),
                T(On("released"), S("../../closed"), Effect(close_at_rest)),
                T(On(E), S("../../closed"), Effect(close_out_failed, close_at_rest)),
                T(On(FAULT), S("../../closed"), Effect(fault_releasing)),
            ),
        ),
        hsm.State(
            "queued",
            mirrored("queued"),
            T(On(FAULT), S("../resending"), Effect(take_end_fault)),
            T(On("join_commit", "join_rollback"), S("../joined"), into("joined")),
            T(On("join_abandon"), S("../abandoned"), into("abandoned")),
            # a withdrawal reaches a focus graft only from its source: here it holds no request,
            # so the withdrawal changes nothing (top level) and is not sent back
            # a message that reached it after its join: passed back once, as one its source keeps
            T(On("user_message"), Call(return_to_source)),
            T(On("inbox_handed"), Call(forward_to_source)),
            T(On("rest_request"), Call(answer_joined)),  # its run has ended by joining
        ),
        hsm.Choice(
            "resending",  # a step after its join failed: its join event goes if it did not, then the fault
            T(S("telling"), G(hand_back_unsent), Effect(resend_hand_back)),
            T(S("telling"), Effect(checked(hand_back_unsent))),
        ),
        hsm.Choice("telling", T(S("queued"), Effect(tell_end_fault))),
        hsm.Final("joined"),
        hsm.Final("abandoned"),
        hsm.Final("closed"),
    )


TIMELINE = _timeline("Timeline")

_T, _S, _On, _G, _E = hsm.Transition, hsm.Target, hsm.On, Guard, hsm.ErrorEvent
BACKGROUND = hsm.Define(
    "BackgroundTimeline",
    hsm.Initial(_S("start")),
    hsm.Choice(
        "start",  # made in a turn its caller drives: it does not run; the caller joins it
        _T(_S("live/background/held"), _G(is_held)),
        _T(_S("live/background/running"), _G(starts_running)),
        # made for a request withdrawn before it was made: it never runs, and joins cancelled
        _T(_S("live/recording"), _G(starts_cancelled), Effect(ended_as(reason=CANCELLED, close_first=False))),
        _T(_S("live/recording"), Effect(checked(is_held, starts_running, starts_cancelled,
                                                effects=(ended_as(reason="failed", close_first=False),)))),
    ),
    _T(_On(*REQUESTS, "rest_request", "may_continue"), Effect(refuse_request)),  # not a timeline that rests
    _T(_On("when_queued"), Effect(answer_true)),
    _T(_On("result_unstored"), Effect(keep_unstored)),
    _T(_On("cancel_request"), Effect(refuse_cancel_call)),
    _T(_On("clone_request"), Effect(refuse_clone_call)),
    _T(_On("cancel", "shutdown", "finish", "closing"), Effect(ignore)),
    _T(_On(FAULT), Effect(bg_fault_at_end)),  # joined, abandoned: the fault is told
    hsm.State(
        "live",  # until its join event is raised: callers waiting for it are kept
        _T(_On("when_queued"), Effect(add_watcher)),
        _T(_On(FAULT), Effect(bg_fault_noted)),
        hsm.State(
            "background",
            mirrored("background"),
            _T(_On(FAULT), _S("../recording"), Effect(bg_fault_stops)),  # a step failed: it stops, joins failed
            _T(_On("finish"), _S("../ending")),
            _T(_On("cancel"), _S("../recording"), Effect(ended_as(reason=CANCELLED, close_first=True))),
            _T(_On("shutdown"), _S("../recording"), Effect(ended_as(reason="failed", close_first=True))),
            _T(_On("join"), _S("../recording"), Effect(forced)),
            _T(_On(_E), _S("../recording"), Effect(run_raised)),
            hsm.State("held"),
            # its run goes on until the graft leaves `background` (finish, cancel, shutdown, join);
            # it asks before each step: it stops (joins cancelled) once the request it was made for
            # is withdrawn by its record, else its source answers, or stops it (`cancel`)
            hsm.State(
                "running", Activity(work=background_activity, done="run"),
                _T(_On("may_continue"), _G(made_for_live), Effect(ask_source)),
                _T(_On("may_continue"), _S("../../recording"), _G(made_for_withdrawn),
                   Effect(ended_as(reason=CANCELLED, close_first=True))),
                _T(_On("may_continue"), _S("../../recording"), Effect(checked(
                    made_for_live, made_for_withdrawn, effects=(ended_as(reason="failed", close_first=True),)))),
            ),
        ),
        hsm.Choice(
            "ending",  # how its run ended: its join reason
            _T(_S("recording"), _G(run_joined), Effect(ended_by_join)),
            _T(_S("recording"), _G(run_replied), Effect(ended_as(reason="done", close_first=False))),
            _T(_S("recording"), Effect(checked(run_joined, run_replied,
                                               effects=(ended_as(reason="failed", close_first=False),)))),
        ),
        hsm.State(
            "recording",  # its join reason is written, each failure is told, then its join is raised
            hsm.Initial(_S("writing")),
            hsm.State(
                "writing", io(done="recorded", work=_record),
                _T(_On("recorded"), _S("../report_choice"), Effect(recorded)),
                _T(_On(_E), _S("../report_choice"), Effect(record_crashed)),
                # the step that entered it failed (its write held back): written now
                _T(_On(FAULT), _S("../writing"), _G(held_back_in(activity="recorded")), Effect(bg_fault_rewrites)),
                _T(_On(FAULT), Effect(checked(held_back_in(activity="recorded"), effects=(bg_fault_noted,), first=True))),
            ),
            hsm.State(
                "reporting", Entry(report_next),
                _T(_On("report_next"), _S("../report_choice")),
                _T(_On(FAULT), _S("../report_choice"), Effect(bg_fault_noted)),
            ),
            hsm.Choice(
                "report_choice",
                _T(_S("reporting"), _G(has_unreported), Effect(report_one)),
                _T(_S("../../queued"), Effect(checked(has_unreported, effects=(join_raised,)))),
            ),
        ),
    ),
    hsm.State(
        "queued",
        mirrored("queued"),
        _T(_On("join_commit", "join_rollback"), _S("../joined"), into("joined")),
        _T(_On("join_abandon"), _S("../abandoned"), into("abandoned")),
        _T(_On(FAULT), _S("../resending"), Effect(bg_take_end_fault)),
    ),
    hsm.Choice(
        "resending",  # a step after its join was raised failed: the join goes if it did not, then the fault
        _T(_S("telling"), _G(join_unsent), Effect(resend_join)),
        _T(_S("telling"), Effect(checked(join_unsent))),
    ),
    hsm.Choice("telling", _T(_S("queued"), Effect(bg_tell_end_fault))),
    hsm.Final("joined"),
    hsm.Final("abandoned"),
)


# ------------------------------------------------------------------ the scribe
def take_note(ctx, inst, event) -> None:
    inst.note = event.data


def keeps(ctx, inst, event) -> bool:
    return inst.note[1] is not None


def kept_after_report(ctx, inst, event) -> bool:
    """A report line is out (or could not go out), and the error it told is kept for `settle`."""
    return keeps(ctx=ctx, inst=inst, event=event) and not is_write(ctx=ctx, inst=inst, event=event)


def told_after_write(ctx, inst, event) -> bool:
    """A store write is made whose note carries an error no caller took: it is told next."""
    return keeps(ctx=ctx, inst=inst, event=event) and is_write(ctx=ctx, inst=inst, event=event)


def is_write(ctx, inst, event) -> bool:
    return inst.note[0][0] != "report"


def s_keep(ctx, inst, event) -> None:
    _forward(target=inst.machines.session, name="failure", data=((inst.note[1], True),))


def s_tell_kept(ctx, inst, event) -> None:
    """The error a write's note carried is told, one line, and then kept (`kept_after_report`)."""
    error = inst.note[1]
    line = dict(kind="machine", src=None, clones=[], error=error, quiet=False, background=False)
    _self(inst=inst, name="note", data=(("report", line), error))


def s_keep_lost_line(ctx, inst, event) -> None:
    """A report line the user's callbacks could not take: that failure is kept for `settle`."""
    _forward(target=inst.machines.session, name="failure", data=((event.data, True),))


def s_report_write(ctx, inst, event) -> None:
    """A store write the scribe could not make (a state's store copy, a trace line): the user is
    told and the session keeps it for `settle`. Not retried: the next copy of a state overwrites
    it, and the machines never read the store copy."""
    line = dict(kind="mirror", src=None, clones=[], error=event.data, quiet=False, background=False)
    _self(inst=inst, name="note", data=(("report", line), event.data))


def scribe_fault(ctx, inst, event) -> None:
    """A step of the scribe failed: the note it was taking is lost, and the session keeps the
    error for `settle` (the scribe cannot write its line), with the error that note was to have
    kept, if any: the note the failed step took (nothing of it went out), or the one it was
    finishing (its line went out)."""
    fault = _take_fault(inst=inst, event=event)
    taking = fault.event.name == "note"  # it failed taking the note (nothing of it went out), or after
    note = fault.event.data if taking else inst.note
    kept = tuple((e, not taking) for e in (note[1],) if e is not None)
    _forward(target=inst.machines.session, name="failure", data=kept + tuple((e, False) for e in fault.errors))


SCRIBE = hsm.Define(
    "Scribe",
    hsm.Initial(_S("ready")),
    _T(_On(FAULT), _S("ready"), Effect(scribe_fault)),
    hsm.State("ready", _T(_On("note"), _S("../writing"), Effect(take_note))),
    hsm.State(
        "writing", io(done="written", work=_scribe_write),
        hsm.Defer("note"),
        _T(_On("written"), _S("../ready"), _G(kept_after_report), Effect(s_keep)),
        _T(_On("written"), _S("../ready"), _G(told_after_write), Effect(s_tell_kept)),
        _T(_On("written"), _S("../ready"), Effect(checked(kept_after_report, told_after_write))),
        _T(_On(_E), _S("../ready"), _G(is_write), Effect(s_report_write)),
        _T(_On(_E), _S("../lost_line"), Effect(checked(is_write, effects=(s_keep_lost_line,)))),  # callbacks failed
    ),
    hsm.Choice(
        "lost_line",  # the error the report was about is still kept
        _T(_S("ready"), _G(keeps), Effect(s_keep)),
        _T(_S("ready"), Effect(checked(keeps))),
    ),
)


# ------------------------------------------------------------------ the session
def _apply_session(inst: SessionSM, name: str, data: Any) -> None:
    """How each event changes the session's own record of what is in flight."""
    if name == "engaged":
        inst.engaged.add(data)
    elif name in ("rested", "root_closed"):
        inst.engaged.discard(data)
    elif name == "handed":  # (the focus graft that joined, the timeline it handed back to)
        inst.engaged.discard(data[0])
        inst.engaged.add(data[1])
    elif name == "suspended":  # the original handed the conversation over; its focus graft counts itself
        inst.engaged.discard(data)
    elif name == "grafts_made":
        grafts, held = data
        (inst.held if held else inst.grafts).update(grafts)
    elif name == "graft_received":
        inst.grafts.discard(data)
        inst.held.discard(data)
    elif name == "grafts_undone":  # a clone call's grafts, stopped and forgotten
        inst.engaged.difference_update(data)
        inst.grafts.difference_update(data)
        inst.held.difference_update(data)
    elif name == "resync":
        _restate(inst=inst, record=data)


def _restate(inst: SessionSM, record: Record) -> None:
    """A timeline's whole record, re-stated after a fault: it replaces what the session had for it
    (the records passed on with it first: the suspended original's, re-stated by its focus graft)."""
    for passed_on in record.also:
        _restate(inst=inst, record=passed_on)
    if record.engaged is not None:
        (inst.engaged.add if record.engaged else inst.engaged.discard)(record.tl)
    if record.tl == inst.root:  # no timeline holds the conversation but the original or its focus graft
        inst.engaged = {tl for tl in inst.engaged if tl in (record.tl, record.focus)}
    if record.grafts is not None:  # the original's background grafts still running
        running, held = set(record.grafts), set(record.held)
        inst.grafts, inst.held = (running | held, set()) if inst.closing else (running, held)


def _busy_after(inst: SessionSM, event) -> bool:
    probe = SessionSM.__new__(SessionSM)
    probe.engaged, probe.grafts, probe.held = set(inst.engaged), set(inst.grafts), set(inst.held)
    probe.closing, probe.root = inst.closing, inst.root
    _apply_session(inst=probe, name=event.name, data=event.data)
    return bool(probe.engaged or probe.grafts)


def s_busy_after(ctx, inst, event) -> bool:
    return _busy_after(inst=inst, event=event)


def s_quiet_after(ctx, inst, event) -> bool:
    return not _busy_after(inst=inst, event=event)


def _keep(inst: SessionSM, errors: list[BaseException], shown: bool = True) -> None:
    """Errors kept for `settle` (each once): every error no caller was given, once its line is out
    (`shown`), with the user calls in flight then. Once one of those calls is done (`call_done`)
    the error is that call's (its line is in that call's output) and no longer kept. An error whose
    line could not be made (the scribe failed taking it) is kept for `settle` whatever is in flight."""
    calls = frozenset(inst.calls) if shown else frozenset()
    inst.errors += [(e, calls) for e in errors if all(e is not k for k, _ in inst.errors)]


def _tell(inst: SessionSM, errors: list[BaseException]) -> None:
    """A fault's errors that no caller was given: shown as one line (the scribe's), and kept for
    `settle` once it is out (the scribe hands it back, `failure`)."""
    if errors:
        error = errors[HEAD] if len(errors) == 1 else ExceptionGroup("a step failed", errors)
        _note(inst=inst, method="report", keep=error, kind="machine", src=None, clones=[], error=error, quiet=False,
              background=False)


def s_record(ctx, inst, event) -> None:
    """A record event changes the session's record; a re-stated one (`resync`) carries the errors
    no caller was given, which are told."""
    _apply_session(inst=inst, name=event.name, data=event.data)
    if event.name == "resync":
        _tell(inst=inst, errors=list(event.data.errors))


def s_call_began(ctx, inst, event) -> None:
    """A user's call is in flight (its request id): a line out from now on is in its output."""
    inst.calls.add(event.data)


def s_call_done(ctx, inst, event) -> None:
    """A user's call is done (its request id): the errors kept while it was in flight are that
    call's (their lines are in its output, or, if it raised, in the next call's), so `settle`
    does not raise them again."""
    inst.calls.discard(event.data)
    inst.errors = [(e, calls) for e, calls in inst.errors if event.data not in calls]


def s_record_and_answer(ctx, inst, event) -> None:
    s_record(ctx=ctx, inst=inst, event=event)
    s_answer(ctx=ctx, inst=inst, event=event)


def s_answer(ctx, inst, event) -> None:
    """Nothing is in flight: every `settle` waiting is answered, with the errors kept since the
    last answer (each raised once). A settle leaves the session's data only once it is answered,
    and the errors once the first is given them: a step that fails here leaves them for the
    recovery's answer (the session's record re-stated, then checked again)."""
    waiting = [s for s in inst.settles if not s.done()]  # a caller gone is not answered
    if not waiting:
        inst.settles = []
        return
    first, *rest = waiting
    errors = [e for e, _ in inst.errors]
    if len(errors) == 1:
        _fail(reply=first, error=errors[HEAD])
    elif errors:
        _fail(reply=first, error=ExceptionGroup("background grafts failed", errors))
    else:
        _resolve(reply=first, value=None)
    inst.errors = [(e, calls) for e, calls in inst.errors if all(e is not x for x in errors)]
    for later in rest:
        _resolve(reply=later, value=None)
    inst.settles = [s for s in inst.settles if not s.done()]


def s_withdraw(ctx, inst, event) -> None:
    """A settle whose caller went away: it is no longer waiting (the errors stay kept)."""
    inst.settles = [s for s in inst.settles if s is not event.data]


def s_shut_late(ctx, inst, event) -> None:
    """Grafts made by a clone call that was in flight when closing began: they are shut down
    too (one per step, `shutting`), and their joins waited for like the others'."""
    _apply_session(inst=inst, name=event.name, data=event.data)
    inst.to_shut = sorted(inst.grafts | inst.held)
    inst.grafts |= inst.held
    inst.held = set()


def s_stash_settle(ctx, inst, event) -> None:
    inst.settles.append(event.data)


def s_check(ctx, inst, event) -> None:
    """A settle asked while nothing is in flight: the original is asked to apply the joins it
    still has queued; the answer comes when `checking` is entered and checked."""
    inst.settles.append(event.data)
    _forward(target=inst.machines.timelines.get(inst.root), name="poke")


def s_checked(ctx, inst, event) -> None:  # entry of `checking`
    _self(inst=inst, name="checked")


def s_failure(ctx, inst, event) -> None:
    """Errors no caller was given: kept for `settle`. Those whose line is out (`shown`) are not,
    once a user's call that was in flight then is done: its output showed them (see `_keep`)."""
    for error, shown in event.data:
        _keep(inst=inst, errors=[error], shown=shown)


def s_fault(ctx, inst, event) -> None:
    """A step of the session failed, so its record may be wrong, and so may any answer it was
    about to give: the caller of its event and every `settle` waiting get the error (once), what
    the event said of a user's call is kept, and the original is asked to re-state its record
    (`restate`; suspended, it passes the question on to its focus graft), which comes back with
    the errors no caller was given, to keep and show. A state that ticks on entry is entered again by its own `fault`
    transition."""
    fault = _take_fault(inst=inst, event=event)
    if fault.event.name == "call_began":  # a call in flight from now on, whatever the step did
        inst.calls.add(fault.event.data)
    elif fault.event.name == "call_done":
        s_call_done(ctx=ctx, inst=inst, event=fault.event)
    elif fault.event.name == "failure":  # the errors that step was keeping are kept, whatever the step did
        s_failure(ctx=ctx, inst=inst, event=fault.event)
    elif fault.event.name == "shutdown":  # the session closes as without the fault (its caller has the error)
        _close_session(inst=inst)
    closing = fault.event.name in CLOSES  # the step that closed the session: `close` waits on it
    waiting = list(inst.settles) + (list(inst.closers) if closing else [])
    replies = _caller_of(event=fault.event) + waiting
    told = _told(fault=fault, replies=replies)
    fault.told = _fail_all(replies=replies, error=fault.error) or fault.told
    _forward(target=inst.machines.timelines.get(inst.root), name="restate",
             data=Restate(errors=tuple(_errors_for_session(errors=fault.errors, told=told))))
    inst.settles, inst.closers = [], [] if closing else inst.closers





def s_closing(ctx, inst, event) -> None:
    """Close: the original is told first (nothing is woken from now on); the grafts are shut
    down one per step (`shutting`), then the original once their joins have reached it."""
    inst.closers += [event.data] if event.data is not None else []
    _close_session(inst=inst)
    _forward(target=inst.machines.timelines.get(inst.root), name="closing")


def _close_session(inst: SessionSM) -> None:
    """Closing: every graft still running is to be shut down (one per step, `shutting`), and its
    join waited for like the others'."""
    inst.closing = True
    inst.to_shut = sorted(inst.grafts | inst.held)
    inst.grafts |= inst.held
    inst.held = set()


def s_shut_next(ctx, inst, event) -> None:  # entry of `shutting`
    _self(inst=inst, name="shut_next")


def s_shut_one(ctx, inst, event) -> None:
    _forward(target=inst.machines.timelines.get(inst.to_shut.pop(HEAD)), name="shutdown")


def s_wait_joins(ctx, inst, event) -> None:  # entry of `draining`
    _self(inst=inst, name="joins_check")


def s_shut_root(ctx, inst, event) -> None:  # entry of `finishing`
    """Every graft's join has reached the original: it is shut down (it applies what it can and
    closes). Entered again if the step fails, so the original is always told."""
    _forward(target=inst.machines.timelines.get(inst.root), name="shutdown")


def s_stash_closer(ctx, inst, event) -> None:
    inst.closers += [event.data] if event.data is not None else []


def s_closed(ctx, inst, event) -> None:
    """Closed: every close and settle waiting is answered; a machine that could not be stopped
    is the first close's error (the one error, or an ExceptionGroup of all of them)."""
    closers, settles, errors = list(inst.closers), list(inst.settles), list(inst.stop_errors)
    if closers and errors:
        first, *closers = closers
        _fail(reply=first, error=errors[0] if len(errors) == 1 else ExceptionGroup("stopping machines failed", errors))
    for reply in closers + settles:
        _resolve(reply=reply, value=None)
    # left only once answered: a step that fails here leaves them for its recovery to answer
    inst.closers, inst.settles, inst.stop_errors = [], [], []


def s_take_stop_errors(ctx, inst, event) -> None:
    inst.stop_errors = list(event.data)


def s_stop_crashed(ctx, inst, event) -> None:
    inst.stop_errors = [event.data]


def s_stops_clean(ctx, inst, event) -> bool:
    return not event.data


async def s_stop_machines(ctx, inst: SessionSM, event) -> None:
    """The original has closed: every timeline's machine and the scribe are stopped (each stop
    waits for the step it is in), so nothing stays live after `close`. A machine that cannot be
    stopped does not keep the others live: each failure is collected, the rest are still
    stopped, and the outcome (`stopped`) carries the failures. Later requests to the stopped
    machines are refused by the registry."""
    errors: list[BaseException] = []
    for sm in [*inst.machines.timelines.values(), inst.machines.scribe]:
        if sm is not None and sm.state():
            try:
                await hsm.Stop(sm)
            except Exception as e:  # noqa: BLE001 - the outcome event carries it
                errors.append(e)
    inst.dispatch(inst.context(), hsm.Event(name="stopped", data=errors))


def _s_tell(inst, event):
    """The machines that could not be stopped are told to the user (the scribe is stopped by now,
    so the session's own activity calls the host), each whatever telling another one raised. The
    outcome is what the user's callbacks raised meanwhile, which close raises with the rest."""
    failed: list[BaseException] = []
    for error in inst.stop_errors:
        try:
            inst.machines.host.report(kind="machine", src=None, clones=[], error=error, quiet=False,
                                      background=False)
        except Exception as e:  # noqa: BLE001 - the outcome event carries it
            failed.append(e)
    return failed


def s_told(ctx, inst, event) -> None:
    """Every stop failure was told: what telling them raised is added to what close raises."""
    inst.stop_errors += list(event.data)
    s_closed(ctx=ctx, inst=inst, event=event)


def s_tell_crashed(ctx, inst, event) -> None:
    inst.stop_errors.append(event.data)
    s_closed(ctx=ctx, inst=inst, event=event)


def s_answer_now(ctx, inst, event) -> None:
    _resolve(reply=event.data, value=None)


def left_to_shut(ctx, inst, event) -> bool:
    return bool(inst.to_shut)


def all_joins_in(ctx, inst, event) -> bool:
    return not inst.grafts


def last_join_in(ctx, inst, event) -> bool:
    return inst.grafts == {event.data}


def is_root_event(ctx, inst, event) -> bool:
    return event.data == inst.root


def root_record_closed(ctx, inst, event) -> bool:
    """A record re-stated by the original after a fault says it has closed."""
    return event.data.tl == inst.root and event.data.closed


# The session's steps that close it (the machines stopped, or told): a `close` waits on them.
CLOSES = ("stopped", "told", hsm.ErrorEvent.name)
RECORDS = ("engaged", "rested", "handed", "suspended", "grafts_made", "graft_received", "grafts_undone", "resync")

SESSION = hsm.Define(
    "Session",
    hsm.Initial(_S("open/quiet")),
    _T(_On(*REQUESTS[:-2], "when_queued", "rest_request"), Effect(refuse_request)),
    _T(_On("failure"), Effect(s_failure)),
    _T(_On(*RECORDS), Effect(s_record)),
    _T(_On("settle_request", "shutdown"), Effect(s_answer_now)),
    _T(_On("settle_withdrawn"), Effect(s_withdraw)),
    _T(_On("call_began"), Effect(s_call_began)),
    _T(_On("call_done"), Effect(s_call_done)),
    _T(_On("checked", "root_closed", "shut_next", "joins_check"), Effect(ignore)),
    _T(_On(FAULT), Effect(s_fault)),
    hsm.State(
        "open",
        _T(_On(*RECORDS), _S("working"), _G(s_busy_after), Effect(s_record)),
        _T(_On(*RECORDS), Effect(checked(s_busy_after, effects=(s_record,)))),
        _T(_On("settle_request"), Effect(s_stash_settle)),
        _T(_On("shutdown"), _S("../closing"), Effect(s_closing)),
        hsm.State(
            "quiet",
            # a settle left waiting by a step that failed is answered by the next record that leaves
            # the session quiet (one waiting otherwise goes to `checking`)
            _T(_On(*RECORDS), _G(s_quiet_after), Effect(s_record_and_answer)),
            _T(_On(*RECORDS), _S("../working"), _G(s_busy_after), Effect(s_record)),
            _T(_On(*RECORDS), Effect(checked(s_quiet_after, s_busy_after, effects=(s_record,)))),
            _T(_On("settle_request"), _S("../checking"), Effect(s_check)),
        ),
        hsm.State("checking", Entry(s_checked), _T(_On("checked"), _S("../quiet"), Effect(s_answer)),
                  _T(_On(FAULT), _S("../checking"), Effect(s_fault))),
        hsm.State(
            "working",
            _T(_On(*RECORDS), _S("../quiet"), _G(s_quiet_after), Effect(s_record_and_answer)),
            _T(_On(*RECORDS), _G(s_busy_after), Effect(s_record)),
            _T(_On(*RECORDS), Effect(checked(s_quiet_after, s_busy_after, effects=(s_record,)))),
        ),
    ),
    hsm.State(
        "closing",
        _T(_On("settle_request"), Effect(s_stash_settle)),
        _T(_On("shutdown"), Effect(s_stash_closer)),
        _T(_On("root_closed"), _S("stopping"), _G(is_root_event)),
        _T(_On("root_closed"), Effect(checked(is_root_event, effects=(ignore,)))),  # a focus graft closed
        # the original closed, but its step failed before it told the session: its record says so
        _T(_On("resync"), _S("stopping"), _G(root_record_closed), Effect(s_record)),
        _T(_On("resync"), Effect(checked(root_record_closed, effects=(s_record,)))),
        _T(_On("grafts_made"), _S("shutting"), Effect(s_shut_late)),
        hsm.Initial(_S("shutting")),
        hsm.State(
            "shutting", Entry(s_shut_next),
            _T(_On("shut_next"), _S("../shutting"), _G(left_to_shut), Effect(s_shut_one)),
            _T(_On("shut_next"), _S("../draining"), Effect(checked(left_to_shut))),
            _T(_On(FAULT), _S("../shutting"), Effect(s_fault)),
        ),
        hsm.State(
            "draining", Entry(s_wait_joins),
            _T(_On("joins_check"), _S("../finishing"), _G(all_joins_in)),
            _T(_On("joins_check"), Effect(checked(all_joins_in, effects=(ignore,)))),
            _T(_On("graft_received"), _S("../finishing"), _G(last_join_in), Effect(s_record)),
            _T(_On("graft_received"), Effect(checked(last_join_in, effects=(s_record,)))),
            # a record re-stated after a fault may leave no join to wait for: checked again
            _T(_On("resync"), _S("../draining"), Effect(s_record)),
            _T(_On(FAULT), _S("../draining"), Effect(s_fault)),
        ),
        hsm.State("finishing", Entry(s_shut_root), _T(_On(FAULT), _S("../finishing"), Effect(s_fault))),
        hsm.State(
            "stopping", Activity(work=s_stop_machines, done="stopped"),  # every machine but this one, then closed
            _T(_On("stopped"), _S("../../closed"), _G(s_stops_clean), Effect(s_closed)),
            _T(_On("stopped"), _S("../telling"), Effect(checked(s_stops_clean, effects=(s_take_stop_errors,)))),
            _T(_On(_E), _S("../telling"), Effect(s_stop_crashed)),
            _T(_On(FAULT), _S("../stopping"), _G(held_back_in(activity="stopped")), Effect(s_fault)),
            _T(_On(FAULT), Effect(checked(held_back_in(activity="stopped"), effects=(s_fault,), first=True))),
        ),
        hsm.State(
            "telling", io(done="told", work=_s_tell),  # a machine that could not be stopped: reported
            _T(_On("told"), _S("../../closed"), Effect(s_told)),
            _T(_On(_E), _S("../../closed"), Effect(s_tell_crashed)),
            _T(_On(FAULT), _S("../telling"), _G(held_back_in(activity="told")), Effect(s_fault)),
            _T(_On(FAULT), Effect(checked(held_back_in(activity="told"), effects=(s_fault,), first=True))),
        ),
    ),
    hsm.Final("closed"),
)


class Machines:
    """The registry of machines and the requests the harness puts to them. It decides nothing and
    exposes nothing of a machine's data: every request is an event, and the graph answers it."""

    def __init__(self, host: Host):
        self.host = host
        self.timelines: dict[str, TimelineSM] = {}
        self.session: SessionSM | None = None
        self.scribe: ScribeSM | None = None
        self.root: str | None = None
        self.requests = 0  # user requests made so far: the next one's id

    # ------------------------------------------------------------------ making machines
    async def start_root(self, tl: str) -> None:
        if self.root is not None:
            raise TransitionRefused("root already started")
        self.scribe = ScribeSM(self)
        await hsm.Started(None, self.scribe, SCRIBE, hsm.Config(ID="scribe"))
        self.session = SessionSM(self)
        self.session.root = tl
        await hsm.Started(None, self.session, SESSION, hsm.Config(ID="session"))
        sm = TimelineSM(machines=self, timeline_id=tl, source_id=None, kind="original")
        self.timelines[tl] = sm
        self.root = tl
        await hsm.Started(None, sm, TIMELINE, hsm.Config(ID=tl))

    # ------------------------------------------------------------------ requests
    def future(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    def request(self, text: str) -> Request:
        """A new user request: a fresh id and the future that carries its answer."""
        self.requests += 1
        return Request(id=self.requests, text=text, reply=self.future())

    def _target(self, tl: str, name: str) -> hsm.Instance:
        sm = self.session if tl == "session" else self.timelines.get(tl)
        if sm is None:
            raise TransitionRefused(f"no machine for {tl}")
        if not sm.state():  # stopped by `close`: readiness only, nothing about the graph is read
            raise TransitionRefused(f"{tl}: '{name}' refused in state closed (the harness is closed)")
        return sm

    async def send(self, tl: str, name: str, data: Any = None) -> None:
        await hsm.Dispatch(None, self._target(tl=tl, name=name), hsm.Event(name=name, data=data))

    def post(self, tl: str, name: str, data: Any = None) -> None:
        """Put an event on tl's machine now, without waiting for it to be processed (the machine
        takes its events in order). For a caller that waits on a reply carried in the event: its
        only await is that reply, so a cancel of the caller always lands on it.

        Nothing a post sets off is lost. A stopped machine is refused here (TransitionRefused,
        raised to the caller) before hsm is asked, so hsm's "not started" result never arises;
        hsm's event queue is unbounded, so the push cannot fail; and posts go to the original,
        which is stopped only once it is `closed` (a final state with no activity), so an event
        cannot land in its queue while it is being stopped and be cleared with it. What the event
        then sets off fails only as an activity (handled by the graph's `hsm/error` transitions)
        or as a guard, effect, entry or exit, which is a fault the machine recovers from
        (`_step`): the caller waiting on the event's reply gets the error, whichever task
        processes the event; a fault no caller waits on (a withdrawal's) is told and kept for
        `settle`."""
        hsm.Dispatch(None, self._target(tl=tl, name=name), hsm.Event(name=name, data=data))

    async def ask(self, tl: str, name: str, build: Callable[[asyncio.Future], Any]) -> Any:
        """A request to tl's machine, answered on the reply carried in its event. The request is put
        on the machine without waiting (`post`), so the caller's only await is its reply: a caller
        that goes away (a run its machine ended) cancels that reply at once, and an answer that comes
        later is not left to no one."""
        reply = self.future()
        self.post(tl=tl, name=name, data=build(reply))
        return await reply

    async def stop(self) -> None:
        for sm in list(self.timelines.values()):
            if sm.state():
                await hsm.Stop(sm)
        for sm in (self.session, self.scribe):
            if sm is not None and sm.state():
                await hsm.Stop(sm)
