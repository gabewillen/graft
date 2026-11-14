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
  transitions. Fire-and-forget writes (the store copy of each state, report
  lines, trace) go to the scribe machine, which does them one at a time in its `writing` state.
* Machines talk only by events. A caller outside the machines (`user_says`, `settle`, `close`, a
  clone or cancel call made by a model loop) gets its answer from a future carried in the event's
  data, which an effect resolves.
* A user's call is a Request (id, text, reply) the original's machine owns from arrival until it
  is answered or withdrawn, wherever it is (inbox, `requesting`, the request a run answers, the
  waiters, handed to a focus graft with the conversation). `withdraw` (by id) is handled in every
  state: recorded while the request is held (the record is pruned as requests leave), a waiting
  message dropped and answered (one shared response, which the session hears of in the event
  the step sends anyway), forwarded to a focus graft (also one made in `cloning`, once its machine is started), and a
  run whose request is withdrawn ends (`_run_withdrawn`, the one rule). Every point that would
  start a run or let a graft act is guarded on its request being live (not withdrawn, reply not
  done): `requesting` -> busy, the next inbox message, the wake / hand-back / routing runs,
  `noting` -> busy, the clone gate, a focus graft's start (one whose request was withdrawn before
  it could act marks its row in `withdrawing` and rests), and a background graft's start and
  each of its steps (a background graft made for a request withdrawn before it is answered never
  runs, or stops at its next step, and joins `cancelled`; that join wakes nothing). Whether a run
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
      cloned          suspended while its focus graft holds the conversation
      releasing       closed while suspended: its focus graft is shut down first
    queued, joined, abandoned, closed

A background graft: `background` (its run, or `held` when a caller drives the turn that made it;
never entered if the request it was made for was withdrawn before it was made)
--finish (`ending`, a choice on how the run ended) | cancel | shutdown | join (forced) | error
| `may_continue` once the request it was made for is withdrawn (reason cancelled)-->
`recording` (`writing`, an activity: its open
calls and its join reason; then each failure told, one per step, in `reporting`) --> `queued`
--join commit | rollback--> `joined`, --join abandon--> `abandoned`.

The session machine knows whether anything is in flight (the original or a running focus graft
engaged, a background graft whose join has not reached it) and answers `settle` when nothing is;
`close` is `shutdown`. It keeps the errors `settle` raises, each once, unless a user's call was
waiting when it happened. At the end of `close` it stops every machine (`stopping`); one that
cannot be stopped is reported (`telling`) and raised by `close`, and the rest are still stopped.

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
    handed to). The caller withdraws it by id (`withdraw`); a request is live while it is not
    withdrawn and its reply is not done. `row`: its user row once written (a withdrawn request
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
        # Ids of withdrawn requests it holds (and, for the original, of requests its background
        # grafts were made for, until their joins are applied): pruned as requests leave it.
        self.withdrawn: set[int] = set(withdrawn or ())
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
        self.clone_text = ""
        self.spawn_plan: list[GraftPlan] = []
        self.rejection = ""  # the result text of a clone call refused before or in its preparation
        self.handing: Clone | None = None  # a clone call that hands the conversation over, being finished
        self.ended = ""  # the final state it reached: joined | abandoned | active (closed)
        self.cancel: Cancel | None = None  # the cancel calls being carried out
        self.to_check: list[str] = []  # ids of those calls not yet decided
        self.pending: set[str] = set()  # grafts they cancelled whose joins are not queued yet
        self.outcomes: dict[str, str] = {}
        self.claims: dict[str, int] = {}  # graft -> the call of the response that names it first
        self.unstored: dict[str, tuple[str, bool]] = {}  # tool_use id -> a result not yet written
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


class SessionSM(hsm.Instance):
    def __init__(self, machines: "Machines"):
        super().__init__()
        self.machines = machines
        self.engaged: set[str] = set()
        self.grafts: set[str] = set()
        self.held: set[str] = set()
        self.settles: list[asyncio.Future] = []
        self.errors: list[BaseException] = []
        self.closers: list[asyncio.Future] = []
        self.root: str | None = None
        self.users: set[int] = set()
        self.to_shut: list[str] = []
        self.stop_errors: list[BaseException] = []  # machines that could not be stopped at close


class ScribeSM(hsm.Instance):
    def __init__(self, machines: "Machines"):
        super().__init__()
        self.host = machines.host
        self.machines = machines
        self.note: tuple = ((), None)


# ------------------------------------------------------------------ plumbing
def _resolve(reply: asyncio.Future | None, value: Any = None) -> None:
    if reply is not None and not reply.done():
        reply.set_result(value)


def _fail(reply: asyncio.Future | None, error: BaseException) -> None:
    if reply is not None and not reply.done():
        reply.set_exception(error)


def _forward(target: hsm.Instance | None, name: str, data: Any = None) -> None:
    if target is not None:
        hsm.Dispatch(None, target, hsm.Event(name=name, data=data))


def _self(inst: hsm.Instance, name: str, data: Any = None) -> None:
    hsm.Dispatch(None, inst, hsm.Event(name=name, data=data))


def _session(inst: TimelineSM, name: str, data: Any = None) -> None:
    _forward(target=inst.machines.session, name=name, data=data)


def _note(inst, method: str, keep: BaseException | None = None, **args: Any) -> None:
    """A fire-and-forget write (store copy of a state, a report line, trace): the scribe calls the
    host's `method` with `args`, by name. `keep`: an error the session keeps for `settle` once the
    line is out (or failed to go out)."""
    _forward(target=inst.machines.scribe, name="note", data=((method, args), keep))


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
    return hsm.Entry(mirror)


def into(status: str) -> hsm.RedefinableBehaviors:
    """The effect of a transition into a final state: the state reached is recorded (`ended`) and
    copied to the store."""
    def mirror(ctx, inst, event) -> None:
        inst.ended = status
        _note(inst=inst, method="mirror", tl=inst.timeline_id, status=status)
    mirror.__name__ = f"mirror_into_{status}"
    return hsm.Effect(mirror)


def io(done: str, work: Callable) -> hsm.RedefinableBehaviors:
    """A state's activity that does I/O: `work(inst, event)` runs, and its result comes back as the
    event `done`. If it raises, the library dispatches `hsm/error`, handled by the state."""
    async def activity(ctx, inst, event) -> None:
        result = work(inst=inst, event=event)
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
    withdrawn."""
    return request.id in inst.withdrawn or request.reply.cancelled()


def _run_withdrawn(inst: TimelineSM, withdrawing: int | None = None) -> bool:
    """The one rule for ending a run on a withdrawal: the run answers a user request, and that
    request is not live (or is the one being withdrawn now, `withdrawing`)."""
    request = inst.answering
    return request is not None and (request.id == withdrawing or not _live(inst=inst, request=request))


def _held(inst: TimelineSM) -> list[Request]:
    """Every request this machine holds: waiting, being written, answered by its run, waiting at rest."""
    held = list(inst.inbox) + [w.request for w in inst.waiters if w.request is not None]
    return held + [r for r in (inst.answering, inst.request) if r is not None]


def _prune(inst: TimelineSM) -> None:
    """The withdrawal record keeps only ids of requests it still holds, and of requests its
    background grafts were made for whose joins are not applied yet: it is bounded by them."""
    kept = {r.id for r in _held(inst)} | {r.id for r in inst.graft_requests.values()}
    inst.withdrawn &= kept


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
    return _run_withdrawn(inst=inst, withdrawing=event.data)


def arrives_withdrawn(ctx, inst, event) -> bool:
    return not _live(inst=inst, request=event.data)


def arrives_behind_joins(ctx, inst, event) -> bool:
    """A live message arrives while joins are queued: they are applied first."""
    return _live(inst=inst, request=event.data) and bool(inst.queue)


def arrives_to_answer(ctx, inst, event) -> bool:
    return _live(inst=inst, request=event.data) and not inst.queue


def request_live(ctx, inst, event) -> bool:
    return _live(inst=inst, request=inst.request)


def answering_live(ctx, inst, event) -> bool:
    """A run may start: it answers no user request (a wake for grafts that finished while it
    was idle), or at least one it answers is live."""
    return not _run_withdrawn(inst)


def run_withdrawn(ctx, inst, event) -> bool:
    return _run_withdrawn(inst)


def _withdrawn_rows(inst: TimelineSM) -> list[int]:
    """The row of the request its run answers, if it was withdrawn after its row was written."""
    request = inst.answering
    return [request.row] if _run_withdrawn(inst) and request.row is not None else []


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


def made_for_withdrawn(ctx, inst, event) -> bool:
    return _made_for_withdrawn(inst)


def starts_cancelled(ctx, inst, event) -> bool:
    """A background graft that runs on its own, made for a request withdrawn before it was made."""
    return not inst.held and _made_for_withdrawn(inst)


def own_graft(ctx, inst, event) -> bool:
    """A join from a graft it holds: one of a clone call undone because a machine could not start
    (its join held for the step) is never applied."""
    return event.data.child in inst.grafts


def is_focus(ctx, inst, event) -> bool:
    return inst.kind == "focus"


def carries_cancels(ctx, inst, event) -> bool:
    return event.data.cancels is not None


def no_cancels(ctx, inst, event) -> bool:
    return event.data.cancels is None


def declined(ctx, inst, event) -> bool:
    return decline_why(inst=inst, clone=inst.clone) is not None


def rejected_in_preparation(ctx, inst, event) -> bool:
    return isinstance(event.data, str)


def graph_refuses(ctx, inst, event) -> bool:
    """A prepared call (not one the harness refused in its preparation) the graph refuses."""
    prepared = not rejected_in_preparation(ctx=ctx, inst=inst, event=event)
    return prepared and clone_why(inst=inst, clone=inst.clone) is not None


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
    before it withdraws). The outcome is an event: `spawned`, or `spawn_failed` with the machines
    made so far (they are undone in `unspawning`)."""
    machines, source, plans = inst.machines, inst.timeline_id, list(inst.spawn_plan)
    held, withdrawn = inst.mode.name == "external", set(inst.withdrawn)
    made: list[str] = []
    try:
        for plan in plans:
            sm = TimelineSM(machines=machines, timeline_id=plan.child, source_id=source,
                            kind="focus" if plan.focus else "background", held=held,
                            waiters=plan.waiters, answering=plan.answering, made_for=plan.made_for,
                            withdrawn=withdrawn)
            machines.timelines[plan.child] = sm
            made.append(plan.child)
            await hsm.Started(None, sm, TIMELINE if plan.focus else BACKGROUND, hsm.Config(ID=plan.child))
    except Exception as e:  # noqa: BLE001 - the outcome event carries it
        inst.dispatch(inst.context(), hsm.Event(name="spawn_failed", data=(made, e)))
        return
    inst.dispatch(inst.context(), hsm.Event(name="spawned"))


async def unspawn_activity(ctx, inst: TimelineSM, event) -> None:
    """Undo a clone call whose machines could not all start: the machines made are stopped and
    forgotten, and the committed grafts are marked abandoned in the store. The refusal follows."""
    for child in inst.spawned_ids:
        sm = inst.machines.timelines.pop(child, None)
        if sm is not None and sm.state():
            await hsm.Stop(sm)
    inst.host.abandon_clone(src=inst.timeline_id, grafts=[g for g, _ in inst.clone.pairs])
    inst.dispatch(inst.context(), hsm.Event(name="unspawned", data=inst.spawn_error))


def _write_request(inst, event):
    return inst.host.begin_request(tl=inst.timeline_id, text=inst.request.text)


def _viewers(inst: TimelineSM) -> list[str]:
    """The timelines whose views a withdrawal mark binds: the one the request was asked of and, for
    a focus graft, its source, which holds the same conversation and gets the graft's rows back."""
    return [inst.timeline_id] + ([inst.source_id] if inst.kind == "focus" else [])


def _mark_dropped(inst, event):
    """The row just written for a request withdrawn meanwhile gets the withdrawal mark (the
    activity of `dropping`; the event that entered it carries the row)."""
    inst.host.mark_withdrawn(timelines=_viewers(inst), rows=[event.data])


def _mark_withdrawn(inst, event):
    """The row of the withdrawn request its run answered gets the withdrawal mark (the activity of
    `withdrawing`, entered only when there is one)."""
    inst.host.mark_withdrawn(timelines=_viewers(inst), rows=_withdrawn_rows(inst))


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
    text = inst.host.refuse_clone(src=inst.timeline_id, clone=inst.clone,
                                  why=clone_why(inst=inst, clone=inst.clone) or "internal error", error=error)
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
    join = inst.queue[0]
    return inst.host.apply_join(src=inst.timeline_id, join=join, cloned_for=inst.cloned_for.get(join.child),
                                running=_still_running(inst))


def _wake_note(inst, event):
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


def _scribe_write(inst, event):
    (method, args), _keep = inst.note
    getattr(inst.host, method)(**args)


# ------------------------------------------------------------------ effects: user requests
def _waiting(inst: TimelineSM, request: Request) -> list[int]:
    return [id(request.reply)]


def _answer_dead(requests: list[Request]) -> list[int]:
    """The one response to a request dropped because it is no longer live: it is answered (nothing
    is run for it); returns the ids the session counts it by, for the one event the caller sends."""
    for r in requests:
        _resolve(reply=r.reply, value="ended")
    return [id(r.reply) for r in requests]


def _drop_dead(inst: TimelineSM) -> list[int]:
    """The waiting messages whose callers have gone leave the inbox, each answered; their ids for
    the session."""
    dead = [r for r in inst.inbox if not _live(inst=inst, request=r)]
    inst.inbox = [r for r in inst.inbox if all(r is not d for d in dead)]
    return _answer_dead(dead)


def take_message(ctx, inst, event) -> None:
    inst.request = event.data
    _session(inst=inst, name="request_taken", data=(inst.timeline_id, _waiting(inst=inst, request=event.data), []))


def drop_message(ctx, inst, event) -> None:
    """A message whose caller had gone before it arrived: nothing is written or run for it."""
    _session(inst=inst, name="user_answered", data=_answer_dead([event.data]))


def take_inbox(ctx, inst, event) -> None:
    """The first live waiting message is taken (a guard saw one); the dead ones are dropped."""
    dead = _drop_dead(inst)
    inst.request = inst.inbox.pop(HEAD)
    _session(inst=inst, name="user_answered", data=dead)


def request_written(ctx, inst, event) -> None:
    """The message's row is written: it is the run's request; the bounds of the last one are
    cleared (a new request may clone again)."""
    request, inst.request = inst.request, None
    request.row = event.data
    inst.answering = request
    inst.mode, inst.clone_tries = TURN, 0
    inst.blocks, inst.clone_calls, inst.wake_clones = set(), 0, 0
    inst.waiters.append(Waiter(reply=request.reply, user=True, request=request))


def request_dropped(ctx, inst, event) -> None:
    """The message's row is written, but its caller withdrew it meanwhile: no run starts for it.
    The row stays (rows are never removed) and is marked withdrawn next (`dropping`), so no later
    turn reads it as a live request. A new message was written, so the bounds of the last one
    are cleared."""
    request, inst.request = inst.request, None
    inst.blocks, inst.clone_calls, inst.wake_clones = set(), 0, 0
    _prune(inst)
    _session(inst=inst, name="user_answered", data=_answer_dead([request]))


def request_failed(ctx, inst, event) -> None:
    """The message's row could not be written: its caller gets the error, the user is told, and no
    run starts for it."""
    request, inst.request = inst.request, None
    inst.waiters.append(Waiter(reply=request.reply, user=True, error=event.data, request=request))
    _report(inst=inst, kind="request", clones=[], error=event.data, keep=False)


def stash_message(ctx, inst, event) -> None:
    inst.inbox.append(event.data)
    _session(inst=inst, name="user_waiting", data=_waiting(inst=inst, request=event.data))


def stash_and_engage(ctx, inst, event) -> None:
    inst.inbox.append(event.data)
    _session(inst=inst, name="request_taken", data=(inst.timeline_id, _waiting(inst=inst, request=event.data), []))


def note_withdrawn(ctx, inst, event) -> None:
    """A caller withdrew its request (by id): recorded if this machine holds it, wherever it is;
    a waiting message is dropped and answered at once. An unknown or already answered id changes
    nothing."""
    gone = [r for r in _held(inst) if r.id == event.data]
    inst.withdrawn |= {event.data} if gone else set()
    dead = _drop_dead(inst)
    _prune(inst)
    _session(inst=inst, name="user_answered", data=dead + [id(r.reply) for r in gone])


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
    dead = _drop_dead(inst)
    inst.request = inst.inbox.pop(HEAD)
    _session(inst=inst, name="request_taken",
             data=(inst.timeline_id, _waiting(inst=inst, request=inst.request), dead))


def take_handed_and_engage(ctx, inst, event) -> None:
    """Handed messages reach it at rest with joins queued: the joins are applied first."""
    take_handed(ctx=ctx, inst=inst, event=event)
    _session(inst=inst, name="engaged", data=inst.timeline_id)


def begin_run(ctx, inst, event) -> None:
    mode, reply = event.data
    _session(inst=inst, name="engaged", data=inst.timeline_id)
    inst.mode, inst.answering, inst.clone_tries = mode, None, 0
    inst.waiters.append(Waiter(reply=reply, user=False))


def begin_driven(ctx, inst, event) -> None:
    mode, reply = event.data
    _session(inst=inst, name="engaged", data=inst.timeline_id)
    inst.mode, inst.answering, inst.clone_tries = mode, None, 0
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
    inst.reported = None


def error_for_user(ctx, inst, event) -> None:
    [w for w in inst.waiters if w.error is None][-1].error = event.data
    inst.reported = None


def end_driven(ctx, inst, event) -> None:
    _resolve(reply=event.data, value="ended")


def unprepare_failed(ctx, inst, event) -> None:
    """Rolling the cut-short call back failed: the user is told, and settle raises it."""
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


def mark_failed(ctx, inst, event) -> None:
    _report(inst=inst, kind="task", clones=[], error=event.data)


def calls_closed(ctx, inst, event) -> None:
    inst.unstored = {}


def keep_unstored(ctx, inst, event) -> None:
    """A tool its run executed whose result row could not be written: its real result is kept,
    and written when the run's open calls are closed (it ran: it is never closed as not run)."""
    use, content, is_error = event.data
    inst.unstored[use["id"]] = (content, is_error)


def calls_not_closed(ctx, inst, event) -> None:
    _report(inst=inst, kind="task", clones=[], error=event.data)


def _answer_all(inst: TimelineSM) -> list[int]:
    """Every caller waiting on this timeline is answered; returns the user calls answered."""
    waiters, inst.waiters = inst.waiters, []
    users = [id(w.reply) for w in waiters if w.user]
    for w in waiters:
        if w.error is not None:
            _fail(reply=w.reply, error=w.error)
        else:
            _resolve(reply=w.reply, value="ended")
    users += [id(r.reply) for r in inst.inbox]
    for r in inst.inbox:
        _fail(reply=r.reply, error=TransitionRefused("the harness was closed"))
    inst.inbox, inst.answering = [], None
    _prune(inst)
    return users


def come_to_rest(ctx, inst, event) -> None:
    """At rest: every caller waiting is answered, waiting messages whose callers have gone are
    dropped, and the session told in one event."""
    waiters, inst.waiters = inst.waiters, []
    for w in waiters:
        if w.error is not None:
            _fail(reply=w.reply, error=w.error)
        else:
            _resolve(reply=w.reply, value="ended")
    dead = _drop_dead(inst)
    inst.answering = None
    _prune(inst)
    _session(inst=inst, name="rested", data=(inst.timeline_id, [id(w.reply) for w in waiters if w.user] + dead))


def close_root(ctx, inst, event) -> None:
    _session(inst=inst, name="root_closed", data=(inst.timeline_id, _answer_all(inst)))


def close_at_rest(ctx, inst, event) -> None:
    _session(inst=inst, name="root_closed", data=(inst.timeline_id, _answer_all(inst)))


def release_next(ctx, inst, event) -> None:  # entry of `releasing`
    _self(inst=inst, name="released")


def close_and_release(ctx, inst, event) -> None:
    """Closed while suspended: the focus graft is shut down (it answers whoever waits on it and
    closes); the joins on the queue stay queued."""
    _forward(target=inst.machines.timelines.get(inst.focus), name="shutdown")


def forward_to_focus(ctx, inst, event) -> None:
    _forward(target=inst.machines.timelines.get(inst.focus), name=event.name, data=event.data)


def forward_to_source(ctx, inst, event) -> None:
    _forward(target=inst.machines.timelines.get(inst.source_id), name=event.name, data=event.data)


# ------------------------------------------------------------------ effects: clone
def take_clone(ctx, inst, event) -> None:
    inst.clone = event.data
    inst.cancel = event.data.cancels


def clone_refused_written(ctx, inst, event) -> None:
    clone, inst.clone = inst.clone, None
    _resolve(reply=clone.reply, value=(None, None, event.data))


def clone_refusal_not_written(ctx, inst, event) -> None:
    clone, inst.clone = inst.clone, None
    _report(inst=inst, kind="task", clones=[], error=event.data)
    _resolve(reply=clone.reply, value=(None, None, f"clone refused; no clone was created ({event.data})"))


def attempt_clone(ctx, inst, event) -> None:
    """The call may be tried: it uses one of the run's clone tries, and is prepared next."""
    inst.clone_tries += 1


def decline(ctx, inst, event) -> None:
    """The whole call is not tried (nothing is seeded): its result says why."""
    inst.rejection = f"clone not executed: {decline_why(inst=inst, clone=inst.clone)}"


def take_prepared(ctx, inst, event) -> None:
    """The harness prepared the call: its grafts, parts and batch, or the text of its refusal."""
    if isinstance(event.data, str):
        inst.rejection = event.data
    else:
        inst.clone.pairs, inst.clone.parts, inst.clone.data = event.data


def preparation_crashed(ctx, inst, event) -> None:
    error = event.data
    inst.rejection = f"clone refused; no clone was created: internal error ({type(error).__name__}: {error})"


def committed(ctx, inst, event) -> None:
    """The clone call is in the store: its grafts' machines are made next (`spawning`). A focus
    graft is made with the user's calls waiting on this run, and answers the same request."""
    inst.clone_text = event.data
    users = [w for w in inst.waiters if w.user]
    inst.spawn_plan = [GraftPlan(child=child, focus=f, waiters=users if f else [],
                                 answering=inst.answering if f else None,
                                 made_for=None if f else inst.answering)
                       for child, f in inst.clone.pairs]


def row_not_written(ctx, inst, event) -> None:
    """The clone call's result row could not be written: kept, and written with the run's open
    calls, by the run's closing (`closing_calls`) or, before a hand-over, by `flushing`."""
    inst.unstored[inst.clone.use["id"]] = (inst.clone_text, False)
    _report(inst=inst, kind="task", clones=[], error=event.data)


def spawned(ctx, inst, event) -> None:
    """Every graft's machine started: only now are they the original's, and the session told."""
    clone = inst.clone
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
    background = [g for g, f in clone.pairs if not f]
    _session(inst=inst, name="grafts_made", data=(background, inst.mode.name == "external"))


def spawn_failed(ctx, inst, event) -> None:
    """A graft's machine could not be started: the ones made are undone (`unspawning`)."""
    inst.spawned_ids, inst.spawn_error = event.data
    inst.spawn_plan = []


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
    if clone is not None:
        _resolve(reply=clone.reply, value=(None, None, "clone not executed: the turn ended before it ran"))
        if clone.cancels is not None and clone.cancels is not cancel:
            _resolve(reply=clone.cancels.reply, value="the turn ended before the cancel ran")
    if cancel is not None:
        _resolve(reply=cancel.reply, value="the turn ended before the cancel ran")


def pass_inbox(ctx, inst, event) -> None:
    """The messages waiting on the original go to the focus graft with the conversation, in
    order and with the withdrawals recorded, ahead of any message forwarded to it later."""
    inbox, inst.inbox = inst.inbox, []
    _forward(target=inst.machines.timelines.get(inst.focus), name="inbox_handed", data=(inbox, set(inst.withdrawn)))


def hand_over(ctx, inst, event) -> None:
    """Suspended: the user's calls now wait on the focus graft, which was made with them and
    with the request it answers; the session no longer counts the original as engaged (the
    focus graft counts itself while it runs)."""
    clone, inst.handing = inst.handing, None
    for w in inst.waiters:
        if not w.user:
            _resolve(reply=w.reply, value="handed_off")
    inst.waiters, inst.answering = [], None
    _prune(inst)
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
    _session(inst=inst, name="graft_received", data=join.child)
    inst.pending.discard(join.child)
    inst.outcomes[join.child] = join.reason
    inst.blocks |= {(inst.cancel.request, inst.parts.get(join.child, frozenset()))} if join.reason == CANCELLED else set()


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
    _session(inst=inst, name="engaged", data=inst.timeline_id)
    add_run_waiter(ctx=ctx, inst=inst, event=event)


def close_and_engage(ctx, inst, event) -> None:
    _session(inst=inst, name="engaged", data=inst.timeline_id)
    inst.closing = inst.final = True


def focus_back(ctx, inst, event) -> None:
    """Its focus graft joined: the conversation is back, with the calls waiting on the graft,
    the messages it never got to (in order, ahead of any that came back since), the request
    it was answering and the withdrawals it recorded."""
    join, waiters, inbox, answering, withdrawn = event.data
    _session(inst=inst, name="handed", data=(inst.focus, inst.timeline_id))
    inst.focus = None
    inst.queue.append(join)
    inst.waiters.extend(waiters)
    inst.inbox = list(inbox) + inst.inbox
    inst.answering = answering
    inst.withdrawn |= withdrawn


def start_joining(ctx, inst, event) -> None:  # entry of `joining`: a commit point
    inst.hold, inst.news, inst.ends_on_request = False, [], False


def applied(ctx, inst, event) -> None:
    """One join applied: it leaves the queue, the graft is told, and a hand-back is counted under
    the request the graft was working on (blocking that part if it failed or came back twice)."""
    committed_, request, ends = event.data
    join = inst.queue.pop(HEAD)
    inst.grafts.pop(join.child, None)
    _forward(target=inst.machines.timelines.get(join.child), name="join_commit" if committed_ else "join_rollback")
    if (not inst.made.get(join.child) or join.reason != "done") and not _unwanted(inst=inst, child=join.child):
        inst.news.append((join.child, join.reason))
    inst.graft_requests.pop(join.child, None)
    inst.ends_on_request = ends
    if join.reason not in ("done", CANCELLED):
        part = inst.parts.get(join.child, frozenset())
        inst.handbacks[request] = inst.handbacks.get(request, NONE_YET) + 1
        n = inst.arrivals[(request, part)] = inst.arrivals.get((request, part), NONE_YET) + 1
        inst.blocks |= {(request, part)} if join.reason == "failed" or n >= 2 else set()


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


def note_failed(ctx, inst, event) -> None:
    """The wake note could not be written: the user is told; the original comes to rest next."""
    _report(inst=inst, kind="wake", clones=[c for c, _ in inst.news], error=event.data)


# ------------------------------------------------------------------ effects: a focus graft's join
def _hand_back(inst: TimelineSM, join: QueuedJoin, also: list[Waiter]) -> None:
    users = [w for w in inst.waiters if w.user] + also
    for w in inst.waiters:
        if not w.user:
            _resolve(reply=w.reply, value="joined")
    inst.waiters = []
    inbox, inst.inbox = inst.inbox, []
    answering, inst.answering = inst.answering, None
    _forward(target=inst.machines.timelines.get(inst.source_id), name="focus_joined",
             data=(join, users, inbox, answering, set(inst.withdrawn)))


def focus_join(ctx, inst, event) -> None:
    """A focus graft raises its join event: frozen, its join goes on its source's queue, with the
    calls and messages waiting on it; the caller of the join is answered at the source's rest."""
    join, reply = event.data
    _hand_back(inst=inst, join=join, also=[Waiter(reply=reply, user=False)] if reply is not None else [])


def hand_back_join(ctx, inst, event) -> None:
    """Its join event (computed by the state just left) goes to its source."""
    _hand_back(inst=inst, join=event.data, also=[])


def focus_failure_not_recorded(ctx, inst, event) -> None:
    """Its failure could not be recorded: the user is told; it joins `failed` next."""
    _report(inst=inst, kind="task", clones=[inst.timeline_id], error=event.data, src=inst.source_id)


def focus_failed_unrecorded(ctx, inst, event) -> None:
    _hand_back(inst=inst, join=QueuedJoin(child=inst.timeline_id, reason="failed", trigger=None), also=[])


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
    inst.unreported = _unreported(inst=inst, errors=event.data)


def record_crashed(ctx, inst, event) -> None:
    inst.unreported = _unreported(inst=inst, errors=[event.data])


def join_raised(ctx, inst, event) -> None:
    """Its join event goes to its source; whoever waited for it is answered."""
    watchers, inst.watchers = inst.watchers, []
    for w in watchers:
        _resolve(reply=w, value=True)
    _resolve(reply=inst.forced, value=True)
    _forward(target=inst.machines.timelines.get(inst.source_id), name="graft_queued",
             data=QueuedJoin(child=inst.timeline_id, reason=inst.reason, trigger=None))


# ------------------------------------------------------------------ effects: failures told one per step
def report_next(ctx, inst, event) -> None:  # entry of a reporting state: the next failure
    _self(inst=inst, name="report_next")


def has_unreported(ctx, inst, event) -> bool:
    return bool(inst.unreported)


def report_one(ctx, inst, event) -> None:
    kind, clones, error, src = inst.unreported.pop(HEAD)
    _report(inst=inst, kind=kind, clones=clones, error=error, src=src)


def add_watcher(ctx, inst, event) -> None:
    inst.watchers.append(event.data)


# ------------------------------------------------------------------ the graphs
REQUESTS = ("turn_request", "drain_request", "join", "user_message", "settle_request", "driven_done")


def _work(name: str, *activity: hsm.RedefinableBehaviors) -> hsm.RedefinableState:
    """What a running timeline does: run its model loop (`run`, an activity) or wait for the
    caller that drives the turn (`driven`). Both clone and cancel, step by step. A clone call is
    gated whole first (`clone_gate`: nothing is seeded for a call that may not be tried), then
    prepared (`preparing`, the harness's activity: leaving the run ends it), then decided and carried
    out in `steps/cloning`; the cancel calls of one response are `steps/cancels`, and a clone call
    carried with them goes to the gate after them. While a step runs, shutdown and a driven end
    wait (`steps`)."""
    T, S, On, G, E = hsm.Transition, hsm.Target, hsm.On, hsm.Guard, hsm.ErrorEvent
    return hsm.State(
        name,
        *activity,
        hsm.Exit(answer_pending),  # however the run is left, a clone or cancel request it holds is answered
        hsm.Initial(S("running")),
        hsm.State(
            "running",
            T(On("clone_request"), S("../steps/cancels"), G(carries_cancels), hsm.Effect(take_clone)),
            T(On("clone_request"), S("../clone_gate"), G(no_cancels), hsm.Effect(take_clone)),
            T(On("cancel_request"), S("../steps/cancels"), hsm.Effect(take_cancel)),
        ),
        hsm.Choice(
            "clone_gate",  # the whole call: may it be tried at all? Decided before anything is seeded
            T(S("steps/cloning/rejecting"), G(declined), hsm.Effect(decline)),
            T(S("preparing"), hsm.Effect(attempt_clone)),
        ),
        hsm.State(
            "preparing", hsm.Activity(prepare_activity),
            T(On("prepared"), S("../prepared_gate"), hsm.Effect(take_prepared)),
            T(On(E), S("../steps/cloning/rejecting"), hsm.Effect(preparation_crashed)),
            # left early (the run's request withdrawn, the run raised, shutdown, a driven turn
            # ended): the call it was preparing is rolled back in a state of its own before the
            # run's closing. A run that raised is told the way `busy` tells it (only the original
            # prepares a clone call, so the focus graft's branch does not arise here)
            T(On("withdraw"), S("../../../unpreparing"), G(ends_run), hsm.Effect(note_withdrawn)),
            T(On("run_failed"), S("../../../unpreparing"), G(error_to_user), hsm.Effect(error_for_user)),
            T(On("run_failed"), S("../../../unpreparing"), G(error_already_shown), hsm.Effect(error_shown)),
            T(On("run_failed"), S("../../../unpreparing"), hsm.Effect(error_reported)),
            T(On("shutdown"), S("../../../unpreparing"), hsm.Effect(interrupted_closing)),
            T(On("driven_done"), S("../../../unpreparing"), hsm.Effect(end_driven)),
        ),
        hsm.Choice(
            "prepared_gate",
            T(S("steps/cloning/rejecting"), G(rejected_in_preparation)),
            T(S("steps/cloning/refusing"), G(graph_refuses)),
            T(S("steps/cloning/committing")),
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
                T(On("withdraw"), G(withdraws_waiting), hsm.Effect(note_withdrawn)),
                T(On("withdraw"), hsm.Effect(withdraw_and_forward)),
                hsm.State(
                    "rejecting", io(done="rejected", work=_reject),  # refused before or in its preparation
                    T(On("rejected"), S("../../../resume"), hsm.Effect(clone_refused_written)),
                    T(On(E), S("../../../resume"), hsm.Effect(clone_refusal_not_written)),
                ),
                hsm.State(
                    "refusing", io(done="refused", work=_refuse),
                    T(On("refused"), S("../../../resume"), hsm.Effect(clone_refused_written)),
                    T(On(E), S("../../../resume"), hsm.Effect(clone_refusal_not_written)),
                ),
                hsm.State(
                    "committing", io(done="committed", work=_commit),
                    T(On("committed"), S("../spawning"), hsm.Effect(committed)),
                    T(On(E), S("../refusing")),
                ),
                hsm.State(
                    "spawning", hsm.Activity(spawn_activity),
                    # a withdrawal waits for the grafts' machines (the activity reads no data of
                    # this one after it starts), then is handled by `cloning` like any later one
                    hsm.Defer("withdraw"),
                    T(On("spawned"), S("../recording"), hsm.Effect(spawned)),
                    T(On("spawn_failed"), S("../unspawning"), hsm.Effect(spawn_failed)),
                ),
                hsm.State(
                    "unspawning", hsm.Activity(unspawn_activity),
                    T(On("unspawned"), S("../refusing")),
                    T(On(E), S("../refusing")),
                ),
                hsm.State(
                    "recording", io(done="row_written", work=_clone_row),
                    T(On("row_written"), S("../../../clone_choice")),
                    T(On(E), S("../../../clone_choice"), hsm.Effect(row_not_written)),
                ),
                hsm.State(
                    "flushing", io(done="flushed", work=_close_calls),  # before a hand-over: the run's rows
                    T(On("flushed"), S("../../../handed"), hsm.Effect(calls_closed)),
                    T(On(E), S("../../../handed"), hsm.Effect(calls_not_closed)),
                ),
            ),
            hsm.State(
                "cancels",  # the cancel calls of one response, id by id; a clone call carried with them waits
                hsm.Entry(checks_from),
                hsm.Defer("clone_request"),
                T(On("withdraw"), hsm.Effect(note_withdrawn)),  # the run ends once the cancels are done
                T(On("graft_queued"), G(one_of_pending), hsm.Effect(cancel_landed)),
                hsm.Initial(S("checking")),
                hsm.State(
                    "checking", hsm.Entry(start_checking),
                    T(On("check"), S("../check_choice")),
                ),
                hsm.Choice(
                    "check_choice",
                    T(S("checking"), G(head_cancellable), hsm.Effect(cancel_head)),
                    T(S("checking"), G(head_refused), hsm.Effect(refuse_head)),
                    T(S("waiting"), G(checked_with_pending)),
                    T(S("writing")),
                ),
                hsm.State(
                    "waiting",  # until each cancelled graft's join is queued
                    T(On("graft_queued"), S("../writing"), G(last_pending), hsm.Effect(cancel_landed)),
                ),
                hsm.State(
                    "writing", io(done="cancel_rows", work=_cancel_rows),  # each call's real result
                    T(On("cancel_rows"), S("../../../cancels_done"), G(rows_written), hsm.Effect(cancels_written)),
                    T(On("cancel_rows"), S("../row_choice"), G(rows_failed), hsm.Effect(take_row_failures)),
                    T(On(E), S("../../../cancels_done"), hsm.Effect(cancel_rows_crashed)),
                ),
                hsm.State(
                    "reporting",  # a row that could not be written: each failure is told, one per step
                    hsm.Entry(report_next),
                    T(On("report_next"), S("../row_choice")),
                ),
                hsm.Choice(
                    "row_choice",
                    T(S("reporting"), G(has_unreported), hsm.Effect(report_one)),
                    T(S("../../cancels_done"), hsm.Effect(cancel_rows_failed)),
                ),
            ),
        ),
        hsm.Choice(
            "clone_choice",  # a hand-over leaves the run: its open calls and kept rows are written first
            T(S("steps/cloning/flushing"), G(hands_over), hsm.Effect(take_hand_over)),
            T(S("resume"), hsm.Effect(clone_done)),
        ),
        hsm.Choice(
            "resume",  # after a clone step: a run whose request was withdrawn meanwhile ends
            T(S("../../closing_calls"), G(run_withdrawn)),
            T(S("running")),
        ),
        hsm.Choice("handed", T(S("../../../passing"), hsm.Effect(pass_inbox))),
        hsm.Choice(
            "cancels_done",  # a withdrawal that came meanwhile ends the run: what waited is not run
            T(S("../../closing_calls"), G(run_withdrawn)),
            T(S("clone_gate"), G(clone_waits)),
            T(S("running"), G(clone_dropped), hsm.Effect(drop_clone)),  # the run then ends with the error
            T(S("running")),
        ),
    )


def _timeline(name: str) -> hsm.Model:
    T, S, On, G, E = hsm.Transition, hsm.Target, hsm.On, hsm.Guard, hsm.ErrorEvent
    return hsm.Define(
        name,
        hsm.Initial(S("start")),
        hsm.Choice(
            "start",  # a focus graft starts on its request, unless a caller drives the turn that made it
            T(S("live/engaged/busy/run"), G(starts_busy), hsm.Effect(engage)),
            # its request was withdrawn before it could act: it never acts on it, and the request's
            # written row is marked (in its view and its source's) before it is at rest
            T(S("live/engaged/withdrawing"), G(starts_withdrawn), hsm.Effect(engage)),
            T(S("live/idle")),
        ),
        T(On(*REQUESTS, "when_queued", "may_continue"), hsm.Effect(refuse_request)),
        T(On("withdraw"), hsm.Effect(note_withdrawn)),  # in every state: recorded, never ignored
        T(On("result_unstored"), hsm.Effect(keep_unstored)),
        T(On("inbox_handed"), hsm.Effect(take_handed)),
        T(On("rest_request"), G(has_joined), hsm.Effect(answer_joined)),
        T(On("rest_request"), hsm.Effect(refuse_request)),  # closed, or its join abandoned
        T(On("graft_queued"), G(own_graft), hsm.Effect(enqueue)),
        T(On("graft_queued"), hsm.Effect(ignore)),  # a graft of a clone call that was undone at its start
        T(On("clone_request"), hsm.Effect(refuse_clone_call)),
        T(On("cancel_request"), hsm.Effect(refuse_cancel_call)),
        T(On("poke", "shutdown", "check"), hsm.Effect(ignore)),
        # a run that raised during a step that then ended it (a hand-over, its request withdrawn)
        T(On("run_failed"), hsm.Effect(run_failed_after_it_ended)),
        T(On("closing"), hsm.Effect(set_closing)),
        hsm.State(
            "live",
            T(On("join"), S("../queued"), G(is_focus), hsm.Effect(focus_join)),
            T(On("shutdown"), S("../closed"), hsm.Effect(close_at_rest)),
            hsm.State(
                "idle",
                mirrored("active"),
                T(On("user_message"), G(arrives_withdrawn), hsm.Effect(drop_message)),
                T(On("user_message"), S("../engaged/joining"), G(arrives_behind_joins), hsm.Effect(stash_and_engage)),
                T(On("user_message"), S("../engaged/requesting"), G(arrives_to_answer), hsm.Effect(take_message)),
                T(On("turn_request"), S("../engaged/busy/driven"), G(is_external), hsm.Effect(begin_driven)),
                T(On("turn_request"), S("../engaged/busy/run"), G(not_external), hsm.Effect(begin_run)),
                T(On("graft_queued"), S("../receiving"), G(own_graft), hsm.Effect(engage)),
                T(On("drain_request"), S("../engaged/joining"), G(has_queue), hsm.Effect(engage_with_waiter)),
                T(On("drain_request"), G(no_queue), hsm.Effect(answer_ended)),
                T(On("rest_request"), hsm.Effect(answer_ended)),
                T(On("poke"), S("../engaged/joining"), G(has_queue), hsm.Effect(engage)),
                T(On("shutdown"), S("../engaged/joining"), G(has_queue), hsm.Effect(close_and_engage)),
                # messages handed over with the conversation (or back, past a focus graft that had
                # already joined) reach it at rest
                T(On("inbox_handed"), S("../engaged/joining"), G(handed_live_with_queue),
                  hsm.Effect(take_handed_and_engage)),
                T(On("inbox_handed"), S("../engaged/requesting"), G(handed_live_to_answer),
                  hsm.Effect(take_handed_and_begin)),
            ),
            hsm.State(
                "engaged",
                hsm.Defer("shutdown"),  # taken by `busy` and `joining`; every other step finishes first
                T(On("user_message"), hsm.Effect(stash_message)),
                T(On("rest_request", "drain_request"), hsm.Effect(add_run_waiter)),
                hsm.State(
                    "requesting", mirrored("active"), io(done="request_written", work=_write_request),
                    T(On("request_written"), S("../busy/run"), G(request_live), hsm.Effect(request_written)),
                    T(On("request_written"), S("../dropping"), hsm.Effect(request_dropped)),  # withdrawn meanwhile
                    T(On(E), S("../rest"), hsm.Effect(request_failed)),
                ),
                hsm.State(
                    "dropping", io(done="dropped", work=_mark_dropped),  # its row gets the withdrawal mark
                    T(On("dropped"), S("../rest")),
                    T(On(E), S("../rest"), hsm.Effect(mark_failed)),
                ),
                hsm.State(
                    "busy",
                    mirrored("active"),
                    T(On("run_failed", E), S("../failing"), G(is_focus)),
                    T(On("run_failed", E), S("../closing_calls"), G(error_to_user), hsm.Effect(error_for_user)),
                    T(On("run_failed", E), S("../closing_calls"), G(error_already_shown), hsm.Effect(error_shown)),
                    T(On("run_failed", E), S("../closing_calls"), hsm.Effect(error_reported)),
                    T(On("withdraw"), S("../closing_calls"), G(ends_run), hsm.Effect(note_withdrawn)),
                    # the run asks before each step that could act: it goes on while the request it
                    # answers, if any, is live; once it is withdrawn the run is ended the way a
                    # withdrawal ends it (leaving `busy` ends the run that asked: nothing is left to answer)
                    T(On("may_continue"), S("../closing_calls"), G(run_withdrawn)),
                    T(On("may_continue"), hsm.Effect(answer_true)),
                    T(On("inbox_handed"), S("../closing_calls"), G(handed_withdraws_run), hsm.Effect(take_handed)),
                    T(On("shutdown"), S("../closing_calls"), hsm.Effect(interrupted_closing)),
                    _work("run", hsm.Activity(busy_activity), T(On("turn_done"), S("../../ran"), hsm.Effect(end_run))),
                    _work("driven", T(On("driven_done"), S("../../rest"), hsm.Effect(end_driven))),
                ),
                hsm.State(
                    "unpreparing", io(done="unprepared", work=_unprepare),  # a clone call cut short is rolled back
                    T(On("unprepared"), S("../unwound")),
                    T(On(E), S("../unwound"), hsm.Effect(unprepare_failed)),
                ),
                hsm.Choice(
                    "unwound",
                    T(S("rest"), G(is_driven)),  # a driven turn: its caller closes its own calls
                    T(S("closing_calls")),
                ),
                hsm.Choice(
                    "ran",  # how the run ended: a focus graft that joins hands back; anything else rests
                    T(S("handing_back"), G(joins_back)),
                    T(S("rest")),
                ),
                hsm.State(
                    "handing_back", io(done="join_computed", work=_join_back),  # a focus graft's join event
                    T(On("join_computed"), S("../../../queued"), hsm.Effect(hand_back_join)),
                    T(On(E), S("../unrecorded"), hsm.Effect(focus_failure_not_recorded)),
                ),
                hsm.State(
                    "closing_calls", io(done="calls_closed", work=_close_calls),
                    T(On("calls_closed"), S("../withdrawn_choice"), hsm.Effect(calls_closed)),
                    T(On(E), S("../withdrawn_choice"), hsm.Effect(calls_not_closed)),
                ),
                hsm.Choice(
                    "withdrawn_choice",  # the run it cut short answered a request that was withdrawn
                    T(S("withdrawing"), G(cut_short)),
                    T(S("rest")),
                ),
                hsm.State(
                    # the withdrawn request it answered gets the withdrawal mark on its row
                    "withdrawing", io(done="rows_marked", work=_mark_withdrawn),
                    T(On("rows_marked"), S("../rest")),
                    T(On(E), S("../rest"), hsm.Effect(mark_failed)),
                ),
                hsm.State(
                    "failing", io(done="failure_recorded", work=_focus_failure),
                    T(On("failure_recorded"), S("../../../queued"), hsm.Effect(hand_back_join)),
                    T(On(E), S("../unrecorded"), hsm.Effect(focus_failure_not_recorded)),
                ),
                hsm.Choice("unrecorded", T(S("../../queued"), hsm.Effect(focus_failed_unrecorded))),
                hsm.State(
                    "joining",
                    mirrored("joining"),
                    hsm.Entry(start_joining),
                    T(On("shutdown"), hsm.Effect(set_final)),
                    hsm.Initial(S("next")),
                    hsm.Choice(
                        "next",  # one join per step, while one can be applied
                        T(S("applying"), G(applies_next)),
                        T(S("../after")),
                    ),
                    hsm.State(
                        "applying", io(done="applied", work=_apply),
                        T(On("applied"), S("../next"), hsm.Effect(applied)),
                        T(On(E), S("../next"), G(retry_left), hsm.Effect(apply_failed)),
                        T(On(E), S("../giving_up"), G(last_attempt), hsm.Effect(last_commit_failed)),
                    ),
                    hsm.Choice("giving_up", T(S("abandoning"), hsm.Effect(report_abandoned))),
                    hsm.Choice("abandoning", T(S("next"), hsm.Effect(abandon_head))),
                ),
                hsm.State(
                    "noting", mirrored("active"), io(done="noted", work=_wake_note),
                    T(On("noted"), S("../busy/run"), G(answering_live)),
                    T(On("noted"), S("../withdrawing"), G(noted_withdrawn)),  # its caller withdrew meanwhile
                    T(On("noted"), S("../rest")),
                    T(On(E), S("../unnoted"), hsm.Effect(note_failed)),
                ),
                hsm.Choice("unnoted", T(S("../idle"), hsm.Effect(come_to_rest))),
                hsm.Choice(
                    "rest",
                    T(S("joining"), G(drains_now)),
                    T(S("../shutting"), G(closes_now), into("active")),
                    T(S("requesting"), G(answers_inbox), hsm.Effect(take_inbox)),
                    T(S("../idle"), hsm.Effect(come_to_rest)),
                ),
                hsm.Choice(
                    "after",
                    T(S("../shutting"), G(is_final), into("active")),
                    T(S("../idle"), G(final_waits), hsm.Effect(come_to_rest)),
                    T(S("withdrawing"), G(withdrawn_after)),  # closed out before anything else is taken
                    T(S("requesting"), G(inbox_after), hsm.Effect(take_inbox)),
                    T(S("../idle"), G(quiet_after), hsm.Effect(come_to_rest)),
                    T(S("busy/run"), G(answers_handback), hsm.Effect(answer_handback)),
                    T(S("noting"), G(wakes_to_route), hsm.Effect(wake_route)),
                    T(S("noting"), hsm.Effect(wake_report)),
                ),
            ),
            hsm.Choice("shutting", T(S("../closed"), hsm.Effect(close_root))),
            hsm.Choice("passing", T(S("cloned"), hsm.Effect(hand_over))),
            hsm.Choice("receiving", T(S("engaged/joining"), hsm.Effect(enqueue))),
            hsm.State(
                "cloned",
                mirrored("cloned"),
                T(On("user_message"), hsm.Effect(forward_to_focus)),
                T(On("withdraw"), hsm.Effect(withdraw_and_forward)),
                T(On("focus_joined"), S("../engaged/joining"), hsm.Effect(focus_back)),
                T(On("rest_request"), hsm.Effect(answer_handed_off)),
                T(On("shutdown"), S("../releasing"), hsm.Effect(close_and_release)),
            ),
            hsm.State(
                "releasing", hsm.Entry(release_next),
                T(On("released"), S("../../closed"), hsm.Effect(close_at_rest)),
            ),
        ),
        hsm.State(
            "queued",
            mirrored("queued"),
            T(On("join_commit", "join_rollback"), S("../joined"), into("joined")),
            T(On("join_abandon"), S("../abandoned"), into("abandoned")),
            # a withdrawal reaches a focus graft only from its source: here it holds no request,
            # so the withdrawal changes nothing (top level) and is not sent back
            T(On("user_message", "inbox_handed"), hsm.Effect(forward_to_source)),
            T(On("rest_request"), hsm.Effect(answer_joined)),  # its run has ended by joining
        ),
        hsm.Final("joined"),
        hsm.Final("abandoned"),
        hsm.Final("closed"),
    )


TIMELINE = _timeline("Timeline")

_T, _S, _On, _G, _E = hsm.Transition, hsm.Target, hsm.On, hsm.Guard, hsm.ErrorEvent
BACKGROUND = hsm.Define(
    "BackgroundTimeline",
    hsm.Initial(_S("start")),
    hsm.Choice(
        "start",  # made in a turn its caller drives: it does not run; the caller joins it
        _T(_S("live/background/held"), _G(is_held)),
        # made for a request withdrawn before it was made: it never runs, and joins cancelled
        _T(_S("live/recording"), _G(starts_cancelled), hsm.Effect(ended_as(reason=CANCELLED, close_first=False))),
        _T(_S("live/background/running")),
    ),
    _T(_On(*REQUESTS, "rest_request", "may_continue"), hsm.Effect(refuse_request)),  # not a timeline that rests
    _T(_On("when_queued"), hsm.Effect(answer_true)),
    _T(_On("result_unstored"), hsm.Effect(keep_unstored)),
    _T(_On("cancel_request"), hsm.Effect(refuse_cancel_call)),
    _T(_On("clone_request"), hsm.Effect(refuse_clone_call)),
    _T(_On("cancel", "shutdown", "finish", "closing"), hsm.Effect(ignore)),
    hsm.State(
        "live",  # until its join event is raised: callers waiting for it are kept
        _T(_On("when_queued"), hsm.Effect(add_watcher)),
        hsm.State(
            "background",
            mirrored("background"),
            _T(_On("finish"), _S("../ending")),
            _T(_On("cancel"), _S("../recording"), hsm.Effect(ended_as(reason=CANCELLED, close_first=True))),
            _T(_On("shutdown"), _S("../recording"), hsm.Effect(ended_as(reason="failed", close_first=True))),
            _T(_On("join"), _S("../recording"), hsm.Effect(forced)),
            _T(_On(_E), _S("../recording"), hsm.Effect(run_raised)),
            hsm.State("held"),
            # its run goes on until the graft leaves `background` (finish, cancel, shutdown, join);
            # it asks before each step, and stops (joins cancelled) once its request is withdrawn
            hsm.State(
                "running", hsm.Activity(background_activity),
                _T(_On("may_continue"), _S("../../recording"), _G(made_for_withdrawn),
                   hsm.Effect(ended_as(reason=CANCELLED, close_first=True))),
                _T(_On("may_continue"), hsm.Effect(answer_true)),
            ),
        ),
        hsm.Choice(
            "ending",  # how its run ended: its join reason
            _T(_S("recording"), _G(run_joined), hsm.Effect(ended_by_join)),
            _T(_S("recording"), _G(run_replied), hsm.Effect(ended_as(reason="done", close_first=False))),
            _T(_S("recording"), hsm.Effect(ended_as(reason="failed", close_first=False))),
        ),
        hsm.State(
            "recording",  # its join reason is written, each failure is told, then its join is raised
            hsm.Initial(_S("writing")),
            hsm.State(
                "writing", io(done="recorded", work=_record),
                _T(_On("recorded"), _S("../report_choice"), hsm.Effect(recorded)),
                _T(_On(_E), _S("../report_choice"), hsm.Effect(record_crashed)),
            ),
            hsm.State(
                "reporting", hsm.Entry(report_next),
                _T(_On("report_next"), _S("../report_choice")),
            ),
            hsm.Choice(
                "report_choice",
                _T(_S("reporting"), _G(has_unreported), hsm.Effect(report_one)),
                _T(_S("../../queued"), hsm.Effect(join_raised)),
            ),
        ),
    ),
    hsm.State(
        "queued",
        mirrored("queued"),
        _T(_On("join_commit", "join_rollback"), _S("../joined"), into("joined")),
        _T(_On("join_abandon"), _S("../abandoned"), into("abandoned")),
    ),
    hsm.Final("joined"),
    hsm.Final("abandoned"),
)


# ------------------------------------------------------------------ the scribe
def take_note(ctx, inst, event) -> None:
    inst.note = event.data


def keeps(ctx, inst, event) -> bool:
    return inst.note[1] is not None


def is_write(ctx, inst, event) -> bool:
    return inst.note[0][0] != "report"


def s_keep(ctx, inst, event) -> None:
    _forward(target=inst.machines.session, name="failure", data=inst.note[1])


def s_keep_lost_line(ctx, inst, event) -> None:
    """A report line the user's callbacks could not take: that failure is kept for `settle`."""
    _forward(target=inst.machines.session, name="failure", data=event.data)


def s_report_write(ctx, inst, event) -> None:
    """A store write the scribe could not make (a state's store copy, a trace line): the user is
    told and the session keeps it for `settle`. Not retried: the next copy of a state overwrites
    it, and the machines never read the store copy."""
    line = dict(kind="mirror", src=None, clones=[], error=event.data, quiet=False, background=False)
    _self(inst=inst, name="note", data=(("report", line), event.data))


SCRIBE = hsm.Define(
    "Scribe",
    hsm.Initial(_S("ready")),
    hsm.State("ready", _T(_On("note"), _S("../writing"), hsm.Effect(take_note))),
    hsm.State(
        "writing", io(done="written", work=_scribe_write),
        hsm.Defer("note"),
        _T(_On("written"), _S("../ready"), _G(keeps), hsm.Effect(s_keep)),
        _T(_On("written"), _S("../ready")),
        _T(_On(_E), _S("../ready"), _G(is_write), hsm.Effect(s_report_write)),
        _T(_On(_E), _S("../lost_line"), hsm.Effect(s_keep_lost_line)),  # a report the callbacks failed
    ),
    hsm.Choice(
        "lost_line",  # the error the report was about is still kept
        _T(_S("ready"), _G(keeps), hsm.Effect(s_keep)),
        _T(_S("ready")),
    ),
)


# ------------------------------------------------------------------ the session
def _apply_session(inst: SessionSM, name: str, data: Any) -> None:
    """How each event changes the session's own record of what is in flight."""
    if name == "engaged":
        inst.engaged.add(data)
    elif name == "request_taken":  # (timeline, the user call taken, waiting messages dropped)
        inst.engaged.add(data[0])
        inst.users.update(data[1])
        inst.users.difference_update(data[2])
    elif name in ("rested", "root_closed"):
        inst.engaged.discard(data[0])
        inst.users.difference_update(data[1])
    elif name == "handed":
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
    elif name == "user_waiting":
        inst.users.update(data)
    elif name == "user_answered":
        inst.users.difference_update(data)


def _busy_after(inst: SessionSM, event) -> bool:
    engaged, grafts = set(inst.engaged), set(inst.grafts)
    probe = SessionSM.__new__(SessionSM)
    probe.engaged, probe.grafts, probe.held, probe.users = engaged, grafts, set(inst.held), set(inst.users)
    _apply_session(inst=probe, name=event.name, data=event.data)
    return bool(probe.engaged or probe.grafts)


def s_busy_after(ctx, inst, event) -> bool:
    return _busy_after(inst=inst, event=event)


def s_quiet_after(ctx, inst, event) -> bool:
    return not _busy_after(inst=inst, event=event)


def s_record(ctx, inst, event) -> None:
    _apply_session(inst=inst, name=event.name, data=event.data)


def s_record_and_answer(ctx, inst, event) -> None:
    _apply_session(inst=inst, name=event.name, data=event.data)
    s_answer(ctx=ctx, inst=inst, event=event)


def s_answer(ctx, inst, event) -> None:
    """Nothing is in flight: every `settle` waiting is answered, with the errors kept since the
    last answer (each raised once)."""
    settles, inst.settles = [s for s in inst.settles if not s.done()], []  # a caller gone is not answered
    errors, inst.errors = inst.errors, []
    if not settles:
        inst.errors = errors  # kept for the next settle
        return
    first, *rest = settles
    if len(errors) == 1:
        _fail(reply=first, error=errors[0])
    elif errors:
        _fail(reply=first, error=ExceptionGroup("background grafts failed", errors))
    else:
        _resolve(reply=first, value=None)
    for later in rest:
        _resolve(reply=later, value=None)


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
    """Kept for `settle` unless a user's call is waiting: then it is that call's to report."""
    inst.errors += [event.data] if not inst.users and not any(event.data is e for e in inst.errors) else []


def s_closing(ctx, inst, event) -> None:
    """Close: the original is told first (nothing is woken from now on); the grafts are shut
    down one per step (`shutting`), then the original once their joins have reached it."""
    inst.closers += [event.data] if event.data is not None else []
    inst.to_shut = sorted(inst.grafts | inst.held)
    inst.grafts |= inst.held
    inst.held = set()
    _forward(target=inst.machines.timelines.get(inst.root), name="closing")


def s_shut_next(ctx, inst, event) -> None:  # entry of `shutting`
    _self(inst=inst, name="shut_next")


def s_shut_one(ctx, inst, event) -> None:
    _forward(target=inst.machines.timelines.get(inst.to_shut.pop(HEAD)), name="shutdown")


def s_wait_joins(ctx, inst, event) -> None:  # entry of `draining`
    _self(inst=inst, name="joins_check")


def s_shut_root(ctx, inst, event) -> None:
    _apply_session(inst=inst, name=event.name, data=event.data)
    _forward(target=inst.machines.timelines.get(inst.root), name="shutdown")


def s_stash_closer(ctx, inst, event) -> None:
    inst.closers += [event.data] if event.data is not None else []


def s_closed(ctx, inst, event) -> None:
    """Closed: every close and settle waiting is answered; a machine that could not be stopped
    is the first close's error (the one error, or an ExceptionGroup of all of them)."""
    closers, inst.closers = inst.closers, []
    settles, inst.settles = inst.settles, []
    errors, inst.stop_errors = inst.stop_errors, []
    if closers and errors:
        first, *closers = closers
        _fail(reply=first, error=errors[0] if len(errors) == 1 else ExceptionGroup("stopping machines failed", errors))
    for reply in closers + settles:
        _resolve(reply=reply, value=None)


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
    return event.data[0] == inst.root


RECORDS = ("engaged", "request_taken", "rested", "handed", "suspended", "grafts_made", "graft_received",
           "user_waiting", "user_answered")

SESSION = hsm.Define(
    "Session",
    hsm.Initial(_S("open/quiet")),
    _T(_On(*REQUESTS[:-2], "when_queued", "rest_request"), hsm.Effect(refuse_request)),
    _T(_On("failure"), hsm.Effect(s_failure)),
    _T(_On(*RECORDS), hsm.Effect(s_record)),
    _T(_On("settle_request", "shutdown"), hsm.Effect(s_answer_now)),
    _T(_On("settle_withdrawn"), hsm.Effect(s_withdraw)),
    _T(_On("checked", "root_closed", "shut_next", "joins_check"), hsm.Effect(ignore)),
    hsm.State(
        "open",
        _T(_On(*RECORDS), _S("working"), _G(s_busy_after), hsm.Effect(s_record)),
        _T(_On("settle_request"), hsm.Effect(s_stash_settle)),
        _T(_On("shutdown"), _S("../closing"), hsm.Effect(s_closing)),
        hsm.State(
            "quiet",
            _T(_On(*RECORDS), _G(s_quiet_after), hsm.Effect(s_record)),
            _T(_On("settle_request"), _S("../checking"), hsm.Effect(s_check)),
        ),
        hsm.State("checking", hsm.Entry(s_checked), _T(_On("checked"), _S("../quiet"), hsm.Effect(s_answer))),
        hsm.State(
            "working",
            _T(_On(*RECORDS), _S("../quiet"), _G(s_quiet_after), hsm.Effect(s_record_and_answer)),
            _T(_On(*RECORDS), _G(s_busy_after), hsm.Effect(s_record)),
        ),
    ),
    hsm.State(
        "closing",
        _T(_On("settle_request"), hsm.Effect(s_stash_settle)),
        _T(_On("shutdown"), hsm.Effect(s_stash_closer)),
        _T(_On("root_closed"), _S("stopping"), _G(is_root_event)),
        _T(_On("grafts_made"), _S("shutting"), hsm.Effect(s_shut_late)),
        hsm.Initial(_S("shutting")),
        hsm.State(
            "shutting", hsm.Entry(s_shut_next),
            _T(_On("shut_next"), _S("../shutting"), _G(left_to_shut), hsm.Effect(s_shut_one)),
            _T(_On("shut_next"), _S("../draining")),
        ),
        hsm.State(
            "draining", hsm.Entry(s_wait_joins),
            _T(_On("joins_check"), _S("../finishing"), _G(all_joins_in), hsm.Effect(s_shut_root)),
            _T(_On("graft_received"), _S("../finishing"), _G(last_join_in), hsm.Effect(s_shut_root)),
        ),
        hsm.State("finishing"),
        hsm.State(
            "stopping", hsm.Activity(s_stop_machines),  # every machine but this one, then closed
            _T(_On("stopped"), _S("../../closed"), _G(s_stops_clean), hsm.Effect(s_closed)),
            _T(_On("stopped"), _S("../telling"), hsm.Effect(s_take_stop_errors)),
            _T(_On(_E), _S("../telling"), hsm.Effect(s_stop_crashed)),
        ),
        hsm.State(
            "telling", io(done="told", work=_s_tell),  # a machine that could not be stopped: reported
            _T(_On("told"), _S("../../closed"), hsm.Effect(s_told)),
            _T(_On(_E), _S("../../closed"), hsm.Effect(s_tell_crashed)),
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
        then sets off fails as an activity, handled by the graph's `hsm/error` transitions."""
        hsm.Dispatch(None, self._target(tl=tl, name=name), hsm.Event(name=name, data=data))

    async def ask(self, tl: str, name: str, build: Callable[[asyncio.Future], Any]) -> Any:
        reply = self.future()
        await self.send(tl=tl, name=name, data=build(reply))
        return await reply

    async def stop(self) -> None:
        for sm in list(self.timelines.values()):
            if sm.state():
                await hsm.Stop(sm)
        for sm in (self.session, self.scribe):
            if sm is not None and sm.state():
                await hsm.Stop(sm)
