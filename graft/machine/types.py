"""Shared data types, constants, protocols, and grouped sub-state for graft's state machines."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

CANCELLED = "cancelled"  # the join reason of a cancelled graft: set by the harness only
MAX_JOIN_ATTEMPTS = 3  # join commit retries at successive commit points before abandoning
MAX_STOP_ATTEMPTS = 3  # stops tried for one unwanted background graft before giving up
MAX_WRITE_ATTEMPTS = 3  # event-log write retries before waiting for the next session event
MAX_LOST = 64  # cap on kept scribe failures waiting for a consumer

HEAD = 0
NONE_YET = 0
STATE_HEAD = 2
EXITS = (KeyboardInterrupt, SystemExit)

RETURNED = "message_returned"
MESSAGES = ("user_message", RETURNED)
REQUESTS = ("turn_request", "drain_request", "join", "user_message", "driven_done")

CLOSED_TEXT = "the agent was closed before this request finished"
a2a_SUBMITTED = "TASK_STATE_SUBMITTED"
a2a_WORKING = "TASK_STATE_WORKING"
a2a_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
a2a_COMPLETED = "TASK_STATE_COMPLETED"
a2a_FAILED = "TASK_STATE_FAILED"
a2a_CANCELED = "TASK_STATE_CANCELED"


class TransitionRefused(Exception):
    pass


class Unwritten(str):
    """A branch call's refusal by a timeline that is not running (its audit and result row are the
    caller's to write: no step of this machine wrote them)."""


class Aborted(Exception):
    """An activity raised a non-Exception BaseException (not CancelledError/EXITS): wrapped as an
    Exception so hsm dispatches `hsm/error` and the state handles it."""


class QueuedJoin(NamedTuple):
    """A join event waiting on its source: who, why, and the user message it hands back."""
    child: str
    reason: str
    trigger: int | None
    closing: dict[str, tuple[str, bool]] | None = None
    replied_cleanly: bool = False  # focus graft ended via plain reply (no explicit merge(), not failed/withdrawn)


class Mode(NamedTuple):
    """How a run of the model loop goes."""
    name: str
    answers: bool = False
    user: bool = False
    about: tuple[str, ...] = ()


TURN = Mode(name="turn", user=True)


class RunOutcome(NamedTuple):
    """How a run of the model loop ended: replied, empty, join, or exhausted."""
    kind: str
    reason: str | None = None


class Prepared(NamedTuple):
    """A branch call the harness has seeded and validated."""
    pairs: list[tuple[str, bool]]
    data: Any
    names: Any = None
    wake: Any = None
    can_fork: Any = None


@dataclass(eq=False)
class Request:
    """One user's message (a host's SendMessage): owned by the original's machine until answered
    or withdrawn, and until its task is terminal (`call_done`)."""
    id: int
    text: str
    reply: asyncio.Future
    row: int | None = None
    task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GraftPlan(NamedTuple):
    """A graft of a committed branch call, to be made in `spawning`."""
    child: str
    focus: bool
    waiters: list["Waiter"]
    answering: "Request | None"
    made_for: "tuple[Request, ...]"
    can_fork: bool = False


@dataclass
class Waiter:
    """A caller waiting for a timeline to come to rest."""
    reply: asyncio.Future
    user: bool
    error: BaseException | None = None
    request: Request | None = None


@dataclass
class Branch:
    """A branch call put to the original's machine as the model made it."""
    use: dict
    reply: asyncio.Future
    store: bool = False
    cancels: "Cancel | None" = None
    pairs: list[tuple[str, bool]] = field(default_factory=list)
    data: Any = None
    names: dict[str, str] = field(default_factory=dict)
    wake: dict[str, bool] = field(default_factory=dict)
    can_fork: dict[str, bool] = field(default_factory=dict)


@dataclass
class CancelCall:
    use: dict
    ids: list[str]
    problems: list[str]


@dataclass
class Cancel:
    calls: list[CancelCall]
    reply: asyncio.Future


@dataclass
class ToolCall:
    """A host tool call its run asks its machine to make (activity of `calling`)."""
    use: dict
    reply: asyncio.Future
    interrupted: str
    cancelled: str = "interrupted by the tool: it may have acted"
    failed: str = "failed after it was invoked: it may have acted"


@dataclass
class Submission:
    """A host's message for the session (`submit`)."""
    task: str
    follow_up: bool
    message: dict
    request: Request


@dataclass
class TaskRecord:
    """The session's record of an open task."""
    state: str
    requests: dict[int, str]
    failed: bool = False
    reply: dict | None = None


class Rested(NamedTuple):
    """A timeline that holds the conversation came to rest."""
    tl: str
    answered: tuple[tuple[int, str | None], ...]
    owed: frozenset[str] | None


# ------------------------------------------------------------------ grouped Timeline sub-state
@dataclass
class RequestState:
    """User requests, waiters, and withdrawal tracking for a timeline."""
    inbox: list[Request] = field(default_factory=list)
    request: Request | None = None
    waiters: list[Waiter] = field(default_factory=list)
    answering: Request | None = None
    answered: bool = False
    withdrawn: set[int] = field(default_factory=set)
    returned: list[Request] = field(default_factory=list)
    woken_for: list[Request] = field(default_factory=list)
    unmarked: list[int] = field(default_factory=list)


@dataclass
class GraftState:
    """Child grafts owned by this timeline (original only)."""
    live: dict[str, bool] = field(default_factory=dict)  # child -> focus, until join applied
    made: dict[str, bool] = field(default_factory=dict)  # every child ever made -> focus
    wake: dict[str, bool] = field(default_factory=dict)  # child -> whether main wants to wake after clean reply
    requests: dict[str, tuple[Request, ...]] = field(default_factory=dict)  # bg child -> owning requests
    focus: str | None = None
    stopped_unwanted: set[str] = field(default_factory=set)
    stop_attempts: dict[str, int] = field(default_factory=dict)
    stops_told: set[str] = field(default_factory=set)


@dataclass
class JoinState:
    """Join queue (source) and join lifecycle (graft)."""
    queue: list[QueuedJoin] = field(default_factory=list)
    hold: bool = False
    attempts: dict[str, int] = field(default_factory=dict)
    news: list[tuple[str, str]] = field(default_factory=list)
    needs_wake: bool = False  # source: whether any join in this drain requires waking main
    ends_on_request: bool = False
    join_reason: str = "done"  # focus graft: decided on entry of handing_back
    replied_cleanly: bool = False  # focus graft: ended via plain reply (no explicit merge())
    quiet_handback: bool = False  # focus graft: handing back quietly after a clean child reply with wake=False
    reason: str = ""  # background graft: join reason while recording
    close_first: bool = False  # background graft: close open calls before recording join
    forced: asyncio.Future | None = None
    watchers: list[asyncio.Future] = field(default_factory=list)
    unclosed: dict[str, tuple[str, bool]] | None = None


@dataclass
class StepState:
    """In-flight step state: branch, cancel, host tool call, and deferred store close-out."""
    branch: Branch | None = None
    branch_text: str = ""
    spawn_plan: list[GraftPlan] = field(default_factory=list)
    rejection: str = ""
    handing: Branch | None = None
    spawned_ids: list[str] = field(default_factory=list)
    spawn_error: BaseException | None = None
    cancel: Cancel | None = None
    to_check: list[str] = field(default_factory=list)
    pending: set[str] = field(default_factory=set)
    outcomes: dict[str, str] = field(default_factory=dict)
    claims: dict[str, int] = field(default_factory=dict)
    claims_next: list[tuple[int, str]] = field(default_factory=list)
    tool: ToolCall | None = None
    unstored: dict[str, tuple[str, bool]] = field(default_factory=dict)
    unreported: list[tuple[str, list[str], BaseException, str | None]] = field(default_factory=list)
    reported: BaseException | None = None
    close_out: list[dict[str, Any]] = field(default_factory=list)


class Host(Protocol):
    """What the machines need from the harness: I/O only, always called from an activity."""
    def run_busy(self, tl: str, kind: str, woken: list[tuple[str, str]] | None = None,
                 tasks: tuple[str, ...] = ()) -> Any: ...
    def run_background(self, tl: str) -> Any: ...
    def mirror(self, tl: str, status: str) -> None: ...
    def report(self, kind: str, src: str | None, branches: list[str], error: BaseException, quiet: bool,
               background: bool, fails: tuple[str, ...] = ()) -> tuple[str | None, tuple[str, ...]]: ...
    def flush(self) -> None: ...
    def append_events(self, events: list[tuple[int, dict]]) -> int: ...
    def trace(self, text: str) -> None: ...
    def trace_line(self, text: str) -> None: ...
    def graft_failed(self, tl: str, error: BaseException) -> None: ...
    def begin_request(self, tl: str, text: str) -> int: ...
    def mark_withdrawn(self, timelines: list[str], rows: list[int]) -> None: ...
    async def prepare_branch(self, src: str, branch: Branch) -> "Prepared | str": ...
    def roll_back_branch(self, src: str) -> None: ...
    def commit_branch(self, src: str, branch: Branch) -> str: ...
    def refuse_branch(self, src: str, branch: Branch, why: str, error: BaseException | None) -> str: ...
    def store_result(self, tl: str, use: dict, content: str, is_error: bool) -> None: ...
    async def call_tool(self, tl: str, use: dict) -> tuple[str, bool]: ...
    def close_tool_uses(self, tl: str, known: dict[str, tuple[str, bool]]) -> None: ...
    def record_join(self, child: str, reason: str) -> None: ...
    def apply_join(self, src: str, join: QueuedJoin, running: list[str]) -> tuple[bool, bool]: ...
    def cancel_text(self, call: CancelCall, outcomes: dict[str, str]) -> tuple[str, bool]: ...
    def join_of(self, tl: str, reason: str, focus: bool = False) -> QueuedJoin: ...
    def abandon_branch(self, src: str, grafts: list[str]) -> None: ...
    def recover(self, tl: str, known: dict[str, tuple[str, bool]], close_calls: bool, viewers: list[str],
                rows: list[int], cancels: list[tuple[CancelCall, dict[str, str]]]) -> None: ...
