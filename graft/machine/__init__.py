"""State machines (stateforward.hsm) for timelines, the session, and the scribe.

Split into focused modules:
- types: shared payloads, constants, Host protocol, and grouped Timeline sub-state
- plumbing: event dispatch, future helpers, and I/O activity wrappers
- scribe: fire-and-forget write serializer
- session: A2A task lifecycle, event log writer, and coordinated shutdown
- timeline_ops / timeline: original, focus graft, and background graft HSMs
- registry: `Machines` coordinator
"""

from .plumbing import described
from .registry import Machines
from .scribe import SCRIBE, Scribe
from .session import SESSION, Session
from .timeline import BACKGROUND, TIMELINE
from .timeline_ops import (
    Timeline,
    _cancel_rows,
    answer_pending,
    cancel_refusal,
    branch_refusal,
    branch_refusal_not_written,
    branch_why,
    decline,
    decline_why,
    drop_branch,
    preparation_crashed,
    refuse_cancel_call,
    refuse_branch_call,
)
from .types import (
    CANCELLED,
    CLOSED_TEXT,
    MAX_JOIN_ATTEMPTS,
    MAX_STOP_ATTEMPTS,
    MAX_WRITE_ATTEMPTS,
    TURN,
    Aborted,
    Cancel,
    CancelCall,
    Branch,
    GraftPlan,
    GraftState,
    Host,
    JoinState,
    Mode,
    Prepared,
    QueuedJoin,
    Request,
    RequestState,
    Rested,
    RunOutcome,
    StepState,
    Submission,
    TaskRecord,
    ToolCall,
    TransitionRefused,
    Unwritten,
    Waiter,
)

__all__ = [
    "BACKGROUND", "CANCELLED", "CLOSED_TEXT", "MAX_JOIN_ATTEMPTS", "MAX_STOP_ATTEMPTS",
    "MAX_WRITE_ATTEMPTS", "SCRIBE", "SESSION", "TIMELINE", "TURN",
    "Aborted", "Cancel", "CancelCall", "Branch", "GraftPlan", "GraftState", "Host",
    "JoinState", "Machines", "Mode", "Prepared", "QueuedJoin", "Request", "RequestState",
    "Rested", "RunOutcome", "Scribe", "Session", "StepState", "Submission", "TaskRecord",
    "Timeline", "ToolCall", "TransitionRefused", "Unwritten", "Waiter", "described",
]
