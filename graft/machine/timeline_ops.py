"""Timeline instance, domain rules, guards, sync I/O helpers, and transition effects."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import hsm

from .plumbing import _fail, _forward, _note, _resolve, _self, _session, described
from .types import (CANCELLED, HEAD, MAX_JOIN_ATTEMPTS, MAX_STOP_ATTEMPTS, NONE_YET, RETURNED,
                    TURN, Cancel, Branch, GraftPlan, GraftState, JoinState, Mode, QueuedJoin,
                    Request, RequestState, Rested, RunOutcome, StepState, ToolCall,
                    TransitionRefused, Unwritten, Waiter)


class Timeline(hsm.Instance):
    """State machine instance for the original timeline, a focus graft, or a background graft."""

    def __init__(self, machines, timeline_id: str, source_id: str | None, kind: str,
                 held: bool = False, waiters: list[Waiter] | None = None,
                 answering: Request | None = None, made_for: tuple[Request, ...] = (),
                 withdrawn: set[int] | None = None, can_fork: bool = False):
        super().__init__()
        self.machines = machines
        self.host = machines.host
        self.timeline_id = timeline_id
        self.source_id = source_id
        self.kind = kind  # original | focus | background
        self.can_fork = can_fork
        self.held = held
        self.mode = TURN
        self.made_for: tuple[Request, ...] = tuple(made_for)
        self.closing = False
        self.final = False
        self.closed = False
        self.ended = ""

        self.req = RequestState(
            waiters=list(waiters or []),
            answering=answering,
            withdrawn=set(withdrawn or ()),
        )
        self.grafts = GraftState()
        self.joins = JoinState()
        self.step = StepState()


# ------------------------------------------------------------------ domain rules & helpers
def _background(inst: Timeline, branches: list[str]) -> bool:
    return all(inst.grafts.made.get(c) is False or (c == inst.timeline_id and inst.kind == "background")
               for c in branches)


def _report(inst: Timeline, kind: str, branches: list[str], error: BaseException, quiet: bool = False,
            src: str | None = None, fails: tuple[str, ...] = ()) -> None:
    _note(inst=inst, method="report", kind=kind, src=src or inst.timeline_id, branches=branches, error=error,
          quiet=quiet, background=_background(inst, branches), fails=fails)


def branch_refusal(inst: Timeline, grafts: list[tuple[str, bool]]) -> str | None:
    if not grafts:
        return "a fork call needs at least one branch"
    focus = [g for g, f in grafts if f]
    if len(focus) > 1:
        return f"at most one branch may carry the conversation, got {focus}"
    if inst.kind == "background" and focus:
        return "a detached branch can only fork detached branches"
    return None


def decline_why(inst: Timeline, branch: Branch | None = None) -> str | None:
    if inst.closing:
        return "the harness is closing; answer the user instead"
    if _run_withdrawn(inst):
        return "the user's call this turn answers was withdrawn"
    if not inst.can_fork:
        return "you fork only from main (or a branch with child skills)"
    return None


def branch_why(inst: Timeline, branch: Branch) -> str | None:
    if any(g in inst.grafts.made for g, _ in branch.pairs):
        return f"branch ids must be new: {[branch.names.get(g, g) for g, _ in branch.pairs]}"
    rule = branch_refusal(inst, branch.pairs)
    if rule is not None:
        return f"forking is not possible now: {rule}"
    return None


def may_cancel(inst: Timeline, child: str) -> bool:
    return (inst.can_fork and inst.grafts.live.get(child) is False
            and all(q.child != child for q in inst.joins.queue))


def cancel_refusal(inst: Timeline, child: str) -> str:
    if not inst.can_fork:
        return "you stop branches only from main; a branch never stops another branch"
    if child == inst.timeline_id:
        return "main cannot be stopped" if inst.source_id is None else "this branch cannot stop itself"
    if child not in inst.grafts.made:
        return "no such branch"
    if inst.grafts.made[child]:
        return "it was forked as the current branch; only a detached branch can be stopped"
    return "it has already finished or was already stopped; nothing to stop"


def _live(inst: Timeline, request: Request) -> bool:
    return request.id not in inst.req.withdrawn and not request.reply.done()


def _withdrawn(inst: Timeline, request: Request) -> bool:
    return request.id in inst.req.withdrawn or request.reply.cancelled()


def _run_withdrawn(inst: Timeline, withdrawing: int | None = None) -> bool:
    request = inst.req.answering
    if request is not None:
        return request.id == withdrawing or not _live(inst, request)
    woken = inst.req.woken_for
    return bool(woken) and all(r.id == withdrawing or _withdrawn(inst, r) for r in woken)


def _held(inst: Timeline) -> list[Request]:
    held = list(inst.req.inbox) + [w.request for w in inst.req.waiters if w.request is not None]
    return held + [r for r in (inst.req.answering, inst.req.request) if r is not None]


def _answered(inst: Timeline) -> list[Request]:
    answered: list[Request] = []
    for r in [*inst.req.returned, *(r for rs in inst.grafts.requests.values() for r in rs), *inst.req.woken_for]:
        if all(r is not a for a in answered):
            answered.append(r)
    return answered


def _prune(inst: Timeline) -> None:
    known_ids = {r.id for r in _held(inst) + _answered(inst)}
    inst.req.withdrawn &= known_ids
    live = set(inst.grafts.live)
    inst.grafts.stopped_unwanted &= live
    inst.grafts.stop_attempts = {g: n for g, n in inst.grafts.stop_attempts.items() if g in live}
    inst.grafts.stops_told &= live


def _unwanted(inst: Timeline, child: str) -> bool:
    return any(_withdrawn(inst, r) for r in inst.grafts.requests.get(child, ()))


def _owners(inst: Timeline) -> tuple[Request, ...]:
    runs_for: list[Request] = []
    sources = ([inst.req.answering] if inst.req.answering is not None else []) + list(inst.req.woken_for) + list(inst.made_for)
    for r in sources:
        if all(r is not o for o in runs_for):
            runs_for.append(r)
    live = [r for r in runs_for if not _withdrawn(inst, r)]
    return tuple(live or runs_for)


def _tasks_of(requests) -> tuple[str, ...]:
    out: list[str] = []
    for r in requests:
        if r.task is not None and r.task not in out:
            out.append(r.task)
    return tuple(out)


def _owed(inst: Timeline) -> frozenset[str] | None:
    if inst.kind != "original":
        return None
    return frozenset(t for rs in inst.grafts.requests.values() for t in _tasks_of(rs))


def _made_for_withdrawn(inst: Timeline) -> bool:
    return any(_withdrawn(inst, r) for r in inst.made_for)


def _still_running(inst: Timeline) -> list[str]:
    queued = {q.child for q in inst.joins.queue}
    return [g for g, focus in inst.grafts.live.items() if not focus and g not in queued]


def _viewers(inst: Timeline) -> list[str]:
    out = [inst.timeline_id]
    cur: Timeline | None = inst
    while cur is not None and cur.kind == "focus" and cur.source_id is not None:
        out.append(cur.source_id)
        cur = inst.machines.timelines.get(cur.source_id)
    return out


def _withdrawn_rows(inst: Timeline) -> list[int]:
    request = inst.req.answering
    withdrawn = request is not None and _withdrawn(inst, request)
    return [request.row] if withdrawn and request.row is not None else []


def _answers_requests(inst: Timeline) -> bool:
    return inst.mode.user or inst.mode.answers


def _handed_back(inst: Timeline) -> bool:
    return any(r != "done" and inst.grafts.made.get(c) for c, r in inst.joins.news)


def _marks_after(inst: Timeline) -> bool:
    return _handed_back(inst) and bool(_withdrawn_rows(inst))


def _inbox_live(inst: Timeline) -> bool:
    return any(_live(inst, r) for r in inst.req.inbox)


def _late(inst: Timeline, rid: int) -> list[Request]:
    held = {r.id for r in _held(inst)}
    return [r for r in _answered(inst) if r.id == rid and rid not in held and rid not in inst.req.withdrawn
            and not r.reply.cancelled()]


def _focus_withdrawn(inst: Timeline) -> bool:
    if inst.kind != "focus" or inst.closing or inst.final:
        return False
    answering = inst.req.answering is not None and _withdrawn(inst, inst.req.answering)
    return answering or any(_withdrawn(inst, r) for r in inst.req.returned)


def _unstopped(inst: Timeline) -> list[str]:
    return [g for g in _still_running(inst) if _unwanted(inst, g) and g not in inst.grafts.stopped_unwanted]


def _stop_target(inst: Timeline) -> str | None:
    return next((g for g in _unstopped(inst) if inst.grafts.stop_attempts.get(g, 0) < MAX_STOP_ATTEMPTS), None)


def _returned(inst: Timeline, requests: list) -> None:
    inst.req.returned += [r for r in requests if isinstance(r, Request) and r.reply.done()
                          and not r.reply.cancelled() and all(r is not k for k in inst.req.returned)]


# ------------------------------------------------------------------ deferred store close-out
def _mark_pending(inst: Timeline) -> None:
    if inst.req.unmarked:
        inst.host.mark_withdrawn(timelines=_viewers(inst), rows=list(inst.req.unmarked))


def _close_out(inst: Timeline) -> None:
    for args in inst.step.close_out:
        inst.host.recover(**args)


def _close_out_of(inst: Timeline) -> dict[str, Any]:
    return dict(tl=inst.timeline_id, known=dict(inst.step.unstored), close_calls=inst.mode.name != "external",
                viewers=_viewers(inst), rows=list(inst.req.unmarked), cancels=[])


def closed_out(inst: Timeline) -> None:
    inst.step.close_out = []


# ------------------------------------------------------------------ guards
def has_queue(ctx, inst: Timeline, event) -> bool:
    return bool(inst.joins.queue)


def is_external(ctx, inst: Timeline, event) -> bool:
    return event.data[0].name == "external"


def ends_run(ctx, inst: Timeline, event) -> bool:
    return inst.req.answering is not None and _run_withdrawn(inst, withdrawing=event.data)


def ends_wake(ctx, inst: Timeline, event) -> bool:
    return inst.req.answering is None and withdraws_late(ctx, inst, event) and _run_withdrawn(inst, withdrawing=event.data)


def withdraws_late(ctx, inst: Timeline, event) -> bool:
    return bool(_late(inst, event.data))


def withdraws_late_only(ctx, inst: Timeline, event) -> bool:
    return withdraws_late(ctx, inst, event) and not ends_wake(ctx, inst, event)


def withdraws_own_focus(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus" and not (inst.closing or inst.final) and withdraws_late(ctx, inst, event)


def withdraws_late_at_rest(ctx, inst: Timeline, event) -> bool:
    return withdraws_late(ctx, inst, event) and not withdraws_own_focus(ctx, inst, event)


def withdraws_waiting(ctx, inst: Timeline, event) -> bool:
    return any(r.id == event.data for r in inst.req.inbox)


def arrives_withdrawn(ctx, inst: Timeline, event) -> bool:
    return not _live(inst, event.data)


def arrives_behind_joins(ctx, inst: Timeline, event) -> bool:
    return _live(inst, event.data) and bool(inst.joins.queue)


def request_live(ctx, inst: Timeline, event) -> bool:
    return _live(inst, inst.req.request)


def nothing_to_mark(ctx, inst: Timeline, event) -> bool:
    return not (_answers_requests(inst) and bool(_withdrawn_rows(inst)))


def focus_cancelled(ctx, inst: Timeline, event) -> bool:
    return _focus_withdrawn(inst)


def withdrawn_after(ctx, inst: Timeline, event) -> bool:
    return not inst.final and _marks_after(inst)


def handed_live(ctx, inst: Timeline, event) -> bool:
    inbox, withdrawn = event.data
    return any(r.id not in withdrawn and _live(inst, r) for r in inbox)


def handed_withdraws_run(ctx, inst: Timeline, event) -> bool:
    _, withdrawn = event.data
    request = inst.req.answering
    return request is not None and (request.id in withdrawn or not _live(inst, request))


def handed_live_with_queue(ctx, inst: Timeline, event) -> bool:
    return bool(inst.joins.queue) and handed_live(ctx, inst, event)


def handed_live_to_answer(ctx, inst: Timeline, event) -> bool:
    return not inst.joins.queue and handed_live(ctx, inst, event)


def starts_busy(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus" and not inst.held and not _run_withdrawn(inst)


def starts_withdrawn(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus" and not inst.held and _run_withdrawn(inst)


def is_held(ctx, inst: Timeline, event) -> bool:
    return inst.held


def made_for_live(ctx, inst: Timeline, event) -> bool:
    return not _made_for_withdrawn(inst)


def starts_running(ctx, inst: Timeline, event) -> bool:
    return not inst.held and not _made_for_withdrawn(inst)


def own_graft(ctx, inst: Timeline, event) -> bool:
    return event.data.child in inst.grafts.live and event.data.child not in inst.step.pending


def is_focus(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus"


def carries_cancels(ctx, inst: Timeline, event) -> bool:
    return event.data.cancels is not None


def may_try(ctx, inst: Timeline, event) -> bool:
    return decline_why(inst) is None


def rejected_in_preparation(ctx, inst: Timeline, event) -> bool:
    return isinstance(event.data, str)


def graph_allows(ctx, inst: Timeline, event) -> bool:
    return not isinstance(event.data, str) and branch_why(inst, inst.step.branch) is None


def run_goes_on(ctx, inst: Timeline, event) -> bool:
    return not _run_withdrawn(inst)


def cancels_only(ctx, inst: Timeline, event) -> bool:
    return inst.step.branch is None and not _run_withdrawn(inst)


def branch_waits(ctx, inst: Timeline, event) -> bool:
    return inst.step.branch is not None and not _run_withdrawn(inst) and inst.step.reported is None


def branch_dropped(ctx, inst: Timeline, event) -> bool:
    return inst.step.branch is not None and not _run_withdrawn(inst) and inst.step.reported is not None


def hands_over(ctx, inst: Timeline, event) -> bool:
    return inst.grafts.focus is not None


def row_lost(ctx, inst: Timeline, event) -> bool:
    return (inst.grafts.focus is None and inst.step.branch is not None
            and inst.step.branch.use.get("id") in inst.step.unstored)


def has_joined(ctx, inst: Timeline, event) -> bool:
    return inst.ended == "joined"


def joins_back(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus"


def run_joined(ctx, inst: Timeline, event) -> bool:
    return event.data.kind == "join"


def run_replied(ctx, inst: Timeline, event) -> bool:
    return event.data.kind == "replied"


def head_cancellable(ctx, inst: Timeline, event) -> bool:
    return (bool(inst.step.to_check) and inst.step.to_check[0] not in inst.step.claims
            and may_cancel(inst, inst.step.to_check[0]))


def head_refused(ctx, inst: Timeline, event) -> bool:
    return bool(inst.step.to_check) and not head_cancellable(ctx, inst, event)


def checked_with_pending(ctx, inst: Timeline, event) -> bool:
    return not inst.step.to_check and bool(inst.step.pending)


def last_pending(ctx, inst: Timeline, event) -> bool:
    return inst.step.pending == {event.data.child}


def other_pending(ctx, inst: Timeline, event) -> bool:
    return event.data.child in inst.step.pending and inst.step.pending != {event.data.child}


def one_of_pending(ctx, inst: Timeline, event) -> bool:
    return event.data.child in inst.step.pending


def rows_written(ctx, inst: Timeline, event) -> bool:
    return not event.data[1]


def error_to_user(ctx, inst: Timeline, event) -> bool:
    return inst.kind != "focus" and inst.mode.user and any(w.error is None for w in inst.req.waiters)


def error_already_shown(ctx, inst: Timeline, event) -> bool:
    return inst.kind != "focus" and inst.step.reported is event.data and not error_to_user(ctx, inst, event)


def drains_now(ctx, inst: Timeline, event) -> bool:
    return bool(inst.joins.queue) and not inst.joins.hold


def closes_now(ctx, inst: Timeline, event) -> bool:
    return inst.final and not drains_now(ctx, inst, event) and not _still_running(inst)


def closes_clean(ctx, inst: Timeline, event) -> bool:
    return closes_now(ctx, inst, event) and not inst.step.close_out


def closes_out_first(ctx, inst: Timeline, event) -> bool:
    return closes_now(ctx, inst, event) and bool(inst.step.close_out)


def answers_inbox(ctx, inst: Timeline, event) -> bool:
    return _inbox_live(inst) and not inst.final and not drains_now(ctx, inst, event)


def applies_next(ctx, inst: Timeline, event) -> bool:
    return bool(inst.joins.queue) and not inst.joins.hold


def retry_left(ctx, inst: Timeline, event) -> bool:
    return inst.joins.attempts.get(inst.joins.queue[0].child, NONE_YET) + 1 < MAX_JOIN_ATTEMPTS


def is_final(ctx, inst: Timeline, event) -> bool:
    return inst.final and not _still_running(inst)


def final_clean(ctx, inst: Timeline, event) -> bool:
    return is_final(ctx, inst, event) and not inst.step.close_out


def final_out_first(ctx, inst: Timeline, event) -> bool:
    return is_final(ctx, inst, event) and bool(inst.step.close_out)


def closes_out_at_rest(ctx, inst: Timeline, event) -> bool:
    return not inst.joins.queue and bool(inst.step.close_out)


def final_waits(ctx, inst: Timeline, event) -> bool:
    return inst.final and bool(_still_running(inst))


def _after_open(inst: Timeline) -> bool:
    return not inst.final and not _marks_after(inst)


def inbox_after(ctx, inst: Timeline, event) -> bool:
    return _after_open(inst) and _inbox_live(inst)


def quiet_after(ctx, inst: Timeline, event) -> bool:
    return (_after_open(inst) and not _inbox_live(inst) and not inst.joins.ends_on_request
            and (not inst.joins.news or not inst.joins.needs_wake or inst.closing or _run_withdrawn(inst)))


def answers_handback(ctx, inst: Timeline, event) -> bool:
    return (_after_open(inst) and not _inbox_live(inst) and bool(inst.joins.news)
            and not inst.closing and not _run_withdrawn(inst) and inst.joins.ends_on_request)


def is_driven(ctx, inst: Timeline, event) -> bool:
    return inst.mode.name == "external"


def stops_due(ctx, inst: Timeline, event) -> bool:
    return _stop_target(inst) is not None


def graft_wanted(ctx, inst: Timeline, event) -> bool:
    child, _ = event.data
    return not _unwanted(inst, child)


def has_unreported(ctx, inst: Timeline, event) -> bool:
    return bool(inst.step.unreported)


# ------------------------------------------------------------------ sync I/O helpers (for io())
def _closing_out(inst: Timeline, event):
    _close_out(inst)


def _write_request(inst: Timeline, event):
    _close_out(inst)
    return inst.host.begin_request(tl=inst.timeline_id, text=inst.req.request.text)


def _mark_rows(inst: Timeline, event):
    _close_out(inst)
    inst.host.mark_withdrawn(timelines=_viewers(inst), rows=list(inst.req.unmarked))


def _unprepare(inst: Timeline, event):
    inst.host.roll_back_branch(src=inst.timeline_id)


def _commit(inst: Timeline, event):
    return inst.host.commit_branch(src=inst.timeline_id, branch=inst.step.branch)


def _branch_row(inst: Timeline, event):
    if inst.step.branch.store:
        inst.host.store_result(tl=inst.timeline_id, use=inst.step.branch.use,
                               content=inst.step.branch_text, is_error=False)


def _refuse(inst: Timeline, event):
    error = event.data if isinstance(event.data, BaseException) else None
    why = branch_why(inst, inst.step.branch) if error is None else None
    text = inst.host.refuse_branch(src=inst.timeline_id, branch=inst.step.branch,
                                  why=why or "internal error", error=error)
    if inst.step.branch.store:
        inst.host.store_result(tl=inst.timeline_id, use=inst.step.branch.use, content=text, is_error=True)
    return text


def _reject(inst: Timeline, event):
    if inst.step.branch.store:
        inst.host.store_result(tl=inst.timeline_id, use=inst.step.branch.use,
                               content=inst.step.rejection, is_error=True)
    return inst.step.rejection


def _join_back(inst: Timeline, event):
    return inst.host.join_of(tl=inst.timeline_id, reason=inst.joins.join_reason, focus=True)


def _cancel_rows(inst: Timeline, event):
    results, failures = [], []
    again = "it has already finished or was already stopped; nothing to stop"
    for index, call in enumerate(inst.step.cancel.calls):
        outcomes = {c: inst.step.outcomes[c] if inst.step.claims.get(c) == index else again for c in call.ids}
        content, is_error = inst.host.cancel_text(call=call, outcomes=outcomes)
        try:
            inst.host.store_result(tl=inst.timeline_id, use=call.use, content=content, is_error=is_error)
        except Exception as e:  # noqa: BLE001
            stopped = [c for c in call.ids if outcomes.get(c) == CANCELLED]
            failures.append((call.use["id"], content, is_error, stopped, e))
        results.append((content, is_error))
    return results, failures


def _close_calls(inst: Timeline, event):
    inst.host.close_tool_uses(tl=inst.timeline_id, known=dict(inst.step.unstored))


def _focus_failure(inst: Timeline, event):
    inst.host.close_tool_uses(tl=inst.timeline_id, known=dict(inst.step.unstored))
    inst.host.graft_failed(tl=inst.timeline_id, error=event.data)
    return inst.host.join_of(tl=inst.timeline_id, reason="failed")


def _apply(inst: Timeline, event):
    _close_out(inst)
    _mark_pending(inst)
    return inst.host.apply_join(src=inst.timeline_id, join=inst.joins.queue[0], running=_still_running(inst))


def _record(inst: Timeline, event):
    errors, closed = [], True
    if inst.joins.close_first:
        try:
            inst.host.close_tool_uses(tl=inst.timeline_id, known=dict(inst.step.unstored))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            closed = False
    try:
        inst.host.record_join(child=inst.timeline_id, reason=inst.joins.reason)
    except Exception as e:  # noqa: BLE001
        errors.append(e)
    return errors, closed


# ------------------------------------------------------------------ effects: user requests & lifecycle
def _answer_dead(requests: list[Request]) -> None:
    for r in requests:
        _resolve(r.reply, "ended")


def _drop_dead(inst: Timeline) -> None:
    dead = [r for r in inst.req.inbox if not _live(inst, r)]
    inst.req.inbox = [r for r in inst.req.inbox if all(r is not d for d in dead)]
    _answer_dead(dead)


def take_message(ctx, inst: Timeline, event) -> None:
    inst.req.request = event.data


def drop_message(ctx, inst: Timeline, event) -> None:
    _answer_dead([event.data])


def take_inbox(ctx, inst: Timeline, event) -> None:
    _drop_dead(inst)
    inst.req.request = inst.req.inbox.pop(HEAD)


def request_written(ctx, inst: Timeline, event) -> None:
    closed_out(inst)
    request, inst.req.request = inst.req.request, None
    request.row = event.data
    inst.req.answering, inst.req.answered = request, False
    inst.mode = TURN
    inst.req.waiters.append(Waiter(reply=request.reply, user=True, request=request))
    _session(inst, "answering", (inst.timeline_id, request.task))


def request_dropped(ctx, inst: Timeline, event) -> None:
    closed_out(inst)
    request, inst.req.request = inst.req.request, None
    inst.req.unmarked.append(event.data)
    _prune(inst)
    _answer_dead([request])


def request_failed(ctx, inst: Timeline, event) -> None:
    request, inst.req.request = inst.req.request, None
    inst.req.waiters.append(Waiter(reply=request.reply, user=True, error=event.data, request=request))


def stash_message(ctx, inst: Timeline, event) -> None:
    inst.req.inbox.append(event.data)


def note_withdrawn(ctx, inst: Timeline, event) -> None:
    if any(r.id == event.data for r in _held(inst)):
        inst.req.withdrawn.add(event.data)
    _drop_dead(inst)
    _prune(inst)


def _reply_error(reply: asyncio.Future) -> BaseException | None:
    return reply.exception() if reply.done() and not reply.cancelled() else None


def note_late_withdrawal(ctx, inst: Timeline, event) -> None:
    request, = _late(inst, event.data)
    inst.req.withdrawn.add(request.id)
    _note(inst, "mark_withdrawn", keep=_reply_error(request.reply), timelines=_viewers(inst),
          rows=[request.row] if request.row is not None else [])


def tick_stop(ctx, inst: Timeline, event) -> None:
    _self(inst, "stop_tick")


def stop_one(ctx, inst: Timeline, event) -> None:
    child = _stop_target(inst)
    if child is None:
        return
    inst.grafts.stop_attempts[child] = inst.grafts.stop_attempts.get(child, 0) + 1
    inst.grafts.stopped_unwanted.add(child)
    _forward(inst.machines.timelines.get(child), "cancel")


def tell_given_up(ctx, inst: Timeline, event) -> None:
    given = [g for g in _unstopped(inst) if g not in inst.grafts.stops_told
             and inst.grafts.stop_attempts.get(g, 0) >= MAX_STOP_ATTEMPTS]
    if given:
        inst.grafts.stops_told |= set(given)
        _report(inst, "task", given, RuntimeError(
            f"{', '.join(given)} could not be stopped after {MAX_STOP_ATTEMPTS} attempts; "
            f"{'it stops' if len(given) == 1 else 'they stop'} at its next step"))


def call_finished(ctx, inst: Timeline, event) -> None:
    inst.req.returned = [r for r in inst.req.returned if r.id != event.data]
    _prune(inst)


def call_finished_and_forward(ctx, inst: Timeline, event) -> None:
    call_finished(ctx, inst, event)
    _forward(inst.machines.timelines.get(inst.grafts.focus), "call_done", event.data)


def let_graft_go(ctx, inst: Timeline, event) -> None:
    _, reply = event.data
    _resolve(reply, True)


def stop_graft(ctx, inst: Timeline, event) -> None:
    child, _ = event.data
    _forward(inst.machines.timelines.get(child), "cancel")


def ask_source(ctx, inst: Timeline, event) -> None:
    _forward(inst.machines.timelines.get(inst.source_id), "graft_step", (inst.timeline_id, event.data))


def withdraw_and_forward(ctx, inst: Timeline, event) -> None:
    if any(r.id == event.data for r in _held(inst)):
        inst.req.withdrawn.add(event.data)
    if inst.grafts.focus:
        _forward(inst.machines.timelines.get(inst.grafts.focus), "withdraw", event.data)


def take_handed(ctx, inst: Timeline, event) -> None:
    inbox, withdrawn = event.data
    inst.req.inbox = list(inbox) + inst.req.inbox
    inst.req.withdrawn |= withdrawn
    _prune(inst)


def take_handed_and_begin(ctx, inst: Timeline, event) -> None:
    take_handed(ctx, inst, event)
    _drop_dead(inst)
    inst.req.request = inst.req.inbox.pop(HEAD)


def begin_run(ctx, inst: Timeline, event) -> None:
    mode, reply = event.data
    inst.mode, inst.req.answering = mode, None
    inst.req.waiters.append(Waiter(reply=reply, user=False))


def begin_driven(ctx, inst: Timeline, event) -> None:
    mode, reply = event.data
    inst.mode, inst.req.answering = mode, None
    _resolve(reply, "started")


def add_run_waiter(ctx, inst: Timeline, event) -> None:
    inst.req.waiters.append(Waiter(reply=event.data, user=False))


def answer_ended(ctx, inst: Timeline, event) -> None:
    _resolve(event.data, "ended")


def answer_handed_off(ctx, inst: Timeline, event) -> None:
    _resolve(event.data, "handed_off")


def answer_joined(ctx, inst: Timeline, event) -> None:
    _resolve(event.data, "joined")


def answer_true(ctx, inst: Timeline, event) -> None:
    _resolve(event.data, True)


def ignore(ctx, inst: Timeline, event) -> None:
    return None


def refuse_branch_call(ctx, inst: Timeline, event) -> None:
    branch: Branch = event.data
    why = Unwritten(f"fork refused; no branch was started: {inst.timeline_id} is not running a turn that may fork")
    _resolve(branch.reply, (None, None, why))
    if branch.cancels is not None:
        _resolve(branch.cancels.reply, "the fork call they came with was not run")


def refuse_cancel_call(ctx, inst: Timeline, event) -> None:
    cancel: Cancel = event.data
    why = ("you stop branches only from main; a branch never stops another branch" if not inst.can_fork
           else "you can stop branches only during your own turn, while you hold the conversation")
    _resolve(cancel.reply, why)


# ------------------------------------------------------------------ effects: end of run & tool calls
def end_run(ctx, inst: Timeline, event) -> None:
    closed_out(inst)
    inst.step.reported, inst.req.unmarked = None, []
    inst.req.answered = inst.req.answered or _answers_requests(inst)


def error_for_user(ctx, inst: Timeline, event) -> None:
    [w for w in inst.req.waiters if w.error is None][-1].error = event.data
    inst.step.reported = None


def end_driven(ctx, inst: Timeline, event) -> None:
    _resolve(event.data, "ended")


def unprepare_failed(ctx, inst: Timeline, event) -> None:
    inst.step.close_out = [*inst.step.close_out, _close_out_of(inst)]
    _report(inst, "task", [], event.data)


def error_shown(ctx, inst: Timeline, event) -> None:
    inst.step.reported = None


def error_reported(ctx, inst: Timeline, event) -> None:
    _report(inst, "wake", list(inst.mode.about), event.data)
    inst.step.reported = None


def set_closing(ctx, inst: Timeline, event) -> None:
    inst.closing = True


def set_final(ctx, inst: Timeline, event) -> None:
    inst.closing = inst.final = True


def run_failed_after_it_ended(ctx, inst: Timeline, event) -> None:
    _report(inst, "task", [], event.data)


def rows_marked(ctx, inst: Timeline, event) -> None:
    inst.req.unmarked = []
    closed_out(inst)


def to_mark(ctx, inst: Timeline, event) -> None:
    inst.req.unmarked += [r for r in _withdrawn_rows(inst) if r not in inst.req.unmarked]


def mark_failed(ctx, inst: Timeline, event) -> None:
    _report(inst, "task", [], event.data)


def calls_closed(ctx, inst: Timeline, event) -> None:
    inst.step.unstored = {}


def take_tool(ctx, inst: Timeline, event) -> None:
    inst.step.tool = event.data


def tool_result_kept(ctx, inst: Timeline, event) -> None:
    inst.step.unstored[inst.step.tool.use["id"]] = event.data


def _trace_tool(inst: Timeline, call: ToolCall, content: str) -> None:
    _note(inst, "trace_line", text=f"[{inst.timeline_id}] tool {call.use['name']}"
          f"({json.dumps(call.use['input'])[:120]}) -> {content[:80]}")


def tool_row_written(ctx, inst: Timeline, event) -> None:
    call, inst.step.tool = inst.step.tool, None
    content, _ = inst.step.unstored.pop(call.use["id"])
    _trace_tool(inst, call, content)
    _resolve(call.reply)


def tool_row_failed(ctx, inst: Timeline, event) -> None:
    call, inst.step.tool = inst.step.tool, None
    content, _ = inst.step.unstored[call.use["id"]]
    _trace_tool(inst, call, content)
    _fail(call.reply, event.data)


def tool_row_failed_late(ctx, inst: Timeline, event) -> None:
    _report(inst, "task", [inst.timeline_id], event.data, src=inst.source_id)


def tool_raised(ctx, inst: Timeline, event) -> None:
    inst.step.unstored[inst.step.tool.use["id"]] = (inst.step.tool.failed, True)


def keep_unstored(ctx, inst: Timeline, event) -> None:
    use, content, is_error = event.data
    inst.step.unstored[use["id"]] = (content, is_error)


def calls_not_closed(ctx, inst: Timeline, event) -> None:
    inst.step.close_out = [*inst.step.close_out, _close_out_of(inst)]
    _report(inst, "task", [], event.data)


def close_out_written(ctx, inst: Timeline, event) -> None:
    closed_out(inst)


def close_out_failed(ctx, inst: Timeline, event) -> None:
    _report(inst, "task", [], event.data)


def _answer_all(inst: Timeline) -> None:
    for w in inst.req.waiters:
        if w.error is not None:
            _fail(w.reply, w.error)
        else:
            _resolve(w.reply, "ended")
    for r in inst.req.inbox:
        _fail(r.reply, TransitionRefused("the harness was closed"))
    inst.req.waiters, inst.req.inbox, inst.req.answering = [], [], None
    _prune(inst)


def come_to_rest(ctx, inst: Timeline, event) -> None:
    waiters = list(inst.req.waiters)
    for w in waiters:
        if w.error is not None:
            _fail(w.reply, w.error)
        else:
            _resolve(w.reply, "ended")
    _returned(inst, [w.request for w in waiters if w.request is not None and not _withdrawn(inst, w.request)])
    inst.req.waiters = []
    _drop_dead(inst)
    inst.req.answering, inst.req.woken_for = None, []
    _prune(inst)
    answered = tuple((w.request.id, None if w.error is None else described(w.error))
                     for w in waiters if w.request is not None)
    _session(inst, "rested", Rested(tl=inst.timeline_id, answered=answered, owed=_owed(inst)))


def close_root(ctx, inst: Timeline, event) -> None:
    inst.closed = True
    _answer_all(inst)
    _session(inst, "root_closed", inst.timeline_id)


def close_and_release(ctx, inst: Timeline, event) -> None:
    _forward(inst.machines.timelines.get(inst.grafts.focus), "shutdown")


def forward_to_focus(ctx, inst: Timeline, event) -> None:
    _forward(inst.machines.timelines.get(inst.grafts.focus), event.name, event.data)


def forward_to_source(ctx, inst: Timeline, event) -> None:
    _forward(inst.machines.timelines.get(inst.source_id), event.name, event.data)


def return_to_source(ctx, inst: Timeline, event) -> None:
    _forward(inst.machines.timelines.get(inst.source_id), RETURNED, event.data)


# ------------------------------------------------------------------ effects: branch
def take_branch(ctx, inst: Timeline, event) -> None:
    inst.step.branch = event.data
    inst.step.cancel = event.data.cancels


def decline(ctx, inst: Timeline, event) -> None:
    inst.step.rejection = f"fork not executed: {decline_why(inst)}"


def take_prepared(ctx, inst: Timeline, event) -> None:
    if isinstance(event.data, str):
        inst.step.rejection = event.data
    else:
        inst.step.branch.pairs = event.data.pairs
        inst.step.branch.data = event.data.data
        inst.step.branch.names = dict(event.data.names or {})
        inst.step.branch.wake = dict(event.data.wake or {})
        inst.step.branch.can_fork = dict(event.data.can_fork or {})


def preparation_crashed(ctx, inst: Timeline, event) -> None:
    inst.step.rejection = f"fork refused; no branch was started: internal error ({described(event.data)})"


def branch_refused_written(ctx, inst: Timeline, event) -> None:
    branch, inst.step.branch = inst.step.branch, None
    _resolve(branch.reply, (None, None, event.data))


def branch_refusal_not_written(ctx, inst: Timeline, event) -> None:
    branch, inst.step.branch = inst.step.branch, None
    text = f"fork refused; no branch was started ({described(event.data)})"
    if branch.store:
        inst.step.unstored[branch.use["id"]] = (text, True)
    inst.step.close_out = [*inst.step.close_out, _close_out_of(inst)]
    inst.step.reported = event.data
    _report(inst, "task", [], event.data)
    _fail(branch.reply, event.data)


def committed(ctx, inst: Timeline, event) -> None:
    inst.step.branch_text = event.data
    if inst.step.branch.store:
        inst.step.unstored[inst.step.branch.use["id"]] = (inst.step.branch_text, False)
    users = [w for w in inst.req.waiters if w.user]
    inst.step.spawn_plan = [GraftPlan(child=child, focus=f, waiters=users if f else [],
                                      answering=inst.req.answering if f else None,
                                      made_for=() if f else _owners(inst),
                                      can_fork=inst.step.branch.can_fork.get(child, False))
                            for child, f in inst.step.branch.pairs]


def spawned(ctx, inst: Timeline, event) -> None:
    branch: Branch = event.data
    owners = _owners(inst)
    for child, focus in branch.pairs:
        inst.grafts.live[child] = focus
        inst.grafts.made[child] = focus
        inst.grafts.wake[child] = branch.wake.get(child, True) if focus else True
        if not focus and owners:
            inst.grafts.requests[child] = owners
    inst.grafts.focus = next((g for g, f in branch.pairs if f), None)
    inst.step.spawn_plan = []
    background = [g for g, f in branch.pairs if not f]
    _session(inst, "grafts_made", (background, inst.mode.name == "external", {
        g: _tasks_of(inst.grafts.requests.get(g, ())) for g in background}))


def spawn_failed(ctx, inst: Timeline, event) -> None:
    inst.step.spawned_ids, inst.step.spawn_error = event.data
    inst.step.spawn_plan = []
    inst.step.unstored.pop(inst.step.branch.use["id"], None)


def unspawn_failed(ctx, inst: Timeline, event) -> None:
    _report(inst, "task", [g for g, _ in inst.step.branch.pairs], event.data)


def undone(ctx, inst: Timeline, event) -> None:
    _session(inst, "grafts_undone", [g for g, _ in inst.step.branch.pairs])


def row_recorded(ctx, inst: Timeline, event) -> None:
    inst.step.unstored.pop(inst.step.branch.use["id"], None)


def row_not_written(ctx, inst: Timeline, event) -> None:
    inst.step.unstored[inst.step.branch.use["id"]] = (inst.step.branch_text, False)
    inst.step.reported = event.data
    _report(inst, "task", [], event.data)


def branch_row_lost(ctx, inst: Timeline, event) -> None:
    branch, inst.step.branch = inst.step.branch, None
    _fail(branch.reply, inst.step.reported)


def take_hand_over(ctx, inst: Timeline, event) -> None:
    inst.step.handing, inst.step.branch = inst.step.branch, None


def drop_branch(ctx, inst: Timeline, event) -> None:
    branch, inst.step.branch = inst.step.branch, None
    _resolve(branch.reply, (None, None, "fork not executed: the stop calls beside it could not be recorded"))


def answer_pending(ctx, inst: Timeline, event) -> None:
    branch, inst.step.branch = inst.step.branch, None
    tool, inst.step.tool = inst.step.tool, None
    if tool is not None:
        tool.reply.cancel()
    cancel, inst.step.cancel = inst.step.cancel, None
    if branch is not None:
        _resolve(branch.reply, (None, None, "fork not executed: the turn ended before it ran"))
        if branch.cancels is not None and branch.cancels is not cancel:
            _resolve(branch.cancels.reply, "the turn ended before the stop ran")
    if cancel is not None:
        _resolve(cancel.reply, "the turn ended before the stop ran")


def pass_inbox(ctx, inst: Timeline, event) -> None:
    _forward(inst.machines.timelines.get(inst.grafts.focus), "inbox_handed",
             (list(inst.req.inbox), set(inst.req.withdrawn)))
    inst.req.inbox = []


def hand_over(ctx, inst: Timeline, event) -> None:
    branch, inst.step.handing = inst.step.handing, None
    for w in inst.req.waiters:
        if not w.user:
            _resolve(w.reply, "handed_off")
    inst.req.waiters, inst.req.answering = [], None
    _prune(inst)
    if branch is not None:
        _resolve(branch.reply, ([g for g, _ in branch.pairs], inst.grafts.focus, inst.step.branch_text))


def branch_done(ctx, inst: Timeline, event) -> None:
    branch, inst.step.branch = inst.step.branch, None
    _resolve(branch.reply, ([g for g, _ in branch.pairs], None, inst.step.branch_text))


# ------------------------------------------------------------------ effects: cancel
def take_cancel(ctx, inst: Timeline, event) -> None:
    inst.step.cancel = event.data
    inst.step.branch = None


def start_checking(ctx, inst: Timeline, event) -> None:
    _self(inst, "check")


def checks_from(ctx, inst: Timeline, event) -> None:
    cancel = inst.step.cancel
    inst.step.pending, inst.step.outcomes, inst.step.claims = set(), {}, {}
    inst.step.to_check = [c for call in cancel.calls for c in call.ids]
    inst.step.claims_next = [(i, c) for i, call in enumerate(cancel.calls) for c in call.ids]


def cancel_head(ctx, inst: Timeline, event) -> None:
    child = inst.step.to_check.pop(HEAD)
    index, _ = inst.step.claims_next.pop(HEAD)
    inst.step.claims[child] = index
    inst.grafts.live.pop(child, None)
    inst.step.pending.add(child)
    _forward(inst.machines.timelines.get(child), "cancel")


def refuse_head(ctx, inst: Timeline, event) -> None:
    child = inst.step.to_check.pop(HEAD)
    index, _ = inst.step.claims_next.pop(HEAD)
    inst.step.outcomes.setdefault(child, cancel_refusal(inst, child))
    inst.step.claims.setdefault(child, index)


def cancel_landed(ctx, inst: Timeline, event) -> None:
    join: QueuedJoin = event.data
    inst.joins.queue.append(join)
    inst.step.pending.discard(join.child)
    inst.step.outcomes[join.child] = join.reason
    _session(inst, "graft_received", join.child)


def cancels_written(ctx, inst: Timeline, event) -> None:
    cancel, inst.step.cancel = inst.step.cancel, None
    inst.step.outcomes, inst.step.claims = {}, {}
    _resolve(cancel.reply, event.data[0])


def take_row_failures(ctx, inst: Timeline, event) -> None:
    _, failures = event.data
    for use_id, content, is_error, _stopped, _error in failures:
        inst.step.unstored[use_id] = (content, is_error)
    inst.step.unreported = [("cancel_result", stopped, error, None) for _, _, _, stopped, error in failures]
    inst.step.reported = failures[0][4]


def cancel_rows_failed(ctx, inst: Timeline, event) -> None:
    cancel, inst.step.cancel = inst.step.cancel, None
    inst.step.outcomes, inst.step.claims = {}, {}
    _fail(cancel.reply, inst.step.reported)


def cancel_rows_crashed(ctx, inst: Timeline, event) -> None:
    cancel, inst.step.cancel = inst.step.cancel, None
    _report(inst, "machine", [], event.data)
    inst.step.reported = event.data
    _fail(cancel.reply, event.data)


# ------------------------------------------------------------------ effects: joins
def enqueue(ctx, inst: Timeline, event) -> None:
    inst.joins.queue.append(event.data)
    _session(inst, "graft_received", event.data.child)


def focus_back(ctx, inst: Timeline, event) -> None:
    join, waiters, inbox, answering, withdrawn, unmarked, returned = event.data
    inst.grafts.focus = None
    inst.joins.queue.append(join)
    inst.req.waiters.extend(waiters)
    inst.req.inbox = list(inbox) + inst.req.inbox
    inst.req.answering, inst.req.answered = answering, join.reason == "done"
    inst.req.withdrawn |= withdrawn
    inst.req.unmarked += [r for r in unmarked if r not in inst.req.unmarked]
    inst.req.returned += [r for r in returned if all(r is not k for k in inst.req.returned)]


def start_joining(ctx, inst: Timeline, event) -> None:
    inst.joins.hold = False
    inst.joins.news = []
    inst.joins.needs_wake = False
    inst.joins.ends_on_request = False
    inst.req.woken_for = []


def applied(ctx, inst: Timeline, event) -> None:
    committed_, ends = event.data
    closed_out(inst)
    join = inst.joins.queue.pop(HEAD)
    inst.grafts.live.pop(join.child, None)
    wants_wake = inst.grafts.wake.pop(join.child, True)
    made_for = inst.grafts.requests.pop(join.child, ())
    news = not any(_withdrawn(inst, r) for r in made_for)
    if news:
        inst.joins.news.append((join.child, join.reason))
        if wants_wake or not join.replied_cleanly or bool(inst.grafts.live):
            inst.joins.needs_wake = True
        inst.req.woken_for += [r for r in made_for if all(r is not w for w in inst.req.woken_for)]
    inst.joins.ends_on_request = ends
    _forward(inst.machines.timelines.get(join.child), "join_commit" if committed_ else "join_rollback")


def apply_failed(ctx, inst: Timeline, event) -> None:
    child = inst.joins.queue[0].child
    attempts = inst.joins.attempts[child] = inst.joins.attempts.get(child, NONE_YET) + 1
    inst.joins.hold = True
    _report(inst, "commit", [j.child for j in inst.joins.queue], event.data, quiet=attempts > 1)


def last_commit_failed(ctx, inst: Timeline, event) -> None:
    _report(inst, "commit", [inst.joins.queue[0].child], event.data, quiet=True)


def report_abandoned(ctx, inst: Timeline, event) -> None:
    child = inst.joins.queue[0].child
    _report(inst, "abandon", [child], event.data, fails=_tasks_of(inst.grafts.requests.get(child, ())))


def abandon_head(ctx, inst: Timeline, event) -> None:
    join = inst.joins.queue.pop(HEAD)
    inst.grafts.live.pop(join.child, None)
    inst.grafts.wake.pop(join.child, None)
    inst.grafts.requests.pop(join.child, None)
    _forward(inst.machines.timelines.get(join.child), "join_abandon")


def answer_handback(ctx, inst: Timeline, event) -> None:
    inst.mode = Mode(name="turn", user=True, about=tuple(c for c, _ in inst.joins.news))


def wake_run(ctx, inst: Timeline, event) -> None:
    inst.mode = Mode(name="wake", answers=_handed_back(inst), about=tuple(c for c, _ in inst.joins.news))


# ------------------------------------------------------------------ effects: focus & background join
def decide_join_reason(ctx, inst: Timeline, event) -> None:
    if inst.joins.quiet_handback:
        inst.joins.quiet_handback = False
        return
    outcome = event.data if isinstance(event.data, RunOutcome) else None
    reason = "done" if outcome is not None and outcome.kind in ("join", "replied", "empty") else "failed"
    inst.joins.join_reason = CANCELLED if _focus_withdrawn(inst) else reason
    inst.joins.replied_cleanly = (not _focus_withdrawn(inst)) and outcome is not None and outcome.kind == "replied"


def _hand_back(inst: Timeline, event: hsm.Event | None) -> None:
    data = event.data if event is not None else None
    join, also = None, []
    if isinstance(data, QueuedJoin):
        join = data
    elif isinstance(data, tuple) and data and isinstance(data[HEAD], QueuedJoin):
        join, reply = data
        if reply is not None:
            also = [Waiter(reply=reply, user=False)]
    join = join or QueuedJoin(child=inst.timeline_id, reason="failed", trigger=None)
    if inst.joins.unclosed is not None:
        join = join._replace(closing=inst.joins.unclosed)
    if inst.joins.replied_cleanly and join.reason == "done" and join.trigger is None:
        join = join._replace(replied_cleanly=True)
    inst.joins.replied_cleanly = False
    users = [w for w in inst.req.waiters if w.user] + also
    others = [w for w in inst.req.waiters if not w.user]
    payload = (join, users, list(inst.req.inbox), inst.req.answering,
               set(inst.req.withdrawn), list(inst.req.unmarked), list(inst.req.returned))
    inst.req.waiters, inst.req.inbox, inst.req.answering = [], [], None
    for w in others:
        _resolve(w.reply, "joined")
    _forward(inst.machines.timelines.get(inst.source_id), "focus_joined", payload)


def hand_back_join(ctx, inst: Timeline, event) -> None:
    _hand_back(inst, event)


def focus_failure_not_recorded(ctx, inst: Timeline, event) -> None:
    _report(inst, "task", [inst.timeline_id], event.data, src=inst.source_id)
    inst.joins.unclosed = dict(inst.step.unstored)


def ended_as(reason: str, close_first: bool):
    def effect(ctx, inst: Timeline, event) -> None:
        inst.joins.reason, inst.joins.close_first = reason, close_first
    effect.__name__ = f"ended_{reason}{'_early' if close_first else ''}"
    return effect


def ended_by_join(ctx, inst: Timeline, event) -> None:
    inst.joins.reason, inst.joins.close_first = event.data.reason, False


def forced(ctx, inst: Timeline, event) -> None:
    join, inst.joins.forced = event.data
    inst.joins.reason, inst.joins.close_first = join.reason, True


def run_raised(ctx, inst: Timeline, event) -> None:
    inst.joins.reason, inst.joins.close_first = "failed", True
    _note(inst, "graft_failed", tl=inst.timeline_id, error=event.data)


def bg_run_failed_late(ctx, inst: Timeline, event) -> None:
    _report(inst, "task", [inst.timeline_id], event.data, src=inst.source_id)


def _unreported(inst: Timeline, errors: list[BaseException]) -> list[tuple]:
    kind = "cancel" if inst.joins.reason == CANCELLED else "task"
    return [(kind, [inst.timeline_id], e, inst.source_id) for e in errors]


def recorded(ctx, inst: Timeline, event) -> None:
    errors, closed = event.data
    inst.step.unreported += _unreported(inst, errors)
    if not closed:
        inst.joins.unclosed = dict(inst.step.unstored)


def record_crashed(ctx, inst: Timeline, event) -> None:
    inst.step.unreported += _unreported(inst, [event.data])
    if inst.joins.close_first:
        inst.joins.unclosed = dict(inst.step.unstored)


def join_raised(ctx, inst: Timeline, event) -> None:
    for w in inst.joins.watchers:
        _resolve(w, True)
    _resolve(inst.joins.forced, True)
    inst.joins.watchers = []
    closing = inst.joins.unclosed if inst.joins.unclosed is not None else (
        dict(inst.step.unstored) if inst.joins.close_first else None)
    _forward(inst.machines.timelines.get(inst.source_id), "graft_queued",
             QueuedJoin(child=inst.timeline_id, reason=inst.joins.reason, trigger=None, closing=closing))


def close_abandoned(ctx, inst: Timeline, event) -> None:
    if inst.joins.unclosed is not None:
        _note(inst, "close_tool_uses", tl=inst.timeline_id, known=dict(inst.joins.unclosed))


def report_next(ctx, inst: Timeline, event) -> None:
    _self(inst, "report_next")


def report_one(ctx, inst: Timeline, event) -> None:
    kind, branches, error, src = inst.step.unreported.pop(HEAD)
    _report(inst, kind, branches, error, src=src)


def add_watcher(ctx, inst: Timeline, event) -> None:
    inst.joins.watchers.append(event.data)


# ------------------------------------------------------------------ depth-N child orchestration
def can_fork_guard(ctx, inst: Timeline, event) -> bool:
    return inst.can_fork


def has_running_children(ctx, inst: Timeline, event) -> bool:
    return bool(_still_running(inst))


def no_running_children(ctx, inst: Timeline, event) -> bool:
    return not _still_running(inst)


def has_queued_only(ctx, inst: Timeline, event) -> bool:
    return not _still_running(inst) and bool(inst.joins.queue)


def all_children_queued(ctx, inst: Timeline, event) -> bool:
    return own_graft(ctx, inst, event) and _still_running(inst) == [event.data.child]


def focus_runs_children(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus" and bool(_still_running(inst))


def focus_drains_children(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus" and not _still_running(inst) and bool(inst.joins.queue)


def focus_waits_after(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus" and not inst.final and bool(_still_running(inst)) and not _focus_withdrawn(inst)


def focus_quiet_handback(ctx, inst: Timeline, event) -> bool:
    return inst.kind == "focus" and quiet_after(ctx, inst, event) and not _still_running(inst)


def focus_cancelled_with_children(ctx, inst: Timeline, event) -> bool:
    return _focus_withdrawn(inst) and bool(_still_running(inst))


def set_quiet_handback(ctx, inst: Timeline, event) -> None:
    inst.joins.join_reason = "done"
    inst.joins.replied_cleanly = bool(inst.joins.news) and not inst.joins.needs_wake
    inst.joins.quiet_handback = True


def set_closing_and_forward(ctx, inst: Timeline, event) -> None:
    inst.closing = True
    if inst.grafts.focus:
        _forward(inst.machines.timelines.get(inst.grafts.focus), "closing")


def stop_live_children(ctx, inst: Timeline, event) -> None:
    for c in list(_still_running(inst)):
        _forward(inst.machines.timelines.get(c), "cancel")
    _self(inst, "check")


def abandon_queued_children(ctx, inst: Timeline, event) -> None:
    while inst.joins.queue:
        join = inst.joins.queue.pop(HEAD)
        inst.grafts.live.pop(join.child, None)
        inst.grafts.wake.pop(join.child, None)
        inst.grafts.requests.pop(join.child, None)
        _forward(inst.machines.timelines.get(join.child), "join_abandon")
