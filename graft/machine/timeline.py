"""Timeline (original / focus graft) and background graft state machine graphs and async activities."""

from __future__ import annotations

import asyncio

import hsm

from .plumbing import (Activity, Effect, Entry, Exit, _E, _G, _On, _S, _T, _failure, described,
                       into, io, mirrored, refuse_request)
from .timeline_ops import (
    Timeline, _apply, _cancel_rows, _branch_row, _close_calls, _close_out, _closing_out, _commit,
    _focus_failure, _join_back, _mark_pending, _mark_rows, _record, _refuse, _reject, _tasks_of,
    _unprepare, _write_request, abandon_head, add_run_waiter, add_watcher, answer_ended,
    answer_handback, answer_handed_off, answer_joined, answer_pending, answer_true, answers_handback,
    answers_inbox, applied, applies_next, apply_failed, arrives_behind_joins, arrives_withdrawn,
    ask_source, begin_driven, begin_run, bg_run_failed_late, call_finished, call_finished_and_forward,
    calls_closed, calls_not_closed, cancel_head, cancel_landed, cancel_rows_crashed, cancel_rows_failed,
    cancels_only, cancels_written, carries_cancels, checked_with_pending, checks_from, branch_done,
    branch_dropped, branch_refusal_not_written, branch_refused_written, branch_row_lost, branch_waits,
    close_abandoned, close_and_release, close_out_failed, close_out_written, close_root, closes_clean,
    closes_out_at_rest, closes_out_first, come_to_rest, committed, decide_join_reason, decline,
    drains_now, drop_branch, drop_message, end_driven, end_run, ended_as, ended_by_join, ends_run,
    ends_wake, enqueue, error_already_shown, error_for_user, error_reported, error_shown, error_to_user, final_clean,
    final_out_first, final_waits, focus_back, focus_cancelled, focus_failure_not_recorded, forced,
    forward_to_focus, forward_to_source, graft_wanted, graph_allows, hand_back_join, hand_over,
    handed_live_to_answer, handed_live_with_queue, handed_withdraws_run, hands_over, has_joined,
    has_queue, has_unreported, head_cancellable, head_refused, ignore, inbox_after, is_driven,
    is_external, is_focus, is_held, join_raised, joins_back, keep_unstored, last_commit_failed,
    last_pending, let_graft_go, made_for_live, mark_failed, may_try, note_late_withdrawal,
    note_withdrawn, nothing_to_mark, one_of_pending, other_pending, own_graft, pass_inbox,
    preparation_crashed, quiet_after, record_crashed, recorded, refuse_cancel_call, refuse_branch_call,
    refuse_head, rejected_in_preparation, report_abandoned, report_next, report_one, request_dropped,
    request_failed, request_live, request_written, retry_left, return_to_source, row_lost,
    row_not_written, row_recorded, rows_marked, rows_written, run_failed_after_it_ended, run_goes_on,
    run_joined, run_raised, run_replied, set_closing, set_final, spawn_failed, spawned,
    start_checking, start_joining, starts_busy, starts_running, starts_withdrawn, stash_message,
    stop_graft, stop_one, stops_due, take_cancel, take_branch, take_hand_over, take_handed,
    take_handed_and_begin, take_inbox, take_message, take_prepared, take_row_failures, take_tool,
    tell_given_up, tick_stop, to_mark, tool_raised, tool_result_kept, tool_row_failed,
    tool_row_failed_late, tool_row_written, undone, unprepare_failed, unspawn_failed, wake_run,
    withdraw_and_forward, withdrawn_after, withdraws_late, withdraws_late_at_rest,
    withdraws_late_only, withdraws_own_focus, withdraws_waiting,
)
from .types import CANCELLED, MESSAGES, REQUESTS, RETURNED, Aborted


# ------------------------------------------------------------------ async activities
async def busy_activity(ctx, inst: Timeline, event) -> None:
    try:
        _close_out(inst)
        _mark_pending(inst)
        woken = list(inst.joins.news) if inst.mode.name == "wake" else None
        runs_for = ([inst.req.answering] if inst.req.answering is not None else []) + list(inst.req.woken_for)
        outcome = await inst.host.run_busy(tl=inst.timeline_id, kind=inst.kind, woken=woken,
                                           tasks=_tasks_of(runs_for))
    except BaseException as error:  # noqa: BLE001
        error = _failure(ctx, error)
        if not ctx.is_done():
            inst.dispatch(inst.context(), hsm.Event(name="run_failed", data=error))
        return
    await inst.dispatch(inst.context(), hsm.Event(name="turn_done", data=outcome))


async def background_activity(ctx, inst: Timeline, event) -> None:
    try:
        outcome = await inst.host.run_background(tl=inst.timeline_id)
    except BaseException as error:  # noqa: BLE001
        error = _failure(ctx, error)
        if not ctx.is_done():
            inst.dispatch(inst.context(), hsm.Event(name="run_failed", data=error))
        return
    await inst.dispatch(inst.context(), hsm.Event(name="finish", data=outcome))


async def tool_activity(ctx, inst: Timeline, event) -> None:
    host, tl, call = inst.host, inst.timeline_id, inst.step.tool
    inst.step.unstored[call.use["id"]] = (call.interrupted, True)
    try:
        content, is_error = await host.call_tool(tl=tl, use=call.use)
    except BaseException as error:  # noqa: BLE001
        _failure(ctx, error)
        cancelled = isinstance(error, asyncio.CancelledError)
        content, is_error = (call.cancelled if cancelled else call.failed), True
    inst.dispatch(inst.context(), hsm.Event(name="tool_returned", data=(content, is_error)))


async def store_activity(ctx, inst: Timeline, event) -> None:
    host, tl, call = inst.host, inst.timeline_id, inst.step.tool
    content, is_error = inst.step.unstored[call.use["id"]]
    try:
        host.store_result(tl=tl, use=call.use, content=content, is_error=is_error)
    except BaseException as error:  # noqa: BLE001
        error = _failure(ctx, error)
        inst.dispatch(inst.context(), hsm.Event(
            name="tool_unstored", data=error if isinstance(error, Exception) else Aborted(described(error))))
        return
    inst.dispatch(inst.context(), hsm.Event(name="tool_stored"))


async def prepare_activity(ctx, inst: Timeline, event) -> None:
    _mark_pending(inst)
    prepared = await inst.host.prepare_branch(src=inst.timeline_id, branch=inst.step.branch)
    inst.dispatch(inst.context(), hsm.Event(name="prepared", data=prepared))


async def spawn_activity(ctx, inst: Timeline, event) -> None:
    machines, source, branch = inst.machines, inst.timeline_id, inst.step.branch
    plans = sorted(inst.step.spawn_plan, key=lambda plan: plan.focus)  # focus graft last
    held, withdrawn = inst.mode.name == "external", set(inst.req.withdrawn)
    made: list[Timeline] = []
    try:
        for plan in plans:
            sm = Timeline(machines=machines, timeline_id=plan.child, source_id=source,
                            kind="focus" if plan.focus else "background", held=held,
                            waiters=plan.waiters, answering=plan.answering, made_for=plan.made_for,
                            withdrawn=withdrawn)
            machines.timelines[plan.child] = sm
            made.append(sm)
            await hsm.Started(None, sm, TIMELINE if plan.focus else BACKGROUND, hsm.Config(ID=plan.child))
    except BaseException as e:  # noqa: BLE001
        e = _failure(ctx, e)
        inst.dispatch(inst.context(), hsm.Event(name="spawn_failed", data=([sm.timeline_id for sm in made], e)))
        return
    inst.dispatch(inst.context(), hsm.Event(name="spawned", data=branch))


async def unspawn_activity(ctx, inst: Timeline, event) -> None:
    made = list(inst.step.spawned_ids)
    grafts = [g for g, _ in inst.step.branch.pairs]
    error = inst.step.spawn_error
    machines, host, source = inst.machines, inst.host, inst.timeline_id
    for child in made:
        sm = machines.timelines.pop(child, None)
        if sm is not None and sm.state():
            await hsm.Stop(sm)
    host.abandon_branch(src=source, grafts=grafts)
    inst.dispatch(inst.context(), hsm.Event(name="unspawned", data=error))


# ------------------------------------------------------------------ graph builders
def _stopping() -> list:
    return [
        hsm.Initial(_S("gate")),
        hsm.Choice("gate", _T(_S("stopping"), _G(stops_due)), _T(_S("still"), Effect(tell_given_up))),
        hsm.State("still"),
        hsm.State("stopping", Entry(tick_stop), _T(_On("stop_tick"), _S("../stop_next"))),
        hsm.Choice("stop_next",
                   _T(_S("stopping"), _G(stops_due), Effect(stop_one)),
                   _T(_S("still"), Effect(tell_given_up))),
    ]


def _work(name: str, *activity: hsm.RedefinableBehaviors) -> hsm.RedefinableState:
    return hsm.State(
        name,
        *activity,
        Exit(answer_pending),
        hsm.Initial(_S("running")),
        hsm.State(
            "running",
            _T(_On("branch_request"), _S("../steps/cancels"), _G(carries_cancels), Effect(take_branch)),
            _T(_On("branch_request"), _S("../branch_gate"), Effect(take_branch)),
            _T(_On("cancel_request"), _S("../steps/cancels"), Effect(take_cancel)),
            _T(_On("tool_request"), _S("../calling"), Effect(take_tool)),
        ),
        hsm.State(
            "calling", Activity(work=tool_activity, done="tool_returned"),
            _T(_On("tool_returned"), _S("../storing"), Effect(tool_result_kept)),
            _T(_On(_E), _S("../storing"), Effect(tool_raised)),
        ),
        hsm.State(
            "storing", Activity(work=store_activity, done="tool_stored"),
            _T(_On("tool_stored"), _S("../running"), Effect(tool_row_written)),
            _T(_On("tool_unstored", _E), _S("../running"), Effect(tool_row_failed)),
        ),
        hsm.Choice(
            "branch_gate",
            _T(_S("preparing"), _G(may_try)),
            _T(_S("steps/branching/rejecting"), Effect(decline)),
        ),
        hsm.State(
            "preparing", Activity(work=prepare_activity, done="prepared"),
            _T(_On("prepared"), _S("../prepared_gate"), Effect(take_prepared)),
            _T(_On(_E), _S("../steps/branching/rejecting"), Effect(preparation_crashed)),
            _T(_On("withdraw"), _S("../../../unpreparing"), _G(ends_run), Effect(note_withdrawn)),
            _T(_On("withdraw"), _G(withdraws_late), Effect(note_late_withdrawal)),
            _T(_On("withdraw"), Effect(note_withdrawn)),
            _T(_On("run_failed"), _S("../../../unpreparing"), _G(error_to_user), Effect(error_for_user)),
            _T(_On("run_failed"), _S("../../../unpreparing"), _G(error_already_shown), Effect(error_shown)),
            _T(_On("run_failed"), _S("../../../unpreparing"), Effect(error_reported)),
            _T(_On("shutdown"), _S("../../../unpreparing"), Effect(set_final)),
            _T(_On("driven_done"), _S("../../../unpreparing"), Effect(end_driven)),
        ),
        hsm.Choice(
            "prepared_gate",
            _T(_S("steps/branching/rejecting"), _G(rejected_in_preparation)),
            _T(_S("steps/branching/committing"), _G(graph_allows)),
            _T(_S("steps/branching/refusing")),
        ),
        hsm.State(
            "steps",
            hsm.Defer("shutdown", "driven_done", "run_failed"),
            hsm.State(
                "branching",
                hsm.Defer("focus_joined", "graft_queued"),
                _T(_On("withdraw"), _G(withdraws_waiting), Effect(note_withdrawn)),
                _T(_On("withdraw"), _G(withdraws_late), Effect(note_late_withdrawal)),
                _T(_On("withdraw"), Effect(withdraw_and_forward)),
                hsm.State(
                    "rejecting", io(done="rejected", work=_reject),
                    _T(_On("rejected"), _S("../../../resume"), Effect(branch_refused_written)),
                    _T(_On(_E), _S("../../../resume"), Effect(branch_refusal_not_written)),
                ),
                hsm.State(
                    "refusing", io(done="refused", work=_refuse),
                    _T(_On("refused"), _S("../../../resume"), Effect(branch_refused_written)),
                    _T(_On(_E), _S("../../../resume"), Effect(branch_refusal_not_written)),
                ),
                hsm.State(
                    "committing", io(done="committed", work=_commit),
                    _T(_On("committed"), _S("../spawning"), Effect(committed)),
                    _T(_On(_E), _S("../refusing")),
                ),
                hsm.State(
                    "spawning", hsm.Activity(spawn_activity),
                    hsm.Defer("withdraw", "graft_step"),
                    _T(_On("spawned"), _S("../recording"), Effect(spawned)),
                    _T(_On("spawn_failed"), _S("../unspawning"), Effect(spawn_failed)),
                ),
                hsm.State(
                    "unspawning", Activity(work=unspawn_activity, done="unspawned"),
                    hsm.Defer("graft_step"),
                    _T(_On("unspawned"), _S("../refusing"), Effect(undone)),
                    _T(_On(_E), _S("../refusing"), Effect(unspawn_failed)),
                ),
                hsm.State(
                    "recording", io(done="row_written", work=_branch_row),
                    _T(_On("row_written"), _S("../../../branch_choice"), Effect(row_recorded)),
                    _T(_On(_E), _S("../../../branch_choice"), Effect(row_not_written)),
                ),
                hsm.State(
                    "flushing", io(done="flushed", work=_close_calls),
                    _T(_On("flushed"), _S("../../../handed"), Effect(calls_closed)),
                    _T(_On(_E), _S("../../../handed"), Effect(calls_not_closed)),
                ),
            ),
            hsm.State(
                "cancels",
                Entry(checks_from),
                hsm.Defer("branch_request"),
                _T(_On("withdraw"), _G(withdraws_late), Effect(note_late_withdrawal)),
                _T(_On("withdraw"), Effect(note_withdrawn)),
                _T(_On("graft_queued"), _G(one_of_pending), Effect(cancel_landed)),
                _T(_On("graft_queued"), _G(own_graft), Effect(enqueue)),
                _T(_On("graft_queued"), Effect(ignore)),
                hsm.Initial(_S("checking")),
                hsm.State("checking", Entry(start_checking), _T(_On("check"), _S("../check_choice"))),
                hsm.Choice(
                    "check_choice",
                    _T(_S("checking"), _G(head_cancellable), Effect(cancel_head)),
                    _T(_S("checking"), _G(head_refused), Effect(refuse_head)),
                    _T(_S("waiting"), _G(checked_with_pending)),
                    _T(_S("writing")),
                ),
                hsm.State(
                    "waiting",
                    _T(_On("graft_queued"), _S("../writing"), _G(last_pending), Effect(cancel_landed)),
                    _T(_On("graft_queued"), _G(other_pending), Effect(cancel_landed)),
                    _T(_On("graft_queued"), _G(own_graft), Effect(enqueue)),
                    _T(_On("graft_queued"), Effect(ignore)),
                ),
                hsm.State(
                    "writing", io(done="cancel_rows", work=_cancel_rows),
                    _T(_On("cancel_rows"), _S("../../../cancels_done"), _G(rows_written), Effect(cancels_written)),
                    _T(_On("cancel_rows"), _S("../row_choice"), Effect(take_row_failures)),
                    _T(_On(_E), _S("../../../cancels_done"), Effect(cancel_rows_crashed)),
                ),
                hsm.State("reporting", Entry(report_next), _T(_On("report_next"), _S("../row_choice"))),
                hsm.Choice(
                    "row_choice",
                    _T(_S("reporting"), _G(has_unreported), Effect(report_one)),
                    _T(_S("../../cancels_done"), Effect(cancel_rows_failed)),
                ),
            ),
        ),
        hsm.Choice(
            "branch_choice",
            _T(_S("steps/branching/flushing"), _G(hands_over), Effect(take_hand_over)),
            _T(_S("resume"), _G(row_lost), Effect(branch_row_lost)),
            _T(_S("resume"), Effect(branch_done)),
        ),
        hsm.Choice("resume", _T(_S("running"), _G(run_goes_on)), _T(_S("../../closing_calls"))),
        hsm.Choice("handed", _T(_S("../../../passing"), Effect(pass_inbox))),
        hsm.Choice(
            "cancels_done",
            _T(_S("branch_gate"), _G(branch_waits)),
            _T(_S("running"), _G(branch_dropped), Effect(drop_branch)),
            _T(_S("running"), _G(cancels_only)),
            _T(_S("../../closing_calls")),
        ),
    )


TIMELINE = hsm.Define(
    "Timeline",
    hsm.Initial(_S("start")),
    hsm.Choice(
        "start",
        _T(_S("live/engaged/busy/run"), _G(starts_busy)),
        _T(_S("live/engaged/withdrawing"), _G(starts_withdrawn)),
        _T(_S("live/idle")),
    ),
    _T(_On(*REQUESTS, RETURNED, "when_queued", "may_continue", "tool_request"), Effect(refuse_request)),
    _T(_On("withdraw"), _G(withdraws_late), Effect(note_late_withdrawal)),
    _T(_On("withdraw"), Effect(note_withdrawn)),
    _T(_On("call_done"), Effect(call_finished)),
    _T(_On("graft_step"), _G(graft_wanted), Effect(let_graft_go)),
    _T(_On("graft_step"), Effect(stop_graft)),
    _T(_On("result_unstored"), Effect(keep_unstored)),
    _T(_On("inbox_handed"), Effect(take_handed)),
    _T(_On("rest_request"), _G(has_joined), Effect(answer_joined)),
    _T(_On("rest_request"), Effect(refuse_request)),
    _T(_On("graft_queued"), _G(own_graft), Effect(enqueue)),
    _T(_On("graft_queued"), Effect(ignore)),
    _T(_On("branch_request"), Effect(refuse_branch_call)),
    _T(_On("cancel_request"), Effect(refuse_cancel_call)),
    _T(_On("poke", "shutdown", "check", "tool_returned", "tool_stored", "stop_tick"), Effect(ignore)),
    _T(_On("tool_unstored"), Effect(tool_row_failed_late)),
    _T(_On("run_failed"), Effect(run_failed_after_it_ended)),
    _T(_On("closing"), Effect(set_closing)),
    hsm.State(
        "live",
        _T(_On("join"), _S("../queued"), _G(is_focus), Effect(hand_back_join)),
        _T(_On("join"), Effect(refuse_request)),
        _T(_On("shutdown"), _S("../closed"), Effect(close_root)),
        hsm.State(
            "idle",
            mirrored("active"),
            _T(_On(*MESSAGES), _G(arrives_withdrawn), Effect(drop_message)),
            _T(_On(*MESSAGES), _S("../engaged/joining"), _G(arrives_behind_joins), Effect(stash_message)),
            _T(_On(*MESSAGES), _S("../engaged/requesting"), Effect(take_message)),
            _T(_On("turn_request"), _S("../engaged/busy/driven"), _G(is_external), Effect(begin_driven)),
            _T(_On("turn_request"), _S("../engaged/busy/run"), Effect(begin_run)),
            _T(_On("graft_queued"), _S("../receiving"), _G(own_graft)),
            _T(_On("graft_queued"), Effect(ignore)),
            _T(_On("drain_request"), _S("../engaged/joining"), _G(has_queue), Effect(add_run_waiter)),
            _T(_On("drain_request"), Effect(answer_ended)),
            _T(_On("rest_request"), Effect(answer_ended)),
            _T(_On("poke"), _S("../engaged/joining"), _G(has_queue)),
            _T(_On("poke"), Effect(ignore)),
            _T(_On("shutdown"), _S("../engaged/joining"), _G(has_queue), Effect(set_final)),
            _T(_On("shutdown"), _S("../engaged/closing_out"), _G(closes_out_at_rest), Effect(set_final)),
            _T(_On("shutdown"), _S("../../closed"), Effect(close_root)),
            _T(_On("inbox_handed"), _S("../engaged/joining"), _G(handed_live_with_queue), Effect(take_handed)),
            _T(_On("inbox_handed"), _S("../engaged/requesting"), _G(handed_live_to_answer), Effect(take_handed_and_begin)),
            _T(_On("inbox_handed"), Effect(take_handed)),
            _T(_On("withdraw"), _S("../engaged/handing_back"), _G(withdraws_own_focus), Effect(note_late_withdrawal)),
            _T(_On("withdraw"), _S("gate"), _G(withdraws_late_at_rest), Effect(note_late_withdrawal)),
            _T(_On("withdraw"), _S("gate"), Effect(note_withdrawn)),
            *_stopping(),
        ),
        hsm.State(
            "engaged",
            hsm.Defer("shutdown"),
            _T(_On(*MESSAGES), Effect(stash_message)),
            _T(_On("rest_request", "drain_request"), Effect(add_run_waiter)),
            hsm.State(
                "requesting", mirrored("active"), io(done="request_written", work=_write_request),
                _T(_On("request_written"), _S("../busy/run"), _G(request_live), Effect(request_written)),
                _T(_On("request_written"), _S("../dropping"), Effect(request_dropped)),
                _T(_On(_E), _S("../rest"), Effect(request_failed)),
            ),
            hsm.State(
                "dropping", io(done="dropped", work=_mark_rows),
                _T(_On("dropped"), _S("../rest"), Effect(rows_marked)),
                _T(_On(_E), _S("../rest"), Effect(mark_failed)),
            ),
            hsm.State(
                "busy",
                mirrored("active"),
                _T(_On("run_failed", _E), _S("../failing"), _G(is_focus)),
                _T(_On("run_failed", _E), _S("../closing_calls"), _G(error_to_user), Effect(error_for_user)),
                _T(_On("run_failed", _E), _S("../closing_calls"), _G(error_already_shown), Effect(error_shown)),
                _T(_On("run_failed", _E), _S("../closing_calls"), Effect(error_reported)),
                _T(_On("withdraw"), _S("../closing_calls"), _G(ends_run), Effect(note_withdrawn)),
                _T(_On("withdraw"), _S("../closing_calls"), _G(ends_wake), Effect(note_late_withdrawal)),
                _T(_On("withdraw"), _G(withdraws_late_only), Effect(note_late_withdrawal)),
                _T(_On("withdraw"), Effect(note_withdrawn)),
                _T(_On("may_continue"), _G(run_goes_on), Effect(answer_true)),
                _T(_On("may_continue"), _S("../closing_calls")),
                _T(_On("inbox_handed"), _S("../closing_calls"), _G(handed_withdraws_run), Effect(take_handed)),
                _T(_On("inbox_handed"), Effect(take_handed)),
                _T(_On("shutdown"), _S("../closing_calls"), Effect(set_final)),
                _work("run", Activity(work=busy_activity, done="run"),
                      _T(_On("turn_done"), _S("../../ran"), Effect(end_run))),
                _work("driven", _T(_On("driven_done"), _S("../../rest"), Effect(end_driven))),
            ),
            hsm.State(
                "unpreparing", io(done="unprepared", work=_unprepare),
                _T(_On("unprepared"), _S("../unwound")),
                _T(_On(_E), _S("../unwound"), Effect(unprepare_failed)),
            ),
            hsm.Choice("unwound", _T(_S("rest"), _G(is_driven)), _T(_S("closing_calls"))),
            hsm.Choice("ran", _T(_S("handing_back"), _G(joins_back)), _T(_S("rest"))),
            hsm.State(
                "handing_back", Entry(decide_join_reason), io(done="join_computed", work=_join_back),
                _T(_On("join_computed"), _S("../../../queued"), Effect(hand_back_join)),
                _T(_On(_E), _S("../unrecorded"), Effect(focus_failure_not_recorded)),
            ),
            hsm.State(
                "closing_calls", io(done="calls_closed", work=_close_calls),
                _T(_On("calls_closed"), _S("../withdrawn_choice"), Effect(calls_closed)),
                _T(_On(_E), _S("../withdrawn_choice"), Effect(calls_not_closed)),
            ),
            hsm.Choice("withdrawn_choice", _T(_S("withdrawn_end"), _G(nothing_to_mark)), _T(_S("withdrawing"))),
            hsm.State(
                "withdrawing", Entry(to_mark), io(done="rows_marked", work=_mark_rows),
                _T(_On("rows_marked"), _S("../withdrawn_end"), Effect(rows_marked)),
                _T(_On(_E), _S("../withdrawn_end"), Effect(mark_failed)),
            ),
            hsm.Choice("withdrawn_end", _T(_S("handing_back"), _G(focus_cancelled)), _T(_S("rest"))),
            hsm.State(
                "failing", io(done="failure_recorded", work=_focus_failure),
                _T(_On("failure_recorded"), _S("../../../queued"), Effect(hand_back_join)),
                _T(_On(_E), _S("../unrecorded"), Effect(focus_failure_not_recorded)),
            ),
            hsm.Choice("unrecorded", _T(_S("../../queued"), Effect(hand_back_join))),
            hsm.State(
                "joining",
                mirrored("joining"),
                Entry(start_joining),
                _T(_On("shutdown"), Effect(set_final)),
                hsm.Initial(_S("next")),
                hsm.Choice("next", _T(_S("applying"), _G(applies_next)), _T(_S("../after"))),
                hsm.State(
                    "applying", io(done="applied", work=_apply),
                    _T(_On("applied"), _S("../next"), Effect(applied)),
                    _T(_On(_E), _S("../next"), _G(retry_left), Effect(apply_failed)),
                    _T(_On(_E), _S("../giving_up"), Effect(last_commit_failed)),
                ),
                hsm.Choice("giving_up", _T(_S("abandoning"), Effect(report_abandoned))),
                hsm.Choice("abandoning", _T(_S("next"), Effect(abandon_head))),
            ),
            hsm.Choice("rest", _T(_S("stops"), _G(stops_due)), _T(_S("resting"), Effect(tell_given_up))),
            hsm.State("stops", Entry(tick_stop), _T(_On("stop_tick"), _S("../stops_next"))),
            hsm.Choice("stops_next",
                       _T(_S("stops"), _G(stops_due), Effect(stop_one)),
                       _T(_S("resting"), Effect(tell_given_up))),
            hsm.Choice(
                "resting",
                _T(_S("joining"), _G(drains_now)),
                _T(_S("../shutting"), _G(closes_clean), into("active")),
                _T(_S("closing_out"), _G(closes_out_first)),
                _T(_S("requesting"), _G(answers_inbox), Effect(take_inbox)),
                _T(_S("../idle"), Effect(come_to_rest)),
            ),
            hsm.Choice(
                "after",
                _T(_S("../shutting"), _G(final_clean), into("active")),
                _T(_S("closing_out"), _G(final_out_first)),
                _T(_S("../idle"), _G(final_waits), Effect(come_to_rest)),
                _T(_S("withdrawing"), _G(withdrawn_after)),
                _T(_S("requesting"), _G(inbox_after), Effect(take_inbox)),
                _T(_S("../idle"), _G(quiet_after), Effect(come_to_rest)),
                _T(_S("busy/run"), _G(answers_handback), Effect(answer_handback)),
                _T(_S("busy/run"), Effect(wake_run)),
            ),
            hsm.State(
                "closing_out", io(done="closed_out", work=_closing_out),
                _T(_On("closed_out"), _S("../../shutting"), Effect(close_out_written), into("active")),
                _T(_On(_E), _S("../../shutting"), Effect(close_out_failed), into("active")),
            ),
        ),
        hsm.Choice("shutting", _T(_S("../closed"), Effect(close_root))),
        hsm.Choice("passing", _T(_S("branched"), Effect(hand_over))),
        hsm.Choice("receiving", _T(_S("engaged/joining"), Effect(enqueue))),
        hsm.State(
            "branched",
            mirrored("branched"),
            _T(_On("user_message"), Effect(forward_to_focus)),
            _T(_On(RETURNED), Effect(stash_message)),
            _T(_On("withdraw"), _S("gate"), _G(withdraws_late), Effect(note_late_withdrawal)),
            _T(_On("withdraw"), _S("gate"), Effect(withdraw_and_forward)),
            *_stopping(),
            _T(_On("call_done"), Effect(call_finished_and_forward)),
            _T(_On("focus_joined"), _S("../engaged/joining"), Effect(focus_back)),
            _T(_On("rest_request"), Effect(answer_handed_off)),
            _T(_On("shutdown"), _S("../releasing"), Effect(close_and_release)),
        ),
        hsm.State(
            "releasing", io(done="released", work=_closing_out),
            _T(_On("released"), _S("../../closed"), Effect(close_root)),
            _T(_On(_E), _S("../../closed"), Effect(close_out_failed, close_root)),
        ),
    ),
    hsm.State(
        "queued",
        mirrored("queued"),
        _T(_On("join_commit", "join_rollback"), _S("../joined"), into("joined")),
        _T(_On("join_abandon"), _S("../abandoned"), into("abandoned"), Effect(close_abandoned)),
        _T(_On("user_message"), Effect(return_to_source)),
        _T(_On("inbox_handed"), Effect(forward_to_source)),
        _T(_On("rest_request"), Effect(answer_joined)),
    ),
    hsm.Final("joined"),
    hsm.Final("abandoned"),
    hsm.Final("closed"),
)


BACKGROUND = hsm.Define(
    "BackgroundTimeline",
    hsm.Initial(_S("start")),
    hsm.Choice(
        "start",
        _T(_S("live/background/held"), _G(is_held)),
        _T(_S("live/background/running"), _G(starts_running)),
        _T(_S("live/recording"), Effect(ended_as(reason=CANCELLED, close_first=False))),
    ),
    _T(_On(*REQUESTS, "rest_request", "may_continue", "tool_request"), Effect(refuse_request)),
    _T(_On("when_queued"), Effect(answer_true)),
    _T(_On("result_unstored"), Effect(keep_unstored)),
    _T(_On("cancel_request"), Effect(refuse_cancel_call)),
    _T(_On("branch_request"), Effect(refuse_branch_call)),
    _T(_On("shutdown", "finish", "closing", "tool_returned", "tool_stored", "cancel"), Effect(ignore)),
    _T(_On("run_failed"), Effect(bg_run_failed_late)),
    _T(_On("tool_unstored"), Effect(tool_row_failed_late)),
    hsm.State(
        "live",
        _T(_On("when_queued"), Effect(add_watcher)),
        hsm.State(
            "background",
            mirrored("background"),
            _T(_On("finish"), _S("../ending")),
            _T(_On("cancel"), _S("../recording"), Effect(ended_as(reason=CANCELLED, close_first=True))),
            _T(_On("shutdown"), _S("../recording"), Effect(ended_as(reason="failed", close_first=True))),
            _T(_On("join"), _S("../recording"), Effect(forced)),
            _T(_On("run_failed", _E), _S("../recording"), Effect(run_raised)),
            hsm.State("held"),
            hsm.State(
                "running", Activity(work=background_activity, done="run"),
                _T(_On("may_continue"), _G(made_for_live), Effect(ask_source)),
                _T(_On("may_continue"), _S("../../recording"), Effect(ended_as(reason=CANCELLED, close_first=True))),
                hsm.Initial(_S("stepping")),
                hsm.State("stepping", _T(_On("tool_request"), _S("../calling"), Effect(take_tool))),
                hsm.State(
                    "calling", Activity(work=tool_activity, done="tool_returned"),
                    _T(_On("tool_returned"), _S("../storing"), Effect(tool_result_kept)),
                    _T(_On(_E), _S("../storing"), Effect(tool_raised)),
                ),
                hsm.State(
                    "storing", Activity(work=store_activity, done="tool_stored"),
                    _T(_On("tool_stored"), _S("../stepping"), Effect(tool_row_written)),
                    _T(_On("tool_unstored", _E), _S("../stepping"), Effect(tool_row_failed)),
                ),
            ),
        ),
        hsm.Choice(
            "ending",
            _T(_S("recording"), _G(run_joined), Effect(ended_by_join)),
            _T(_S("recording"), _G(run_replied), Effect(ended_as(reason="done", close_first=False))),
            _T(_S("recording"), Effect(ended_as(reason="failed", close_first=False))),
        ),
        hsm.State(
            "recording",
            hsm.Initial(_S("writing")),
            hsm.State(
                "writing", io(done="recorded", work=_record),
                _T(_On("recorded"), _S("../report_choice"), Effect(recorded)),
                _T(_On(_E), _S("../report_choice"), Effect(record_crashed)),
            ),
            hsm.State("reporting", Entry(report_next), _T(_On("report_next"), _S("../report_choice"))),
            hsm.Choice(
                "report_choice",
                _T(_S("reporting"), _G(has_unreported), Effect(report_one)),
                _T(_S("../../queued"), Effect(join_raised)),
            ),
        ),
    ),
    hsm.State(
        "queued",
        mirrored("queued"),
        _T(_On("join_commit", "join_rollback"), _S("../joined"), into("joined")),
        _T(_On("join_abandon"), _S("../abandoned"), into("abandoned"), Effect(close_abandoned)),
    ),
    hsm.Final("joined"),
    hsm.Final("abandoned"),
)
