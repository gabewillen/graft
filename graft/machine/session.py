"""The session machine: owns host A2A tasks, the event log, and coordinated shutdown."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import hsm

from .plumbing import (Activity, Effect, Entry, _E, _G, _On, _S, _T, _fail, _failure, _forward,
                       _reply_of, _resolve, _self, io, refuse_request)
from .types import (CLOSED_TEXT, HEAD, MAX_LOST, MAX_WRITE_ATTEMPTS, NONE_YET, REQUESTS,
                    Rested, TaskRecord, a2a_CANCELED, a2a_COMPLETED, a2a_FAILED,
                    a2a_INPUT_REQUIRED, a2a_SUBMITTED, a2a_WORKING)


class Session(hsm.Instance):
    def __init__(self, machines, context_id: str, position: int):
        super().__init__()
        self.machines = machines
        self.root: str | None = None
        self.context_id = context_id
        self.tasks: dict[str, TaskRecord] = {}
        self.ended: dict[str, str] = {}
        self.answered_ended: set[int] = set()
        self.req_task: dict[int, str] = {}
        self.graft_tasks: dict[str, tuple[str, ...]] = {}
        self.grafts: set[str] = set()
        self.held: set[str] = set()
        self.pending: list[tuple[int, dict]] = []
        self.next_seq = position + 1
        self.written = position
        self.outbox: list[tuple[str, str, Any]] = []
        self.after_write: list[tuple[int, tuple[str, str, Any]]] = []
        self.waits: list[tuple[int, asyncio.Future]] = []
        self.root_told = False
        self.root_closed = False
        self.scribe_flushed = False
        self.closers: list[asyncio.Future] = []
        self.lost: list[BaseException] = []
        self.blocked = False
        self.attempts = NONE_YET
        self.write_error: BaseException | None = None
        self.stop_errors: list[BaseException] = []

    # ---- event log helpers
    def message(self, seq: int, task: str | None, text: str, error: bool = False) -> dict:
        m: dict[str, Any] = {
            "messageId": f"{self.context_id}-m{seq}", "contextId": self.context_id,
            "role": "ROLE_AGENT", "parts": [{"text": text}],
        }
        if task is not None:
            m["taskId"] = task
        if error:
            m["metadata"] = {"graft": {"error": True}}
        return m

    def status_event(self, task: str, state: str, message: dict | None) -> dict:
        status = {"state": state} | ({"message": message} if message is not None else {})
        return {"statusUpdate": {"taskId": task, "contextId": self.context_id, "status": status}}

    def emit(self, event: dict) -> int:
        seq = self.next_seq
        self.next_seq += 1
        self.pending.append((seq, event))
        return seq

    def emit_status(self, task: str, state: str, message: dict | None = None) -> int:
        rec = self.tasks[task]
        rec.state = state
        return self.emit(self.status_event(task, state, message))

    def say(self, task: str | None, text: str, error: bool = False) -> None:
        if task is None:
            self.emit({"message": self.message(self.next_seq, None, text, error)})
            return
        rec = self.tasks[task]
        msg = self.message(self.next_seq, task, text, error)
        self.emit_status(task, rec.state if error else a2a_WORKING, msg)
        rec.reply = msg

    def working(self, task: str | None) -> None:
        rec = self.tasks.get(task) if task is not None else None
        if rec is not None and rec.state != a2a_WORKING:
            self.emit_status(task, a2a_WORKING)

    def tell(self, line: str) -> None:
        if not self.tasks:
            self.say(None, line, error=True)
        for task in sorted(self.tasks):
            self.say(task, line, error=True)

    def end(self, task: str, state: str, reason: str | None = None) -> None:
        rec = self.tasks.pop(task)
        msg = rec.reply if reason is None else self.message(self.next_seq, task, reason, error=True)
        seq = self.emit(self.status_event(task, state, msg))
        self.ended[task] = state
        self.answered_ended |= {rid for rid, where in rec.requests.items() if where == "answered"}
        for rid, where in sorted(rec.requests.items()):
            if where != "queued":
                self.after_write.append((seq, (self.root, "call_done", rid)))
        self.after_write = [(s, send) for s, send in self.after_write
                            if not (send[1] == "user_message" and send[2].id in rec.requests)]

    def answer(self, rid: int, error: str | None) -> None:
        task = self.req_task.get(rid)
        rec = self.tasks.get(task) if task is not None else None
        if rec is None:
            if error is not None and rid not in self.answered_ended:
                self.tell(f"(harness error: {error})")
            return
        if rec.requests.get(rid) == "answered":
            return
        rec.requests[rid] = "answered"
        if error is not None:
            rec.failed = True
            self.say(task, f"(harness error: {error})", error=True)

    def decide(self, owed: frozenset[str]) -> None:
        for task in sorted(self.tasks):
            rec = self.tasks[task]
            if any(w != "answered" for w in rec.requests.values()):
                continue
            if task in owed:
                if rec.state != a2a_INPUT_REQUIRED:
                    self.emit_status(task, a2a_INPUT_REQUIRED, rec.reply)
            else:
                self.end(task, a2a_FAILED if rec.failed else a2a_COMPLETED)

    def tasks_of(self, requests) -> list[str]:
        return sorted({self.req_task[r] for r in requests if r in self.req_task and self.req_task[r] in self.tasks})


def _on_event(fn: Callable) -> Callable:
    """Wrap a session effect to prune done waits and clear write-blocked on new events."""
    def effect(ctx, inst: Session, event) -> None:
        inst.blocked = False
        inst.waits = [(c, r) for c, r in inst.waits if not r.done()]
        fn(inst, event)
    effect.__name__ = fn.__name__
    return effect


# ------------------------------------------------------------------ handlers
def _submit_new(inst: Session, event) -> None:
    sub, reply = event.data
    inst.tasks[sub.task] = TaskRecord(state=a2a_SUBMITTED, requests={sub.request.id: "queued"})
    seq = inst.emit({"task": {"id": sub.task, "contextId": inst.context_id,
                              "status": {"state": a2a_SUBMITTED}, "history": [sub.message]}})
    inst.req_task[sub.request.id] = sub.task
    inst.after_write.append((seq, (inst.root, "user_message", sub.request)))
    _resolve(reply, "accepted")


def _submit_follow_up(inst: Session, event) -> None:
    sub, reply = event.data
    rec = inst.tasks[sub.task]
    rec.requests[sub.request.id] = "queued"
    rec.reply = None
    state = a2a_SUBMITTED if rec.state == a2a_INPUT_REQUIRED else rec.state
    seq = inst.emit_status(sub.task, state, sub.message)
    inst.req_task[sub.request.id] = sub.task
    inst.after_write.append((seq, (inst.root, "user_message", sub.request)))
    _resolve(reply, "accepted")


def _refuse(answer: str) -> Callable:
    def handler(inst: Session, event) -> None:
        _resolve(event.data[1], answer)
    handler.__name__ = f"refuse_{answer}"
    return _on_event(handler)


def _cancel(inst: Session, event) -> None:
    task, reply = event.data
    rec = inst.tasks[task]
    for rid, where in sorted(rec.requests.items()):
        if where != "queued":
            inst.outbox.append((inst.root, "withdraw", rid))
    rec.reply = None
    inst.end(task, a2a_CANCELED)
    _resolve(reply, "accepted")


def _answer_wait(inst: Session, event) -> None:
    _resolve(event.data[1], (inst.written, False))


def _fail_wait(inst: Session, event) -> None:
    error = inst.lost.pop(HEAD)
    _fail(event.data[1], error)


def _keep_wait(inst: Session, event) -> None:
    inst.waits.append(event.data)


def _said(inst: Session, event) -> None:
    text, tasks = event.data
    if not tasks:
        inst.say(None, text)
    for task in sorted(set(tasks)):
        if task in inst.tasks:
            inst.say(task, text)


def _answering(inst: Session, event) -> None:
    _tl, task = event.data
    inst.working(task)


def _grafts_made(inst: Session, event) -> None:
    grafts, held, tasks = event.data
    (inst.held if held else inst.grafts).update(grafts)
    inst.graft_tasks.update({g: tuple(t) for g, t in tasks.items()})


def _grafts_made_closing(inst: Session, event) -> None:
    grafts, _held, tasks = event.data
    inst.grafts |= set(grafts) | inst.held
    inst.held = set()
    inst.graft_tasks.update({g: tuple(t) for g, t in tasks.items()})
    inst.outbox += [(g, "shutdown", None) for g in sorted(set(grafts))]


def _graft_received(inst: Session, event) -> None:
    child = event.data
    inst.grafts.discard(child)
    inst.held.discard(child)
    for task in inst.graft_tasks.pop(child, ()):
        inst.working(task)


def _grafts_undone(inst: Session, event) -> None:
    for g in event.data:
        inst.grafts.discard(g)
        inst.held.discard(g)
        inst.graft_tasks.pop(g, None)


def _rested_root(inst: Session, event) -> None:
    rested: Rested = event.data
    for rid, error in rested.answered:
        inst.answer(rid, error)
    inst.decide(rested.owed)


def _rested_focus(inst: Session, event) -> None:
    rested: Rested = event.data
    for rid, error in rested.answered:
        inst.answer(rid, error)
    for task in inst.tasks_of(rid for rid, _ in rested.answered):
        rec = inst.tasks[task]
        if rec.state != a2a_INPUT_REQUIRED:
            inst.emit_status(task, a2a_INPUT_REQUIRED, rec.reply)


def _report(inst: Session, event) -> None:
    line, fails = event.data
    for task in fails:
        if task in inst.tasks:
            inst.tasks[task].failed = True
    inst.tell(line)


def _failure_ev(inst: Session, event) -> None:
    errors = [e for e in event.data if all(e is not k for k in inst.lost)]
    waiting = [r for _, r in inst.waits if not r.done()]
    if waiting and errors:
        err = errors.pop(HEAD)
        inst.waits = []
        for r in waiting:
            _fail(r, err)
    inst.lost += errors[:MAX_LOST - len(inst.lost)]


def _begin_closing(inst: Session, event) -> None:
    if event.data is not None:
        inst.closers.append(event.data)
    inst.after_write = [(s, send) for s, send in inst.after_write if send[1] != "user_message"]
    inst.outbox = [send for send in inst.outbox if send[1] != "user_message"]
    inst.outbox.append((inst.root, "closing", None))
    inst.grafts |= inst.held
    inst.held = set()
    inst.outbox += [(g, "shutdown", None) for g in sorted(inst.grafts)]


def _stash_closer(inst: Session, event) -> None:
    if event.data is not None:
        inst.closers.append(event.data)


def _root_closed(inst: Session, event) -> None:
    inst.root_closed = True
    inst.outbox.append(("scribe", "note", (("flush", {}), None)))
    for task in sorted(inst.tasks):
        inst.end(task, a2a_FAILED, reason=CLOSED_TEXT)


def _tell_root(ctx, inst: Session, event) -> None:
    inst.root_told = True
    inst.outbox.append((inst.root, "shutdown", None))


def _scribe_flushed(inst: Session, event) -> None:
    inst.scribe_flushed = True


def _ignore(ctx, inst, event) -> None:
    return None


# ------------------------------------------------------------------ guards
def _submits_new(ctx, inst: Session, event) -> bool:
    sub, _ = event.data
    return not sub.follow_up and sub.task not in inst.tasks and sub.task not in inst.ended


def _submits_follow_up(ctx, inst: Session, event) -> bool:
    sub, _ = event.data
    return sub.follow_up and sub.task in inst.tasks


def _submits_to_ended(ctx, inst: Session, event) -> bool:
    sub, _ = event.data
    return sub.task in inst.ended or (not sub.follow_up and sub.task in inst.tasks)


def _cancels_open(ctx, inst: Session, event) -> bool:
    return event.data[0] in inst.tasks


def _cancels_ended(ctx, inst: Session, event) -> bool:
    return event.data[0] in inst.ended


def _log_past(ctx, inst: Session, event) -> bool:
    return inst.written > event.data[0]


def _lost_kept(ctx, inst: Session, event) -> bool:
    return inst.written <= event.data[0] and bool(inst.lost)


def _rested_root_g(ctx, inst: Session, event) -> bool:
    return event.data.owed is not None and event.data.tl == inst.root


def _rested_focus_g(ctx, inst: Session, event) -> bool:
    return event.data.owed is None


def _root_closes(ctx, inst: Session, event) -> bool:
    return event.data == inst.root and not inst.root_closed


def _has_outbox(ctx, inst: Session, event) -> bool:
    return bool(inst.outbox)


def _writes_due(ctx, inst: Session, event) -> bool:
    return not inst.outbox and bool(inst.pending) and not inst.blocked


def _root_due(ctx, inst: Session, event) -> bool:
    return not inst.outbox and not (inst.pending and not inst.blocked) and not inst.grafts and not inst.root_told


def _stops_now(ctx, inst: Session, event) -> bool:
    return (inst.root_closed and inst.scribe_flushed and not inst.outbox
            and (not inst.pending or inst.blocked) and inst.root_told)


# ------------------------------------------------------------------ flush / write / close activities
def _send_head(ctx, inst: Session, event) -> None:
    target, name, data = inst.outbox.pop(HEAD)
    machine = inst.machines.scribe if target == "scribe" else inst.machines.timelines.get(target)
    if name == "user_message":
        task = inst.req_task.get(data.id)
        rec = inst.tasks.get(task) if task is not None else None
        if rec is not None and rec.requests.get(data.id) == "queued":
            rec.requests[data.id] = "delivered"
    _forward(target=machine, name=name, data=data)


def _tick(ctx, inst: Session, event) -> None:
    _self(inst=inst, name="sent")


def _write_events(inst: Session, event):
    return inst.machines.host.append_events(events=list(inst.pending))


def _written(ctx, inst: Session, event) -> None:
    written = max(inst.written, event.data)
    inst.written = written
    inst.pending = [(s, e) for s, e in inst.pending if s > written]
    due = [send for s, send in inst.after_write if s <= written]
    inst.after_write = [(s, send) for s, send in inst.after_write if s > written]
    inst.outbox += due
    passed = [r for c, r in inst.waits if c < written]
    inst.waits = [(c, r) for c, r in inst.waits if c >= written and not r.done()]
    inst.attempts = NONE_YET
    for reply in passed:
        _resolve(reply, (written, False))


def _write_failed(ctx, inst: Session, event) -> None:
    error = event.data
    waiting = [r for _, r in inst.waits]
    attempts = inst.attempts + 1
    inst.waits = []
    inst.attempts = attempts if attempts < MAX_WRITE_ATTEMPTS else NONE_YET
    inst.blocked = attempts >= MAX_WRITE_ATTEMPTS
    inst.write_error = error
    for reply in waiting:
        _fail(reply, error)


def _closed(ctx, inst: Session, event) -> None:
    errors = list(inst.stop_errors) + ([inst.write_error] if inst.blocked and inst.write_error else [])
    closers, waits = list(inst.closers), [r for _, r in inst.waits]
    inst.closers, inst.waits, inst.stop_errors = [], [], []
    if closers and errors:
        first, *closers = closers
        _fail(first, errors[0] if len(errors) == 1 else BaseExceptionGroup("closing failed", errors))
    for reply in closers:
        _resolve(reply, None)
    for reply in waits:
        _resolve(reply, (inst.written, True))


def _answer_closed(ctx, inst: Session, event) -> None:
    data = event.data
    if event.name == "events_wait":
        _resolve(data[1], (inst.written, True))
    elif event.name in ("submit", "cancel_task"):
        _resolve(data[1], "closing")
    else:
        _resolve(_reply_of(data), None)


async def _stop_machines(ctx, inst: Session, event) -> None:
    errors: list[BaseException] = []
    for sm in [*inst.machines.timelines.values(), inst.machines.scribe]:
        if sm is not None and sm.state():
            try:
                await hsm.Stop(sm)
            except BaseException as e:  # noqa: BLE001
                errors.append(_failure(ctx, e))
    inst.dispatch(inst.context(), hsm.Event(name="stopped", data=errors))


def _take_stop_errors(ctx, inst: Session, event) -> None:
    inst.stop_errors = list(event.data)


def _stop_crashed(ctx, inst: Session, event) -> None:
    inst.stop_errors = [event.data]


def _stops_clean(ctx, inst: Session, event) -> bool:
    return not event.data


def _tell_stop_errors(inst: Session, event):
    failed: list[BaseException] = []
    position = inst.written
    for error in list(inst.stop_errors):
        try:
            line, _ = inst.machines.host.report(kind="machine", src=None, branches=[], error=error,
                                                quiet=False, background=False)
            seq = max(inst.next_seq, position + 1)
            position = inst.machines.host.append_events(events=[(seq, {"message": inst.message(
                seq=seq, task=None, text=line, error=True)})])
        except Exception as e:  # noqa: BLE001
            failed.append(e)
    return position, failed


def _told(ctx, inst: Session, event) -> None:
    position, failed = event.data
    inst.written = max(inst.written, position)
    inst.next_seq = max(inst.next_seq, position + 1)
    inst.stop_errors += list(failed)
    _closed(ctx, inst, event)


def _tell_crashed(ctx, inst: Session, event) -> None:
    inst.stop_errors.append(event.data)
    _closed(ctx, inst, event)


# ------------------------------------------------------------------ graph
def _writer(flush: tuple, takes: tuple) -> list:
    at_ready, on_composite = [], []
    for name, branches in takes:
        for guard, effect in branches:
            extra = (_G(guard),) if guard is not None else ()
            at_ready.append(_T(_On(name), _S("../flush"), *extra, Effect(effect)))
            on_composite.append(_T(_On(name), *extra, Effect(effect)))
    return [
        *on_composite,
        _T(_On("sent", "written", "write_failed"), Effect(_ignore)),
        hsm.Initial(_S("ready")),
        hsm.State("ready", *at_ready),
        hsm.Choice("flush", *flush),
        hsm.State("sending", Entry(_tick), _T(_On("sent"), _S("../flush"))),
        hsm.State(
            "writing", io(done="written", work=_write_events),
            hsm.Defer("shutdown"),
            _T(_On("written"), _S("../flush"), Effect(_written)),
            _T(_On(_E), _S("../flush"), Effect(_write_failed)),
        ),
    ]


_COMMON_TAKES = (
    ("events_wait", [(_log_past, _on_event(_answer_wait)), (_lost_kept, _on_event(_fail_wait)),
                     (None, _on_event(_keep_wait))]),
    ("said", [(None, _on_event(_said))]),
    ("answering", [(None, _on_event(_answering))]),
    ("graft_received", [(None, _on_event(_graft_received))]),
    ("grafts_undone", [(None, _on_event(_grafts_undone))]),
    ("rested", [(_rested_root_g, _on_event(_rested_root)), (_rested_focus_g, _on_event(_rested_focus)),
                (None, _ignore)]),
    ("report", [(None, _on_event(_report))]),
    ("failure", [(None, _on_event(_failure_ev))]),
    ("scribe_flushed", [(None, _on_event(_scribe_flushed))]),
)
_OPEN_TAKES = _COMMON_TAKES + (
    ("submit", [(_submits_new, _on_event(_submit_new)), (_submits_follow_up, _on_event(_submit_follow_up)),
                (_submits_to_ended, _refuse("terminal")), (None, _refuse("not_found"))]),
    ("cancel_task", [(_cancels_open, _on_event(_cancel)), (_cancels_ended, _refuse("terminal")),
                     (None, _refuse("not_found"))]),
    ("grafts_made", [(None, _on_event(_grafts_made))]),
)
_CLOSING_TAKES = _COMMON_TAKES + (
    ("submit", [(None, _refuse("closing"))]),
    ("cancel_task", [(None, _refuse("closing"))]),
    ("grafts_made", [(None, _on_event(_grafts_made_closing))]),
    ("shutdown", [(None, _on_event(_stash_closer))]),
    ("root_closed", [(_root_closes, _on_event(_root_closed)), (None, _ignore)]),
)
_TAKEN = tuple(sorted({name for name, _ in _OPEN_TAKES + _CLOSING_TAKES}))

SESSION = hsm.Define(
    "Session",
    hsm.Initial(_S("open")),
    _T(_On(*REQUESTS, "when_queued", "rest_request"), Effect(refuse_request)),
    hsm.State(
        "open",
        _T(_On("shutdown"), _S("../closing/flush"), Effect(_on_event(_begin_closing))),
        *_writer(flush=(
            _T(_S("sending"), _G(_has_outbox), Effect(_send_head)),
            _T(_S("writing"), _G(_writes_due)),
            _T(_S("ready")),
        ), takes=_OPEN_TAKES),
    ),
    hsm.State(
        "closing",
        *_writer(flush=(
            _T(_S("sending"), _G(_has_outbox), Effect(_send_head)),
            _T(_S("writing"), _G(_writes_due)),
            _T(_S("sending"), _G(_root_due), Effect(_tell_root, _send_head)),
            _T(_S("../ending"), _G(_stops_now)),
            _T(_S("ready")),
        ), takes=_CLOSING_TAKES),
    ),
    hsm.State(
        "ending",
        _T(_On(*_TAKEN), Effect(_answer_closed)),
        hsm.Initial(_S("stopping")),
        hsm.State(
            "stopping", Activity(work=_stop_machines, done="stopped"),
            _T(_On("stopped"), _S("../closed"), _G(_stops_clean), Effect(_closed)),
            _T(_On("stopped"), _S("../telling"), Effect(_take_stop_errors)),
            _T(_On(_E), _S("../telling"), Effect(_stop_crashed)),
        ),
        hsm.State(
            "telling", io(done="told", work=_tell_stop_errors),
            _T(_On("told"), _S("../closed"), Effect(_told)),
            _T(_On(_E), _S("../closed"), Effect(_tell_crashed)),
        ),
        hsm.State("closed"),
    ),
)
