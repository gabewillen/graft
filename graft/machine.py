"""State machines (stateforward.hsm) for timelines and the single conversation slot.

A timeline that can hold the conversation (the original, and a focus graft):

    live
      active                                 holds the conversation
        idle        at rest: no model call in flight, nothing half-written
        busy
          turn      answering a request (a user message, or one a clone handed back): may clone
          reporting woken only to tell the user what background grafts did: clone is refused
      cloned        suspended while its focus graft holds the conversation
      joining       applying joins: every join waiting on its queue, in order
    queued          (a focus graft) frozen, its join waiting on its source; not final
    joined          final: its join was applied

    idle  --turn_begin-->   turn            idle --report_begin--> reporting
    busy  --turn_end-->     idle            (one transition, on the parent of turn and reporting)
    turn  --clone_commit [original, one focus graft, room for the background grafts]--> cloned
    turn  --clone_background [original, room]--> turn   (internal: the source keeps the conversation)
    cloned --clone_abort--> turn            (the clone call could not be committed: undone)
    idle  --drain_begin [joins queued]--> joining
    cloned --focus_joined--> joining        (the focus graft's join is put on the queue)
    joining --join_applied [the head of the queue]--> joining   (internal: that join leaves the queue)
    joining --join_commit | join_rollback--> active (idle)
    active --join [a graft: it has a source]--> queued
    queued --join_commit | join_rollback--> joined
    queued --join_abandon--> abandoned      (final: the join could not be applied; it is given up)
    live  --graft_queued--> live            (internal: a background graft's join is put on the queue)
    live  --grafts_dropped--> live          (internal: grafts that never joined are forgotten)

A background graft (it runs concurrently and never holds the conversation):

    background --finish--> queued --join_commit | join_rollback--> joined (final)
    queued --join_abandon--> abandoned      (final: the join could not be applied; it is given up)
    background --abandon--> abandoned       (final: its join event could not be raised)

`queued` means the graft is frozen and its join event, with its reason and trigger, is waiting on
its source; `joined` means that join was applied to the source (committed, or rolled back to a
record). A focus graft and a background graft go through the same two steps, so no join is ever
lost between the event and its commit. The queue is the source's: finish order, kept on the
source's machine instance, changed only by transition effects, and an entry leaves it only when
its join has been applied.

Global invariant: at most one branch holds the conversation. Exactly one timeline is in `active`
or `joining`, and it is the holder of the single global `Slot` machine, which only accepts a
handoff from the current holder. Background grafts are outside the slot: creating one is not a
handoff and their joins never move it.

The graph is the authority for what may happen. A drain is refused mid-turn, while suspended, or
with nothing queued; a clone is refused outside `turn`, from a graft (grafts are one level deep),
and beyond MAX_LIVE_BACKGROUND live background grafts; a join is refused from the original and
a second time (a graft never has grafts of its own to leave behind: only the original clones). Callers dispatch and are told accepted or refused;
neither they nor this module decide by reading a state first. An event is accepted when a
transition's effect ran for it (each event carries a token the effect records), so internal
transitions are observable too.

Every `Machines` operation is one critical section: it takes `Machines._op`, so two operations
never interleave even if a dispatch yields to the event loop, and it runs to completion even if
the calling task is cancelled meanwhile (the cancellation is delivered afterwards). An operation
that needs several dispatches therefore cannot be left half done.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any, Awaitable, Callable, NamedTuple, TypeVar

import hsm

RUNNING = ("active", "joining")  # lifecycle states that hold the conversation
MAX_LIVE_BACKGROUND = 4  # live (running or queued) background grafts per source

T = TypeVar("T")


class TransitionRefused(Exception):
    pass


class QueuedJoin(NamedTuple):
    """A join event waiting on its source: who, why, and the user message it hands back."""
    child: str
    reason: str
    trigger: int | None


class Sent(NamedTuple):
    """Event data: a token (recorded by the effect of the transition that takes the event) and
    the payload the guards and effects read."""
    token: int
    payload: Any


class TimelineSM(hsm.Instance):
    def __init__(self, timeline_id: str, source_id: str | None, kind: str):
        super().__init__()
        self.timeline_id = timeline_id
        self.source_id = source_id
        self.kind = kind  # original | focus | background: fixed at creation
        self.queue: list[QueuedJoin] = []  # joins waiting on this timeline, in finish order
        self.grafts: dict[str, bool] = {}  # its grafts that have not joined: id -> is a focus graft
        self.accepted: int | None = None  # token of the last event a transition took


def clone_refusal(inst: TimelineSM, grafts: list[tuple[str, bool]]) -> str | None:
    """Why inst may not make this clone call, apart from where it is in its lifecycle. The one
    statement of these rules: it is the guard of both clone transitions."""
    if inst.source_id is not None:
        return "only the original clones (grafts are one level deep)"
    if not grafts:
        return "a clone call needs at least one graft"
    focus = [g for g, f in grafts if f]
    if len(focus) > 1:
        return f"at most one graft may take the conversation, got {focus}"
    live = [g for g, is_focus in inst.grafts.items() if not is_focus]
    if len(live) + sum(1 for _, f in grafts if not f) > MAX_LIVE_BACKGROUND:
        return (f"at most {MAX_LIVE_BACKGROUND} background clones may be live at once "
                f"({', '.join(live) or 'none'} still {'is' if len(live) == 1 else 'are'})")
    return None


def _may_clone_focus(ctx, inst: TimelineSM, event) -> bool:
    grafts = event.data.payload
    return clone_refusal(inst, grafts) is None and any(f for _, f in grafts)


def _may_clone_background(ctx, inst: TimelineSM, event) -> bool:
    grafts = event.data.payload
    return clone_refusal(inst, grafts) is None and not any(f for _, f in grafts)


def _may_join(ctx, inst: TimelineSM, event) -> bool:
    return inst.source_id is not None  # the original has nobody to join


def _has_queue(ctx, inst: TimelineSM, event) -> bool:
    return bool(inst.queue)


def _is_head(ctx, inst: TimelineSM, event) -> bool:
    return bool(inst.queue) and inst.queue[0].child == event.data.payload


def _add_grafts(inst: TimelineSM, grafts: list[tuple[str, bool]]) -> None:
    inst.grafts.update(dict(grafts))


def _drop_grafts(inst: TimelineSM, ids: list[str]) -> None:
    for g in ids:
        inst.grafts.pop(g, None)


def _enqueue(inst: TimelineSM, join: QueuedJoin) -> None:
    inst.queue.append(join)


def _dequeue(inst: TimelineSM, child: str) -> None:
    inst.queue.pop(0)
    inst.grafts.pop(child, None)


def _fx(change: Callable[[TimelineSM, Any], None] | None = None):
    """The effect of a transition: apply `change` with the event's payload, then record the
    event's token. Every transition has one, so taking a transition is always observable."""
    def effect(ctx, inst, event) -> None:
        if change is not None:
            change(inst, event.data.payload)
        inst.accepted = event.data.token
    return hsm.Effect(effect)


TIMELINE = hsm.Define(
    "Timeline",
    hsm.Initial(hsm.Target("live")),
    hsm.State(
        "live",
        hsm.Initial(hsm.Target("active")),
        hsm.Transition(hsm.On("graft_queued"), _fx(_enqueue)),
        hsm.Transition(hsm.On("grafts_dropped"), _fx(_drop_grafts)),
        hsm.State(
            "active",
            hsm.Initial(hsm.Target("idle")),
            hsm.Transition(hsm.On("join"), hsm.Target("../../queued"), hsm.Guard(_may_join), _fx()),
            hsm.State(
                "idle",
                hsm.Transition(hsm.On("turn_begin"), hsm.Target("../busy/turn"), _fx()),
                hsm.Transition(hsm.On("report_begin"), hsm.Target("../busy/reporting"), _fx()),
                hsm.Transition(hsm.On("drain_begin"), hsm.Target("../../joining"), hsm.Guard(_has_queue), _fx()),
            ),
            hsm.State(
                "busy",
                hsm.Transition(hsm.On("turn_end"), hsm.Target("../idle"), _fx()),
                hsm.State(
                    "turn",
                    hsm.Transition(hsm.On("clone_commit"), hsm.Target("../../../cloned"),
                                   hsm.Guard(_may_clone_focus), _fx(_add_grafts)),
                    hsm.Transition(hsm.On("clone_background"), hsm.Guard(_may_clone_background),
                                   _fx(_add_grafts)),
                ),
                hsm.State("reporting"),
            ),
        ),
        hsm.State(
            "cloned",
            hsm.Transition(hsm.On("focus_joined"), hsm.Target("../joining"), _fx(_enqueue)),
            hsm.Transition(hsm.On("clone_abort"), hsm.Target("../active/busy/turn"), _fx()),
        ),
        hsm.State(
            "joining",
            hsm.Transition(hsm.On("join_applied"), hsm.Guard(_is_head), _fx(_dequeue)),
            hsm.Transition(hsm.On("join_commit"), hsm.Target("../active"), _fx()),
            hsm.Transition(hsm.On("join_rollback"), hsm.Target("../active"), _fx()),
        ),
    ),
    hsm.State(
        "queued",
        hsm.Transition(hsm.On("join_commit"), hsm.Target("../joined"), _fx()),
        hsm.Transition(hsm.On("join_rollback"), hsm.Target("../joined"), _fx()),
        hsm.Transition(hsm.On("join_abandon"), hsm.Target("../abandoned"), _fx()),
    ),
    hsm.Final("joined"),
    hsm.Final("abandoned"),
)

BACKGROUND = hsm.Define(
    "BackgroundTimeline",
    hsm.Initial(hsm.Target("background")),
    hsm.State(
        "background",
        hsm.Transition(hsm.On("finish"), hsm.Target("../queued"), _fx()),
        hsm.Transition(hsm.On("abandon"), hsm.Target("../abandoned"), _fx()),
    ),
    hsm.State(
        "queued",
        hsm.Transition(hsm.On("join_commit"), hsm.Target("../joined"), _fx()),
        hsm.Transition(hsm.On("join_rollback"), hsm.Target("../joined"), _fx()),
        hsm.Transition(hsm.On("join_abandon"), hsm.Target("../abandoned"), _fx()),
    ),
    hsm.Final("joined"),
    hsm.Final("abandoned"),
)


class SlotSM(hsm.Instance):
    def __init__(self, holder: str):
        super().__init__()
        self.holder = holder


def _handoff_ok(ctx, inst: SlotSM, event) -> bool:
    frm, _to = event.data
    return inst.holder == frm


def _handoff(ctx, inst: SlotSM, event) -> None:
    inst.holder = event.data[1]


SLOT = hsm.Define(
    "Slot",
    hsm.Initial(hsm.Target("running")),
    hsm.State(
        "running",
        hsm.Transition(hsm.On("handoff"), hsm.Guard(_handoff_ok), hsm.Effect(_handoff)),
    ),
)


def lifecycle(state_path: str) -> str:
    """'/Timeline/live/active/busy/turn' -> 'active': the lifecycle state, the one mirrored to
    the store (`live` only groups the states that share the queue). A stopped machine has no
    state: 'stopped'."""
    parts = state_path.split("/")[2:]
    if parts[:1] == ["live"]:
        parts = parts[1:]
    return parts[0] if parts else "stopped"


def short(state_path: str) -> str:
    return state_path.rsplit("/", 1)[-1]


class Machines:
    def __init__(self):
        self.timelines: dict[str, TimelineSM] = {}
        self.slot: SlotSM | None = None
        self._op = asyncio.Lock()  # one operation at a time (see the module docstring)
        self._tokens = itertools.count(1)

    # ------------------------------------------------------------------ reads
    def state(self, tl: str) -> str:
        """Lifecycle state: active | cloned | joining | queued | joined | background | abandoned."""
        return lifecycle(self.timelines[tl].state())

    def phase(self, tl: str) -> str | None:
        """Within `active`: idle | turn | reporting. None in any other lifecycle state."""
        path = self.timelines[tl].state()
        return short(path) if lifecycle(path) == "active" else None

    def kind(self, tl: str) -> str:
        """original | focus | background: what tl was created as. It never changes."""
        return self.timelines[tl].kind

    def source(self, tl: str) -> str | None:
        return self.timelines[tl].source_id

    def holder(self) -> str | None:
        """The timeline holding the conversation."""
        return self.slot.holder if self.slot is not None else None

    def queued(self, source: str) -> list[QueuedJoin]:
        """The joins waiting on source, in finish order (a copy)."""
        return list(self.timelines[source].queue)

    def waiting(self) -> list[str]:
        """Sources that have joins waiting (whether or not they can be drained now)."""
        return [t for t, sm in self.timelines.items() if sm.queue]

    def running(self) -> list[str]:
        """Timelines holding the conversation (active or joining); background grafts excluded."""
        return [t for t in self.timelines if self.state(t) in RUNNING]

    def background(self, source: str | None = None) -> list[str]:
        """Background grafts still running (their join event not yet raised), optionally of one source."""
        return [t for t, sm in self.timelines.items()
                if self.state(t) == "background" and (source is None or sm.source_id == source)]

    def clone_refusal(self, source: str, grafts: list[tuple[str, bool]]) -> str | None:
        """Why source may not make this clone call (depth, the focus rule, the cap on live
        background grafts), or None. The same rule the clone transitions are guarded by, for a
        caller that wants the reason before it does any work. It says nothing about whether
        source is in a turn: only `clone` answers that."""
        return clone_refusal(self.timelines[source], grafts)

    def check(self) -> None:
        run = self.running()
        if len(run) != 1 or self.slot is None or run[0] != self.slot.holder:
            raise AssertionError(f"invariant violated: running={run} slot={self.slot and self.slot.holder}")

    # ------------------------------------------------------------------ primitives (lock held)
    async def _atomic(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one operation under the lock and to completion: if the calling task is cancelled
        while it runs, the operation still finishes, and only then is the cancellation raised."""
        async def locked() -> T:
            async with self._op:
                return await operation()

        task = asyncio.ensure_future(locked())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
            except Exception:
                break  # the operation's own error: raised below
        if cancelled:
            task.exception()  # retrieved; the cancellation wins
            raise asyncio.CancelledError
        return task.result()

    async def _accepts(self, tl: str, name: str, payload: Any = None) -> bool:
        """Dispatch an event to tl. True if a transition took it."""
        sm = self.timelines[tl]
        token = next(self._tokens)
        await hsm.Dispatch(None, sm, hsm.Event(name=name, data=Sent(token=token, payload=payload)))
        return sm.accepted == token

    async def _must(self, tl: str, name: str, payload: Any = None) -> None:
        if not await self._accepts(tl, name=name, payload=payload):
            where = self.timelines[tl].state().split("/", 2)[-1] or "stopped"
            raise TransitionRefused(f"{tl}: '{name}' refused in state {where}")

    async def _handoff(self, frm: str, to: str) -> None:
        assert self.slot is not None
        await hsm.Dispatch(None, self.slot, hsm.Event(name="handoff", data=(frm, to)))
        if self.slot.holder != to:
            raise TransitionRefused(f"slot handoff {frm}->{to} refused; holder={self.slot.holder}")

    # ------------------------------------------------------------------ operations
    async def start_root(self, tl: str) -> None:
        async def operation() -> None:
            if self.slot is not None:
                raise TransitionRefused("root already started")
            sm = TimelineSM(tl, source_id=None, kind="original")
            await hsm.Started(None, sm, TIMELINE, hsm.Config(ID=tl))
            self.timelines[tl] = sm
            self.slot = SlotSM(tl)
            await hsm.Started(None, self.slot, SLOT, hsm.Config(ID="slot"))
            self.check()
        await self._atomic(operation)

    async def begin_turn(self, tl: str) -> None:
        """tl starts answering a request: idle -> turn."""
        await self._atomic(lambda: self._must(tl, name="turn_begin"))

    async def begin_report(self, tl: str) -> None:
        """tl is woken only to report background grafts: idle -> reporting (no clone)."""
        await self._atomic(lambda: self._must(tl, name="report_begin"))

    async def end_turn(self, tl: str) -> None:
        """A turn or a report ends: busy -> idle."""
        await self._atomic(lambda: self._must(tl, name="turn_end"))

    async def try_end_turn(self, tl: str) -> bool:
        """End tl's turn if it is in one. For a caller that lost track because its run raised."""
        return await self._atomic(lambda: self._accepts(tl, name="turn_end"))

    async def clone(self, source: str, grafts: list[tuple[str, bool]]) -> None:
        """One clone call: `grafts` is [(timeline id, focus)]. The source must be the original, in
        a turn, with room for the background grafts. A focus graft takes the conversation
        (source turn->cloned, slot handoff). With background grafts only, the source stays in
        its turn and keeps the conversation. All of it or none: if a step fails, what was done
        is undone before the error leaves."""
        async def operation() -> None:
            ids = [g for g, _ in grafts]
            if any(g in self.timelines for g in ids) or len(set(ids)) != len(ids):
                raise TransitionRefused(f"graft ids must be new and distinct: {ids}")
            focus = next((g for g, f in grafts if f), None)
            event = "clone_commit" if focus is not None else "clone_background"
            if not await self._accepts(source, name=event, payload=list(grafts)):
                reason = clone_refusal(self.timelines[source], grafts)
                raise TransitionRefused(reason or f"{source} is not in a turn while holding the conversation")
            handed = False
            try:
                for child, is_focus in grafts:
                    sm = TimelineSM(child, source_id=source, kind="focus" if is_focus else "background")
                    await hsm.Started(None, sm, TIMELINE if is_focus else BACKGROUND, hsm.Config(ID=child))
                    self.timelines[child] = sm
                if focus is not None:
                    await self._handoff(source, focus)
                    handed = True
                self.check()
            except BaseException:
                await self._undo_clone(source, grafts, suspended=focus is not None, handed=handed)
                raise
        await self._atomic(operation)

    async def _undo_clone(self, source: str, grafts: list[tuple[str, bool]], suspended: bool, handed: bool) -> None:
        """Undo a clone call. The caller says how far it got; nothing is decided from a state.
        The grafts' machines never ran: they are dropped, not stopped."""
        focus = next((g for g, f in grafts if f), None)
        if handed and focus is not None:
            await self._handoff(focus, source)
        if suspended:
            await self._must(source, name="clone_abort")
        await self._must(source, name="grafts_dropped", payload=[g for g, _ in grafts])
        for child, _ in grafts:
            self.timelines.pop(child, None)
        self.check()

    async def unclone(self, source: str, grafts: list[tuple[str, bool]]) -> None:
        """Undo a completed `clone(source, grafts)` whose grafts could not be committed to the
        store, so the machine and the store never disagree about which grafts exist. Only valid
        right after that clone: none of the grafts has run."""
        focus = any(f for _, f in grafts)
        await self._atomic(lambda: self._undo_clone(source, grafts, suspended=focus, handed=focus))

    async def join_focus(self, child: str, reason: str, trigger: int | None
                         ) -> tuple[str, list[QueuedJoin]] | None:
        """A focus graft's join event. Accepted: the graft is `queued` (frozen, not final), its
        join is last on its source's queue, the source is `joining` and holds the conversation
        again; returns (source, the queue to apply, in order). Refused (None): the timeline is
        the original, or has already raised its join event."""
        async def operation() -> tuple[str, list[QueuedJoin]] | None:
            if not await self._accepts(child, name="join"):
                return None
            src = self.timelines[child].source_id
            await self._must(src, name="focus_joined", payload=QueuedJoin(child=child, reason=reason, trigger=trigger))
            await self._handoff(child, src)
            self.check()
            return src, list(self.timelines[src].queue)
        return await self._atomic(operation)

    async def finish_background(self, child: str, reason: str) -> str | None:
        """A background graft's join event. Accepted: the graft is `queued` and its join is on
        its source's queue; returns the source. Refused (None): it already raised its join
        event. Nothing else moves."""
        async def operation() -> str | None:
            if not await self._accepts(child, name="finish"):
                return None
            src = self.timelines[child].source_id
            await self._must(src, name="graft_queued", payload=QueuedJoin(child=child, reason=reason, trigger=None))
            self.check()
            return src
        return await self._atomic(operation)

    async def abandon(self, child: str) -> bool:
        """A background graft whose join event cannot be raised ends in `abandoned` (final) and
        its source forgets it. False if it is not a running background graft."""
        async def operation() -> bool:
            if not await self._accepts(child, name="abandon"):
                return False
            await self._must(self.timelines[child].source_id, name="grafts_dropped", payload=[child])
            self.check()
            return True
        return await self._atomic(operation)

    async def try_begin_drain(self, source: str) -> list[QueuedJoin] | None:
        """Ask to start applying the joins waiting on source: idle->joining. Returns the queue
        (finish order), or None if the graph refuses: source is mid-turn, suspended, or has
        nothing queued."""
        async def operation() -> list[QueuedJoin] | None:
            if not await self._accepts(source, name="drain_begin"):
                return None
            self.check()
            return list(self.timelines[source].queue)
        return await self._atomic(operation)

    async def commit_queued(self, child: str, committed: bool) -> None:
        """The join at the head of its source's queue was applied to the source. The source
        (which must be `joining`) takes `join_applied` and drops the entry; the graft goes
        queued->joined. Refused if the source is not joining or child is not the head."""
        async def operation() -> None:
            src = self.timelines[child].source_id
            if src is None:
                raise TransitionRefused(f"{child} has no source")
            await self._must(src, name="join_applied", payload=child)
            await self._must(child, name="join_commit" if committed else "join_rollback")
            self.check()
        await self._atomic(operation)

    async def abandon_queued(self, child: str) -> None:
        """The join at the head of its source's queue cannot be applied and is given up: the
        source (which must be `joining`) drops the entry, and the graft ends `abandoned`. The
        joins behind it can then be applied."""
        async def operation() -> None:
            src = self.timelines[child].source_id
            if src is None:
                raise TransitionRefused(f"{child} has no source")
            await self._must(src, name="join_applied", payload=child)
            await self._must(child, name="join_abandon")
            self.check()
        await self._atomic(operation)

    def has(self, tl: str) -> bool:
        """Does the machine know this timeline? For a caller whose clone was interrupted and
        that must find out whether the step was taken before it undoes it."""
        return tl in self.timelines

    async def finish_join(self, source: str, committed: bool) -> None:
        """The joins source was applying are done: joining->active (idle). `committed` is the
        outcome: False if any of them rolled back or could not be applied."""
        async def operation() -> None:
            await self._must(source, name="join_commit" if committed else "join_rollback")
            self.check()
        await self._atomic(operation)

    async def stop(self) -> None:
        async def operation() -> None:
            for sm in self.timelines.values():
                await hsm.Stop(sm)
            if self.slot is not None:
                await hsm.Stop(self.slot)
        await self._atomic(operation)
