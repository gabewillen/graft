"""State machines (stateforward.hsm) for timelines and the single conversation slot.

A timeline that can hold the conversation (the original, and a focus graft):

    active                                   holds the conversation
      idle        at rest: no model call in flight, nothing half-written
      busy
        turn      answering a request (a user message, or one a clone handed back): may clone
        reporting woken only to tell the user what background grafts did: clone is refused
    cloned        suspended while its focus graft holds the conversation
    joining       committing joins (a focus graft's, and/or queued background joins)
    joined        final: frozen

    idle  --turn_begin-->   turn            idle --report_begin--> reporting
    busy  --turn_end-->     idle            (one transition, on the parent of turn and reporting)
    turn  --clone_commit--> cloned          (the call has a focus graft)
    turn  --clone_background--> turn        (internal: background grafts only; the source keeps the conversation)
    idle  --drain_begin [queued joins, slot holder]--> joining
    cloned --focus_joined--> joining
    joining --join_commit | join_rollback--> active (idle)
    active --join [has a source]--> joined

A background graft (it runs concurrently and never holds the conversation):

    background --finish--> queued --join_commit | join_rollback--> joined (final)

`queued` means the graft is frozen and its join event is waiting on its source; `joined` means
the join was applied to the source (committed, or rolled back to a record). The queue is the
source's: finish order, kept on the source's machine instance, and an entry leaves it only when
its join has been applied.

Global invariant: at most one branch holds the conversation. Exactly one timeline is in `active`
or `joining`, and it is the holder of the single global `Slot` machine, which only accepts a
handoff from the current holder. Background grafts are outside the slot: creating one is not a
handoff and their joins never move it.

The machine is the authority for what may happen: a queued-join commit is refused mid-turn, when
nothing is queued, or from a timeline that does not hold the conversation; a clone is refused
outside `turn`. Callers ask (`try_begin_drain`) or are refused (`TransitionRefused`); they do not
read states to decide.

Every `Machines` operation is one critical section. Each takes `Machines._op`, so two
operations never interleave even if a dispatch yields to the event loop, and the invariant is
checked before the lock is released.
"""

from __future__ import annotations

import asyncio

import hsm

RUNNING = ("active", "joining")  # lifecycle states that hold the conversation
MAX_LIVE_BACKGROUND = 4  # live (running or queued) background grafts per source


class TransitionRefused(Exception):
    pass


class TimelineSM(hsm.Instance):
    def __init__(self, timeline_id: str, source_id: str | None):
        super().__init__()
        self.timeline_id = timeline_id
        self.source_id = source_id
        self.queue: list[str] = []  # this timeline's background grafts with a join waiting, finish order
        self.is_holder = source_id is None  # mirrors the slot, for guards
        self.background_clone_calls = 0  # clone_background transitions taken


def _has_source(ctx, inst: TimelineSM, event) -> bool:
    return inst.source_id is not None


def _can_drain(ctx, inst: TimelineSM, event) -> bool:
    return bool(inst.queue) and inst.is_holder


def _count_background_clone(ctx, inst: TimelineSM, event) -> None:
    inst.background_clone_calls += 1


TIMELINE = hsm.Define(
    "Timeline",
    hsm.Initial(hsm.Target("active")),
    hsm.State(
        "active",
        hsm.Initial(hsm.Target("idle")),
        hsm.Transition(hsm.On("join"), hsm.Target("../joined"), hsm.Guard(_has_source)),
        hsm.State(
            "idle",
            hsm.Transition(hsm.On("turn_begin"), hsm.Target("../busy/turn")),
            hsm.Transition(hsm.On("report_begin"), hsm.Target("../busy/reporting")),
            hsm.Transition(hsm.On("drain_begin"), hsm.Target("../../joining"), hsm.Guard(_can_drain)),
        ),
        hsm.State(
            "busy",
            hsm.Transition(hsm.On("turn_end"), hsm.Target("../idle")),
            hsm.State(
                "turn",
                hsm.Transition(hsm.On("clone_commit"), hsm.Target("../../../cloned")),
                hsm.Transition(hsm.On("clone_background"), hsm.Effect(_count_background_clone)),
            ),
            hsm.State("reporting"),
        ),
    ),
    hsm.State("cloned", hsm.Transition(hsm.On("focus_joined"), hsm.Target("../joining"))),
    hsm.State(
        "joining",
        hsm.Transition(hsm.On("join_commit"), hsm.Target("../active")),
        hsm.Transition(hsm.On("join_rollback"), hsm.Target("../active")),
    ),
    hsm.Final("joined"),
)

BACKGROUND = hsm.Define(
    "BackgroundTimeline",
    hsm.Initial(hsm.Target("background")),
    hsm.State("background", hsm.Transition(hsm.On("finish"), hsm.Target("../queued"))),
    hsm.State(
        "queued",
        hsm.Transition(hsm.On("join_commit"), hsm.Target("../joined")),
        hsm.Transition(hsm.On("join_rollback"), hsm.Target("../joined")),
    ),
    hsm.Final("joined"),
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
    """'/Timeline/active/busy/turn' -> 'active': the top-level state, the one mirrored to the store.
    A stopped machine has no state: 'stopped'."""
    parts = state_path.split("/")
    return parts[2] if len(parts) > 2 else "stopped"


def short(state_path: str) -> str:
    return state_path.rsplit("/", 1)[-1]


class Machines:
    def __init__(self):
        self.timelines: dict[str, TimelineSM] = {}
        self.slot: SlotSM | None = None
        self._op = asyncio.Lock()  # one operation at a time (see the module docstring)

    # ------------------------------------------------------------------ reads
    def state(self, tl: str) -> str:
        """Lifecycle state: active | cloned | joining | joined | background | queued."""
        return lifecycle(self.timelines[tl].state())

    def phase(self, tl: str) -> str | None:
        """Within `active`: idle | turn | reporting. None in any other lifecycle state."""
        path = self.timelines[tl].state()
        return short(path) if lifecycle(path) == "active" else None

    def queued(self, source: str) -> list[str]:
        """source's background grafts whose join is waiting, in finish order (a copy)."""
        return list(self.timelines[source].queue)

    def waiting(self) -> list[str]:
        """Sources that have queued background joins (whether or not they can be drained now)."""
        return [t for t, sm in self.timelines.items() if sm.queue]

    def running(self) -> list[str]:
        """Timelines holding the conversation (active or joining); background grafts excluded."""
        return [t for t in self.timelines if self.state(t) in RUNNING]

    def background(self, source: str | None = None) -> list[str]:
        """Background grafts still running (not yet finished), optionally of one source."""
        return [t for t, sm in self.timelines.items()
                if self.state(t) == "background" and (source is None or sm.source_id == source)]

    def live_background(self, source: str) -> list[str]:
        """source's background grafts that are running or have a join waiting."""
        return [t for t, sm in self.timelines.items()
                if sm.source_id == source and self.state(t) in ("background", "queued")]

    def check(self) -> None:
        run = self.running()
        if len(run) != 1 or self.slot is None or run[0] != self.slot.holder:
            raise AssertionError(f"invariant violated: running={run} slot={self.slot and self.slot.holder}")
        holders = [t for t, sm in self.timelines.items() if sm.is_holder]
        if holders != [self.slot.holder]:
            raise AssertionError(f"invariant violated: holder flags {holders} slot={self.slot.holder}")

    # ------------------------------------------------------------------ primitives (lock held)
    async def _send(self, tl: str, name: str) -> tuple[str, str]:
        sm = self.timelines[tl]
        before = sm.state()
        await hsm.Dispatch(None, sm, hsm.Event(name=name))
        return before, sm.state()

    async def _dispatch(self, tl: str, name: str, expect: str) -> None:
        """Dispatch an event that must move tl to the state named `expect` (its last path
        segment); anything else is a refusal."""
        before, after = await self._send(tl, name)
        if after == before or short(after) != expect:
            raise TransitionRefused(f"{tl}: '{name}' refused in state {before[1:].split('/', 1)[1]} "
                                    f"(now {after[1:].split('/', 1)[1]})")

    async def _handoff(self, frm: str, to: str) -> None:
        assert self.slot is not None
        if self.slot.holder != frm:
            raise TransitionRefused(f"slot handoff {frm}->{to} refused; holder={self.slot.holder}")
        await hsm.Dispatch(None, self.slot, hsm.Event(name="handoff", data=(frm, to)))
        if self.slot.holder != to:
            raise TransitionRefused(f"slot handoff {frm}->{to} refused; holder={self.slot.holder}")
        self.timelines[frm].is_holder, self.timelines[to].is_holder = False, True

    def _refuse_clone(self, source: str, grafts: list[tuple[str, bool]]) -> str | None:
        if self.slot is None or self.slot.holder != source or self.phase(source) != "turn":
            return f"{source} is not in a turn while holding the conversation"
        if not grafts:
            return "a clone call needs at least one graft"
        focus = [g for g, f in grafts if f]
        if len(focus) > 1:
            return f"at most one graft may take the conversation, got {focus}"
        if any(g in self.timelines for g, _ in grafts) or len({g for g, _ in grafts}) != len(grafts):
            return f"graft ids must be new and distinct: {[g for g, _ in grafts]}"
        live = len(self.live_background(source)) + sum(1 for _, f in grafts if not f)
        if live > MAX_LIVE_BACKGROUND:
            return f"{source} would have {live} live background grafts (the maximum is {MAX_LIVE_BACKGROUND})"
        return None

    # ------------------------------------------------------------------ operations
    async def start_root(self, tl: str) -> None:
        async with self._op:
            if self.slot is not None:
                raise TransitionRefused("root already started")
            sm = TimelineSM(tl, None)
            await hsm.Started(None, sm, TIMELINE, hsm.Config(ID=tl))
            self.timelines[tl] = sm
            self.slot = SlotSM(tl)
            await hsm.Started(None, self.slot, SLOT, hsm.Config(ID="slot"))
            self.check()

    async def begin_turn(self, tl: str) -> None:
        """tl starts answering a request: idle -> turn."""
        async with self._op:
            await self._dispatch(tl, "turn_begin", "turn")

    async def begin_report(self, tl: str) -> None:
        """tl is woken only to report background grafts: idle -> reporting (no clone)."""
        async with self._op:
            await self._dispatch(tl, "report_begin", "reporting")

    async def end_turn(self, tl: str) -> None:
        """A turn or a report ends: busy -> idle."""
        async with self._op:
            await self._dispatch(tl, "turn_end", "idle")

    async def try_end_turn(self, tl: str) -> bool:
        """End tl's turn if it is in one. For a caller that lost track because its run raised."""
        async with self._op:
            before, after = await self._send(tl, "turn_end")
            return after != before

    def clone_refusal(self, source: str, grafts: list[tuple[str, bool]]) -> str | None:
        """Why `clone(source, grafts)` would be refused now, or None. Lets a caller refuse before
        it writes anything it could not undo."""
        return self._refuse_clone(source, grafts)

    async def clone(self, source: str, grafts: list[tuple[str, bool]]) -> None:
        """One clone call: `grafts` is [(timeline id, focus)]. The source must be in a turn. A
        focus graft takes the conversation (source turn->cloned, slot handoff). With background
        grafts only, the source stays in its turn and keeps the conversation."""
        async with self._op:
            refusal = self._refuse_clone(source, grafts)
            if refusal:
                raise TransitionRefused(refusal)
            focus = next((g for g, f in grafts if f), None)
            if focus is None:
                taken = self.timelines[source].background_clone_calls
                await self._send(source, "clone_background")
                if self.timelines[source].background_clone_calls != taken + 1:
                    raise TransitionRefused(f"{source}: 'clone_background' refused")
            else:
                await self._dispatch(source, "clone_commit", "cloned")
            for child, is_focus in grafts:
                sm = TimelineSM(child, source)
                await hsm.Started(None, sm, TIMELINE if is_focus else BACKGROUND, hsm.Config(ID=child))
                self.timelines[child] = sm
            if focus is not None:
                await self._handoff(source, focus)
            self.check()

    async def join_focus(self, child: str) -> str:
        """A focus graft's join event: active->joined, its source cloned->joining, and the slot
        returns to the source. `finish_join` completes it."""
        async with self._op:
            src = self.timelines[child].source_id
            if src is None:
                raise TransitionRefused("the original cannot join")
            await self._dispatch(child, "join", "joined")
            await self._dispatch(src, "focus_joined", "joining")
            await self._handoff(child, src)
            self.check()
            return src

    async def finish_background(self, child: str) -> str:
        """A background graft's join event: background->queued, and it is appended to its
        source's queue. Nothing else moves."""
        async with self._op:
            await self._dispatch(child, "finish", "queued")
            src = self.timelines[child].source_id
            self.timelines[src].queue.append(child)
            self.check()
            return src

    async def try_begin_drain(self, source: str) -> list[str] | None:
        """Ask to start committing source's queued background joins: idle->joining. Returns the
        queue (finish order), or None if the machine refuses: source is mid-turn, suspended, not
        the conversation holder, or has nothing queued."""
        async with self._op:
            before, after = await self._send(source, "drain_begin")
            if after == before:
                return None
            self.check()
            return list(self.timelines[source].queue)

    async def begin_drain(self, source: str) -> list[str]:
        """`try_begin_drain`, refused loudly."""
        queue = await self.try_begin_drain(source)
        if queue is None:
            raise TransitionRefused(f"{source}: 'drain_begin' refused (not at rest holding the "
                                    "conversation, or nothing is queued)")
        return queue

    async def commit_queued(self, child: str, committed: bool) -> None:
        """A queued background join was applied to its source (which must be `joining`):
        queued->joined, and only now does it leave the queue."""
        async with self._op:
            src = self.timelines[child].source_id
            if src is None or self.state(src) != "joining" or child not in self.timelines[src].queue:
                raise TransitionRefused(f"{child} is not queued on a joining source")
            await self._dispatch(child, "join_commit" if committed else "join_rollback", "joined")
            self.timelines[src].queue.remove(child)
            self.check()

    async def finish_join(self, source: str, committed: bool) -> None:
        """The joins source was committing are done: joining->active (idle). `committed` is the
        outcome: False if any of them rolled back or failed."""
        async with self._op:
            await self._dispatch(source, "join_commit" if committed else "join_rollback", "idle")
            self.check()

    async def stop(self) -> None:
        async with self._op:
            for sm in self.timelines.values():
                await hsm.Stop(sm)
            if self.slot is not None:
                await hsm.Stop(self.slot)
