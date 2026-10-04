"""State machines (stateforward.hsm) for timelines and the single running slot.

Per-timeline lifecycle (identical for every timeline; the original is just a
timeline with no source):

    active --clone_commit--> cloned --child_joined--> joining --join_commit|join_rollback--> active
    active --join [guard: has source]--> joined (frozen; no outgoing transitions)

Global invariant: exactly one timeline is running (active or joining) at any
depth. A single global `Slot` machine holds the running timeline id and only
accepts a handoff from the current holder (guarded internal transition).
`Machines` drives both and checks the invariant after every operation.
"""

from __future__ import annotations

import hsm

RUNNING = ("active", "joining")


class TransitionRefused(Exception):
    pass


class TimelineSM(hsm.Instance):
    def __init__(self, timeline_id: str, source_id: str | None):
        super().__init__()
        self.timeline_id = timeline_id
        self.source_id = source_id


def _has_source(ctx, inst: TimelineSM, event) -> bool:
    return inst.source_id is not None


TIMELINE = hsm.Define(
    "Timeline",
    hsm.Initial(hsm.Target("active")),
    hsm.State(
        "active",
        hsm.Transition(hsm.On("clone_commit"), hsm.Target("../cloned")),
        hsm.Transition(hsm.On("join"), hsm.Target("../joined"), hsm.Guard(_has_source)),
    ),
    hsm.State("cloned", hsm.Transition(hsm.On("child_joined"), hsm.Target("../joining"))),
    hsm.State(
        "joining",
        hsm.Transition(hsm.On("join_commit"), hsm.Target("../active")),
        hsm.Transition(hsm.On("join_rollback"), hsm.Target("../active")),
    ),
    hsm.State("joined"),
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


def short(state_path: str) -> str:
    return state_path.rsplit("/", 1)[-1]


class Machines:
    def __init__(self):
        self.timelines: dict[str, TimelineSM] = {}
        self.slot: SlotSM | None = None

    def state(self, tl: str) -> str:
        return short(self.timelines[tl].state())

    async def _dispatch(self, tl: str, name: str, expect: str) -> None:
        before = self.state(tl)
        await hsm.Dispatch(None, self.timelines[tl], hsm.Event(name=name))
        after = self.state(tl)
        if after != expect or after == before:
            raise TransitionRefused(f"{tl}: '{name}' refused in state {before} (now {after})")

    async def _handoff(self, frm: str, to: str) -> None:
        assert self.slot is not None
        if self.slot.holder != frm:
            raise TransitionRefused(f"slot handoff {frm}->{to} refused; holder={self.slot.holder}")
        await hsm.Dispatch(None, self.slot, hsm.Event(name="handoff", data=(frm, to)))
        if self.slot.holder != to:
            raise TransitionRefused(f"slot handoff {frm}->{to} refused; holder={self.slot.holder}")

    async def start_root(self, tl: str) -> None:
        if self.slot is not None:
            raise TransitionRefused("root already started")
        sm = TimelineSM(tl, None)
        await hsm.Started(None, sm, TIMELINE, hsm.Config(ID=tl))
        self.timelines[tl] = sm
        self.slot = SlotSM(tl)
        await hsm.Started(None, self.slot, SLOT, hsm.Config(ID="slot"))
        self.check()

    async def clone(self, source: str, child: str) -> None:
        if self.slot is None or self.slot.holder != source or self.state(source) != "active":
            raise TransitionRefused(f"{source} is not the running active timeline")
        if any(self.timelines[t].source_id == source and self.state(t) != "joined"
               for t in self.timelines):
            raise TransitionRefused(f"{source} already has a live clone")
        await self._dispatch(source, "clone_commit", "cloned")
        sm = TimelineSM(child, source)
        await hsm.Started(None, sm, TIMELINE, hsm.Config(ID=child))
        self.timelines[child] = sm
        await self._handoff(source, child)
        self.check()

    async def join(self, child: str) -> str:
        """Deterministic join event: clone active->joined, source cloned->joining."""
        src = self.timelines[child].source_id
        await self._dispatch(child, "join", "joined")
        if src is None:
            raise TransitionRefused("original cannot join")
        await self._dispatch(src, "child_joined", "joining")
        await self._handoff(child, src)
        self.check()
        return src

    async def finish_join(self, source: str, committed: bool) -> None:
        await self._dispatch(source, "join_commit" if committed else "join_rollback", "active")
        self.check()

    def running(self) -> list[str]:
        return [t for t in self.timelines if self.state(t) in RUNNING]

    def check(self) -> None:
        run = self.running()
        if len(run) != 1 or self.slot is None or run[0] != self.slot.holder:
            raise AssertionError(f"invariant violated: running={run} slot={self.slot and self.slot.holder}")

    async def stop(self) -> None:
        for sm in self.timelines.values():
            await hsm.Stop(sm)
        if self.slot is not None:
            await hsm.Stop(self.slot)
