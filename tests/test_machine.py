"""The state machines through their public surface: operations, and the reads `state`, `phase`,
`holder`, `queued`, `running`, `background`, `live_background`."""

import asyncio

import hsm
import pytest

from graft.machine import MAX_LIVE_BACKGROUND, Machines, QueuedJoin, TransitionRefused


def run(coro):
    return asyncio.run(coro)


async def root_in_turn():
    m = Machines()
    await m.start_root("o")
    await m.begin_turn("o")
    return m


def test_lifecycle_and_single_running_invariant():
    async def go():
        m = Machines()
        await m.start_root("o")
        assert (m.state("o"), m.phase("o"), m.kind("o")) == ("active", "idle", "original")
        assert m.running() == ["o"] and m.holder() == "o"
        await m.begin_turn("o")
        assert (m.state("o"), m.phase("o")) == ("active", "turn")
        await m.clone("o", [("c1", True)])
        assert (m.state("o"), m.state("c1"), m.phase("c1"), m.kind("c1")) == ("cloned", "active", "idle", "focus")
        assert m.phase("o") is None and m.running() == ["c1"] and m.holder() == "c1"
        src, queue = await m.join_focus("c1", reason="done", trigger=None)
        assert src == "o" and queue == [QueuedJoin("c1", "done", None)]
        assert (m.state("o"), m.state("c1")) == ("joining", "queued")  # frozen, but not joined yet
        assert m.running() == ["o"] and m.holder() == "o"
        await m.commit_queued("c1", committed=True)
        assert m.state("c1") == "joined" and m.queued("o") == []
        await m.finish_join("o", committed=True)
        assert (m.state("o"), m.phase("o")) == ("active", "idle")
        await m.begin_turn("o")
        await m.clone("o", [("c2", True)])
        await m.begin_turn("c2")
        await m.join_focus("c2", reason="out_of_scope", trigger=7)  # a focus graft joins from inside its turn too
        assert m.queued("o") == [QueuedJoin("c2", "out_of_scope", 7)]  # the queue carries reason and trigger
        await m.commit_queued("c2", committed=False)
        await m.finish_join("o", committed=False)  # rollback also returns to rest
        assert (m.state("o"), m.phase("o"), m.state("c2")) == ("active", "idle", "joined")
        await m.stop()
        assert m.state("o") == "stopped"
    run(go())


def test_turn_and_report_phases():
    async def go():
        m = Machines()
        await m.start_root("o")
        with pytest.raises(TransitionRefused):
            await m.end_turn("o")  # not in a turn
        assert await m.try_end_turn("o") is False
        await m.begin_turn("o")
        with pytest.raises(TransitionRefused):
            await m.begin_turn("o")  # one turn at a time
        with pytest.raises(TransitionRefused):
            await m.begin_report("o")
        await m.end_turn("o")
        await m.begin_report("o")
        assert m.phase("o") == "reporting"
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c1", True)])  # a report may not clone
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c1", False)])
        assert list(m.timelines) == ["o"] and m.live_background("o") == []
        assert await m.try_end_turn("o") is True and m.phase("o") == "idle"
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c1", False)])  # nor may a timeline at rest: a clone belongs to a turn
        await m.stop()
    run(go())


def test_refusals():
    async def go():
        m = await root_in_turn()
        assert await m.join_focus("o", reason="done", trigger=None) is None  # the original has no source
        assert (m.state("o"), m.phase("o")) == ("active", "turn")
        await m.clone("o", [("c1", True)])
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c9", True)])  # a suspended source cannot clone
        with pytest.raises(TransitionRefused):
            await m.finish_join("o", committed=True)  # cloned -> active only via joining
        with pytest.raises(TransitionRefused):
            await m.begin_turn("o")  # suspended: no turn
        await m.join_focus("c1", reason="done", trigger=None)
        assert await m.join_focus("c1", reason="failed", trigger=None) is None  # a second join event
        assert m.queued("o") == [QueuedJoin("c1", "done", None)]  # the first one stands
        with pytest.raises(TransitionRefused):
            await m.begin_turn("c1")  # frozen
        await m.commit_queued("c1", committed=True)
        assert await m.join_focus("c1", reason="failed", trigger=None) is None  # joined is final
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c1", committed=True)
        m.check()
        await m.stop()
    run(go())


def test_only_the_holder_of_the_conversation_moves_it():
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", True), ("c2", False)])
        assert m.holder() == "c1" and m.running() == ["c1"]
        await m.finish_background("c2", reason="done")  # a background join never moves the conversation
        assert m.holder() == "c1" and m.running() == ["c1"]
        with pytest.raises(TransitionRefused):
            await m.begin_turn("o")  # the suspended source cannot act
        await m.join_focus("c1", reason="done", trigger=None)
        assert m.holder() == "o" and m.running() == ["o"]
        await m.stop()
    run(go())


def test_background_grafts_are_outside_the_slot():
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", False), ("c2", False)])  # not a handoff: the source keeps its turn
        assert (m.state("o"), m.phase("o"), m.state("c1"), m.state("c2")) == \
            ("active", "turn", "background", "background")
        assert m.running() == ["o"] and m.holder() == "o" and m.background() == ["c1", "c2"]
        assert m.phase("c1") is None and m.kind("c1") == "background" and m.source("c1") == "o"
        await m.clone("o", [("c3", True), ("c4", False)])  # live background grafts do not block a clone call
        assert (m.state("o"), m.state("c3"), m.state("c4")) == ("cloned", "active", "background")
        assert m.running() == ["c3"] and m.holder() == "c3"
        assert m.background("o") == ["c1", "c2", "c4"] and m.background("c3") == []
        assert await m.finish_background("c1", reason="done") == "o"  # frozen and queued; nothing else moves
        assert (m.state("c1"), m.state("o"), m.holder()) == ("queued", "cloned", "c3")
        assert m.queued("o") == [QueuedJoin("c1", "done", None)] and m.live_background("o") == ["c1", "c2", "c4"]
        assert await m.try_begin_drain("o") is None  # a suspended source is not at rest
        _, queue = await m.join_focus("c3", reason="done", trigger=None)
        assert [q.child for q in queue] == ["c1", "c3"]  # the background join first, the focus join last
        await m.commit_queued("c1", committed=True)  # queued joins are applied while the source is joining
        await m.commit_queued("c3", committed=True)
        assert m.state("c1") == "joined" and m.queued("o") == [] and m.live_background("o") == ["c2", "c4"]
        await m.finish_join("o", committed=True)
        await m.finish_background("c2", reason="failed")
        await m.finish_background("c4", reason="done")
        queue = await m.try_begin_drain("o")  # at rest: idle -> joining, finish order
        assert queue == [QueuedJoin("c2", "failed", None), QueuedJoin("c4", "done", None)]
        assert m.state("o") == "joining" and m.running() == ["o"]
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c4", committed=True)  # not the head of the queue: order is kept
        await m.commit_queued("c2", committed=True)
        await m.commit_queued("c4", committed=False)  # a rolled-back join also leaves the queue
        await m.finish_join("o", committed=False)
        assert m.background() == [] and m.queued("o") == [] and m.running() == ["o"]
        assert (m.state("c2"), m.state("c4")) == ("joined", "joined") and m.live_background("o") == []
        await m.stop()
    run(go())


def test_drain_is_refused_unless_at_rest_with_something_queued():
    async def go():
        m = Machines()
        await m.start_root("o")
        assert await m.try_begin_drain("o") is None  # nothing queued
        with pytest.raises(TransitionRefused):
            await m.begin_drain("o")
        await m.begin_turn("o")
        await m.clone("o", [("c1", False), ("c2", False)])
        await m.finish_background("c1", reason="done")
        assert await m.try_begin_drain("o") is None  # mid-turn
        await m.end_turn("o")
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c1", committed=True)  # the source is not joining
        assert [q.child for q in await m.begin_drain("o")] == ["c1"]
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c2", committed=True)  # still running: not queued
        await m.finish_join("o", committed=False)  # begin, then rollback: nothing was applied
        assert (m.state("o"), m.phase("o"), m.state("c1")) == ("active", "idle", "queued")
        assert [q.child for q in await m.begin_drain("o")] == ["c1"]  # the join was not lost: it is offered again
        await m.commit_queued("c1", committed=True)
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c1", committed=True)  # a second join on a joined background graft
        assert await m.finish_background("c1", reason="failed") is None
        await m.finish_join("o", committed=True)
        m.check()
        await m.stop()
    run(go())


def test_drain_is_refused_from_a_timeline_that_does_not_hold_the_conversation():
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", True), ("c2", False)])
        await m.finish_background("c2", reason="done")
        assert [q.child for q in m.queued("o")] == ["c2"] and m.holder() == "c1" and m.waiting() == ["o"]
        assert await m.try_begin_drain("o") is None  # queued on o, but c1 holds the conversation
        assert await m.try_begin_drain("c1") is None  # the holder has nothing queued
        with pytest.raises(TransitionRefused):
            await m.begin_drain("o")
        await m.stop()
    run(go())


def test_clone_call_refusals():
    async def go():
        m = await root_in_turn()
        with pytest.raises(TransitionRefused, match="at most one graft"):
            await m.clone("o", [("c1", True), ("c2", True)])
        with pytest.raises(TransitionRefused):
            await m.clone("o", [])
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c1", False), ("c1", False)])
        too_many = [(f"c{i}", False) for i in range(MAX_LIVE_BACKGROUND + 1)]
        assert "background clones may be live" in m.clone_refusal("o", too_many)
        with pytest.raises(TransitionRefused, match="background clones may be live"):
            await m.clone("o", too_many)  # the cap on live background grafts per source
        assert list(m.timelines) == ["o"] and m.phase("o") == "turn"  # refused calls created nothing
        await m.clone("o", too_many[:-1])
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c9", False)])  # still at the cap: queued grafts count until they join
        assert m.clone_refusal("o", [("c9", True)]) is None  # a focus graft is not capped
        await m.stop()
    run(go())


def test_a_graft_cannot_clone():
    """Grafts are one level deep: the graph refuses a clone from a focus graft in a turn, and a
    background graft has no clone transition at all."""
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", True), ("c2", False)])
        await m.begin_turn("c1")
        assert (m.state("c1"), m.phase("c1"), m.holder()) == ("active", "turn", "c1")
        for grafts in ([("c5", True)], [("c5", False)]):
            assert "only the original clones" in m.clone_refusal("c1", grafts)
            with pytest.raises(TransitionRefused, match="only the original clones"):
                await m.clone("c1", grafts)
            with pytest.raises(TransitionRefused):
                await m.clone("c2", grafts)
        assert list(m.timelines) == ["o", "c1", "c2"] and m.phase("c1") == "turn"
        await m.stop()
    run(go())


def test_a_timeline_with_grafts_cannot_join():
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", True)])
        m.timelines["c1"].grafts["x"] = False  # a state the graph never produces (depth 1): forced here
        assert await m.join_focus("c1", reason="done", trigger=None) is None  # refused by the join guard
        assert (m.state("c1"), m.state("o"), m.holder()) == ("active", "cloned", "c1")
        del m.timelines["c1"].grafts["x"]
        assert await m.join_focus("c1", reason="done", trigger=None) is not None
        await m.stop()
    run(go())


def test_abandon_is_a_terminal_state_for_a_running_background_graft():
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", False), ("c2", False)])
        assert await m.abandon("c1") is True
        assert m.state("c1") == "abandoned" and m.live_background("o") == ["c2"] and m.background() == ["c2"]
        assert await m.abandon("c1") is False and await m.finish_background("c1", reason="failed") is None
        await m.finish_background("c2", reason="done")
        assert await m.abandon("c2") is False  # it raised its join event: it joins, it is not abandoned
        assert await m.abandon("o") is False
        m.check()
        await m.stop()
    run(go())


@pytest.mark.parametrize("grafts", [[("c1", True), ("c2", False)], [("c1", False), ("c2", False)]])
def test_a_clone_that_fails_halfway_is_undone(monkeypatch, grafts):
    async def go():
        m = await root_in_turn()
        started = hsm.Started

        async def fail_on_c2(ctx, sm, model, config):
            if getattr(sm, "timeline_id", None) == "c2":
                raise RuntimeError("injected")
            return await started(ctx, sm, model, config)

        monkeypatch.setattr(hsm, "Started", fail_on_c2)
        with pytest.raises(RuntimeError, match="injected"):
            await m.clone("o", grafts)
        assert list(m.timelines) == ["o"] and m.holder() == "o"  # no graft is left behind
        assert (m.state("o"), m.phase("o")) == ("active", "turn")  # the source is back in its turn
        assert m.live_background("o") == []
        m.check()
        monkeypatch.setattr(hsm, "Started", started)
        await m.clone("o", grafts)  # and can clone again
        assert list(m.timelines) == ["o", "c1", "c2"]
        await m.stop()
    run(go())


def test_unclone_restores_the_source_and_forgets_the_grafts():
    async def go():
        m = await root_in_turn()
        grafts = [("c1", True), ("c2", False)]
        await m.clone("o", grafts)
        assert (m.state("o"), m.holder()) == ("cloned", "c1")
        await m.unclone("o", grafts)  # the store commit failed: the clone call never happened
        assert list(m.timelines) == ["o"] and m.holder() == "o" and m.phase("o") == "turn"
        assert m.live_background("o") == []
        await m.end_turn("o")
        m.check()
        await m.stop()
    run(go())


def test_an_operation_cancelled_midway_still_completes(monkeypatch):
    """A cancellation that arrives while an undo is between its dispatches does not leave a
    phantom graft counting against the live cap: the operation finishes, then the cancellation
    is raised."""
    async def go():
        m = await root_in_turn()
        grafts = [("c1", True), ("c2", False)]
        await m.clone("o", grafts)
        dispatch = hsm.Dispatch
        gate = asyncio.Event()
        reached = asyncio.Event()

        async def slow(ctx, sm, event):
            if event.name == "clone_abort":
                reached.set()
                await gate.wait()  # the undo is suspended between two of its steps
            return await dispatch(ctx, sm, event)

        monkeypatch.setattr(hsm, "Dispatch", slow)
        undo = asyncio.create_task(m.unclone("o", grafts))
        await reached.wait()
        undo.cancel()
        await asyncio.sleep(0)
        assert not undo.done()  # the cancellation waits for the operation
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await undo
        monkeypatch.setattr(hsm, "Dispatch", dispatch)
        assert list(m.timelines) == ["o"] and m.holder() == "o" and m.phase("o") == "turn"
        assert m.live_background("o") == []  # nothing counts against the cap
        m.check()
        await m.clone("o", [(f"g{i}", False) for i in range(MAX_LIVE_BACKGROUND)])
        await m.stop()
    run(go())


def test_a_queued_join_that_is_given_up_ends_abandoned_and_frees_the_queue():
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", False), ("c2", False)])
        await m.finish_background("c1", reason="done")
        await m.finish_background("c2", reason="done")
        with pytest.raises(TransitionRefused):
            await m.abandon_queued("c1")  # only while the source is applying its queue
        await m.end_turn("o")
        assert [q.child for q in await m.try_begin_drain("o")] == ["c1", "c2"]
        with pytest.raises(TransitionRefused):
            await m.abandon_queued("c2")  # not the head
        await m.abandon_queued("c1")
        assert m.state("c1") == "abandoned" and [q.child for q in m.queued("o")] == ["c2"]
        await m.commit_queued("c2", committed=True)  # the join behind it is applied
        await m.finish_join("o", committed=False)
        assert m.clone_refusal("o", [(f"g{i}", False) for i in range(MAX_LIVE_BACKGROUND)]) is None  # no slot is held
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c1", committed=True)  # abandoned is final
        assert m.has("c1") and not m.has("c9")
        m.check()
        await m.stop()
    run(go())


def test_a_focus_graft_whose_join_is_given_up_ends_abandoned():
    async def go():
        m = await root_in_turn()
        await m.clone("o", [("c1", True)])
        await m.join_focus("c1", reason="out_of_scope", trigger=3)
        await m.abandon_queued("c1")
        await m.finish_join("o", committed=False)
        assert (m.state("c1"), m.state("o"), m.phase("o"), m.holder()) == ("abandoned", "active", "idle", "o")
        assert m.queued("o") == [] and await m.join_focus("c1", reason="done", trigger=None) is None
        await m.stop()
    run(go())
