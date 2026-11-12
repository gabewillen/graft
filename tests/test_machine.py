import asyncio

import hsm
import pytest

from graft.machine import MAX_LIVE_BACKGROUND, Machines, TransitionRefused


def run(coro):
    return asyncio.run(coro)


def test_lifecycle_and_single_running_invariant():
    async def go():
        m = Machines()
        await m.start_root("o")
        assert (m.state("o"), m.phase("o")) == ("active", "idle") and m.running() == ["o"]
        await m.begin_turn("o")
        assert (m.state("o"), m.phase("o")) == ("active", "turn")
        await m.clone("o", [("c1", True)])
        assert (m.state("o"), m.state("c1"), m.phase("c1")) == ("cloned", "active", "idle")
        assert m.phase("o") is None and m.running() == ["c1"] and m.slot.holder == "c1"
        src = await m.join_focus("c1")
        assert src == "o"
        assert (m.state("o"), m.state("c1")) == ("joining", "joined")
        assert m.running() == ["o"]
        await m.finish_join("o", committed=True)
        assert (m.state("o"), m.phase("o")) == ("active", "idle")
        await m.begin_turn("o")
        await m.clone("o", [("c2", True)])
        await m.begin_turn("c2")
        await m.join_focus("c2")  # a focus graft joins from inside its turn too
        await m.finish_join("o", committed=False)  # rollback also returns to rest
        assert (m.state("o"), m.phase("o"), m.state("c2")) == ("active", "idle", "joined")
        await m.stop()
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
        assert m.clone_refusal("o", [("c1", False)]) and list(m.timelines) == ["o"]
        assert await m.try_end_turn("o") is True and m.phase("o") == "idle"
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c1", False)])  # nor may a timeline at rest: a clone belongs to a turn
        await m.stop()
    run(go())


def test_refusals():
    async def go():
        m = Machines()
        await m.start_root("o")
        with pytest.raises(TransitionRefused):
            await m.join_focus("o")  # the original has no source
        assert m.state("o") == "active"
        await m.begin_turn("o")
        await m.clone("o", [("c1", True)])
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c9", True)])  # a suspended source cannot clone
        with pytest.raises(TransitionRefused):
            await m.finish_join("o", committed=True)  # cloned -> active only via joining
        with pytest.raises(TransitionRefused):
            await m.begin_turn("o")  # suspended: no turn
        await m.join_focus("c1")
        with pytest.raises(TransitionRefused):
            await m.join_focus("c1")  # joined is final
        with pytest.raises(TransitionRefused):
            await m.begin_turn("c1")
        m.check()
        await m.stop()
    run(go())


def test_slot_rejects_handoff_from_non_holder():
    async def go():
        m = Machines()
        await m.start_root("o")
        with pytest.raises(TransitionRefused):
            await m._handoff("c7", "x")
        # the HSM guard itself also refuses a handoff from a non-holder
        await hsm.Dispatch(None, m.slot, hsm.Event(name="handoff", data=("c7", "x")))
        assert m.slot.holder == "o"
        await m.stop()
    run(go())


def test_background_grafts_are_outside_the_slot():
    async def go():
        m = Machines()
        await m.start_root("o")
        await m.begin_turn("o")
        await m.clone("o", [("c1", False), ("c2", False)])  # not a handoff: the source keeps its turn
        assert (m.state("o"), m.phase("o"), m.state("c1"), m.state("c2")) == \
            ("active", "turn", "background", "background")
        assert m.running() == ["o"] and m.slot.holder == "o" and m.background() == ["c1", "c2"]
        assert m.phase("c1") is None
        await m.clone("o", [("c3", True), ("c4", False)])  # live background grafts do not block a clone call
        assert (m.state("o"), m.state("c3"), m.state("c4")) == ("cloned", "active", "background")
        assert m.running() == ["c3"] and m.slot.holder == "c3"
        assert m.background("o") == ["c1", "c2", "c4"] and m.background("c3") == []
        assert await m.finish_background("c1") == "o"  # frozen and queued; nothing else moves
        assert (m.state("c1"), m.state("o"), m.slot.holder) == ("queued", "cloned", "c3")
        assert m.queued("o") == ["c1"] and m.live_background("o") == ["c1", "c2", "c4"]
        assert await m.try_begin_drain("o") is None  # a suspended source is not at rest
        await m.join_focus("c3")
        await m.commit_queued("c1", committed=True)  # queued joins are applied while the source is joining
        assert m.state("c1") == "joined" and m.queued("o") == []
        await m.finish_join("o", committed=True)
        await m.finish_background("c2")
        await m.finish_background("c4")
        assert await m.try_begin_drain("o") == ["c2", "c4"]  # at rest: idle -> joining, finish order
        assert m.state("o") == "joining" and m.running() == ["o"]
        await m.commit_queued("c2", committed=True)
        await m.commit_queued("c4", committed=False)  # a rolled-back join also leaves the queue
        await m.finish_join("o", committed=False)
        assert m.background() == [] and m.queued("o") == [] and m.running() == ["o"]
        assert (m.state("c2"), m.state("c4")) == ("joined", "joined")
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
        await m.finish_background("c1")
        assert await m.try_begin_drain("o") is None  # mid-turn
        await m.end_turn("o")
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c1", committed=True)  # the source is not joining
        assert await m.begin_drain("o") == ["c1"]
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c2", committed=True)  # still running: not queued
        await m.finish_join("o", committed=False)  # begin, then rollback: nothing was applied
        assert (m.state("o"), m.phase("o"), m.state("c1"), m.queued("o")) == ("active", "idle", "queued", ["c1"])
        assert await m.begin_drain("o") == ["c1"]  # the join was not lost: it is offered again
        await m.commit_queued("c1", committed=True)
        with pytest.raises(TransitionRefused):
            await m.commit_queued("c1", committed=True)  # a second join on a joined background graft
        with pytest.raises(TransitionRefused):
            await m.finish_background("c1")
        await m.finish_join("o", committed=True)
        m.check()
        await m.stop()
    run(go())


def test_drain_is_refused_from_a_timeline_that_does_not_hold_the_conversation():
    async def go():
        m = Machines()
        await m.start_root("o")
        await m.begin_turn("o")
        await m.clone("o", [("c1", True), ("c2", False)])
        await m.finish_background("c2")
        assert m.queued("o") == ["c2"] and m.slot.holder == "c1"
        assert await m.try_begin_drain("o") is None  # queued on o, but c1 holds the conversation
        assert await m.try_begin_drain("c1") is None  # the holder has nothing queued
        with pytest.raises(TransitionRefused):
            await m.begin_drain("o")
        await m.stop()
    run(go())


def test_clone_call_refusals():
    async def go():
        m = Machines()
        await m.start_root("o")
        await m.begin_turn("o")
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c1", True), ("c2", True)])  # at most one graft takes the conversation
        with pytest.raises(TransitionRefused):
            await m.clone("o", [])
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c1", False), ("c1", False)])
        too_many = [(f"c{i}", False) for i in range(MAX_LIVE_BACKGROUND + 1)]
        with pytest.raises(TransitionRefused):
            await m.clone("o", too_many)  # the cap on live background grafts per source
        assert list(m.timelines) == ["o"] and m.phase("o") == "turn"  # refused calls created nothing
        await m.clone("o", too_many[:-1])
        with pytest.raises(TransitionRefused):
            await m.clone("o", [("c9", False)])  # still at the cap: queued grafts count until they join
        await m.stop()
    run(go())
