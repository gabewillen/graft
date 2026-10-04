import asyncio

import hsm
import pytest

from graft.machine import Machines, TransitionRefused


def run(coro):
    return asyncio.run(coro)


def test_lifecycle_and_single_running_invariant():
    async def go():
        m = Machines()
        await m.start_root("o")
        assert m.state("o") == "active" and m.running() == ["o"]
        await m.clone("o", "c1")
        assert (m.state("o"), m.state("c1")) == ("cloned", "active")
        assert m.running() == ["c1"] and m.slot.holder == "c1"
        src = await m.join("c1")
        assert src == "o"
        assert (m.state("o"), m.state("c1")) == ("joining", "joined")
        assert m.running() == ["o"]
        await m.finish_join("o", committed=True)
        assert m.state("o") == "active"
        await m.clone("o", "c2")
        await m.join("c2")
        await m.finish_join("o", committed=False)  # rollback also returns to active
        assert m.state("o") == "active" and m.state("c2") == "joined"
        await m.stop()
    run(go())


def test_refusals():
    async def go():
        m = Machines()
        await m.start_root("o")
        with pytest.raises(TransitionRefused):
            await m.join("o")  # the original has no source: guard refuses
        assert m.state("o") == "active"
        await m.clone("o", "c1")
        with pytest.raises(TransitionRefused):
            await m.clone("o", "c9")  # suspended source cannot clone; one clone at a time
        with pytest.raises(TransitionRefused):
            await m.finish_join("o", True)  # cloned -> active only via joining
        await m.join("c1")
        with pytest.raises(TransitionRefused):
            await m.join("c1")  # joined is frozen
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
