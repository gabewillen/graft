"""Every branch turn ends in a merge, and a merge is messages (spec: Join; What the model is told,
Fork and merge are messages): the branch's merge call (its own, or the harness's when the turn ends
without one) comes back to main as an assistant message, followed by main's merge result, which
says what the wake note said; main always runs once after a merge, and may end that run silently."""

import asyncio
import json
import threading

from _util import holder, ran
from graft.db import Store, validate
from graft.harness import Harness
from graft.providers import FakeProvider
from graft.skills import install

REQUEST = "Build the practice a website and a slide deck."


def text(t):
    return [{"type": "text", "text": t}]


def fork_use(use_id="f1", **inp):
    return {"type": "tool_use", "id": use_id, "name": "fork", "input": inp}


def make(tmp_path, policy, **kw):
    store = Store(str(tmp_path / "merge.db"))
    install(store)
    said = []
    h = Harness(store, FakeProvider(policy), seeder=None, emit=said.append, **kw)
    return store, h, said


def main_sees(store):
    """main's view as (role, kind, tool name, harness) rows."""
    return [(r["role"], r["kind"], r["tool_name"], r["harness"]) for r in store.view("o")]


def merge_results(store, tl="o"):
    return [json.loads(r["content"])["content"] for r in store.view(tl)
            if r["kind"] == "tool_result" and r["tool_name"] == "merge" and r["owner_timeline"] == tl]


class Script:
    """main forks `entry` (current or detached) for REQUEST and, after a merge, does what `after`
    says (nothing: ends silently); the branch answers with `branch(messages)`."""

    def __init__(self, entry, branch, after=None, current=True):
        self.entry, self.branch, self.after, self.current = entry, branch, after, current
        self.main_after_merge: list = []

    def __call__(self, system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "fork" in names:
            if last["type"] == "text" and not last.get("harness") and last["text"].endswith(REQUEST):
                return [fork_use(**({"current": self.entry} if self.current else {"detached": [self.entry]}))]
            if ran(last) and last["content"].startswith("forked"):
                return text("Started.") if not self.current else []
            self.main_after_merge.append(messages)
            return self.after(messages) if self.after else []
        return self.branch(messages)


def run(h, *says):
    out = []

    async def go():
        await h.start()
        for s in says:
            out.append(await h.user_says(s))
        await h.settle()
        await h.close()

    asyncio.run(go())
    return out


SITE = {"id": "site", "skills": ["website"], "messages": [REQUEST], "reason": "Build the website."}


def test_a_current_branch_that_replies_merges_at_the_end_of_its_turn_and_main_runs_once(tmp_path):
    script = Script(entry=SITE, branch=lambda m: text("What pages should the site have?"))
    store, h, said = make(tmp_path, script)
    out = run(h, REQUEST)
    assert out == [["What pages should the site have?"]]  # main's run after the merge said nothing
    assert len(script.main_after_merge) == 1  # main always runs once after a merge
    # the branch's timeline ends on the harness's argument-less merge call and its result
    tail = [(r["role"], r["kind"], r["tool_name"], r["harness"]) for r in store.view("c1")][-3:]
    assert tail == [("assistant", "text", None, 0), ("assistant", "tool_use", "merge", 1),
                    ("user", "tool_result", "merge", 1)]
    use = json.loads(store.view("c1")[-2]["content"])
    assert use["input"] == {}
    assert json.loads(store.view("c1")[-1]["content"])["content"] == "merged"
    # main's view: its fork turn and result, the branch's reply and merge call, main's merge result
    assert main_sees(store)[-5:] == [("assistant", "tool_use", "fork", 0), ("user", "tool_result", "fork", 0),
                                     ("assistant", "text", None, 0), ("assistant", "tool_use", "merge", 1),
                                     ("user", "tool_result", "merge", 1)]
    result = merge_results(store)[0]
    assert result.startswith("site merged. Its replies above already reached the user: do not repeat them.")
    assert "end your turn without saying anything" in result
    # no user-role harness note: the merge result carries what the wake note carried
    assert not [r for r in store.view("o") if r["harness"] and r["kind"] == "text"]
    assert store.timeline("c1")["join_reason"] == "done" and holder(store) == "o"
    assert validate(store.view("o"), "o") == []
    # main's run after the merge sees the merge call as an assistant message
    last_two = script.main_after_merge[0][-2:]
    assert last_two[0]["role"] == "assistant" and last_two[0]["content"][-1]["name"] == "merge"


def test_the_next_message_goes_to_main_never_to_the_merged_branch(tmp_path):
    seen = []

    def branch(messages):
        seen.append(messages)
        return text("What pages?")

    script = Script(entry=SITE, branch=branch, after=lambda m: [])
    store, h, said = make(tmp_path, script)
    run(h, REQUEST, "Home and contact.")
    assert len(seen) == 1  # the branch ran one turn
    assert store.view("o")[-1]["content"] == "Home and contact." or any(
        r["content"] == "Home and contact." and r["owner_timeline"] == "o" for r in store.view("o"))
    assert store.timeline("c1")["status"] == "joined"


def test_merge_with_a_reason_and_no_reply_hands_the_request_back_and_main_reads_the_reason(tmp_path):
    branch = lambda m: [{"type": "tool_use", "id": "m1", "name": "merge",
                         "input": {"reason": "The user also wants slides; I only build sites."}}]
    script = Script(entry=SITE, branch=branch, after=lambda m: text("I'll do the slides too."))
    store, h, said = make(tmp_path, script)
    out = run(h, REQUEST)
    assert out == [["I'll do the slides too."]]
    assert store.timeline("c1")["join_reason"] == "out_of_scope"  # graft's own word; the model never sees it
    merge = [r for r in store.view("o") if r["kind"] == "tool_use" and r["tool_name"] == "merge"][0]
    assert merge["owner_timeline"] == "c1" and merge["harness"] == 0
    assert json.loads(merge["content"])["input"]["reason"] == "The user also wants slides; I only build sites."
    result = merge_results(store)[0]
    assert result.startswith("site merged without answering the user's last message")
    seen = script.main_after_merge[0]
    assert seen[-2]["content"][-1]["input"]["reason"].startswith("The user also wants slides")


def test_calls_beside_a_merge_are_not_run_and_stay_in_the_branch(tmp_path):
    branch = lambda m: (text("Done: home page written.")
                        + [{"type": "tool_use", "id": "w1", "name": "site_write", "input": {"page": "x", "content": "y"}},
                           {"type": "tool_use", "id": "m1", "name": "merge", "input": {}}])
    script = Script(entry=SITE, branch=branch)
    store, h, said = make(tmp_path, script)
    run(h, REQUEST)
    assert h.world.site == {}
    mine = [(r["kind"], r["tool_name"]) for r in store.view("o") if r["owner_timeline"] == "c1"]
    assert mine == [("text", None), ("tool_use", "merge")]  # the sibling call is not pointed at
    assert validate(store.view("o"), "o") == []


def test_a_detached_branch_merges_and_main_is_told_the_user_has_not_seen_it(tmp_path):
    gate = threading.Event()

    def branch(messages):
        assert gate.wait(5)
        return text("Deck built: five slides.")

    entry = {"id": "deck", "skills": ["slides"], "messages": [], "reason": "A five-slide deck."}
    script = Script(entry=entry, branch=branch, after=lambda m: text("I built the deck."), current=False)
    store, h, said = make(tmp_path, script)

    async def go():
        await h.start()
        first = await h.user_says(REQUEST)
        assert first == ["Started."]
        gate.set()
        await h.join_queued("c1")
        assert await h.settle() == ["I built the deck."]
        await h.close()

    asyncio.run(go())
    assert said[-1] == "I built the deck."
    result = merge_results(store)[0]
    assert result.startswith("deck merged. What you did there is in the messages above, and the user has not seen it")
    assert main_sees(store)[-4:] == [("assistant", "text", None, 0), ("assistant", "tool_use", "merge", 1),
                                     ("user", "tool_result", "merge", 1), ("assistant", "text", None, 0)]
    assert validate(store.view("o"), "o", require_user_end=False) == []
