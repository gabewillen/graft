"""What a branch starts with (spec: Clone, The seed and the graft's view; What the model is told,
Fork and merge are messages): the messages main points at (row ids or exact quotes, in stored
order), then main's own fork turn (its reason included), then that turn's results as the branch
sees them: its own fork result, and main's results of the other calls. Nothing is injected with
the user role, and main chooses what each branch sees."""

import asyncio
import json
import threading

from _util import chat_of, conversation, install_fake
from graft.db import Store
from graft.harness import NOT_RUN_BESIDE_CLONE, Host, clone_entries, fork_result
from graft.providers import FakeProvider

ASK = "I run a dental practice in Austin."
SECRET = "My password for the old host is hunter2."
REQUEST = "Build the practice a website."


def text(t):
    return [{"type": "text", "text": t}]


def fork_use(use_id="f1", **inp):
    return {"type": "tool_use", "id": use_id, "name": "fork", "input": inp}


def make(tmp_path, policy, **kw):
    store = Store(str(tmp_path / "fork.db"))
    install_fake(store)
    h = conversation(store, policy, seeder=kw.pop("seeder", None), **kw).h
    return store, h


class Main:
    """A scripted main that forks `blocks` for REQUEST, and a branch that records the messages it
    was first sent and replies."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.branch_sent = None
        self.done = threading.Event()

    def __call__(self, system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "fork" in names:
            if last["type"] == "text" and not last.get("harness") and last["text"].endswith(REQUEST):
                return self.blocks
            return []  # after the merge: nothing more to say
        if self.branch_sent is None:
            self.branch_sent = messages
        return text("Here is your site.")


def run(h, *says):
    async def go():
        await chat_of(h).start()
        for s in says:
            await chat_of(h).say(s)
        await chat_of(h).quiet()
        await h.close()
    asyncio.run(go())


def test_entry_shape_messages_and_reason():
    _, v = clone_entries({"current": {"skills": ["website"]}})
    assert any("messages must be a list" in x for x in v), v
    _, v = clone_entries({"current": {"skills": ["website"], "messages": []}})
    assert v == ["current: give it messages, a reason, or both: with neither it would start from nothing"]
    es, v = clone_entries({"current": {"skills": ["website"], "messages": [], "reason": "Build it."}})
    assert v == [] and es[0].reason == "Build it." and es[0].messages == []
    _, v = clone_entries({"current": {"skills": ["website"], "messages": [], "reason": "  "}})
    assert v and "neither" in v[0]  # a blank reason is no reason
    _, v = clone_entries({"current": {"skills": ["website"], "messages": [1.5, True, "q"]}})
    assert v == ["current: each item of messages is a short exact quote (a string) or a row id (an integer)"]
    _, v = clone_entries({"current": {"skills": ["website"], "messages": ["q"], "reason": 3}})
    assert v == ["current: reason must be a string (your own words for what this branch is for)"]
    _, v = clone_entries({"current": {"skills": ["website"], "messages": ["q"], "purpose": "x"}})
    assert v == ["current: unknown field 'purpose' (a branch is {id?, skills, messages, reason?})"]


def test_a_branch_sees_only_the_messages_main_points_at_then_main_fork_turn_and_its_own_result(tmp_path):
    main = Main([{"type": "text", "text": "On it."},
                 fork_use(current={"id": "site", "skills": ["website"], "messages": [REQUEST, "dental practice"],
                                   "reason": "Build the site; ask what pages they want."})])
    store, h = make(tmp_path, main)
    run(h, ASK, SECRET, REQUEST)
    sent = main.branch_sent
    flat = json.dumps(sent)
    assert "hunter2" not in flat  # main did not point at it: the branch never sees it
    # stored order: the quoted messages, then main's fork turn (text + call), then its own result
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert [b.get("text", "").split("\n")[-1] for b in sent[0]["content"]] == [ASK, REQUEST]
    assert [b["type"] for b in sent[1]["content"]] == ["text", "tool_use"]
    assert sent[1]["content"][1]["input"]["current"]["reason"] == "Build the site; ask what pages they want."
    result = sent[2]["content"]
    assert len(result) == 1 and result[0]["type"] == "tool_result" and result[0]["tool_use_id"] == "f1"
    assert result[0]["content"] == fork_result(name="site", focus=True, skills=["website"], reason="yes")
    assert result[0]["content"].startswith("You are now on branch site (current: ")
    assert result[0]["content"].endswith("Your part: the reason you wrote for site above.")
    # main's own result stays "forked -> ..." and is never in the branch's view
    mine = [json.loads(r["content"]) for r in store.view("o") if r["kind"] == "tool_result" and r["tool_name"] == "fork"]
    assert mine[0]["content"].startswith("forked -> site (current;")
    assert "forked ->" not in flat
    # nothing is injected with the user role: the branch's rows are pointers plus its one result row
    own = [r for r in store.view("c1") if r["owner_timeline"] == "c1" and r["pos"] <= 4]
    assert [(r["role"], r["kind"], r["harness"]) for r in own] == [("user", "tool_result", 1)]


def test_a_branch_with_no_messages_starts_on_main_fork_turn(tmp_path):
    main = Main([fork_use(detached=[{"id": "deck", "skills": ["slides"], "messages": [],
                                     "reason": "A five-slide deck for a dental practice."}])])
    store, h = make(tmp_path, main)
    run(h, REQUEST)
    sent = main.branch_sent
    assert [m["role"] for m in sent] == ["assistant", "user"]
    assert sent[1]["content"][0]["content"].startswith("You are now on branch deck (detached: ")


def test_a_pointer_that_names_no_message_refuses_the_call_and_lists_the_messages(tmp_path):
    calls = []

    def policy(system, messages, tools):
        names = {t["name"] for t in tools}
        last = messages[-1]["content"][-1]
        if "fork" not in names:
            return text("built")
        if last["type"] == "tool_result":
            calls.append(last["content"])
            return text("Sorry.")
        if not last["text"].endswith(REQUEST):
            return text("Noted.")
        return [fork_use(current={"skills": ["website"], "messages": [
            "nothing like this", "dental", 999, 1, "Build"]})]

    store, h = make(tmp_path, policy)
    run(h, ASK, "I run a dental clinic too.", REQUEST)
    refusal = calls[0]
    assert refusal.startswith("fork refused; no branch was started:")
    assert "current: quote 'nothing like this' matches no message" in refusal
    assert "current: quote 'dental' matches 2 messages" in refusal
    assert "current: row 999 is not in your conversation" in refusal
    assert "Your messages (row id, role: start of the text):" in refusal
    assert f"  1 user: {ASK}" in refusal
    assert store.con.execute("SELECT count(*) FROM timelines WHERE source_id IS NOT NULL").fetchone()[0] == 0


def test_pointers_to_tool_rows_harness_rows_and_withdrawn_messages_are_refused(tmp_path):
    store, h = make(tmp_path, lambda *a: text("ok"))

    async def go():
        await chat_of(h).start()
        u = store.add_turn("o", "user", "text", REQUEST)
        tu = store.add_turn("o", "assistant", "tool_use", {"id": "x", "input": {}}, tool_name="load_skill")
        tr = store.add_turn("o", "user", "tool_result", {"tool_use_id": "x", "content": "ok"}, tool_name="load_skill")
        note = store.add_turn("o", "user", "text", "(a note)", harness=True)
        gone = store.add_turn("o", "user", "text", "Forget it.")
        store.mark_withdrawn(timelines=["o"], rows=[gone])
        rows, problems = h.resolve_messages("o", [u, tu, tr, note, gone, "Forget it.", u], exclude=set())
        assert rows == [u]
        assert problems == [f"row {tu} is a tool call or tool result, not a message",
                            f"row {tr} is a tool call or tool result, not a message",
                            f"row {note} is a harness note, not a message",
                            f"row {gone} is a message the user withdrew; it is not a request and cannot be seeded",
                            "quote 'Forget it.' matches only a message the user withdrew; it cannot be seeded"]
        await h.close()

    asyncio.run(go())


def test_other_calls_of_the_fork_turn_carry_main_results_in_the_branch_and_thinking_stays_out(tmp_path):
    main = Main([{"type": "thinking", "thinking": "hmm", "signature": "s"},
                 fork_use(current={"skills": ["website"], "messages": [REQUEST]}),
                 {"type": "tool_use", "id": "l1", "name": "load_skill", "input": {"name": "fork"}}])
    store, h = make(tmp_path, main)
    run(h, REQUEST)
    sent = main.branch_sent
    assert [b["type"] for b in sent[1]["content"]] == ["tool_use", "tool_use"]
    results = {b["tool_use_id"]: b["content"] for b in sent[2]["content"]}
    assert results["l1"] == NOT_RUN_BESIDE_CLONE
    assert results["f1"].endswith("Your part: what the messages above ask for that your skills cover.")


def test_messages_and_refusals_are_bounded():
    from graft.harness import MAX_MESSAGES
    _, v = clone_entries({"current": {"skills": ["website"], "messages": ["q"] * (MAX_MESSAGES + 1)}})
    assert v == [f"current: messages names {MAX_MESSAGES + 1} items; give at most {MAX_MESSAGES}"]


def test_a_refusal_lists_a_bounded_number_of_problems_and_cuts_long_row_ids(tmp_path):
    from graft.harness import MAX_PROBLEMS
    calls = []

    def policy(system, messages, tools):
        last = messages[-1]["content"][-1]
        if "fork" not in {t["name"] for t in tools}:
            return text("built")
        if last["type"] == "tool_result":
            calls.append(last["content"])
            return text("Sorry.")
        return [fork_use(current={"skills": ["website"], "messages": [10 ** 200] + list(range(900, 940))})]

    store, h = make(tmp_path, policy)
    run(h, REQUEST)
    refusal = calls[0]
    assert refusal.count("\n- ") == MAX_PROBLEMS + 1 and f"and {41 - MAX_PROBLEMS} more" in refusal
    assert "1" + "0" * 200 not in refusal and "row 100000000000000000000000... is not in" in refusal
