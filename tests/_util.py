"""Shared test helpers. Tests observe the harness through what it stores and says: each
machine's state is copied to `timelines.status` in the step that enters it, so after any awaited
call the store says where every timeline is."""

import re


def conversation_rows(store, src="o"):
    """The row ids of src's conversation a test branch starts from by default: its messages (text
    rows that are not harness notes or withdrawn) from the first user message to the last."""
    rows = [r for r in store.working_view(src)
            if r["kind"] == "text" and not r["harness"] and not r["withdrawn"]]
    users = [i for i, r in enumerate(rows) if r["role"] == "user"]
    return [r["row_id"] for r in rows[users[0]:users[-1] + 1]] if users else []


def pointing(store, use, src="o"):
    """`use` with every entry that names no messages pointed at src's conversation (by row id):
    a driven turn stores no fork call, so its branches start from their messages alone."""
    args = use.get("input")
    if not isinstance(args, dict):
        return use
    seeded = lambda e: dict(e, messages=conversation_rows(store=store, src=src)) if isinstance(e, dict) and not e.get(
        "messages") and "reason" not in e else e
    out = dict(args)
    if "current" in out:
        out["current"] = seeded(out["current"])
    if isinstance(out.get("detached"), list):
        out["detached"] = [seeded(e) for e in out["detached"]]
    return dict(use, input=out)


async def clone_in_turn(h, use, src="o"):
    """Make a clone call the way the harness loop does, inside a turn of `src` (a turn its caller
    drives, so the call is not a row of src's view: an entry that names no messages is pointed at
    the conversation, `pointing`). A turn that still holds the conversation afterwards (background
    grafts only, or a refused call) is ended, so `src` is left at rest. Returns (new timeline ids or
    None, result text)."""
    use = pointing(store=h.store, use=use, src=src)
    await h.begin_turn(src)
    children, focus, text = await h.clone_call(src=src, use=use)
    if focus is None:
        await h.end_turn(src)
    return children, text


def status(store, tl):
    """tl's state, as its machine copied it to the store."""
    return store.timeline(tl)["status"]


def holders(store):
    """The timelines holding the conversation: one, whatever happened."""
    return [r[0] for r in store.con.execute(
        "SELECT timeline_id FROM timelines WHERE status IN ('active', 'joining') ORDER BY timeline_id")]


def clones(store):
    """The grafts made, in order (a clone call undone because its machines could not start is
    left out: its grafts are `abandoned` without ever running)."""
    return [r[0] for r in store.con.execute(
        "SELECT t.timeline_id FROM timelines t WHERE t.source_id IS NOT NULL AND NOT (t.status = 'abandoned' "
        "AND t.join_reason IS NULL) ORDER BY t.rowid")]


def holder(store):
    """The timeline holding the conversation, or None if the store says it is not exactly one."""
    run = holders(store)
    return run[0] if len(run) == 1 else None


def branch_input(*entries):
    """A fork call's input from test entries: each {"skills", "focus", "id"?, "messages"?,
    "reason"?} (no messages: an empty list); the one with focus=True is `current`, the others are
    `detached`, in order. A dict without `focus` is
    passed through as a detached entry (a malformed one, for refusal tests)."""
    current = [e for e in entries if isinstance(e, dict) and e.get("focus") is True]
    detached = [e for e in entries if not (isinstance(e, dict) and e.get("focus") is True)]
    strip = lambda e: {"messages": [], **{k: v for k, v in e.items() if k != "focus"}} if isinstance(e, dict) else e
    out = {}
    if current:
        out["current"] = strip(current[0])
    if detached or not current:
        out["detached"] = [strip(e) for e in detached]
    return out


def _quotes(messages):
    """The user's own messages in a rendered view, as quotes (no harness rows, no nonce). A text
    sent more than once is left out: a quote of it would match more than one message."""
    texts = []
    for m in messages:
        if m["role"] != "user":
            continue
        for b in m["content"]:
            if b["type"] == "text" and not b.get("harness"):
                q = re.sub(r"^\[conversation [^\]]*\]\s*", "", b["text"]).strip()
                if q:
                    texts.append(q)
    return [q for q in dict.fromkeys(texts) if texts.count(q) == 1]


def quoting(policy):
    """A scripted policy whose fork calls point every entry that names no messages and gives no
    reason at the user's messages in its view (as quotes), as a test default: the branch then
    starts from the conversation's requests, as a model that quotes them would make it."""
    def wrapped(system, messages, tools):
        blocks = policy(system, messages, tools)
        if not isinstance(blocks, list):
            return blocks
        out = []
        for b in blocks:
            args = b.get("input") if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "fork" else None
            if isinstance(args, dict):
                fill = lambda e: dict(e, messages=_quotes(messages)) if isinstance(e, dict) and not e.get(
                    "messages") and "reason" not in e else e
                args = dict(args)
                if "current" in args:
                    args["current"] = fill(args["current"])
                if isinstance(args.get("detached"), list):
                    args["detached"] = [fill(e) for e in args["detached"]]
                b = dict(b, input=args)
            out.append(b)
        return out
    return wrapped


def started(block):
    """Is this block a branch's own fork result, the last row of its first view (spec: Clone)?"""
    return (block.get("type") == "tool_result" and isinstance(block.get("content"), str)
            and block["content"].startswith("You are now on branch "))


def merged(block):
    """Is this block main's result of a branch's merge call (spec: Join), which ends main's view
    when it is woken?"""
    content = block.get("content")
    return (block.get("type") == "tool_result" and isinstance(content, str)
            and bool(re.match(r"[a-z0-9][a-z0-9-]* (merged\b|was stopped before it finished)", content)))


def ran(block):
    """Is this block the result of a tool call the timeline made itself (not a branch's fork
    result, which ends a branch's first view, nor main's result of a branch's merge call)?"""
    return block.get("type") == "tool_result" and not started(block) and not merged(block)


def woke(block):
    """Is this block what the harness wrote for main to read, a note or a merge result (what a
    wake run's view ends on)?"""
    if merged(block):
        block.setdefault("text", block["content"])  # read like a note's text
        return True
    return bool(block.get("harness"))


# ------------------------------------------------------------------ the host interface, for tests
import asyncio
import contextlib
import itertools
import weakref

from graft import a2a
from graft.agent import GraftAgent, GraftConfig, HarnessError, fold
from graft.fake import FakeWorld
from graft.providers import FakeProvider
from graft.skills import install

_CHATS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def install_fake(store, *, world=None):
    """The skill catalog, with the simulated tools' definitions (graft/fake.py)."""
    install(store, tools=(world or FakeWorld()).definitions())


class Chat:
    """A test's conversation with one agent, through its host interface (`GraftAgent`): `say` sends
    a message as a new task (blocking SendMessage), `quiet` follows every open task until it is
    terminal (SubscribeToTask). Both return the agent's texts told since the last return, read from
    the event log in order (harness error lines included), and raise `HarnessError` as the interface
    does. `said` is every text told so far; `h` is the agent's internal host, for driven turns."""

    def __init__(self, agent: GraftAgent):
        self.agent = agent
        self.h = agent._host
        self.store = agent.store
        self.cursor = 0
        self.seen: set[str] = set()
        self.last_error: tuple[int, str] | None = None  # the last error line read, at its position
        self.said: list[str] = []
        self.returned = 0  # said[:returned] was returned by `say` or `quiet`
        self.raised: list[HarnessError] = []  # what the interface raised to `say` and `quiet`
        self.ids = itertools.count(1)
        _CHATS[self.h] = self

    async def start(self):
        await self.agent.start()

    def _texts(self, events) -> list[str]:
        """The agent's texts, each once. A text told on several open tasks at once (one message per
        task, written together: a run for several tasks, or a harness error) is one line."""
        out = []
        for seq, event in events:
            body = event.get("statusUpdate", {}).get("status", {}).get("message") or event.get("message")
            if not body or body["role"] != a2a.ROLE_AGENT or body["messageId"] in self.seen:
                continue
            self.seen.add(body["messageId"])
            text = "".join(p["text"] for p in body["parts"])
            if self.last_error == (seq - 1, text):  # the same text told on another task at once
                self.last_error = (seq, text)
                continue
            self.last_error = (seq, text)
            out.append(text)
        return out

    def _sync(self) -> None:
        """Every text told so far, in order, into `said`."""
        events = self.store.events_after(self.cursor, limit=10 ** 9)
        if events:
            self.cursor = events[-1][0]
        self.said += self._texts(events)

    def heard(self) -> list[str]:
        """The texts told since the last `say` or `quiet` returned."""
        self._sync()
        texts, self.returned = self.said[self.returned:], len(self.said)
        return texts

    def errors(self) -> list[str]:
        """Every harness error line in the log, in order, each once (one told on several open tasks
        at once is one line)."""
        out, last = [], None
        for seq, event in self.store.events_after(0, limit=10 ** 9):
            body = event.get("statusUpdate", {}).get("status", {}).get("message") or event.get("message")
            if body and (body.get("metadata") or {}).get("graft", {}).get("error"):
                text = "".join(p["text"] for p in body["parts"])
                if last != (seq - 1, text):
                    out.append(text)
                last = (seq, text)
        return out

    async def written(self) -> list[str]:
        """Every text told so far, once the session has written what it decided (the log stays
        still for a moment): for a test that waited on the machines, not on a task."""
        position = -1
        while position != self.store.event_position():
            position = self.store.event_position()
            try:
                await asyncio.wait_for(self.agent._wait(position), 0.05)
            except (asyncio.TimeoutError, HarnessError):
                pass
        self._sync()
        return list(self.said)

    async def told(self, phrase: str) -> list[str]:
        """The harness error lines containing `phrase`, waiting (bounded) until one is written:
        errors are events the session writes as it decides them."""
        for _ in range(200):
            found = [line for line in self.errors() if phrase in line]
            if found:
                return found
            position = self.store.event_position()
            try:
                await asyncio.wait_for(self.agent._wait(position), 0.05)
            except (asyncio.TimeoutError, HarnessError):
                pass
        return []

    def message(self, text, task_id=None) -> dict:
        m = {"messageId": f"u{next(self.ids)}", "role": "ROLE_USER", "parts": [{"text": text}]}
        return {"message": m | ({"taskId": task_id} if task_id else {})}

    async def send(self, text, task_id=None) -> a2a.Task:
        """One message, followed until main's turn ends (its task's next INPUT_REQUIRED or terminal
        state, as a blocking SendMessage returns); returns its task. A cancel of this call is the
        user's Stop: the task is canceled (CancelTask) before the cancel goes on."""
        request = self.message(text, task_id)
        task, error = task_id, None
        try:
            async for event in self.agent.stream(request):
                if event.task is not None:
                    task = event.task.id
                    continue
                status = event.status_update.status
                if status.message is not None and (status.message.metadata or {}).get("graft", {}).get("error"):
                    error = error or status.message.text
                if status.state in a2a.INTERRUPTED | a2a.TERMINAL:
                    break
        except asyncio.CancelledError:
            task = task or await self._task_of(request["message"]["messageId"])
            with contextlib.suppress(a2a.A2AError):
                await self.agent.cancel_task({"id": task})
            raise
        if error is not None:
            raise HarnessError(error)
        return self.agent.get_task({"id": task})

    async def _task_of(self, message_id) -> str:
        """The task a message of this chat opened, once its row is in the log (the session took
        the message before the call that sent it could be cancelled)."""
        cursor = 0
        for _ in range(1000):
            for seq, event in self.store.events_after(cursor, limit=10 ** 9):
                cursor = seq
                body = event.get("task") or event.get("statusUpdate", {}).get("status", {})
                messages = body.get("history", []) if "task" in event else [body.get("message") or {}]
                if any(m.get("messageId") == message_id for m in messages):
                    return event["task"]["id"] if "task" in event else event["statusUpdate"]["taskId"]
            await self.agent._wait(cursor)
        raise AssertionError(f"no task holds message {message_id}")

    async def say(self, text, task_id=None) -> list[str]:
        """One message (`send`); the texts told since the last return, harness error lines
        included. A HarnessError the interface raised is kept in `raised` (its line is in the
        texts), so a test reads what the user was told; `send` is the strict call."""
        try:
            await self.send(text, task_id)
        except HarnessError as error:
            self.raised.append(error)
        return self.heard()

    def open_tasks(self) -> list[str]:
        return [t for t in self.store.task_ids()
                if fold(events=[e for _, e in self.store.task_events(t)]).status.state not in a2a.TERMINAL]

    async def quiet(self) -> list[str]:
        """Follow every open task until it is terminal (a HarnessError is kept in `raised`); the
        texts told meanwhile."""
        while self.open_tasks():
            for task_id in self.open_tasks():
                try:
                    async for _event in self.agent.subscribe({"id": task_id}):
                        pass
                except a2a.UnsupportedOperation:  # it ended meanwhile
                    pass
                except HarnessError as error:
                    self.raised.append(error)
        return self.heard()

    # a list of every text told so far
    def __len__(self):
        self._sync()
        return len(self.said)

    def __getitem__(self, i):
        self._sync()
        return self.said[i]

    def __iter__(self):
        self._sync()
        return iter(list(self.said))

    def __eq__(self, other):
        self._sync()
        return list(self.said) == list(other)


def conversation(store, policy=None, *, provider=None, world=None, tools=None, quote=True, **config) -> Chat:
    """An agent on `store` with a scripted model (`policy`, wrapped by `quoting` unless `quote` is
    false) or a `provider`, and the simulated tools of `world`; returns its Chat."""
    if provider is None:
        provider = FakeProvider(quoting(policy) if quote else policy)
    world = world if world is not None else FakeWorld()
    count = itertools.count(1)
    config.setdefault("ids", lambda: f"t{next(count)}")
    config.setdefault("context_id", "ctx")
    if not store.skill_names():
        install_fake(store, world=world)
    agent = GraftAgent(store, provider, tools=world.tools() if tools is None else tools, config=GraftConfig(**config))
    chat = Chat(agent)
    chat.world = world
    return chat


def chat_of(h) -> Chat:
    """The Chat of an agent's internal host."""
    return _CHATS[h]
